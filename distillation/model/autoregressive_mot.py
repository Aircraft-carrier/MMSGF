"""Independent autoregressive MoT model family.

The class is parameter-compatible with VAMOTTransformer3DModel.  Its
training path owns the AR order natively; rollout integration is added through
the normal model forward boundary so FSDP hooks remain effective.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch

from wan_va.modules.model_va_mot import (
    VAMOTBlock,
    VAMOTTransformer3DModel,
)

from .autoregressive_types import (
    AutoregressiveMOTLayerRequest,
    AutoregressiveModelOutput,
    AutoregressiveModelRequest,
    AutoregressiveStreamInput,
)


class AutoregressiveVAMOTBlock(VAMOTBlock):
    """Parameter-compatible MoT block reserved for native AR execution."""

    def forward(self, states, *args, **kwargs):
        if isinstance(states, AutoregressiveMOTLayerRequest):
            return self.forward_incremental(states)
        return super().forward(states, *args, **kwargs)

    def forward_incremental(self, request: AutoregressiveMOTLayerRequest):
        stream = request.stream
        hidden = request.hidden_state
        block = (
            self.video_block
            if stream.block_kind == "video"
            else self.action_block
        )
        modulation = self._modulation(block, stream.conditioning)
        query, current_key, current_value = self._self_qkv(
            block,
            hidden,
            modulation,
            stream.rotary,
        )
        request.cache.append(
            request.layer_id,
            current_key,
            current_value,
            transaction_id=request.transaction_id,
        )
        key, value = request.cache.materialize(
            request.layer_id,
            transaction_id=request.transaction_id,
        )
        # Causality is automatic: committed history plus the current transaction
        # are the only visible tokens, and future frames are not cached yet.
        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        attention_output = attention_output.transpose(1, 2)
        text_attn = self.video_block.attn2
        text_key = text_attn.norm_k(text_attn.to_k(request.text)).unflatten(
            2, (text_attn.heads, -1)
        )
        text_value = text_attn.to_v(request.text).unflatten(
            2, (text_attn.heads, -1)
        )
        self_output = self._attention_output(block.attn1, attention_output)
        return self._finish_block(
            block,
            hidden,
            self_output,
            modulation,
            text_key,
            text_value,
        )

class AutoregressiveVAMOTTransformer3DModel(
    VAMOTTransformer3DModel
):
    """MOT transformer with AR metadata and AR component construction."""

    _no_split_modules = ["AutoregressiveVAMOTBlock"]
    _repeated_blocks = ["AutoregressiveVAMOTBlock"]

    def __init__(self, *args, generation_shape=None, **kwargs):
        generation_shape = dict(generation_shape or {})
        self.history_frames = int(generation_shape.get("history_frames", 4))
        self.chunk_size = int(generation_shape.get("chunk_size", 4))
        self.window_size = int(generation_shape.get("window_size", 16))
        super().__init__(*args, **kwargs)
        self.register_to_config(generation_shape=generation_shape)

    def _build_mot_block(self, **kwargs):
        return AutoregressiveVAMOTBlock(**kwargs)

    def _prepare_metadata(
        self,
        input_dict: dict,
        prepared: Any,
    ) -> tuple[Any, dict]:
        """把原生 MOT metadata 改写成分段式 Video+Action 因果顺序。"""
        metadata, diagnostics = super()._prepare_metadata(input_dict, prepared)
        chunk_size = int(input_dict.get("chunk_size", self.chunk_size))
        history_frames = int(self.history_frames)
        if metadata.frame_ids is None:
            raise ValueError("segmented distillation order requires frame_ids")

        num_frames = int(metadata.frame_ids.max().item()) + 1
        frame_ids = torch.arange(num_frames, device=metadata.device)
        history_segments = (history_frames + chunk_size - 1) // chunk_size
        history_order = torch.div(
            frame_ids,
            chunk_size,
            rounding_mode="floor",
        ) * 2
        target_order = 2 * history_segments + 2 * (frame_ids - history_frames)
        video_order = torch.where(
            frame_ids < history_frames,
            history_order,
            target_order,
        )
        action_order = video_order + 1
        frame_order = video_order[metadata.frame_ids]
        metadata.order_ids = torch.where(
            metadata.stream_ids == 1,
            action_order[metadata.frame_ids],
            frame_order,
        )
        metadata.cache_key = None
        metadata.structure_cache_key = None
        return metadata

    def _text(self, text_emb: torch.Tensor, batch_size: int) -> torch.Tensor:
        if text_emb.shape[0] == 1 and batch_size != 1:
            text_emb = text_emb.expand(batch_size, *text_emb.shape[1:])
        if text_emb.shape[0] != batch_size:
            raise ValueError(
                f"text batch {text_emb.shape[0]} does not match {batch_size}"
            )
        embedder = self.condition_embedder.text_embedder
        dtype = embedder.linear_1.weight.dtype
        return embedder(text_emb.to(device=next(self.parameters()).device, dtype=dtype))

    def _stream_axes(
        self,
        sample: torch.Tensor,
        *,
        frame_ids: Iterable[int] | torch.Tensor,
        timesteps: float | int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _channels, frames = sample.shape[:3]
        frame_ids = torch.as_tensor(frame_ids, dtype=torch.long, device=sample.device)
        if frame_ids.ndim == 1:
            frame_ids = frame_ids[None].expand(batch_size, -1)
        timesteps = torch.as_tensor(timesteps, dtype=torch.float32, device=sample.device)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size, frames)
        elif timesteps.ndim == 1:
            if timesteps.numel() == 1:
                timesteps = timesteps.expand(batch_size, frames)
            elif timesteps.numel() == frames:
                timesteps = timesteps[None].expand(batch_size, -1)
        return frame_ids, timesteps

    def _video_rotary(
        self,
        frame_ids: torch.Tensor,
        *,
        views: int,
        h_tokens: int,
        w_tokens: int,
    ) -> torch.Tensor:
        # LINK: distillation/model/autoregressive_mot.py:890
        # frame_ids: [B, F] absolute frame index per sample position,
        # e.g. [[0, 1, 2], [5, 6, 7]].
        batch_size, frames = frame_ids.shape
        device = frame_ids.device
        # Stage 1/4: Spatial token coordinates along height/width.
        height = torch.arange(h_tokens, device=device, dtype=torch.float32)
        width = torch.arange(w_tokens, device=device, dtype=torch.float32)
        grids = []
        # Stage 2/4: Per sample, build the [F, V, H, W] frame/view/height/width
        # grid, stack the four positional channels [frame, height, width, 0],
        # and flatten each token column to [4, F*V*H*W]. The view index only
        # affects token layout, not an explicit coordinate channel.
        for batch in range(batch_size):
            frame = frame_ids[batch].to(dtype=torch.float32)
            ff, _vv, hh, ww = torch.meshgrid(
                frame,
                torch.arange(views, device=device, dtype=torch.float32),
                height,
                width,
                indexing="ij",
            )
            grids.append(
                torch.stack([ff, hh, ww, torch.zeros_like(ff)], dim=0).reshape(4, -1)
            )
        # Stage 3/4: Stack per-sample grids into [B, 4, N] with N=F*V*H*W.
        grid = torch.stack(grids, dim=0)
        expected = frames * views * h_tokens * w_tokens
        if grid.shape[-1] != expected:
            raise RuntimeError("video rotary grid length mismatch")

        # Stage 4/4: RoPE over the positional channels, then add a head axis
        # ([B, D, N] -> [B, D, 1, N]) for attention-head broadcasting.
        return self.rope(grid)[:, :, None]

    def _action_rotary(
        self,
        frame_ids: torch.Tensor,
        *,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        # LINK: wan_va/modules/model_va_mot.py:914
        # frame_ids: [B, F] frame index per position; tokens_per_frame is the
        # number of action tokens packed into each frame.
        # Stage 1/3: Per-frame token slot indices [K] -> [1, 1, K].
        slots = torch.arange(
            tokens_per_frame,
            device=frame_ids.device,
            dtype=torch.long,
        )[None, None]
        # Stage 2/3: One continuous position per token:
        # positions[b, f, k] = frame_ids[b, f] * K + k -> [B, F, K].
        positions = frame_ids[:, :, None] * tokens_per_frame + slots
        # Stage 3/3: Flatten per-frame slots into one [B, F*K] sequence and
        # rotate those positions with action_rope.
        return self.action_rope(positions.reshape(frame_ids.shape[0], -1))

    def _video_input(
        self,
        latents: torch.Tensor,
        *,
        frame_ids: torch.Tensor,
        timesteps: torch.Tensor,
        stream_ids: torch.Tensor,
    ) -> tuple[AutoregressiveStreamInput, tuple[int, ...]]:
        hidden, shape = self._embed_video(latents, stream_ids)
        batch_size, frames, views, _height, _width, h_tokens, w_tokens = shape
        tokens_per_frame = views * h_tokens * w_tokens
        _, conditioning = self._time_embed_repeated(
            timesteps,
            tokens_per_frame,
            hidden.dtype,
            action=False,
        )
        return (
            AutoregressiveStreamInput(
                hidden=hidden,
                conditioning=conditioning,
                rotary=self._video_rotary(
                    frame_ids,
                    views=views,
                    h_tokens=h_tokens,
                    w_tokens=w_tokens,
                ),
                block_kind="video",
            ),
            shape,
        )

    def _action_input(
        self,
        actions: torch.Tensor,
        *,
        frame_ids: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> AutoregressiveStreamInput:
        hidden = self._embed_action(actions)
        _batch_size, _channels, _frames, action_per_frame, width = actions.shape
        tokens_per_frame = action_per_frame * width
        _, conditioning = self._time_embed_repeated(
            timesteps,
            tokens_per_frame,
            hidden.dtype,
            action=True,
        )
        return AutoregressiveStreamInput(
            hidden=hidden,
            conditioning=conditioning,
            rotary=self._action_rotary(
                frame_ids,
                tokens_per_frame=tokens_per_frame,
            ),
            block_kind="action",
        )

    def _run_stream(
        self,
        stream: AutoregressiveStreamInput,
        *,
        text_emb: torch.Tensor,
        cache,
        transaction_id: int,
    ) -> torch.Tensor:
        batch_size = stream.hidden.shape[0]
        text = self._text(text_emb, batch_size)
        hidden = stream.hidden

        for layer_id, mot_block in enumerate(self.mot_blocks):
            hidden = mot_block(
                AutoregressiveMOTLayerRequest(
                    stream=stream,
                    hidden_state=hidden,
                    text=text,
                    cache=cache,
                    transaction_id=transaction_id,
                    layer_id=layer_id,
                )
            )

        return hidden

    def _run_transaction(
        self,
        stream: AutoregressiveStreamInput,
        *,
        text_emb: torch.Tensor,
        cache,
        transaction_id: int,
        commit: bool = False,
    ) -> torch.Tensor:
        try:
            hidden = self._run_stream(
                stream,
                text_emb=text_emb,
                cache=cache,
                transaction_id=transaction_id,
            )
            if commit:
                cache.commit(transaction_id)
            return hidden
        finally:
            cache.discard(transaction_id)

    def predict_video(
        self,
        sample: torch.Tensor,
        *,
        timestep: float | int | torch.Tensor,
        frame_ids: Iterable[int] | torch.Tensor,
        stream_ids: torch.Tensor,
        text_emb: torch.Tensor,
        cache,
    ) -> torch.Tensor:
        frame_ids, timesteps = self._stream_axes(
            sample,
            frame_ids=frame_ids,
            timesteps=timestep,
        )
        transaction_id = cache.new_transaction_id()
        stream, shape = self._video_input(
            sample,
            frame_ids=frame_ids,
            timesteps=timesteps,
            stream_ids=stream_ids,
        )
        hidden = self._run_transaction(
            stream,
            text_emb=text_emb,
            cache=cache,
            transaction_id=transaction_id,
        )
        return self._final_video(hidden, timesteps, shape)

    def predict_action(
        self,
        sample: torch.Tensor,
        *,
        timestep: float | int | torch.Tensor,
        frame_ids: Iterable[int] | torch.Tensor,
        text_emb: torch.Tensor,
        cache,
    ) -> torch.Tensor:
        frame_ids, timesteps = self._stream_axes(
            sample,
            frame_ids=frame_ids,
            timesteps=timestep,
        )
        transaction_id = cache.new_transaction_id()
        stream = self._action_input(
            sample,
            frame_ids=frame_ids,
            timesteps=timesteps,
        )
        hidden = self._run_transaction(
            stream,
            text_emb=text_emb,
            cache=cache,
            transaction_id=transaction_id,
        )
        return self._final_action(hidden, timesteps, tuple(sample.shape))

    def commit_video(
        self,
        latents: torch.Tensor,
        *,
        frame_ids: Iterable[int] | torch.Tensor,
        stream_ids: torch.Tensor,
        text_emb: torch.Tensor,
        cache,
    ) -> None:
        frame_ids, zeros = self._stream_axes(
            latents,
            frame_ids=frame_ids,
            timesteps=0,
        )
        transaction_id = cache.new_transaction_id()
        clean, _ = self._video_input(
            latents,
            frame_ids=frame_ids,
            timesteps=zeros,
            stream_ids=stream_ids,
        )
        self._run_transaction(
            clean,
            text_emb=text_emb,
            cache=cache,
            transaction_id=transaction_id,
            commit=True,
        )

    def commit_action(
        self,
        actions: torch.Tensor,
        *,
        frame_ids: Iterable[int] | torch.Tensor,
        text_emb: torch.Tensor,
        cache,
    ) -> None:
        frame_ids, zeros = self._stream_axes(
            actions,
            frame_ids=frame_ids,
            timesteps=0,
        )
        transaction_id = cache.new_transaction_id()
        clean = self._action_input(
            actions,
            frame_ids=frame_ids,
            timesteps=zeros,
        )
        self._run_transaction(
            clean,
            text_emb=text_emb,
            cache=cache,
            transaction_id=transaction_id,
            commit=True,
        )

    def forward(self, input_dict, *, mode: str, return_diagnostics: bool = False, **kwargs):
        if mode != "self_rollout":
            return super().forward(
                input_dict,
                mode=mode,
                return_diagnostics=return_diagnostics,
                **kwargs,
            )
        if not isinstance(input_dict, AutoregressiveModelRequest):
            raise TypeError("mode='self_rollout' requires AutoregressiveModelRequest")
        return self.forward_autoregressive(input_dict)

    @torch.no_grad()
    def forward_autoregressive(
        self,
        request: AutoregressiveModelRequest,
    ) -> AutoregressiveModelOutput:
        """Execute one rollout operation through the AR model boundary."""

        p = request.payload
        op = request.operation
        if op == "predict_video":
            prediction = self.predict_video(**p)
            return AutoregressiveModelOutput(prediction=prediction)
        if op == "predict_action":
            prediction = self.predict_action(**p)
            return AutoregressiveModelOutput(prediction=prediction)
        if op == "commit_video":
            self.commit_video(**p)
            return AutoregressiveModelOutput()
        if op == "commit_action":
            self.commit_action(**p)
            return AutoregressiveModelOutput()
        raise ValueError(f"unsupported autoregressive operation {op!r}")
