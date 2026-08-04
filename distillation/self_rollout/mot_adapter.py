"""Incremental video/action adapter for the existing ThreeDVAMOT weights.

The adapter deliberately uses model submodules without calling the fixed-window
``forward_inference`` path.  All cache ownership and attention policy live in
``distillation``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .attention import (
    NOISE_CLEAN,
    NOISE_NOISY,
    STREAM_ACTION,
    STREAM_VIDEO,
    TokenMetadataBatch,
    build_cache_visibility,
    build_token_metadata,
    incremental_attention,
)
from .cache import KVSegment
from .state import CacheSource, RolloutState


@dataclass(frozen=True, slots=True)
class _StreamInput:
    hidden: torch.Tensor
    conditioning: torch.Tensor
    rotary: torch.Tensor
    metadata: TokenMetadataBatch
    block_kind: str


class MOTIncrementalAdapter:
    """Run current video/action streams against committed rectangular K/V.

    The two transaction lifetimes are intentionally small::

        prediction: [committed K/V | current noisy K/V] -> output -> discard
        commit:     [committed K/V | current clean K/V] -> output -> commit

    A commit never creates a noisy cache segment.  Noisy K/V exists only so a
    denoising query can attend to its own transaction during that model call.
    """

    def __init__(
        self,
        transformer,
        *,
        history_frames: int,
        chunk_size: int,
        window_size: int,
    ) -> None:
        self.model = getattr(transformer, "module", transformer)
        self.history_frames = int(history_frames)
        self.chunk_size = int(chunk_size)
        self.window_size = int(window_size)
        if self.history_frames <= 0 or self.chunk_size <= 0 or self.window_size < 0:
            raise ValueError("invalid history/chunk/window configuration")
        required = (
            "mot_blocks",
            "_embed_video",
            "_embed_action",
            "_time_embed_repeated",
            "_final_video",
            "_final_action",
            "rope",
            "action_rope",
            "condition_embedder",
        )
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise TypeError(
                "self_rollout requires ThreeDVAMOT-compatible transformer attributes: "
                + ", ".join(missing)
            )

    def _text(self, text_emb: torch.Tensor, batch_size: int) -> torch.Tensor:
        if text_emb.shape[0] == 1 and batch_size != 1:
            text_emb = text_emb.expand(batch_size, *text_emb.shape[1:])
        if text_emb.shape[0] != batch_size:
            raise ValueError(
                f"text batch {text_emb.shape[0]} does not match {batch_size}"
            )
        embedder = self.model.condition_embedder.text_embedder
        dtype = embedder.linear_1.weight.dtype
        return embedder(text_emb.to(device=next(self.model.parameters()).device, dtype=dtype))

    @staticmethod
    def _normalize_frame_ids(
        frame_ids: Iterable[int] | torch.Tensor,
        *,
        batch_size: int,
        frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        frame_ids = torch.as_tensor(frame_ids, dtype=torch.long, device=device)
        if frame_ids.ndim == 1:
            frame_ids = frame_ids[None].expand(batch_size, -1)
        if tuple(frame_ids.shape) != (batch_size, frames):
            raise ValueError(
                f"frame_ids must be [{batch_size},{frames}], got {tuple(frame_ids.shape)}"
            )
        return frame_ids

    @staticmethod
    def _normalize_timesteps(
        timesteps: float | int | torch.Tensor,
        *,
        batch_size: int,
        frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        timesteps = torch.as_tensor(timesteps, dtype=torch.float32, device=device)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size, frames)
        elif timesteps.ndim == 1:
            if timesteps.numel() == 1:
                timesteps = timesteps.expand(batch_size, frames)
            elif timesteps.numel() == frames:
                timesteps = timesteps[None].expand(batch_size, -1)
        if tuple(timesteps.shape) != (batch_size, frames):
            raise ValueError(
                f"timesteps must broadcast to [{batch_size},{frames}], got {tuple(timesteps.shape)}"
            )
        return timesteps

    def _stream_axes(
        self,
        sample: torch.Tensor,
        *,
        frame_ids: Iterable[int] | torch.Tensor,
        timesteps: float | int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize the shared frame/timestep axes for video and action."""

        batch_size, _channels, frames = sample.shape[:3]
        return (
            self._normalize_frame_ids(
                frame_ids,
                batch_size=batch_size,
                frames=frames,
                device=sample.device,
            ),
            self._normalize_timesteps(
                timesteps,
                batch_size=batch_size,
                frames=frames,
                device=sample.device,
            ),
        )

    def _video_rotary(
        self,
        frame_ids: torch.Tensor,
        *,
        views: int,
        h_tokens: int,
        w_tokens: int,
    ) -> torch.Tensor:
        batch_size, frames = frame_ids.shape
        device = frame_ids.device
        height = torch.arange(h_tokens, device=device, dtype=torch.float32)
        width = torch.arange(w_tokens, device=device, dtype=torch.float32)
        grids = []
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
        grid = torch.stack(grids, dim=0)
        expected = frames * views * h_tokens * w_tokens
        if grid.shape[-1] != expected:
            raise RuntimeError("video rotary grid length mismatch")
        return self.model.rope(grid)[:, :, None]

    def _action_rotary(
        self,
        frame_ids: torch.Tensor,
        *,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        slots = torch.arange(
            tokens_per_frame,
            device=frame_ids.device,
            dtype=torch.long,
        )[None, None]
        positions = frame_ids[:, :, None] * tokens_per_frame + slots
        return self.model.action_rope(positions.reshape(frame_ids.shape[0], -1))

    def _video_input(
        self,
        latents: torch.Tensor,
        *,
        frame_ids: torch.Tensor,
        timesteps: torch.Tensor,
        stream_ids: torch.Tensor,
        noise_id: int,
        committed: bool,
        transaction_id: int,
        source: CacheSource,
        version_id: int,
        valid_frames: torch.Tensor | None,
    ) -> tuple[_StreamInput, tuple[int, ...]]:
        hidden, shape = self.model._embed_video(latents, stream_ids)
        batch_size, frames, views, _height, _width, h_tokens, w_tokens = shape
        tokens_per_frame = views * h_tokens * w_tokens
        _, conditioning = self.model._time_embed_repeated(
            timesteps,
            tokens_per_frame,
            hidden.dtype,
            action=False,
        )
        metadata = build_token_metadata(
            batch_size=batch_size,
            frame_ids=frame_ids,
            tokens_per_frame=tokens_per_frame,
            stream_id=STREAM_VIDEO,
            noise_id=noise_id,
            history_frames=self.history_frames,
            chunk_size=self.chunk_size,
            device=hidden.device,
            valid_ids=valid_frames,
            committed=committed,
            transaction_id=transaction_id,
            source_id=int(source),
            version_id=version_id,
        )
        return (
            _StreamInput(
                hidden=hidden,
                conditioning=conditioning,
                rotary=self._video_rotary(
                    frame_ids,
                    views=views,
                    h_tokens=h_tokens,
                    w_tokens=w_tokens,
                ),
                metadata=metadata,
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
        noise_id: int,
        committed: bool,
        transaction_id: int,
        source: CacheSource,
        version_id: int,
        valid_mask: torch.Tensor | None,
    ) -> _StreamInput:
        hidden = self.model._embed_action(actions)
        batch_size, _channels, frames, action_per_frame, width = actions.shape
        tokens_per_frame = action_per_frame * width
        _, conditioning = self.model._time_embed_repeated(
            timesteps,
            tokens_per_frame,
            hidden.dtype,
            action=True,
        )
        token_valid = None
        if valid_mask is not None:
            token_valid = valid_mask.to(device=hidden.device, dtype=torch.bool).any(dim=1)
            token_valid = token_valid.reshape(batch_size, frames * tokens_per_frame)
        metadata = build_token_metadata(
            batch_size=batch_size,
            frame_ids=frame_ids,
            tokens_per_frame=tokens_per_frame,
            stream_id=STREAM_ACTION,
            noise_id=noise_id,
            history_frames=self.history_frames,
            chunk_size=self.chunk_size,
            device=hidden.device,
            valid_ids=token_valid,
            committed=committed,
            transaction_id=transaction_id,
            source_id=int(source),
            version_id=version_id,
        )
        return _StreamInput(
            hidden=hidden,
            conditioning=conditioning,
            rotary=self._action_rotary(
                frame_ids,
                tokens_per_frame=tokens_per_frame,
            ),
            metadata=metadata,
            block_kind="action",
        )

    def _run_streams(
        self,
        streams: list[_StreamInput],
        *,
        text_emb: torch.Tensor,
        state: RolloutState,
        transaction_id: int,
    ) -> list[torch.Tensor]:
        if not streams:
            raise ValueError("at least one stream is required")
        batch_size = streams[0].hidden.shape[0]
        if any(stream.hidden.shape[0] != batch_size for stream in streams):
            raise ValueError("all streams must share a batch size")
        text = self._text(text_emb, batch_size)
        hidden_states = [stream.hidden for stream in streams]
        for layer_id, mot_block in enumerate(self.model.mot_blocks):
            qkv_parts = []
            modulations = []
            blocks = []
            for stream, hidden in zip(streams, hidden_states):
                block = (
                    mot_block.video_block
                    if stream.block_kind == "video"
                    else mot_block.action_block
                )
                modulation = mot_block._modulation(block, stream.conditioning)
                qkv_parts.append(
                    mot_block._self_qkv(
                        block,
                        hidden,
                        modulation,
                        stream.rotary,
                    )
                )
                blocks.append(block)
                modulations.append(modulation)
            query, current_key, current_value = (
                torch.cat(parts, dim=1) for parts in zip(*qkv_parts)
            )
            current_meta = TokenMetadataBatch.cat(
                [stream.metadata for stream in streams]
            )
            state.mot_cache.append_transaction(
                layer_id,
                transaction_id,
                KVSegment(current_key, current_value, current_meta),
            )
            key, value, key_meta = state.mot_cache.materialize(
                layer_id,
                transaction_id=transaction_id,
            )
            mask = build_cache_visibility(
                current_meta,
                key_meta,
                window_size=self.window_size,
            )
            attended = incremental_attention(query, key, value, mask)
            lengths = [stream.hidden.shape[1] for stream in streams]
            attended_parts = torch.split(attended, lengths, dim=1)

            text_attn = mot_block.video_block.attn2
            text_key = text_attn.norm_k(text_attn.to_k(text)).unflatten(
                2,
                (text_attn.heads, -1),
            )
            text_value = text_attn.to_v(text).unflatten(
                2,
                (text_attn.heads, -1),
            )
            hidden_states = [
                mot_block._finish_block(
                    block,
                    hidden,
                    mot_block._attention_output(block.attn1, attention_output),
                    modulation,
                    text_key,
                    text_value,
                )
                for block, hidden, attention_output, modulation in zip(
                    blocks,
                    hidden_states,
                    attended_parts,
                    modulations,
                )
            ]
        return hidden_states

    def _run_transaction(
        self,
        stream: _StreamInput,
        *,
        text_emb: torch.Tensor,
        state: RolloutState,
        transaction_id: int,
        commit_source: CacheSource | None = None,
    ) -> torch.Tensor:
        """Run one stream, optionally commit it, and always clear transaction state."""

        try:
            hidden = self._run_streams(
                [stream],
                text_emb=text_emb,
                state=state,
                transaction_id=transaction_id,
            )[0]
            if commit_source is not None:
                state.mot_cache.commit_transaction(
                    transaction_id,
                    source_id=int(commit_source),
                )
            return hidden
        finally:
            state.mot_cache.discard_transaction(transaction_id)

    def _assert_clean_commit(
        self,
        state: RolloutState,
        *,
        frame_id: int,
        stream_id: int,
        source: CacheSource,
        version_id: int,
    ) -> None:
        state.mot_cache.assert_committed_frame(
            range(len(self.model.mot_blocks)),
            frame_id=frame_id,
            stream_id=stream_id,
            noise_ids=(NOISE_CLEAN,),
            source_id=int(source),
            version_id=version_id,
        )

    def predict_video(
        self,
        sample: torch.Tensor,
        *,
        timestep: float | int | torch.Tensor,
        frame_id: int,
        stream_ids: torch.Tensor,
        text_emb: torch.Tensor,
        state: RolloutState,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frames = sample.shape[2]
        if frames != 1:
            raise ValueError("predict_video expects exactly one logical frame")
        frame_ids, timesteps = self._stream_axes(
            sample,
            frame_ids=[frame_id],
            timesteps=timestep,
        )
        transaction_id = state.new_transaction_id()
        stream, shape = self._video_input(
            sample,
            frame_ids=frame_ids,
            timesteps=timesteps,
            stream_ids=stream_ids,
            noise_id=NOISE_NOISY,
            committed=False,
            transaction_id=transaction_id,
            source=CacheSource.PREDICTED,
            version_id=0,
            valid_frames=valid_frames,
        )
        hidden = self._run_transaction(
            stream,
            text_emb=text_emb,
            state=state,
            transaction_id=transaction_id,
        )
        return self.model._final_video(hidden, timesteps, shape)

    def predict_action(
        self,
        sample: torch.Tensor,
        *,
        timestep: float | int | torch.Tensor,
        frame_id: int,
        text_emb: torch.Tensor,
        state: RolloutState,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frames = sample.shape[2]
        if frames != 1:
            raise ValueError("predict_action expects exactly one logical frame")
        frame_ids, timesteps = self._stream_axes(
            sample,
            frame_ids=[frame_id],
            timesteps=timestep,
        )
        transaction_id = state.new_transaction_id()
        stream = self._action_input(
            sample,
            frame_ids=frame_ids,
            timesteps=timesteps,
            noise_id=NOISE_NOISY,
            committed=False,
            transaction_id=transaction_id,
            source=CacheSource.PREDICTED,
            version_id=0,
            valid_mask=valid_mask,
        )
        hidden = self._run_transaction(
            stream,
            text_emb=text_emb,
            state=state,
            transaction_id=transaction_id,
        )
        return self.model._final_action(hidden, timesteps, tuple(sample.shape))

    def commit_video(
        self,
        latents: torch.Tensor,
        *,
        frame_ids: Iterable[int] | torch.Tensor,
        stream_ids: torch.Tensor,
        text_emb: torch.Tensor,
        state: RolloutState,
        source: CacheSource,
        version_id: int,
        valid_frames: torch.Tensor | None = None,
    ) -> None:
        frame_ids, zeros = self._stream_axes(
            latents,
            frame_ids=frame_ids,
            timesteps=0,
        )
        transaction_id = state.new_transaction_id()
        clean, _ = self._video_input(
            latents,
            frame_ids=frame_ids,
            timesteps=zeros,
            stream_ids=stream_ids,
            noise_id=NOISE_CLEAN,
            committed=False,
            transaction_id=transaction_id,
            source=source,
            version_id=version_id,
            valid_frames=valid_frames,
        )
        self._run_transaction(
            clean,
            text_emb=text_emb,
            state=state,
            transaction_id=transaction_id,
            commit_source=source,
        )

    def commit_action(
        self,
        actions: torch.Tensor,
        *,
        frame_ids: Iterable[int] | torch.Tensor,
        text_emb: torch.Tensor,
        state: RolloutState,
        source: CacheSource,
        version_id: int,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        frame_ids, zeros = self._stream_axes(
            actions,
            frame_ids=frame_ids,
            timesteps=0,
        )
        transaction_id = state.new_transaction_id()
        clean = self._action_input(
            actions,
            frame_ids=frame_ids,
            timesteps=zeros,
            noise_id=NOISE_CLEAN,
            committed=False,
            transaction_id=transaction_id,
            source=source,
            version_id=version_id,
            valid_mask=valid_mask,
        )
        self._run_transaction(
            clean,
            text_emb=text_emb,
            state=state,
            transaction_id=transaction_id,
            commit_source=source,
        )

    def assert_video_commit(
        self,
        state: RolloutState,
        *,
        frame_id: int,
        source: CacheSource,
        version_id: int,
    ) -> None:
        self._assert_clean_commit(
            state,
            frame_id=frame_id,
            stream_id=STREAM_VIDEO,
            source=source,
            version_id=version_id,
        )

    def assert_action_commit(
        self,
        state: RolloutState,
        *,
        frame_id: int,
        source: CacheSource,
        version_id: int,
    ) -> None:
        self._assert_clean_commit(
            state,
            frame_id=frame_id,
            stream_id=STREAM_ACTION,
            source=source,
            version_id=version_id,
        )
