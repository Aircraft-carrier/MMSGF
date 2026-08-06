"""Independent autoregressive MoT model family.

The class is parameter-compatible with ThreeDVAMOTTransformer3DModel.  Its
training path owns the AR order natively; rollout integration is added through
the normal model forward boundary so FSDP hooks remain effective.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch

from distillation.self_rollout.attention import segmented_orders
from wan_va.modules.model_3dva_mot import (
    ThreeDVAMOTBlock,
    ThreeDVAMOTTransformer3DModel,
)

from .autoregressive_types import AutoregressiveProfile
from .autoregressive_types import (
    AutoregressiveGeometryJointRequest,
    AutoregressiveMOTLayerRequest,
    AutoregressiveModelOutput,
    AutoregressiveModelRequest,
    AutoregressiveStreamInput,
)
from .autoregressive_vggto import AutoregressiveVGGTOGeometryTower


class AutoregressiveThreeDVAMOTBlock(ThreeDVAMOTBlock):
    """Parameter-compatible MoT block reserved for native AR execution."""

    def forward(self, states, *args, **kwargs):
        if isinstance(states, AutoregressiveGeometryJointRequest):
            return self.forward_geometry_incremental(states)
        if isinstance(states, AutoregressiveMOTLayerRequest):
            return self.forward_incremental(states)
        return super().forward(states, *args, **kwargs)

    def forward_incremental(self, request: AutoregressiveMOTLayerRequest):
        from distillation.self_rollout.attention import (
            STREAM_GEOMETRY,
            build_cache_selection,
            indexed_attention,
        )
        from distillation.self_rollout.cache import KVSegment

        streams = request.streams
        hidden_states = request.hidden_states
        current_meta = streams[0].metadata
        if len(streams) > 1:
            from distillation.self_rollout.attention import TokenMetadataBatch

            current_meta = TokenMetadataBatch.cat([stream.metadata for stream in streams])
        qkv_parts = []
        modulations = []
        blocks = []
        for stream, hidden in zip(streams, hidden_states):
            block = self.video_block if stream.block_kind == "video" else self.action_block
            modulation = self._modulation(block, stream.conditioning)
            qkv_parts.append(self._self_qkv(block, hidden, modulation, stream.rotary))
            blocks.append(block)
            modulations.append(modulation)
        query, current_key, current_value = (
            torch.cat(parts, dim=1) for parts in zip(*qkv_parts)
        )
        request.state.mot_cache.append_transaction(
            request.layer_id,
            request.transaction_id,
            KVSegment(
                current_key,
                current_value,
                current_meta,
                stream_id=request.stream_id,
            ),
        )
        layer_id = request.layer_id
        key, value, key_meta = request.state.mot_cache.materialize(
            layer_id,
            transaction_id=request.transaction_id,
        )
        query_valid, key_valid = build_cache_selection(current_meta, key_meta)
        attention_output, _ = indexed_attention(
            query,
            key,
            value,
            query_valid=query_valid,
            key_valid=key_valid,
        )
        lengths = [hidden.shape[1] for hidden in hidden_states]
        attention_parts = torch.split(attention_output, lengths, dim=1)
        text_attn = self.video_block.attn2
        text_key = text_attn.norm_k(text_attn.to_k(request.text)).unflatten(
            2, (text_attn.heads, -1)
        )
        text_value = text_attn.to_v(request.text).unflatten(
            2, (text_attn.heads, -1)
        )
        updated = []
        for block, hidden, attended, modulation in zip(
            blocks, hidden_states, attention_parts, modulations
        ):
            self_output = self._attention_output(block.attn1, attended)
            updated.append(
                self._finish_block(
                    block,
                    hidden,
                    self_output,
                    modulation,
                    text_key,
                    text_value,
                )
            )
        return updated

    def forward_geometry_incremental(self, request: AutoregressiveGeometryJointRequest):
        from distillation.self_rollout.attention import (
            STREAM_GEOMETRY,
            build_cache_selection,
            indexed_attention,
        )
        from distillation.self_rollout.cache import KVSegment

        geometry = self.geometry
        if geometry is None:
            raise ValueError("geometry incremental request sent to an odd MoT layer")
        registers = request.registers.to(dtype=getattr(geometry.norm1, "weight", request.registers).dtype)
        query, key, value = geometry.qkv_project(registers.flatten(1, 2), request.rotary)
        request.cache.mot_cache.append_transaction(
            request.layer_id,
            request.transaction_id,
            KVSegment(key, value, request.metadata, stream_id=STREAM_GEOMETRY),
        )
        cache_key, cache_value, cache_meta = request.cache.mot_cache.materialize(
            request.layer_id,
            transaction_id=request.transaction_id,
            stream_id=STREAM_GEOMETRY,
        )
        query_valid, key_valid = build_cache_selection(request.metadata, cache_meta)
        attended, visible = indexed_attention(
            query,
            cache_key,
            cache_value,
            query_valid=query_valid,
            key_valid=key_valid,
        )
        delta = geometry.attn_delta(attended).reshape_as(registers)
        visible = visible.reshape(
            registers.shape[0], request.groups * request.slots * request.views, registers.shape[2]
        )
        delta = torch.where(visible[:, :, :, None], delta, torch.zeros_like(delta))
        residual = registers + delta
        updated = residual + geometry.ffn_delta(residual)
        if request.slot_valid_mask is not None:
            valid = (
                request.slot_valid_mask[:, :, :, None]
                .to(device=updated.device, dtype=torch.bool)
                .expand(-1, -1, -1, request.views)
                .reshape(updated.shape[0], request.groups * request.slots * request.views)
            )
            updated = torch.where(valid[:, :, None, None], updated, registers)
        return updated


class AutoregressiveThreeDVAMOTTransformer3DModel(
    ThreeDVAMOTTransformer3DModel
):
    """MOT transformer with AR metadata and AR component construction."""

    _no_split_modules = ["AutoregressiveThreeDVAMOTBlock"]
    _repeated_blocks = ["AutoregressiveThreeDVAMOTBlock"]

    def __init__(self, *args, generation_profile=None, **kwargs):
        if isinstance(generation_profile, dict):
            generation_profile = AutoregressiveProfile.from_generation_shape(generation_profile)
        self.generation_profile = generation_profile or AutoregressiveProfile(
            profile_name="segmented_history_strict_geometry_v1",
            profile_version=2,
            order_mode="segmented",
            history_frames=4,
            chunk_size=4,
            window_size=16,
            geometry_relation="segmented_order_causal",
            x_to_g_relation="strict_order",
        )
        self.generation_profile.validate()
        super().__init__(*args, **kwargs)
        self.vggto.configure_generation_profile(self.generation_profile)
        self.register_to_config(generation_profile=self.generation_profile.as_dict())

    def _build_vggto_tower(self, **kwargs):
        return AutoregressiveVGGTOGeometryTower(
            **kwargs,
            generation_profile=self.generation_profile,
        )

    def _build_mot_block(self, **kwargs):
        return AutoregressiveThreeDVAMOTBlock(**kwargs)

    def configure_generation_profile(self, generation_shape: Any) -> None:
        profile = (
            generation_shape
            if isinstance(generation_shape, AutoregressiveProfile)
            else AutoregressiveProfile.from_generation_shape(generation_shape)
        )
        self.generation_profile = profile
        self.vggto.configure_generation_profile(profile)
        self.register_to_config(generation_profile=profile.as_dict())

    def _prepare_metadata(self, input_dict, prepared, geometry):
        x_meta, mot_meta, diagnostics = super()._prepare_metadata(
            input_dict,
            prepared,
            geometry,
        )
        profile = self.generation_profile
        chunk_size = int(input_dict.get("chunk_size", profile.chunk_size))
        return (
            self._apply_segmented_order(
                x_meta,
                history_frames=profile.history_frames,
                chunk_size=chunk_size,
            ),
            self._apply_segmented_order(
                mot_meta,
                history_frames=profile.history_frames,
                chunk_size=chunk_size,
            ),
            diagnostics,
        )

    @staticmethod
    def _segmented_frame_orders(
        *,
        num_frames: int,
        history_frames: int,
        chunk_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_order = segmented_orders(
            torch.arange(num_frames, device=device),
            history_frames=history_frames,
            chunk_size=chunk_size,
        )
        return video_order, video_order + 1

    @classmethod
    def _apply_segmented_order(
        cls,
        metadata,
        *,
        history_frames: int,
        chunk_size: int,
    ):
        if metadata.frame_ids is None:
            raise ValueError("segmented distillation order requires frame_ids")
        video_order, action_order = cls._segmented_frame_orders(
            num_frames=int(metadata.frame_ids.max().item()) + 1,
            history_frames=history_frames,
            chunk_size=chunk_size,
            device=metadata.device,
        )
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
        return self.rope(grid)[:, :, None]

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
        return self.action_rope(positions.reshape(frame_ids.shape[0], -1))

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
        source,
        version_id: int,
        valid_frames: torch.Tensor | None,
    ) -> tuple[AutoregressiveStreamInput, tuple[int, ...]]:
        from distillation.self_rollout.attention import STREAM_VIDEO, build_token_metadata

        hidden, shape = self._embed_video(latents, stream_ids)
        batch_size, frames, views, _height, _width, h_tokens, w_tokens = shape
        tokens_per_frame = views * h_tokens * w_tokens
        _, conditioning = self._time_embed_repeated(
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
            history_frames=self.generation_profile.history_frames,
            chunk_size=self.generation_profile.chunk_size,
            device=hidden.device,
            valid_ids=valid_frames,
            committed=committed,
            transaction_id=transaction_id,
            source_id=int(source),
            version_id=version_id,
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
        source,
        version_id: int,
        valid_mask: torch.Tensor | None,
    ) -> AutoregressiveStreamInput:
        from distillation.self_rollout.attention import STREAM_ACTION, build_token_metadata

        hidden = self._embed_action(actions)
        batch_size, _channels, frames, action_per_frame, width = actions.shape
        tokens_per_frame = action_per_frame * width
        _, conditioning = self._time_embed_repeated(
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
            history_frames=self.generation_profile.history_frames,
            chunk_size=self.generation_profile.chunk_size,
            device=hidden.device,
            valid_ids=token_valid,
            committed=committed,
            transaction_id=transaction_id,
            source_id=int(source),
            version_id=version_id,
        )
        return AutoregressiveStreamInput(
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
        streams: list[AutoregressiveStreamInput],
        *,
        text_emb: torch.Tensor,
        state,
        transaction_id: int,
    ) -> list[torch.Tensor]:
        from distillation.self_rollout.attention import (
            STREAM_ACTION,
            STREAM_VIDEO,
            TokenMetadataBatch,
        )
        from distillation.self_rollout.cache import KVSegment

        if not streams:
            raise ValueError("at least one stream is required")
        batch_size = streams[0].hidden.shape[0]
        if any(stream.hidden.shape[0] != batch_size for stream in streams):
            raise ValueError("all streams must share a batch size")
        text = self._text(text_emb, batch_size)
        hidden_states = [stream.hidden for stream in streams]
        current_stream_id = None
        if all(stream.block_kind == "video" for stream in streams):
            current_stream_id = STREAM_VIDEO
        elif all(stream.block_kind == "action" for stream in streams):
            current_stream_id = STREAM_ACTION
        current_meta = (
            streams[0].metadata
            if len(streams) == 1
            else TokenMetadataBatch.cat([stream.metadata for stream in streams])
        )

        for layer_id, mot_block in enumerate(self.mot_blocks):
            if hasattr(mot_block, "forward_incremental"):
                hidden_states = mot_block(
                    AutoregressiveMOTLayerRequest(
                        streams=streams,
                        hidden_states=hidden_states,
                        text=text,
                        state=state,
                        transaction_id=transaction_id,
                        layer_id=layer_id,
                        stream_id=current_stream_id,
                    )
                )
                continue
            from distillation.self_rollout import attention as attention_module

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
            state.mot_cache.append_transaction(
                layer_id,
                transaction_id,
                KVSegment(
                    current_key,
                    current_value,
                    current_meta,
                    stream_id=current_stream_id,
                ),
            )
            key, value, key_meta = state.mot_cache.materialize(
                layer_id,
                transaction_id=transaction_id,
            )
            query_valid, key_valid = attention_module.build_cache_selection(
                current_meta, key_meta
            )
            attended, _visible = attention_module.indexed_attention(
                query,
                key,
                value,
                query_valid=query_valid,
                key_valid=key_valid,
            )
            lengths = [hidden.shape[1] for hidden in hidden_states]
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
        stream: AutoregressiveStreamInput,
        *,
        text_emb: torch.Tensor,
        state,
        transaction_id: int,
        commit_source=None,
    ) -> torch.Tensor:
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
        state,
        *,
        frame_id: int,
        stream_id: int,
        source,
        version_id: int,
    ) -> None:
        from distillation.self_rollout.attention import NOISE_CLEAN

        state.mot_cache.assert_committed_frame(
            range(len(self.mot_blocks)),
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
        state,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from distillation.self_rollout.attention import NOISE_NOISY
        from distillation.self_rollout.state import CacheSource

        if sample.shape[2] != 1:
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
        return self._final_video(hidden, timesteps, shape)

    def predict_action(
        self,
        sample: torch.Tensor,
        *,
        timestep: float | int | torch.Tensor,
        frame_id: int,
        text_emb: torch.Tensor,
        state,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from distillation.self_rollout.attention import NOISE_NOISY
        from distillation.self_rollout.state import CacheSource

        if sample.shape[2] != 1:
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
        return self._final_action(hidden, timesteps, tuple(sample.shape))

    def commit_video(
        self,
        latents: torch.Tensor,
        *,
        frame_ids: Iterable[int] | torch.Tensor,
        stream_ids: torch.Tensor,
        text_emb: torch.Tensor,
        state,
        source,
        version_id: int,
        valid_frames: torch.Tensor | None = None,
    ) -> None:
        from distillation.self_rollout.attention import NOISE_CLEAN

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
        state,
        source,
        version_id: int,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        from distillation.self_rollout.attention import NOISE_CLEAN

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
        state,
        *,
        frame_id: int,
        source,
        version_id: int,
    ) -> None:
        from distillation.self_rollout.attention import STREAM_VIDEO

        self._assert_clean_commit(
            state,
            frame_id=frame_id,
            stream_id=STREAM_VIDEO,
            source=source,
            version_id=version_id,
        )

    def assert_action_commit(
        self,
        state,
        *,
        frame_id: int,
        source,
        version_id: int,
    ) -> None:
        from distillation.self_rollout.attention import STREAM_ACTION

        self._assert_clean_commit(
            state,
            frame_id=frame_id,
            stream_id=STREAM_ACTION,
            source=source,
            version_id=version_id,
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
        if op == "encode_geometry_history":
            frame = self.vggto.encode_history_and_commit(self, **p)
            return AutoregressiveModelOutput(geometry_frame=frame)
        if op == "encode_geometry":
            frame = self.vggto.encode_and_commit(self, **p)
            return AutoregressiveModelOutput(geometry_frame=frame)
        raise ValueError(f"unsupported autoregressive operation {op!r}")
