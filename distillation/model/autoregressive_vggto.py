"""Autoregressive VGGTO tower constructed without runtime monkeypatching."""
from __future__ import annotations

import torch

from distillation.self_rollout.attention import (
    NOISE_GEOMETRY,
    STREAM_GEOMETRY,
    build_cache_selection,
    build_token_metadata,
    indexed_attention,
)
from distillation.self_rollout.cache import (
    EncodedGeometryFrame,
    GeometryRolloutCache,
    KVSegment,
)
from distillation.self_rollout.state import CacheSource
from distillation.self_rollout.attention import segmented_orders
from wan_va.modules.vggto_geometry import VGGTOGeometryTower
from wan_va.modules.vggto_vendored.layers.block import SelfAttentionBlock

from .autoregressive_types import AutoregressiveProfile
from .autoregressive_types import AutoregressiveGeometryRelationRequest


class AutoregressiveVGGTOBlock(SelfAttentionBlock):
    """Parameter-compatible VGGTO block reserved for future cache requests.

    The tensor/list path remains exactly the vendored implementation. Keeping a
    distinct type now gives FSDP and activation-checkpoint construction a stable
    owner for the later incremental request path without mutating global classes.
    """

    def forward(
        self,
        x_or_x_list,
        rope_or_rope_list=None,
        attn_bias=None,
        chunk_causal_spec=None,
        attention_backend: str = "dense",
        row_valid_mask=None,
        *,
        mot_mask_metadata=None,
    ):
        if isinstance(x_or_x_list, AutoregressiveGeometryRelationRequest):
            return self.forward_incremental(x_or_x_list)
        if mot_mask_metadata is None:
            return super().forward(
                x_or_x_list,
                rope_or_rope_list,
                attn_bias=attn_bias,
                chunk_causal_spec=chunk_causal_spec,
                attention_backend=attention_backend,
                row_valid_mask=row_valid_mask,
            )
        if not torch.is_tensor(x_or_x_list):
            raise TypeError("mot_mask_metadata requires a single tensor input")
        if attention_backend != "fa4":
            raise ValueError("mot_mask_metadata requires attention_backend='fa4'")
        if attn_bias is not None or chunk_causal_spec is not None or row_valid_mask is not None:
            raise ValueError("mot_mask_metadata cannot be combined with other attention masks")
        from wan_va.modules.fa4_attention import fa4_attention_from_meta

        x_norm = self.norm1(x_or_x_list)
        channels = x_norm.shape[-1]
        qkv = self.attn.qkv(x_norm).reshape(
            x_norm.shape[0], x_norm.shape[1], 3, self.attn.num_heads,
            channels // self.attn.num_heads,
        )
        query, key, value = qkv.unbind(dim=2)
        if self.attn.use_qk_norm:
            query = self.attn.q_norm(query)
            key = self.attn.k_norm(key)
        if rope_or_rope_list is not None:
            query_heads, key_heads = query.transpose(1, 2), key.transpose(1, 2)
            query_heads, key_heads = self.attn.apply_rope(
                query_heads, key_heads, rope_or_rope_list
            )
            query, key = query_heads.transpose(1, 2), key_heads.transpose(1, 2)
        out = fa4_attention_from_meta(query, key, value, mot_mask_metadata)
        out = self.attn.proj(out.reshape(x_norm.shape[0], x_norm.shape[1], channels))
        out = self.attn.proj_drop(out)
        x_attn = x_or_x_list + self.ls1(out)
        return x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

    def forward_incremental(self, request: AutoregressiveGeometryRelationRequest):
        from distillation.self_rollout.attention import (
            STREAM_GEOMETRY,
            build_cache_selection,
            indexed_attention,
        )
        from distillation.self_rollout.cache import KVSegment

        values = request.values.to(dtype=getattr(self.norm1.weight, "dtype", request.values.dtype))
        qkv = self.attn.qkv(self.norm1(values)).reshape(
            values.shape[0], values.shape[1], 3, self.attn.num_heads,
            values.shape[-1] // self.attn.num_heads,
        )
        query, key, value = qkv.unbind(dim=2)
        if self.attn.use_qk_norm:
            query = self.attn.q_norm(query)
            key = self.attn.k_norm(key)
        if request.rope is not None:
            query_heads, key_heads = query.transpose(1, 2), key.transpose(1, 2)
            query_heads, key_heads = self.attn.apply_rope(query_heads, key_heads, request.rope)
            query, key = query_heads.transpose(1, 2), key_heads.transpose(1, 2)
        request.cache.relation_cache.append_transaction(
            request.layer_id,
            request.transaction_id,
            KVSegment(key, value, request.metadata, stream_id=STREAM_GEOMETRY),
        )
        cache_key, cache_value, cache_meta = request.cache.relation_cache.materialize(
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
        projected = self.attn.proj(attended.flatten(2, 3))
        projected = self.attn.proj_drop(projected)
        projected = torch.where(visible[:, :, None], projected, torch.zeros_like(projected))
        values = values + self.ls1(projected)
        values = values + self.ls2(self.mlp(self.norm2(values)))
        return values


class AutoregressiveVGGTOGeometryTower(VGGTOGeometryTower):
    """VGGTO tower with the segmented AR relation order as native behavior."""

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

    def _new_block(self, mlp_ratio: float) -> SelfAttentionBlock:
        return AutoregressiveVGGTOBlock(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_ratio=mlp_ratio,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            ffn_layer=self._mlp_layer(),
            init_values=1e-5,
            use_qk_norm=True,
            mask_k_bias=True,
        )

    @staticmethod
    def _mlp_layer():
        # Import lazily to preserve the vendored module's optional import order.
        from wan_va.modules.vggto_vendored.layers.ffn_layers import Mlp

        return Mlp

    def configure_generation_profile(self, profile: AutoregressiveProfile) -> None:
        profile.validate()
        self.generation_profile = profile

    @staticmethod
    def _norm_input_dtype(block, fallback: torch.dtype) -> torch.dtype:
        norm = getattr(block, "norm1", None)
        weight = getattr(norm, "weight", None)
        return getattr(weight, "dtype", fallback)

    @staticmethod
    def _module_parameter_dtype(module, fallback: torch.dtype) -> torch.dtype:
        if module is None:
            return fallback
        for parameter in module.parameters():
            return parameter.dtype
        return fallback

    def _relation_metadata(
        self,
        *,
        batch_size: int,
        views: int,
        frame_ids: torch.Tensor,
        tokens_per_group: int,
        slot_valid_mask: torch.Tensor | None,
        transaction_id: int,
        source: CacheSource,
        version_id: int,
        device: torch.device,
    ):
        valid = None
        if slot_valid_mask is not None:
            slots = slot_valid_mask.shape[-1]
            tokens_per_slot = tokens_per_group // slots
            valid = (
                slot_valid_mask.to(device=device, dtype=torch.bool)[:, None]
                .expand(-1, views, -1, -1)
                .reshape(batch_size * views, -1)
                .repeat_interleave(tokens_per_slot, dim=1)
            )
        return build_token_metadata(
            batch_size=batch_size * views,
            frame_ids=frame_ids,
            tokens_per_frame=tokens_per_group,
            stream_id=STREAM_GEOMETRY,
            noise_id=NOISE_GEOMETRY,
            history_frames=self.generation_profile.history_frames,
            chunk_size=self.generation_profile.chunk_size,
            device=device,
            valid_ids=valid,
            committed=False,
            transaction_id=transaction_id,
            source_id=int(source),
            version_id=version_id,
        )

    def _run_relation_attention_incremental(
        self,
        grouped_tokens: torch.Tensor,
        *,
        patch_hw: tuple[int, int],
        layer_id: int,
        frame_ids: torch.Tensor,
        slot_valid_mask: torch.Tensor | None,
        transaction_id: int,
        source: CacheSource,
        version_id: int,
        cache: GeometryRolloutCache,
    ) -> torch.Tensor:
        from distillation.model.autoregressive_types import AutoregressiveGeometryRelationRequest

        batch_size, groups, slots, views, num_tokens, channels = grouped_tokens.shape
        view_major = grouped_tokens.permute(0, 3, 1, 2, 4, 5)
        values = view_major.reshape(
            batch_size * views,
            groups * slots * num_tokens,
            channels,
        )
        rope = self._full_token_rope(
            patch_hw,
            groups * slots,
            values.device,
        )
        metadata = self._relation_metadata(
            batch_size=batch_size,
            views=views,
            frame_ids=frame_ids,
            tokens_per_group=slots * num_tokens,
            slot_valid_mask=slot_valid_mask,
            transaction_id=transaction_id,
            source=source,
            version_id=version_id,
            device=values.device,
        )
        block = self.inter_frame_blocks[layer_id]
        if hasattr(block, "forward_incremental"):
            values = block(
                AutoregressiveGeometryRelationRequest(
                    values=values,
                    rope=rope,
                    metadata=metadata,
                    cache=cache,
                    transaction_id=transaction_id,
                    layer_id=layer_id,
                )
            )
        else:
            values = values.to(dtype=self._norm_input_dtype(block, values.dtype))
            qkv = block.attn.qkv(block.norm1(values)).reshape(
                values.shape[0],
                values.shape[1],
                3,
                block.attn.num_heads,
                values.shape[-1] // block.attn.num_heads,
            )
            query, key, value = qkv.unbind(dim=2)
            if block.attn.use_qk_norm:
                query = block.attn.q_norm(query)
                key = block.attn.k_norm(key)
            if rope is not None:
                query_heads, key_heads = query.transpose(1, 2), key.transpose(1, 2)
                query_heads, key_heads = block.attn.apply_rope(query_heads, key_heads, rope)
                query, key = query_heads.transpose(1, 2), key_heads.transpose(1, 2)
            from distillation.self_rollout import attention as attention_module

            cache.relation_cache.append_transaction(
                layer_id,
                transaction_id,
                KVSegment(key, value, metadata, stream_id=STREAM_GEOMETRY),
            )
            cache_key, cache_value, cache_meta = cache.relation_cache.materialize(
                layer_id,
                transaction_id=transaction_id,
                stream_id=STREAM_GEOMETRY,
            )
            query_valid, key_valid = attention_module.build_cache_selection(
                metadata, cache_meta
            )
            attended, visible = attention_module.indexed_attention(
                query,
                cache_key,
                cache_value,
                query_valid=query_valid,
                key_valid=key_valid,
            )
            projected = block.attn.proj(attended.flatten(2, 3))
            projected = block.attn.proj_drop(projected)
            projected = torch.where(
                visible[:, :, None],
                projected,
                torch.zeros_like(projected),
            )
            values = values + block.ls1(projected)
            values = values + block.ls2(block.mlp(block.norm2(values)))
        return (
            values.reshape(batch_size, views, groups, slots, num_tokens, channels)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(batch_size, groups * slots * views, num_tokens, channels)
        )

    def _register_rotary(
        self,
        model,
        *,
        batch_size: int,
        frame_ids: torch.Tensor,
        slots: int,
        views: int,
        register_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        frame_ids = torch.as_tensor(
            frame_ids,
            device=device,
            dtype=torch.float32,
        )
        frame = frame_ids[:, None, None, None].expand(
            -1,
            slots,
            views,
            register_tokens,
        )
        spatial = torch.full_like(frame, -1.0)
        grid = torch.stack(
            [frame, spatial, spatial, torch.zeros_like(frame)],
            dim=0,
        ).reshape(4, -1)
        return model.rope(grid[None].expand(batch_size, -1, -1))[:, :, None]

    def _joint_metadata(
        self,
        *,
        batch_size: int,
        frame_ids: torch.Tensor,
        slots: int,
        views: int,
        registers: int,
        slot_valid_mask: torch.Tensor | None,
        transaction_id: int,
        source: CacheSource,
        version_id: int,
        device: torch.device,
    ):
        valid = None
        if slot_valid_mask is not None:
            valid = (
                slot_valid_mask[:, :, :, None, None]
                .to(device=device, dtype=torch.bool)
                .expand(-1, -1, -1, views, registers)
                .reshape(batch_size, -1)
            )
        return build_token_metadata(
            batch_size=batch_size,
            frame_ids=frame_ids,
            tokens_per_frame=slots * views * registers,
            stream_id=STREAM_GEOMETRY,
            noise_id=NOISE_GEOMETRY,
            history_frames=self.generation_profile.history_frames,
            chunk_size=self.generation_profile.chunk_size,
            device=device,
            valid_ids=valid,
            committed=False,
            transaction_id=transaction_id,
            source_id=int(source),
            version_id=version_id,
        )

    def _run_joint_registers_incremental(
        self,
        model,
        registers: torch.Tensor,
        *,
        layer_id: int,
        frame_ids: torch.Tensor,
        groups: int,
        slots: int,
        views: int,
        slot_valid_mask: torch.Tensor | None,
        transaction_id: int,
        source: CacheSource,
        version_id: int,
        state,
    ) -> torch.Tensor:
        from distillation.model.autoregressive_types import AutoregressiveGeometryJointRequest

        batch_size, _images, register_tokens, _channels = registers.shape
        geometry_stream = model.mot_blocks[layer_id].geometry
        registers = registers.to(
            dtype=self._norm_input_dtype(geometry_stream, registers.dtype)
        )
        rotary = self._register_rotary(
            model,
            batch_size=batch_size,
            frame_ids=frame_ids,
            slots=slots,
            views=views,
            register_tokens=register_tokens,
            device=registers.device,
        )
        metadata = self._joint_metadata(
            batch_size=batch_size,
            frame_ids=frame_ids,
            slots=slots,
            views=views,
            registers=register_tokens,
            slot_valid_mask=slot_valid_mask,
            transaction_id=transaction_id,
            source=source,
            version_id=version_id,
            device=registers.device,
        )
        mot_block = model.mot_blocks[layer_id]
        if hasattr(mot_block, "forward_geometry_incremental"):
            return mot_block(
                AutoregressiveGeometryJointRequest(
                    registers=registers,
                    rotary=rotary,
                    metadata=metadata,
                    cache=state,
                    transaction_id=transaction_id,
                    layer_id=layer_id,
                    groups=groups,
                    slots=slots,
                    views=views,
                    slot_valid_mask=slot_valid_mask,
                )
            )
        with torch.autocast(
            device_type=registers.device.type,
            dtype=torch.bfloat16,
            enabled=registers.device.type == "cuda",
        ):
            query, key, value = geometry_stream.qkv_project(
                registers.flatten(1, 2),
                rotary,
            )
        state.mot_cache.append_transaction(
            layer_id,
            transaction_id,
            KVSegment(key, value, metadata, stream_id=STREAM_GEOMETRY),
        )
        cache_key, cache_value, cache_meta = state.mot_cache.materialize(
            layer_id,
            transaction_id=transaction_id,
            stream_id=STREAM_GEOMETRY,
        )
        from distillation.self_rollout import attention as attention_module

        query_valid, key_valid = attention_module.build_cache_selection(
            metadata, cache_meta
        )
        attended, visible = attention_module.indexed_attention(
            query,
            cache_key,
            cache_value,
            query_valid=query_valid,
            key_valid=key_valid,
        )
        with torch.autocast(
            device_type=registers.device.type,
            dtype=torch.bfloat16,
            enabled=registers.device.type == "cuda",
        ):
            delta = geometry_stream.attn_delta(attended).reshape_as(registers)
        visible = visible.reshape(
            batch_size,
            groups * slots * views,
            register_tokens,
        )
        delta = torch.where(visible[:, :, :, None], delta, torch.zeros_like(delta))
        residual = registers + delta
        with torch.autocast(
            device_type=residual.device.type,
            dtype=torch.bfloat16,
            enabled=residual.device.type == "cuda",
        ):
            updated = residual + geometry_stream.ffn_delta(residual)
        if slot_valid_mask is not None:
            valid = (
                slot_valid_mask[:, :, :, None]
                .to(device=updated.device, dtype=torch.bool)
                .expand(-1, -1, -1, views)
                .reshape(batch_size, groups * slots * views)
            )
            updated = torch.where(valid[:, :, None, None], updated, registers)
        return updated

    def _encode_groups_and_commit(
        self,
        model,
        rgb: torch.Tensor,
        *,
        frame_ids: torch.Tensor,
        slot_valid_mask: torch.Tensor | None,
        state,
        source: CacheSource,
        version_id: int,
        return_points: bool = False,
    ) -> list[EncodedGeometryFrame]:
        if rgb.ndim != 7:
            raise ValueError(
                "geometry transaction expects [B,G,S,V,3,H,W], "
                f"got {tuple(rgb.shape)}"
            )
        batch_size, groups, slots, views, channels, height, width = rgb.shape
        if channels != 3:
            raise ValueError("geometry RGB must have three channels")
        frame_ids = torch.as_tensor(
            frame_ids,
            dtype=torch.long,
            device=rgb.device,
        )
        if tuple(frame_ids.shape) != (groups,):
            raise ValueError(
                f"frame_ids must be [{groups}], got {tuple(frame_ids.shape)}"
            )
        if slot_valid_mask is not None:
            expected = (batch_size, groups, slots)
            if tuple(slot_valid_mask.shape) != expected:
                raise ValueError(
                    f"slot_valid_mask must be {expected}, got {tuple(slot_valid_mask.shape)}"
                )
            if bool(slot_valid_mask.all().item()):
                slot_valid_mask = None
        geometry_cache = state.geometry_cache
        if not isinstance(geometry_cache, GeometryRolloutCache):
            raise TypeError("RolloutState.geometry_cache must be GeometryRolloutCache")

        transaction_id = state.new_transaction_id()
        state_snapshot = state.mot_cache.snapshot()
        geometry_snapshot = geometry_cache.snapshot()
        patch_embed = getattr(self, "patch_embed", None)
        patch_projection = getattr(
            getattr(patch_embed, "patch_embed", None),
            "proj",
            None,
        )
        patch_embed_dtype = getattr(
            getattr(patch_projection, "weight", None),
            "dtype",
            self._module_parameter_dtype(patch_embed, rgb.dtype),
        )
        rgb_for_model = rgb.to(dtype=patch_embed_dtype)
        with torch.autocast(
            device_type=rgb_for_model.device.type,
            dtype=torch.bfloat16,
            enabled=rgb_for_model.device.type == "cuda",
        ):
            geometry_state = self.encode_grouped(
                rgb_for_model,
                slot_valid_mask=slot_valid_mask,
            )
        tokens = geometry_state.tokens
        cached_outputs: list[torch.Tensor | None] = [None] * self.depth
        layer_registers: dict[int, torch.Tensor] = {}
        try:
            for layer_id in range(self.depth):
                frame_blocks = getattr(self, "frame_blocks", None)
                frame_block = frame_blocks[layer_id] if frame_blocks is not None else None
                frame_tokens_input = tokens.to(
                    dtype=self._norm_input_dtype(frame_block, tokens.dtype)
                )
                with torch.autocast(
                    device_type=frame_tokens_input.device.type,
                    dtype=torch.bfloat16,
                    enabled=frame_tokens_input.device.type == "cuda",
                ):
                    frame_tokens = self.run_frame_block(
                        frame_tokens_input,
                        geometry_state.patch_hw,
                        layer_id,
                    )
                if layer_id in self.register_attention_indices:
                    registers = frame_tokens[:, :, : self.patch_start_idx]
                    layer_registers[layer_id] = registers.contiguous()
                    updated = self._run_joint_registers_incremental(
                        model,
                        registers,
                        layer_id=layer_id,
                        frame_ids=frame_ids,
                        groups=groups,
                        slots=slots,
                        views=views,
                        slot_valid_mask=slot_valid_mask,
                        transaction_id=transaction_id,
                        source=source,
                        version_id=version_id,
                        state=state,
                    )
                    tokens = torch.cat(
                        [updated, frame_tokens[:, :, self.patch_start_idx :]],
                        dim=2,
                    )
                else:
                    with torch.autocast(
                        device_type=frame_tokens.device.type,
                        dtype=torch.bfloat16,
                        enabled=frame_tokens.device.type == "cuda",
                    ):
                        grouped = self._run_cross_view_block(
                            frame_tokens,
                            geometry_state.patch_hw,
                            layer_id,
                            groups=groups,
                            group_size=slots,
                            views=views,
                            slot_valid_mask=slot_valid_mask,
                        )
                    tokens = self._run_relation_attention_incremental(
                        grouped,
                        patch_hw=geometry_state.patch_hw,
                        layer_id=layer_id,
                        frame_ids=frame_ids,
                        slot_valid_mask=slot_valid_mask,
                        transaction_id=transaction_id,
                        source=source,
                        version_id=version_id,
                        cache=geometry_cache,
                    )
                    if layer_id in self.cached_layer_set:
                        cached_outputs[layer_id] = torch.cat(
                            [frame_tokens, tokens],
                            dim=-1,
                        )

            state.mot_cache.commit_transaction(
                transaction_id,
                source_id=int(source),
            )
            geometry_cache.relation_cache.commit_transaction(
                transaction_id,
                source_id=int(source),
            )

            source_images = rgb.reshape(
                batch_size,
                groups * slots * views,
                channels,
                height,
                width,
            )
            depth = depth_conf = points = points_conf = None
            if return_points:
                depth, depth_conf = self.dense_forward(cached_outputs, source_images)
                points, points_conf = self.point_forward(cached_outputs, source_images)
            images_per_group = slots * views
            encoded_frames = []
            for group_index, frame_id in enumerate(frame_ids.tolist()):
                start = group_index * images_per_group
                end = start + images_per_group
                frame_cached_outputs = [
                    value[:, start:end].contiguous() if value is not None else None
                    for value in cached_outputs
                ]
                frame_registers = {
                    layer_id: value[:, start:end].contiguous()
                    for layer_id, value in layer_registers.items()
                }

                def frame_slice(value: torch.Tensor | None):
                    if value is None:
                        return None
                    return value[:, start:end].contiguous()

                encoded = EncodedGeometryFrame(
                    frame_id=int(frame_id),
                    rgb=rgb[:, group_index : group_index + 1],
                    final_tokens=tokens[:, start:end].contiguous(),
                    patch_hw=geometry_state.patch_hw,
                    image_hw=geometry_state.image_hw,
                    patch_token_start=geometry_state.patch_token_start,
                    cached_outputs=frame_cached_outputs,
                    layer_registers=frame_registers,
                    depth=frame_slice(depth),
                    depth_conf=frame_slice(depth_conf),
                    points=frame_slice(points),
                    points_conf=frame_slice(points_conf),
                )
                geometry_cache.frames[int(frame_id)] = encoded
                encoded_frames.append(encoded)
            return encoded_frames
        except Exception:
            state.mot_cache.restore(state_snapshot)
            geometry_cache.restore(geometry_snapshot)
            raise

    def encode_history_and_commit(
        self,
        model,
        rgb: torch.Tensor,
        *,
        frame_ids: list[int] | tuple[int, ...] | torch.Tensor,
        slot_valid_mask: torch.Tensor | None,
        state,
        source: CacheSource,
        version_id: int,
    ) -> list[EncodedGeometryFrame]:
        if source is not CacheSource.HISTORY:
            raise ValueError("encode_history_and_commit requires HISTORY source")
        return self._encode_groups_and_commit(
            model,
            rgb,
            frame_ids=torch.as_tensor(frame_ids, device=rgb.device),
            slot_valid_mask=slot_valid_mask,
            state=state,
            source=source,
            version_id=version_id,
        )

    def encode_and_commit(
        self,
        model,
        rgb: torch.Tensor,
        *,
        frame_id: int,
        slot_valid_mask: torch.Tensor | None,
        state,
        source: CacheSource,
        version_id: int,
        return_points: bool = False,
    ) -> EncodedGeometryFrame:
        if rgb.ndim != 7 or rgb.shape[1] != 1:
            raise ValueError(
                "single geometry transaction expects [B,1,S,V,3,H,W], "
                f"got {tuple(rgb.shape)}"
            )
        return self._encode_groups_and_commit(
            model,
            rgb,
            frame_ids=torch.tensor([frame_id], device=rgb.device),
            slot_valid_mask=slot_valid_mask,
            state=state,
            source=source,
            version_id=version_id,
            return_points=return_points,
        )[0]

    def assert_geometry_commit(
        self,
        state,
        *,
        frame_id: int,
        source: CacheSource,
        version_id: int,
    ) -> None:
        geometry_cache = state.geometry_cache
        if not isinstance(geometry_cache, GeometryRolloutCache):
            raise TypeError("RolloutState.geometry_cache must be GeometryRolloutCache")
        if int(frame_id) not in geometry_cache.frames:
            raise RuntimeError(f"geometry frame {int(frame_id)} was not committed")
        joint_layers = tuple(sorted(self.register_attention_indices))
        relation_layers = tuple(
            layer_id
            for layer_id in range(self.depth)
            if layer_id not in self.register_attention_indices
        )
        if joint_layers:
            state.mot_cache.assert_committed_frame(
                joint_layers,
                frame_id=frame_id,
                stream_id=STREAM_GEOMETRY,
                noise_ids=(NOISE_GEOMETRY,),
                source_id=int(source),
                version_id=version_id,
            )
        if relation_layers:
            geometry_cache.relation_cache.assert_committed_frame(
                relation_layers,
                frame_id=frame_id,
                stream_id=STREAM_GEOMETRY,
                noise_ids=(NOISE_GEOMETRY,),
                source_id=int(source),
                version_id=version_id,
            )

    def _run_same_view_inter_frame_block(
        self,
        grouped_tokens,
        patch_hw,
        layer_id,
        *,
        groups_per_chunk,
        slot_valid_mask,
    ):
        """Run the dense segmented relation path as a native virtual method."""

        bsz, groups, group_size, views, num_tokens, channels = grouped_tokens.shape
        view_major = grouped_tokens.permute(0, 3, 1, 2, 4, 5)
        values = view_major.reshape(
            bsz * views,
            groups * group_size * num_tokens,
            channels,
        )
        image_valid_mask = None
        if slot_valid_mask is not None:
            image_valid_mask = (
                slot_valid_mask.to(device=values.device, dtype=torch.bool)[:, None]
                .expand(-1, views, -1, -1)
                .reshape(bsz * views, groups * group_size)
            )

        rope = self._full_token_rope(patch_hw, groups * group_size, values.device)
        if self.masked_attn_backend == "fa4":
            metadata = self._build_segmented_inter_frame_metadata(
                batch_size=values.shape[0],
                groups=groups,
                group_size=group_size,
                tokens_per_image=num_tokens,
                history_frames=self.generation_profile.history_frames,
                chunk_size=int(groups_per_chunk),
                window_size=self.generation_profile.window_size,
                device=values.device,
                image_valid_mask=image_valid_mask,
            )
            values = self.inter_frame_blocks[layer_id](
                values,
                rope,
                mot_mask_metadata=metadata,
                attention_backend="fa4",
            )
        else:
            mask = self._build_segmented_inter_frame_mask(
                groups=groups,
                group_size=group_size,
                tokens_per_image=num_tokens,
                history_frames=self.generation_profile.history_frames,
                chunk_size=int(groups_per_chunk),
                device=values.device,
                image_valid_mask=image_valid_mask,
            )
            values = self.inter_frame_blocks[layer_id](
                values,
                rope,
                attn_bias=mask,
                attention_backend="dense",
            )
        return (
            values.reshape(bsz, views, groups, group_size, num_tokens, channels)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(bsz, groups * group_size * views, num_tokens, channels)
        )

    @staticmethod
    def _build_segmented_inter_frame_mask(
        *,
        groups: int,
        group_size: int,
        tokens_per_image: int,
        history_frames: int,
        chunk_size: int,
        device: torch.device,
        image_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        frame_ids = torch.arange(groups, device=device).repeat_interleave(group_size)
        image_orders = segmented_orders(
            frame_ids,
            history_frames=history_frames,
            chunk_size=chunk_size,
        )
        token_orders = image_orders.repeat_interleave(tokens_per_image)
        mask = token_orders[None, :] <= token_orders[:, None]
        if image_valid_mask is None:
            return mask

        image_valid = image_valid_mask.to(device=device, dtype=torch.bool)
        token_valid = image_valid.repeat_interleave(tokens_per_image, dim=1)
        mask = (
            mask[None, None]
            & token_valid[:, None, :, None]
            & token_valid[:, None, None, :]
        )
        token_count = token_orders.numel()
        eye = torch.eye(token_count, device=device, dtype=torch.bool)[None, None]
        invalid_queries = ~token_valid[:, None, :, None]
        return torch.where(invalid_queries, eye, mask)

    @classmethod
    def _build_segmented_inter_frame_metadata(
        cls,
        *,
        batch_size: int,
        groups: int,
        group_size: int,
        tokens_per_image: int,
        history_frames: int,
        chunk_size: int,
        window_size: int,
        device: torch.device,
        image_valid_mask: torch.Tensor | None,
    ):
        from wan_va.modules.mot_attention import build_geometry_metadata
        from distillation.model.autoregressive_mot import (
            AutoregressiveThreeDVAMOTTransformer3DModel,
        )

        token_valid_ids = None
        if image_valid_mask is not None:
            token_valid_ids = image_valid_mask.to(
                device=device,
                dtype=torch.bool,
            ).repeat_interleave(tokens_per_image, dim=1)
        metadata = build_geometry_metadata(
            batch_size=batch_size,
            geometry_tokens_per_frame=group_size * tokens_per_image,
            num_frames=groups,
            chunk_size=chunk_size,
            window_size=window_size,
            device=device,
            token_valid_ids=token_valid_ids,
        )
        return AutoregressiveThreeDVAMOTTransformer3DModel._apply_segmented_order(
            metadata,
            history_frames=history_frames,
            chunk_size=chunk_size,
        )
