"""Incremental VGGTO geometry encoding with strict historical attention."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .attention import (
    NOISE_GEOMETRY,
    STREAM_GEOMETRY,
    build_cache_visibility,
    build_token_metadata,
    incremental_attention,
)
from .cache import KVCacheSnapshot, KVSegment, SelfRolloutKVCache
from .state import CacheSource, RolloutState


@dataclass(slots=True)
class EncodedGeometryFrame:
    frame_id: int
    rgb: torch.Tensor
    final_tokens: torch.Tensor
    patch_hw: tuple[int, int]
    image_hw: tuple[int, int]
    patch_token_start: int
    cached_outputs: list[torch.Tensor | None]
    layer_registers: dict[int, torch.Tensor]
    depth: torch.Tensor | None = None
    depth_conf: torch.Tensor | None = None
    points: torch.Tensor | None = None
    points_conf: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class GeometryCacheSnapshot:
    relation_cache: KVCacheSnapshot
    frames: dict[int, EncodedGeometryFrame]


class GeometryRolloutCache:
    def __init__(self) -> None:
        self.relation_cache = SelfRolloutKVCache()
        self.frames: dict[int, EncodedGeometryFrame] = {}

    def snapshot(self) -> GeometryCacheSnapshot:
        return GeometryCacheSnapshot(
            relation_cache=self.relation_cache.snapshot(),
            frames=dict(self.frames),
        )

    def restore(self, snapshot: GeometryCacheSnapshot | None) -> None:
        if snapshot is None:
            self.relation_cache = SelfRolloutKVCache()
            self.frames = {}
            return
        self.relation_cache.restore(snapshot.relation_cache)
        self.frames = dict(snapshot.frames)

    def truncate_from(self, frame_id: int) -> None:
        self.relation_cache.truncate_from(frame_id)
        self.frames = {
            index: frame for index, frame in self.frames.items() if index < int(frame_id)
        }


class GeometryIncrementalAdapter:
    """Encode one geometry group while reading only earlier committed G K/V."""

    def __init__(
        self,
        transformer,
        *,
        history_frames: int,
        chunk_size: int,
        window_size: int,
    ) -> None:
        self.model = getattr(transformer, "module", transformer)
        required_model = ("vggto", "mot_blocks", "rope")
        missing_model = [
            name for name in required_model if not hasattr(self.model, name)
        ]
        if missing_model:
            raise TypeError(
                "geometry self_rollout requires ThreeDVAMOT attributes: "
                + ", ".join(missing_model)
            )
        self.vggto = self.model.vggto
        self.history_frames = int(history_frames)
        self.chunk_size = int(chunk_size)
        self.window_size = int(window_size)
        if self.history_frames <= 0 or self.chunk_size <= 0 or self.window_size < 0:
            raise ValueError("invalid history/chunk/window configuration")
        required_vggto = (
            "depth",
            "inter_frame_blocks",
            "register_attention_indices",
            "cached_layer_set",
            "patch_start_idx",
            "encode_grouped",
            "run_frame_block",
            "_run_cross_view_block",
            "_full_token_rope",
        )
        missing_vggto = [
            name for name in required_vggto if not hasattr(self.vggto, name)
        ]
        if missing_vggto:
            raise TypeError(
                "geometry self_rollout requires VGGTO attributes: "
                + ", ".join(missing_vggto)
            )
        if len(self.model.mot_blocks) < int(self.vggto.depth):
            raise TypeError("geometry self_rollout requires one MoT block per VGGTO layer")
        if len(self.vggto.inter_frame_blocks) < int(self.vggto.depth):
            raise TypeError("geometry self_rollout requires one relation block per VGGTO layer")
        for layer_id in self.vggto.register_attention_indices:
            if self.model.mot_blocks[int(layer_id)].geometry is None:
                raise TypeError(
                    f"VGGTO joint layer {int(layer_id)} has no MoT geometry stream"
                )

    @staticmethod
    def _apply_vggto_qkv(block, values: torch.Tensor, rope):
        attention = block.attn
        batch_size, tokens, channels = values.shape
        qkv = attention.qkv(block.norm1(values)).reshape(
            batch_size,
            tokens,
            3,
            attention.num_heads,
            channels // attention.num_heads,
        )
        query, key, value = qkv.unbind(dim=2)
        if attention.use_qk_norm:
            query = attention.q_norm(query)
            key = attention.k_norm(key)
        if rope is not None:
            query_heads, key_heads = query.transpose(1, 2), key.transpose(1, 2)
            query_heads, key_heads = attention.apply_rope(
                query_heads,
                key_heads,
                rope,
            )
            query = query_heads.transpose(1, 2)
            key = key_heads.transpose(1, 2)
        return query, key, value

    def _relation_metadata(
        self,
        *,
        batch_size: int,
        views: int,
        frame_id: int,
        tokens_per_view: int,
        slot_valid_mask: torch.Tensor | None,
        transaction_id: int,
        source: CacheSource,
        version_id: int,
        device: torch.device,
    ):
        valid = None
        if slot_valid_mask is not None:
            slots = slot_valid_mask.shape[-1]
            tokens_per_slot = tokens_per_view // slots
            valid = (
                slot_valid_mask[:, 0, :]
                .to(device=device, dtype=torch.bool)[:, None]
                .expand(-1, views, -1)
                .reshape(batch_size * views, slots)
                .repeat_interleave(tokens_per_slot, dim=1)
            )
        return build_token_metadata(
            batch_size=batch_size * views,
            frame_ids=torch.full(
                (batch_size * views, 1),
                int(frame_id),
                dtype=torch.long,
                device=device,
            ),
            tokens_per_frame=tokens_per_view,
            stream_id=STREAM_GEOMETRY,
            noise_id=NOISE_GEOMETRY,
            history_frames=self.history_frames,
            chunk_size=self.chunk_size,
            device=device,
            valid_ids=valid,
            committed=False,
            transaction_id=transaction_id,
            source_id=int(source),
            version_id=version_id,
        )

    def _run_history_relation(
        self,
        grouped_tokens: torch.Tensor,
        *,
        patch_hw: tuple[int, int],
        layer_id: int,
        frame_id: int,
        slot_valid_mask: torch.Tensor | None,
        transaction_id: int,
        source: CacheSource,
        version_id: int,
        cache: GeometryRolloutCache,
    ) -> torch.Tensor:
        batch_size, _groups, slots, views, num_tokens, channels = grouped_tokens.shape
        view_major = grouped_tokens.permute(0, 3, 1, 2, 4, 5)
        values = view_major.reshape(batch_size * views, slots * num_tokens, channels)
        block = self.vggto.inter_frame_blocks[layer_id]
        rope = self.vggto._full_token_rope(
            patch_hw,
            slots,
            values.device,
        )
        query, current_key, current_value = self._apply_vggto_qkv(block, values, rope)
        metadata = self._relation_metadata(
            batch_size=batch_size,
            views=views,
            frame_id=frame_id,
            tokens_per_view=slots * num_tokens,
            slot_valid_mask=slot_valid_mask,
            transaction_id=transaction_id,
            source=source,
            version_id=version_id,
            device=values.device,
        )
        cache.relation_cache.append_transaction(
            layer_id,
            transaction_id,
            KVSegment(current_key, current_value, metadata),
        )
        key, value, key_meta = cache.relation_cache.materialize(
            layer_id,
            transaction_id=transaction_id,
        )
        mask = build_cache_visibility(metadata, key_meta, window_size=self.window_size)
        attended = incremental_attention(query, key, value, mask)
        projected = block.attn.proj(attended.flatten(2, 3))
        projected = block.attn.proj_drop(projected)
        visible = mask.any(dim=-1)
        projected = torch.where(visible[:, :, None], projected, torch.zeros_like(projected))
        values = values + block.ls1(projected)
        values = values + block.ls2(block.mlp(block.norm2(values)))
        return (
            values.reshape(batch_size, views, 1, slots, num_tokens, channels)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(batch_size, slots * views, num_tokens, channels)
        )

    def _register_rotary(
        self,
        *,
        batch_size: int,
        frame_id: int,
        slots: int,
        views: int,
        register_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        frame = torch.full(
            (slots, views, register_tokens),
            float(frame_id),
            device=device,
        )
        spatial = torch.full_like(frame, -1.0)
        grid = torch.stack(
            [frame, spatial, spatial, torch.zeros_like(frame)],
            dim=0,
        ).reshape(4, -1)
        return self.model.rope(grid[None].expand(batch_size, -1, -1))[:, :, None]

    def _joint_metadata(
        self,
        *,
        batch_size: int,
        frame_id: int,
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
                slot_valid_mask[:, 0, :, None, None]
                .to(device=device, dtype=torch.bool)
                .expand(-1, slots, views, registers)
                .reshape(batch_size, -1)
            )
        return build_token_metadata(
            batch_size=batch_size,
            frame_ids=torch.full(
                (batch_size, 1),
                int(frame_id),
                dtype=torch.long,
                device=device,
            ),
            tokens_per_frame=slots * views * registers,
            stream_id=STREAM_GEOMETRY,
            noise_id=NOISE_GEOMETRY,
            history_frames=self.history_frames,
            chunk_size=self.chunk_size,
            device=device,
            valid_ids=valid,
            committed=False,
            transaction_id=transaction_id,
            source_id=int(source),
            version_id=version_id,
        )

    def _run_joint_registers(
        self,
        registers: torch.Tensor,
        *,
        layer_id: int,
        frame_id: int,
        slots: int,
        views: int,
        slot_valid_mask: torch.Tensor | None,
        transaction_id: int,
        source: CacheSource,
        version_id: int,
        state: RolloutState,
    ) -> torch.Tensor:
        batch_size, _images, register_tokens, _channels = registers.shape
        geometry_stream = self.model.mot_blocks[layer_id].geometry
        rotary = self._register_rotary(
            batch_size=batch_size,
            frame_id=frame_id,
            slots=slots,
            views=views,
            register_tokens=register_tokens,
            device=registers.device,
        )
        query, current_key, current_value = geometry_stream.qkv_project(
            registers.flatten(1, 2),
            rotary,
        )
        metadata = self._joint_metadata(
            batch_size=batch_size,
            frame_id=frame_id,
            slots=slots,
            views=views,
            registers=register_tokens,
            slot_valid_mask=slot_valid_mask,
            transaction_id=transaction_id,
            source=source,
            version_id=version_id,
            device=registers.device,
        )
        state.mot_cache.append_transaction(
            layer_id,
            transaction_id,
            KVSegment(current_key, current_value, metadata),
        )
        key, value, key_meta = state.mot_cache.materialize(
            layer_id,
            transaction_id=transaction_id,
        )
        mask = build_cache_visibility(metadata, key_meta, window_size=self.window_size)
        attended = incremental_attention(query, key, value, mask)
        delta = geometry_stream.attn_delta(attended).reshape_as(registers)
        visible = mask.any(dim=-1).reshape(
            batch_size,
            slots * views,
            register_tokens,
        )
        delta = torch.where(visible[:, :, :, None], delta, torch.zeros_like(delta))
        residual = registers + delta
        updated = residual + geometry_stream.ffn_delta(residual)
        if slot_valid_mask is not None:
            valid = (
                slot_valid_mask[:, 0, :, None]
                .to(device=updated.device, dtype=torch.bool)
                .expand(-1, slots, views)
                .reshape(batch_size, slots * views)
            )
            updated = torch.where(
                valid[:, :, None, None],
                updated,
                registers,
            )
        return updated

    def encode_and_commit(
        self,
        rgb: torch.Tensor,
        *,
        frame_id: int,
        slot_valid_mask: torch.Tensor | None,
        state: RolloutState,
        source: CacheSource,
        version_id: int,
        return_points: bool = False,
    ) -> EncodedGeometryFrame:
        if rgb.ndim != 7 or rgb.shape[1] != 1:
            raise ValueError(
                "geometry transaction expects [B,1,S,V,3,H,W], "
                f"got {tuple(rgb.shape)}"
            )
        batch_size, _groups, slots, views, channels, height, width = rgb.shape
        if channels != 3:
            raise ValueError("geometry RGB must have three channels")
        if slot_valid_mask is not None:
            expected = (batch_size, 1, slots)
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
        geometry_state = self.vggto.encode_grouped(rgb, slot_valid_mask=slot_valid_mask)
        tokens = geometry_state.tokens
        cached_outputs: list[torch.Tensor | None] = [None] * self.vggto.depth
        layer_registers: dict[int, torch.Tensor] = {}
        try:
            for layer_id in range(self.vggto.depth):
                frame_tokens = self.vggto.run_frame_block(
                    tokens,
                    geometry_state.patch_hw,
                    layer_id,
                )
                if layer_id in self.vggto.register_attention_indices:
                    registers = frame_tokens[:, :, : self.vggto.patch_start_idx]
                    layer_registers[layer_id] = registers.contiguous()
                    updated = self._run_joint_registers(
                        registers,
                        layer_id=layer_id,
                        frame_id=frame_id,
                        slots=slots,
                        views=views,
                        slot_valid_mask=slot_valid_mask,
                        transaction_id=transaction_id,
                        source=source,
                        version_id=version_id,
                        state=state,
                    )
                    tokens = torch.cat(
                        [updated, frame_tokens[:, :, self.vggto.patch_start_idx :]],
                        dim=2,
                    )
                else:
                    grouped = self.vggto._run_cross_view_block(
                        frame_tokens,
                        geometry_state.patch_hw,
                        layer_id,
                        groups=1,
                        group_size=slots,
                        views=views,
                        slot_valid_mask=slot_valid_mask,
                    )
                    tokens = self._run_history_relation(
                        grouped,
                        patch_hw=geometry_state.patch_hw,
                        layer_id=layer_id,
                        frame_id=frame_id,
                        slot_valid_mask=slot_valid_mask,
                        transaction_id=transaction_id,
                        source=source,
                        version_id=version_id,
                        cache=geometry_cache,
                    )
                    if layer_id in self.vggto.cached_layer_set:
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
                slots * views,
                channels,
                height,
                width,
            )
            depth = depth_conf = points = points_conf = None
            if return_points:
                depth, depth_conf = self.vggto.dense_forward(cached_outputs, source_images)
                points, points_conf = self.vggto.point_forward(cached_outputs, source_images)
            encoded = EncodedGeometryFrame(
                frame_id=int(frame_id),
                rgb=rgb,
                final_tokens=tokens,
                patch_hw=geometry_state.patch_hw,
                image_hw=geometry_state.image_hw,
                patch_token_start=geometry_state.patch_token_start,
                cached_outputs=cached_outputs,
                layer_registers=layer_registers,
                depth=depth,
                depth_conf=depth_conf,
                points=points,
                points_conf=points_conf,
            )
            geometry_cache.frames[int(frame_id)] = encoded
            return encoded
        except Exception:
            state.mot_cache.restore(state_snapshot)
            geometry_cache.restore(geometry_snapshot)
            raise

    def assert_geometry_commit(
        self,
        state: RolloutState,
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
        joint_layers = tuple(sorted(self.vggto.register_attention_indices))
        relation_layers = tuple(
            layer_id
            for layer_id in range(self.vggto.depth)
            if layer_id not in self.vggto.register_attention_indices
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
