"""Distillation-only attention order profiles."""
from __future__ import annotations

import json
from pathlib import Path
from types import MethodType
from typing import Any, TYPE_CHECKING

import torch

from distillation.self_rollout.attention import segmented_orders

if TYPE_CHECKING:
    from wan_va.modules.mot_attention import MOTMaskMetadata
else:
    MOTMaskMetadata = Any


# Mirrors wan_va.modules.mot_attention without importing optional model modules.
STREAM_ACTION = 1
PROFILE_NAME = "segmented_history_strict_geometry_v1"
# Version 2 changes G->G from strict frame history to segmented order-causal
# visibility.  Old checkpoints must not silently reuse the new topology.
PROFILE_VERSION = 2


def generation_profile_contract(generation_shape: Any) -> dict[str, Any]:
    return {
        "profile_name": str(generation_shape.get("profile_name", PROFILE_NAME)),
        "profile_version": PROFILE_VERSION,
        "order_mode": str(generation_shape.get("order_mode", "chunk")),
        "history_frames": int(generation_shape.get("history_frames", 4)),
        "chunk_size": int(generation_shape["chunk_size"]),
        "window_size": int(generation_shape["window_size"]),
        "geometry_relation": "segmented_order_causal",
        "x_to_g_relation": "strict_order",
    }


def validate_checkpoint_generation_profile(
    checkpoint_path: str | Path,
    generation_shape: Any,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = metadata.get("generation_profile")
    expected = generation_profile_contract(generation_shape)
    if actual != expected:
        raise ValueError(
            "Checkpoint generation profile does not match distillation policy: "
            f"checkpoint={actual}, expected={expected}"
        )


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


def _apply_segmented_order(
    metadata: MOTMaskMetadata,
    *,
    history_frames: int,
    chunk_size: int,
) -> MOTMaskMetadata:
    if metadata.frame_ids is None:
        raise ValueError("segmented distillation order requires frame_ids")
    video_order, action_order = _segmented_frame_orders(
        num_frames=int(metadata.frame_ids.max().item()) + 1,
        history_frames=history_frames,
        chunk_size=chunk_size,
        device=metadata.device,
    )
    frame_order = video_order[metadata.frame_ids]
    metadata.order_ids = torch.where(
        metadata.stream_ids == STREAM_ACTION,
        action_order[metadata.frame_ids],
        frame_order,
    )
    # The native cache key only describes the original chunk-derived order.
    # Do not let a segmented profile reuse that mask.
    metadata.cache_key = None
    metadata.structure_cache_key = None
    return metadata


def _build_segmented_vggto_inter_frame_mask(
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


def _build_segmented_vggto_inter_frame_metadata(
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
) -> MOTMaskMetadata:
    from wan_va.modules.mot_attention import build_geometry_metadata

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
    return _apply_segmented_order(
        metadata,
        history_frames=history_frames,
        chunk_size=chunk_size,
    )


def _run_segmented_vggto_inter_frame_fa4(
    self,
    values: torch.Tensor,
    patch_hw,
    layer_id: int,
    *,
    groups: int,
    group_size: int,
    num_tokens: int,
    image_valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    from wan_va.modules.fa4_attention import fa4_attention_from_meta

    block = self.inter_frame_blocks[layer_id]
    attention = block.attn
    hidden = block.norm1(values)
    qkv = attention.qkv(hidden).reshape(
        values.shape[0],
        values.shape[1],
        3,
        attention.num_heads,
        values.shape[-1] // attention.num_heads,
    )
    query, key, value = qkv.unbind(dim=2)
    if attention.use_qk_norm:
        query = attention.q_norm(query)
        key = attention.k_norm(key)
    rope = self._full_token_rope(patch_hw, groups * group_size, values.device)
    query_heads, key_heads = query.transpose(1, 2), key.transpose(1, 2)
    query_heads, key_heads = attention.apply_rope(query_heads, key_heads, rope)
    query, key = query_heads.transpose(1, 2), key_heads.transpose(1, 2)

    metadata = _build_segmented_vggto_inter_frame_metadata(
        batch_size=values.shape[0],
        groups=groups,
        group_size=group_size,
        tokens_per_image=num_tokens,
        history_frames=int(self._distillation_history_frames),
        chunk_size=int(self._distillation_chunk_size),
        window_size=int(self._distillation_window_size),
        device=values.device,
        image_valid_mask=image_valid_mask,
    )
    attended = fa4_attention_from_meta(query, key, value, metadata)
    projected = attention.proj(attended.flatten(2, 3))
    projected = attention.proj_drop(projected)
    values = values + block.ls1(projected)
    return values + block.ls2(block.mlp(block.norm2(values)))


def _segmented_same_view_inter_frame(
    self,
    grouped_tokens,
    patch_hw,
    layer_id,
    *,
    groups_per_chunk,
    slot_valid_mask,
):
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
    if getattr(self, "masked_attn_backend", None) == "fa4":
        values = _run_segmented_vggto_inter_frame_fa4(
            self,
            values,
            patch_hw,
            layer_id,
            groups=groups,
            group_size=group_size,
            num_tokens=num_tokens,
            image_valid_mask=image_valid_mask,
        )
        return (
            values.reshape(bsz, views, groups, group_size, num_tokens, channels)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(bsz, groups * group_size * views, num_tokens, channels)
        )

    mask = _build_segmented_vggto_inter_frame_mask(
        groups=groups,
        group_size=group_size,
        tokens_per_image=num_tokens,
        history_frames=int(self._distillation_history_frames),
        chunk_size=int(groups_per_chunk),
        device=values.device,
        image_valid_mask=image_valid_mask,
    )
    values = self.inter_frame_blocks[layer_id](
        values,
        self._full_token_rope(patch_hw, groups * group_size, values.device),
        attn_bias=mask,
        attention_backend="dense",
    )
    return (
        values.reshape(bsz, views, groups, group_size, num_tokens, channels)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(bsz, groups * group_size * views, num_tokens, channels)
    )


def _install_segmented_vggto_inter_frame(model: Any, generation_shape: Any) -> None:
    vggto = getattr(model, "vggto", None)
    if vggto is None:
        return
    history_frames = int(generation_shape.get("history_frames", 4))
    chunk_size = int(generation_shape["chunk_size"])
    window_size = int(generation_shape["window_size"])
    if (
        getattr(vggto, "_distillation_vggto_order_profile", None) == "segmented"
        and getattr(vggto, "_distillation_history_frames", None) == history_frames
        and getattr(vggto, "_distillation_chunk_size", None) == chunk_size
        and getattr(vggto, "_distillation_window_size", None) == window_size
    ):
        return
    vggto._distillation_history_frames = history_frames
    vggto._distillation_chunk_size = chunk_size
    vggto._distillation_window_size = window_size
    vggto._run_same_view_inter_frame_block = MethodType(
        _segmented_same_view_inter_frame,
        vggto,
    )
    vggto._distillation_vggto_order_profile = "segmented"


def install_order_profile(model: Any, generation_shape: Any) -> None:
    """Install a distillation-only order wrapper on one MOT transformer."""
    model = getattr(model, "module", model)
    shape = generation_shape
    order_mode = str(shape.get("order_mode", "chunk"))
    if order_mode != "segmented":
        return
    if getattr(model, "_distillation_order_profile", None) == order_mode:
        _install_segmented_vggto_inter_frame(model, generation_shape)
        return

    original_prepare_metadata = model._prepare_metadata

    def prepare_metadata(input_dict, prepared, geometry):
        x_meta, mot_meta, diagnostics = original_prepare_metadata(
            input_dict,
            prepared,
            geometry,
        )
        chunk_size = int(input_dict.get("chunk_size", shape["chunk_size"]))
        history_frames = int(shape.get("history_frames", 4))
        return (
            _apply_segmented_order(
                x_meta,
                history_frames=history_frames,
                chunk_size=chunk_size,
            ),
            _apply_segmented_order(
                mot_meta,
                history_frames=history_frames,
                chunk_size=chunk_size,
            ),
            diagnostics,
        )

    model._prepare_metadata = prepare_metadata
    model._distillation_order_profile = order_mode
    _install_segmented_vggto_inter_frame(model, generation_shape)
