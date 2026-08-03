"""Distillation-only attention order profiles."""
from __future__ import annotations

from typing import Any

import torch

from wan_va.modules.mot_attention import STREAM_ACTION, MOTMaskMetadata


def _segmented_frame_orders(
    *,
    num_frames: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    split = num_frames // 2
    first_half = torch.arange(split, device=device) // chunk_size * 2
    second_half_start = ((split + chunk_size - 1) // chunk_size) * 2
    second_half = (
        torch.arange(num_frames - split, device=device) * 2
        + second_half_start
    )
    video_order = torch.cat([first_half, second_half])
    return video_order, video_order + 1


def _apply_segmented_order(
    metadata: MOTMaskMetadata,
    *,
    chunk_size: int,
) -> MOTMaskMetadata:
    if metadata.frame_ids is None:
        raise ValueError("segmented distillation order requires frame_ids")
    video_order, action_order = _segmented_frame_orders(
        num_frames=int(metadata.frame_ids.max().item()) + 1,
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


def install_order_profile(model: Any, generation_shape: Any) -> None:
    """Install a distillation-only order wrapper on one MOT transformer."""
    model = getattr(model, "module", model)
    shape = generation_shape
    order_mode = str(shape.get("order_mode", "chunk"))
    if order_mode != "segmented":
        return
    if getattr(model, "_distillation_order_profile", None) == order_mode:
        return

    original_prepare_metadata = model._prepare_metadata

    def prepare_metadata(input_dict, prepared, geometry):
        x_meta, mot_meta, diagnostics = original_prepare_metadata(
            input_dict,
            prepared,
            geometry,
        )
        chunk_size = int(input_dict.get("chunk_size", shape["chunk_size"]))
        return (
            _apply_segmented_order(x_meta, chunk_size=chunk_size),
            _apply_segmented_order(mot_meta, chunk_size=chunk_size),
            diagnostics,
        )

    model._prepare_metadata = prepare_metadata
    model._distillation_order_profile = order_mode
