"""Distillation-only attention order profiles."""
from __future__ import annotations

import json
from pathlib import Path
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
