"""Distillation-only attention order profiles."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Mirrors wan_va.modules.mot_attention without importing optional model modules.
PROFILE_NAME = "segmented_history_va_v1"
PROFILE_VERSION = 2


def generation_profile_contract(generation_shape: Any) -> dict[str, Any]:
    return {
        "profile_name": str(generation_shape.get("profile_name", PROFILE_NAME)),
        "profile_version": PROFILE_VERSION,
        "order_mode": str(generation_shape.get("order_mode", "chunk")),
        "history_frames": int(generation_shape.get("history_frames", 4)),
        "chunk_size": int(generation_shape["chunk_size"]),
        "window_size": int(generation_shape["window_size"]),
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
