"""Distillation-only attention order profiles."""
from __future__ import annotations

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
