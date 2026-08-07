"""Small, serialization-free contracts shared by AR model and rollout code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass(frozen=True, slots=True)
class AutoregressiveProfile:
    profile_name: str
    profile_version: int
    order_mode: str
    history_frames: int
    chunk_size: int
    window_size: int
    geometry_relation: str
    x_to_g_relation: str

    @classmethod
    def from_generation_shape(cls, shape: Any) -> "AutoregressiveProfile":
        if isinstance(shape, cls):
            shape.validate()
            return shape
        profile_name = str(shape.get("profile_name", "segmented_history_strict_geometry_v1"))
        order_mode = str(shape.get("order_mode", "segmented"))
        profile = cls(
            profile_name=profile_name,
            profile_version=2,
            order_mode=order_mode,
            history_frames=int(shape.get("history_frames", 4)),
            chunk_size=int(shape["chunk_size"]),
            window_size=int(shape["window_size"]),
            geometry_relation="segmented_order_causal",
            x_to_g_relation="strict_order",
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.profile_name != "segmented_history_strict_geometry_v1":
            raise ValueError(f"unsupported autoregressive profile {self.profile_name!r}")
        if self.order_mode != "segmented":
            raise ValueError(f"unsupported autoregressive order_mode {self.order_mode!r}")
        if self.profile_version != 2:
            raise ValueError(f"unsupported autoregressive profile version {self.profile_version}")
        if self.history_frames <= 0 or self.chunk_size <= 0 or self.window_size < 0:
            raise ValueError("history_frames/chunk_size must be positive and window_size non-negative")
        if self.geometry_relation != "segmented_order_causal":
            raise ValueError(f"unsupported geometry relation {self.geometry_relation!r}")
        if self.x_to_g_relation != "strict_order":
            raise ValueError(f"unsupported x_to_g relation {self.x_to_g_relation!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "order_mode": self.order_mode,
            "history_frames": self.history_frames,
            "chunk_size": self.chunk_size,
            "window_size": self.window_size,
            "geometry_relation": self.geometry_relation,
            "x_to_g_relation": self.x_to_g_relation,
        }


@dataclass(slots=True)
class AutoregressiveModelRequest:
    """One cache transaction requested by the self-rollout engine.

    ``payload`` is intentionally an engine-owned dictionary for this first
    migration step.  The model consumes only the operation-specific keys and
    returns an ``AutoregressiveModelOutput``; it never serializes the request.
    """

    operation: Literal[
        "predict_video",
        "commit_video",
        "predict_action",
        "commit_action",
        "encode_geometry",
        "encode_geometry_history",
    ]
    payload: dict[str, Any]


@dataclass(slots=True)
class AutoregressiveModelOutput:
    prediction: torch.Tensor | None = None
    geometry_frame: Any | None = None
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AutoregressiveStreamInput:
    hidden: torch.Tensor
    conditioning: torch.Tensor
    rotary: torch.Tensor | None
    metadata: Any
    block_kind: Literal["video", "action"]


@dataclass(slots=True)
class AutoregressiveMOTLayerRequest:
    """Internal request consumed by one AR MoT block forward."""

    stream: Any
    hidden_state: torch.Tensor
    text: torch.Tensor
    state: Any
    transaction_id: int
    layer_id: int
    stream_id: int


@dataclass(slots=True)
class AutoregressiveGeometryRelationRequest:
    values: torch.Tensor
    rope: Any
    metadata: Any
    cache: Any
    transaction_id: int
    layer_id: int


@dataclass(slots=True)
class AutoregressiveGeometryJointRequest:
    registers: torch.Tensor
    rotary: Any
    metadata: Any
    cache: Any
    transaction_id: int
    layer_id: int
    groups: int
    slots: int
    views: int
    slot_valid_mask: torch.Tensor | None
