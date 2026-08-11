"""Small, serialization-free contracts shared by AR model and rollout code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass(slots=True)
class AutoregressiveModelRequest:
    """One incremental rollout operation requested by a pipeline.

    ``payload`` is intentionally a pipeline-owned dictionary. The model consumes
    only the operation-specific keys and returns an ``AutoregressiveModelOutput``;
    it never serializes the request.
    """

    operation: Literal[
        "predict_video",
        "commit_video",
        "predict_action",
        "commit_action",
    ]
    payload: dict[str, Any]


@dataclass(slots=True)
class AutoregressiveModelOutput:
    prediction: torch.Tensor | None = None
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AutoregressiveStreamInput:
    hidden: torch.Tensor
    conditioning: torch.Tensor
    rotary: torch.Tensor | None
    block_kind: Literal["video", "action"]


@dataclass(slots=True)
class AutoregressiveMOTLayerRequest:
    """Internal request consumed by one AR MoT block forward."""

    stream: Any
    hidden_state: torch.Tensor
    text: torch.Tensor
    cache: Any
    transaction_id: int
    layer_id: int
