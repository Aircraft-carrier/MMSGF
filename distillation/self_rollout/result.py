"""Result types for self rollout."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class RolloutResult:
    pred_latents: torch.Tensor
    target_latents: torch.Tensor
    pred_actions: torch.Tensor
    target_actions: torch.Tensor
    pred_geometry_rgb: torch.Tensor
    target_geometry_rgb: torch.Tensor
    action_valid_mask: torch.Tensor
    chunk_pairs: int
    chunk_frames: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
