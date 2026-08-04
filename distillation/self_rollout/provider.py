"""Ground-truth providers used by self rollout replacement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True, slots=True)
class GroundTruthStep:
    frame_id: int
    video_latent: torch.Tensor | None = None
    geometry_rgb: torch.Tensor | None = None
    geometry_state: Any | None = None
    action: torch.Tensor | None = None
    video_valid: torch.Tensor | None = None
    action_valid: torch.Tensor | None = None
    caused_by_action_frame_id: int | None = None

    def __post_init__(self) -> None:
        if int(self.frame_id) < 0:
            raise ValueError("frame_id must be non-negative")
        if (
            self.video_latent is None
            and self.geometry_rgb is None
            and self.geometry_state is None
            and self.action is None
        ):
            raise ValueError("GroundTruthStep must provide at least one component")


class GroundTruthProvider(Protocol):
    def maybe_get(
        self,
        *,
        frame_id: int,
        predicted_action: torch.Tensor,
        state: Any,
    ) -> GroundTruthStep | None: ...


class OfflineGroundTruthProvider:
    """Return frame-aligned GT slices from the original training batch."""

    def __init__(self, batch: dict[str, Any], *, replace: frozenset[str] | None = None):
        self.batch = batch
        self.replace = replace or frozenset({"video", "geometry", "action"})

    def maybe_get(
        self,
        *,
        frame_id: int,
        predicted_action: torch.Tensor,
        state: Any,
    ) -> GroundTruthStep | None:
        del predicted_action, state
        frame_id = int(frame_id)
        frames = int(self.batch["latents"].shape[2])
        if frame_id >= frames:
            return None
        return GroundTruthStep(
            frame_id=frame_id,
            video_latent=(
                self.batch["latents"][:, :, frame_id : frame_id + 1].clone()
                if "video" in self.replace
                else None
            ),
            geometry_rgb=(
                self.batch["geometry_rgb"][:, frame_id : frame_id + 1].clone()
                if "geometry" in self.replace and "geometry_rgb" in self.batch
                else None
            ),
            action=(
                self.batch["actions"][:, :, frame_id : frame_id + 1].clone()
                if "action" in self.replace
                else None
            ),
            video_valid=(
                self.batch["video_latent_valid_mask"][:, frame_id : frame_id + 1].clone()
                if "video_latent_valid_mask" in self.batch
                else None
            ),
            action_valid=(
                self.batch["action_valid_mask"][:, :, frame_id : frame_id + 1].clone()
                if "action_valid_mask" in self.batch
                else None
            ),
        )


def resolve_ground_truth_provider(
    mode: str,
    batch: dict[str, Any],
    explicit_provider: GroundTruthProvider | None = None,
) -> GroundTruthProvider | None:
    """Resolve the configured GT source without importing trainer dependencies."""

    mode = str(mode)
    if mode == "none":
        if explicit_provider is not None:
            raise ValueError(
                "an explicit ground_truth_provider requires rollout_gt_mode='provider'"
            )
        return None
    if mode == "offline":
        if explicit_provider is not None:
            raise ValueError(
                "rollout_gt_mode='offline' constructs its provider from the batch"
            )
        return OfflineGroundTruthProvider(batch)
    if mode == "provider":
        if explicit_provider is None:
            raise ValueError(
                "rollout_gt_mode='provider' requires an explicit ground_truth_provider"
            )
        return explicit_provider
    raise ValueError(
        "rollout_gt_mode must be 'none', 'offline', or 'provider'; "
        f"got {mode!r}"
    )
