"""Scheduler-aware adapters for Wan Video+Action flow-matching models."""
from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from distillation.diffusion_utils import flow_to_x0
from distillation.schema import VADiffusionOutput, VAPrediction, VATimesteps


class WanDiffusionWrapper:
    """Borrow a configured model and expose velocity plus clean predictions.

    The wrapper deliberately is not an ``nn.Module``.  Student and score models
    are already owned and possibly FSDP-sharded by their trainer/model owner;
    registering them again would change parameter and checkpoint ownership.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        video_scheduler: Any,
        action_scheduler: Any,
    ) -> None:
        self.model = model
        self.video_scheduler = video_scheduler
        self.action_scheduler = action_scheduler

    def velocity_to_x0(
        self,
        velocity: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        modality: Literal["video", "action"],
    ) -> torch.Tensor:
        if tuple(velocity.shape) != tuple(noisy.shape):
            raise ValueError(
                f"{modality} velocity shape {tuple(velocity.shape)} does not match "
                f"noisy sample {tuple(noisy.shape)}"
            )
        if velocity.ndim < 3:
            raise ValueError(f"{modality} sample must contain B/C/F axes")
        expected = (velocity.shape[0], velocity.shape[2])
        if tuple(timesteps.shape) != expected:
            raise ValueError(
                f"{modality} timesteps must be [B,F]={expected}, got "
                f"{tuple(timesteps.shape)}"
            )
        scheduler = (
            self.video_scheduler if modality == "video" else self.action_scheduler
        )
        return flow_to_x0(velocity, noisy, timesteps, scheduler)

    def predict_velocity(self, input_dict: dict[str, Any]) -> VAPrediction:
        output = self.model(input_dict, mode="train")
        try:
            video = output["latent_pred"]
            action = output["action_pred"]
        except KeyError as exc:
            raise KeyError(
                "Wan diffusion output must contain latent_pred and action_pred"
            ) from exc
        return VAPrediction(video=video, action=action)

    def predict_joint(
        self,
        input_dict: dict[str, Any],
        noisy: VAPrediction,
        timesteps: VATimesteps,
    ) -> VADiffusionOutput:
        velocity = self.predict_velocity(input_dict)
        x0 = VAPrediction(
            video=self.velocity_to_x0(
                velocity.video,
                noisy.video,
                timesteps.video,
                modality="video",
            ),
            action=self.velocity_to_x0(
                velocity.action,
                noisy.action,
                timesteps.action,
                modality="action",
            ),
        )
        return VADiffusionOutput(velocity=velocity, x0=x0)
