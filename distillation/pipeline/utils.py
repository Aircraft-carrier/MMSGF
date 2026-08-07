"""Shared pipeline tensor helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from distillation.schema import VAMasks, VAPrediction, VATimesteps
from distillation.scheduler import add_noise

if TYPE_CHECKING:
    from wan_va.utils.scheduler import FlowMatchScheduler


def add_noise_to_va(
    clean: VAPrediction,
    noise: VAPrediction,
    timesteps: VATimesteps,
    masks: VAMasks,
    video_scheduler: "FlowMatchScheduler",
    action_scheduler: "FlowMatchScheduler",
) -> VAPrediction:
    noisy_video = add_noise(
        clean.video,
        noise.video,
        timesteps.video,
        video_scheduler,
    )
    noisy_action = add_noise(
        clean.action,
        noise.action,
        timesteps.action,
        action_scheduler,
    )
    video_mask = masks.video[:, None, :, None, None, None]
    return VAPrediction(
        video=torch.where(video_mask, noisy_video, clean.video),
        action=torch.where(masks.action, noisy_action, clean.action),
    )


def replace_va_streams(
    base_input: dict,
    noisy: VAPrediction,
    clean: VAPrediction,
    timesteps: VATimesteps,
) -> dict:
    """Install one synthetic V/A state into the native MOT training input.

    ``_prepare_joint_input_dict(add_noise=False)`` intentionally omits the native
    autoregressive ``targets`` tensors because distillation computes its losses
    outside ``VAMOTTransformer3DModel``.  The model's train forward still
    reads ``action_dict["targets"].shape`` when reshaping its output, so the
    adapter must keep that structural key.  The value below is only a shape
    carrier; consistency/replay/DMD losses never read it.

    Example input shapes used throughout stage2/stage3:

    - ``noisy.video`` / ``clean.video``: ``[B,Cv,F,V,H,W]``;
    - ``noisy.action`` / ``clean.action``: ``[B,Ca,F,N,1]``;
    - ``timesteps.video`` / ``timesteps.action``: ``[B,F]``.

    The returned dictionaries preserve text, masks and attention-window
    metadata from ``base_input`` and replace only the V/A trajectory fields.
    """
    latent_dict = {
        **base_input["latent_dict"],
        "noisy_latents": noisy.video,
        "latent": clean.video,
        "targets": base_input["latent_dict"].get("targets", clean.video),
        "timesteps": timesteps.video,
        "cond_timesteps": torch.zeros_like(timesteps.video),
    }
    action_dict = {
        **base_input["action_dict"],
        "noisy_latents": noisy.action,
        "latent": clean.action,
        "targets": base_input["action_dict"].get("targets", clean.action),
        "timesteps": timesteps.action,
        "cond_timesteps": torch.zeros_like(timesteps.action),
    }
    return {
        **base_input,
        "latent_dict": latent_dict,
        "action_dict": action_dict,
    }


def replace_text_condition(input_dict: dict, text_emb: torch.Tensor) -> dict:
    return {
        **input_dict,
        "latent_dict": {
            **input_dict["latent_dict"],
            "text_emb": text_emb,
        },
        "action_dict": {
            **input_dict["action_dict"],
            "text_emb": text_emb,
        },
    }


def mask_clean_targets(
    clean: VAPrediction,
    masks: VAMasks,
) -> VAPrediction:
    video_mask = masks.video[:, None, :, None, None, None]
    return VAPrediction(
        video=torch.where(video_mask, torch.zeros_like(clean.video), clean.video),
        action=torch.where(masks.action, torch.zeros_like(clean.action), clean.action),
    )
