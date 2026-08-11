"""Shared model state, input, and tensor helpers."""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from distillation.diffusion_utils import add_noise
from distillation.schema import VAMasks, VAPair, VATimesteps

if TYPE_CHECKING:
    from wan_va.utils.scheduler import FlowMatchScheduler


_MISSING = object()


def add_noise_to_va(
    clean: VAPair,
    noise: VAPair,
    timesteps: VATimesteps,
    masks: VAMasks,
    video_scheduler: "FlowMatchScheduler",
    action_scheduler: "FlowMatchScheduler",
) -> VAPair:
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
    return VAPair(
        video=torch.where(video_mask, noisy_video, clean.video),
        action=torch.where(masks.action, noisy_action, clean.action),
    )


def replace_va_streams(
    base_input: dict,
    noisy: VAPair,
    clean: VAPair,
    timesteps: VATimesteps,
) -> dict:
    """Install one synthetic V/A state into the native MOT training input.

    ``_prepare_joint_input_dict(add_noise=False)`` intentionally omits the native
    autoregressive ``targets`` tensors because distillation computes its losses
    outside ``VAMOTTransformer3DModel``. The model's train forward still reads
    ``action_dict["targets"].shape`` when reshaping its output, so the adapter
    keeps that structural key as a shape carrier.

    Video values use ``[B,Cv,F,V,H,W]``, action values use
    ``[B,Ca,F,N,1]``, and both timestep tensors use ``[B,F]``. The returned
    dictionaries preserve text, masks, and attention-window metadata from
    ``base_input`` while replacing only the V/A trajectory fields.
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
    clean: VAPair,
    masks: VAMasks,
) -> VAPair:
    video_mask = masks.video[:, None, :, None, None, None]
    return VAPair(
        video=torch.where(video_mask, torch.zeros_like(clean.video), clean.video),
        action=torch.where(masks.action, torch.zeros_like(clean.action), clean.action),
    )


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval().requires_grad_(False)
    return model


def set_trainable(model: nn.Module) -> nn.Module:
    model.train().requires_grad_(True)
    return model


@contextmanager
def temporary_masked_attention_backend(model: nn.Module, backend: str | None):
    if backend is None:
        yield
        return

    if str(backend) not in {"dense", "fa4", "flex"}:
        raise ValueError(f"unsupported masked attention backend: {backend!r}")

    modules = [model]
    previous = [
        (module, getattr(module, "masked_attn_backend", _MISSING))
        for module in modules
    ]
    try:
        for module, _old_backend in previous:
            module.masked_attn_backend = str(backend)
        yield
    finally:
        for module, old_backend in previous:
            if old_backend is _MISSING:
                if hasattr(module, "masked_attn_backend"):
                    delattr(module, "masked_attn_backend")
            else:
                module.masked_attn_backend = old_backend


@contextmanager
def temporary_fsdp_unshard(model: nn.Module):
    """Temporarily materialize FSDP2 DTensor parameters for internal calls.

    Distillation self-rollout intentionally calls MOT submodules directly
    to maintain an incremental KV cache.  That bypasses FSDP2 root pre-forward
    hooks, so FSDP parameters can remain as DTensors while rollout inputs are
    local tensors.  Use this only around no-grad rollout sections, then reshard
    immediately before returning to normal training forwards.
    """

    try:
        from torch.distributed._composable.fsdp import FSDPModule
    except Exception:
        FSDPModule = ()

    fsdp_modules: list[nn.Module] = []
    for module in model.modules():
        is_fsdp_module = isinstance(module, FSDPModule) if FSDPModule else False
        if is_fsdp_module or (
            callable(getattr(module, "unshard", None))
            and callable(getattr(module, "reshard", None))
        ):
            fsdp_modules.append(module)
    unsharded: list[nn.Module] = []
    try:
        for module in fsdp_modules:
            module.unshard()
            unsharded.append(module)
        yield
    finally:
        for module in reversed(unsharded):
            module.reshard()


@torch.no_grad()
def update_ema(ema_model: nn.Module, student_model: nn.Module, decay: float) -> None:
    for ema_param, student_param in zip(
        ema_model.parameters(),
        student_model.parameters(),
        strict=True,
    ):
        ema_param.mul_(decay).add_(student_param, alpha=1.0 - decay)
