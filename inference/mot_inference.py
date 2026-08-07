"""Video+Action MOT inference independent of the training runtime."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from wan_va.utils import FlowMatchScheduler


MOT_INFERENCE_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class MOTVideoInferenceResult:
    pred_latents: torch.Tensor
    pred_rgb: torch.Tensor


@dataclass(frozen=True)
class MOTInferenceResult:
    pred_latents: torch.Tensor
    pred_actions: torch.Tensor
    pred_rgb: torch.Tensor


def decoded_rgb_to_latent_frames(
    decoded_rgb: torch.Tensor,
    *,
    latent_frames: int,
    vae_temporal_factor: int = 4,
) -> torch.Tensor:
    """Select one display frame for each latent frame from streaming VAE output."""

    expected = 1 + int(vae_temporal_factor) * (int(latent_frames) - 1)
    if decoded_rgb.ndim != 6 or decoded_rgb.shape[1] != expected:
        raise ValueError(
            f"decoded MOT RGB must be [B,{expected},V,C,H,W], got {tuple(decoded_rgb.shape)}"
        )
    indices = [0] + [1 + int(vae_temporal_factor) * index for index in range(int(latent_frames) - 1)]
    return decoded_rgb[:, indices]


def _inference_input(
    *,
    latent_noisy: torch.Tensor,
    latent_clean: torch.Tensor,
    action_noisy: torch.Tensor,
    action_clean: torch.Tensor,
    action_loss_mask: torch.Tensor,
    text_emb: torch.Tensor,
    stream_ids: torch.Tensor,
    latent_timesteps: torch.Tensor,
    action_timesteps: torch.Tensor,
    chunk_size: int,
    window_size: int,
    action_valid_mask: torch.Tensor,
    video_latent_valid_mask: torch.Tensor | None,
) -> dict[str, Any]:
    out = {
        "latent_dict": {
            "noisy_latents": latent_noisy,
            "latent": latent_clean,
            "targets": torch.zeros_like(latent_clean),
            "timesteps": latent_timesteps,
            "cond_timesteps": torch.zeros_like(latent_timesteps),
            "text_emb": text_emb,
        },
        "action_dict": {
            "noisy_latents": action_noisy,
            "latent": action_clean,
            "targets": torch.zeros_like(action_clean),
            "timesteps": action_timesteps,
            "cond_timesteps": torch.zeros_like(action_timesteps),
            "action_loss_mask": action_loss_mask,
            "action_valid_mask": action_valid_mask,
            "text_emb": text_emb,
        },
        "stream_ids": stream_ids,
        "chunk_size": int(chunk_size),
        "window_size": int(window_size),
    }
    if video_latent_valid_mask is not None:
        out["latent_dict"]["video_latent_valid_mask"] = video_latent_valid_mask
    return out


def _empty_text_condition(empty_text_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    empty = empty_text_emb.to(device=text_emb.device, dtype=text_emb.dtype)
    if empty.ndim == text_emb.ndim - 1:
        empty = empty.unsqueeze(0)
    if empty.shape[0] == 1 and text_emb.shape[0] > 1:
        empty = empty.expand(text_emb.shape[0], *empty.shape[1:])
    if empty.shape != text_emb.shape:
        raise ValueError(
            f"empty text embedding shape {tuple(empty.shape)} does not match {tuple(text_emb.shape)}"
        )
    return empty


def _validate_complete_window(batch: dict[str, Any], frame_count: int, spec) -> None:
    if int(frame_count) != spec.total_latent_frames:
        raise ValueError(
            f"MOT inference requires {spec.total_latent_frames} latent frames, got {frame_count}"
        )
    if int(batch["latents"].shape[2]) != spec.total_latent_frames:
        raise ValueError("MOT inference batch does not contain one complete latent window")


def _target_slice(frame_count: int, spec) -> tuple[int, slice]:
    history_end = int(spec.history_latent_frames)
    generated_start = history_end + 1
    if generated_start >= frame_count:
        raise ValueError("MOT inference requires a generated frame after the target anchor")
    return history_end, slice(generated_start, frame_count)


def _batch_inputs(batch: dict[str, Any], frame_count: int, device: torch.device):
    latents = batch["latents"][:, :, :frame_count].to(device=device, dtype=MOT_INFERENCE_DTYPE)
    actions = batch["actions"][:, :, :frame_count].to(device=device, dtype=MOT_INFERENCE_DTYPE)
    loss_mask = batch["action_loss_mask"][:, :, :frame_count].to(device)
    valid_mask = batch.get("action_valid_mask", batch["action_loss_mask"])[
        :, :, :frame_count
    ].to(device)
    video_valid = batch.get("video_latent_valid_mask")
    if video_valid is not None:
        video_valid = video_valid[:, :frame_count].to(device=device, dtype=torch.bool)
    return latents, actions, loss_mask, valid_mask, video_valid


@torch.no_grad()
def run_video_inference(
    batch: dict[str, Any],
    frame_count: int,
    *,
    transformer,
    config,
    spec,
    device: torch.device,
    empty_text_emb: torch.Tensor,
    decode_latents_to_rgb_views: Callable[..., torch.Tensor],
    scheduler_factory=FlowMatchScheduler,
) -> MOTVideoInferenceResult:
    """Generate the target video stream from GT history and anchor."""

    _validate_complete_window(batch, frame_count, spec)
    latents, actions, action_loss, action_valid, video_valid = _batch_inputs(
        batch, frame_count, device
    )
    text = batch["text_emb"].to(device=device, dtype=MOT_INFERENCE_DTYPE)
    empty_text = _empty_text_condition(empty_text_emb, text)
    stream_ids = batch["stream_ids"].to(device)
    history_end, target_slice = _target_slice(frame_count, spec)
    scheduler = scheduler_factory(shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(int(config.num_inference_steps))

    action_condition = actions.clone()
    action_condition[:, :, target_slice] = 0
    action_condition_valid = action_valid.clone()
    action_condition_valid[:, :, target_slice] = False
    sample = torch.randn_like(latents[:, :, target_slice])
    for timestep in scheduler.timesteps.to(device):
        noisy = latents.clone()
        clean = latents.clone()
        noisy[:, :, target_slice] = sample
        clean[:, :, target_slice] = 0
        video_t = torch.zeros((latents.shape[0], frame_count), device=device)
        video_t[:, target_slice] = timestep
        action_t = torch.zeros_like(video_t)
        common = dict(
            latent_noisy=noisy,
            latent_clean=clean,
            action_noisy=action_condition,
            action_clean=action_condition,
            action_loss_mask=action_loss,
            stream_ids=stream_ids,
            latent_timesteps=video_t,
            action_timesteps=action_t,
            chunk_size=int(spec.latent_frames_per_action_chunk_per_view),
            window_size=int(spec.attention_window_size),
            action_valid_mask=action_condition_valid,
            video_latent_valid_mask=video_valid,
        )
        conditional = transformer(
            _inference_input(text_emb=text, **common), mode="inference_video"
        )["latent_pred"]
        guidance = float(config.guidance_scale)
        prediction = conditional
        if guidance != 1.0:
            unconditional = transformer(
                _inference_input(text_emb=empty_text, **common), mode="inference_video"
            )["latent_pred"]
            prediction = unconditional + guidance * (conditional - unconditional)
        sample = scheduler.step(prediction[:, :, target_slice], timestep, sample)

    pred_latents = latents.clone()
    pred_latents[:, :, target_slice] = sample
    decoded = decode_latents_to_rgb_views(pred_latents[:, :, history_end:frame_count])
    pred_rgb = decoded_rgb_to_latent_frames(
        decoded,
        latent_frames=int(spec.target_latent_frames),
        vae_temporal_factor=int(config.vae_temporal_factor),
    )
    return MOTVideoInferenceResult(pred_latents=pred_latents, pred_rgb=pred_rgb)


@torch.no_grad()
def run_mot_inference(
    batch: dict[str, Any],
    frame_count: int,
    *,
    transformer,
    config,
    spec,
    device: torch.device,
    empty_text_emb: torch.Tensor,
    decode_latents_to_rgb_views: Callable[..., torch.Tensor],
    scheduler_factory=FlowMatchScheduler,
) -> MOTInferenceResult:
    """Generate target Video, then generate Action from the resulting VA context."""

    video = run_video_inference(
        batch,
        frame_count,
        transformer=transformer,
        config=config,
        spec=spec,
        device=device,
        empty_text_emb=empty_text_emb,
        decode_latents_to_rgb_views=decode_latents_to_rgb_views,
        scheduler_factory=scheduler_factory,
    )
    _latents, actions, action_loss, action_valid, video_valid = _batch_inputs(
        batch, frame_count, device
    )
    if float(config.action_guidance_scale) != 1.0:
        raise ValueError("Current MOT inference requires action_guidance_scale=1")
    scheduler = scheduler_factory(
        shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True
    )
    scheduler.set_timesteps(int(config.action_num_inference_steps))
    text = batch["text_emb"].to(device=device, dtype=MOT_INFERENCE_DTYPE)
    stream_ids = batch["stream_ids"].to(device)
    _history_end, target_slice = _target_slice(frame_count, spec)
    pred_actions = actions.clone()
    pred_actions[:, :, target_slice] = 0
    sample = torch.randn_like(actions[:, :, target_slice])
    target_mask = action_loss[:, :, target_slice].to(dtype=sample.dtype)
    sample = sample * target_mask
    for timestep in scheduler.timesteps.to(device):
        noisy = pred_actions.clone()
        clean = pred_actions.clone()
        noisy[:, :, target_slice] = sample
        clean[:, :, target_slice] = 0
        action_t = torch.zeros((actions.shape[0], frame_count), device=device)
        action_t[:, target_slice] = timestep
        video_t = torch.zeros_like(action_t)
        prediction = transformer(
            _inference_input(
                latent_noisy=video.pred_latents,
                latent_clean=video.pred_latents,
                action_noisy=noisy,
                action_clean=clean,
                action_loss_mask=action_loss,
                text_emb=text,
                stream_ids=stream_ids,
                latent_timesteps=video_t,
                action_timesteps=action_t,
                chunk_size=int(spec.latent_frames_per_action_chunk_per_view),
                window_size=int(spec.attention_window_size),
                action_valid_mask=action_valid,
                video_latent_valid_mask=video_valid,
            ),
            mode="inference_action",
        )["action_pred"]
        sample = scheduler.step(prediction[:, :, target_slice], timestep, sample)
        sample = sample * target_mask
    pred_actions[:, :, target_slice] = sample
    return MOTInferenceResult(
        pred_latents=video.pred_latents,
        pred_actions=pred_actions,
        pred_rgb=video.pred_rgb,
    )
