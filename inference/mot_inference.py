"""MOT inference sampling independent of the training runtime."""

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
    action_geometry_rgb: torch.Tensor


@dataclass(frozen=True)
class MOTGeometryInferenceResult:
    pred_depth: torch.Tensor
    pred_depth_conf: torch.Tensor
    pred_points: torch.Tensor
    geometry_rgb: torch.Tensor


@dataclass(frozen=True)
class MOTInferenceResult:
    pred_latents: torch.Tensor
    pred_actions: torch.Tensor
    pred_rgb: torch.Tensor
    pred_depth: torch.Tensor
    pred_depth_conf: torch.Tensor
    pred_points: torch.Tensor
    action_geometry_rgb: torch.Tensor


def decoded_rgb_to_geometry_groups(
    decoded_rgb: torch.Tensor,
    *,
    latent_frames: int,
    vae_temporal_factor: int = 4,
) -> torch.Tensor:
    latent_frames = int(latent_frames)
    vae_temporal_factor = int(vae_temporal_factor)
    expected_frames = 1 + vae_temporal_factor * (latent_frames - 1)
    if decoded_rgb.ndim != 6 or decoded_rgb.shape[1] != expected_frames:
        raise ValueError(
            "decoded MOT RGB must be "
            f"[B,{expected_frames},V,C,H,W], got {tuple(decoded_rgb.shape)}"
        )
    groups = [decoded_rgb[:, 0:1].expand(-1, 4, -1, -1, -1, -1)]
    for latent_idx in range(1, latent_frames):
        start = 1 + vae_temporal_factor * (latent_idx - 1)
        groups.append(decoded_rgb[:, start : start + vae_temporal_factor])
    return torch.stack(groups, dim=1)


def representative_geometry_frames(values: torch.Tensor, latent_frames: int) -> torch.Tensor:
    latent_frames = int(latent_frames)
    if values.shape[1] == latent_frames:
        return values
    if values.shape[1] == latent_frames * 4:
        return values.reshape(values.shape[0], latent_frames, 4, *values.shape[2:])[:, :, 0]
    raise ValueError(f"Cannot select representative geometry values for {latent_frames} frames from {tuple(values.shape)}")


def _inference_input(
    *,
    latent_noisy: torch.Tensor,
    latent_clean: torch.Tensor,
    action_noisy: torch.Tensor,
    action_clean: torch.Tensor,
    action_loss_mask: torch.Tensor,
    text_emb: torch.Tensor,
    rgb: torch.Tensor,
    stream_ids: torch.Tensor,
    latent_timesteps: torch.Tensor,
    action_timesteps: torch.Tensor,
    geometry_slot_valid_mask: torch.Tensor,
    geometry_condition: dict[str, Any],
    chunk_size: int,
    window_size: int,
    action_valid_mask: torch.Tensor,
    video_latent_valid_mask: torch.Tensor | None,
) -> dict[str, Any]:
    zeros = torch.zeros_like(latent_timesteps)
    out = {
        "latent_dict": {
            "noisy_latents": latent_noisy,
            "latent": latent_clean,
            "targets": torch.zeros_like(latent_clean),
            "timesteps": latent_timesteps,
            "cond_timesteps": zeros,
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
        "geometry_dict": {
            "rgb": rgb.to(dtype=MOT_INFERENCE_DTYPE),
            "stream_ids": stream_ids,
            "slot_valid_mask": geometry_slot_valid_mask.to(device=rgb.device, dtype=torch.bool),
            "precomputed_condition": geometry_condition,
        },
        "chunk_size": int(chunk_size),
        "window_size": int(window_size),
    }
    if video_latent_valid_mask is not None:
        out["latent_dict"]["video_latent_valid_mask"] = video_latent_valid_mask.to(
            device=latent_noisy.device,
            dtype=torch.bool,
        )
    return out


def _geometry_input(
    *,
    rgb: torch.Tensor,
    stream_ids: torch.Tensor,
    slot_valid_mask: torch.Tensor,
    chunk_size: int,
    window_size: int,
    return_points: bool,
) -> dict[str, Any]:
    return {
        "rgb": rgb.to(dtype=MOT_INFERENCE_DTYPE),
        "stream_ids": stream_ids,
        "chunk_size": int(chunk_size),
        "window_size": int(window_size),
        "return_points": bool(return_points),
        "slot_valid_mask": slot_valid_mask,
    }


def _empty_text_condition(empty_text_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    empty = empty_text_emb.to(device=text_emb.device, dtype=text_emb.dtype)
    if empty.ndim == text_emb.ndim - 1:
        empty = empty.unsqueeze(0)
    if empty.shape[0] == 1 and text_emb.shape[0] > 1:
        empty = empty.expand(text_emb.shape[0], *empty.shape[1:])
    if empty.shape != text_emb.shape:
        raise ValueError(
            f"empty text embedding shape {tuple(empty.shape)} does not match text embedding {tuple(text_emb.shape)}"
        )
    return empty


def _validate_complete_window(
    batch: dict[str, Any],
    frame_count: int,
    spec,
    *,
    require_latents: bool,
) -> None:
    if int(frame_count) != spec.total_latent_frames:
        raise ValueError(
            "MOT inference requires one complete GT-history/target window with "
            f"{spec.total_latent_frames} latent frames, got {frame_count}"
        )
    if require_latents and int(batch["latents"].shape[2]) != spec.total_latent_frames:
        raise ValueError(
            f"MOT inference batch must contain {spec.total_latent_frames} "
            f"latent frames, got {batch['latents'].shape[2]}"
        )
    if int(batch["geometry_rgb"].shape[1]) != spec.total_latent_frames:
        raise ValueError(
            f"MOT inference batch must contain {spec.total_latent_frames} "
            f"geometry groups, got {batch['geometry_rgb'].shape[1]}"
        )


def _target_slice(frame_count: int, spec) -> tuple[int, slice]:
    history_end = int(spec.history_latent_frames)
    generated_target_start = history_end + 1
    if generated_target_start >= frame_count:
        raise ValueError(
            "MOT inference requires at least one generated target latent after "
            "the GT target anchor"
        )
    return history_end, slice(generated_target_start, frame_count)


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
    """Generate target V while conditioning only on available GT history G."""

    _validate_complete_window(batch, frame_count, spec, require_latents=True)
    chunk_size = int(spec.latent_frames_per_action_chunk_per_view)
    window_size = int(spec.attention_window_size)
    video_scheduler = scheduler_factory(
        shift=config.snr_shift,
        sigma_min=0.0,
        extra_one_step=True,
    )
    video_scheduler.set_timesteps(int(config.num_inference_steps))
    guidance_scale = float(config.guidance_scale)

    gt_latents = batch["latents"][:, :, :frame_count].to(
        device=device,
        dtype=MOT_INFERENCE_DTYPE,
    )
    gt_actions = batch["actions"][:, :, :frame_count].to(device)
    action_loss_mask = batch["action_loss_mask"][:, :, :frame_count].to(device)
    action_valid_mask = batch.get(
        "action_valid_mask",
        batch["action_loss_mask"],
    )[:, :, :frame_count].to(device)
    text_emb = batch["text_emb"].to(device=device, dtype=MOT_INFERENCE_DTYPE)
    uncond_text_emb = _empty_text_condition(empty_text_emb, text_emb)
    geometry_rgb_gt = batch["geometry_rgb"][:, :frame_count].to(
        device,
        dtype=MOT_INFERENCE_DTYPE,
    )
    geometry_slot_valid_mask = batch["geometry_group_valid_mask"][
        :, :frame_count
    ].to(device, dtype=torch.bool)
    stream_ids = batch["stream_ids"].to(device)
    video_latent_valid_mask = batch.get("video_latent_valid_mask")
    if video_latent_valid_mask is not None:
        video_latent_valid_mask = video_latent_valid_mask[:, :frame_count].to(
            device,
            dtype=torch.bool,
        )

    history_end, target_slice = _target_slice(frame_count, spec)
    video_geometry_rgb = geometry_rgb_gt.clone()
    video_geometry_rgb[:, target_slice] = 0
    video_geometry_slot_valid = geometry_slot_valid_mask.clone()
    video_geometry_slot_valid[:, target_slice] = False
    video_geometry_condition = transformer(
        _geometry_input(
            rgb=video_geometry_rgb,
            stream_ids=stream_ids,
            slot_valid_mask=video_geometry_slot_valid,
            chunk_size=chunk_size,
            window_size=window_size,
            return_points=False,
        ),
        mode="precompute_geometry",
    )

    video_action_noisy = gt_actions.clone()
    video_action_noisy[:, :, target_slice] = 0
    video_action_clean = video_action_noisy.clone()
    video_action_valid_mask = action_valid_mask.clone()
    video_action_valid_mask[:, :, target_slice] = False
    cur_latents = torch.randn_like(gt_latents[:, :, target_slice])
    for timestep in video_scheduler.timesteps.to(device):
        latent_noisy = gt_latents.clone()
        latent_clean = gt_latents.clone()
        latent_noisy[:, :, target_slice] = cur_latents
        latent_clean[:, :, target_slice] = 0
        latent_timesteps = torch.zeros(
            (gt_latents.shape[0], frame_count),
            device=device,
            dtype=torch.float32,
        )
        latent_timesteps[:, target_slice] = timestep
        action_timesteps = torch.zeros(
            (gt_actions.shape[0], frame_count),
            device=device,
            dtype=torch.float32,
        )
        common_kwargs = {
            "latent_noisy": latent_noisy,
            "latent_clean": latent_clean,
            "action_noisy": video_action_noisy,
            "action_clean": video_action_clean,
            "action_loss_mask": action_loss_mask,
            "rgb": video_geometry_rgb,
            "stream_ids": stream_ids,
            "latent_timesteps": latent_timesteps,
            "action_timesteps": action_timesteps,
            "geometry_condition": video_geometry_condition,
            "action_valid_mask": video_action_valid_mask,
            "video_latent_valid_mask": video_latent_valid_mask,
            "geometry_slot_valid_mask": video_geometry_slot_valid,
            "chunk_size": chunk_size,
            "window_size": window_size,
        }
        cond_out = transformer(
            _inference_input(text_emb=text_emb, **common_kwargs),
            mode="inference_video",
        )
        uncond_out = transformer(
            _inference_input(text_emb=uncond_text_emb, **common_kwargs),
            mode="inference_video",
        )
        latent_prediction = uncond_out["latent_pred"] + guidance_scale * (
            cond_out["latent_pred"] - uncond_out["latent_pred"]
        )
        cur_latents = video_scheduler.step(
            latent_prediction[:, :, target_slice],
            timestep,
            cur_latents,
        )

    pred_latents = gt_latents.clone()
    pred_latents[:, :, target_slice] = cur_latents
    decoded_target_rgb = decode_latents_to_rgb_views(
        pred_latents[:, :, history_end:frame_count]
    )
    decoded_target_geometry = decoded_rgb_to_geometry_groups(
        decoded_target_rgb,
        latent_frames=int(spec.target_latent_frames),
        vae_temporal_factor=int(config.vae_temporal_factor),
    )
    pred_rgb = geometry_rgb_gt[:, :, 0].clone()
    pred_rgb[:, target_slice] = decoded_target_geometry[:, 1:, 0].to(
        dtype=pred_rgb.dtype
    )
    action_geometry_rgb = geometry_rgb_gt.clone()
    action_geometry_rgb[:, target_slice] = decoded_target_geometry[:, 1:].to(
        dtype=action_geometry_rgb.dtype
    )
    return MOTVideoInferenceResult(
        pred_latents=pred_latents,
        pred_rgb=pred_rgb,
        action_geometry_rgb=action_geometry_rgb,
    )


def _run_geometry_from_rgb(
    batch: dict[str, Any],
    frame_count: int,
    *,
    geometry_rgb: torch.Tensor,
    transformer,
    spec,
    device: torch.device,
) -> tuple[MOTGeometryInferenceResult, dict[str, Any]]:
    chunk_size = int(spec.latent_frames_per_action_chunk_per_view)
    window_size = int(spec.attention_window_size)
    geometry_rgb = geometry_rgb[:, :frame_count].to(
        device,
        dtype=MOT_INFERENCE_DTYPE,
    )
    condition = transformer(
        _geometry_input(
            rgb=geometry_rgb,
            stream_ids=batch["stream_ids"].to(device),
            slot_valid_mask=batch["geometry_group_valid_mask"][
                :, :frame_count
            ].to(device, dtype=torch.bool),
            chunk_size=chunk_size,
            window_size=window_size,
            return_points=True,
        ),
        mode="precompute_geometry",
    )
    result = MOTGeometryInferenceResult(
        pred_depth=representative_geometry_frames(
            condition["depth"],
            frame_count,
        ),
        pred_depth_conf=representative_geometry_frames(
            condition["depth_conf"],
            frame_count,
        ),
        pred_points=condition["points"],
        geometry_rgb=geometry_rgb,
    )
    return result, condition


@torch.no_grad()
def run_geometry_inference(
    batch: dict[str, Any],
    frame_count: int,
    *,
    transformer,
    spec,
    device: torch.device,
) -> MOTGeometryInferenceResult:
    """Predict G outputs from the dataset's GT synchronized RGB views."""

    _validate_complete_window(batch, frame_count, spec, require_latents=False)
    result, _condition = _run_geometry_from_rgb(
        batch,
        frame_count,
        geometry_rgb=batch["geometry_rgb"],
        transformer=transformer,
        spec=spec,
        device=device,
    )
    return result


def _sample_actions(
    batch: dict[str, Any],
    frame_count: int,
    *,
    pred_latents: torch.Tensor,
    geometry_rgb: torch.Tensor,
    geometry_condition: dict[str, Any],
    transformer,
    config,
    spec,
    device: torch.device,
    scheduler_factory,
) -> torch.Tensor:
    if float(config.action_guidance_scale) != 1.0:
        raise ValueError("Current MOT inference requires action_guidance_scale=1")
    action_scheduler = scheduler_factory(
        shift=config.action_snr_shift,
        sigma_min=0.0,
        extra_one_step=True,
    )
    action_scheduler.set_timesteps(int(config.action_num_inference_steps))
    gt_actions = batch["actions"][:, :, :frame_count].to(device)
    action_loss_mask = batch["action_loss_mask"][:, :, :frame_count].to(device)
    action_valid_mask = batch.get(
        "action_valid_mask",
        batch["action_loss_mask"],
    )[:, :, :frame_count].to(device)
    text_emb = batch["text_emb"].to(device=device, dtype=MOT_INFERENCE_DTYPE)
    geometry_slot_valid_mask = batch["geometry_group_valid_mask"][
        :, :frame_count
    ].to(device, dtype=torch.bool)
    video_latent_valid_mask = batch.get("video_latent_valid_mask")
    if video_latent_valid_mask is not None:
        video_latent_valid_mask = video_latent_valid_mask[:, :frame_count].to(
            device,
            dtype=torch.bool,
        )
    stream_ids = batch["stream_ids"].to(device)
    chunk_size = int(spec.latent_frames_per_action_chunk_per_view)
    window_size = int(spec.attention_window_size)
    _history_end, target_slice = _target_slice(frame_count, spec)

    pred_actions = gt_actions.clone()
    pred_actions[:, :, target_slice] = 0
    cur_actions = torch.randn_like(gt_actions[:, :, target_slice])
    target_action_mask = action_loss_mask[:, :, target_slice].to(gt_actions.dtype)
    cur_actions = cur_actions * target_action_mask
    for timestep in action_scheduler.timesteps.to(device):
        action_noisy = pred_actions.clone()
        action_clean = pred_actions.clone()
        action_noisy[:, :, target_slice] = cur_actions
        action_clean[:, :, target_slice] = 0
        action_timesteps = torch.zeros(
            (gt_actions.shape[0], frame_count),
            device=device,
            dtype=torch.float32,
        )
        action_timesteps[:, target_slice] = timestep
        latent_timesteps = torch.zeros(
            (pred_latents.shape[0], frame_count),
            device=device,
            dtype=torch.float32,
        )
        out = transformer(
            _inference_input(
                latent_noisy=pred_latents,
                latent_clean=pred_latents,
                action_noisy=action_noisy,
                action_clean=action_clean,
                action_loss_mask=action_loss_mask,
                text_emb=text_emb,
                rgb=geometry_rgb,
                stream_ids=stream_ids,
                latent_timesteps=latent_timesteps,
                action_timesteps=action_timesteps,
                geometry_condition=geometry_condition,
                action_valid_mask=action_valid_mask,
                video_latent_valid_mask=video_latent_valid_mask,
                geometry_slot_valid_mask=geometry_slot_valid_mask,
                chunk_size=chunk_size,
                window_size=window_size,
            ),
            mode="inference_action",
        )
        cur_actions = action_scheduler.step(
            out["action_pred"][:, :, target_slice],
            timestep,
            cur_actions,
        )
        cur_actions = cur_actions * target_action_mask
    pred_actions[:, :, target_slice] = cur_actions
    return pred_actions


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
    """Generate V, recompute G from generated RGB, then generate A from V/G."""

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
    geometry, geometry_condition = _run_geometry_from_rgb(
        batch,
        frame_count,
        geometry_rgb=video.action_geometry_rgb,
        transformer=transformer,
        spec=spec,
        device=device,
    )
    pred_actions = _sample_actions(
        batch,
        frame_count,
        pred_latents=video.pred_latents,
        geometry_rgb=video.action_geometry_rgb,
        geometry_condition=geometry_condition,
        transformer=transformer,
        config=config,
        spec=spec,
        device=device,
        scheduler_factory=scheduler_factory,
    )
    return MOTInferenceResult(
        pred_latents=video.pred_latents,
        pred_actions=pred_actions,
        pred_rgb=video.pred_rgb,
        pred_depth=geometry.pred_depth,
        pred_depth_conf=geometry.pred_depth_conf,
        pred_points=geometry.pred_points,
        action_geometry_rgb=video.action_geometry_rgb,
    )
