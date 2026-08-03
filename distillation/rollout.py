"""Autoregressive MOT rollout helpers used by consistency training."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from inference.mot_inference import run_mot_inference
from wan_va.utils import FlowMatchScheduler

from distillation.schema import ReplayContext, VAMasks, VAPrediction, VATimesteps


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


@dataclass(frozen=True)
class _RecordedDenoiseStep:
    noisy: torch.Tensor
    timestep: torch.Tensor


class _RecordingFlowMatchScheduler(FlowMatchScheduler):
    def __init__(self, *args, record_step: int, **kwargs):
        self.record_step = int(record_step)
        self.step_index = 0
        self.record: _RecordedDenoiseStep | None = None
        super().__init__(*args, **kwargs)

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if self.step_index == self.record_step:
            self.record = _RecordedDenoiseStep(
                noisy=sample.detach().clone(),
                timestep=torch.as_tensor(timestep, device=sample.device).detach().clone(),
            )
        self.step_index += 1
        return super().step(model_output, timestep, sample, to_final=to_final, **kwargs)


class _RecordingSchedulerFactory:
    def __init__(self, video_step: int, action_step: int):
        self.record_steps = (int(video_step), int(action_step))
        self.schedulers: list[_RecordingFlowMatchScheduler] = []

    def __call__(self, **kwargs) -> _RecordingFlowMatchScheduler:
        if len(self.schedulers) >= len(self.record_steps):
            raise RuntimeError("MOT rollout created more schedulers than expected")
        scheduler = _RecordingFlowMatchScheduler(
            **kwargs,
            record_step=self.record_steps[len(self.schedulers)],
        )
        self.schedulers.append(scheduler)
        return scheduler

    def records(self) -> tuple[_RecordedDenoiseStep, _RecordedDenoiseStep]:
        if len(self.schedulers) != 2 or any(scheduler.record is None for scheduler in self.schedulers):
            raise RuntimeError("MOT rollout did not record both video and action denoising states")
        return self.schedulers[0].record, self.schedulers[1].record


def _chunk_batch(
    batch: dict[str, Any],
    frame_slice: slice,
) -> dict[str, Any]:
    out = dict(batch)
    for key in ("latents", "actions", "action_loss_mask", "action_valid_mask"):
        out[key] = batch[key][:, :, frame_slice].clone()
    for key in ("video_latent_loss_mask", "video_latent_valid_mask"):
        out[key] = batch[key][:, frame_slice].clone()
    for key in (
        "geometry_rgb",
        "geometry_pts3d",
        "geometry_point_valid_mask",
        "geometry_group_valid_mask",
    ):
        if key in batch:
            out[key] = batch[key][:, frame_slice].clone()
    return out


def _set_generation_masks(
    batch: dict[str, Any],
    generated_frames: torch.Tensor,
) -> None:
    video_valid = batch["video_latent_valid_mask"].to(dtype=torch.bool)
    batch["video_latent_loss_mask"] = video_valid & generated_frames
    action_frames = generated_frames[:, None, :, None, None]
    batch["action_loss_mask"] = batch["action_valid_mask"].to(dtype=torch.bool) & action_frames


def _recorded_rollout(
    batch: dict[str, Any],
    *,
    transformer,
    config: Any,
    spec: Any,
    device: torch.device,
    empty_text_emb: torch.Tensor,
    decode_latents_to_rgb_views: Callable[[torch.Tensor], torch.Tensor],
    video_record_step: int,
    action_record_step: int,
    run_inference: Callable[..., Any],
) -> tuple[Any, _RecordedDenoiseStep, _RecordedDenoiseStep]:
    scheduler_factory = _RecordingSchedulerFactory(
        video_record_step,
        action_record_step,
    )
    result = run_inference(
        batch,
        frame_count=int(spec.total_latent_frames),
        transformer=transformer,
        config=config,
        spec=spec,
        device=device,
        empty_text_emb=batch.get("empty_text_emb", empty_text_emb),
        decode_latents_to_rgb_views=decode_latents_to_rgb_views,
        scheduler_factory=scheduler_factory,
    )
    video_record, action_record = scheduler_factory.records()
    return result, video_record, action_record


@torch.no_grad()
def record_sgf_trajectory(
    batch: dict[str, Any],
    *,
    transformer,
    config: Any,
    spec: Any,
    device: torch.device,
    empty_text_emb: torch.Tensor,
    decode_latents_to_rgb_views: Callable[[torch.Tensor], torch.Tensor],
    video_record_step: int,
    action_record_step: int,
    run_inference: Callable[..., Any] = run_mot_inference,
) -> ReplayContext:
    """Record history and target rollouts, then assemble one full-window replay.

    以 ``chunk=4`` 为例，输入索引是 ``H0 H1 H2 H3 T0 T1 T2 T3``：

    - 第一次把 history slice 变成局部 4-frame generation window，H0 mask=0，
      H1..H3 mask=1；完整执行 video denoise、decode/recompute geometry、action
      denoise，并同时记录指定 video/action denoise step 的 noisy state；
    - 把 final pH1..pH3 写回 full batch，保留 T0 clean；
    - 第二次在 predicted history 条件下生成 pT1..pT3，并记录第二组 noisy；
    - 合并为 generated full window，replay mask 固定排除 H0/T0，并继续与原始
      valid mask 相交，因此 padding 和 invalid action token 不会进入 loss。

    假设 video scheduler 的 record timestep 是 600，action 是 320，则最终
    ``VATimesteps.video`` 在六个 replay frame 写 600，action 写 320；二者不可
    合并成一个 tensor。anchor 的 timestep 写 0、noisy=generated clean。

    函数对输入 batch 做 clone 后再改写，调用方的 dataset batch 保持不变。
    """
    chunk = int(spec.latent_frames_per_action_chunk_per_view)
    if (
        int(spec.history_latent_frames) != chunk
        or int(spec.target_latent_frames) != chunk
        or int(spec.total_latent_frames) != 2 * chunk
    ):
        raise ValueError("SGF recording requires one history chunk and one target chunk")

    video_steps = int(config.num_inference_steps)
    action_steps = int(config.action_num_inference_steps)
    if not 0 <= int(video_record_step) < video_steps:
        raise ValueError(f"video_record_step must be in [0,{video_steps}), got {video_record_step}")
    if not 0 <= int(action_record_step) < action_steps:
        raise ValueError(f"action_record_step must be in [0,{action_steps}), got {action_record_step}")

    batch_size = int(batch["latents"].shape[0])
    local_spec = replace(
        spec,
        history_chunks=0,
        target_chunks=1,
        history_latent_frames=0,
        target_latent_frames=chunk,
        total_latent_frames=chunk,
        attention_window_size=2,
    )
    history_batch = _chunk_batch(batch, slice(0, chunk))
    history_generated_frames = torch.ones(
        (batch_size, chunk),
        dtype=torch.bool,
        device=history_batch["latents"].device,
    )
    history_generated_frames[:, 0] = False
    _set_generation_masks(history_batch, history_generated_frames)
    history_result, history_video_record, history_action_record = _recorded_rollout(
        history_batch,
        transformer=transformer,
        config=config,
        spec=local_spec,
        device=device,
        empty_text_emb=empty_text_emb,
        decode_latents_to_rgb_views=decode_latents_to_rgb_views,
        video_record_step=video_record_step,
        action_record_step=action_record_step,
        run_inference=run_inference,
    )

    recorded_batch = dict(batch)
    for key in ("latents", "actions", "geometry_rgb"):
        recorded_batch[key] = batch[key].clone()
    history_target = slice(1, chunk)
    recorded_batch["latents"][:, :, history_target] = history_result.pred_latents[:, :, history_target]
    recorded_batch["actions"][:, :, history_target] = history_result.pred_actions[:, :, history_target]
    recorded_batch["geometry_rgb"][:, history_target] = history_result.action_geometry_rgb[:, history_target]
    del history_result

    target_generated_frames = torch.zeros(
        (batch_size, 2 * chunk),
        dtype=torch.bool,
        device=recorded_batch["latents"].device,
    )
    target_generated_frames[:, chunk + 1 :] = True
    _set_generation_masks(recorded_batch, target_generated_frames)
    target_result, target_video_record, target_action_record = _recorded_rollout(
        recorded_batch,
        transformer=transformer,
        config=config,
        spec=spec,
        device=device,
        empty_text_emb=empty_text_emb,
        decode_latents_to_rgb_views=decode_latents_to_rgb_views,
        video_record_step=video_record_step,
        action_record_step=action_record_step,
        run_inference=run_inference,
    )

    target_slice = slice(chunk + 1, 2 * chunk)
    recorded_batch["latents"][:, :, target_slice] = target_result.pred_latents[:, :, target_slice]
    recorded_batch["actions"][:, :, target_slice] = target_result.pred_actions[:, :, target_slice]
    recorded_batch["geometry_rgb"][:, target_slice] = target_result.action_geometry_rgb[:, target_slice]
    del target_result

    replay_frames = torch.zeros_like(batch["video_latent_valid_mask"], dtype=torch.bool)
    replay_frames[:, 1:chunk] = True
    replay_frames[:, chunk + 1 : 2 * chunk] = True
    video_mask = batch["video_latent_valid_mask"].to(dtype=torch.bool) & replay_frames
    action_mask = batch["action_valid_mask"].to(dtype=torch.bool) & replay_frames[:, None, :, None, None]
    recorded_batch["video_latent_loss_mask"] = video_mask
    recorded_batch["action_loss_mask"] = action_mask

    generated = VAPrediction(
        recorded_batch["latents"].detach(),
        recorded_batch["actions"].detach(),
    )
    noisy_video = generated.video.clone()
    noisy_action = generated.action.clone()
    noisy_video[:, :, history_target] = history_video_record.noisy
    noisy_video[:, :, target_slice] = target_video_record.noisy
    noisy_action[:, :, history_target] = history_action_record.noisy
    noisy_action[:, :, target_slice] = target_action_record.noisy

    video_timesteps = torch.zeros(
        (batch_size, 2 * chunk),
        dtype=torch.float32,
        device=generated.video.device,
    )
    action_timesteps = torch.zeros_like(video_timesteps)
    video_timesteps[:, history_target] = history_video_record.timestep
    video_timesteps[:, target_slice] = target_video_record.timestep
    action_timesteps[:, history_target] = history_action_record.timestep
    action_timesteps[:, target_slice] = target_action_record.timestep
    video_timesteps = torch.where(video_mask, video_timesteps, 0)
    action_timesteps = torch.where(action_mask.any(dim=(1, 3, 4)), action_timesteps, 0)

    return ReplayContext(
        batch=recorded_batch,
        timesteps=VATimesteps(video_timesteps, action_timesteps),
        noisy=VAPrediction(noisy_video, noisy_action),
        generated=generated,
        masks=VAMasks(video_mask, action_mask),
    )


def _frame_pair(
    batch: dict[str, Any],
    working: dict[str, torch.Tensor],
    pair: int,
    chunk: int,
) -> dict[str, Any]:
    history = slice(pair * chunk, (pair + 1) * chunk)
    target = slice((pair + 1) * chunk, (pair + 2) * chunk)
    local = {
        "latents": torch.cat(
            [working["latents"][:, :, history], working["latents"][:, :, target]],
            dim=2,
        ),
        "actions": torch.cat(
            [working["actions"][:, :, history], working["actions"][:, :, target]],
            dim=2,
        ),
        "action_loss_mask": torch.cat(
            [
                batch["action_loss_mask"][:, :, history],
                batch["action_loss_mask"][:, :, target],
            ],
            dim=2,
        ),
        "action_valid_mask": torch.cat(
            [
                batch["action_valid_mask"][:, :, history],
                batch["action_valid_mask"][:, :, target],
            ],
            dim=2,
        ),
        "video_latent_valid_mask": torch.cat(
            [
                batch["video_latent_valid_mask"][:, history],
                batch["video_latent_valid_mask"][:, target],
            ],
            dim=1,
        ),
        "geometry_rgb": torch.cat(
            [working["geometry_rgb"][:, history], working["geometry_rgb"][:, target]],
            dim=1,
        ),
        "geometry_group_valid_mask": torch.cat(
            [
                batch["geometry_group_valid_mask"][:, history],
                batch["geometry_group_valid_mask"][:, target],
            ],
            dim=1,
        ),
        "text_emb": batch["text_emb"],
        "stream_ids": batch["stream_ids"],
    }
    if "empty_text_emb" in batch:
        local["empty_text_emb"] = batch["empty_text_emb"]
    return local


@torch.no_grad()
def autoregressive_rollout(
    batch: dict[str, Any],
    *,
    transformer,
    config: Any,
    spec: Any,
    device: torch.device,
    empty_text_emb: torch.Tensor,
    decode_latents_to_rgb_views: Callable[[torch.Tensor], torch.Tensor],
    video_num_steps: int,
    action_num_steps: int,
    chunk_pairs: int,
) -> RolloutResult:
    """Generate target chunks sequentially, feeding each prediction into the next window."""
    chunk = int(spec.latent_frames_per_action_chunk_per_view)
    history = int(spec.history_latent_frames)
    if history != chunk:
        raise ValueError("MOT rollout currently requires one history chunk per fixed inference window")
    chunk_pairs = int(chunk_pairs)
    if chunk_pairs <= 0:
        raise ValueError(f"chunk_pairs must be positive, got {chunk_pairs}")
    if int(video_num_steps) <= 0 or int(action_num_steps) <= 0:
        raise ValueError(
            "rollout video/action denoising steps must be positive, got "
            f"{video_num_steps} and {action_num_steps}"
        )
    frames = int(batch["latents"].shape[2])
    available_pairs = (frames - history) // chunk
    if available_pairs < chunk_pairs:
        raise ValueError(
            f"rollout_chunk_pairs={chunk_pairs} needs {history + chunk_pairs * chunk} latent frames, "
            f"but batch has {frames}"
        )

    working = {
        "latents": batch["latents"].to(device=device).clone(),
        "actions": batch["actions"].to(device=device).clone(),
        "geometry_rgb": batch["geometry_rgb"].to(device=device).clone(),
    }
    target = {
        "latents": working["latents"].clone(),
        "actions": working["actions"].clone(),
        "geometry_rgb": working["geometry_rgb"].clone(),
    }
    rollout_config = copy.copy(config)
    rollout_config.num_inference_steps = int(video_num_steps)
    rollout_config.action_num_inference_steps = int(action_num_steps)

    for pair in range(chunk_pairs):
        local_batch = _frame_pair(batch, working, pair, chunk)
        result = run_mot_inference(
            local_batch,
            frame_count=int(spec.total_latent_frames),
            transformer=transformer,
            config=rollout_config,
            spec=spec,
            device=device,
            empty_text_emb=local_batch.get("empty_text_emb", empty_text_emb),
            decode_latents_to_rgb_views=decode_latents_to_rgb_views,
        )
        target_start = (pair + 1) * chunk
        target_end = target_start + chunk
        working["latents"][:, :, target_start:target_end] = result.pred_latents[:, :, history:]
        working["actions"][:, :, target_start:target_end] = result.pred_actions[:, :, history:]
        working["geometry_rgb"][:, target_start:target_end] = result.action_geometry_rgb[:, history:]

    horizon = history + chunk_pairs * chunk
    return RolloutResult(
        pred_latents=working["latents"][:, :, :horizon],
        target_latents=target["latents"][:, :, :horizon],
        pred_actions=working["actions"][:, :, :horizon],
        target_actions=target["actions"][:, :, :horizon],
        pred_geometry_rgb=working["geometry_rgb"][:, :horizon],
        target_geometry_rgb=target["geometry_rgb"][:, :horizon],
        action_valid_mask=batch["action_valid_mask"][:, :, :horizon].to(device=device),
        chunk_pairs=chunk_pairs,
        chunk_frames=chunk,
    )


def _wide_frames(rgb: torch.Tensor) -> np.ndarray:
    rgb = rgb[0].detach().float().cpu().permute(0, 3, 1, 4, 2).contiguous()
    rgb = rgb.reshape(rgb.shape[0], rgb.shape[1], rgb.shape[2] * rgb.shape[3], rgb.shape[4])
    return (rgb.clamp(0, 1).numpy() * 255).round().astype(np.uint8)


def _annotate(frame: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    banner = max(24, min(42, image.height // 10))
    draw.rectangle((0, 0, image.width, banner), fill=color)
    draw.text((8, max(4, banner // 4)), label, fill=(255, 255, 255))
    return np.asarray(image)


def _save_video(
    result: RolloutResult,
    decode: Callable[[torch.Tensor], torch.Tensor],
    path: Path,
    fps: int,
) -> None:
    import imageio.v2 as imageio

    pred_frames = []
    target_frames = []
    chunk = result.chunk_frames
    for pair in range(result.chunk_pairs):
        start, end = (pair + 1) * chunk, (pair + 2) * chunk
        pred_frames.append(_wide_frames(decode(result.pred_latents[:, :, start:end])))
        target_frames.append(_wide_frames(decode(result.target_latents[:, :, start:end])))
    pred = np.concatenate(pred_frames, axis=0)
    target = np.concatenate(target_frames, axis=0)
    frames = []
    for idx, (gt, generated) in enumerate(zip(target, pred, strict=True)):
        frames.append(
            np.concatenate(
                [
                    _annotate(gt, f"TARGET | frame {idx}", (37, 99, 235)),
                    _annotate(generated, f"GENERATED | frame {idx}", (217, 119, 6)),
                ],
                axis=1,
            )
        )
    imageio.mimsave(path, frames, fps=int(fps))


def _action_vectors(
    actions: torch.Tensor,
    mask: torch.Tensor,
    q01=None,
    q99=None,
) -> np.ndarray:
    values = actions[0, :, :, :, 0].permute(1, 2, 0)
    valid = mask[0, :, :, :, 0].all(dim=0)
    values = values[valid]
    if q01 is not None and q99 is not None:
        q01 = torch.as_tensor(q01, device=actions.device, dtype=values.dtype)
        q99 = torch.as_tensor(q99, device=actions.device, dtype=values.dtype)
        if q01.ndim > 1:
            q01, q99 = q01[0], q99[0]
        q01, q99 = q01.view(1, -1), q99.view(1, -1)
        values = (values + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    return values.detach().cpu().numpy()


def _save_action_plot(result: RolloutResult, batch: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    q01, q99 = batch.get("action_q01"), batch.get("action_q99")
    pred = _action_vectors(result.pred_actions, result.action_valid_mask, q01, q99)
    target = _action_vectors(result.target_actions, result.action_valid_mask, q01, q99)
    if pred.shape != target.shape or pred.shape[0] == 0:
        raise ValueError(
            "Rollout action visualization has incompatible shapes: "
            f"{pred.shape} and {target.shape}"
        )
    channels = pred.shape[1]
    columns = 4
    rows = (channels + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(16, max(3, rows * 2.5)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    x = np.arange(pred.shape[0])
    for idx, ax in enumerate(axes):
        if idx >= channels:
            ax.axis("off")
            continue
        ax.plot(x, target[:, idx], color="#2563eb", linewidth=1.2, label="target")
        ax.plot(x, pred[:, idx], color="#d97706", linewidth=1.0, label="generated")
        ax.set_title(f"action_{idx}", fontsize=8)
        ax.grid(True, linewidth=0.3)
    valid = result.action_valid_mask[0, :, :, :, 0].all(dim=0)
    chunk_counts = [
        int(valid[idx * result.chunk_frames : (idx + 1) * result.chunk_frames].sum())
        for idx in range(result.chunk_pairs + 1)
    ]
    boundaries = np.cumsum(chunk_counts)[:-1]
    for ax in axes[:channels]:
        for boundary_idx, boundary in enumerate(boundaries):
            ax.axvline(
                boundary - 0.5,
                color="#b45309" if boundary_idx == 0 else "#6b7280",
                linestyle="--" if boundary_idx == 0 else ":",
                linewidth=0.8,
            )
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rollout_artifacts(
    result: RolloutResult,
    batch: dict[str, Any],
    *,
    decode_latents_to_rgb_views: Callable[[torch.Tensor], torch.Tensor],
    output_dir: str | Path,
    video_fps: int = 10,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "rollout_target_vs_generated.mp4"
    action_path = output_dir / "rollout_actions.png"
    _save_video(result, decode_latents_to_rgb_views, video_path, video_fps)
    _save_action_plot(result, batch, action_path)
    torch.save(
        {
            "pred_latents": result.pred_latents.detach().cpu(),
            "target_latents": result.target_latents.detach().cpu(),
            "pred_actions": result.pred_actions.detach().cpu(),
            "target_actions": result.target_actions.detach().cpu(),
            "pred_geometry_rgb": result.pred_geometry_rgb.detach().cpu(),
            "target_geometry_rgb": result.target_geometry_rgb.detach().cpu(),
            "chunk_pairs": result.chunk_pairs,
            "chunk_frames": result.chunk_frames,
        },
        output_dir / "rollout.pt",
    )
    metadata = {
        "chunk_pairs": result.chunk_pairs,
        "chunk_frames": result.chunk_frames,
        "clean_target_anchor_per_chunk": True,
        "video": str(video_path),
        "actions": str(action_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "video": str(video_path),
        "actions": str(action_path),
        "tensor": str(output_dir / "rollout.pt"),
    }
