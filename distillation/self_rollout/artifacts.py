"""Visualization and artifact export for incremental self rollout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .result import RolloutResult


def rollout_artifact_due(completed_step: int, interval: int) -> bool:
    """Return whether this completed optimizer step should emit artifacts."""

    interval = int(interval)
    return interval > 0 and int(completed_step) % interval == 0


def _generated_bounds(result: RolloutResult) -> tuple[int, int]:
    total_frames = int(result.pred_latents.shape[2])
    rollout_frames = int(result.diagnostics.get("rollout_frames", 0))
    if not 0 < rollout_frames <= total_frames:
        raise ValueError(
            "rollout diagnostics must provide a valid positive rollout_frames"
        )
    return total_frames - rollout_frames, total_frames


def _wide_frames(rgb: torch.Tensor) -> np.ndarray:
    """Join all camera views horizontally for the first batch sample."""

    rgb = rgb[0].detach().float().cpu().permute(0, 3, 1, 4, 2).contiguous()
    rgb = rgb.reshape(
        rgb.shape[0],
        rgb.shape[1],
        rgb.shape[2] * rgb.shape[3],
        rgb.shape[4],
    )
    return (rgb.clamp(0, 1).numpy() * 255).round().astype(np.uint8)


def _annotate(
    frame: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> np.ndarray:
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
) -> int:
    """Save target/generated RGB views for the actually generated latent range."""

    import imageio.v2 as imageio

    generated_start, generated_end = _generated_bounds(result)
    pred = _wide_frames(
        decode(result.pred_latents[:, :, generated_start:generated_end])
    )
    target = _wide_frames(
        decode(result.target_latents[:, :, generated_start:generated_end])
    )
    if pred.shape != target.shape:
        raise ValueError(
            "decoded target/generated rollout shapes differ: "
            f"{target.shape} and {pred.shape}"
        )
    frames = [
        np.concatenate(
            [
                _annotate(gt, f"TARGET | decoded frame {index}", (37, 99, 235)),
                _annotate(
                    generated,
                    f"GENERATED | decoded frame {index}",
                    (217, 119, 6),
                ),
            ],
            axis=1,
        )
        for index, (gt, generated) in enumerate(zip(target, pred, strict=True))
    ]
    imageio.mimsave(path, frames, fps=int(fps))
    return len(frames)


def _action_vectors(
    actions: torch.Tensor,
    mask: torch.Tensor,
    q01=None,
    q99=None,
) -> tuple[np.ndarray, torch.Tensor]:
    values = actions[0, :, :, :, 0].permute(1, 2, 0)
    valid = mask[0, :, :, :, 0].all(dim=0)
    values = values[valid]
    if q01 is not None and q99 is not None:
        q01 = torch.as_tensor(q01, device=actions.device, dtype=values.dtype)
        q99 = torch.as_tensor(q99, device=actions.device, dtype=values.dtype)
        if q01.ndim > 1:
            q01, q99 = q01[0], q99[0]
        values = (values + 1.0) / 2.0 * (
            q99.view(1, -1) - q01.view(1, -1) + 1e-6
        ) + q01.view(1, -1)
    return values.detach().cpu().numpy(), valid


def _save_action_plot(
    result: RolloutResult,
    batch: dict[str, Any],
    path: Path,
) -> None:
    """Plot the full action trajectory and mark where generated frames begin."""

    import matplotlib.pyplot as plt

    q01, q99 = batch.get("action_q01"), batch.get("action_q99")
    pred, valid = _action_vectors(
        result.pred_actions,
        result.action_valid_mask,
        q01,
        q99,
    )
    target, _ = _action_vectors(
        result.target_actions,
        result.action_valid_mask,
        q01,
        q99,
    )
    if pred.shape != target.shape or pred.shape[0] == 0:
        raise ValueError(
            "rollout action visualization has incompatible shapes: "
            f"{pred.shape} and {target.shape}"
        )

    channels = pred.shape[1]
    columns = 4
    rows = (channels + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, max(3, rows * 2.5)),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(-1)
    x = np.arange(pred.shape[0])
    generated_start, _ = _generated_bounds(result)
    generated_boundary = int(valid[:generated_start].sum().item())

    for index, axis in enumerate(axes):
        if index >= channels:
            axis.axis("off")
            continue
        axis.plot(x, target[:, index], color="#2563eb", linewidth=1.2, label="target")
        axis.plot(x, pred[:, index], color="#d97706", linewidth=1.0, label="generated")
        if 0 < generated_boundary < pred.shape[0]:
            axis.axvline(
                generated_boundary - 0.5,
                color="#b45309",
                linestyle="--",
                linewidth=0.9,
            )
        axis.set_title(f"action_{index}", fontsize=8)
        axis.grid(True, linewidth=0.3)
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
    """Save consistency-training rollout video, action plot, tensors, and metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "rollout_target_vs_generated.mp4"
    action_path = output_dir / "rollout_actions.png"
    tensor_path = output_dir / "rollout.pt"
    metadata_path = output_dir / "metadata.json"

    decoded_frames = _save_video(
        result,
        decode_latents_to_rgb_views,
        video_path,
        video_fps,
    )
    _save_action_plot(result, batch, action_path)
    torch.save(
        {
            "pred_latents": result.pred_latents.detach().cpu(),
            "target_latents": result.target_latents.detach().cpu(),
            "pred_actions": result.pred_actions.detach().cpu(),
            "target_actions": result.target_actions.detach().cpu(),
            "pred_geometry_rgb": result.pred_geometry_rgb.detach().cpu(),
            "target_geometry_rgb": result.target_geometry_rgb.detach().cpu(),
            "action_valid_mask": result.action_valid_mask.detach().cpu(),
            "chunk_pairs": result.chunk_pairs,
            "chunk_frames": result.chunk_frames,
            "diagnostics": result.diagnostics,
        },
        tensor_path,
    )

    generated_start, generated_end = _generated_bounds(result)
    metadata = {
        "profile": result.diagnostics.get("profile"),
        "profile_version": result.diagnostics.get("profile_version"),
        "chunk_pairs": result.chunk_pairs,
        "chunk_frames": result.chunk_frames,
        "rollout_frames": generated_end - generated_start,
        "generated_start_frame": generated_start,
        "generated_end_frame_exclusive": generated_end,
        "decoded_video_frames": decoded_frames,
        "video": str(video_path),
        "actions": str(action_path),
        "tensor": str(tensor_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "video": str(video_path),
        "actions": str(action_path),
        "tensor": str(tensor_path),
        "metadata": str(metadata_path),
    }
