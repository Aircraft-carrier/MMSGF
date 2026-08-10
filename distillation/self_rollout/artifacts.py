"""Single-MP4 visualization export for incremental self rollout."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .result import RolloutResult

HEAD_STREAM_ID = 1
ARM_NAMES = ("left", "right")
ARM_MARKERS = ("o", "^")


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


def _decoded_head_video(
    result: RolloutResult,
    batch: dict[str, Any],
    decode: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the rollout anchor plus generated range and select head camera."""

    generated_start, generated_end = _generated_bounds(result)
    # Include T0 so three generated latent intervals decode to the complete
    # anchor-to-T3 video rather than losing the first temporal VAE interval.
    decode_start = max(0, generated_start - 1)
    pred = decode(result.pred_latents[:, :, decode_start:generated_end])
    target = decode(result.target_latents[:, :, decode_start:generated_end])
    if pred.shape != target.shape or pred.ndim != 6:
        raise ValueError(
            "decoded rollout must produce matching [B,F,V,C,H,W] tensors, "
            f"got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    stream_ids = batch.get("stream_ids")
    if not torch.is_tensor(stream_ids) or stream_ids.ndim != 2:
        raise ValueError("rollout visualization requires stream_ids [B,V]")
    head = torch.nonzero(stream_ids[0].detach().cpu() == HEAD_STREAM_ID).flatten()
    if head.numel() != 1:
        raise ValueError(
            f"rollout visualization requires exactly one head camera stream_id={HEAD_STREAM_ID}, "
            f"found {head.numel()}"
        )
    view = int(head.item())
    return (
        pred[0, :, view].detach().float().cpu(),
        target[0, :, view].detach().float().cpu(),
    )


def _absolute_generated_actions(
    result: RolloutResult,
    batch: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Restore generated GT/pred 20D relative actions to absolute 16D EEF."""

    from wan_va.dataset.mot_dataset import relative_20d_to_absolute_actions

    generated_start, generated_end = _generated_bounds(result)
    references = batch.get("action_reference_states")
    q01 = batch.get("action_q01")
    q99 = batch.get("action_q99")
    if not all(torch.is_tensor(value) for value in (references, q01, q99)):
        raise ValueError(
            "absolute action visualization requires action_reference_states, "
            "action_q01, and action_q99 tensors"
        )

    gt_normalized = result.target_actions[
        0, :, generated_start:generated_end, :, 0
    ].permute(1, 2, 0)
    pred_normalized = result.pred_actions[
        0, :, generated_start:generated_end, :, 0
    ].permute(1, 2, 0)
    valid = result.action_valid_mask[
        0, :, generated_start:generated_end, :, 0
    ].all(dim=0)
    reference_values = references[
        0, :, generated_start:generated_end, :, 0
    ].permute(1, 2, 0)
    q01 = q01[0] if q01.ndim > 1 else q01
    q99 = q99[0] if q99.ndim > 1 else q99
    q01 = q01.to(device=gt_normalized.device, dtype=gt_normalized.dtype).view(1, 1, -1)
    q99 = q99.to(device=gt_normalized.device, dtype=gt_normalized.dtype).view(1, 1, -1)

    def denormalize(values: torch.Tensor) -> np.ndarray:
        relative = (values + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01
        return relative[valid].detach().float().cpu().numpy()

    reference_values = reference_values[valid].detach().float().cpu().numpy()
    gt = relative_20d_to_absolute_actions(
        reference_values,
        denormalize(gt_normalized),
    )
    pred = relative_20d_to_absolute_actions(
        reference_values,
        denormalize(pred_normalized),
    )
    if gt.shape != pred.shape or gt.shape[0] == 0:
        raise ValueError(
            "rollout absolute action visualization has incompatible or empty "
            f"trajectories: {gt.shape} and {pred.shape}"
        )
    return gt, pred


def _axis_limits(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.concatenate(
        [gt[:, 0:3], gt[:, 8:11], pred[:, 0:3], pred[:, 8:11]], axis=0
    )
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    center = (low + high) * 0.5
    radius = max(float((high - low).max()) * 0.62, 0.025)
    return center - radius, center + radius


def _colored_path(axis, positions, current, *, cmap, marker) -> None:
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    shown = positions[: current + 1]
    if len(shown) > 1:
        segments = np.stack([shown[:-1], shown[1:]], axis=1)
        progress = np.linspace(0.35, 1.0, max(len(positions) - 1, 1))[
            : len(segments)
        ]
        collection = Line3DCollection(
            segments,
            cmap=cmap,
            norm=plt.Normalize(0.0, 1.0),
            linewidth=3.0,
            alpha=0.95,
        )
        collection.set_array(progress)
        axis.add_collection3d(collection)
    axis.scatter(
        *shown[-1],
        color=plt.get_cmap(cmap)(1.0),
        marker=marker,
        s=55,
        depthshade=False,
    )


def _orientation_axes(
    axis,
    action: np.ndarray,
    arm_index: int,
    *,
    scale: float,
    linestyle: str,
    alpha: float,
) -> None:
    from wan_va.dataset.mot_dataset import quaternion_xyzw_to_matrix

    offset = arm_index * 8
    origin = action[offset : offset + 3]
    rotation = quaternion_xyzw_to_matrix(action[offset + 3 : offset + 7])
    for column, color in enumerate(("#ef4444", "#22c55e", "#3b82f6")):
        endpoint = origin + rotation[:, column] * scale
        axis.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            [origin[2], endpoint[2]],
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            alpha=alpha,
        )


def _render_action_panel(
    gt: np.ndarray,
    pred: np.ndarray,
    current: int,
    *,
    size: int,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    axis = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f8fafc")
    axis.set_facecolor("#f8fafc")
    scale = float((high - low).max()) * 0.07

    for arm_index, marker in enumerate(ARM_MARKERS):
        offset = arm_index * 8
        _colored_path(
            axis,
            gt[:, offset : offset + 3],
            current,
            cmap="Blues",
            marker=marker,
        )
        _colored_path(
            axis,
            pred[:, offset : offset + 3],
            current,
            cmap="Oranges",
            marker=marker,
        )
        _orientation_axes(
            axis,
            gt[current],
            arm_index,
            scale=scale,
            linestyle="--",
            alpha=0.65,
        )
        _orientation_axes(
            axis,
            pred[current],
            arm_index,
            scale=scale,
            linestyle="-",
            alpha=1.0,
        )
        endpoint = pred[current, offset : offset + 3]
        axis.text(
            *endpoint,
            f" {ARM_NAMES[arm_index][0].upper()} g={pred[current, offset + 7]:.2f}",
            fontsize=8,
        )

    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(low[2], high[2])
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24, azim=-58)
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.grid(True, alpha=0.28)
    axis.set_title(
        "ABSOLUTE EEF ACTION TRAJECTORY\n"
        f"light -> dark = time | action {current + 1}/{len(gt)}",
        fontsize=13,
        weight="bold",
        pad=16,
    )
    axis.legend(
        handles=[
            Line2D([0], [0], color=plt.get_cmap("Blues")(0.9), lw=3, label="GT trajectory"),
            Line2D([0], [0], color=plt.get_cmap("Oranges")(0.9), lw=3, label="Pred trajectory"),
            Line2D([0], [0], color="#475569", marker="o", lw=0, label="Left arm"),
            Line2D([0], [0], color="#475569", marker="^", lw=0, label="Right arm"),
            Line2D([0], [0], color="#475569", ls="--", label="GT rotation frame"),
            Line2D([0], [0], color="#475569", ls="-", label="Pred rotation frame"),
        ],
        loc="upper left",
        fontsize=8,
    )
    fig.tight_layout(pad=1.2)
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return panel


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _video_panel(
    frame: torch.Tensor,
    label: str,
    banner_color: str,
    *,
    width: int,
    height: int,
):
    from PIL import Image, ImageDraw

    array = (
        frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).round().astype(np.uint8)
    source = Image.fromarray(array, mode="RGB")
    source.thumbnail((width, height), Image.Resampling.LANCZOS)
    image = Image.new("RGB", (width, height), "black")
    image.paste(source, ((width - source.width) // 2, (height - source.height) // 2))
    draw = ImageDraw.Draw(image)
    banner_height = max(34, height // 12)
    draw.rectangle((0, 0, width, banner_height), fill=banner_color)
    draw.text((12, 8), label, fill="white", font=_font(18))
    return image


def _composite_frame(
    pred_video: torch.Tensor,
    gt_video: torch.Tensor,
    video_index: int,
    action_panel: np.ndarray,
) -> np.ndarray:
    from PIL import Image

    panel_height, panel_width = action_panel.shape[:2]
    left_width = panel_width * 2 // 3
    half_height = panel_height // 2
    pred = _video_panel(
        pred_video[video_index],
        "PRED VIDEO - HEAD CAMERA",
        "#c2410c",
        width=left_width,
        height=half_height,
    )
    gt = _video_panel(
        gt_video[video_index],
        "GT VIDEO - HEAD CAMERA",
        "#1d4ed8",
        width=left_width,
        height=panel_height - half_height,
    )
    canvas = Image.new("RGB", (left_width + panel_width, panel_height), "white")
    canvas.paste(pred, (0, 0))
    canvas.paste(gt, (0, half_height))
    canvas.paste(Image.fromarray(action_panel, mode="RGB"), (left_width, 0))
    return np.asarray(canvas)


def _save_comparison_video(
    result: RolloutResult,
    batch: dict[str, Any],
    decode: Callable[[torch.Tensor], torch.Tensor],
    path: Path,
    fps: int,
) -> None:
    import imageio.v2 as imageio

    pred_video, gt_video = _decoded_head_video(result, batch, decode)
    gt_actions, pred_actions = _absolute_generated_actions(result, batch)
    low, high = _axis_limits(gt_actions, pred_actions)
    panel_size = 960
    with imageio.get_writer(
        path,
        fps=int(fps),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
    ) as writer:
        for action_index in range(len(gt_actions)):
            if len(gt_actions) == 1:
                video_index = len(gt_video) - 1
            else:
                video_index = round(
                    action_index * (len(gt_video) - 1) / (len(gt_actions) - 1)
                )
            action_panel = _render_action_panel(
                gt_actions,
                pred_actions,
                action_index,
                size=panel_size,
                low=low,
                high=high,
            )
            writer.append_data(
                _composite_frame(pred_video, gt_video, video_index, action_panel)
            )


def save_rollout_artifacts(
    result: RolloutResult,
    batch: dict[str, Any],
    *,
    decode_latents_to_rgb_views: Callable[[torch.Tensor], torch.Tensor],
    output_dir: str | Path,
    video_fps: int = 10,
) -> dict[str, str]:
    """Save the training rollout as one video/action comparison MP4."""

    if int(video_fps) <= 0:
        raise ValueError("video_fps must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "rollout_video_action_comparison.mp4"
    _save_comparison_video(
        result,
        batch,
        decode_latents_to_rgb_views,
        video_path,
        int(video_fps),
    )
    return {"video": str(video_path)}
