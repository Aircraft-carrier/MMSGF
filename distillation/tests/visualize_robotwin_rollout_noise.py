#!/usr/bin/env python3
"""Create one RoboTwin self-rollout comparison MP4.

Layout:
  - top left: predicted head-camera video (GT + Gaussian noise)
  - bottom left: GT head-camera video
  - right: GT/pred absolute dual-arm 3D action trajectories

The action panel denormalizes the model's 20D relative actions and uses the
dataset reference state to reconstruct absolute position, quaternion rotation,
and gripper values with the production conversion formula.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wan_va.dataset.mot_dataset import (
    MotTrainData,
    quaternion_xyzw_to_matrix,
    relative_20d_to_absolute_actions,
)

HEAD_STREAM_ID = 1
ARM_NAMES = ("left", "right")
ARM_MARKERS = ("o", "^")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/robotwin_one_task_test"),
    )
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=48)
    parser.add_argument("--rollout-frames", type=int, default=3)
    parser.add_argument(
        "--video-frame-count",
        type=int,
        default=49,
        help="Number of consecutive raw head-camera frames to render.",
    )
    parser.add_argument("--video-noise-std", type=float, default=0.008)
    parser.add_argument("--action-noise-std", type=float, default=0.01)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("distillation/tests/visu/robotwin_rollout_comparison.mp4"),
    )
    return parser.parse_args()


def _dataset_from_root(root: Path) -> tuple[MotTrainData, dict]:
    config = json.loads((root / "meta/mot_config.json").read_text(encoding="utf-8"))
    manifest = root / "meta/mot_final_training_manifest.jsonl"
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    text_cache = {
        str(row["segment"]["action_text"]): torch.empty(0) for row in rows
    }
    dataset = MotTrainData(
        manifest_path=manifest,
        action_sequence_length=int(config["action_sequence_length"]),
        action_dim=int(config["action_dim"]),
        norm_stats_by_task=config["norm_stats_by_task"],
        action_chunk_size=int(config["action_chunk_size"]),
        video_downsample_ratio=int(config["video_downsample_ratio"]),
        text_emb_cache_path=None,
        text_emb_cache=text_cache,
        action_cache_manifest_path=config.get("action_cache_manifest_path") or None,
        random_start=False,
    )
    return dataset, config


def _continuous_head_frames(
    dataset: MotTrainData,
    row_index: int,
    start_frame: int,
    frame_count: int,
) -> torch.Tensor:
    """Load consecutive episode frames as [F,C,H,W] from the head camera."""

    row = dataset.rows[int(row_index) % len(dataset.rows)]
    segment_start = int(row["segment"]["start_frame"])
    segment_end = int(row["segment"]["end_frame"])
    end_frame = int(start_frame) + int(frame_count)
    if int(start_frame) < segment_start or end_frame > segment_end:
        raise ValueError(
            "continuous video range exceeds the episode segment: "
            f"requested [{start_frame},{end_frame}), available [{segment_start},{segment_end})"
        )
    head_views = [
        index
        for index, view in enumerate(row["views"])
        if int(view.get("stream_id", -1)) == HEAD_STREAM_ID
    ]
    if len(head_views) != 1:
        raise ValueError(f"expected exactly one head camera, found {len(head_views)}")
    frame_ids = list(range(int(start_frame), end_frame))
    return dataset._load_rgb(row, frame_ids)[:, head_views[0]].contiguous()


def _generated_absolute_actions(
    sample: dict,
    pred_normalized: torch.Tensor,
    generated_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert valid generated GT/pred actions to absolute 16D EEF actions."""

    gt_normalized = sample["actions"][:, generated_start:, :, 0].permute(1, 2, 0)
    pred_normalized = pred_normalized[:, generated_start:, :, 0].permute(1, 2, 0)
    valid = sample["action_valid_mask"][:, generated_start:, :, 0].all(dim=0)
    references = sample["action_reference_states"][:, generated_start:, :, 0].permute(1, 2, 0)
    q01 = sample["action_q01"].view(1, 1, -1)
    q99 = sample["action_q99"].view(1, 1, -1)

    def denormalize(values: torch.Tensor) -> np.ndarray:
        relative = (values + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01
        return relative[valid].detach().cpu().numpy()

    reference_values = references[valid].detach().cpu().numpy()
    gt_absolute = relative_20d_to_absolute_actions(
        reference_values,
        denormalize(gt_normalized),
    )
    pred_absolute = relative_20d_to_absolute_actions(
        reference_values,
        denormalize(pred_normalized),
    )
    if gt_absolute.shape[0] == 0:
        raise ValueError("selected rollout window contains no valid generated actions")
    return gt_absolute, pred_absolute


def _axis_limits(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.concatenate(
        [gt[:, 0:3], gt[:, 8:11], pred[:, 0:3], pred[:, 8:11]], axis=0
    )
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    center = (low + high) * 0.5
    radius = max(float((high - low).max()) * 0.62, 0.025)
    return center - radius, center + radius


def _colored_path(
    axis,
    positions: np.ndarray,
    current: int,
    *,
    cmap: str,
    marker: str,
) -> None:
    shown = positions[: current + 1]
    if len(shown) > 1:
        segments = np.stack([shown[:-1], shown[1:]], axis=1)
        progress = np.linspace(0.35, 1.0, max(len(positions) - 1, 1))[: len(segments)]
        collection = Line3DCollection(
            segments,
            cmap=cmap,
            norm=plt.Normalize(0.0, 1.0),
            linewidth=3.0,
            alpha=0.95,
        )
        collection.set_array(progress)
        axis.add_collection3d(collection)
    color = plt.get_cmap(cmap)(1.0)
    axis.scatter(*shown[-1], color=color, marker=marker, s=55, depthshade=False)


def _orientation_axes(
    axis,
    action: np.ndarray,
    arm_index: int,
    *,
    scale: float,
    linestyle: str,
    alpha: float,
) -> None:
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
    width: int,
    height: int,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    axis = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f8fafc")
    axis.set_facecolor("#f8fafc")

    for arm_index, marker in enumerate(ARM_MARKERS):
        offset = arm_index * 8
        _colored_path(axis, gt[:, offset : offset + 3], current, cmap="Blues", marker=marker)
        _colored_path(axis, pred[:, offset : offset + 3], current, cmap="Oranges", marker=marker)
        _orientation_axes(
            axis,
            gt[current],
            arm_index,
            scale=float((high - low).max()) * 0.07,
            linestyle="--",
            alpha=0.65,
        )
        _orientation_axes(
            axis,
            pred[current],
            arm_index,
            scale=float((high - low).max()) * 0.07,
            linestyle="-",
            alpha=1.0,
        )
        endpoint = pred[current, offset : offset + 3]
        gripper = pred[current, offset + 7]
        axis.text(*endpoint, f" {ARM_NAMES[arm_index][0].upper()} g={gripper:.2f}", fontsize=8)

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
        f"light → dark = time | action {current + 1}/{len(gt)}",
        fontsize=13,
        weight="bold",
        pad=16,
    )
    legend = [
        Line2D([0], [0], color=plt.get_cmap("Blues")(0.9), lw=3, label="GT trajectory"),
        Line2D([0], [0], color=plt.get_cmap("Oranges")(0.9), lw=3, label="Pred trajectory"),
        Line2D([0], [0], color="#475569", marker="o", lw=0, label="Left arm"),
        Line2D([0], [0], color="#475569", marker="^", lw=0, label="Right arm"),
        Line2D([0], [0], color="#475569", ls="--", label="GT rotation frame"),
        Line2D([0], [0], color="#475569", ls="-", label="Pred rotation frame"),
    ]
    axis.legend(handles=legend, loc="upper left", fontsize=8)
    fig.tight_layout(pad=1.2)
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return panel


def _video_panel(frame: torch.Tensor, label: str, banner_color: str) -> Image.Image:
    array = (
        frame.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    ).round().astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image)
    banner_height = max(34, image.height // 12)
    draw.rectangle((0, 0, image.width, banner_height), fill=banner_color)
    draw.text((12, 9), label, fill="white", font=ImageFont.load_default(size=18))
    return image


def _composite_frame(
    pred_video: torch.Tensor,
    gt_video: torch.Tensor,
    video_index: int,
    action_panel: np.ndarray,
) -> np.ndarray:
    pred = _video_panel(pred_video[video_index], "PRED VIDEO - HEAD CAMERA", "#c2410c")
    gt = _video_panel(gt_video[video_index], "GT VIDEO - HEAD CAMERA", "#1d4ed8")
    left_width = max(pred.width, gt.width)
    half_height = action_panel.shape[0] // 2
    pred = pred.resize((left_width, half_height), Image.Resampling.LANCZOS)
    gt = gt.resize((left_width, action_panel.shape[0] - half_height), Image.Resampling.LANCZOS)
    left = Image.new("RGB", (left_width, action_panel.shape[0]), "white")
    left.paste(pred, (0, 0))
    left.paste(gt, (0, half_height))
    right = Image.fromarray(action_panel, mode="RGB")
    canvas = Image.new("RGB", (left.width + right.width, right.height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return np.asarray(canvas)


def main() -> None:
    args = _parse_args()
    if args.video_noise_std < 0 or args.action_noise_std < 0:
        raise ValueError("noise standard deviations must be non-negative")
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    dataset, config = _dataset_from_root(args.dataset_root.resolve())
    sample = dataset.get_window(args.row_index, args.start_frame)
    logical_video_frames = int(config["latent_frames_per_action_chunk_per_view"]) * 2
    if not 0 < args.rollout_frames <= logical_video_frames:
        raise ValueError(
            f"rollout_frames must be in [1,{logical_video_frames}], got {args.rollout_frames}"
        )
    if args.video_frame_count <= 0:
        raise ValueError("video_frame_count must be positive")
    generated_start = logical_video_frames - args.rollout_frames
    gt_video = _continuous_head_frames(
        dataset,
        args.row_index,
        args.start_frame,
        args.video_frame_count,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    pred_video = (
        gt_video
        + torch.randn(gt_video.shape, generator=generator) * args.video_noise_std
    ).clamp_(0, 1)
    pred_normalized = sample["actions"].clone()
    action_noise = torch.randn(
        pred_normalized[:, generated_start:].shape,
        generator=generator,
    )
    pred_normalized[:, generated_start:] += (
        action_noise
        * args.action_noise_std
        * sample["action_valid_mask"][:, generated_start:]
    )
    gt_actions, pred_actions = _generated_absolute_actions(
        sample,
        pred_normalized,
        generated_start,
    )

    low, high = _axis_limits(gt_actions, pred_actions)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_size = 960
    with imageio.get_writer(
        output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
    ) as writer:
        for video_index in range(len(gt_video)):
            if len(gt_video) == 1:
                action_index = len(gt_actions) - 1
            else:
                action_index = round(
                    video_index * (len(gt_actions) - 1) / (len(gt_video) - 1)
                )
            action_panel = _render_action_panel(
                gt_actions,
                pred_actions,
                action_index,
                width=panel_size,
                height=panel_size,
                low=low,
                high=high,
            )
            writer.append_data(
                _composite_frame(pred_video, gt_video, video_index, action_panel)
            )
    print(output)


if __name__ == "__main__":
    main()
