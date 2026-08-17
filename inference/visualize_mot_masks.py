#!/usr/bin/env python
"""Render fixed-window and autoregressive VA-MOT attention masks."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap

from wan_va.modules.mot_attention import (
    STREAM_ACTION,
    build_dense_mot_mask,
    build_x_metadata,
)


FRAMES = 8
CHUNK_SIZE = 4
WINDOW_SIZE = 16
VIDEO_TOKENS_PER_FRAME = 1
ACTION_TOKENS_PER_FRAME = 1


def _token_valid_ids() -> torch.Tensor:
    video = torch.ones((1, FRAMES * VIDEO_TOKENS_PER_FRAME), dtype=torch.bool)
    action = torch.zeros((1, FRAMES, ACTION_TOKENS_PER_FRAME), dtype=torch.bool)
    action[:, 1:4] = True
    action[:, 5:8] = True
    action = action.reshape(1, -1)
    return torch.cat([video, video, action, action], dim=1)


def _metadata(*, autoregressive: bool):
    metadata = build_x_metadata(
        batch_size=1,
        video_tokens_per_frame=VIDEO_TOKENS_PER_FRAME,
        action_tokens_per_frame=ACTION_TOKENS_PER_FRAME,
        num_frames=FRAMES,
        chunk_size=CHUNK_SIZE,
        window_size=WINDOW_SIZE,
        device=torch.device("cpu"),
        token_valid_ids=_token_valid_ids(),
    )
    if not autoregressive:
        return metadata

    frame_ids = torch.arange(FRAMES)
    history_order = torch.div(frame_ids, CHUNK_SIZE, rounding_mode="floor") * 2
    target_order = 2 + 2 * (frame_ids - CHUNK_SIZE)
    video_order = torch.where(frame_ids < CHUNK_SIZE, history_order, target_order)
    action_order = video_order + 1
    metadata.order_ids = torch.where(
        metadata.stream_ids == STREAM_ACTION,
        action_order[metadata.frame_ids],
        video_order[metadata.frame_ids],
    )
    metadata.cache_key = None
    metadata.structure_cache_key = None
    return metadata


def _group_layout() -> tuple[list[int], list[str]]:
    video_length = FRAMES * VIDEO_TOKENS_PER_FRAME
    action_length = FRAMES * ACTION_TOKENS_PER_FRAME
    boundaries = [
        0,
        video_length,
        2 * video_length,
        2 * video_length + action_length,
        2 * video_length + 2 * action_length,
    ]
    labels = ["NV", "CV", "NA", "CA"]
    return boundaries, labels


def _token_labels(group_labels: list[str]) -> list[str]:
    return [
        f"{group}{frame}"
        for group in group_labels
        for frame in range(FRAMES)
    ]


def _draw_frame_boundaries(ax, offset: int, tokens_per_frame: int) -> None:
    for frame in range(1, FRAMES):
        position = offset + frame * tokens_per_frame - 0.5
        ax.axhline(position, color="#94a3b8", linewidth=0.25, alpha=0.55)
        ax.axvline(position, color="#94a3b8", linewidth=0.25, alpha=0.55)


def render_mask(path: Path, *, autoregressive: bool) -> None:
    metadata = _metadata(autoregressive=autoregressive)
    mask = build_dense_mot_mask(metadata)[0].cpu().numpy()
    boundaries, group_labels = _group_layout()
    token_labels = _token_labels(group_labels)

    figure, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
    ax.imshow(
        mask,
        interpolation="nearest",
        origin="upper",
        cmap=ListedColormap(["#ffffff", "#111827"]),
        vmin=0,
        vmax=1,
        rasterized=False,
    )
    ax.set_xticks(range(len(token_labels)), labels=token_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(token_labels)), labels=token_labels, fontsize=7)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    kind = "Autoregressive target-frame" if autoregressive else "Bidirectional fixed-window"
    ax.set_title(
        f"{kind} VA-MOT attention mask\n"
        "8 frames; 1 view × 1 video token/frame; 1 action token/frame",
        pad=24,
    )

    for boundary in boundaries[1:-1]:
        position = boundary - 0.5
        ax.axhline(position, color="#dc2626", linewidth=1.0)
        ax.axvline(position, color="#dc2626", linewidth=1.0)

    video_length = FRAMES * VIDEO_TOKENS_PER_FRAME
    action_length = FRAMES * ACTION_TOKENS_PER_FRAME
    _draw_frame_boundaries(ax, 0, VIDEO_TOKENS_PER_FRAME)
    _draw_frame_boundaries(ax, video_length, VIDEO_TOKENS_PER_FRAME)
    _draw_frame_boundaries(ax, 2 * video_length, ACTION_TOKENS_PER_FRAME)
    _draw_frame_boundaries(
        ax, 2 * video_length + action_length, ACTION_TOKENS_PER_FRAME
    )

    ax.text(
        0.01,
        -0.035,
        "NV/CV/NA/CA = noisy video / clean video / noisy action / clean action; suffix = frame id\n"
        "black = visible, white = masked; red = stream boundary; gray = frame boundary",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, facecolor="white")
    plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "mask_visualizations",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    render_mask(args.output_dir / "bidirectional_mask.png", autoregressive=False)
    render_mask(args.output_dir / "autoregressive_mask.png", autoregressive=True)


if __name__ == "__main__":
    main()
