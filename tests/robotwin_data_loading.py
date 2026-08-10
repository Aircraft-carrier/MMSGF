#!/usr/bin/env python3
"""Interactively load one prepared RoboTwin MOT sample.

Example:
    python -m ipdb tests/robotwin_data_loading.py \
        --dataset-root /path/to/prepared_robotwin_dataset --sample-index 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.dataset.mot_dataset import MotTrainData


def _path_from_metadata(dataset_root: Path, value: str | None, *, field: str) -> Path:
    if not value:
        raise ValueError(f"meta/mot_config.json is missing {field!r}")
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }
    if value.numel() and value.is_floating_point():
        summary["min"] = float(value.min())
        summary["max"] = float(value.max())
    return summary


def load_robotwin_sample(
    dataset_root: Path,
    *,
    sample_index: int,
    start_frame: int | None,
    random_start: bool,
    no_text_emb: bool = False,
) -> tuple[MotTrainData, dict[str, Any]]:
    """Construct the runtime dataset from RoboTwin metadata and load one sample."""

    dataset_root = dataset_root.resolve()
    metadata_path = dataset_root / "meta" / "mot_config.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Prepared RoboTwin MOT metadata was not found: {metadata_path}. "
            "Run build_robotwin_eef_training_dataset.py first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_path = _path_from_metadata(dataset_root, metadata.get("mot_manifest_path"), field="mot_manifest_path")
    text_emb_cache = None
    if no_text_emb:
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        text_emb_cache = {row["segment"]["action_text"]: torch.empty(0) for row in rows}
    dataset = MotTrainData(
        manifest_path=manifest_path,
        action_sequence_length=int(metadata["action_sequence_length"]),
        action_dim=int(metadata["action_dim"]),
        norm_stats_by_task=metadata["norm_stats_by_task"],
        action_chunk_size=int(metadata["action_chunk_size"]),
        video_downsample_ratio=int(metadata["video_downsample_ratio"]),
        text_emb_cache_path=(
            None if no_text_emb else _path_from_metadata(
                dataset_root, metadata.get("text_emb_cache_path"), field="text_emb_cache_path"
            )
        ),
        empty_emb_path=(
            None if no_text_emb else _path_from_metadata(dataset_root, metadata.get("empty_emb_path"), field="empty_emb_path")
        ),
        text_emb_cache=text_emb_cache,
        action_cache_manifest_path=(
            _path_from_metadata(dataset_root, metadata["action_cache_manifest_path"], field="action_cache_manifest_path")
            if metadata.get("action_cache_manifest_path")
            else None
        ),
        random_start=random_start,
    )
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample_index={sample_index} is outside [0, {len(dataset)})")
    sample = dataset[sample_index] if start_frame is None else dataset.get_window(sample_index, start_frame)
    return dataset, sample


def main() -> None:
    """Run the one-task CPU loading smoke test.

    python tests/robotwin_data_loading.py \\
      --dataset-root data/robotwin_one_task_test \\
      --sample-index 0 \\
      --start-frame 0 \\
      --no-text-emb
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.getenv("ROBOTWIN_MOT_DATASET_ROOT") or os.getenv("MOT_DATASET_ROOT"),
        required=not (os.getenv("ROBOTWIN_MOT_DATASET_ROOT") or os.getenv("MOT_DATASET_ROOT")),
        help="Prepared dataset root containing meta/mot_config.json. Defaults to ROBOTWIN_MOT_DATASET_ROOT or MOT_DATASET_ROOT.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, help="Load this exact window start instead of sampling one.")
    parser.add_argument("--random-start", action="store_true", help="Randomize the window start when --start-frame is omitted.")
    parser.add_argument("--no-text-emb", action="store_true", help="Use in-memory empty text tensors; validates video/action loading only, not model input readiness.")
    args = parser.parse_args()

    dataset, sample = load_robotwin_sample(
        args.dataset_root,
        sample_index=args.sample_index,
        start_frame=args.start_frame,
        random_start=args.random_start,
        no_text_emb=args.no_text_emb,
    )
    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "num_samples": len(dataset),
        "sample_index": args.sample_index,
        "sample": {
            key: _tensor_summary(value) if torch.is_tensor(value) else str(type(value))
            for key, value in sample.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
