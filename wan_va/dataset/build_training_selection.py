"""Select a frozen set of real-data Video+Action MOT episodes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mot_dataset import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    validate_dataset_index_bounds,
)


SELECTION_CONFIG_NAME = "training_selection_config.json"
SELECTED_MANIFEST_NAME = "selected_episode_manifest.jsonl"


@dataclass
class RealTrainSelectionBuildResult:
    output_root: Path
    selection_config_path: Path
    manifest_path: Path
    num_samples: int


def _read_base_rows(base_root: Path) -> tuple[list[dict[str, Any]], Path]:
    base_config = _read_json(base_root / "meta" / "base_config.json")
    manifest_path = Path(base_config["base_valid_episode_manifest_path"])
    rows = _read_jsonl(manifest_path)
    validate_dataset_index_bounds(rows, context=str(manifest_path))
    return rows, manifest_path


def _limit_rows_round_robin_by_task(
    rows: list[dict[str, Any]],
    max_samples: int | None,
) -> list[dict[str, Any]]:
    if max_samples is None:
        return rows
    target = max(0, int(max_samples))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task_uid"]), []).append(row)
    selected: list[dict[str, Any]] = []
    cursors = {key: 0 for key in grouped}
    while len(selected) < target:
        progressed = False
        for key in sorted(grouped):
            cursor = cursors[key]
            if cursor >= len(grouped[key]):
                continue
            selected.append(grouped[key][cursor])
            cursors[key] += 1
            progressed = True
            if len(selected) == target:
                break
        if not progressed:
            break
    return selected


def build_real_mot_training_selection(
    *,
    base_root: str | Path,
    output_root: str | Path,
    sample_from_existing_selection_path: str | Path | None = None,
    max_samples: int | None = None,
) -> RealTrainSelectionBuildResult:
    base_root = Path(base_root).resolve()
    output_root = Path(output_root).resolve()
    meta_root = output_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)

    if sample_from_existing_selection_path is None:
        rows, source_manifest = _read_base_rows(base_root)
        mode = "build_from_base"
    else:
        source_root = Path(sample_from_existing_selection_path).resolve()
        source_config = _read_json(source_root / "meta" / SELECTION_CONFIG_NAME)
        source_manifest = Path(source_config["manifest_path"])
        rows = _read_jsonl(source_manifest)
        mode = "sample_from_existing_selection"
    rows = _limit_rows_round_robin_by_task(rows, max_samples)
    if not rows:
        raise ValueError("No Video+Action training rows were selected")

    manifest_path = meta_root / SELECTED_MANIFEST_NAME
    config_path = meta_root / SELECTION_CONFIG_NAME
    _write_jsonl(manifest_path, rows)
    _write_json(
        config_path,
        {
            "format": "umi_mot_real_training_selection_va_v1",
            "selection_mode": mode,
            "base_root": str(base_root),
            "base_valid_episode_manifest_path": str(source_manifest),
            "manifest_path": str(manifest_path),
            "max_samples": None if max_samples is None else int(max_samples),
            "num_samples": len(rows),
        },
    )
    return RealTrainSelectionBuildResult(
        output_root=output_root,
        selection_config_path=config_path,
        manifest_path=manifest_path,
        num_samples=len(rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select real Video+Action MOT episodes.")
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-from-existing-selection-path", type=Path)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    result = build_real_mot_training_selection(
        base_root=args.base_root,
        output_root=args.output_root,
        sample_from_existing_selection_path=args.sample_from_existing_selection_path,
        max_samples=args.max_samples,
    )
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.__dict__.items()}, indent=2))


if __name__ == "__main__":
    main()
