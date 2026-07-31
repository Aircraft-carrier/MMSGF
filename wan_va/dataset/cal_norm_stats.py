#!/usr/bin/env python3
"""Compute per-task MOT real train norm stats before building train metadata."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .build_training_dataset import (
    _norm_stats_path_for_task,
    load_required_task_norm_stats,
    load_real_mot_training_selection,
)
from .mot_dataset import (
    MOT_DEFAULT_ACTION_CHUNK_SIZE,
    MOT_MAX_RIGHT_PADDING_RAW_STEPS,
    RAW_ACTION_DIM,
    RELATIVE_ACTION_DIM,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    absolute_actions_to_relative_20d,
)


_NORM_STAT_KEYS = ("q01", "q99", "mean", "std")
_MIN_NORM_STATS_EPISODE_FRAMES = 30
_NORM_STATS_SKIPPED_REPORT_NAME = "norm_stats_skipped.json"
_NORM_STATS_SKIPPED_JSONL_NAME = "norm_stats_skipped.jsonl"


def _slice_episode_arrays_for_row(
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    absolute_start = int(row["dataset_from_index"]) + int(row["segment"]["start_frame"])
    absolute_end = int(row["dataset_from_index"]) + int(row["segment"]["end_frame"])
    if any(value.shape[0] != indices.shape[0] for value in arrays.values()):
        lengths = {key: int(value.shape[0]) for key, value in arrays.items()}
        raise ValueError(f"LeRobot norm-stats action/index length mismatch in {row['data_file']}: arrays={lengths} index={indices.shape[0]}")
    expected_len = absolute_end - absolute_start
    positions = np.flatnonzero((indices >= absolute_start) & (indices < absolute_end))
    positions = positions[np.argsort(indices[positions])]
    expected_indices = np.arange(absolute_start, absolute_end, dtype=np.int64)
    if positions.size != expected_len or not np.array_equal(indices[positions], expected_indices):
        raise ValueError(
            "LeRobot norm-stats index coverage mismatch: "
            f"data_file={row['data_file']} expected absolute index range [{absolute_start}, {absolute_end}) "
            f"with {expected_len} rows, found {int(positions.size)}"
        )
    return {key: value[positions] for key, value in arrays.items()}


def _relative_target_values_for_arrays_vectorized(
    row: dict[str, Any],
    actions: np.ndarray,
    states: np.ndarray,
    action_chunk_size: int,
) -> tuple[np.ndarray, int]:
    target_actions = int(action_chunk_size)
    # 记录所有 relative actions
    rel_values = []
    chunks = 0
    valid_range = row.get("valid_start_range", [])
    if not valid_range:
        return np.zeros((0, RELATIVE_ACTION_DIM), dtype=np.float32), chunks
    start, end = valid_range
    # 当前 state/actions 已经被切片，所以要转换为从 0 开始的下标
    starts = np.arange(int(start), int(end) + 1, dtype=np.int64) - int(row["segment"]["start_frame"])
    starts = starts[(starts >= 0) & (starts < actions.shape[0])]
    if starts.size == 0:
        raise ValueError(
            "Selected norm-stats row has no start indices after episode slicing: "
            f"task_uid={row.get('task_uid')}, episode_index={row.get('episode_index')}, "
            f"segment={row.get('segment')}, valid_start_range={[int(start), int(end)]}, "
            f"num_actions={actions.shape[0]}"
        )
    # 构造二维 action chunk indices
    offsets = np.arange(target_actions, dtype=np.int64)
    action_indices = starts[:, None] + offsets[None, :]

    valid = action_indices < actions.shape[0]
    if not bool(valid.any()):
        raise ValueError(
            "Selected norm-stats row has no valid action indices: "
            f"task_uid={row.get('task_uid')}, episode_index={row.get('episode_index')}, "
            f"segment={row.get('segment')}, valid_start_range={[int(start), int(end)]}, "
            f"action_chunk_size={int(action_chunk_size)}, num_actions={actions.shape[0]}"
        )

    # 对于每个 action chunk，所有 target action 都使用 chunk起点的 state 作为 reference state
    ref = np.broadcast_to(states[starts][:, None, :], (starts.shape[0], target_actions, RAW_ACTION_DIM))
    padded_indices = np.minimum(action_indices, actions.shape[0] - 1)
    # 计算 relative actions
    rel = absolute_actions_to_relative_20d(ref, actions[padded_indices])
    # valid 筛选掉用于 padding 的 action
    rel_values.append(rel[valid])
    chunks += int(starts.shape[0])
    if not rel_values:
        return np.zeros((0, RELATIVE_ACTION_DIM), dtype=np.float32), chunks
    return np.concatenate(rel_values, axis=0), chunks


def _valid_existing_stats(path: Path, *, action_chunk_size: int) -> dict[str, list[float]] | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    if int(payload.get("action_chunk_size", -1)) != int(action_chunk_size):
        return None
    stats: dict[str, list[float]] = {}
    for key in _NORM_STAT_KEYS:
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != RELATIVE_ACTION_DIM:
            return None
        stats[key] = [float(value) for value in values]
    return stats


def _compute_one_task(
    args: tuple[str, list[dict[str, Any]], int],
) -> tuple[str, dict[str, list[float]] | None, dict[str, Any], list[dict[str, Any]]]:
    task_uid, task_rows, action_chunk_size = args
    stats, payload, skipped_rows = compute_task_norm_stats_payload(
        task_uid=task_uid,
        task_rows=task_rows,
        action_chunk_size=action_chunk_size,
    )
    return task_uid, stats, payload, skipped_rows


def _min_required_episode_frames_for_stats(action_chunk_size: int) -> int:
    max_start_required_frames = int(action_chunk_size) + 1 - MOT_MAX_RIGHT_PADDING_RAW_STEPS
    return max(_MIN_NORM_STATS_EPISODE_FRAMES, max_start_required_frames)


def _norm_stats_skipped_row(
    row: dict[str, Any],
    *,
    task_uid: str,
    action_chunk_size: int,
    reason: str,
) -> dict[str, Any]:
    segment = row["segment"]
    segment_start = int(segment["start_frame"])
    segment_end = int(segment["end_frame"])
    return {
        "reason": reason,
        "task_uid": task_uid,
        "episode_index": int(row["episode_index"]),
        "source_lerobot_task_dir": row.get("source_lerobot_task_dir"),
        "data_file": row.get("data_file"),
        "segment": segment,
        "num_frames": int(segment_end - segment_start),
        "min_required_frames": _min_required_episode_frames_for_stats(action_chunk_size),
        "action_chunk_size": int(action_chunk_size),
        "max_right_padding_raw_steps": MOT_MAX_RIGHT_PADDING_RAW_STEPS,
    }


def compute_task_norm_stats_payload(
    *,
    task_uid: str,
    task_rows: list[dict[str, Any]],
    action_chunk_size: int,
) -> tuple[dict[str, list[float]] | None, dict[str, Any], list[dict[str, Any]]]:
    # 记录得到的 relative actions
    values: list[np.ndarray] = []
    # 实际参与统计的episode 数
    episode_indices: list[int] = []
    # 有效 start frame数
    num_chunks = 0
    arrays_by_file: dict[Path, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    skipped_rows: list[dict[str, Any]] = []
    for base_row in sorted(task_rows, key=lambda item: int(item["episode_index"])):
        # NOTE max start frame，和训练对齐
        valid_range = _full_valid_start_range_for_stats(base_row, action_chunk_size)
        if not valid_range:
            skipped_rows.append(
                _norm_stats_skipped_row(
                    base_row,
                    task_uid=task_uid,
                    action_chunk_size=action_chunk_size,
                    reason="episode_too_short_for_norm_stats",
                )
            )
            continue
        data_file = Path(base_row["data_file"])
        cached = arrays_by_file.get(data_file)
        if cached is None:
            schema_names = set(pq.read_schema(data_file).names)
            if "index" not in schema_names:
                raise ValueError(f"LeRobot norm-stats loading requires an index column: {data_file}")
            table = pq.read_table(data_file, columns=["action", "observation.state", "index"])
            arrays = {
                "action": np.asarray(table["action"].to_pylist(), dtype=np.float32),
                "observation.state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
            }
            indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
            cached = (arrays, indices)
            arrays_by_file[data_file] = cached
        # 从原始混合的 parquet 中切出当前 episode 对应的那一部分
        arrays = _slice_episode_arrays_for_row(base_row, cached[0], cached[1])
        stats_row = json.loads(json.dumps(base_row))
        stats_row["valid_start_range"] = valid_range
        # NOTE 核心：计算 relative action chunk，遍历所有有效 start frame
        row_values, row_chunks = _relative_target_values_for_arrays_vectorized(
            stats_row,
            arrays["action"],
            arrays["observation.state"],
            action_chunk_size,
        )
        if row_values.size:
            values.append(row_values)
            episode_indices.append(int(base_row["episode_index"]))
            num_chunks += int(row_chunks)
    if not values:
        task_root = Path(task_rows[0]["source_lerobot_task_dir"])
        return None, {
            "task_uid": task_uid,
            "action_chunk_size": int(action_chunk_size),
            "action_dim": RELATIVE_ACTION_DIM,
            "source_lerobot_task_dir": str(task_root),
            "episode_indices": [],
            "num_episodes": 0,
            "num_chunks": 0,
            "num_values": 0,
            "skipped": True,
            "skip_reason": "no_valid_rows_for_norm_stats",
        }, skipped_rows
    arr = np.concatenate(values, axis=0)
    stats = {
        "q01": [float(value) for value in np.quantile(arr, 0.01, axis=0)],
        "q99": [float(value) for value in np.quantile(arr, 0.99, axis=0)],
        "mean": [float(value) for value in np.mean(arr, axis=0)],
        "std": [float(value) for value in np.std(arr, axis=0)],
    }
    task_root = Path(task_rows[0]["source_lerobot_task_dir"])
    return stats, {
        **stats,
        "task_uid": task_uid,
        "action_chunk_size": int(action_chunk_size),
        "action_dim": RELATIVE_ACTION_DIM,
        "action_representation": "relative_to_chunk_reference_state_rot6d_cols",
        "source_lerobot_task_dir": str(task_root),
        "episode_indices": episode_indices,
        "num_episodes": len(episode_indices),
        "num_chunks": int(num_chunks),
        "num_values": int(arr.shape[0]),
    }, skipped_rows


def _full_valid_start_range_for_stats(row: dict[str, Any], action_chunk_size: int) -> list[int]:
    num_frames = int(row["segment"]["end_frame"]) - int(row["segment"]["start_frame"])
    if num_frames < _min_required_episode_frames_for_stats(action_chunk_size):
        return []
    max_current = (
        int(row["segment"]["end_frame"])
        + MOT_MAX_RIGHT_PADDING_RAW_STEPS
        - int(action_chunk_size)
        - 1
    )
    min_current = int(row["segment"]["start_frame"])
    if max_current < min_current:
        return []
    return [min_current, max_current]


def cal_real_mot_norm_stats(
    *,
    selection_root: str | Path,
    action_chunk_size: int = MOT_DEFAULT_ACTION_CHUNK_SIZE,
    dataset_root: str | Path | None = None,
    num_workers: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    # 读取 manifest 里的 rows，根据训练要用的数据算 stats
    loaded_selection = load_real_mot_training_selection(selection_root)
    selection_config = loaded_selection.config
    action_chunk_size = int(action_chunk_size)
    selected_rows = loaded_selection.pointcloud_rows + loaded_selection.non_pointcloud_rows
    selected_task_uids = {str(row["task_uid"]) for row in selected_rows}
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        task_uid = str(row["task_uid"])
        rows_by_task.setdefault(task_uid, []).append(row)

    task_stats: dict[str, dict[str, list[float]]] = {}
    pending: list[tuple[str, list[dict[str, Any]], int]] = []
    existing_task_count = 0
    for task_uid, task_rows in sorted(rows_by_task.items()):
        task_root = Path(task_rows[0]["source_lerobot_task_dir"])
        existing = None
        if not overwrite:
            existing = _valid_existing_stats(
                _norm_stats_path_for_task(task_root, action_chunk_size),
                action_chunk_size=action_chunk_size,
        )
        if existing is not None:
            task_stats[task_uid] = existing
            existing_task_count += 1
            continue
        pending.append((task_uid, task_rows, int(action_chunk_size)))

    print(
        f"Selected {len(selected_task_uids)} tasks; using {len(task_stats)} existing stats; computing {len(pending)} missing stats.",
        flush=True,
    )
    skipped_rows: list[dict[str, Any]] = []
    skipped_task_uids: set[str] = set()
    computed_task_count = 0
    if int(num_workers) > 1 and pending:
        with futures.ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            future_to_task = {executor.submit(_compute_one_task, task): task[0] for task in pending}
            for done_count, future in enumerate(futures.as_completed(future_to_task), start=1):
                task_uid, stats, payload, task_skipped_rows = future.result()
                skipped_rows.extend(task_skipped_rows)
                if stats is None:
                    skipped_task_uids.add(task_uid)
                    print(f"[{done_count}/{len(pending)}] skipped norm stats: {task_uid}", flush=True)
                    continue
                task_stats[task_uid] = stats
                _write_json(_norm_stats_path_for_task(Path(payload["source_lerobot_task_dir"]), action_chunk_size), payload)
                computed_task_count += 1
                print(f"[{done_count}/{len(pending)}] wrote norm stats: {task_uid}", flush=True)
    else:
        for done_count, task in enumerate(pending, start=1):
            task_uid, stats, payload, task_skipped_rows = _compute_one_task(task)
            skipped_rows.extend(task_skipped_rows)
            if stats is None:
                skipped_task_uids.add(task_uid)
                print(f"[{done_count}/{len(pending)}] skipped norm stats: {task_uid}", flush=True)
                continue
            task_stats[task_uid] = stats
            _write_json(_norm_stats_path_for_task(Path(payload["source_lerobot_task_dir"]), action_chunk_size), payload)
            computed_task_count += 1
            print(f"[{done_count}/{len(pending)}] wrote norm stats: {task_uid}", flush=True)

    missing = sorted(selected_task_uids - set(task_stats))
    unexpected_missing = sorted(set(missing) - skipped_task_uids)
    if unexpected_missing:
        raise ValueError("No relative action values found for selected task norm stats: " + ", ".join(missing[:10]))

    skipped_counts: dict[str, int] = {}
    for row in skipped_rows:
        reason = str(row["reason"])
        skipped_counts[reason] = skipped_counts.get(reason, 0) + 1
    reports_dir = loaded_selection.selection_root / "reports"
    _write_jsonl(reports_dir / _NORM_STATS_SKIPPED_JSONL_NAME, skipped_rows)
    _write_json(
        reports_dir / _NORM_STATS_SKIPPED_REPORT_NAME,
        {
            "skipped": skipped_counts,
            "num_skipped_rows": len(skipped_rows),
            "num_skipped_tasks": len(skipped_task_uids),
            "skipped_task_uids": sorted(skipped_task_uids),
            "min_required_frames": _min_required_episode_frames_for_stats(action_chunk_size),
            "action_chunk_size": int(action_chunk_size),
            "max_right_padding_raw_steps": MOT_MAX_RIGHT_PADDING_RAW_STEPS,
        },
    )

    dataset_root_path = Path(dataset_root).resolve() if dataset_root is not None else None
    if dataset_root_path is not None:
        mot_config_path = dataset_root_path / "meta" / "mot_config.json"
        if mot_config_path.is_file():
            mot_config = _read_json(mot_config_path)
            selected_config_rows = _rows_from_mot_config(dataset_root_path, mot_config)
            task_stats = load_required_task_norm_stats(
                selected_rows=selected_config_rows,
                action_chunk_size=int(mot_config.get("action_chunk_size", action_chunk_size)),
            )
            default_norm_key = str(selected_config_rows[0]["norm_stats_key"])
            mot_config["norm_stat"] = task_stats[default_norm_key]
            mot_config["norm_stats_by_task"] = task_stats
            mot_config["norm_stat_policy"] = f"by_task_uid_selected_episodes_relative20_chunk{int(action_chunk_size)}"
            _write_json(mot_config_path, mot_config)

    return {
        "selection_root": loaded_selection.selection_root,
        "base_root": Path(selection_config["base_root"]).resolve(),
        "num_pointcloud_samples": len(loaded_selection.pointcloud_rows),
        "num_non_pointcloud_samples": len(loaded_selection.non_pointcloud_rows),
        "num_tasks": len(task_stats),
        "num_existing_tasks": existing_task_count,
        "num_computed_tasks": computed_task_count,
        "num_skipped_tasks": len(skipped_task_uids),
        "num_skipped_rows": len(skipped_rows),
        "skipped_report_path": reports_dir / _NORM_STATS_SKIPPED_REPORT_NAME,
        "skipped_report_jsonl_path": reports_dir / _NORM_STATS_SKIPPED_JSONL_NAME,
        "task_uids": sorted(task_stats),
    }


def _resolve_dataset_path(dataset_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return dataset_root / path


def _rows_from_mot_config(dataset_root: Path, mot_config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("mot_manifest_path", "non_pointcloud_manifest_path"):
        manifest_path = _resolve_dataset_path(dataset_root, mot_config.get(key))
        if manifest_path is not None and manifest_path.is_file():
            rows.extend(_read_jsonl(manifest_path))
    if not rows:
        raise ValueError(f"No train rows found from {dataset_root / 'meta' / 'mot_config.json'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-task MOT real train norm stats.")
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional existing train root whose mot_config should be updated.")
    parser.add_argument("--action-chunk-size", type=int, default=MOT_DEFAULT_ACTION_CHUNK_SIZE)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing task norm stats.")
    args = parser.parse_args()

    result = cal_real_mot_norm_stats(
        selection_root=args.selection_root,
        action_chunk_size=args.action_chunk_size,
        dataset_root=args.dataset_root,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
