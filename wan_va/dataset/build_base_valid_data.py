"""Build the reusable valid real-data MOT base snapshot."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .mot_dataset import (
    DEFAULT_VIDEO_KEYS,
    VIDEO_KEY_TO_STREAM_ID,
    MOT_TIMESTAMP_POLICY,
    MOT_VIDEO_TIMESTAMP_TOLERANCE_S,
    RAW_ACTION_DIM,
    _read_json,
    _write_json,
    _read_jsonl,
    _write_jsonl,
)


GENROBOT_FAMILY_DIRS = ("genrobot_lerobot_v1_v2", "genrobot_lerobot_v3", "genrobot_lerobot_v3_1", "genrobot_lerobot_v3_2")


@dataclass
class BaseBuildResult:
    output_root: Path
    base_valid_episode_manifest_path: Path
    base_config_path: Path
    task_stats_path: Path
    num_tasks: int
    num_episodes: int
    skipped: dict[str, dict[str, int]]


def _source_dataset_for_task(task_root: Path) -> str:
    parts = set(task_root.parts)
    if "lumos_lerobot" in parts:
        return "lumos_lerobot"
    if any(part.startswith("genrobot_lerobot") for part in parts):
        return "genrobot_lerobot"
    raise ValueError(f"Cannot infer real MOT source dataset from {task_root}")


def _task_uid(umi_data_root: Path, task_root: Path) -> str:
    try:
        return str(task_root.resolve().relative_to(umi_data_root.resolve()))
    except ValueError:
        return str(task_root.resolve())


def _invaild_lerobot_reason(task_root: Path) -> str | None:
    for name in ("data", "videos", "meta"):
        if not (task_root / name).is_dir():
            return f"missing_{name}_dir"
    if not (task_root / "meta" / "info.json").is_file():
        return "missing_meta_info"
    return None


def _step1_iter_task_roots_filter_lerobot(umi_data_root: Path) -> tuple[list[Path], list[dict[str, str]]]:
    roots: list[Path] = []
    missing_meta_info_rows: list[dict[str, str]] = []
    lumos_root = umi_data_root / "lumos_lerobot"
    if lumos_root.is_dir():
        for date_root in sorted(path for path in lumos_root.iterdir() if path.is_dir()):
            for task_root in sorted(path for path in date_root.iterdir() if path.is_dir()):
                missing_reason = _invaild_lerobot_reason(task_root)
                if missing_reason is None:
                    roots.append(task_root)
                else:
                    missing_meta_info_rows.append({"task_root": str(task_root), "reason": missing_reason})
    for family_dir in GENROBOT_FAMILY_DIRS:
        family_root = umi_data_root / family_dir
        if not family_root.is_dir():
            continue
        for task_root in sorted(path for path in family_root.iterdir() if path.is_dir()):
            missing_reason = _invaild_lerobot_reason(task_root)
            if missing_reason is None:
                roots.append(task_root)
            else:
                missing_meta_info_rows.append({"task_root": str(task_root), "reason": missing_reason})
    return sorted(set(roots), key=lambda path: str(path)), sorted(missing_meta_info_rows, key=lambda row: row["task_root"])


def _is_dual_arm_info(info: dict[str, Any]) -> bool:
    features = info.get("features", {})
    # 2 个 view 的 rgb 都要存在
    if any(key not in features for key in DEFAULT_VIDEO_KEYS):
        return False
    action_shape = (features.get("action") or {}).get("shape")
    state_shape = (features.get("observation.state") or {}).get("shape")
    # 双臂 raw action shape = 16
    return bool(action_shape) and bool(state_shape) and int(action_shape[0]) == RAW_ACTION_DIM and int(state_shape[0]) == RAW_ACTION_DIM


def _read_episode_records(task_root: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for pq_path in sorted((task_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(pq_path)
        for record in table.to_pylist():
            # 记录每一条 episode 自己的信息，其中包含了 prompt、起止idx 等
            records[int(record["episode_index"])] = record
    return records


def _episode_text(task_root: Path, record: dict[str, Any]) -> str:
    tasks = record.get("tasks") or []
    text = str(tasks[0]).strip() if tasks else ""
    if not text:
        raise ValueError(f"Missing LeRobot episode text: task={task_root} episode={record.get('episode_index')}")
    return text


def _episode_len(record: dict[str, Any]) -> int:
    return int(record["dataset_to_index"]) - int(record["dataset_from_index"])


def _video_path(task_root: Path, info: dict[str, Any], record: dict[str, Any], video_key: str) -> Path:
    return task_root / info["video_path"].format(
        video_key=video_key,
        chunk_index=int(record[f"videos/{video_key}/chunk_index"]),
        file_index=int(record[f"videos/{video_key}/file_index"]),
    )


def _data_path(task_root: Path, info: dict[str, Any], record: dict[str, Any]) -> Path:
    return task_root / info["data_path"].format(
        chunk_index=int(record["data/chunk_index"]),
        file_index=int(record["data/file_index"]),
    )


def _missing_payload_files_for_episode(
    task_root: Path,
    info: dict[str, Any],
    record: dict[str, Any],
) -> list[str]:
    paths = [_data_path(task_root, info, record)]
    paths.extend(_video_path(task_root, info, record, video_key) for video_key in DEFAULT_VIDEO_KEYS)
    return [str(path) for path in paths if not path.is_file()]


def _data_index_timestamp_issue_for_episode(
    task_root: Path,
    info: dict[str, Any],
    record: dict[str, Any],
    data_cache: dict[Path, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any] | None:
    data_path = _data_path(task_root, info, record)
    if data_path not in data_cache:
        schema_names = set(pq.read_schema(data_path).names)
        if "index" not in schema_names:
            return {"reason": "missing_data_index_column", "data_file": str(data_path)}
        if "timestamp" not in schema_names:
            return {"reason": "missing_data_timestamp_column", "data_file": str(data_path)}
        table = pq.read_table(data_path, columns=["index", "timestamp"])
        indices = np.asarray(table["index"].to_pylist(), dtype=np.int64).reshape(-1)
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        if timestamps.ndim > 1:
            timestamps = timestamps.reshape(timestamps.shape[0], -1)[:, 0]
        if timestamps.shape[0] != indices.shape[0]:
            return {
                "reason": "data_timestamp_index_length_mismatch",
                "data_file": str(data_path),
                "timestamp_rows": int(timestamps.shape[0]),
                "index_rows": int(indices.shape[0]),
            }
        # Episode records using one data file are adjacent in normal LeRobot metadata.
        # Keep only that file resident so the offline validation cache is bounded too.
        data_cache.clear()
        data_cache[data_path] = (indices, timestamps)

    # 排除缺失索引、重复索引或索引范围错误等问题
    absolute_start = int(record["dataset_from_index"])
    absolute_end = int(record["dataset_to_index"])
    indices, timestamps = data_cache[data_path]
    positions = np.flatnonzero((indices >= absolute_start) & (indices < absolute_end))
    positions = positions[np.argsort(indices[positions])]
    expected_indices = np.arange(absolute_start, absolute_end, dtype=np.int64)
    if positions.size != expected_indices.size or not np.array_equal(indices[positions], expected_indices):
        return {
            "reason": "data_index_coverage_mismatch",
            "data_file": str(data_path),
            "expected_index_start": absolute_start,
            "expected_index_end": absolute_end,
            "expected_rows": int(expected_indices.size),
            "found_rows": int(positions.size),
        }
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        return {"reason": "invalid_fps", "data_file": str(data_path), "fps": fps}
    episode_timestamps = timestamps[positions]
    nonfinite = np.flatnonzero(~np.isfinite(episode_timestamps))
    if nonfinite.size:
        local_frame = int(nonfinite[0])
        return {
            "reason": "data_timestamp_nonfinite",
            "data_file": str(data_path),
            "local_frame": local_frame,
            "parquet_timestamp": float(episode_timestamps[local_frame]),
        }
    expected_timestamps = np.arange(expected_indices.size, dtype=np.float64) / fps
    deltas = np.abs(episode_timestamps - expected_timestamps)
    bad = np.flatnonzero(deltas >= MOT_VIDEO_TIMESTAMP_TOLERANCE_S)
    if bad.size:
        local_frame = int(bad[0])
        return {
            "reason": "data_timestamp_mismatch",
            "data_file": str(data_path),
            "local_frame": local_frame,
            "parquet_timestamp": float(episode_timestamps[local_frame]),
            "expected_timestamp": float(expected_timestamps[local_frame]),
            "delta": float(deltas[local_frame]),
            "tolerance_s": MOT_VIDEO_TIMESTAMP_TOLERANCE_S,
        }
    return None


def build_real_mot_base_cache(
    *,
    umi_data_root: str | Path,
    output_root: str | Path,
) -> BaseBuildResult:
    # 转化为绝对路径
    umi_data_root = Path(umi_data_root).resolve()
    output_root = Path(output_root).resolve()

    output_root.mkdir(parents=True, exist_ok=True)

    # 记录最终 vaild 的数据
    rows: list[dict[str, Any]] = []
    # 记录 build 过程中失败的 task/episode，保存统计信息和详细的失败原因
    task_skipped: dict[str, int] = {}
    task_skipped_rows: list[dict[str, Any]] = []
    episode_skipped: dict[str, int] = {}
    episode_skipped_rows: list[dict[str, Any]] = []
    data_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    # 1. 初步筛选出完整的 lerobot 格式的 tasks
    step1_vaild_task_roots, step1_iter_roots_invaild_lerobot_rows = _step1_iter_task_roots_filter_lerobot(umi_data_root)
    step1_check_lerobot_skipped: dict[str, int] = {}
    for skipped_row in step1_iter_roots_invaild_lerobot_rows:
        reason = str(skipped_row["reason"])
        step1_check_lerobot_skipped[reason] = step1_check_lerobot_skipped.get(reason, 0) + 1

    # 逐 task 进行
    for task_root in step1_vaild_task_roots:
        reason = ""
        # task 层面上筛掉非双臂/非3.0的数据
        try:
            info = _read_json(task_root / "meta" / "info.json")
        except Exception as exc:
            reason = f"bad_info:{type(exc).__name__}"
        if not reason and info.get("codebase_version") != "v3.0":
            reason = "not_lerobot_v3"
        if not reason and not _is_dual_arm_info(info):
            reason = "not_dual_arm"
        if not reason:
            try:
                # 2. 读取 lerobot meta episode，因为这里是逐 task 遍历，这里拿到的 episode idx 就是这个 task 内部的
                step2_vaild_episode_records = _read_episode_records(task_root)
            except Exception as exc:
                reason = f"bad_episode_metadata:{type(exc).__name__}"
        # 写入 task 层面失败原因
        if reason:
            task_skipped[reason] = task_skipped.get(reason, 0) + 1
            task_skipped_rows.append({"task_root": str(task_root), "reason": reason})
            continue
        if not step2_vaild_episode_records:
            task_skipped["empty_episode_metadata"] = task_skipped.get("empty_episode_metadata", 0) + 1
            task_skipped_rows.append({"task_root": str(task_root), "reason": "empty_episode_metadata"})
            continue

        # 标志 lumos / genrobot，用于训练时 sample
        source_dataset = _source_dataset_for_task(task_root)
        # 任务标识，用相对于 root 的路径来代表，用于后续 norm 的计算
        task_uid = _task_uid(umi_data_root, task_root)
        
        # episode 层面上的筛选
        for episode_index, record in sorted(step2_vaild_episode_records.items()):
            text = _episode_text(task_root, record)
            length = _episode_len(record)
            # 筛掉长度为 0 的 episode
            if length <= 0:
                episode_skipped["empty_episode"] = episode_skipped.get("empty_episode", 0) + 1
                episode_skipped_rows.append({"task_root": str(task_root), "episode_index": episode_index, "reason": "empty_episode"})
                continue
            # 测试对应的 data/video 文件是否正确可读
            missing_files = _missing_payload_files_for_episode(task_root, info, record)
            if missing_files:
                episode_skipped["missing_payload_file"] = episode_skipped.get("missing_payload_file", 0) + 1
                episode_skipped_rows.append(
                    {
                        "task_root": str(task_root),
                        "episode_index": episode_index,
                        "reason": "missing_payload_file",
                        "missing_files": missing_files,
                    }
                )
                continue
            # 验证每个 episode 对应的 Parquet 索引及 timestamp，训练时不再重复读取整列。
            data_issue = _data_index_timestamp_issue_for_episode(task_root, info, record, data_cache)
            if data_issue is not None:
                reason = str(data_issue["reason"])
                episode_skipped[reason] = episode_skipped.get(reason, 0) + 1
                episode_skipped_rows.append(
                    {
                        "task_root": str(task_root),
                        "episode_index": episode_index,
                        **data_issue,
                    }
                )
                continue
            # Base 只筛选 lerobot，因此这里只写入基本的 video 相关信息
            views = []
            for video_key in DEFAULT_VIDEO_KEYS:
                views.append(
                    {
                        "stream_id": VIDEO_KEY_TO_STREAM_ID[video_key],
                        "video_key": video_key,
                        "video_path": str(_video_path(task_root, info, record, video_key)),
                        "video_from_timestamp": float(record[f"videos/{video_key}/from_timestamp"]),
                    }
                )
            # 最终 vaild episode 的信息
            rows.append(
                {
                    "task_uid": task_uid,
                    "norm_stats_key": task_uid,
                    "source_dataset": source_dataset,
                    "episode_index": int(episode_index),        # 这里的 episode idx 是该 task 内部的
                    "fps": int(info["fps"]),
                    "timestamp_policy": MOT_TIMESTAMP_POLICY,
                    "source_lerobot_task_dir": str(task_root),
                    "data_file": str(_data_path(task_root, info, record)),
                    "dataset_from_index": int(record["dataset_from_index"]),
                    "dataset_to_index": int(record["dataset_to_index"]),
                    "segment": {
                        "start_frame": 0,
                        "end_frame": length,        # episode 内部 end_frame，这里是左闭右开，因此实际上取到的是 length - 1
                        "tasks": [text],
                        "action_text": text,
                    },
                    "views": views,
                }
            )

    if not rows:
        raise ValueError(f"No valid real dual-arm LeRobot episodes found under {umi_data_root}")

    base_valid_episode_manifest_path = output_root / "meta" / "base_valid_episode_manifest.jsonl"
    task_stats_path = output_root / "meta" / "task_norm_stats.json"
    base_config_path = output_root / "meta" / "base_config.json"
    _write_jsonl(base_valid_episode_manifest_path, rows)
    _write_json(task_stats_path, {})
    _write_jsonl(output_root / "reports" / "task_skipped.jsonl", task_skipped_rows)
    _write_jsonl(output_root / "reports" / "episode_skipped.jsonl", episode_skipped_rows)
    _write_jsonl(output_root / "reports" / "step1_iter_roots_invaild_lerobot.jsonl", step1_iter_roots_invaild_lerobot_rows)
    _write_json(
        output_root / "reports" / "text_check.json",
        {
            "policy": "LeRobot meta/episodes.tasks[0] must be present and non-empty; build fails otherwise.",
            "num_episodes": len(rows),
            "num_unique_texts": len({row["segment"]["action_text"] for row in rows}),
        },
    )
    base_config = {
        "format": "umi_mot_real_base",
        "umi_data_root": str(umi_data_root),
        "base_valid_episode_manifest_path": str(base_valid_episode_manifest_path),
        "task_stats_path": str(task_stats_path),
        "num_episodes": len(rows),
        "num_tasks": len({row["task_uid"] for row in rows}),
        "video_keys": list(DEFAULT_VIDEO_KEYS),
        "timestamp_policy": MOT_TIMESTAMP_POLICY,
        "skipped": {
            "step1_invaild_lerobot": step1_check_lerobot_skipped,
            "task": task_skipped,
            "episode": episode_skipped,
        },
    }
    _write_json(base_config_path, base_config)

    return BaseBuildResult(
        output_root=output_root,
        base_valid_episode_manifest_path=base_valid_episode_manifest_path,
        base_config_path=base_config_path,
        task_stats_path=task_stats_path,
        num_tasks=len({row["task_uid"] for row in rows}),
        num_episodes=len(rows),
        skipped={
            "step1_invaild_lerobot": step1_check_lerobot_skipped,
            "task": task_skipped,
            "episode": episode_skipped,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real UMI MOT base valid-data snapshot.")
    parser.add_argument("--umi-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_real_mot_base_cache(
        umi_data_root=args.umi_data_root,
        output_root=args.output_root,
    )
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.__dict__.items()}, indent=2))


if __name__ == "__main__":
    main()
