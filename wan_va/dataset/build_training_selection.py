"""Select the frozen real-data MOT training episodes before preprocessing."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .build_base_valid_data import _source_dataset_for_task
from .mot_dataset import (
    HAND_TO_VIDEO_KEY,
    HAND_TO_STREAM_ID,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    validate_dataset_index_bounds,
)


SELECTION_CONFIG_NAME = "training_selection_config.json"
SELECTED_POINTCLOUD_MANIFEST_NAME = "selected_pointcloud_episode_manifest.jsonl"
SELECTED_NON_POINTCLOUD_MANIFEST_NAME = "selected_non_pointcloud_episode_manifest.jsonl"
LUMOS_LEGACY_MAPPING_REPORT_NAME = "selection_lumos_legacy_mapping.jsonl"


@dataclass
class RealTrainSelectionBuildResult:
    output_root: Path
    selection_config_path: Path
    pointcloud_manifest_path: Path
    non_pointcloud_manifest_path: Path
    num_pointcloud_samples: int
    num_non_pointcloud_samples: int


@dataclass
class RealTrainSelection:
    pointcloud_rows: list[dict[str, Any]]
    non_pointcloud_rows: list[dict[str, Any]]
    pointcloud_skipped: dict[str, int]
    pointcloud_skipped_rows: list[dict[str, Any]]
    lumos_legacy_mapping_rows: list[dict[str, Any]]
    blocked_pointcloud_episode_keys: set[tuple[str, int]]


def _read_base_rows_for_selection(base_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    base_config = _read_json(base_root / "meta" / "base_config.json")
    base_valid_episode_manifest_path = Path(base_config["base_valid_episode_manifest_path"])
    base_rows = _read_jsonl(base_valid_episode_manifest_path)
    validate_dataset_index_bounds(base_rows, context=str(base_valid_episode_manifest_path))
    return base_config, base_rows, base_valid_episode_manifest_path


def _limit_rows_round_robin_by_task(rows: list[dict[str, Any]], max_rows: int | None) -> list[dict[str, Any]]:
    if max_rows is None:
        return rows
    target = max(0, int(max_rows))
    if target <= 0 or not rows:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task_uid"]), []).append(row)
    task_keys = sorted(grouped)
    selected: list[dict[str, Any]] = []
    cursor = {key: 0 for key in task_keys}
    while len(selected) < target:
        progressed = False
        for key in task_keys:
            idx = cursor[key]
            if idx >= len(grouped[key]):
                continue
            selected.append(grouped[key][idx])
            cursor[key] += 1
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    return selected


def _limit_rows_balanced_by_source_then_task(rows: list[dict[str, Any]], max_rows: int | None) -> list[dict[str, Any]]:
    if max_rows is None:
        return rows
    target = max(0, int(max_rows))
    if target <= 0 or not rows:
        return []

    source_order = ("lumos_lerobot", "genrobot_lerobot")
    by_source = {source: [row for row in rows if row["source_dataset"] == source] for source in source_order}
    quotas = {source: target // len(source_order) for source in source_order}
    remaining = target - sum(quotas.values())
    for source in sorted(source_order, key=lambda item: (-len(by_source[item]), item)):
        if remaining <= 0:
            break
        if len(by_source[source]) > quotas[source]:
            quotas[source] += 1
            remaining -= 1

    selected_counts = {source: min(quotas[source], len(by_source[source])) for source in source_order}
    remaining = target - sum(selected_counts.values())
    while remaining > 0:
        progressed = False
        for source in sorted(source_order, key=lambda item: (selected_counts[item] >= len(by_source[item]), item)):
            if selected_counts[source] >= len(by_source[source]):
                continue
            selected_counts[source] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break

    selected: list[dict[str, Any]] = []
    for source in source_order:
        selected.extend(_limit_rows_round_robin_by_task(by_source[source], selected_counts[source]))
    return selected


def _source_sample_counts(pointcloud_rows: list[dict[str, Any]], non_pointcloud_rows: list[dict[str, Any]]) -> dict[str, int]:
    lumos_pointcloud = sum(1 for row in pointcloud_rows if row["source_dataset"] == "lumos_lerobot")
    genrobot_pointcloud = sum(1 for row in pointcloud_rows if row["source_dataset"] == "genrobot_lerobot")
    lumos_non_pointcloud = sum(1 for row in non_pointcloud_rows if row["source_dataset"] == "lumos_lerobot")
    genrobot_non_pointcloud = sum(1 for row in non_pointcloud_rows if row["source_dataset"] == "genrobot_lerobot")
    return {
        "lumos_pointcloud": lumos_pointcloud,
        "genrobot_pointcloud": genrobot_pointcloud,
        "lumos_non_pointcloud": lumos_non_pointcloud,
        "genrobot_non_pointcloud": genrobot_non_pointcloud,
        "total_pointcloud": lumos_pointcloud + genrobot_pointcloud,
        "total_non_pointcloud": lumos_non_pointcloud + genrobot_non_pointcloud,
        "total_train_data_num": lumos_pointcloud + genrobot_pointcloud + lumos_non_pointcloud + genrobot_non_pointcloud,
    }


def _hand_dirs(session_root: Path) -> dict[str, Path] | None:
    left = sorted(path for path in session_root.iterdir() if path.is_dir() and path.name.startswith("left_hand"))
    right = sorted(path for path in session_root.iterdir() if path.is_dir() and path.name.startswith("right_hand"))
    if not left or not right:
        return None
    return {"left": left[0], "right": right[0]}


def _iter_pointcloud_session_roots(task_root: Path) -> list[Path]:
    pointcloud_root = task_root / "pointcloud"
    if not pointcloud_root.is_dir():
        return []
    if _source_dataset_for_task(task_root) == "lumos_lerobot":
        sessions = [
            session_root
            for multi_root in sorted(path for path in pointcloud_root.iterdir() if path.is_dir())
            for session_root in sorted(path for path in multi_root.iterdir() if path.is_dir() and path.name.startswith("session_"))
        ]
    else:
        sessions = [
            session_root
            for group_root in sorted(path for path in pointcloud_root.iterdir() if path.is_dir())
            for session_root in sorted(path for path in group_root.iterdir() if path.is_dir())
        ]
    return sorted(sessions, key=lambda path: str(path))


def _load_lumos_session_mapping(task_root: Path, legacy_rows: list[dict[str, Any]] | None = None) -> dict[str, int]:
    path = task_root / "raw_lerobot_idx.jsonl"
    if not path.is_file():
        return {}
    mapping: dict[str, int] = {}
    for row_index, row in enumerate(_read_jsonl(path)):
        if "filename" in row and "lerobot_episode_index" in row:
            mapping[str(row["filename"])] = int(row["lerobot_episode_index"])
            continue
        if "session_path" in row and "episode_index" in row:
            session_path = str(row["session_path"])
            episode_index = int(row["episode_index"])
            mapping[session_path] = episode_index
            if legacy_rows is not None:
                legacy_rows.append(
                    {
                        "task_root": str(task_root),
                        "mapping_path": str(path),
                        "row_index": int(row_index),
                        "schema": "legacy_session_path_episode_index",
                        "session_path": session_path,
                        "episode_index": episode_index,
                    }
                )
    return mapping


def _load_genrobot_session_mapping(task_root: Path) -> dict[str, int]:
    path = task_root / "mcap_lerobot_idx_mapping.jsonl"
    mapping: dict[str, int] = {}
    if path.is_file():
        for row in _read_jsonl(path):
            episode_index = int(row["episode_index"])
            mcap = Path(str(row["mcap_file"]))
            mapping[mcap.stem] = episode_index
            mapping[mcap.name] = episode_index
    return mapping


def _episode_index_for_pointcloud_session(
    task_root: Path,
    session_root: Path,
    session_mapping: dict[str, int],
) -> tuple[int | None, str]:
    source_dataset = _source_dataset_for_task(task_root)
    if source_dataset == "lumos_lerobot":
        key = str(session_root.relative_to(task_root / "pointcloud"))
        return session_mapping.get(key), "raw_lerobot_idx"

    if session_root.name in session_mapping:
        return session_mapping[session_root.name], "mcap_lerobot_idx_mapping"
    return None, "missing_genrobot_mapping"


def _build_planned_pointcloud_rows(
    base_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[tuple[str, int]], dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    # task + episode idx 唯一确定一条 base row
    by_task_episode = {(row["task_uid"], int(row["episode_index"])): row for row in base_rows}
    # 遍历所有 vaild 的 task 目录
    task_roots = {row["task_uid"]: Path(row["source_lerobot_task_dir"]) for row in base_rows}

    rows: list[dict[str, Any]] = []
    blocked_episode_keys: set[tuple[str, int]] = set()
    skipped: dict[str, int] = {}
    skipped_rows: list[dict[str, Any]] = []
    lumos_legacy_mapping_rows: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()

    for task_uid, task_root in sorted(task_roots.items()):
        if _source_dataset_for_task(task_root) == "lumos_lerobot":
            session_mapping = _load_lumos_session_mapping(task_root, legacy_rows=lumos_legacy_mapping_rows)
        else:
            session_mapping = _load_genrobot_session_mapping(task_root)
        for session_root in _iter_pointcloud_session_roots(task_root):
            # NOTE:  这里的 session 就是一条带 pointcloud 的 episode
            session_key = str(session_root)
            if session_key in seen_sessions:
                continue
            seen_sessions.add(session_key)
            hand_dirs = _hand_dirs(session_root)
            if hand_dirs is None:
                skipped["missing_hand_dirs"] = skipped.get("missing_hand_dirs", 0) + 1
                skipped_rows.append({"session_root": str(session_root), "reason": "missing_hand_dirs"})
                continue

            # 两步筛选
            # 1. 先找当前带 pointcloud 的 session 能不能找到 lerobot video 中对应的 episode idx
            episode_index, mapping_policy = _episode_index_for_pointcloud_session(
                task_root,
                session_root,
                session_mapping,
            )
            if episode_index is None:
                skipped["missing_mapping"] = skipped.get("missing_mapping", 0) + 1
                skipped_rows.append({"session_root": str(session_root), "reason": "missing_mapping", "mapping_policy": mapping_policy})
                continue
            # 2. 找到映射还不够，还需要确定 base vaild row 中是否存在这条 episode
            base_row = by_task_episode.get((task_uid, int(episode_index)))
            if base_row is None:
                skipped["missing_base_episode"] = skipped.get("missing_base_episode", 0) + 1
                skipped_rows.append({"session_root": str(session_root), "reason": "missing_base_episode", "episode_index": episode_index})
                continue
            # 两步筛选都满足后，说明 pointcloud session 能够成功映射到 base episode，就加入 blocked_episode_keys，表示不能再进入 non-pointcloud pool
            key = (str(base_row["task_uid"]), int(base_row["episode_index"]))
            blocked_episode_keys.add(key)
            # 前面不满足的都会 continue，所以走到这的都是满足的，就可以加入 pointcloud pool
            views = []
            source_dataset = str(base_row["source_dataset"])
            skip_row: dict[str, Any] | None = None
            for hand in ("left", "right"):
                hand_dir = hand_dirs[hand]
                if source_dataset == "lumos_lerobot" and not (hand_dir / "clips").is_dir():
                    skip_row = {"session_root": str(session_root), "reason": "missing_hand_clips", "episode_index": int(episode_index), "detail": str(hand_dir)}
                    break
                if source_dataset != "lumos_lerobot" and not (hand_dir / "preprocessed_pointcloud" / "_SUCCESS").is_file():
                    skip_row = {
                        "session_root": str(session_root),
                        "reason": "missing_preprocessed_pointcloud",
                        "episode_index": int(episode_index),
                        "detail": str(hand_dir / "preprocessed_pointcloud"),
                    }
                    break
                base_view = next(view for view in base_row["views"] if view["video_key"] == HAND_TO_VIDEO_KEY[hand])
                # 上面是对本地文件结构做遍历，即遍历所有 task 下的所有 session，但是通过 task 和 episode_idx 可以定位到 manifest 的唯一行 base_row
                # 走到这，pointcloud 都是合法的，就可以把 pointcloud session 信息更新到 base_row 里了
                # 第一步的 base_row 中只记录了 video 信息 LINK: wan_va/dataset/build_base_valid_data.py:225
                # 这里补充 pointcloud 信息 （通过上述流程，step2 其实完成了把一个 pointcloud session 映射到一个 LeRobot episode）
                view = dict(base_view)
                view.update(
                    {
                        "hand": hand,
                        "hand_dir": str(hand_dir),
                        "stream_id": HAND_TO_STREAM_ID[hand],
                    }
                )
                views.append(view)
            if skip_row is not None:
                reason = str(skip_row["reason"])
                skipped[reason] = skipped.get(reason, 0) + 1
                skipped_rows.append(skip_row)
                continue
            out = json.loads(json.dumps(base_row))
            out.update(
                {
                    "has_pointcloud": True,
                    "source_pointcloud_task_dir": str(task_root),
                    "pointcloud_session_dir": str(session_root),
                    "pointcloud_mapping_policy": mapping_policy,
                    "views": views,
                }
            )
            rows.append(out)
    return rows, blocked_episode_keys, skipped, skipped_rows, lumos_legacy_mapping_rows


def select_real_mot_train_rows(
    *,
    base_rows: list[dict[str, Any]],
) -> RealTrainSelection:
    # 筛选 vaild pointcloud pool
    pointcloud_rows, blocked_pointcloud_episode_keys, pc_skipped, pc_skipped_rows, lumos_legacy_mapping_rows = _build_planned_pointcloud_rows(
        base_rows,
    )
    if not pointcloud_rows:
        raise ValueError("No pointcloud-labeled windows found in the current real data snapshot")

    # 构建 non-pointcloud pool，核心逻辑是排除 pointcloud pool 中的 episode
    non_pointcloud_rows: list[dict[str, Any]] = []
    for base_row in base_rows:
        key = (str(base_row["task_uid"]), int(base_row["episode_index"]))
        if key in blocked_pointcloud_episode_keys:
            continue
        source_dataset = str(base_row["source_dataset"])
        if source_dataset not in {"lumos_lerobot", "genrobot_lerobot"}:
            continue
        row = json.loads(json.dumps(base_row))
        row.update(
            {
                "has_pointcloud": False,
            }
        )
        non_pointcloud_rows.append(row)
    return RealTrainSelection(
        pointcloud_rows=pointcloud_rows,
        non_pointcloud_rows=non_pointcloud_rows,
        pointcloud_skipped=pc_skipped,
        pointcloud_skipped_rows=pc_skipped_rows,
        lumos_legacy_mapping_rows=lumos_legacy_mapping_rows,
        blocked_pointcloud_episode_keys=blocked_pointcloud_episode_keys,
    )


def _sample_existing_selection(
    selection_root: Path,
    *,
    max_pointcloud_samples: int | None,
    max_non_pointcloud_samples: int | None,
) -> tuple[RealTrainSelection, dict[str, Any]]:
    meta_root = selection_root / "meta"
    source_config = _read_json(meta_root / SELECTION_CONFIG_NAME)
    pointcloud_rows = _read_jsonl(meta_root / SELECTED_POINTCLOUD_MANIFEST_NAME)
    non_pointcloud_rows = _read_jsonl(meta_root / SELECTED_NON_POINTCLOUD_MANIFEST_NAME)
    pointcloud_rows = _limit_rows_balanced_by_source_then_task(pointcloud_rows, max_pointcloud_samples)
    non_pointcloud_rows = _limit_rows_balanced_by_source_then_task(non_pointcloud_rows, max_non_pointcloud_samples)
    if not pointcloud_rows:
        raise ValueError("No pointcloud-labeled windows found in the sampled real data selection")

    return RealTrainSelection(
        pointcloud_rows=pointcloud_rows,
        non_pointcloud_rows=non_pointcloud_rows,
        pointcloud_skipped={},
        pointcloud_skipped_rows=[],
        lumos_legacy_mapping_rows=[],
        blocked_pointcloud_episode_keys=set(),
    ), source_config


def build_real_mot_training_selection(
    *,
    base_root: str | Path,
    output_root: str | Path,
    sample_from_existing_selection_path: str | Path | None = None,
    max_pointcloud_samples: int | None = None,
    max_non_pointcloud_samples: int | None = None,
) -> RealTrainSelectionBuildResult:
    base_root = Path(base_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)

    source_selection_path = None if sample_from_existing_selection_path is None else Path(sample_from_existing_selection_path).resolve()
    # 从 base valid episode manifest 构建完整 selection -> 用于全量训练
    if source_selection_path is None:
        if max_pointcloud_samples is not None or max_non_pointcloud_samples is not None:
            raise ValueError("max_pointcloud_samples/max_non_pointcloud_samples require sample_from_existing_selection_path")
        _base_config, base_rows, base_valid_episode_manifest_path = _read_base_rows_for_selection(base_root)
        selection = select_real_mot_train_rows(
            base_rows=base_rows,
        )
        selection_mode = "build_from_base"
    # 从已有 selection 中采样 -> 用于小规模测试
    else:
        selection, source_config = _sample_existing_selection(
            source_selection_path,
            max_pointcloud_samples=max_pointcloud_samples,
            max_non_pointcloud_samples=max_non_pointcloud_samples,
        )
        base_valid_episode_manifest_path = Path(str(source_config.get("base_valid_episode_manifest_path", "")))
        selection_mode = "sample_from_existing_selection"

    pointcloud_manifest_path = output_root / "meta" / SELECTED_POINTCLOUD_MANIFEST_NAME
    non_pointcloud_manifest_path = output_root / "meta" / SELECTED_NON_POINTCLOUD_MANIFEST_NAME
    selection_config_path = output_root / "meta" / SELECTION_CONFIG_NAME
    _write_jsonl(pointcloud_manifest_path, selection.pointcloud_rows)
    _write_jsonl(non_pointcloud_manifest_path, selection.non_pointcloud_rows)
    _write_jsonl(output_root / "reports" / "selection_pointcloud_skipped.jsonl", selection.pointcloud_skipped_rows)
    _write_json(output_root / "reports" / "selection_pointcloud_skipped.json", {"skipped": selection.pointcloud_skipped})
    _write_jsonl(output_root / "reports" / LUMOS_LEGACY_MAPPING_REPORT_NAME, selection.lumos_legacy_mapping_rows)

    # 统计数据
    source_counts = _source_sample_counts(selection.pointcloud_rows, selection.non_pointcloud_rows)
    selection_config = {
        "format": "umi_mot_real_training_selection",
        "selection_mode": selection_mode,
        "base_root": str(base_root),
        "output_root": str(output_root),
        "base_valid_episode_manifest_path": str(base_valid_episode_manifest_path),
        "sample_from_existing_selection_path": None if source_selection_path is None else str(source_selection_path),
        "pointcloud_manifest_path": str(pointcloud_manifest_path),
        "non_pointcloud_manifest_path": str(non_pointcloud_manifest_path),
        "source_sample_counts": source_counts,
        "selection_limit_policy": {
            "max_pointcloud_samples": None if max_pointcloud_samples is None else int(max_pointcloud_samples),
            "max_non_pointcloud_samples": None if max_non_pointcloud_samples is None else int(max_non_pointcloud_samples),
        },
        "pointcloud_selection_policy": (
            "planned before Lumos preprocessing; mapped pointcloud episodes are excluded from "
            "non-pointcloud selection"
        ),
    }
    _write_json(selection_config_path, selection_config)
    return RealTrainSelectionBuildResult(
        output_root=output_root,
        selection_config_path=selection_config_path,
        pointcloud_manifest_path=pointcloud_manifest_path,
        non_pointcloud_manifest_path=non_pointcloud_manifest_path,
        num_pointcloud_samples=len(selection.pointcloud_rows),
        num_non_pointcloud_samples=len(selection.non_pointcloud_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select real UMI MOT training episodes.")
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-from-existing-selection-path", type=Path, default=None)
    parser.add_argument("--max-pointcloud-samples", type=int, default=None)
    parser.add_argument("--max-non-pointcloud-samples", type=int, default=None)
    args = parser.parse_args()

    result = build_real_mot_training_selection(
        base_root=args.base_root,
        output_root=args.output_root,
        sample_from_existing_selection_path=args.sample_from_existing_selection_path,
        max_pointcloud_samples=args.max_pointcloud_samples,
        max_non_pointcloud_samples=args.max_non_pointcloud_samples,
    )
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.__dict__.items()}, indent=2))


if __name__ == "__main__":
    main()
