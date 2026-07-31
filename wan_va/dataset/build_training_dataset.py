"""Build the final real-data MOT training dataset metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .action_cache import build_mot_action_cache
from . import build_training_selection
from .mot_dataset import (
    DEFAULT_REAL_TRAIN_ROOT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_VIDEO_KEYS,
    HAND_TO_VIDEO_KEY,
    HAND_TO_STREAM_ID,
    MOT_HISTORY_CHUNKS,
    MOT_TARGET_CHUNKS,
    MOT_ACTION_CHUNKS,
    MOT_DEFAULT_ACTION_CHUNK_SIZE,
    MOT_GEOMETRY_GROUP_SIZE,
    MOT_MAX_RIGHT_PADDING_RAW_STEPS,
    MOT_TIMESTAMP_POLICY,
    RELATIVE_ACTION_DIM,
    WAN_VAE_TEMPORAL_FACTOR,
    _read_json,
    _write_json,
    _read_jsonl,
    _write_jsonl,
    validate_dataset_index_bounds,
    mot_real_window_frame_ids,
    mot_action_per_frame,
    mot_action_sequence_length,
    mot_geometry_groups,
    mot_latent_frames_per_action_chunk_per_view,
    mot_sampled_video_frames_per_action_chunk_per_view,
)
from .pointcloud_store import PointStore


@dataclass
class RealTrainBuildResult:
    output_root: Path
    mot_config_path: Path
    mot_manifest_path: Path
    non_pointcloud_manifest_path: Path
    num_pointcloud_samples: int
    num_non_pointcloud_samples: int
    action_cache_manifest_path: Path | None


@dataclass
class LoadedRealTrainSelection:
    selection_root: Path
    config: dict[str, Any]
    pointcloud_rows: list[dict[str, Any]]
    non_pointcloud_rows: list[dict[str, Any]]


def _pointcloud_views_for_row(row: dict[str, Any], session_root: Path) -> tuple[list[dict[str, Any]] | None, str, str]:
    # 找到 left/right 具体的路径
    hand_dirs = build_training_selection._hand_dirs(session_root)
    if hand_dirs is None:
        return None, "missing_hand_clips", ""
    views = []
    for hand in ("left", "right"):
        # 确保 pointcloud 存在
        store_dir = hand_dirs[hand] / "preprocessed_pointcloud"
        if not (store_dir / "_SUCCESS").is_file():
            return None, "missing_preprocessed_pointcloud", str(store_dir)

        # 在 step2 的基础上，写入最终的 preprocess pointcloud 路径 LINK: wan_va/dataset/build_training_selection.py:295
        base_view = next(view for view in row["views"] if view["video_key"] == HAND_TO_VIDEO_KEY[hand])
        view = dict(base_view)
        view.update(
            {
                "hand": hand,
                "hand_dir": str(hand_dirs[hand]),
                "stream_id": HAND_TO_STREAM_ID[hand],
                "preprocessed_pointcloud_dir": str(store_dir),
            }
        )
        views.append(view)
    return views, "", ""


def _covered_frames(view: dict[str, Any]) -> set[int]:
    store = PointStore.open(view["preprocessed_pointcloud_dir"])
    return set(range(store.num_rows))


def _pointcloud_store_lengths(views: list[dict[str, Any]]) -> list[int]:
    return [PointStore.open(view["preprocessed_pointcloud_dir"]).num_rows for view in views]


def _window_max_current(
    row: dict[str, Any],
    action_chunk_size: int,
    max_right_padding_raw_steps: int = MOT_MAX_RIGHT_PADDING_RAW_STEPS,
) -> int:
    return (
        int(row["segment"]["end_frame"])
        + int(max_right_padding_raw_steps)
        - int(action_chunk_size)
        - 1
    )


def _full_valid_start_range(row: dict[str, Any], action_chunk_size: int) -> list[int]:
    max_current = _window_max_current(row, action_chunk_size)
    min_current = int(row["segment"]["start_frame"])
    if max_current < min_current:
        return []
    return [min_current, max_current]


def _pointcloud_valid_start_range(
    row: dict[str, Any],
    views: list[dict[str, Any]],
    *,
    action_chunk_size: int,
    video_downsample_ratio: int,
) -> list[int]:
    # 获取当前 episode 可采样范围，max_start_idx 考虑了 padding
    full_range = _full_valid_start_range(row, action_chunk_size)
    if not full_range:
        return []
    # 获取 left/right hand 的 pointcloud行数（其实就是pointcloud 对应的 episode length）
    covered = [_covered_frames(view) for view in views]
    first_valid = int(row["segment"]["start_frame"])
    last_valid = int(row["segment"]["end_frame"]) - 1
    valid_starts: list[int] = []
    # 遍历所有可选 chunk
    start, end = full_range
    for value in range(int(start), int(end) + 1):
        # 用真实训练的 dataset 逻辑算出窗口会读哪些帧，即给定 start_idx，实际用哪些帧做训练
        history_ids, target_ids, _geometry_ids, _geometry_group_mask = mot_real_window_frame_ids(
            current_frame=value,
            video_downsample_ratio=video_downsample_ratio,
            action_chunk_size=action_chunk_size,
        )
        frame_ids = history_ids + target_ids
        padded_frames = [
            first_valid if int(frame) < first_valid else min(int(frame), last_valid)
            for frame in frame_ids
        ]
        # 确保实际的 pointcloud 覆盖了这些帧
        if all(all(frame in frame_set for frame in padded_frames) for frame_set in covered):
            valid_starts.append(value)
    if not valid_starts:
        return []
    return [valid_starts[0], valid_starts[-1]]


def _materialize_selected_pointcloud_rows(
    selected_rows: list[dict[str, Any]],
    *,
    action_chunk_size: int,
    video_downsample_ratio: int,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    skipped_rows: list[dict[str, Any]] = []

    # NOTE: 对于genrobot数据，由于decode的问题，会出现pointcloud 比rgb少一帧的情况，保留这部分数据，截断最后一帧
    truncated_rows: list[dict[str, Any]] = []

    for selected_row in selected_rows:
        # 获取 pointcloud 保存目录，step2 完成了 pointcloud session 和 lerobot episode 间的对应，所以这里能直接读
        session_root = Path(selected_row["pointcloud_session_dir"])
        # 校验路径，确保 preprocess pointcloud 存在
        views, view_skip_reason, view_skip_detail = _pointcloud_views_for_row(selected_row, session_root)
        if views is None:
            reason = view_skip_reason or "missing_hand_clips"
            skipped[reason] = skipped.get(reason, 0) + 1
            row = {
                "task_uid": selected_row.get("task_uid"),
                "episode_index": selected_row.get("episode_index"),
                "session_root": str(session_root),
                "reason": reason,
            }
            if view_skip_detail:
                row["detail"] = view_skip_detail
            skipped_rows.append(row)
            continue
        # 长度校验：left/hand pointcloud长度应一致，且都等于 episode length
        segment_start = int(selected_row["segment"]["start_frame"])
        segment_end = int(selected_row["segment"]["end_frame"])
        episode_len = segment_end - segment_start
        lengths = _pointcloud_store_lengths(views)
        row_for_windows = selected_row
        truncated_row: dict[str, Any] | None = None
        if any(length != episode_len for length in lengths):
            pointcloud_lengths = [int(length) for length in lengths]
            pointcloud_len = pointcloud_lengths[0] if pointcloud_lengths else -1

            # 对于 genrobot 数据，保存截断最后一帧的数据
            can_drop_tail_frame = (
                selected_row.get("source_dataset") == "genrobot_lerobot"
                and len(pointcloud_lengths) == 2
                and len(set(pointcloud_lengths)) == 1
                and pointcloud_len == episode_len - 1
                and pointcloud_len > 0
            )
            if can_drop_tail_frame:
                row_for_windows = json.loads(json.dumps(selected_row))
                original_dataset_to_index = int(row_for_windows["dataset_to_index"])
                new_segment_end = segment_start + pointcloud_len
                row_for_windows["segment"]["end_frame"] = new_segment_end
                row_for_windows["dataset_to_index"] = int(row_for_windows["dataset_from_index"]) + new_segment_end
                row_for_windows.update(
                    {
                        "pointcloud_truncation_policy": "drop_tail_frame",
                        "original_segment_end_frame": segment_end,
                        "original_dataset_to_index": original_dataset_to_index,
                        "original_episode_len": int(episode_len),
                        "effective_episode_len": int(pointcloud_len),
                        "pointcloud_lengths": pointcloud_lengths,
                    }
                )
                truncated_row = {
                    "task_uid": selected_row.get("task_uid"),
                    "episode_index": selected_row.get("episode_index"),
                    "session_root": str(session_root),
                    "reason": "pointcloud_tail_frame_truncated",
                    "original_episode_len": int(episode_len),
                    "effective_episode_len": int(pointcloud_len),
                    "original_segment_end_frame": int(segment_end),
                    "new_segment_end_frame": int(new_segment_end),
                    "pointcloud_lengths": pointcloud_lengths,
                }
            else:
                skipped["pointcloud_length_mismatch"] = skipped.get("pointcloud_length_mismatch", 0) + 1
                skipped_rows.append(
                    {
                        "task_uid": selected_row.get("task_uid"),
                        "episode_index": selected_row.get("episode_index"),
                        "session_root": str(session_root),
                        "reason": "pointcloud_length_mismatch",
                        "episode_len": int(episode_len),
                        "pointcloud_lengths": pointcloud_lengths,
                    }
                )
                continue
        # 逐 chunk 校验：
        valid_range = _pointcloud_valid_start_range(
            row_for_windows,
            views,
            action_chunk_size=action_chunk_size,
            video_downsample_ratio=video_downsample_ratio,
        )
        if not valid_range:
            skipped["no_valid_pointcloud_window"] = skipped.get("no_valid_pointcloud_window", 0) + 1
            skipped_rows.append(
                {
                    "task_uid": selected_row.get("task_uid"),
                    "episode_index": selected_row.get("episode_index"),
                    "session_root": str(session_root),
                    "reason": "no_valid_pointcloud_window",
                }
            )
            continue
        out = json.loads(json.dumps(row_for_windows))
        out.update(
            {
                "has_pointcloud": True,
                "video_downsample_ratio": int(video_downsample_ratio),
                "valid_start_range": valid_range,
                "views": views,
            }
        )
        rows.append(out)
        if truncated_row is not None:
            truncated_rows.append(truncated_row)
    return rows, skipped, skipped_rows, truncated_rows


def _materialize_selected_non_pointcloud_rows(
    selected_rows: list[dict[str, Any]],
    *,
    action_chunk_size: int,
    video_downsample_ratio: int,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    skipped_rows: list[dict[str, Any]] = []
    for selected_row in selected_rows:
        valid_range = _full_valid_start_range(selected_row, action_chunk_size)
        if not valid_range:
            skipped["no_valid_window"] = skipped.get("no_valid_window", 0) + 1
            skipped_rows.append(
                {
                    "task_uid": selected_row.get("task_uid"),
                    "episode_index": selected_row.get("episode_index"),
                    "source_lerobot_task_dir": selected_row.get("source_lerobot_task_dir"),
                    "reason": "no_valid_window",
                }
            )
            continue
        out = json.loads(json.dumps(selected_row))
        out.update(
            {
                "has_pointcloud": False,
                "video_downsample_ratio": int(video_downsample_ratio),
                "valid_start_range": valid_range,
            }
        )
        rows.append(out)
    return rows, skipped, skipped_rows


def _make_text_cache(
    texts: list[str],
    output_root: Path,
    model_root: Path,
    device: str,
    text_embedder=None,
) -> tuple[Path, Path]:
    if text_embedder is None:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        from wan_va.modules.utils import load_text_encoder, load_tokenizer

        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        tokenizer = load_tokenizer(str(model_root / "tokenizer"))
        text_encoder = load_text_encoder(str(model_root / "text_encoder"), torch_dtype=dtype, torch_device=device).eval()

        def text_embedder(text: str) -> torch.Tensor:
            prompt = prompt_clean(text)
            text_inputs = tokenizer(
                [prompt],
                padding="max_length",
                max_length=512,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            mask = text_inputs.attention_mask.to(device)
            seq_len = int(mask.gt(0).sum(dim=1)[0].item())
            with torch.no_grad():
                embeds = text_encoder(input_ids, mask).last_hidden_state.to(dtype=dtype)
            embeds = embeds[0, :seq_len]
            return torch.cat([embeds, embeds.new_zeros(512 - embeds.shape[0], embeds.shape[1])], dim=0).cpu()

    def to_tensor(value) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.cpu()
        return torch.as_tensor(value).cpu()

    unique_texts = sorted(set(texts))
    text_cache = {text: to_tensor(text_embedder(text)) for text in unique_texts}
    empty_emb = to_tensor(text_embedder(""))
    empty_path = output_root / "empty_emb.pt"
    cache_path = output_root / "text_emb_cache.pt"
    torch.save(empty_emb, empty_path)
    torch.save(text_cache, cache_path)
    return empty_path, cache_path


def _resolve_selection_path(selection_root: Path, value: str | None) -> Path:
    if not value:
        from .build_training_selection import SELECTION_CONFIG_NAME

        raise ValueError(f"Missing path in {selection_root / 'meta' / SELECTION_CONFIG_NAME}")
    path = Path(value)
    return path if path.is_absolute() else selection_root / path


def load_real_mot_training_selection(selection_root: str | Path) -> LoadedRealTrainSelection:
    from .build_training_selection import SELECTION_CONFIG_NAME

    selection_root = Path(selection_root).resolve()
    selection_config_path = selection_root / "meta" / SELECTION_CONFIG_NAME
    config = _read_json(selection_config_path)
    pointcloud_manifest_path = _resolve_selection_path(selection_root, config.get("pointcloud_manifest_path"))
    non_pointcloud_manifest_path = _resolve_selection_path(selection_root, config.get("non_pointcloud_manifest_path"))
    pointcloud_rows = _read_jsonl(pointcloud_manifest_path)
    non_pointcloud_rows = _read_jsonl(non_pointcloud_manifest_path)
    validate_dataset_index_bounds(pointcloud_rows + non_pointcloud_rows, context=str(selection_config_path))
    return LoadedRealTrainSelection(
        selection_root=selection_root,
        config=config,
        pointcloud_rows=pointcloud_rows,
        non_pointcloud_rows=non_pointcloud_rows,
    )


def _norm_stats_path_for_task(task_root: Path, action_chunk_size: int) -> Path:
    return task_root / "meta" / f"norm_stats_deltarot6d_chunk{int(action_chunk_size)}.json"


_NORM_STAT_KEYS = ("q01", "q99", "mean", "std")


def load_required_task_norm_stats(
    *,
    selected_rows: list[dict[str, Any]],
    action_chunk_size: int,
) -> dict[str, dict[str, list[float]]]:
    task_roots = {
        str(row["task_uid"]): Path(row["source_lerobot_task_dir"])
        for row in selected_rows
    }
    stats_by_task: dict[str, dict[str, list[float]]] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for task_uid, task_root in sorted(task_roots.items()):
        path = _norm_stats_path_for_task(task_root, action_chunk_size)
        if not path.is_file():
            missing.append(f"{task_uid} ({path})")
            continue
        payload = _read_json(path)
        if int(payload.get("action_chunk_size", -1)) != int(action_chunk_size):
            invalid.append(f"{task_uid} has action_chunk_size={payload.get('action_chunk_size')}")
            continue
        stats: dict[str, list[float]] = {}
        for key in _NORM_STAT_KEYS:
            values = payload.get(key)
            if not isinstance(values, list) or len(values) != RELATIVE_ACTION_DIM:
                invalid.append(f"{task_uid} has invalid q01/q99/mean/std")
                break
            stats[key] = [float(value) for value in values]
        else:
            stats_by_task[task_uid] = stats
    if missing or invalid:
        message = (
            "Missing or invalid task norm stats. Run python -m wan_va.dataset.cal_norm_stats "
            "with the same selection-root and action-chunk-size first."
        )
        details = []
        if missing:
            details.append("missing: " + "; ".join(missing[:10]))
        if invalid:
            details.append("invalid: " + "; ".join(invalid[:10]))
        raise FileNotFoundError(message + " " + " ".join(details))
    return stats_by_task


def build_real_mot_train_dataset(
    *,
    selection_root: str | Path,
    output_root: str | Path = DEFAULT_REAL_TRAIN_ROOT,
    action_chunk_size: int = MOT_DEFAULT_ACTION_CHUNK_SIZE,
    video_downsample_ratio: int = 4,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    device: str = "cuda:0",
    text_embedder=None,
    write_action_cache: bool = True,
    num_action_cache_workers: int = 16,
) -> RealTrainBuildResult:
    from .build_training_selection import SELECTION_CONFIG_NAME

    # 读取 step 2 选出来的 manifest
    loaded_selection = load_real_mot_training_selection(selection_root)
    selection_config = loaded_selection.config

    base_root = Path(selection_config["base_root"]).resolve()
    output_root = Path(output_root).resolve()
    action_chunk_size = int(action_chunk_size)
    video_downsample_ratio = int(video_downsample_ratio)
    output_root.mkdir(parents=True, exist_ok=True)

    # 遍历所有 row 进行校验， pointcloud 会做 preprocess_pointcloud 存在性校验和可用 chunk 校验
    # 可用 chunk 校验时也会同步写入 vaild sample idx
    # NOTE: 因此这一步需要最终真实训练用到的 action_chunk_size 和 video_downsample_ratio
    pointcloud_rows, materialize_skipped, materialize_skipped_rows, materialize_truncated_rows = _materialize_selected_pointcloud_rows(
        loaded_selection.pointcloud_rows,
        action_chunk_size=action_chunk_size,
        video_downsample_ratio=video_downsample_ratio,
    )
    # non-pointcloud 只做可用 chunk 校验
    non_pointcloud_rows, pure_skipped, pure_skipped_rows = _materialize_selected_non_pointcloud_rows(
        loaded_selection.non_pointcloud_rows,
        action_chunk_size=action_chunk_size,
        video_downsample_ratio=video_downsample_ratio,
    )
    _write_json(output_root / "reports" / "pointcloud_skipped.json", {"skipped": materialize_skipped})
    _write_jsonl(output_root / "reports" / "pointcloud_skipped.jsonl", materialize_skipped_rows)
    _write_json(
        output_root / "reports" / "pointcloud_truncated.json",
        {"truncated": {"pointcloud_tail_frame_truncated": len(materialize_truncated_rows)}},
    )
    _write_jsonl(output_root / "reports" / "pointcloud_truncated.jsonl", materialize_truncated_rows)
    _write_json(output_root / "reports" / "non_pointcloud_skipped.json", {"skipped": pure_skipped})
    _write_jsonl(output_root / "reports" / "non_pointcloud_skipped.jsonl", pure_skipped_rows)
    train_rows = pointcloud_rows + non_pointcloud_rows
    if not train_rows:
        raise ValueError(f"No train rows remained after materialization. See {output_root / 'reports'}")
    if any(row.get("timestamp_policy") != MOT_TIMESTAMP_POLICY for row in train_rows):
        raise ValueError(
            "Training selection has an unsupported or missing timestamp_policy. "
            "Rebuild the base cache and training selection."
        )
    
    # 检查 norm stats 是否都存在
    task_stats = load_required_task_norm_stats(
        selected_rows=train_rows,
        action_chunk_size=action_chunk_size,
    )

    mot_final_training_pointcloud_manifest_path = (
        output_root / "meta" / "mot_final_training_pointcloud_manifest.jsonl"
    )
    mot_final_training_non_pointcloud_manifest_path = (
        output_root / "meta" / "mot_final_training_non_pointcloud_manifest.jsonl"
    )
    _write_jsonl(mot_final_training_pointcloud_manifest_path, pointcloud_rows)
    _write_jsonl(mot_final_training_non_pointcloud_manifest_path, non_pointcloud_rows)

    # 提前算好所有 text embedding
    texts = [row["segment"]["action_text"] for row in train_rows]
    empty_emb_path, text_emb_cache_path = _make_text_cache(
        texts=texts,
        output_root=output_root,
        model_root=Path(model_root),
        device=device,
        text_embedder=text_embedder,
    )

    # 生成统计信息
    source_counts = build_training_selection._source_sample_counts(pointcloud_rows, non_pointcloud_rows)
    default_norm_key = train_rows[0]["norm_stats_key"]

    sampled_video_frames_per_view = mot_sampled_video_frames_per_action_chunk_per_view(
        action_chunk_size,
        video_downsample_ratio,
    )
    latent_frames_per_view = mot_latent_frames_per_action_chunk_per_view(
        action_chunk_size,
        video_downsample_ratio,
    )
    action_sequence_length = mot_action_sequence_length(action_chunk_size)
    action_per_frame = mot_action_per_frame(video_downsample_ratio)
    geometry_groups = mot_geometry_groups(action_chunk_size, video_downsample_ratio)
    mot_config = {
        "format": "umi_mot_real_train",
        "base_root": str(base_root),
        "selection_root": str(loaded_selection.selection_root),
        "training_selection_config_path": str(loaded_selection.selection_root / "meta" / SELECTION_CONFIG_NAME),
        "output_root": str(output_root),
        "mot_manifest_path": str(mot_final_training_pointcloud_manifest_path),
        "non_pointcloud_manifest_path": str(mot_final_training_non_pointcloud_manifest_path),
        "empty_emb_path": str(empty_emb_path),
        "text_emb_cache_path": str(text_emb_cache_path),
        "obs_cam_keys": list(DEFAULT_VIDEO_KEYS),
        "timestamp_policy": MOT_TIMESTAMP_POLICY,
        "action_dim": RELATIVE_ACTION_DIM,
        "action_representation": "relative_to_chunk_reference_state_rot6d_cols",
        "video_downsample_ratio": int(video_downsample_ratio),
        "vae_temporal_factor": WAN_VAE_TEMPORAL_FACTOR,
        "action_chunk_size": int(action_chunk_size),
        "action_sequence_length": action_sequence_length,
        "action_per_frame": action_per_frame,
        "action_frames": geometry_groups,
        "action_chunks": MOT_ACTION_CHUNKS,
        "history_chunks": MOT_HISTORY_CHUNKS,
        "target_chunks": MOT_TARGET_CHUNKS,
        "sampled_video_frames_per_action_chunk_per_view": sampled_video_frames_per_view,
        "latent_frames_per_action_chunk_per_view": latent_frames_per_view,
        "sampled_video_images_per_action_chunk": sampled_video_frames_per_view * len(DEFAULT_VIDEO_KEYS),
        "latent_view_items_per_action_chunk": latent_frames_per_view * len(DEFAULT_VIDEO_KEYS),
        "geometry_groups": geometry_groups,
        "geometry_group_size": MOT_GEOMETRY_GROUP_SIZE,
        "rollout_stride_raw_steps": int(action_chunk_size),
        "norm_stat": task_stats[default_norm_key],
        "norm_stats_by_task": task_stats,
        "norm_stat_policy": f"by_task_uid_selected_episodes_relative20_chunk{int(action_chunk_size)}",
        "source_sample_counts": source_counts,
        "selection_limit_policy": selection_config.get("selection_limit_policy", {}),
        "pointcloud_snapshot_policy": "selected before preprocessing; final build materializes the selected rows only",
    }
    mot_config_path = output_root / "meta" / "mot_config.json"
    _write_json(mot_config_path, mot_config)

    action_cache_manifest_path = None
    if write_action_cache:
        action_cache_manifest_path = build_mot_action_cache(
            dataset_root=output_root,
            output_root=output_root / "cache" / "actions",
            manifest_paths=[
                mot_final_training_pointcloud_manifest_path,
                mot_final_training_non_pointcloud_manifest_path,
            ],
            num_workers=int(num_action_cache_workers),
            update_mot_config=True,
        )

    return RealTrainBuildResult(
        output_root=output_root,
        mot_config_path=mot_config_path,
        mot_manifest_path=mot_final_training_pointcloud_manifest_path,
        non_pointcloud_manifest_path=mot_final_training_non_pointcloud_manifest_path,
        num_pointcloud_samples=len(pointcloud_rows),
        num_non_pointcloud_samples=len(non_pointcloud_rows),
        action_cache_manifest_path=action_cache_manifest_path,
    )




def main() -> None:
    parser = argparse.ArgumentParser(description="Build real UMI MOT training dataset metadata.")
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REAL_TRAIN_ROOT)
    parser.add_argument("--action-chunk-size", type=int, default=MOT_DEFAULT_ACTION_CHUNK_SIZE)
    parser.add_argument("--video-downsample-ratio", type=int, default=4)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-action-cache", action="store_true")
    parser.add_argument("--num-action-cache-workers", type=int, default=16)
    args = parser.parse_args()
    result = build_real_mot_train_dataset(
        selection_root=args.selection_root,
        output_root=args.output_root,
        action_chunk_size=args.action_chunk_size,
        video_downsample_ratio=args.video_downsample_ratio,
        model_root=args.model_root,
        device=args.device,
        write_action_cache=not args.no_action_cache,
        num_action_cache_workers=args.num_action_cache_workers,
    )
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.__dict__.items()}, indent=2))


if __name__ == "__main__":
    main()
