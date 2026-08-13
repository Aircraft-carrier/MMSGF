"""Prepare the RoboTwin EEF clean-50 LeRobot-v2.1 snapshot for MOT training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .action_cache import build_mot_action_cache
from .build_training_dataset import _make_text_cache
from .mot_dataset import (
    MOT_TIMESTAMP_POLICY,
    absolute_actions_to_relative_20d,
    _write_json,
    _write_jsonl,
    mot_action_per_frame,
    mot_action_sequence_length,
    mot_latent_frames_per_action_chunk_per_view,
    mot_sampled_video_frames_per_action_chunk_per_view,
)

CAMERAS = (
    ("observation.images.cam_high", 1),
    ("observation.images.cam_left_wrist", 0),
    ("observation.images.cam_right_wrist", 2),
)
ACTION_CHUNK_SIZE = 48
VIDEO_DOWNSAMPLE_RATIO = 4


def _task_roots(source_root: Path) -> list[Path]:
    direct = [
        path
        for path in sorted(source_root.iterdir())
        if (path / "meta" / "info.json").is_file()
    ]
    if direct:
        return direct

    containers = [
        path
        for path in sorted(source_root.iterdir())
        if path.is_dir()
        and any((child / "meta" / "info.json").is_file() for child in path.iterdir())
    ]
    if len(containers) == 1:
        return [
            path
            for path in sorted(containers[0].iterdir())
            if (path / "meta" / "info.json").is_file()
        ]
    return []


def _episode_rows(source_root: Path, *, task_name: str | None = None):
    for task_root in _task_roots(source_root):
        if task_name is not None and task_root.name != task_name:
            continue
        info_path = task_root / "meta" / "info.json"
        episodes_path = task_root / "meta" / "episodes.jsonl"
        if not info_path.is_file() or not episodes_path.is_file():
            continue
        info = json.loads(info_path.read_text())
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"expected LeRobot v2.1: {task_root}")
        if (info["features"]["action"]["shape"] != [16] or
                info["features"]["observation.state"]["shape"] != [16]):
            raise ValueError(f"expected 16D EEF state/action: {task_root}")
        for camera, _ in CAMERAS:
            if camera not in info["features"]:
                raise ValueError(f"missing camera {camera}: {task_root}")
        for line in episodes_path.read_text().splitlines():
            record = json.loads(line)
            episode = int(record["episode_index"])
            text = str(record["tasks"][0]).strip()
            parquet = task_root / "data/chunk-000" / f"episode_{episode:06d}.parquet"
            videos = [task_root / "videos/chunk-000" / camera / f"episode_{episode:06d}.mp4" for camera, _ in CAMERAS]
            if not text or not parquet.is_file() or any(not path.is_file() for path in videos):
                continue
            table = pq.read_table(parquet, columns=["index", "action", "observation.state"])
            indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            if actions.shape != states.shape or actions.shape[1] != 16 or len(indices) != int(record["length"]):
                raise ValueError(f"invalid EEF episode payload: {parquet}")
            yield task_root.name, task_root, episode, text, parquet, videos, indices, actions, states


def _norm_stats(episodes):
    values = []
    for _, _, _, _, _, _, _, actions, states in episodes:
        starts = np.arange(0, max(1, len(actions) - ACTION_CHUNK_SIZE + 17), dtype=np.int64)
        target = np.minimum(starts[:, None] + np.arange(ACTION_CHUNK_SIZE), len(actions) - 1)
        ref = np.broadcast_to(states[starts, None], (len(starts), ACTION_CHUNK_SIZE, 16))
        values.append(absolute_actions_to_relative_20d(ref, actions[target]).reshape(-1, 20))
    joined = np.concatenate(values, axis=0)
    return {"q01": np.quantile(joined, .01, axis=0).astype(float).tolist(),
            "q99": np.quantile(joined, .99, axis=0).astype(float).tolist()}


def build(
    source_root: Path,
    output_root: Path,
    model_root: Path | None,
    *,
    device: str,
    action_cache: bool,
    task_name: str | None = None,
    text_embeddings: bool = True,
):
    source_root, output_root = source_root.resolve(), output_root.resolve()
    grouped = {}
    for item in _episode_rows(source_root, task_name=task_name):
        grouped.setdefault(item[0], []).append(item)
    if not grouped:
        suffix = f" for task {task_name!r}" if task_name is not None else ""
        raise ValueError(f"no complete EEF episodes under {source_root}{suffix}")
    rows, norms, texts = [], {}, []
    for task, episodes in grouped.items():
        norms[task] = _norm_stats(episodes)
        for _, task_root, episode, text, parquet, videos, indices, actions, _ in episodes:
            rows.append({"task_uid": task, "norm_stats_key": task, "source_dataset": "robotwin_eef_clean_50",
                "episode_index": episode, "fps": 50, "timestamp_policy": MOT_TIMESTAMP_POLICY,
                "source_lerobot_task_dir": str(task_root), "data_file": str(parquet),
                "dataset_from_index": int(indices[0]), "dataset_to_index": int(indices[-1]) + 1,
                "segment": {"start_frame": 0, "end_frame": len(actions), "tasks": [text], "action_text": text},
                "valid_start_range": [0, len(actions) - ACTION_CHUNK_SIZE + 16],
                "views": [{"stream_id": sid, "video_key": camera, "video_path": str(video), "video_from_timestamp": 0.0}
                          for (camera, sid), video in zip(CAMERAS, videos)]})
            texts.append(text)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    manifest = output_root / "meta/mot_final_training_manifest.jsonl"
    _write_jsonl(manifest, rows)
    if text_embeddings:
        if model_root is None:
            raise ValueError("model_root is required unless text_embeddings=False")
        _make_text_cache(texts, output_root, model_root, device)
    config = {"format": "robotwin_eef_clean50_mot_v1", "mot_manifest_path": str(manifest),
        "text_embeddings_cached": text_embeddings,
        "obs_cam_keys": [name for name, _ in CAMERAS], "action_dim": 20,
        "action_representation": "relative_to_chunk_reference_state_rot6d_cols", "video_downsample_ratio": 4,
        "vae_temporal_factor": 4, "action_chunk_size": 48, "action_sequence_length": mot_action_sequence_length(48),
        "action_per_frame": mot_action_per_frame(4), "action_frames": 8,
        "sampled_video_frames_per_action_chunk_per_view": mot_sampled_video_frames_per_action_chunk_per_view(48, 4),
        "latent_frames_per_action_chunk_per_view": mot_latent_frames_per_action_chunk_per_view(48, 4),
        "selected_tasks": sorted(grouped), "norm_stat": next(iter(norms.values())),
        "norm_stats_by_task": norms, "source_sample_counts": {"robotwin_eef_clean_50": len(rows)}}
    if text_embeddings:
        config.update({
            "empty_emb_path": str(output_root / "empty_emb.pt"),
            "text_emb_cache_path": str(output_root / "text_emb_cache.pt"),
        })
    _write_json(output_root / "meta/mot_config.json", config)
    if action_cache:
        build_mot_action_cache(dataset_root=output_root, output_root=output_root / "cache/actions", manifest_paths=[manifest], num_workers=8, update_mot_config=True)
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--model-root", type=Path, help="Lingbot model root; required unless --no-text-emb is used.")
    p.add_argument(
        "--task",
        help="Optional exact task directory name; omit to prepare every RoboTwin task.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-action-cache", action="store_true")
    p.add_argument("--no-text-emb", action="store_true", help="Skip text embedding generation; this output cannot train until embeddings are added.")
    a = p.parse_args()
    print(json.dumps({"episodes": build(
        a.source_root,
        a.output_root,
        a.model_root,
        device=a.device,
        action_cache=not a.no_action_cache,
        task_name=a.task,
        text_embeddings=not a.no_text_emb,
    ), "task": a.task}, indent=2))


if __name__ == "__main__":
    main()
