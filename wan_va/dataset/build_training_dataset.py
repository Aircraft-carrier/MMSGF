"""Build the final Video+Action MOT training metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .action_cache import build_mot_action_cache
from .mot_dataset import (
    DEFAULT_MODEL_ROOT,
    DEFAULT_REAL_TRAIN_ROOT,
    DEFAULT_VIDEO_KEYS,
    MOT_ACTION_CHUNKS,
    MOT_DEFAULT_ACTION_CHUNK_SIZE,
    MOT_HISTORY_CHUNKS,
    MOT_MAX_RIGHT_PADDING_RAW_STEPS,
    MOT_TARGET_CHUNKS,
    MOT_TIMESTAMP_POLICY,
    RELATIVE_ACTION_DIM,
    WAN_VAE_TEMPORAL_FACTOR,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    mot_action_per_frame,
    mot_action_sequence_length,
    mot_latent_frames_per_action_chunk_per_view,
    mot_sampled_video_frames_per_action_chunk_per_view,
    validate_dataset_index_bounds,
)


@dataclass
class RealTrainBuildResult:
    output_root: Path
    mot_config_path: Path
    mot_manifest_path: Path
    num_samples: int
    action_cache_manifest_path: Path | None


@dataclass
class LoadedRealTrainSelection:
    selection_root: Path
    config: dict[str, Any]
    rows: list[dict[str, Any]]


def _window_max_current(row: dict[str, Any], action_chunk_size: int) -> int:
    return (
        int(row["segment"]["end_frame"])
        + MOT_MAX_RIGHT_PADDING_RAW_STEPS
        - int(action_chunk_size)
        - 1
    )


def _full_valid_start_range(row: dict[str, Any], action_chunk_size: int) -> list[int]:
    start = int(row["segment"]["start_frame"])
    end = _window_max_current(row, action_chunk_size)
    return [] if end < start else [start, end]


def _materialize_rows(
    selected_rows: list[dict[str, Any]],
    *,
    action_chunk_size: int,
    video_downsample_ratio: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    skipped = []
    for selected_row in selected_rows:
        valid_range = _full_valid_start_range(selected_row, action_chunk_size)
        if not valid_range:
            skipped.append(
                {
                    "task_uid": selected_row.get("task_uid"),
                    "episode_index": selected_row.get("episode_index"),
                    "reason": "no_valid_window",
                }
            )
            continue
        row = json.loads(json.dumps(selected_row))
        row.update(
            video_downsample_ratio=int(video_downsample_ratio),
            valid_start_range=valid_range,
        )
        rows.append(row)
    return rows, skipped


def _make_text_cache(texts, output_root, model_root, device, text_embedder=None):
    if text_embedder is None:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        from wan_va.modules.utils import load_text_encoder, load_tokenizer

        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        tokenizer = load_tokenizer(str(model_root / "tokenizer"))
        text_encoder = load_text_encoder(
            str(model_root / "text_encoder"),
            torch_dtype=dtype,
            torch_device=device,
        ).eval()

        def text_embedder(text: str) -> torch.Tensor:
            inputs = tokenizer(
                [prompt_clean(text)],
                padding="max_length",
                max_length=512,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.to(device)
            mask = inputs.attention_mask.to(device)
            seq_len = int(mask.gt(0).sum(dim=1)[0].item())
            with torch.no_grad():
                embeds = text_encoder(input_ids, mask).last_hidden_state.to(dtype=dtype)
            embeds = embeds[0, :seq_len]
            return torch.cat(
                [embeds, embeds.new_zeros(512 - embeds.shape[0], embeds.shape[1])]
            ).cpu()

    def to_tensor(value):
        return value.cpu() if torch.is_tensor(value) else torch.as_tensor(value).cpu()

    text_cache = {text: to_tensor(text_embedder(text)) for text in sorted(set(texts))}
    empty_path = output_root / "empty_emb.pt"
    cache_path = output_root / "text_emb_cache.pt"
    torch.save(to_tensor(text_embedder("")), empty_path)
    torch.save(text_cache, cache_path)
    return empty_path, cache_path


def load_real_mot_training_selection(selection_root: str | Path) -> LoadedRealTrainSelection:
    from .build_training_selection import SELECTION_CONFIG_NAME

    selection_root = Path(selection_root).resolve()
    config_path = selection_root / "meta" / SELECTION_CONFIG_NAME
    config = _read_json(config_path)
    manifest_path = Path(config["manifest_path"])
    rows = _read_jsonl(manifest_path)
    validate_dataset_index_bounds(rows, context=str(config_path))
    return LoadedRealTrainSelection(selection_root, config, rows)


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
    stats_by_task = {}
    missing = []
    for task_uid, task_root in sorted(task_roots.items()):
        path = task_root / "meta" / f"norm_stats_deltarot6d_chunk{action_chunk_size}.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        payload = _read_json(path)
        if int(payload.get("action_chunk_size", -1)) != int(action_chunk_size):
            missing.append(str(path))
            continue
        stats = {}
        for key in _NORM_STAT_KEYS:
            values = payload.get(key)
            if not isinstance(values, list) or len(values) != RELATIVE_ACTION_DIM:
                missing.append(str(path))
                break
            stats[key] = [float(value) for value in values]
        else:
            stats_by_task[task_uid] = stats
    if missing:
        raise FileNotFoundError(
            "Missing or invalid task norm stats: " + ", ".join(missing[:10])
        )
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

    selection = load_real_mot_training_selection(selection_root)
    output_root = Path(output_root).resolve()
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    rows, skipped = _materialize_rows(
        selection.rows,
        action_chunk_size=int(action_chunk_size),
        video_downsample_ratio=int(video_downsample_ratio),
    )
    _write_jsonl(output_root / "reports" / "skipped.jsonl", skipped)
    if not rows:
        raise ValueError("No train rows remained after materialization")
    if any(row.get("timestamp_policy") != MOT_TIMESTAMP_POLICY for row in rows):
        raise ValueError("Training selection has an unsupported timestamp_policy")

    task_stats = load_required_task_norm_stats(
        selected_rows=rows,
        action_chunk_size=int(action_chunk_size),
    )
    manifest_path = output_root / "meta" / "mot_final_training_manifest.jsonl"
    _write_jsonl(manifest_path, rows)
    empty_path, text_cache_path = _make_text_cache(
        [row["segment"]["action_text"] for row in rows],
        output_root,
        Path(model_root),
        device,
        text_embedder,
    )
    sampled_frames = mot_sampled_video_frames_per_action_chunk_per_view(
        action_chunk_size, video_downsample_ratio
    )
    latent_frames = mot_latent_frames_per_action_chunk_per_view(
        action_chunk_size, video_downsample_ratio
    )
    action_frames = MOT_ACTION_CHUNKS * latent_frames
    source_counts = {}
    for row in rows:
        source = str(row.get("source_dataset", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
    config = {
        "format": "umi_mot_real_train_va_v1",
        "selection_root": str(selection.selection_root),
        "training_selection_config_path": str(
            selection.selection_root / "meta" / SELECTION_CONFIG_NAME
        ),
        "output_root": str(output_root),
        "mot_manifest_path": str(manifest_path),
        "empty_emb_path": str(empty_path),
        "text_emb_cache_path": str(text_cache_path),
        "obs_cam_keys": list(DEFAULT_VIDEO_KEYS),
        "timestamp_policy": MOT_TIMESTAMP_POLICY,
        "action_dim": RELATIVE_ACTION_DIM,
        "action_representation": "relative_to_chunk_reference_state_rot6d_cols",
        "video_downsample_ratio": int(video_downsample_ratio),
        "vae_temporal_factor": WAN_VAE_TEMPORAL_FACTOR,
        "action_chunk_size": int(action_chunk_size),
        "action_sequence_length": mot_action_sequence_length(action_chunk_size),
        "action_per_frame": mot_action_per_frame(video_downsample_ratio),
        "action_frames": action_frames,
        "action_chunks": MOT_ACTION_CHUNKS,
        "history_chunks": MOT_HISTORY_CHUNKS,
        "target_chunks": MOT_TARGET_CHUNKS,
        "sampled_video_frames_per_action_chunk_per_view": sampled_frames,
        "latent_frames_per_action_chunk_per_view": latent_frames,
        "rollout_stride_raw_steps": int(action_chunk_size),
        "norm_stat": task_stats[rows[0]["norm_stats_key"]],
        "norm_stats_by_task": task_stats,
        "source_sample_counts": source_counts,
    }
    config_path = output_root / "meta" / "mot_config.json"
    _write_json(config_path, config)
    action_cache_manifest_path = None
    if write_action_cache:
        action_cache_manifest_path = build_mot_action_cache(
            dataset_root=output_root,
            output_root=output_root / "cache" / "actions",
            manifest_paths=[manifest_path],
            num_workers=int(num_action_cache_workers),
            update_mot_config=True,
        )
    return RealTrainBuildResult(
        output_root,
        config_path,
        manifest_path,
        len(rows),
        action_cache_manifest_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Video+Action MOT metadata.")
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
