#!/usr/bin/env python3
"""Visualize one Lumos preprocessed pointcloud hand store against its LeRobot video.

PYTHONPATH=. python -m wan_va.dataset.check_preprocessed_pointcloud \
  /team_data/umi_data/lumos_lerobot/20260126/task_20260123O004/pointcloud/multi_sessions_20260126_092933/session_003/left_hand_250801DR48FP25002247 \
  --sample-stride 20 \
  --max-frames 2 \
  --fps 2 \
  --frame-idx 100


"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
from torchcodec.decoders import VideoDecoder

from wan_va.dataset.mot_dataset import MOT_VIDEO_FPS_TOLERANCE, MOT_VIDEO_TIMESTAMP_TOLERANCE_S
from wan_va.dataset.ply_io import write_ply
from wan_va.dataset.pointcloud_store import PointStore, confidence_to_probability


DEFAULT_OUTPUT_ROOT = Path("data/check_lumos_preprocess")
HAND_TO_VIDEO_KEY = {
    "left": "observation.images.robot_0",
    "right": "observation.images.robot_1",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _read_frame_mapping(path: Path) -> list[dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [
            {
                "local_frame_index": int(row["local_frame_index"]),
                "aligned_frame_index": int(row["aligned_frame_index"]),
                "raw_frame_index": int(row["raw_frame_index"]),
            }
            for row in csv.DictReader(f)
        ]
    if not rows:
        raise ValueError(f"empty frame mapping: {path}")
    rows = sorted(rows, key=lambda row: int(row["local_frame_index"]))
    for expected, row in enumerate(rows):
        local = int(row["local_frame_index"])
        if local != expected:
            raise ValueError(f"{path} local_frame_index must be dense 0..N-1; row {expected} has {local}")
    return rows


def _squeeze_points(array: np.ndarray, path: Path) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected local_points [1,F,H,W,3] or [F,H,W,3], got {array.shape} from {path}")
    return array.astype(np.float32, copy=False)


def _squeeze_conf(array: np.ndarray, path: Path) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"expected conf [1,F,H,W], [1,F,H,W,1], or [F,H,W], got {array.shape} from {path}")
    return array.astype(np.float32, copy=False)


def _load_downsample(hand_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None, list[dict[str, int]] | None]:
    downsample_dir = hand_dir / "downsample"
    res_npz = downsample_dir / "res.npz"
    mapping_csv = downsample_dir / "frame_mapping.csv"
    if not mapping_csv.is_file():
        return None, None, None
    mapping = _read_frame_mapping(mapping_csv)
    if not res_npz.is_file():
        return None, None, mapping
    with np.load(res_npz, allow_pickle=False) as data:
        for key in ("local_points", "conf"):
            if key not in data:
                raise KeyError(f"{res_npz} missing required key: {key}")
        points = _squeeze_points(np.asarray(data["local_points"]), res_npz)
        conf = _squeeze_conf(np.asarray(data["conf"]), res_npz)
    if len(mapping) != int(points.shape[0]):
        raise ValueError(f"{mapping_csv} has {len(mapping)} rows, expected {int(points.shape[0])} from {res_npz}")
    if int(conf.shape[0]) != int(points.shape[0]):
        raise ValueError(f"{res_npz} conf has {conf.shape[0]} rows, expected {points.shape[0]}")
    return points, conf, mapping


def _resolve_hand_dir(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    if path.name == "preprocessed_pointcloud":
        store_dir = path
        hand_dir = path.parent
    else:
        hand_dir = path
        store_dir = hand_dir / "preprocessed_pointcloud"
    if not store_dir.is_dir():
        raise FileNotFoundError(store_dir)
    return hand_dir, store_dir


def _task_and_session_roots(hand_dir: Path) -> tuple[Path, Path]:
    session_root = hand_dir.parent
    for parent in hand_dir.parents:
        if parent.name == "pointcloud":
            return parent.parent, session_root
    raise ValueError(f"cannot find pointcloud ancestor for {hand_dir}")


def _hand_name(hand_dir: Path) -> str:
    if hand_dir.name.startswith("left_hand"):
        return "left"
    if hand_dir.name.startswith("right_hand"):
        return "right"
    raise ValueError(f"cannot infer hand from {hand_dir.name}")


def _lumos_episode_index(task_root: Path, session_root: Path) -> int:
    mapping_path = task_root / "raw_lerobot_idx.jsonl"
    pointcloud_root = task_root / "pointcloud"
    session_key = str(session_root.relative_to(pointcloud_root))
    for row in _read_jsonl(mapping_path):
        if "filename" in row and "lerobot_episode_index" in row and str(row["filename"]) == session_key:
            return int(row["lerobot_episode_index"])
        if "session_path" in row and "episode_index" in row and str(row["session_path"]) == session_key:
            return int(row["episode_index"])
    raise ValueError(f"cannot map session {session_key} to episode via {mapping_path}")


def _load_episode_record(task_root: Path, episode_index: int) -> dict[str, Any]:
    for pq_path in sorted((task_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        for record in pq.read_table(pq_path).to_pylist():
            if int(record["episode_index"]) == int(episode_index):
                return record
    raise ValueError(f"episode_index={episode_index} not found under {task_root / 'meta' / 'episodes'}")


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


def _local_timestamp_for_frame(task_root: Path, info: dict[str, Any], record: dict[str, Any], episode_frame: int) -> float:
    data_file = _data_path(task_root, info, record)
    schema_names = set(pq.read_schema(data_file).names)
    if "timestamp" not in schema_names:
        raise ValueError(f"LeRobot video validation requires a timestamp column: {data_file}")
    if "index" not in schema_names:
        raise ValueError(f"LeRobot video validation requires an index column: {data_file}")

    table = pq.read_table(data_file, columns=["timestamp", "index"])
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    if timestamps.ndim > 1:
        timestamps = timestamps.reshape(timestamps.shape[0], -1)[:, 0]
    indices = np.asarray(table["index"].to_pylist(), dtype=np.int64).reshape(-1)
    if timestamps.shape[0] != indices.shape[0]:
        raise ValueError(
            f"LeRobot timestamp/index length mismatch in {data_file}: "
            f"{timestamps.shape[0]} timestamps vs {indices.shape[0]} indices"
        )

    absolute_id = int(record["dataset_from_index"]) + int(episode_frame)
    positions = np.flatnonzero(indices == absolute_id)
    if positions.size != 1:
        raise KeyError(f"Missing LeRobot timestamp row for absolute frame index {absolute_id} in {data_file}")

    local_timestamp = float(timestamps[int(positions[0])])
    fps = float(info["fps"])
    expected_timestamp = int(episode_frame) / fps
    delta = abs(local_timestamp - expected_timestamp)
    if delta >= MOT_VIDEO_TIMESTAMP_TOLERANCE_S:
        raise ValueError(
            "LeRobot timestamp mismatch: "
            f"data_file={data_file} local_frame={int(episode_frame)} "
            f"parquet_timestamp={local_timestamp} expected_timestamp={expected_timestamp} "
            f"delta={delta} tolerance_s={MOT_VIDEO_TIMESTAMP_TOLERANCE_S}"
        )
    return local_timestamp


def _video_frame_index(
    task_root: Path,
    info: dict[str, Any],
    record: dict[str, Any],
    video_key: str,
    episode_frame: int,
    *,
    video_fps: float,
) -> int:
    manifest_fps = float(info["fps"])
    if not np.isfinite(video_fps) or abs(float(video_fps) - manifest_fps) > MOT_VIDEO_FPS_TOLERANCE:
        raise ValueError(
            "Video FPS mismatch: "
            f"manifest_fps={manifest_fps} video_average_fps={float(video_fps)} "
            f"tolerance={MOT_VIDEO_FPS_TOLERANCE}"
        )
    local_timestamp = _local_timestamp_for_frame(task_root, info, record, episode_frame)
    query_timestamp = float(record[f"videos/{video_key}/from_timestamp"]) + local_timestamp
    return int(round(query_timestamp * float(video_fps)))


class _VideoFrameReader:
    def __init__(self, video_path: Path) -> None:
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        self.video_path = video_path
        self.decoder = VideoDecoder(video_path, dimension_order="NHWC", device="cpu", seek_mode="approximate")
        self.average_fps = float(self.decoder.metadata.average_fps)

    def read(self, frame_idx: int) -> np.ndarray:
        frames = self.decoder.get_frames_at([int(frame_idx)]).data
        if int(frames.shape[0]) != 1:
            raise ValueError(f"failed to read frame {frame_idx} from {self.video_path}")
        return frames[0].cpu().numpy()

    def close(self) -> None:
        self.decoder = None


def _resize_rgb(rgb: np.ndarray, h_w: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(h_w[0]), int(h_w[1])
    if rgb.shape[:2] == (target_h, target_w):
        return rgb[..., :3]
    return cv2.resize(rgb[..., :3], (target_w, target_h), interpolation=cv2.INTER_AREA)


def _normalize_panel(values: np.ndarray, valid: np.ndarray, *, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.float32)
    valid_values = np.asarray(values, dtype=np.float32)[valid]
    valid_values = valid_values[np.isfinite(valid_values)]
    if valid_values.size == 0:
        return out
    lo = float(np.percentile(valid_values, 1)) if vmin is None else float(vmin)
    hi = float(np.percentile(valid_values, 99)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-6
    out[valid] = np.clip((np.asarray(values, dtype=np.float32)[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _colorize(norm: np.ndarray, valid: np.ndarray, colormap: int) -> np.ndarray:
    gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, colormap)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb


def _draw_label(rgb: np.ndarray, label: str) -> np.ndarray:
    out = rgb.copy()
    cv2.putText(out, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray, valid: np.ndarray) -> int:
    if colors.shape[:2] != points.shape[:2]:
        colors = _resize_rgb(colors, points.shape[:2])
    return write_ply(path, points, colors, valid)


def _sample_indices(num_rows: int, stride: int, max_frames: int | None) -> list[int]:
    indices = list(range(0, int(num_rows), int(stride)))
    if max_frames is not None:
        indices = indices[: int(max_frames)]
    if not indices and num_rows > 0:
        indices = [0]
    return indices


def _prepare_output_dir(output_dir: Path) -> None:
    for subdir_name in ("ply", "downsample_ply"):
        subdir = output_dir / subdir_name
        if subdir.is_dir():
            for path in subdir.glob("*.ply"):
                path.unlink()
    for filename in ("rgb_depth_conf.mp4", "metadata.json"):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _point_valid_from_conf(points: np.ndarray, conf_logits: np.ndarray, threshold: float) -> np.ndarray:
    conf_prob = confidence_to_probability(conf_logits)
    return (
        np.isfinite(points).all(axis=-1)
        & (np.asarray(points)[..., 2] > 0)
        & np.isfinite(conf_prob)
        & (conf_prob > float(threshold))
    )


def _sample_frame_refs(
    *,
    store: PointStore,
    episode_frames: np.ndarray,
    downsample_mapping: list[dict[str, int]] | None,
    sample_stride: int,
    max_frames: int | None,
    frame_idx: int | None,
) -> list[tuple[int | None, int]]:
    if frame_idx is not None:
        row_idx = int(frame_idx)
        if row_idx < 0 or row_idx >= int(store.num_rows):
            raise ValueError(f"--frame-idx must be in [0, {int(store.num_rows) - 1}], got {row_idx}")
        downsample_idx = None
        if downsample_mapping is not None:
            episode_frame = int(episode_frames[row_idx])
            downsample_by_episode = {
                int(row["aligned_frame_index"]): int(row["local_frame_index"]) for row in downsample_mapping
            }
            downsample_idx = downsample_by_episode.get(episode_frame)
        return [(downsample_idx, row_idx)]

    sample_refs: list[tuple[int | None, int]] = []
    if downsample_mapping is not None:
        for downsample_idx in _sample_indices(len(downsample_mapping), sample_stride, max_frames):
            episode_frame = int(downsample_mapping[downsample_idx]["aligned_frame_index"])
            row_idx = store.row_for_episode_frame(episode_frame)
            if row_idx < 0:
                raise ValueError(
                    f"downsample local_frame_index={downsample_idx} aligned_frame_index={episode_frame} "
                    f"is outside preprocessed store length {store.num_rows}"
                )
            if int(episode_frames[row_idx]) != episode_frame:
                raise ValueError(
                    f"store row mismatch for downsample local_frame_index={downsample_idx}: "
                    f"row {row_idx} has episode_frame={int(episode_frames[row_idx])}, expected {episode_frame}"
                )
            sample_refs.append((downsample_idx, row_idx))
    else:
        for row_idx in _sample_indices(store.num_rows, sample_stride, max_frames):
            sample_refs.append((None, row_idx))
    return sample_refs


def export_check_visualization(
    hand_path: Path,
    *,
    output_root: Path,
    sample_stride: int,
    max_frames: int | None,
    fps: float,
    mask_threshold: float | None,
    frame_idx: int | None = None,
) -> Path:
    if sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")
    hand_dir, store_dir = _resolve_hand_dir(hand_path)
    task_root, session_root = _task_and_session_roots(hand_dir)
    hand = _hand_name(hand_dir)
    video_key = HAND_TO_VIDEO_KEY[hand]
    episode_index = _lumos_episode_index(task_root, session_root)
    episode_record = _load_episode_record(task_root, episode_index)
    info = _read_json(task_root / "meta" / "info.json")
    video = _video_path(task_root, info, episode_record, video_key)

    store = PointStore.open(store_dir, mask_threshold=mask_threshold)
    points = store.points
    has_conf_224 = "conf_224" in store.metadata.files
    conf_prob = confidence_to_probability(np.asarray(store.metadata["conf_224"], dtype=np.float32)) if has_conf_224 else None
    valid_mask = np.asarray(store.valid_mask, dtype=bool)
    episode_frames = np.asarray(store.metadata["episode_frame_indices"], dtype=np.int64)
    downsample_points, downsample_conf, downsample_mapping = _load_downsample(hand_dir)
    has_downsample_mapping = downsample_mapping is not None
    has_downsample_res_npz = downsample_points is not None and downsample_conf is not None

    output_dir = output_root / _safe_name(hand_dir.name)
    ply_dir = output_dir / "ply"
    downsample_ply_dir = output_dir / "downsample_ply"
    output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_output_dir(output_dir)
    sample_refs = _sample_frame_refs(
        store=store,
        episode_frames=episode_frames,
        downsample_mapping=downsample_mapping,
        sample_stride=sample_stride,
        max_frames=max_frames,
        frame_idx=frame_idx,
    )
    sample_rows = [row_idx for _downsample_idx, row_idx in sample_refs]

    depth_values = np.asarray(points[sample_rows, ..., 2], dtype=np.float32)
    valid_values = valid_mask[sample_rows] & np.isfinite(depth_values) & (depth_values > 0)
    if valid_values.any():
        depth_vmin = float(np.percentile(depth_values[valid_values], 1))
        depth_vmax = float(np.percentile(depth_values[valid_values], 99))
    else:
        depth_vmin, depth_vmax = 0.0, 1.0

    first_points = np.asarray(points[sample_rows[0]], dtype=np.float32)
    h, w = int(first_points.shape[0]), int(first_points.shape[1])
    video_out = output_dir / "rgb_depth_conf.mp4"
    writer = imageio.get_writer(video_out, fps=float(fps), codec="libx264", macro_block_size=1)
    reader = _VideoFrameReader(video)
    video_start = int(round(float(episode_record[f"videos/{video_key}/from_timestamp"]) * reader.average_fps))

    frame_records: list[dict[str, Any]] = []
    try:
        for ordinal, (downsample_idx, row_idx) in enumerate(sample_refs):
            episode_frame = int(episode_frames[row_idx])
            video_frame = _video_frame_index(
                task_root,
                info,
                episode_record,
                video_key,
                episode_frame,
                video_fps=reader.average_fps,
            )
            rgb = reader.read(video_frame)
            rgb_small = _resize_rgb(rgb, (h, w)).astype(np.uint8)

            frame_points = np.asarray(points[row_idx], dtype=np.float32)
            frame_valid = np.asarray(valid_mask[row_idx], dtype=bool)
            depth = frame_points[..., 2]
            depth_valid = frame_valid & np.isfinite(depth) & (depth > 0)

            depth_norm = _normalize_panel(depth, depth_valid, vmin=depth_vmin, vmax=depth_vmax)
            if conf_prob is not None:
                frame_conf = np.asarray(conf_prob[row_idx], dtype=np.float32)
                conf_norm = np.where(np.isfinite(frame_conf), np.clip(frame_conf, 0.0, 1.0), 0.0).astype(np.float32)
                conf_valid = np.isfinite(frame_conf)
                conf_label = "conf sigmoid"
            else:
                conf_norm = frame_valid.astype(np.float32)
                conf_valid = np.ones_like(frame_valid, dtype=bool)
                conf_label = "active mask"

            depth_rgb = _colorize(depth_norm, depth_valid, cv2.COLORMAP_TURBO)
            conf_rgb = _colorize(conf_norm, conf_valid, cv2.COLORMAP_VIRIDIS)
            panel = np.concatenate(
                [
                    _draw_label(rgb_small, f"rgb f={episode_frame}"),
                    _draw_label(depth_rgb, "depth z"),
                    _draw_label(conf_rgb, conf_label),
                ],
                axis=1,
            )
            writer.append_data(panel)

            ply_path = ply_dir / f"frame_{ordinal:06d}_episode_{episode_frame:06d}_video_{video_frame:06d}.ply"
            vertex_count = _write_ply(ply_path, frame_points, rgb_small, frame_valid)
            frame_record = {
                "row_index": int(row_idx),
                "downsample_local_frame_index": None if downsample_idx is None else int(downsample_idx),
                "downsample_raw_frame_index": None
                if downsample_idx is None or downsample_mapping is None
                else int(downsample_mapping[downsample_idx]["raw_frame_index"]),
                "episode_frame": episode_frame,
                "video_frame": video_frame,
                "ply_path": str(ply_path),
                "downsample_ply_path": None,
                "num_valid_points": int(vertex_count),
                "num_downsample_valid_points": None,
            }
            if has_downsample_res_npz and downsample_idx is not None:
                assert downsample_points is not None
                assert downsample_conf is not None
                down_points = np.asarray(downsample_points[downsample_idx], dtype=np.float32)
                down_conf = np.asarray(downsample_conf[downsample_idx], dtype=np.float32)
                down_valid = _point_valid_from_conf(
                    down_points,
                    down_conf,
                    float(store.meta.get("active_mask_threshold", 0.03) if mask_threshold is None else mask_threshold),
                )
                down_ply_path = downsample_ply_dir / (
                    f"frame_{ordinal:06d}_downsample_{downsample_idx:06d}_"
                    f"episode_{episode_frame:06d}_video_{video_frame:06d}.ply"
                )
                down_vertex_count = _write_ply(down_ply_path, down_points, rgb, down_valid)
                frame_record["downsample_ply_path"] = str(down_ply_path)
                frame_record["num_downsample_valid_points"] = int(down_vertex_count)
            frame_records.append(frame_record)
    finally:
        reader.close()
        writer.close()
        store.metadata.close()

    _write_json(
        output_dir / "metadata.json",
        {
            "input_path": str(hand_path),
            "hand_dir": str(hand_dir),
            "preprocessed_pointcloud_dir": str(store_dir),
            "task_root": str(task_root),
            "session_root": str(session_root),
            "episode_index": int(episode_index),
            "hand": hand,
            "video_key": video_key,
            "video_path": str(video),
            "video_start_frame": int(video_start),
            "sample_stride": int(sample_stride),
            "max_frames": None if max_frames is None else int(max_frames),
            "frame_idx": None if frame_idx is None else int(frame_idx),
            "fps": float(fps),
            "mask_threshold": mask_threshold,
            "has_conf_224": bool(has_conf_224),
            "num_store_rows": int(len(points)),
            "has_downsample_mapping": bool(has_downsample_mapping),
            "has_downsample_res_npz": bool(has_downsample_res_npz),
            "num_downsample_rows": None if downsample_mapping is None else int(len(downsample_mapping)),
            "num_sampled_frames": len(frame_records),
            "mp4_path": str(video_out),
            "frames": frame_records,
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check a Lumos preprocessed pointcloud hand directory by exporting an "
            "RGB/depth/conf MP4, sampled preprocessed local_points PLY files, "
            "and matching downsample/res.npz PLY files when the raw downsample npz still exists."
        )
    )
    parser.add_argument("path", type=Path, help="Lumos hand dir or its preprocessed_pointcloud dir.")
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-idx", type=int, default=None, help="Visualize only this preprocessed store row index.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--mask-threshold", type=float, default=None, help="Optional PointStore mask threshold override.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    out = export_check_visualization(
        args.path,
        output_root=args.output_root,
        sample_stride=args.sample_stride,
        max_frames=args.max_frames,
        fps=args.fps,
        mask_threshold=args.mask_threshold,
        frame_idx=args.frame_idx,
    )
    print(json.dumps({"output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
