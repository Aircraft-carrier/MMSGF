"""Preprocess lumos raw pointcloud clips into runtime pointcloud stores."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .pointcloud_store import (
    DEFAULT_MASK_THRESHOLDS,
    MASK_POLICY,
    STORE_FORMAT_V2,
    default_mask_threshold,
    generate_masks,
    mask_path_for_threshold,
    set_active_mask,
    valid_point_mask,
)
from .build_training_dataset import load_real_mot_training_selection


DEFAULT_TARGET_HW = (224, 224)
MIN_VALID_RAY_Z = 0.2
MAX_VALID_DEPTH_Z = 5.0


def _configure_worker_torch_threads() -> None:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


_CLIP_NAME_RE = re.compile(r"^clip_\d+_(\d+)_(\d+)$")


def _clip_episode_frame_bounds(clip_dir: Path) -> tuple[int, int]:
    match = _CLIP_NAME_RE.match(clip_dir.name)
    if match is None:
        raise ValueError(f"cannot parse pointcloud clip episode frame range from {clip_dir.name}")
    start, end = int(match.group(1)), int(match.group(2))
    if end < start:
        raise ValueError(f"invalid pointcloud clip episode frame range in {clip_dir.name}")
    return start, end


def _collect_clip_specs(hand_dir: Path) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for clip_id, clip_dir in enumerate(sorted((hand_dir / "clips").glob("clip_*"))):
        res_npz = clip_dir / "res.npz"
        # 用来后续做 scale alignment
        frame_mapping_csv = clip_dir / "frame_mapping.csv"
        if not res_npz.is_file():
            continue
        if not frame_mapping_csv.is_file():
            raise FileNotFoundError(frame_mapping_csv)
        # 按照正则规则解析出每个 clip 的起止 idx
        episode_start, episode_end = _clip_episode_frame_bounds(clip_dir)
        clips.append(
            {
                "clip_id": int(clip_id),
                "name": clip_dir.name,
                "clip_dir": str(clip_dir),
                "res_npz": str(res_npz),
                "frame_mapping_csv": str(frame_mapping_csv),
                "episode_start": int(episode_start),
                "episode_end": int(episode_end),
            }
        )
    if not clips:
        raise FileNotFoundError(f"no clips/*/res.npz under {hand_dir}")
    return clips


def _delete_original_res_npz(hand_dir: Path) -> None:
    for res_npz in sorted((hand_dir / "clips").glob("clip_*/res.npz")):
        res_npz.unlink(missing_ok=True)
    (hand_dir / "downsample" / "res.npz").unlink(missing_ok=True)


def _read_frame_mapping(frame_mapping_csv: Path) -> list[dict[str, int]]:
    with frame_mapping_csv.open("r", encoding="utf-8", newline="") as f:
        rows = [
            {
                "local_frame_index": int(row["local_frame_index"]),
                "aligned_frame_index": int(row["aligned_frame_index"]),
                "raw_frame_index": int(row["raw_frame_index"]),
            }
            for row in csv.DictReader(f)
        ]
    if not rows:
        raise ValueError(f"empty frame mapping: {frame_mapping_csv}")
    return sorted(rows, key=lambda row: int(row["local_frame_index"]))


def _validate_dense_local_mapping(frame_mapping_csv: Path, rows: list[dict[str, int]], frame_count: int) -> None:
    if len(rows) != int(frame_count):
        raise ValueError(f"{frame_mapping_csv} has {len(rows)} rows, expected {int(frame_count)}")
    for expected_local, row in enumerate(rows):
        # 确认 idx 是从 0 开始连续的
        local = int(row["local_frame_index"])
        if local != expected_local:
            raise ValueError(
                f"{frame_mapping_csv} local_frame_index must be dense 0..N-1; "
                f"row {expected_local} has local_frame_index={local}"
            )


def _squeeze_frame_axis(array: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    if name == "points" and (array.ndim != 4 or array.shape[-1] != 3):
        raise ValueError(f"expected local_points [1,F,H,W,3] or [F,H,W,3], got {array.shape}")
    if name == "conf":
        if array.ndim == 4 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 3:
            raise ValueError(f"expected conf [1,F,H,W], [1,F,H,W,1], or [F,H,W], got {array.shape}")
    if name == "rays" and (array.ndim != 4 or array.shape[-1] != 3):
        raise ValueError(f"expected rays [1,F,H,W,3] or [F,H,W,3], got {array.shape}")
    return array


def _protection_stats(valid_depth: torch.Tensor, valid_ray: torch.Tensor) -> dict[str, int]:
    valid = valid_depth & valid_ray
    return {
        "invalid_depth_points": int((~valid_depth).sum().item()),
        "invalid_ray_points": int((~valid_ray).sum().item()),
        "protected_points": int((~valid).sum().item()),
    }


def _resize_depth_rays_frame(
    points: np.ndarray,
    conf: np.ndarray,
    rays: np.ndarray,
    target_h_w: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    target_h, target_w = int(target_h_w[0]), int(target_h_w[1])
    points = np.nan_to_num(np.asarray(points, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(np.asarray(conf, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    rays = np.nan_to_num(np.asarray(rays, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if tuple(points.shape[:2]) == (target_h, target_w):
        points_t = torch.from_numpy(points)
        rays_t = torch.from_numpy(rays)
        valid_depth = (points_t[..., 2:3] > 0.0) & (points_t[..., 2:3] <= MAX_VALID_DEPTH_Z)
        valid_ray = rays_t[..., 2:3] > MIN_VALID_RAY_Z
        stats = _protection_stats(valid_depth, valid_ray)
        out_points = torch.where(valid_depth & valid_ray, points_t, torch.zeros_like(points_t)).numpy()
        return out_points.astype(np.float32, copy=False), conf, rays, stats

    depth = torch.from_numpy(points[..., 2:3]).permute(2, 0, 1).unsqueeze(0)
    depth = F.interpolate(depth, size=(target_h, target_w), mode="nearest").squeeze(0).permute(1, 2, 0)
    valid_depth = (depth > 0.0) & (depth <= MAX_VALID_DEPTH_Z)
    rays_t = torch.from_numpy(rays).permute(2, 0, 1).unsqueeze(0)
    rays_t = F.interpolate(rays_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    rays_t = F.normalize(rays_t.squeeze(0).permute(1, 2, 0), dim=-1)
    ray_z = rays_t[..., 2:3]
    valid_ray = ray_z > MIN_VALID_RAY_Z
    xy = rays_t[..., :2] / ray_z.clamp_min(MIN_VALID_RAY_Z)
    points_t = torch.cat([xy * depth, depth], dim=-1)
    stats = _protection_stats(valid_depth, valid_ray)
    points_t = torch.where(valid_depth & valid_ray, points_t, torch.zeros_like(points_t))
    out_rays = rays_t.numpy().astype(np.float32)
    out_points = np.nan_to_num(points_t.numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    conf_t = torch.from_numpy(conf).unsqueeze(0).unsqueeze(0)
    out_conf = F.interpolate(conf_t, size=(target_h, target_w), mode="nearest").squeeze(0).squeeze(0).numpy()
    return out_points, out_conf, out_rays, stats


def _load_downsample_points(hand_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    downsample_npz = hand_dir / "downsample" / "res.npz"
    if not downsample_npz.is_file():
        raise FileNotFoundError(downsample_npz)
    with np.load(downsample_npz, allow_pickle=False) as data:
        for key in ("local_points", "conf", "rays"):
            if key not in data:
                raise KeyError(f"{downsample_npz} missing required key: {key}")
        points = _squeeze_frame_axis(np.asarray(data["local_points"]), name="points")
        conf = _squeeze_frame_axis(np.asarray(data["conf"]), name="conf")
        rays = _squeeze_frame_axis(np.asarray(data["rays"]), name="rays")
    return points.astype(np.float32), conf.astype(np.float32), rays.astype(np.float32)


def _frame_radius_p90(points: np.ndarray, mask: np.ndarray) -> float:
    if not bool(mask.any()):
        return 0.0
    radius = np.linalg.norm(np.asarray(points, dtype=np.float32)[mask], axis=-1)
    radius = radius[np.isfinite(radius)]
    if radius.size == 0:
        return 0.0
    return float(np.percentile(radius, 90))


def _valid_mask(points: np.ndarray, conf: np.ndarray | None, threshold: float) -> np.ndarray:
    return valid_point_mask(points, conf, threshold, source_dataset="lumos_lerobot")


def _scale_filter(points: np.ndarray, conf: np.ndarray | None, source_dataset: str) -> np.ndarray:
    threshold = default_mask_threshold(source_dataset)
    return _valid_mask(points, conf, threshold)


def _frame_scales_for_rows(
    hand_dir: Path,
    clip_specs: list[dict[str, Any]],
    resized_points: list[np.ndarray],
    resized_conf: list[np.ndarray],
    row_clip_ids: list[int],
    row_clip_local_indices: list[int],
    source_dataset: str,
    target_h_w: tuple[int, int],
) -> tuple[list[float], dict[str, Any]]:
    # 和之前 clip 类似的逻辑，读取 npz 和 mapping，校验二者长度一致
    down_points, down_conf, down_rays = _load_downsample_points(hand_dir)
    down_mapping_csv = hand_dir / "downsample" / "frame_mapping.csv"
    if not down_mapping_csv.is_file():
        raise FileNotFoundError(down_mapping_csv)
    down_rows = _read_frame_mapping(down_mapping_csv)
    _validate_dense_local_mapping(down_mapping_csv, down_rows, int(down_points.shape[0]))

    down_by_aligned: dict[int, dict[str, int]] = {}
    for row in down_rows:
        aligned = int(row["aligned_frame_index"])
        if aligned in down_by_aligned:
            raise ValueError(f"{down_mapping_csv} duplicate downsample aligned_frame_index={aligned}")
        # 用 aligned_frame_index 作为 key，存储 downsample 内的 local-align-raw 对应关系
        down_by_aligned[aligned] = row

    # 现在各个 clip 的 pointcloud 已经被合并到一个里了，因此需要知道 episode 内的某一个 idx 对应原始哪个 clip 的哪一帧
    # 这里是一个反映射，clip_id + clip_local_idx => episode_idx
    row_by_clip_local: dict[tuple[int, int], int] = {}
    for row_idx, (clip_id, local_idx) in enumerate(zip(row_clip_ids, row_clip_local_indices, strict=True)):
        row_by_clip_local[(int(clip_id), int(local_idx))] = int(row_idx)

    clip_scale_by_id: dict[int, float] = {}
    clip_reports: list[dict[str, Any]] = []
    threshold = default_mask_threshold(source_dataset)

    # clip-level scale -> 这里认为一个 clip 内是对齐的，核心是需要对齐不同 clip
    for clip in clip_specs:
        clip_id = int(clip["clip_id"])
        clip_mapping_csv = Path(clip["frame_mapping_csv"])
        clip_rows = _read_frame_mapping(clip_mapping_csv)

        # 先保存能找到对应关系的所有帧上每一帧的 scale 比值
        candidate_scales: list[float] = []
        anchor_rows: list[dict[str, Any]] = []
        
        invalid_scale_anchors = 0

        # 找出当前 clip 内哪些帧出现在了 downsample 内，这些帧上计算 clip-level scale
        anchor_clip_rows = [row for row in clip_rows if int(row["aligned_frame_index"]) in down_by_aligned]
        matched_anchors = len(anchor_clip_rows)
        # 只需要遍历 anchor frames
        for clip_row in anchor_clip_rows:
            aligned = int(clip_row["aligned_frame_index"])
            down_row = down_by_aligned[aligned]
            clip_raw = int(clip_row["raw_frame_index"])
            down_raw = int(down_row["raw_frame_index"])
            # 二次校验，确保是同一帧
            if clip_raw != down_raw:
                raise ValueError(
                    f"raw_frame_index mismatch for {clip['name']} aligned_frame_index={aligned}: "
                    f"clip raw_frame_index={clip_raw}, downsample raw_frame_index={down_raw}"
                )

            clip_local_idx = int(clip_row["local_frame_index"])
            # 已有的未做 scale 的 mix-clip pointcloud array 里的 idx
            row_idx = row_by_clip_local.get((clip_id, clip_local_idx))
            if row_idx is None:
                raise ValueError(f"{clip_mapping_csv} local_frame_index={clip_local_idx} has no preprocessed row")
            # downsample 里对应的同一帧的 idx
            downsample_idx = int(down_row["local_frame_index"])
            # downsample 也 resize 到相同分辨率
            down_resized, down_conf_resized, _down_rays_resized, _down_protection = _resize_depth_rays_frame(
                down_points[downsample_idx],
                down_conf[downsample_idx],
                down_rays[downsample_idx],
                target_h_w,
            )
            points = resized_points[row_idx]
            # 过滤 vaild pointcloud，这里已经 apply 了 lumos 默认的 0.03 conf
            mask_src = _scale_filter(points, resized_conf[row_idx], source_dataset)
            mask_down = _scale_filter(down_resized, down_conf_resized, source_dataset)
            # 获取点云半径，这里用了 0.9 分位数
            src_radius = _frame_radius_p90(points, mask_src)
            down_radius = _frame_radius_p90(down_resized, mask_down)
            if src_radius <= 1e-6 or down_radius <= 1e-6:
                invalid_scale_anchors += 1
                continue
            # 计算 scale factor
            scale = float(down_radius / src_radius)
            candidate_scales.append(scale)
            anchor_rows.append(
                {
                    "aligned_frame_index": aligned,
                    "clip_local_frame_index": clip_local_idx,
                    "downsample_local_frame_index": downsample_idx,
                    "raw_frame_index": clip_raw,
                    "scale": scale,
                    "clip_radius_p90": float(src_radius),
                    "downsample_radius_p90": float(down_radius),
                }
            )

        if matched_anchors == 0:
            raise ValueError(
                f"{clip['name']} has no downsample anchors under {hand_dir}; "
                f"clip mapping={clip_mapping_csv}, downsample mapping={down_mapping_csv}"
            )
        if not candidate_scales:
            raise ValueError(f"{clip['name']} has no valid scale anchors under {hand_dir}")

        scale_values = np.asarray(candidate_scales, dtype=np.float32)
        # 由所有 anchor frame scale 计算 clip scale，用中位数
        scale = float(np.median(scale_values))
        clip_scale_by_id[clip_id] = scale
        clip_reports.append(
            {
                "clip_id": clip_id,
                "clip_name": clip["name"],
                "frame_mapping_csv": str(clip_mapping_csv),
                "num_mapping_rows": len(clip_rows),
                "num_matched_anchors": int(matched_anchors),
                "num_invalid_scale_anchors": int(invalid_scale_anchors),
                "num_valid_scale_anchors": int(scale_values.size),
                "scale": scale,
                "candidate_scale_min": float(np.min(scale_values)),
                "candidate_scale_p25": float(np.percentile(scale_values, 25)),
                "candidate_scale_median": float(np.median(scale_values)),
                "candidate_scale_p75": float(np.percentile(scale_values, 75)),
                "candidate_scale_max": float(np.max(scale_values)),
                "scale_filter_conf_threshold": float(threshold),
                "anchors": anchor_rows,
            }
        )

    scales = [float(clip_scale_by_id[int(clip_id)]) for clip_id in row_clip_ids]
    return scales, {
        "source_hand_dir": str(hand_dir),
        "downsample_frame_mapping_csv": str(down_mapping_csv),
        "num_downsample_mapping_rows": len(down_rows),
        "num_clips": len(clip_reports),
        "clips": clip_reports,
    }


def build_hand_store(
    hand_dir: str | Path,
    *,
    source_dataset: str,
    target_h_w: tuple[int, int] = DEFAULT_TARGET_HW,
    mask_thresholds: Iterable[float] = DEFAULT_MASK_THRESHOLDS,
    overwrite: bool = False,
    save_conf: bool = False,
    save_rays: bool = False,
    del_ori_npz: bool = True,
) -> Path:
    if str(source_dataset) != "lumos_lerobot":
        raise ValueError(f"preprocess_lumos_pointcloud only builds lumos stores, got {source_dataset}")
    # 左右 hand 是独立处理的
    hand_dir = Path(hand_dir)

    out_dir = hand_dir / "preprocessed_pointcloud"
    if out_dir.exists():
        completed = (
            (out_dir / "_SUCCESS").is_file()
            and (out_dir / "protection_report.json").is_file()
            and (out_dir / "scale_alignment_report.json").is_file()
        )
        if not overwrite and completed:
            with np.load(out_dir / "metadata.npz", allow_pickle=False) as metadata:
                has_requested_optional = (
                    (not save_conf or "conf_224" in metadata.files)
                    and (not save_rays or "rays_224" in metadata.files)
                )
            if has_requested_optional:
                return out_dir

        clip_dirs = sorted(path for path in (hand_dir / "clips").glob("clip_*") if path.is_dir())
        missing_raw_res = [path / "res.npz" for path in clip_dirs if not (path / "res.npz").is_file()]
        downsample_res = hand_dir / "downsample" / "res.npz"
        if not downsample_res.is_file():
            missing_raw_res.append(downsample_res)
        if not clip_dirs or missing_raw_res:
            missing = [str(path) for path in missing_raw_res]
            if not clip_dirs:
                missing.append(str(hand_dir / "clips" / "clip_*/res.npz"))
            raise FileNotFoundError(
                f"Cannot rebuild {out_dir}: required raw res.npz files are missing; "
                f"existing store was preserved. Missing: {missing}"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    # preprocess 所有 clip
    clip_specs = _collect_clip_specs(hand_dir)
    # 存储所有 clip 的 points/conf/rays/camera_poses，合并为一个，方便后续训练读取
    points_rows: list[np.ndarray] = []
    conf_rows: list[np.ndarray] = []
    rays_rows: list[np.ndarray] = []
    camera_poses_rows: list[np.ndarray] = []
    episode_frame_indices: list[int] = []
    clip_ids: list[int] = []
    clip_local_indices: list[int] = []
    clip_table_rows: list[dict[str, Any]] = []
    protection_events: list[dict[str, Any]] = []

    for clip in clip_specs:
        frame_mapping_csv = Path(clip["frame_mapping_csv"])
        frame_mapping_rows = _read_frame_mapping(frame_mapping_csv)
        # 读取原始 clip 数据
        with np.load(clip["res_npz"], allow_pickle=False) as data:
            for key in ("local_points", "conf", "rays", "camera_poses"):
                if key not in data:
                    raise KeyError(f"{clip['res_npz']} missing required key: {key}")
            local_points = _squeeze_frame_axis(np.asarray(data["local_points"]), name="points").astype(np.float32)
            conf = _squeeze_frame_axis(np.asarray(data["conf"]), name="conf").astype(np.float32)
            rays = _squeeze_frame_axis(np.asarray(data["rays"]), name="rays").astype(np.float32)
            camera_poses = np.asarray(data["camera_poses"], dtype=np.float32)
            if camera_poses.ndim >= 1 and camera_poses.shape[0] == 1:
                camera_poses = camera_poses[0]
            if camera_poses.ndim != 3 or camera_poses.shape[1:] != (4, 4):
                raise ValueError(f"expected camera_poses [1,F,4,4] or [F,4,4], got {camera_poses.shape}")
            if int(camera_poses.shape[0]) != int(local_points.shape[0]):
                raise ValueError(
                    f"{clip['res_npz']} has {camera_poses.shape[0]} camera poses, "
                    f"expected {local_points.shape[0]} from local_points"
                )
        # 校验 frame_mapping.csv 行数等于 local_points.shape[0]，且 local_frame_index 是 0..F-1
        _validate_dense_local_mapping(frame_mapping_csv, frame_mapping_rows, int(local_points.shape[0]))
        clip_table_rows.append(
            {
                "clip_id": int(clip["clip_id"]),        # 标志是哪个 clip
                "name": clip["name"],
                "res_npz": clip["res_npz"],
                "frame_mapping_csv": clip["frame_mapping_csv"],
                "episode_start": int(clip["episode_start"]),
                "episode_end": int(clip["episode_end"]),
                "num_npz_rows": int(local_points.shape[0]),
            }
        )
        # 长度校验 -> frame count must match frame range encoded in clip directory name
        expected_rows = int(clip["episode_end"]) - int(clip["episode_start"]) + 1
        if int(local_points.shape[0]) != expected_rows:
            raise ValueError(
                f"{clip['res_npz']} has {local_points.shape[0]} rows, expected {expected_rows} from {clip['name']}"
        )
        # 逐帧resize + 类似 unik3d 中 rays/depth 的 max 过滤
        for local_idx in range(int(local_points.shape[0])):
            resized_points, resized_conf, resized_rays, protection = _resize_depth_rays_frame(
                local_points[local_idx],
                conf[local_idx],
                rays[local_idx],
                target_h_w,
            )
            # 直接读 mapping csv，获取真实的 episode_idx
            frame_mapping_row = frame_mapping_rows[local_idx]
            episode_frame = int(frame_mapping_row["aligned_frame_index"])
            if int(protection["protected_points"]) > 0:
                protection_events.append(
                    {
                        "clip_id": int(clip["clip_id"]),
                        "clip_name": clip["name"],
                        "res_npz": clip["res_npz"],
                        "local_idx": int(local_idx),
                        "episode_frame": int(episode_frame),
                        "invalid_depth_points": int(protection["invalid_depth_points"]),
                        "invalid_ray_points": int(protection["invalid_ray_points"]),
                        "protected_points": int(protection["protected_points"]),
                    }
                )
            points_rows.append(resized_points)
            conf_rows.append(resized_conf.astype(np.float32))
            rays_rows.append(resized_rays.astype(np.float32))
            camera_poses_rows.append(np.asarray(camera_poses[local_idx], dtype=np.float32).copy())
            episode_frame_indices.append(episode_frame)
            clip_ids.append(int(clip["clip_id"]))
            clip_local_indices.append(int(local_idx))

    order = np.argsort(np.asarray(episode_frame_indices, dtype=np.int64), kind="stable")
    points_rows = [points_rows[int(idx)] for idx in order]
    conf_rows = [conf_rows[int(idx)] for idx in order]
    rays_rows = [rays_rows[int(idx)] for idx in order]
    camera_poses_rows = [camera_poses_rows[int(idx)] for idx in order]
    episode_frames = np.asarray([episode_frame_indices[int(idx)] for idx in order], dtype=np.int32)
    clip_ids_arr = np.asarray([clip_ids[int(idx)] for idx in order], dtype=np.int32)        # 标记某一帧来源于哪个 clip
    clip_local_indices_arr = np.asarray([clip_local_indices[int(idx)] for idx in order], dtype=np.int32)        # 标记某一帧在原始 clip 内的是第几帧
    # episode idx 应该是唯一的，不能重复
    if len(set(int(value) for value in episode_frames)) != int(episode_frames.shape[0]):
        raise ValueError(f"duplicate episode frame values under {hand_dir}")
    # NOTE 确保 lerobot video idx 和 pointcloud idx 之间是对齐的
    expected_episode_frames = np.arange(int(episode_frames.shape[0]), dtype=np.int32)
    if not np.array_equal(episode_frames, expected_episode_frames):
        raise ValueError(
            f"{hand_dir} does not contain dense episode-local pointcloud rows 0..N-1; "
            f"min={int(episode_frames.min()) if episode_frames.size else None}, "
            f"max={int(episode_frames.max()) if episode_frames.size else None}, rows={int(episode_frames.shape[0])}"
        )

    # 根据 downsample 计算一个scale，对各个 clip 进行 scale 缩放
    scale_values, scale_alignment_report = _frame_scales_for_rows(
        hand_dir,
        clip_specs,
        points_rows,
        conf_rows,
        [int(value) for value in clip_ids_arr.tolist()],
        [int(value) for value in clip_local_indices_arr.tolist()],
        source_dataset,
        target_h_w,
    )
    scales = np.asarray(scale_values, dtype=np.float32)
    points_arr = np.stack(points_rows, axis=0).astype(np.float32)
    points_arr *= scales[:, None, None, None]
    poses_arr = np.stack(camera_poses_rows, axis=0).astype(np.float32)
    poses_arr[:, :3, 3] *= scales[:, None]

    np.save(out_dir / "local_points_224.npy", points_arr.astype(np.float16))
    conf_arr = np.stack(conf_rows, axis=0).astype(np.float32)
    metadata_payload: dict[str, np.ndarray] = {
        "camera_poses": poses_arr,
        "episode_frame_indices": episode_frames,
        "clip_ids": clip_ids_arr,
        "frame_scales": scales,
    }
    if save_conf:
        metadata_payload["conf_224"] = conf_arr.astype(np.float16)
    if save_rays:
        metadata_payload["rays_224"] = np.stack(rays_rows, axis=0).astype(np.float16)
    np.savez_compressed(out_dir / "metadata.npz", **metadata_payload)
    _write_jsonl(out_dir / "clip_table.jsonl", clip_table_rows)
    _write_json(
        out_dir / "alignment_report.json",
        {
            "num_rows": int(episode_frames.shape[0]),
            "episode_frame_min": int(episode_frames.min()) if episode_frames.size else None,
            "episode_frame_max": int(episode_frames.max()) if episode_frames.size else None,
            "num_clips": len(clip_table_rows),
        },
    )
    _write_json(
        out_dir / "protection_report.json",
        {
            "source_hand_dir": str(hand_dir),
            "min_valid_ray_z": float(MIN_VALID_RAY_Z),
            "max_valid_depth_z": float(MAX_VALID_DEPTH_Z),
            "num_events": len(protection_events),
            "total_invalid_depth_points": int(sum(int(row["invalid_depth_points"]) for row in protection_events)),
            "total_invalid_ray_points": int(sum(int(row["invalid_ray_points"]) for row in protection_events)),
            "total_protected_points": int(sum(int(row["protected_points"]) for row in protection_events)),
            "events": protection_events,
        },
    )
    _write_json(out_dir / "scale_alignment_report.json", scale_alignment_report)
    thresholds = sorted({float(value) for value in mask_thresholds} | {0.0, default_mask_threshold(source_dataset)})
    generate_masks(out_dir, thresholds, source_dataset=source_dataset, conf=conf_arr)
    default_threshold = default_mask_threshold(source_dataset)
    _write_json(
        out_dir / "meta.json",
        {
            "format": STORE_FORMAT_V2,
            "source_hand_dir": str(hand_dir),
            "source_dataset": str(source_dataset),
            "target_point_hw": [int(target_h_w[0]), int(target_h_w[1])],
            "points": "local_points_224.npy",
            "metadata": "metadata.npz",
            "active_mask": str(mask_path_for_threshold(out_dir, default_threshold).relative_to(out_dir)),
            "active_mask_threshold": float(default_threshold),
            "mask_policy": MASK_POLICY,
            "mask_source_dataset": str(source_dataset),
            "mask_thresholds": thresholds,
            "saved_conf_224": bool(save_conf),
            "saved_rays_224": bool(save_rays),
            "num_rows": int(episode_frames.shape[0]),
        },
    )
    (out_dir / "_SUCCESS").write_text("", encoding="utf-8")
    if del_ori_npz:
        _delete_original_res_npz(hand_dir)
    return out_dir


def _source_dataset_for_path(path: Path) -> str:
    parts = set(path.parts)
    if "lumos_lerobot" in parts:
        return "lumos_lerobot"
    raise ValueError(f"preprocess_lumos_pointcloud only supports lumos_lerobot paths, got {path}")


def _hand_dirs(session_root: Path) -> dict[str, Path] | None:
    left = sorted(path for path in session_root.iterdir() if path.is_dir() and path.name.startswith("left_hand"))
    right = sorted(path for path in session_root.iterdir() if path.is_dir() and path.name.startswith("right_hand"))
    if not left or not right:
        return None
    if not (left[0] / "clips").is_dir() or not (right[0] / "clips").is_dir():
        return None
    return {"left": left[0], "right": right[0]}


def _build_hand_store_task(task: dict[str, Any]) -> dict[str, Any]:
    _configure_worker_torch_threads()
    start = time.perf_counter()
    row = {
        key: value
        for key, value in task.items()
        if key not in {"target_h_w", "overwrite", "save_conf", "save_rays", "del_ori_npz"}
    }
    try:
        store_dir = build_hand_store(
            task["hand_dir"],
            source_dataset=str(task["source_dataset"]),
            target_h_w=tuple(task["target_h_w"]),
            overwrite=bool(task["overwrite"]),
            save_conf=bool(task.get("save_conf", False)),
            save_rays=bool(task.get("save_rays", False)),
            del_ori_npz=bool(task.get("del_ori_npz", True)),
        )
        protection_report_path = store_dir / "protection_report.json"
        protection_report = json.loads(protection_report_path.read_text(encoding="utf-8"))
        return {
            **row,
            "ok": True,
            "preprocessed_pointcloud_dir": str(store_dir),
            "protection_report_path": str(protection_report_path),
            "num_protection_events": int(protection_report["num_events"]),
            "total_protected_points": int(protection_report["total_protected_points"]),
            "protection_events": protection_report["events"],
            "seconds": time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            **row,
            "ok": False,
            "error_source": "_build_hand_store_task",
            "error_stage": "build_hand_store",
            "reason": type(exc).__name__,
            "detail": str(exc),
            "seconds": time.perf_counter() - start,
        }


def _run_hand_store_tasks(tasks: list[dict[str, Any]], *, num_workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total = len(tasks)
    workers = max(1, int(num_workers))
    print(f"[build] hand_tasks={total} workers={workers}", flush=True)
    if total == 0:
        return rows, skipped

    def consume(result: dict[str, Any], done: int) -> None:
        if bool(result.pop("ok")):
            rows.append(result)
        else:
            skipped.append(result)
        if done == total or done % 10 == 0:
            last = Path(result.get("hand_dir", "")).name
            print(
                f"[build] done {done}/{total} ok={len(rows)} errors={len(skipped)} last={last} seconds={result.get('seconds', 0.0):.2f}",
                flush=True,
            )

    if workers == 1:
        for done, task in enumerate(tasks, start=1):
            consume(_build_hand_store_task(task), done)
    else:
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
            futures = [executor.submit(_build_hand_store_task, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                consume(future.result(), done)

    rows.sort(key=lambda row: (str(row.get("session_root", "")), str(row.get("hand", ""))))
    skipped.sort(key=lambda row: (str(row.get("session_root", "")), str(row.get("hand", "")), str(row.get("reason", ""))))
    return rows, skipped


def _selected_lumos_sessions(selection_root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    # 读取 step 2 选出来的 manifest
    loaded_selection = load_real_mot_training_selection(selection_root)
    sessions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    for row in loaded_selection.pointcloud_rows:
        if str(row["source_dataset"]) != "lumos_lerobot":
            continue
        task_root = Path(row["source_lerobot_task_dir"])
        session_root = Path(row["pointcloud_session_dir"])
        session_key = str(session_root)
        if session_key in seen_sessions:
            continue
        seen_sessions.add(session_key)
        if not session_root.is_dir():
            skipped.append(
                {
                    "task_uid": row.get("task_uid"),
                    "episode_index": row.get("episode_index"),
                    "session_root": str(session_root),
                    "reason": "missing_session_dir",
                }
            )
            continue
        hand_dirs = _hand_dirs(session_root)
        if hand_dirs is None:
            skipped.append(
                {
                    "task_uid": row.get("task_uid"),
                    "episode_index": row.get("episode_index"),
                    "session_root": str(session_root),
                    "reason": "missing_hand_clips",
                }
            )
            continue
        sessions.append(
            {
                "task_uid": row["task_uid"],
                "task_root": str(task_root),
                "source_dataset": row["source_dataset"],
                "episode_index": int(row["episode_index"]),
                "session": str(session_root.relative_to(task_root / "pointcloud")),
                "session_root": str(session_root),
                "mapping_policy": row.get("pointcloud_mapping_policy"),
                "left_hand_dir": str(hand_dirs["left"]),
                "right_hand_dir": str(hand_dirs["right"]),
            }
        )
    sessions.sort(key=lambda item: (str(item["task_uid"]), int(item["episode_index"]), str(item["session_root"])))
    skipped.sort(key=lambda item: (str(item.get("task_uid", "")), str(item.get("session_root", "")), str(item.get("reason", ""))))
    return sessions, skipped, loaded_selection.config


def build_from_root(
    *,
    selection_root: str | Path,
    target_h_w: tuple[int, int] = DEFAULT_TARGET_HW,
    overwrite: bool = False,
    save_conf: bool = False,
    save_rays: bool = False,
    del_ori_npz: bool = True,
    report_path: str | Path | None = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    selection_root = Path(selection_root).resolve()
    # 筛选存在 left/hand 且有 clips 的作为待处理 pool
    sessions, discovery_skipped, selection_config = _selected_lumos_sessions(selection_root)
    tasks = []
    skipped = []
    missing_downsample_hands = []
    # 遍历所有 session，一个 session 其实就是一个 episode
    for session_info in sessions:
        session_root = Path(session_info["session_root"])
        source_dataset = str(session_info["source_dataset"])
        hand_dirs = _hand_dirs(session_root)
        if hand_dirs is None:
            skipped.append({"session_root": str(session_root), "reason": "missing_hands"})
            continue
        # 左/右各自分别处理
        for hand, hand_dir in sorted(hand_dirs.items()):
            downsample_res_npz = hand_dir / "downsample" / "res.npz"
            if not downsample_res_npz.is_file():
                missing_downsample_hands.append(
                    {
                        "session_root": str(session_root),
                        "hand": hand,
                        "hand_dir": str(hand_dir),
                        "source_dataset": source_dataset,
                        "downsample_res_npz": str(downsample_res_npz),
                    }
                )
            tasks.append(
                {
                    "session_root": str(session_root),
                    "task_uid": session_info.get("task_uid"),
                    "episode_index": session_info.get("episode_index"),
                    "mapping_policy": session_info.get("mapping_policy"),
                    "hand": hand,
                    "hand_dir": str(hand_dir),
                    "source_dataset": source_dataset,
                    "target_h_w": [int(target_h_w[0]), int(target_h_w[1])],
                    "overwrite": bool(overwrite),
                    "save_conf": bool(save_conf),
                    "save_rays": bool(save_rays),
                    "del_ori_npz": bool(del_ori_npz),
                }
            )
    rows, exec_errors = _run_hand_store_tasks(tasks, num_workers=num_workers)
    skipped.extend(exec_errors)
    protection_events: list[dict[str, Any]] = []
    total_protected_points = 0
    for row in rows:
        row_events = row.pop("protection_events", [])
        total_protected_points += int(row.get("total_protected_points", 0))
        for event in row_events:
            protection_events.append(
                {
                    "task_uid": row.get("task_uid"),
                    "episode_index": row.get("episode_index"),
                    "session_root": row.get("session_root"),
                    "mapping_policy": row.get("mapping_policy"),
                    "hand": row.get("hand"),
                    "hand_dir": row.get("hand_dir"),
                    "preprocessed_pointcloud_dir": row.get("preprocessed_pointcloud_dir"),
                    **event,
                }
            )
    report = {
        "selection_root": str(selection_root),
        "base_root": str(selection_config["base_root"]),
        "task_source": "training_selection",
        "num_workers": max(1, int(num_workers)),
        "save_conf": bool(save_conf),
        "save_rays": bool(save_rays),
        "del_ori_npz": bool(del_ori_npz),
        "num_eligible_tasks": len({str(row["task_root"]) for row in sessions}),
        "num_eligible_sessions": len(sessions),
        "num_selected_sessions": len(sessions),
        "num_written_hand_stores": len(rows),
        "num_failed_hand_stores": len(exec_errors),
        "num_discovery_skipped_sessions": len(discovery_skipped),
        "num_missing_downsample_hands": len(missing_downsample_hands),
        "num_protection_events": len(protection_events),
        "total_protected_points": int(total_protected_points),
        "selected": rows,
        "skipped": discovery_skipped + skipped,
        "missing_downsample_hands": missing_downsample_hands,
        "protection_events": protection_events,
    }
    if report_path is not None:
        report_path = Path(report_path)
        _write_json(report_path, report)
        _write_json(report_path.with_name("missing_downsample_hands.json"), missing_downsample_hands)
        _write_jsonl(report_path.with_name("lumos_pointcloud_protection_events.jsonl"), protection_events)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="convert raw clips into preprocessed_pointcloud stores")
    build.add_argument("--selection-root", type=Path, default=None)
    build.add_argument("--hand-dir", type=Path, default=None)
    build.add_argument("--target-h", type=int, default=224)
    build.add_argument("--target-w", type=int, default=224)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--save_conf", "--save-conf", action="store_true")
    build.add_argument("--save_rays", "--save-rays", action="store_true")
    build.add_argument(
        "--del_ori_npz",
        "--del-ori-npz",
        dest="del_ori_npz",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    build.add_argument("--report-path", type=Path, default=None)
    build.add_argument("--num-workers", type=int, default=min(32, os.cpu_count() or 1))

    active = sub.add_parser("set-active-mask", help="set meta.json active mask")
    active.add_argument("store_dir", type=Path)
    active.add_argument("--threshold", type=float, required=True)

    args = parser.parse_args()
    if args.cmd == "build":
        target_h_w = (int(args.target_h), int(args.target_w))
        if args.hand_dir is not None:
            source_dataset = _source_dataset_for_path(args.hand_dir)
            out = build_hand_store(
                args.hand_dir,
                source_dataset=source_dataset,
                target_h_w=target_h_w,
                overwrite=args.overwrite,
                save_conf=args.save_conf,
                save_rays=args.save_rays,
                del_ori_npz=args.del_ori_npz,
            )
            # print(json.dumps({"preprocessed_pointcloud_dir": str(out)}, indent=2))
            print('Success')
        else:
            if args.selection_root is None:
                parser.error("preprocess_lumos_pointcloud build requires --selection-root unless --hand-dir is provided")
            report = build_from_root(
                selection_root=args.selection_root,
                target_h_w=target_h_w,
                overwrite=args.overwrite,
                save_conf=args.save_conf,
                save_rays=args.save_rays,
                del_ori_npz=args.del_ori_npz,
                report_path=args.report_path,
                num_workers=args.num_workers,
            )
            print(json.dumps(report, indent=2))
    elif args.cmd == "set-active-mask":
        path = set_active_mask(args.store_dir, threshold=args.threshold)
        print(json.dumps({"active_mask": str(path)}, indent=2))


if __name__ == "__main__":
    _main()
