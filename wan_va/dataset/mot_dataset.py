"""Runtime MOT dataset for UMI 3DWA training/inference."""

from __future__ import annotations

import json
import os
import random
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, get_worker_info
from torchcodec.decoders import VideoDecoder

from .action_cache import _cache_index_key, load_action_cache_index


DEFAULT_UMI_DATA_ROOT = Path("/team_data/umi_data")
DEFAULT_REAL_BASE_ROOT = Path("data/umi_mot_full_data_base")
DEFAULT_REAL_TRAIN_ROOT = Path("data/umi_mot_real_train_stride4_rel20")
DEFAULT_VIDEO_KEYS = ("observation.images.robot_0", "observation.images.robot_1")
WAN_VAE_TEMPORAL_FACTOR = 4
MOT_HISTORY_CHUNKS = 1
MOT_TARGET_CHUNKS = 1
MOT_DEFAULT_ACTION_CHUNK_SIZE = 48
MOT_DEFAULT_VIDEO_DOWNSAMPLE_RATIO = 4
MOT_MAX_RIGHT_PADDING_RAW_STEPS = 16
MOT_ACTION_CHUNKS = MOT_HISTORY_CHUNKS + MOT_TARGET_CHUNKS
MOT_DEFAULT_VIDEO_DECODER_CACHE_SIZE = 256
MOT_DEFAULT_ACTION_CACHE_SIZE = 512
MOT_SUPPORTED_VIEW_COUNTS = (2, 3)
MOT_RUNTIME_CACHE_STAT_FIELDS = ("worker_id", "worker_pid") + tuple(
    f"{prefix}_{field}"
    for prefix in ("video", "action")
    for field in ("size", "max_size", "hits", "misses", "puts", "evictions")
)
_MOT_RUNTIME_CACHE_STAT_STRUCT = struct.Struct(f"<{len(MOT_RUNTIME_CACHE_STAT_FIELDS)}q")


def encode_mot_runtime_cache_stats(stats: dict[str, int]) -> bytes:
    return _MOT_RUNTIME_CACHE_STAT_STRUCT.pack(*(int(stats[key]) for key in MOT_RUNTIME_CACHE_STAT_FIELDS))


def decode_mot_runtime_cache_stats(payload: bytes) -> dict[str, int]:
    if not isinstance(payload, bytes) or len(payload) != _MOT_RUNTIME_CACHE_STAT_STRUCT.size:
        raise ValueError("invalid MOT runtime cache telemetry payload")
    return dict(zip(MOT_RUNTIME_CACHE_STAT_FIELDS, _MOT_RUNTIME_CACHE_STAT_STRUCT.unpack(payload)))


class _BoundedLRUCache:
    def __init__(
        self,
        max_size: int,
        *,
        name: str,
        on_evict: Callable[[Any], None] | None = None,
    ):
        self.max_size = int(max_size)
        if self.max_size <= 0:
            raise ValueError(f"{name} must be positive, got {self.max_size}")
        self._items: OrderedDict[Any, Any] = OrderedDict()
        self._on_evict = on_evict
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: Any) -> Any | None:
        try:
            value = self._items[key]
        except KeyError:
            self._misses += 1
            return None
        self._hits += 1
        self._items.move_to_end(key)
        return value

    def put(self, key: Any, value: Any) -> None:
        self._puts += 1
        previous = self._items.pop(key, None)
        if previous is not None and previous is not value:
            self._evict(previous)
        self._items[key] = value
        if len(self._items) > self.max_size:
            _, oldest = self._items.popitem(last=False)
            self._evict(oldest)

    def clear(self) -> None:
        while self._items:
            _, value = self._items.popitem(last=False)
            self._evict(value)

    def _evict(self, value: Any) -> None:
        self._evictions += 1
        if self._on_evict is not None:
            self._on_evict(value)

    def stats(self) -> dict[str, int]:
        return {
            "size": len(self._items),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "puts": self._puts,
            "evictions": self._evictions,
        }


def _close_action_cache_entry(arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    for array in arrays:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None and not mmap.closed:
            mmap.close()


def mot_sampled_video_frames_per_action_chunk_per_view(
    action_chunk_size: int,
    video_downsample_ratio: int = MOT_DEFAULT_VIDEO_DOWNSAMPLE_RATIO,
) -> int:
    action_chunk_size = int(action_chunk_size)
    video_downsample_ratio = int(video_downsample_ratio)
    if action_chunk_size % video_downsample_ratio != 0:
        raise ValueError(
            f"action_chunk_size={action_chunk_size} must be divisible by video_downsample_ratio={video_downsample_ratio}"
        )
    return action_chunk_size // video_downsample_ratio + 1


def mot_latent_frames_per_action_chunk_per_view(
    action_chunk_size: int,
    video_downsample_ratio: int = MOT_DEFAULT_VIDEO_DOWNSAMPLE_RATIO,
    vae_temporal_factor: int = WAN_VAE_TEMPORAL_FACTOR,
) -> int:
    sampled_frames = mot_sampled_video_frames_per_action_chunk_per_view(action_chunk_size, video_downsample_ratio)
    vae_temporal_factor = int(vae_temporal_factor)

    if (sampled_frames - 1) % vae_temporal_factor != 0:
        raise ValueError(
            f"sampled video frames per view must be 1 + vae_temporal_factor*k, got {sampled_frames}"
        )
    return (sampled_frames - 1) // vae_temporal_factor + 1


def mot_action_sequence_length(action_chunk_size: int) -> int:
    return int(action_chunk_size) + 1


def mot_action_per_frame(
    video_downsample_ratio: int = MOT_DEFAULT_VIDEO_DOWNSAMPLE_RATIO,
    vae_temporal_factor: int = WAN_VAE_TEMPORAL_FACTOR,
) -> int:
    video_downsample_ratio = int(video_downsample_ratio)
    vae_temporal_factor = int(vae_temporal_factor)
    if video_downsample_ratio <= 0 or vae_temporal_factor <= 0:
        raise ValueError(
            f"video_downsample_ratio and vae_temporal_factor must be positive, got {video_downsample_ratio}, {vae_temporal_factor}"
        )
    return video_downsample_ratio * vae_temporal_factor


MOT_ACTION_CHUNK_SIZE = MOT_DEFAULT_ACTION_CHUNK_SIZE
MOT_SAMPLED_VIDEO_FRAMES_PER_ACTION_CHUNK_PER_VIEW = mot_sampled_video_frames_per_action_chunk_per_view(
    MOT_ACTION_CHUNK_SIZE
)
MOT_LATENT_FRAMES_PER_ACTION_CHUNK_PER_VIEW = mot_latent_frames_per_action_chunk_per_view(MOT_ACTION_CHUNK_SIZE)
MOT_TARGET_ACTIONS = MOT_ACTION_CHUNK_SIZE
MOT_ACTION_SEQUENCE_LENGTH = mot_action_sequence_length(MOT_ACTION_CHUNK_SIZE)
MOT_ACTION_PER_FRAME = mot_action_per_frame()
MOT_DATASET_MAX_SAMPLE_ATTEMPTS = 10
MOT_VIDEO_TIMESTAMP_TOLERANCE_S = 1e-4
MOT_VIDEO_FPS_TOLERANCE = 1e-4
MOT_TIMESTAMP_POLICY = "episode_local_frame_over_fps_v1"
RAW_ACTION_DIM = 16
RAW_ARM_ACTION_DIM = 8
RELATIVE_ARM_ACTION_DIM = 10
RELATIVE_ACTION_DIM = 20
DEFAULT_MODEL_ROOT = Path(
    "/workspace/cache/huggingface_cache/hub/models--robbyant--lingbot-va-base/"
    "snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c"
)

HAND_TO_VIDEO_KEY = {
    "left": "observation.images.robot_0",
    "right": "observation.images.robot_1",
}
STREAM_LEFT_WRIST = 0
STREAM_HEAD = 1
STREAM_RIGHT_WRIST = 2
HAND_TO_STREAM_ID = {
    "left": STREAM_LEFT_WRIST,
    "right": STREAM_RIGHT_WRIST,
}
VIDEO_KEY_TO_STREAM_ID = {
    video_key: HAND_TO_STREAM_ID[hand]
    for hand, video_key in HAND_TO_VIDEO_KEY.items()
}


def _stream_id_for_view(view: dict[str, Any]) -> int:
    if "stream_id" in view:
        return int(view["stream_id"])
    hand = view.get("hand")
    if hand in HAND_TO_STREAM_ID:
        return HAND_TO_STREAM_ID[hand]
    video_key = view.get("video_key")
    if video_key in VIDEO_KEY_TO_STREAM_ID:
        return VIDEO_KEY_TO_STREAM_ID[video_key]
    raise KeyError(
        "view must provide semantic stream_id or a known UMI hand/video_key"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    return _read_jsonl(Path(path))


def validate_dataset_index_bounds(rows: Iterable[dict[str, Any]], *, context: str) -> None:
    for idx, row in enumerate(rows):
        # 校验长度一致
        expected = int(row["dataset_from_index"]) + int(row["segment"]["end_frame"])
        actual = int(row["dataset_to_index"])
        if actual != expected:
            raise ValueError(
                f"{context} row {idx} has inconsistent dataset_to_index={actual}; "
                f"expected dataset_from_index + segment.end_frame = {expected}"
            )


def _latent_valid_mask_from_sampled_frames(
    raw_video_valid_mask: torch.Tensor,
    *,
    latent_frames_per_action_chunk_per_view: int,
    vae_temporal_factor: int = WAN_VAE_TEMPORAL_FACTOR,
) -> torch.Tensor:
    raw_video_valid_mask = raw_video_valid_mask.bool()
    if raw_video_valid_mask.ndim != 1:
        raise ValueError(f"raw_video_valid_mask must be 1D, got {tuple(raw_video_valid_mask.shape)}")
    latent_frames = int(latent_frames_per_action_chunk_per_view)
    vae_temporal_factor = int(vae_temporal_factor)
    expected = 1 + vae_temporal_factor * (latent_frames - 1)
    if raw_video_valid_mask.shape[0] != expected:
        raise ValueError(f"raw_video_valid_mask length must be {expected}, got {raw_video_valid_mask.shape[0]}")

    # 第一帧直接取 mask，因为是单帧 encode
    values = [raw_video_valid_mask[0]]
    for latent_idx in range(1, latent_frames):
        # 后续每 4 帧为一个 latent frame
        start = 1 + vae_temporal_factor * (latent_idx - 1)
        end = start + vae_temporal_factor
        # NOTE: 只有全 padding 才是 False
        values.append(raw_video_valid_mask[start:end].any())
    return torch.stack(values)


def mot_real_window_frame_ids(
    *,
    current_frame: int,
    video_downsample_ratio: int,
    action_chunk_size: int = MOT_DEFAULT_ACTION_CHUNK_SIZE,
) -> tuple[list[int], list[int]]:
    current_frame = int(current_frame)
    action_chunk_size = int(action_chunk_size)
    stride = int(video_downsample_ratio)
    if stride <= 0:
        raise ValueError(f"video_downsample_ratio must be positive, got {video_downsample_ratio}")
    sampled_frames_per_view = mot_sampled_video_frames_per_action_chunk_per_view(action_chunk_size, stride)
    # 从 start idx开始，构建 history chunk idx 和 target chunk idx
    history_ids = [current_frame - action_chunk_size - 1 + idx * stride for idx in range(sampled_frames_per_view)]
    target_ids = [current_frame + idx * stride for idx in range(sampled_frames_per_view)]
    return history_ids, target_ids


def _allocate_weighted_counts(total: int, weights: Iterable[float]) -> list[int]:
    total = int(total)
    weights = [float(weight) for weight in weights]
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if not weights:
        return []
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError(f"weights must contain at least one positive value, got {weights}")
    raw = [total * weight / sum(weights) for weight in weights]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(weights)), key=lambda idx: (raw[idx] - counts[idx], weights[idx], -idx), reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-8)


def quaternion_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_xyzw(quat)
    x, y, z, w = np.moveaxis(quat, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[..., 0, 1] = 2.0 * (xy - wz)
    matrix[..., 0, 2] = 2.0 * (xz + wy)
    matrix[..., 1, 0] = 2.0 * (xy + wz)
    matrix[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[..., 1, 2] = 2.0 * (yz - wx)
    matrix[..., 2, 0] = 2.0 * (xz - wy)
    matrix[..., 2, 1] = 2.0 * (yz + wx)
    matrix[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrix


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    flat = matrix.reshape(-1, 3, 3)
    quats = np.empty((flat.shape[0], 4), dtype=np.float32)
    for idx, rot in enumerate(flat):
        trace = float(np.trace(rot))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            quats[idx] = [(rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s, 0.25 * s]
        else:
            axis = int(np.argmax(np.diag(rot)))
            if axis == 0:
                s = np.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 1e-8)) * 2.0
                quats[idx] = [0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[2, 1] - rot[1, 2]) / s]
            elif axis == 1:
                s = np.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 1e-8)) * 2.0
                quats[idx] = [(rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s, (rot[0, 2] - rot[2, 0]) / s]
            else:
                s = np.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 1e-8)) * 2.0
                quats[idx] = [(rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s, (rot[1, 0] - rot[0, 1]) / s]
    quats = _normalize_quat_xyzw(quats)
    return quats.reshape(matrix.shape[:-2] + (4,))


def rotation_matrix_to_rot6d_cols(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def rot6d_cols_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.maximum(np.linalg.norm(a2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def absolute_actions_to_relative_20d(reference_states: np.ndarray, absolute_actions: np.ndarray) -> np.ndarray:
    states = np.asarray(reference_states, dtype=np.float32)
    actions = np.asarray(absolute_actions, dtype=np.float32)
    if states.shape != actions.shape or states.shape[-1] != RAW_ACTION_DIM:
        raise ValueError(f"state/action must both be [...,{RAW_ACTION_DIM}], got {states.shape} and {actions.shape}")
    out = np.empty(states.shape[:-1] + (RELATIVE_ACTION_DIM,), dtype=np.float32)
    for arm_idx in range(2):
        raw_off = arm_idx * RAW_ARM_ACTION_DIM
        rel_off = arm_idx * RELATIVE_ARM_ACTION_DIM
        p_state = states[..., raw_off : raw_off + 3]
        p_action = actions[..., raw_off : raw_off + 3]
        r_state = quaternion_xyzw_to_matrix(states[..., raw_off + 3 : raw_off + 7])
        r_action = quaternion_xyzw_to_matrix(actions[..., raw_off + 3 : raw_off + 7])
        r_state_t = np.swapaxes(r_state, -1, -2)
        delta_p = np.einsum("...ij,...j->...i", r_state_t, p_action - p_state)
        delta_r = np.matmul(r_state_t, r_action)
        out[..., rel_off : rel_off + 3] = delta_p
        out[..., rel_off + 3 : rel_off + 9] = rotation_matrix_to_rot6d_cols(delta_r)
        out[..., rel_off + 9] = actions[..., raw_off + 7]
    return out


def relative_20d_to_absolute_actions(reference_states: np.ndarray, relative_actions: np.ndarray) -> np.ndarray:
    states = np.asarray(reference_states, dtype=np.float32)
    relative = np.asarray(relative_actions, dtype=np.float32)
    if states.shape[:-1] != relative.shape[:-1] or states.shape[-1] != RAW_ACTION_DIM or relative.shape[-1] != RELATIVE_ACTION_DIM:
        raise ValueError(f"state/action must be [...,{RAW_ACTION_DIM}] and [...,{RELATIVE_ACTION_DIM}], got {states.shape} and {relative.shape}")
    out = np.empty(states.shape[:-1] + (RAW_ACTION_DIM,), dtype=np.float32)
    for arm_idx in range(2):
        raw_off = arm_idx * RAW_ARM_ACTION_DIM
        rel_off = arm_idx * RELATIVE_ARM_ACTION_DIM
        p_state = states[..., raw_off : raw_off + 3]
        r_state = quaternion_xyzw_to_matrix(states[..., raw_off + 3 : raw_off + 7])
        delta_p = relative[..., rel_off : rel_off + 3]
        delta_r = rot6d_cols_to_rotation_matrix(relative[..., rel_off + 3 : rel_off + 9])
        p_action = p_state + np.einsum("...ij,...j->...i", r_state, delta_p)
        r_action = np.matmul(r_state, delta_r)
        out[..., raw_off : raw_off + 3] = p_action
        out[..., raw_off + 3 : raw_off + 7] = matrix_to_quaternion_xyzw(r_action)
        out[..., raw_off + 7] = relative[..., rel_off + 9]
    return out


class MotTrainData(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        action_sequence_length: int,
        action_dim: int,
        norm_stats_by_task: dict[str, dict[str, Any]],
        action_chunk_size: int,
        video_downsample_ratio: int,
        text_emb_cache_path: str | Path | None,
        empty_emb_path: str | Path | None = None,
        text_emb_cache: dict[str, torch.Tensor] | None = None,
        action_cache_manifest_path: str | Path | None = None,
        video_decoder_cache_size: int = MOT_DEFAULT_VIDEO_DECODER_CACHE_SIZE,
        action_cache_size: int = MOT_DEFAULT_ACTION_CACHE_SIZE,
        random_start: bool = True,
    ):
        self._video_decoder_cache = _BoundedLRUCache(
            video_decoder_cache_size,
            name="video_decoder_cache_size",
        )
        self._action_cache = _BoundedLRUCache(
            action_cache_size,
            name="action_cache_size",
            on_evict=_close_action_cache_entry,
        )
        self.manifest_path = Path(manifest_path)
        self.rows = load_manifest(self.manifest_path)
        if not self.rows:
            raise ValueError(f"empty MOT manifest: {self.manifest_path}")
        mutable_view_buckets = {view_count: [] for view_count in MOT_SUPPORTED_VIEW_COUNTS}
        for row_index, row in enumerate(self.rows):
            view_count = len(row.get("views", ()))
            if view_count not in mutable_view_buckets:
                raise ValueError(
                    f"MOT row {row_index} has unsupported native view count "
                    f"V={view_count}; expected one of {MOT_SUPPORTED_VIEW_COUNTS}"
                )
            mutable_view_buckets[view_count].append(row_index)
        self.view_buckets = {
            view_count: tuple(indices)
            for view_count, indices in mutable_view_buckets.items()
            if indices
        }
        self.available_view_counts = tuple(sorted(self.view_buckets))
        self._row_bucket_positions = {
            row_index: (view_count, position)
            for view_count, indices in self.view_buckets.items()
            for position, row_index in enumerate(indices)
        }
        validate_dataset_index_bounds(self.rows, context=str(self.manifest_path))
        self.action_sequence_length = int(action_sequence_length)
        self.action_dim = int(action_dim)
        if self.action_dim != RELATIVE_ACTION_DIM:
            raise ValueError(f"MOT real dataset requires action_dim={RELATIVE_ACTION_DIM}, got {self.action_dim}")
        self.norm_stats_by_task = norm_stats_by_task
        self.action_chunk_size = int(action_chunk_size)
        if self.action_chunk_size <= 0:
            raise ValueError(f"action_chunk_size must be positive, got {self.action_chunk_size}")
        self.video_downsample_ratio = int(video_downsample_ratio)
        self.random_start = bool(random_start)
        self.latent_frames_per_action_chunk_per_view = mot_latent_frames_per_action_chunk_per_view(
            self.action_chunk_size,
            self.video_downsample_ratio,
        )
        self.sampled_video_frames_per_action_chunk_per_view = mot_sampled_video_frames_per_action_chunk_per_view(
            self.action_chunk_size,
            self.video_downsample_ratio,
        )
        self.action_per_frame = mot_action_per_frame(self.video_downsample_ratio)
        expected_actions_per_chunk = (self.latent_frames_per_action_chunk_per_view - 1) * self.action_per_frame
        if self.action_chunk_size != expected_actions_per_chunk:
            raise ValueError(
                "action_chunk_size must match latent/action packing: "
                f"got {self.action_chunk_size}, expected {expected_actions_per_chunk}"
            )
        self.history_latent_frames = MOT_HISTORY_CHUNKS * self.latent_frames_per_action_chunk_per_view
        self.target_latent_frames = MOT_TARGET_CHUNKS * self.latent_frames_per_action_chunk_per_view
        self.total_latent_frames = self.history_latent_frames + self.target_latent_frames
        self.vae_input_frame_count = self.sampled_video_frames_per_action_chunk_per_view
        self.target_actions = self.action_chunk_size
        self.action_chunks = MOT_ACTION_CHUNKS
        expected_sequence_length = mot_action_sequence_length(self.action_chunk_size)
        if self.action_sequence_length != expected_sequence_length:
            raise ValueError(
                f"action_sequence_length={self.action_sequence_length} must equal action_chunk_size+1={expected_sequence_length}"
            )
        if text_emb_cache is None:
            if text_emb_cache_path is None:
                raise ValueError("text_emb_cache_path is required unless text_emb_cache is supplied")
            text_emb_cache = torch.load(text_emb_cache_path, map_location="cpu", weights_only=False)
        self.text_emb_cache = text_emb_cache
        self.empty_text_emb = (
            torch.load(empty_emb_path, map_location="cpu", weights_only=False)
            if empty_emb_path is not None
            else None
        )
        self._action_cache_index = load_action_cache_index(action_cache_manifest_path)
        self.video_timestamp_tolerance_s = MOT_VIDEO_TIMESTAMP_TOLERANCE_S

    def __len__(self) -> int:
        return len(self.rows)

    def _norm_arrays_for_row(self, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        key = str(row["norm_stats_key"])
        stat = self.norm_stats_by_task[key]
        return np.asarray(stat["q01"], dtype=np.float32)[None], np.asarray(stat["q99"], dtype=np.float32)[None]

    def _runtime_cache_stats(self) -> bytes:
        worker = get_worker_info()
        stats = {
            "worker_id": -1 if worker is None else int(worker.id),
            "worker_pid": os.getpid(),
        }
        for prefix, cache in (
            ("video", self._video_decoder_cache),
            ("action", self._action_cache),
        ):
            cache_stats = (
                cache.stats()
                if cache is not None
                else {
                    "size": 0,
                    "max_size": 0,
                    "hits": 0,
                    "misses": 0,
                    "puts": 0,
                    "evictions": 0,
                }
            )
            stats.update({f"{prefix}_{key}": value for key, value in cache_stats.items()})
        return encode_mot_runtime_cache_stats(stats)

    def _text_emb_for(self, text: str) -> torch.Tensor:
        if text in self.text_emb_cache:
            return self.text_emb_cache[text]
        raise KeyError(f"No text embedding for prompt {text!r}")

    def _sample_start_from_range(self, valid_range: list[int]) -> int:
        if not valid_range:
            raise ValueError("row has no valid_start_range")
        start, end = (int(valid_range[0]), int(valid_range[1]))
        if end < start:
            raise ValueError("row has invalid valid_start_range")
        if not self.random_start:
            return start
        return random.randint(start, end)

    def _pad_frame_ids_to_segment(self, row: dict[str, Any], frame_ids: list[int]) -> tuple[list[int], torch.Tensor]:
        first_valid = int(row["segment"]["start_frame"])
        last_valid = int(row["segment"]["end_frame"]) - 1
        padded = [
            first_valid if int(frame_id) < first_valid else min(int(frame_id), last_valid)
            for frame_id in frame_ids
        ]
        valid = torch.tensor([first_valid <= int(frame_id) <= last_valid for frame_id in frame_ids], dtype=torch.bool)
        return padded, valid

    def __getitem__(self, idx: int) -> dict[str, Any]:
        requested_index = int(idx) % len(self.rows)
        view_count, start_position = self._row_bucket_positions[requested_index]
        bucket = self.view_buckets[view_count]
        last_exc = None
        attempts = min(len(bucket), MOT_DATASET_MAX_SAMPLE_ATTEMPTS)
        for offset in range(attempts):
            sample_idx = bucket[(start_position + offset) % len(bucket)]
            try:
                sample = self._getitem_window(self.rows[sample_idx], sample_index=sample_idx)
                sample["dataset_skip_count"] = torch.tensor(offset, dtype=torch.int64)
                return sample
            except Exception as exc:
                last_exc = exc
                print(f"[MotTrainData] skipped failed sample idx={sample_idx}: {exc}", flush=True)
        raise RuntimeError(
            "Failed to load MOT sample within the requested native-view bucket "
            f"V={view_count} after {attempts} attempts starting at idx={requested_index}"
        ) from last_exc

    def get_window(self, idx: int, start_frame: int) -> dict[str, Any]:
        return self._getitem_window(
            self.rows[idx % len(self.rows)],
            start_frame=start_frame,
            sample_index=idx,
        )

    def _getitem_window(
        self,
        row: dict[str, Any],
        start_frame: int | None = None,
        sample_index: int | None = None,
    ) -> dict[str, Any]:
        # 在 episode 采样 idx
        start = self._sample_start_from_range(row["valid_start_range"]) if start_frame is None else int(start_frame)
        history_frame_ids, target_frame_ids = mot_real_window_frame_ids(
            current_frame=start,
            video_downsample_ratio=self.video_downsample_ratio,
            action_chunk_size=self.action_chunk_size,
        )
        frame_ids = history_frame_ids + target_frame_ids

        # 处理越界，用 vaild episode idx 做 padding，同时生成 mask
        padded_frame_ids, raw_video_valid_mask = self._pad_frame_ids_to_segment(row, frame_ids)
        history_raw_video_valid_mask = raw_video_valid_mask[: self.sampled_video_frames_per_action_chunk_per_view]
        target_raw_video_valid_mask = raw_video_valid_mask[self.sampled_video_frames_per_action_chunk_per_view :]

        # 把 raw-video mask 转换为 latent frame mask，这里只有全为 padding 时，对应的 latent frame mask 才是 False
        # TODO: 这里是为了兼容另一种设计：只要有 padding 就为 False，当前觉得上面那种更好
        video_latent_valid_mask = torch.cat(
            [
                _latent_valid_mask_from_sampled_frames(
                    history_raw_video_valid_mask,
                    latent_frames_per_action_chunk_per_view=self.latent_frames_per_action_chunk_per_view,
                ),
                _latent_valid_mask_from_sampled_frames(
                    target_raw_video_valid_mask,
                    latent_frames_per_action_chunk_per_view=self.latent_frames_per_action_chunk_per_view,
                ),
            ],
            dim=0,
        )
        video_latent_loss_mask = torch.zeros_like(video_latent_valid_mask)
        # T0 is the target anchor; only valid T1.. frames are diffusion targets.
        target_start = self.latent_frames_per_action_chunk_per_view + 1
        video_latent_loss_mask[target_start:] = video_latent_valid_mask[target_start:]

        # action_loss_mask -> 标记哪些 token 算 loss，即只有 target
        # action_valid_mask -> 标记合法 action token，包含 history 和 target
        actions, action_loss_mask, action_valid_mask, action_reference_states = self._load_actions(
            row,
            start,
            video_latent_loss_mask=video_latent_loss_mask,
        )
        action_q01, action_q99 = self._norm_arrays_for_row(row)
        unique_padded_frame_ids = list(dict.fromkeys(padded_frame_ids))
        decoded_rgb = self._load_rgb(row, unique_padded_frame_ids)
        # frame ID 到 decode 位置的映射, 因为可能有 padding 会去重
        decoded_position = {frame_id: idx for idx, frame_id in enumerate(unique_padded_frame_ids)}
        # 完整原始 padded_frame_ids 到 decode 位置的映射
        frame_positions = torch.tensor(
            [decoded_position[frame_id] for frame_id in padded_frame_ids],
            dtype=torch.long,
            device=decoded_rgb.device,
        )
        # 根据索引获取具体的 rgb frame
        rgb = decoded_rgb.index_select(0, frame_positions)
        vae_rgb_history = rgb[: self.sampled_video_frames_per_action_chunk_per_view]
        vae_rgb_target = rgb[self.sampled_video_frames_per_action_chunk_per_view :]
        text = row["segment"]["action_text"]
        out = {
            "vae_rgb_history": vae_rgb_history,
            "vae_rgb_target": vae_rgb_target,
            "stream_ids": torch.tensor([_stream_id_for_view(view) for view in row["views"]], dtype=torch.long),
            "video_latent_loss_mask": video_latent_loss_mask,
            "video_latent_valid_mask": video_latent_valid_mask,
            "actions": actions,
            "action_loss_mask": action_loss_mask,
            "action_valid_mask": action_valid_mask,
            "action_q01": torch.from_numpy(action_q01[0]).float(),
            "action_q99": torch.from_numpy(action_q99[0]).float(),
            "action_reference_states": action_reference_states,
            "text_emb": self._text_emb_for(text),
            "dataset_skip_count": torch.tensor(0, dtype=torch.int64),
            "_runtime_cache_stats": self._runtime_cache_stats(),
        }
        if self.empty_text_emb is not None:
            out["empty_text_emb"] = self.empty_text_emb
        return out

    def _video_decoder(self, video_path: Path) -> VideoDecoder:
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        cache = getattr(self, "_video_decoder_cache", None)
        if cache is None:
            cache = _BoundedLRUCache(
                MOT_DEFAULT_VIDEO_DECODER_CACHE_SIZE,
                name="video_decoder_cache_size",
            )
            self._video_decoder_cache = cache
        decoder = cache.get(video_path)
        if decoder is None:
            decoder = VideoDecoder(video_path, dimension_order="NCHW", device="cpu", seek_mode="approximate")
            cache.put(video_path, decoder)
        return decoder

    def _video_average_fps(self, video_path: Path) -> float:
        return float(self._video_decoder(video_path).metadata.average_fps)

    def _read_video_frames(self, video_path: Path, frame_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor | None]:
        frames_batch = self._video_decoder(video_path).get_frames_at([int(idx) for idx in frame_indices])
        pts_seconds = getattr(frames_batch, "pts_seconds", None)
        return frames_batch.data.float() / 255.0, pts_seconds

    def _video_tolerance_s(self) -> float:
        return float(getattr(self, "video_timestamp_tolerance_s", MOT_VIDEO_TIMESTAMP_TOLERANCE_S))

    def _local_timestamps_for_frame_ids(self, row: dict[str, Any], local_frame_ids: list[int], fps: float) -> np.ndarray:
        timestamp_policy = row.get("timestamp_policy")
        if timestamp_policy != MOT_TIMESTAMP_POLICY:
            raise ValueError(
                "Unsupported or missing timestamp_policy in MOT manifest: "
                f"got {timestamp_policy!r}, expected {MOT_TIMESTAMP_POLICY!r}. "
                "Rebuild the base cache and training dataset metadata."
            )
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid manifest fps for video timestamps: {fps}")
        return np.asarray(local_frame_ids, dtype=np.float64) / float(fps)

    def _validate_video_fps(self, video_path: Path, manifest_fps: float, video_fps: float) -> None:
        if not np.isfinite(video_fps) or abs(float(video_fps) - float(manifest_fps)) > MOT_VIDEO_FPS_TOLERANCE:
            raise ValueError(
                "Video FPS mismatch: "
                f"video={video_path} manifest_fps={float(manifest_fps)} video_average_fps={float(video_fps)} "
                f"tolerance={MOT_VIDEO_FPS_TOLERANCE}"
            )

    def _unpack_video_read(self, read_result: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(read_result, tuple):
            if not read_result:
                raise ValueError("_read_video_frames returned an empty tuple")
            frames = read_result[0]
            pts_seconds = read_result[1] if len(read_result) > 1 else None
            return frames, pts_seconds
        return read_result, None

    def _validate_video_pts(
        self,
        *,
        video_path: Path,
        frame_indices: list[int],
        query_timestamps: np.ndarray,
        pts_seconds: torch.Tensor | None,
    ) -> None:
        if pts_seconds is None:
            return
        if torch.is_tensor(pts_seconds):
            pts = np.asarray(pts_seconds.detach().cpu(), dtype=np.float64).reshape(-1)
        else:
            pts = np.asarray(pts_seconds, dtype=np.float64).reshape(-1)
        expected = np.asarray(query_timestamps, dtype=np.float64).reshape(-1)
        if pts.shape[0] != expected.shape[0]:
            raise ValueError(
                f"Video timestamp count mismatch for {video_path}: "
                f"{pts.shape[0]} decoded timestamps vs {expected.shape[0]} query timestamps"
            )
        deltas = np.abs(pts - expected)
        bad = np.flatnonzero(deltas >= self._video_tolerance_s())
        if bad.size:
            idx = int(bad[0])
            raise ValueError(
                "Video timestamp mismatch: "
                f"video={video_path} frame_index={int(frame_indices[idx])} "
                f"query_timestamp={float(expected[idx])} decoded_pts={float(pts[idx])} "
                f"delta={float(deltas[idx])} tolerance_s={self._video_tolerance_s()}"
            )

    def _load_rgb(self, row: dict[str, Any], local_frame_ids: list[int]) -> torch.Tensor:
        view_tensors = []
        fps = float(row["fps"])
        local_timestamps = self._local_timestamps_for_frame_ids(row, local_frame_ids, fps)
        for view in row["views"]:
            video_path = Path(view["video_path"])
            video_fps = self._video_average_fps(video_path)
            self._validate_video_fps(video_path, fps, video_fps)
            query_timestamps = float(view["video_from_timestamp"]) + local_timestamps
            global_ids = [int(round(float(ts) * video_fps)) for ts in query_timestamps]
            frames, pts_seconds = self._unpack_video_read(self._read_video_frames(video_path, global_ids))
            self._validate_video_pts(
                video_path=video_path,
                frame_indices=global_ids,
                query_timestamps=query_timestamps,
                pts_seconds=pts_seconds,
            )
            view_tensors.append(frames)
        return torch.stack(view_tensors, dim=1)

    def _load_action_state_index_arrays(self, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data_file = Path(row["data_file"])
        cached = self._action_cache.get(data_file)
        if cached is not None:
            return cached

        # 前序 preprocess 处理好的 action/state cache
        cache_info = self._action_cache_index.get(_cache_index_key(data_file)) or self._action_cache_index.get(str(data_file))
        if cache_info is not None:
            index_path = cache_info.get("index_npy")
            if index_path is None:
                raise ValueError(f"LeRobot action cache requires index_npy for {data_file}")
            actions = np.load(cache_info["actions_npy"], mmap_mode="r")
            states = np.load(cache_info["states_npy"], mmap_mode="r")
            indices = np.load(index_path, mmap_mode="r")
            self._action_cache.put(data_file, (actions, states, indices))
            return actions, states, indices
        schema_names = set(pq.read_schema(data_file).names)
        if "index" not in schema_names:
            raise ValueError(f"LeRobot action loading requires an index column: {data_file}")

        # fallback 到 parquet 读取 -> 理论上不会触发
        table = pq.read_table(data_file, columns=["action", "observation.state", "index"])
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
        self._action_cache.put(data_file, (actions, states, indices))

        return actions, states, indices

    def _absolute_segment_arrays(self, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        local_start = int(row["segment"]["start_frame"])
        local_end = int(row["segment"]["end_frame"])
        absolute_start = int(row["dataset_from_index"]) + local_start
        absolute_end = int(row["dataset_from_index"]) + local_end
        # 这里拿到的一整个 parquet 的数据
        actions_all, states_all, action_indices = self._load_action_state_index_arrays(row)
        if actions_all.shape[0] != action_indices.shape[0] or states_all.shape[0] != action_indices.shape[0]:
            raise ValueError(
                f"LeRobot action/index length mismatch in {row['data_file']}: "
                f"actions={actions_all.shape[0]} states={states_all.shape[0]} index={action_indices.shape[0]}"
            )
        expected_len = absolute_end - absolute_start
        # 找出 action_indices 里属于当前 segment 范围的那些行号
        positions = np.flatnonzero((action_indices >= absolute_start) & (action_indices < absolute_end))
        # 确保按照时间顺序排列
        positions = positions[np.argsort(action_indices[positions])]
        expected_indices = np.arange(absolute_start, absolute_end, dtype=np.int64)
        # 校验长度和连续性
        if positions.size != expected_len or not np.array_equal(action_indices[positions], expected_indices):
            raise ValueError(
                "LeRobot action index coverage mismatch: "
                f"data_file={row['data_file']} expected absolute index range [{absolute_start}, {absolute_end}) "
                f"with {expected_len} rows, found {int(positions.size)}"
            )
        return actions_all[positions], states_all[positions]

    def _load_actions(
        self,
        row: dict[str, Any],
        current_frame: int,
        *,
        video_latent_loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local_start = int(row["segment"]["start_frame"])
        current_frame = int(current_frame)
        local_current = current_frame - local_start
        # 切出当前 episode segment 的 raw action/state，参考原始 lerobot 实现
        actions_source, states_source = self._absolute_segment_arrays(row)

        action_frames = self.total_latent_frames
        tokens_per_frame = self.action_per_frame

        # history + target
        aligned = np.zeros((action_frames, tokens_per_frame, RELATIVE_ACTION_DIM), dtype=np.float32)
        # target action loss
        action_loss_mask = np.zeros_like(aligned, dtype=bool)
        # history action condition
        condition_mask = np.zeros_like(aligned, dtype=bool)
        reference_states = np.zeros((action_frames, tokens_per_frame, RAW_ACTION_DIM), dtype=np.float32)

        action_frame_loss_mask = video_latent_loss_mask.detach().cpu().numpy().astype(bool)
        if action_frame_loss_mask.shape != (action_frames,):
            raise ValueError(
                f"video_latent_loss_mask must be [{action_frames}], got {action_frame_loss_mask.shape}"
            )

        def fill_action_chunk(*, chunk_idx: int, action_start: int, ref_idx: int, supervise: bool) -> None:
            if not (0 <= ref_idx < states_source.shape[0]):
                return
            # 标志 history/target 的起点:  history chunk_idx=0 -> frame_offset=0 / target chunk_idx=1 -> frame_offset=4
            frame_offset = int(chunk_idx) * self.latent_frames_per_action_chunk_per_view

            # 一个 chunk 内所有 action 都相对同一个 state 表示
            ref_state = states_source[ref_idx : ref_idx + 1]
            reference_states[frame_offset : frame_offset + self.latent_frames_per_action_chunk_per_view] = ref_state

            for action_offset in range(self.action_chunk_size):
                # NOTE: 计算 action 放在哪个 latent frame，具体实现了 lingbot 中的 action 偏移一位
                latent_offset = 1 + action_offset // tokens_per_frame
                # 计算 latent 内部的 token 位置 （这里是取余，上面是整除）
                token_idx = action_offset % tokens_per_frame
                # 然后加上 history/target的 全局 offset
                latent_idx = frame_offset + latent_offset
                # NOTE: 最终得到：
                # 全局 latent 0：空
                # 全局 latent 1～3：history action
                # 全局 latent 4：空
                # 全局 latent 5～7：target action

                # 这里是逐 action 遍历和处理，会跳过 invalid 的 action （对于跳过的，即触发 continue 的，其 aligned/condition_mask/action_loss_mask 都会保持默认值 0）
                if supervise and not action_frame_loss_mask[latent_idx]:
                    continue
                action_idx = int(action_start) + action_offset
                if action_idx < 0:
                    continue
                if action_idx >= actions_source.shape[0]:
                    # target 会做 padding
                    if supervise:
                        action_idx = actions_source.shape[0] - 1
                    else:
                        continue

                rel = absolute_actions_to_relative_20d(ref_state, actions_source[action_idx : action_idx + 1])
                # rel 保留了 batch 维度，用 [0] 去掉。由于这里是逐帧的，所以写入对应的 [latent_idx, token_idx]
                aligned[latent_idx, token_idx] = rel[0]
                if supervise:
                    action_loss_mask[latent_idx, token_idx] = True
                else:
                    condition_mask[latent_idx, token_idx] = True

        if 0 <= local_current < states_source.shape[0]:
            history_start = local_current - self.target_actions
            history_ref_idx = max(0, history_start)
            fill_action_chunk(chunk_idx=0, action_start=history_start, ref_idx=history_ref_idx, supervise=False)
            fill_action_chunk(chunk_idx=1, action_start=local_current, ref_idx=local_current, supervise=True)

        q01, q99 = self._norm_arrays_for_row(row)
        keep_mask = action_loss_mask | condition_mask
        valid_mask = keep_mask.transpose(2, 0, 1)
        aligned = (aligned - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        aligned = np.clip(aligned, -1.5, 1.5)
        aligned = aligned * keep_mask
        aligned = aligned.transpose(2, 0, 1)
        action_loss_mask = action_loss_mask.transpose(2, 0, 1)
        reference_states = reference_states.transpose(2, 0, 1)
        return (
            torch.from_numpy(aligned[..., None]).float(),
            torch.from_numpy(action_loss_mask[..., None]).bool(),
            torch.from_numpy(valid_mask[..., None]).bool(),
            torch.from_numpy(reference_states[..., None]).float(),
        )




def validate_mot_batch_for_forward(batch: dict[str, Any], action_sequence_length: int | None = None) -> dict[str, list[int]]:
    required = (
        "latents", "actions", "action_loss_mask", "action_valid_mask",
        "text_emb", "stream_ids", "video_latent_loss_mask",
        "video_latent_valid_mask",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"MOT batch missing keys required by forward_train: {missing}")
    latents = batch["latents"]
    actions = batch["actions"]
    stream_ids = batch["stream_ids"]
    if latents.ndim != 6:
        raise ValueError(f"latents must be [B,C,F,V,H,W], got {tuple(latents.shape)}")
    if actions.ndim != 5:
        raise ValueError(f"actions must be [B,C,F,N,1], got {tuple(actions.shape)}")
    if action_sequence_length is None:
        if actions.shape[2] % MOT_ACTION_CHUNKS != 0:
            raise ValueError(f"action frames must be divisible by {MOT_ACTION_CHUNKS}")
        action_chunk_size = (actions.shape[2] // MOT_ACTION_CHUNKS - 1) * actions.shape[3]
    else:
        action_chunk_size = int(action_sequence_length) - 1
    expected_frames = MOT_ACTION_CHUNKS * mot_latent_frames_per_action_chunk_per_view(action_chunk_size)
    expected_action_per_frame = mot_action_per_frame()
    batch_size, _, latent_frames, views = latents.shape[:4]
    if latent_frames != expected_frames:
        raise ValueError(f"latents must contain {expected_frames} frames, got {latent_frames}")
    if actions.shape[2:] != (expected_frames, expected_action_per_frame, 1):
        raise ValueError(
            f"actions must use [{expected_frames},{expected_action_per_frame},1], "
            f"got {tuple(actions.shape[2:])}"
        )
    for name in ("action_loss_mask", "action_valid_mask"):
        if batch[name].shape != actions.shape:
            raise ValueError(f"{name} shape does not match actions")
    for name in ("video_latent_loss_mask", "video_latent_valid_mask"):
        if batch[name].shape != (batch_size, expected_frames):
            raise ValueError(f"{name} must be [B,{expected_frames}]")
    if stream_ids.shape != (batch_size, views):
        raise ValueError(f"stream_ids must be [B,V], got {tuple(stream_ids.shape)}")
    valid_stream_ids = (stream_ids >= STREAM_LEFT_WRIST) & (stream_ids <= STREAM_RIGHT_WRIST)
    if not bool(valid_stream_ids.all().item()):
        raise ValueError("stream_ids must use LEFT_WRIST=0, HEAD=1, RIGHT_WRIST=2")
    if action_sequence_length is not None and int(action_sequence_length) != mot_action_sequence_length(action_chunk_size):
        raise ValueError(
            f"action_sequence_length must be {mot_action_sequence_length(action_chunk_size)}"
        )
    return {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)}
