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
import torch.nn.functional as F
from torch.utils.data import Dataset, get_worker_info
from torchcodec.decoders import VideoDecoder

from .action_cache import _cache_index_key, load_action_cache_index
from .pointcloud_store import PointStore


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
MOT_GEOMETRY_GROUP_SIZE = 4
MOT_DEFAULT_VIDEO_DECODER_CACHE_SIZE = 256
MOT_DEFAULT_POINT_STORE_CACHE_SIZE = 16
MOT_DEFAULT_ACTION_CACHE_SIZE = 512
MOT_SUPPORTED_VIEW_COUNTS = (2, 3)
MOT_RUNTIME_CACHE_STAT_FIELDS = ("worker_id", "worker_pid") + tuple(
    f"{prefix}_{field}"
    for prefix in ("video", "point", "action")
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


def _close_point_store(store: PointStore) -> None:
    store.close()


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


def mot_geometry_groups(
    action_chunk_size: int,
    video_downsample_ratio: int = MOT_DEFAULT_VIDEO_DOWNSAMPLE_RATIO,
    vae_temporal_factor: int = WAN_VAE_TEMPORAL_FACTOR,
) -> int:
    return MOT_ACTION_CHUNKS * mot_latent_frames_per_action_chunk_per_view(
        action_chunk_size,
        video_downsample_ratio,
        vae_temporal_factor,
    )


MOT_ACTION_CHUNK_SIZE = MOT_DEFAULT_ACTION_CHUNK_SIZE
MOT_SAMPLED_VIDEO_FRAMES_PER_ACTION_CHUNK_PER_VIEW = mot_sampled_video_frames_per_action_chunk_per_view(
    MOT_ACTION_CHUNK_SIZE
)
MOT_LATENT_FRAMES_PER_ACTION_CHUNK_PER_VIEW = mot_latent_frames_per_action_chunk_per_view(MOT_ACTION_CHUNK_SIZE)
MOT_TARGET_ACTIONS = MOT_ACTION_CHUNK_SIZE
MOT_ACTION_SEQUENCE_LENGTH = mot_action_sequence_length(MOT_ACTION_CHUNK_SIZE)
MOT_ACTION_PER_FRAME = mot_action_per_frame()
MOT_GEOMETRY_GROUPS = mot_geometry_groups(MOT_ACTION_CHUNK_SIZE)
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


def _group_mot_frames(
    frame_ids: list[int],
    *,
    latent_frames_per_action_chunk_per_view: int,
) -> tuple[list[list[int]], list[list[bool]]]:
    groups: list[list[int]] = []
    masks: list[list[bool]] = []
    cursor = 0
    # history 和 target 两个 chunk 分别处理
    for _chunk_idx in range(MOT_ACTION_CHUNKS):
        for latent_idx in range(int(latent_frames_per_action_chunk_per_view)):
            # VAE 第一帧会单独 encode，后续都是 4 帧一个 latent frame
            group_len = 1 if latent_idx == 0 else MOT_GEOMETRY_GROUP_SIZE
            # frame_ids 里按当前位置 cursor 切出当前 group 需要的 frame id
            group = [int(value) for value in frame_ids[cursor : cursor + group_len]]
            cursor += group_len
            if len(group) != group_len:
                raise ValueError(f"MOT geometry grouping expected {group_len} frames, got {len(group)}")
            # 第一个 group 只有一帧，为了便于后续模型处理，padding 到 4 帧
            padded = group + [group[-1]] * (MOT_GEOMETRY_GROUP_SIZE - group_len)
            groups.append(padded)
            # mask 标记
            masks.append([True] * group_len + [False] * (MOT_GEOMETRY_GROUP_SIZE - group_len))
    if cursor != len(frame_ids):
        raise ValueError(f"MOT geometry grouping consumed {cursor} frames from {len(frame_ids)}")
    return groups, masks


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
) -> tuple[list[int], list[int], list[list[int]], list[list[bool]]]:
    current_frame = int(current_frame)
    action_chunk_size = int(action_chunk_size)
    stride = int(video_downsample_ratio)
    if stride <= 0:
        raise ValueError(f"video_downsample_ratio must be positive, got {video_downsample_ratio}")
    sampled_frames_per_view = mot_sampled_video_frames_per_action_chunk_per_view(action_chunk_size, stride)
    latent_frames_per_view = mot_latent_frames_per_action_chunk_per_view(action_chunk_size, stride)
    # 从 start idx开始，构建 history chunk idx 和 target chunk idx
    history_ids = [current_frame - action_chunk_size - 1 + idx * stride for idx in range(sampled_frames_per_view)]
    target_ids = [current_frame + idx * stride for idx in range(sampled_frames_per_view)]
    # 每个 chunk 的 13 个 sampled frames 会被分成 4 个 latent groups
    # 共 2 chunks * 4 latent groups = 8 geometry groups
    geometry_ids, geometry_group_mask = _group_mot_frames(
        history_ids + target_ids,
        latent_frames_per_action_chunk_per_view=latent_frames_per_view,
    )
    # geometry_ids, geometry_group_mask都是二维列表，输出类似：
    # history latent 0: [0,0,0,0], mask [T,F,F,F]
    # history latent 1: [1,2,3,4], mask [T,T,T,T]
    # target latent 0: xxx  mask [T,F,F,F]
    # target latent 1: xxx, mask [T,T,T,T]
    # NOTE: 核心就是让后续 G-branch 可以构造与视频 latent 时间对齐的 register token，从而做 MOT attention
    return history_ids, target_ids, geometry_ids, geometry_group_mask


def _flatten_groups(groups: list[list[int]]) -> list[int]:
    return [int(value) for group in groups for value in group]


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


class MotBalancedMixDataset(Dataset):
    def __init__(self, pointcloud_dataset: Dataset, pure_dataset: Dataset):
        if len(pointcloud_dataset) <= 0:
            raise ValueError("pointcloud_dataset must not be empty")
        if len(pure_dataset) <= 0:
            raise ValueError("pure_dataset must not be empty")
        self.pointcloud_dataset = pointcloud_dataset
        self.pure_dataset = pure_dataset
        pointcloud_views = set(pointcloud_dataset.view_buckets)
        pure_views = set(pure_dataset.view_buckets)
        if pointcloud_views != pure_views:
            raise ValueError(
                "pointcloud and non-pointcloud sources must expose the same "
                "nonempty native-view domain"
            )
        self.available_view_counts = tuple(sorted(pointcloud_views))
        self._pairs = max(len(pointcloud_dataset), len(pure_dataset))

    def __len__(self) -> int:
        return 2 * self._pairs

    @staticmethod
    def encode_source_index(source: int, row_index: int) -> int:
        source = int(source)
        if source not in (0, 1):
            raise ValueError(f"source must be 0 or 1, got {source}")
        return 2 * int(row_index) + source

    def _getitem_with_fallback(self, dataset: Dataset, start_idx: int, source_name: str) -> dict[str, Any]:
        size = len(dataset)
        if isinstance(dataset, MotTrainData):
            return dataset[int(start_idx) % size]
        last_exc = None
        attempts = min(size, MOT_DATASET_MAX_SAMPLE_ATTEMPTS)
        for offset in range(attempts):
            sample_idx = (int(start_idx) + offset) % size
            try:
                return dataset[sample_idx]
            except Exception as exc:
                last_exc = exc
                print(
                    f"[MotBalancedMixDataset] skipped failed {source_name} sample idx={sample_idx}: {exc}",
                    flush=True,
                )
        raise RuntimeError(
            f"Failed to load {source_name} sample after {attempts} attempts starting at idx={int(start_idx) % size}"
        ) from last_exc

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair_idx = int(idx) // 2
        if int(idx) % 2 == 0:
            return self._getitem_with_fallback(
                self.pointcloud_dataset,
                pair_idx % len(self.pointcloud_dataset),
                "pointcloud",
            )
        return self._getitem_with_fallback(
            self.pure_dataset,
            pair_idx % len(self.pure_dataset),
            "non_pointcloud",
        )


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
        text_emb_cache_path: str | Path,
        action_cache_manifest_path: str | Path | None = None,
        video_decoder_cache_size: int = MOT_DEFAULT_VIDEO_DECODER_CACHE_SIZE,
        point_store_cache_size: int = MOT_DEFAULT_POINT_STORE_CACHE_SIZE,
        action_cache_size: int = MOT_DEFAULT_ACTION_CACHE_SIZE,
        random_start: bool = True,
        data_profile: str = "joint",
    ):
        self.data_profile = str(data_profile)
        if self.data_profile not in {"geometry", "joint"}:
            raise ValueError(
                f"data_profile must be 'geometry' or 'joint', got {data_profile!r}"
            )
        self._video_decoder_cache = _BoundedLRUCache(
            video_decoder_cache_size,
            name="video_decoder_cache_size",
        )
        self._point_store_cache = _BoundedLRUCache(
            point_store_cache_size,
            name="point_store_cache_size",
            on_evict=_close_point_store,
        )
        self._action_cache = (
            _BoundedLRUCache(
                action_cache_size,
                name="action_cache_size",
                on_evict=_close_action_cache_entry,
            )
            if self.data_profile == "joint"
            else None
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
        if self.data_profile == "joint" and self.action_dim != RELATIVE_ACTION_DIM:
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
        self.geometry_groups = mot_geometry_groups(self.action_chunk_size, self.video_downsample_ratio)
        self.geometry_group_size = MOT_GEOMETRY_GROUP_SIZE
        self.action_chunks = MOT_ACTION_CHUNKS
        expected_sequence_length = mot_action_sequence_length(self.action_chunk_size)
        if self.data_profile == "joint" and self.action_sequence_length != expected_sequence_length:
            raise ValueError(
                f"action_sequence_length={self.action_sequence_length} must equal action_chunk_size+1={expected_sequence_length}"
            )
        if self.data_profile == "joint":
            self.text_emb_cache = torch.load(text_emb_cache_path, map_location="cpu", weights_only=False)
            self._action_cache_index = load_action_cache_index(action_cache_manifest_path)
        else:
            self.text_emb_cache = None
            self._action_cache_index = {}
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
            ("point", self._point_store_cache),
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
        if self.text_emb_cache is None:
            raise RuntimeError("text embeddings are unavailable in the geometry data profile")
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

    def _materialize_geometry_fields(
        self,
        row: dict[str, Any],
        *,
        padded_geometry_frame_ids: list[int],
        geometry_flat_valid_mask: torch.Tensor,
        geometry_group_mask: list[list[bool]],
        decoded_rgb: torch.Tensor,
        decoded_position: dict[int, int],
    ) -> dict[str, torch.Tensor]:
        geometry_positions = torch.tensor(
            [decoded_position[frame_id] for frame_id in padded_geometry_frame_ids],
            dtype=torch.long,
            device=decoded_rgb.device,
        )
        geometry_rgb_flat = decoded_rgb.index_select(0, geometry_positions)
        geometry_rgb = geometry_rgb_flat.reshape(
            self.geometry_groups,
            self.geometry_group_size,
            *geometry_rgb_flat.shape[1:],
        )
        decoded_pts3d, decoded_point_valid_mask = self._load_points(
            row,
            list(decoded_position),
            target_h_w=tuple(decoded_rgb.shape[-2:]),
        )
        geometry_pts3d_flat = decoded_pts3d.index_select(0, geometry_positions)
        geometry_point_valid_mask = decoded_point_valid_mask.index_select(
            0,
            geometry_positions,
        )
        geometry_group_valid_mask = torch.tensor(
            geometry_group_mask,
            dtype=torch.bool,
        ) & geometry_flat_valid_mask.reshape(
            self.geometry_groups,
            self.geometry_group_size,
        )
        geometry_pts3d = geometry_pts3d_flat.reshape(
            self.geometry_groups,
            self.geometry_group_size,
            *geometry_pts3d_flat.shape[1:],
        )
        geometry_point_valid_mask = geometry_point_valid_mask.reshape(
            self.geometry_groups,
            self.geometry_group_size,
            *geometry_point_valid_mask.shape[1:],
        )
        geometry_point_valid_mask = geometry_point_valid_mask & geometry_group_valid_mask[:, :, None, None, None]
        return {
            "geometry_rgb": geometry_rgb,
            "geometry_pts3d": geometry_pts3d,
            "geometry_point_valid_mask": geometry_point_valid_mask,
            "geometry_group_valid_mask": geometry_group_valid_mask,
        }

    def _getitem_geometry_window(
        self,
        row: dict[str, Any],
        *,
        start_frame: int | None = None,
    ) -> dict[str, Any]:
        start = (
            self._sample_start_from_range(row["valid_start_range"])
            if start_frame is None
            else int(start_frame)
        )
        _history, _target, geometry_frame_groups, geometry_group_mask = mot_real_window_frame_ids(
            current_frame=start,
            video_downsample_ratio=self.video_downsample_ratio,
            action_chunk_size=self.action_chunk_size,
        )
        padded_geometry_frame_ids, geometry_flat_valid_mask = self._pad_frame_ids_to_segment(
            row,
            _flatten_groups(geometry_frame_groups),
        )
        unique_frame_ids = list(dict.fromkeys(padded_geometry_frame_ids))
        decoded_rgb = self._load_rgb(row, unique_frame_ids)
        decoded_position = {
            frame_id: index for index, frame_id in enumerate(unique_frame_ids)
        }
        out = self._materialize_geometry_fields(
            row,
            padded_geometry_frame_ids=padded_geometry_frame_ids,
            geometry_flat_valid_mask=geometry_flat_valid_mask,
            geometry_group_mask=geometry_group_mask,
            decoded_rgb=decoded_rgb,
            decoded_position=decoded_position,
        )
        out.update(
            has_pointcloud=torch.tensor(True, dtype=torch.bool),
            dataset_skip_count=torch.tensor(0, dtype=torch.int64),
            _runtime_cache_stats=self._runtime_cache_stats(),
        )
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        requested_index = int(idx) % len(self.rows)
        view_count, start_position = self._row_bucket_positions[requested_index]
        bucket = self.view_buckets[view_count]
        last_exc = None
        attempts = min(len(bucket), MOT_DATASET_MAX_SAMPLE_ATTEMPTS)
        for offset in range(attempts):
            sample_idx = bucket[(start_position + offset) % len(bucket)]
            # 先选一条 episode
            row = self.rows[sample_idx]
            try:
                # 再从 episode 选实际用于训练的 chunk
                sample = (
                    self._getitem_geometry_window(row)
                    if self.data_profile == "geometry"
                    else self._getitem_window(row, sample_index=sample_idx)
                )
                sample["has_pointcloud"] = torch.tensor(bool(row.get("has_pointcloud")), dtype=torch.bool)
                sample["dataset_skip_count"] = torch.tensor(offset, dtype=torch.int64)
                return sample
            except Exception as exc:
                last_exc = exc
                print(
                    f"[MotTrainData] skipped failed sample idx={sample_idx}: {exc}",
                    flush=True,
                )
        raise RuntimeError(
            "Failed to load MOT sample within the requested native-view bucket "
            f"V={view_count} after {attempts} attempts starting at idx={requested_index}"
        ) from last_exc

    def get_window(self, idx: int, start_frame: int) -> dict[str, Any]:
        row = self.rows[idx % len(self.rows)]
        if self.data_profile == "geometry":
            return self._getitem_geometry_window(row, start_frame=start_frame)
        return self._getitem_window(row, start_frame=start_frame, sample_index=idx)

    def _getitem_window(
        self,
        row: dict[str, Any],
        start_frame: int | None = None,
        sample_index: int | None = None,
    ) -> dict[str, Any]:
        # 在 episode 采样 idx
        start = self._sample_start_from_range(row["valid_start_range"]) if start_frame is None else int(start_frame)
        # 训练用的 segment 的 video idx，其中 geometry 部分是二维列表，以默认参数为例，就是 8 个 group（对应 VAE 后的 8 个 latent frame），每组 4 个 slot
        # 这里返回的根据 current_frame直接算出来的，没有考虑越界，越界在下面代码进行 padding
        history_frame_ids, target_frame_ids, geometry_frame_groups, geometry_group_mask = mot_real_window_frame_ids(
            current_frame=start,
            video_downsample_ratio=self.video_downsample_ratio,
            action_chunk_size=self.action_chunk_size,
        )
        frame_ids = history_frame_ids + target_frame_ids
        geometry_frame_ids = _flatten_groups(geometry_frame_groups)

        # 处理越界，用 vaild episode idx 做 padding，同时生成 mask
        padded_frame_ids, raw_video_valid_mask = self._pad_frame_ids_to_segment(row, frame_ids)
        history_raw_video_valid_mask = raw_video_valid_mask[: self.sampled_video_frames_per_action_chunk_per_view]
        target_raw_video_valid_mask = raw_video_valid_mask[self.sampled_video_frames_per_action_chunk_per_view :]
        padded_geometry_frame_ids, geometry_flat_valid_mask = self._pad_frame_ids_to_segment(row, geometry_frame_ids)

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
        geometry_fields = self._materialize_geometry_fields(
            row,
            padded_geometry_frame_ids=padded_geometry_frame_ids,
            geometry_flat_valid_mask=geometry_flat_valid_mask,
            geometry_group_mask=geometry_group_mask,
            decoded_rgb=decoded_rgb,
            decoded_position=decoded_position,
        )
        text = row["segment"]["action_text"]
        out = {
            "vae_rgb_history": vae_rgb_history,
            "vae_rgb_target": vae_rgb_target,
            **geometry_fields,
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
            "has_pointcloud": torch.tensor(bool(row.get("has_pointcloud")), dtype=torch.bool),
            "dataset_skip_count": torch.tensor(0, dtype=torch.int64),
            "_runtime_cache_stats": self._runtime_cache_stats(),
        }
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

    def _point_store(self, view: dict[str, Any]) -> PointStore:
        store_dir = Path(view["preprocessed_pointcloud_dir"])
        cached = self._point_store_cache.get(store_dir)
        if cached is not None:
            return cached
        store = PointStore.open(store_dir)
        self._point_store_cache.put(store_dir, store)
        return store

    def _load_points(
        self,
        row: dict[str, Any],
        local_frame_ids: list[int],
        target_h_w: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(row.get("has_pointcloud")):
            if target_h_w is None:
                raise ValueError("target_h_w is required for non-pointcloud rows")
            frames = len(local_frame_ids)
            views = len(row["views"])
            height, width = int(target_h_w[0]), int(target_h_w[1])
            points = torch.zeros(frames, views, height, width, 3, dtype=torch.float32)
            mask = torch.zeros(frames, views, height, width, dtype=torch.bool)
            return points, mask

        per_frame_points = []
        per_frame_masks = []
        for frame_id in local_frame_ids:
            view_points = []
            view_masks = []
            for view in row["views"]:
                store = self._point_store(view)
                row_idx = store.row_for_episode_frame(int(frame_id))
                if row_idx < 0:
                    raise KeyError(f"No preprocessed pointcloud row covers episode frame {int(frame_id)} in {store.root}")
                points_t = torch.from_numpy(np.array(store.points[row_idx], copy=True)).float()
                mask_t = torch.from_numpy(np.array(store.valid_mask[row_idx], copy=True)).bool()
                view_points.append(points_t)
                view_masks.append(mask_t)
            per_frame_points.append(torch.stack(view_points, dim=0))
            per_frame_masks.append(torch.stack(view_masks, dim=0))
        return torch.stack(per_frame_points, dim=0).float(), torch.stack(per_frame_masks, dim=0).bool()

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




class MotGeometryLeRobotData(MotTrainData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not all(bool(row.get("has_pointcloud")) for row in self.rows):
            raise ValueError("MotGeometryLeRobotData requires all rows to have pointcloud labels")
        for row in self.rows:
            for view in row.get("views", []):
                if "preprocessed_pointcloud_dir" not in view:
                    raise ValueError(
                        "Pointcloud training manifest must use preprocessed_pointcloud_dir. "
                        "Regenerate pointcloud stores with the source-specific pointcloud pipeline, "
                        "then rebuild the train manifest."
                    )


class MotPureLeRobotData(MotTrainData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any(bool(row.get("has_pointcloud")) for row in self.rows):
            raise ValueError("MotPureLeRobotData requires all rows to be non-pointcloud rows")


def _validate_geometry_batch_shapes(
    batch: dict[str, Any],
    *,
    expected_frames: int | None = None,
) -> tuple[int, int]:
    required = (
        "geometry_rgb",
        "geometry_pts3d",
        "geometry_point_valid_mask",
        "geometry_group_valid_mask",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"MOT batch missing geometry fields: {missing}")

    geometry_rgb = batch["geometry_rgb"]
    geometry_pts3d = batch["geometry_pts3d"]
    if geometry_rgb.ndim != 7:
        raise ValueError(
            "geometry_rgb must be [B,G,4,V,C,H,W], "
            f"got {tuple(geometry_rgb.shape)}"
        )
    batch_size, groups, group_size, views, channels, height, width = geometry_rgb.shape
    if expected_frames is not None and groups != int(expected_frames):
        raise ValueError(
            f"geometry_rgb must contain G={int(expected_frames)} groups, got {groups}"
        )
    if group_size != MOT_GEOMETRY_GROUP_SIZE:
        raise ValueError(
            f"geometry_rgb must use S={MOT_GEOMETRY_GROUP_SIZE}, got {group_size}"
        )
    if views not in MOT_SUPPORTED_VIEW_COUNTS:
        raise ValueError(
            f"geometry batch must use native V in {MOT_SUPPORTED_VIEW_COUNTS}, got {views}"
        )
    if channels != 3:
        raise ValueError(f"geometry_rgb must have C=3, got {channels}")

    expected_points = (batch_size, groups, group_size, views, height, width, 3)
    if tuple(geometry_pts3d.shape) != expected_points:
        raise ValueError(
            f"geometry_pts3d must be {expected_points}, got {tuple(geometry_pts3d.shape)}"
        )
    if tuple(batch["geometry_point_valid_mask"].shape) != expected_points[:-1]:
        raise ValueError(
            "geometry_point_valid_mask shape "
            f"{tuple(batch['geometry_point_valid_mask'].shape)} does not match "
            f"geometry_pts3d {tuple(geometry_pts3d.shape)}"
        )
    expected_slots = (batch_size, groups, group_size)
    if tuple(batch["geometry_group_valid_mask"].shape) != expected_slots:
        raise ValueError(
            f"geometry_group_valid_mask must be {expected_slots}, "
            f"got {tuple(batch['geometry_group_valid_mask'].shape)}"
        )
    return batch_size, views


def validate_mot_geometry_batch(
    batch: dict[str, Any],
    action_sequence_length: int | None = None,
) -> dict[str, list[int]]:
    expected_frames = (
        None
        if action_sequence_length is None
        else mot_geometry_groups(int(action_sequence_length) - 1)
    )
    batch_size, _views = _validate_geometry_batch_shapes(
        batch,
        expected_frames=expected_frames,
    )
    if "has_pointcloud" not in batch:
        raise KeyError("MOT geometry batch missing has_pointcloud")
    has_pointcloud = torch.as_tensor(batch["has_pointcloud"], dtype=torch.bool)
    if tuple(has_pointcloud.shape) != (batch_size,):
        raise ValueError(
            f"has_pointcloud must be [B], got {tuple(has_pointcloud.shape)}"
        )
    if not bool(has_pointcloud.all().item()):
        raise ValueError("geometry-only training requires pointcloud labels for every sample")
    return {
        key: list(value.shape)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }


def validate_mot_batch_for_forward(batch: dict[str, Any], action_sequence_length: int | None = None) -> dict[str, list[int]]:
    required = (
        "latents",
        "actions",
        "action_loss_mask",
        "action_valid_mask",
        "text_emb",
        "stream_ids",
        "video_latent_loss_mask",
        "video_latent_valid_mask",
        "geometry_rgb",
        "geometry_pts3d",
        "geometry_point_valid_mask",
        "geometry_group_valid_mask",
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
            raise ValueError(f"action frames must be divisible by {MOT_ACTION_CHUNKS}, got {actions.shape[2]}")
        action_chunk_size = (actions.shape[2] // MOT_ACTION_CHUNKS - 1) * actions.shape[3]
    else:
        action_chunk_size = int(action_sequence_length) - 1
    expected_frames = mot_geometry_groups(action_chunk_size)
    expected_action_per_frame = mot_action_per_frame()
    batch_size, _, latent_frames, views = latents.shape[:4]
    geometry_batch_size, geometry_views = _validate_geometry_batch_shapes(
        batch,
        expected_frames=expected_frames,
    )
    if (geometry_batch_size, geometry_views) != (batch_size, views):
        raise ValueError(
            "geometry batch/view axes must match latents: "
            f"geometry={(geometry_batch_size, geometry_views)}, "
            f"latents={(batch_size, views)}"
        )

    if latent_frames != expected_frames:
        raise ValueError(f"latents must contain {expected_frames} MOT latent groups, got {latent_frames}")
    if actions.shape[2:] != (expected_frames, expected_action_per_frame, 1):
        raise ValueError(
            f"actions must use MOT latent-frame packing [{expected_frames},{expected_action_per_frame},1], "
            f"got {tuple(actions.shape[2:])}"
        )
    for name in ("action_loss_mask", "action_valid_mask"):
        if batch[name].shape != actions.shape:
            raise ValueError(f"{name} shape {tuple(batch[name].shape)} does not match actions {tuple(actions.shape)}")
    for name in ("video_latent_loss_mask", "video_latent_valid_mask"):
        if batch[name].shape != (batch_size, expected_frames):
            raise ValueError(f"{name} must be [B,{expected_frames}], got {tuple(batch[name].shape)}")
    if stream_ids.shape != (batch_size, views):
        raise ValueError(f"stream_ids must be [B,V], got {tuple(stream_ids.shape)}")
    valid_stream_ids = (
        (stream_ids >= STREAM_LEFT_WRIST) & (stream_ids <= STREAM_RIGHT_WRIST)
    )
    if not bool(valid_stream_ids.all().item()):
        raise ValueError(
            "stream_ids must use LEFT_WRIST=0, HEAD=1, RIGHT_WRIST=2"
        )
    if action_sequence_length is not None and int(action_sequence_length) != mot_action_sequence_length(action_chunk_size):
        raise ValueError(
            f"action_sequence_length must be {mot_action_sequence_length(action_chunk_size)}, got {action_sequence_length}"
        )
    return {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)}
