"""Runtime reader and mask helpers for preprocessed MOT pointcloud stores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


STORE_FORMAT_V1 = "mot_preprocessed_pointcloud_v1"
STORE_FORMAT_V2 = "mot_preprocessed_pointcloud_v2"
DEFAULT_MASK_THRESHOLDS = (0.0, 0.03)
MASK_POLICY = "finite_xyz_positive_z_lumos_sigmoid_conf_gt_threshold"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def mask_key(threshold: float) -> str:
    value = float(threshold)
    if value == 0.0:
        return "conf0"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "conf" + text.replace(".", "p").replace("-", "m")


def mask_path_for_threshold(store_dir: str | Path, threshold: float) -> Path:
    return Path(store_dir) / "masks" / f"valid_mask_{mask_key(threshold)}.npy"


def default_mask_threshold(source_dataset: str) -> float:
    return 0.03 if str(source_dataset) == "lumos_lerobot" else 0.0


def confidence_to_probability(conf: np.ndarray) -> np.ndarray:
    conf = np.asarray(conf, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(conf, -88.0, 88.0)))


def valid_point_mask(
    points: np.ndarray,
    conf: np.ndarray | None,
    threshold: float,
    *,
    source_dataset: str | None = None,
) -> np.ndarray:
    points = np.asarray(points)
    mask = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    if conf is not None and str(source_dataset) == "lumos_lerobot":
        conf_prob = confidence_to_probability(conf)
        mask = mask & np.isfinite(conf_prob) & (conf_prob > float(threshold))
    return mask


def generate_masks(
    store_dir: str | Path,
    thresholds: Iterable[float] = DEFAULT_MASK_THRESHOLDS,
    *,
    source_dataset: str | None = None,
    conf: np.ndarray | None = None,
) -> list[Path]:
    store_dir = Path(store_dir)
    threshold_values = sorted({float(value) for value in thresholds})
    meta_path = store_dir / "meta.json"
    meta = _read_json(meta_path) if meta_path.is_file() else {}
    source_dataset = str(source_dataset if source_dataset is not None else meta.get("source_dataset", ""))
    if conf is None:
        with np.load(store_dir / "metadata.npz", allow_pickle=False) as metadata:
            conf = np.asarray(metadata["conf_224"], dtype=np.float32) if "conf_224" in metadata.files else None
    if conf is None and source_dataset == "lumos_lerobot" and any(value > 0.0 for value in threshold_values):
        raise ValueError("cannot generate Lumos confidence masks without conf_224 or an in-memory conf array")
    points = np.load(store_dir / "local_points_224.npy", mmap_mode="r")
    (store_dir / "masks").mkdir(parents=True, exist_ok=True)
    paths = []
    for threshold in threshold_values:
        out = mask_path_for_threshold(store_dir, threshold)
        np.save(
            out,
            valid_point_mask(
                points,
                conf,
                float(threshold),
                source_dataset=source_dataset,
            ).astype(bool),
        )
        paths.append(out)
    if meta_path.is_file():
        meta["mask_policy"] = MASK_POLICY
        meta["mask_source_dataset"] = source_dataset
        _write_json(meta_path, meta)
    return paths


def set_active_mask(store_dir: str | Path, *, threshold: float) -> Path:
    store_dir = Path(store_dir)
    mask_path = mask_path_for_threshold(store_dir, threshold)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    meta_path = store_dir / "meta.json"
    meta = _read_json(meta_path)
    meta["active_mask"] = str(mask_path.relative_to(store_dir))
    meta["active_mask_threshold"] = float(threshold)
    meta["mask_policy"] = MASK_POLICY
    _write_json(meta_path, meta)
    return mask_path


@dataclass
class PointStore:
    root: Path
    points: np.ndarray
    valid_mask: np.ndarray
    metadata: np.lib.npyio.NpzFile
    meta: dict
    active_mask_path: Path

    @classmethod
    def open(cls, root: str | Path, mask_threshold: float | None = None) -> "PointStore":
        root = Path(root)
        success = root / "_SUCCESS"
        if not success.is_file():
            raise FileNotFoundError(success)
        meta = _read_json(root / "meta.json")
        if meta.get("format") not in {STORE_FORMAT_V1, STORE_FORMAT_V2}:
            raise ValueError(f"unsupported pointcloud store format: {meta.get('format')}")
        mask_path = mask_path_for_threshold(root, float(mask_threshold)) if mask_threshold is not None else root / meta["active_mask"]
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        return cls(
            root=root,
            points=np.load(root / meta.get("points", "local_points_224.npy"), mmap_mode="r"),
            valid_mask=np.load(mask_path, mmap_mode="r"),
            metadata=np.load(root / meta.get("metadata", "metadata.npz"), allow_pickle=False),
            meta=meta,
            active_mask_path=mask_path,
        )

    @property
    def num_rows(self) -> int:
        return int(self.points.shape[0])

    def close(self) -> None:
        self.metadata.close()
        for array in (self.points, self.valid_mask):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()

    def row_for_episode_frame(self, frame_id: int) -> int:
        frame_id = int(frame_id)
        if frame_id < 0 or frame_id >= self.num_rows:
            return -1
        return frame_id
