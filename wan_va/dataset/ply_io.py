"""Binary colored PLY output shared by dataset checks and inference."""

from pathlib import Path

import numpy as np


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray, valid: np.ndarray) -> int:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    valid = np.asarray(valid, dtype=bool)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be [H,W,3], got {points.shape}")
    if colors.shape != points.shape or valid.shape != points.shape[:2]:
        raise ValueError(
            f"PLY colors/valid must match points: points={points.shape}, colors={colors.shape}, valid={valid.shape}"
        )

    mask = valid & np.isfinite(points).all(axis=-1)
    vertices = points.reshape(-1, 3)[mask.reshape(-1)]
    vertex_colors = colors.reshape(-1, 3)[mask.reshape(-1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    payload = np.empty(
        len(vertices),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    if len(vertices):
        payload["x"] = vertices[:, 0]
        payload["y"] = vertices[:, 1]
        payload["z"] = vertices[:, 2]
        payload["red"] = vertex_colors[:, 0]
        payload["green"] = vertex_colors[:, 1]
        payload["blue"] = vertex_colors[:, 2]
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        payload.tofile(f)
    return int(len(vertices))
