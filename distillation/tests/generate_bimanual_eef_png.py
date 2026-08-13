#!/usr/bin/env python3
"""Generate a publication-style PNG of synchronized bimanual EEF trajectories.

Install:
    pip install numpy plotly kaleido

Run with the built-in demo:
    python generate_bimanual_eef_png.py -o bimanual_demo.png

Run with your data:
    python generate_bimanual_eef_png.py left_eef.npy right_eef.npy \
        --step 60 --frame-step 15 --axis-length 0.03 -o bimanual.png

Each arm accepts one of these absolute-pose array schemas:
    [N, 3]    = x, y, z
    [N, 7]    = x, y, z, qx, qy, qz, qw
    [N, 8]    = timestamp, x, y, z, qx, qy, qz, qw
    [N, 4, 4] = T_world_eef
    [N, 16]   = flattened T_world_eef

Supported files: .npy, .npz, .csv and .txt.

python generate_bimanual_eef_png.py \
    left_eef.npy \
    right_eef.npy \
    --quat-order xyzw \
    --step 60 \
    --frame-step 15 \
    --axis-length 0.03 \
    --width 1500 \
    --height 1000 \
    --scale 2 \
    -o distillation/tests/visu/bimanual.png
    
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as exc:
    raise SystemExit(
        "Missing Plotly. Install dependencies with: pip install numpy plotly kaleido"
    ) from exc


# -----------------------------------------------------------------------------
# Trajectory loading and pose conversion
# -----------------------------------------------------------------------------


def load_array(path: Path, npz_key: str | None = None) -> np.ndarray:
    """Load one trajectory array from .npy, .npz, .csv or .txt."""
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if npz_key is not None:
                if npz_key not in archive.files:
                    raise ValueError(
                        f"NPZ key {npz_key!r} not found. Available: {archive.files}"
                    )
                return np.asarray(archive[npz_key])

            # Select a conventional key automatically when possible.
            preferred_keys = ("poses", "eef_poses", "trajectory", "traj")
            key = next(
                (candidate for candidate in preferred_keys if candidate in archive.files),
                None,
            )
            if key is None:
                if len(archive.files) != 1:
                    raise ValueError(
                        "NPZ contains multiple arrays. Select one with --left-npz-key "
                        f"or --right-npz-key. Available: {archive.files}"
                    )
                key = archive.files[0]
            return np.asarray(archive[key])

    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        try:
            return np.loadtxt(path, delimiter=delimiter)
        except ValueError:
            # Tolerate one header row such as x,y,z,qx,qy,qz,qw.
            return np.loadtxt(path, delimiter=delimiter, skiprows=1)

    raise ValueError(f"Unsupported input extension: {suffix}")


def quaternion_xyzw_to_rotation(quaternions: np.ndarray) -> np.ndarray:
    """Convert [N,4] xyzw quaternions to [N,3,3] rotation matrices."""
    q = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        indices = np.flatnonzero(norms[:, 0] < 1e-12)[:5].tolist()
        raise ValueError(f"Zero-length quaternion at indices {indices}")

    x, y, z, w = (q / norms).T
    rotation = np.empty((len(q), 3, 3), dtype=np.float64)

    rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[:, 0, 1] = 2.0 * (x * y - z * w)
    rotation[:, 0, 2] = 2.0 * (x * z + y * w)

    rotation[:, 1, 0] = 2.0 * (x * y + z * w)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[:, 1, 2] = 2.0 * (y * z - x * w)

    rotation[:, 2, 0] = 2.0 * (x * z - y * w)
    rotation[:, 2, 1] = 2.0 * (y * z + x * w)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotation


def as_transforms(
    data: np.ndarray,
    quaternion_order: str = "xyzw",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an accepted trajectory schema to T_world_eef [N,4,4]."""
    data = np.asarray(data, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("Trajectory contains NaN or infinity")

    if data.ndim == 3 and data.shape[1:] == (4, 4):
        transforms = data.copy()
        timestamps = np.arange(len(data), dtype=np.float64)

    elif data.ndim == 2 and data.shape[1] == 16:
        transforms = data.reshape(-1, 4, 4).copy()
        timestamps = np.arange(len(data), dtype=np.float64)

    elif data.ndim == 2 and data.shape[1] in (3, 7, 8):
        timestamps = (
            data[:, 0].copy()
            if data.shape[1] == 8
            else np.arange(len(data), dtype=np.float64)
        )
        pose = data[:, 1:] if data.shape[1] == 8 else data

        transforms = np.repeat(np.eye(4)[None], len(data), axis=0)
        transforms[:, :3, 3] = pose[:, :3]

        if pose.shape[1] == 7:
            quaternion = pose[:, 3:7]
            if quaternion_order == "wxyz":
                quaternion = quaternion[:, [1, 2, 3, 0]]
            transforms[:, :3, :3] = quaternion_xyzw_to_rotation(quaternion)

    else:
        raise ValueError(
            "Unsupported trajectory shape. Expected [N,3], [N,7], [N,8], "
            f"[N,4,4] or [N,16], got {data.shape}."
        )

    if len(transforms) == 0:
        raise ValueError("Trajectory is empty")
    if not np.allclose(transforms[:, 3, :], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid T_world_eef: last row must be [0,0,0,1]")

    return transforms, timestamps


def automatic_axis_length(positions: np.ndarray) -> float:
    """Choose an EEF-frame axis length from the spatial trajectory extent."""
    extent = float(np.max(np.ptp(positions, axis=0))) if len(positions) > 1 else 0.0
    return 0.075 * extent if extent > 1e-9 else 0.05


# -----------------------------------------------------------------------------
# Built-in synchronized demo data
# -----------------------------------------------------------------------------


def euler_xyz_to_matrices(
    roll: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
) -> np.ndarray:
    """Convert intrinsic XYZ Euler angles to rotation matrices."""
    matrices = np.empty((len(roll), 3, 3), dtype=np.float64)
    for i, (rx, ry, rz) in enumerate(zip(roll, pitch, yaw)):
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        matrices[i] = np.array(
            [
                [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                [-sy, cy * sx, cy * cx],
            ]
        )
    return matrices


def demo_bimanual(num_poses: int = 140) -> tuple[np.ndarray, np.ndarray]:
    """Generate two synchronized absolute trajectories as [N,4,4]."""
    u = np.linspace(0.0, 1.0, num_poses)

    left_xyz = np.column_stack(
        (
            -0.32 + 0.46 * u + 0.035 * np.sin(2 * np.pi * u),
            0.22 + 0.10 * np.sin(np.pi * u) + 0.025 * np.sin(4 * np.pi * u),
            0.20 + 0.24 * np.sin(np.pi * u),
        )
    )
    right_xyz = np.column_stack(
        (
            -0.30 + 0.49 * u - 0.035 * np.sin(2 * np.pi * u),
            -0.22 - 0.10 * np.sin(np.pi * u) - 0.025 * np.sin(4 * np.pi * u),
            0.20 + 0.22 * np.sin(np.pi * u + 0.08),
        )
    )

    left = np.repeat(np.eye(4)[None], num_poses, axis=0)
    right = np.repeat(np.eye(4)[None], num_poses, axis=0)
    left[:, :3, 3] = left_xyz
    right[:, :3, 3] = right_xyz

    left[:, :3, :3] = euler_xyz_to_matrices(
        0.18 * np.sin(2 * np.pi * u),
        -0.45 + 0.55 * u,
        -0.55 + 1.25 * u,
    )
    right[:, :3, :3] = euler_xyz_to_matrices(
        -0.18 * np.sin(2 * np.pi * u),
        -0.42 + 0.50 * u,
        0.55 - 1.25 * u,
    )
    return left, right


# -----------------------------------------------------------------------------
# Plotly drawing helpers
# -----------------------------------------------------------------------------


def color_mix(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    fraction: float,
) -> str:
    rgb = [round(a + (b - a) * fraction) for a, b in zip(start, end)]
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def add_gradient_path(
    figure: go.Figure,
    positions: np.ndarray,
    start_color: tuple[int, int, int],
    end_color: tuple[int, int, int],
    name: str,
) -> None:
    """Draw a 3D trajectory with a start-to-end color gradient."""
    for i in range(1, len(positions)):
        figure.add_trace(
            go.Scatter3d(
                x=positions[i - 1 : i + 1, 0],
                y=positions[i - 1 : i + 1, 1],
                z=positions[i - 1 : i + 1, 2],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=i == 1,
                line=dict(
                    color=color_mix(
                        start_color,
                        end_color,
                        i / (len(positions) - 1),
                    ),
                    width=8,
                ),
                hoverinfo="skip",
            )
        )


def add_eef_frames(
    figure: go.Figure,
    transforms: np.ndarray,
    indices: np.ndarray,
    axis_length: float,
    opacity: float,
    line_width: int,
    show_legend: bool,
) -> None:
    """Draw multiple local EEF coordinate frames using three Plotly traces."""
    origins = transforms[indices, :3, 3]
    rotations = transforms[indices, :3, :3]

    axes = (
        ("EEF X", 0, "#e76ba7"),
        ("EEF Y", 1, "#f37b39"),
        ("EEF Z", 2, "#4ec576"),
    )
    for label, axis, color in axes:
        x: list[float | None] = []
        y: list[float | None] = []
        z: list[float | None] = []
        ends = origins + axis_length * rotations[:, :, axis]

        # None separates independent line segments inside one Scatter3d trace.
        for begin, end in zip(origins, ends):
            x.extend((float(begin[0]), float(end[0]), None))
            y.extend((float(begin[1]), float(end[1]), None))
            z.extend((float(begin[2]), float(end[2]), None))

        figure.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                name=label,
                legendgroup=label,
                showlegend=show_legend,
                opacity=opacity,
                line=dict(color=color, width=line_width),
                hoverinfo="skip",
            )
        )


def add_arm_endpoints(
    figure: go.Figure,
    positions: np.ndarray,
    prefix: str,
    colors: tuple[str, str],
) -> None:
    figure.add_trace(
        go.Scatter3d(
            x=[positions[0, 0], positions[-1, 0]],
            y=[positions[0, 1], positions[-1, 1]],
            z=[positions[0, 2], positions[-1, 2]],
            mode="markers+text",
            text=[f"{prefix} start", f"{prefix} goal"],
            textposition=["top left", "top right"],
            showlegend=False,
            marker=dict(
                size=8,
                color=list(colors),
                line=dict(color="white", width=2),
            ),
            textfont=dict(size=14, color="#25282b"),
            hoverinfo="skip",
        )
    )


def make_figure(
    left: np.ndarray,
    right: np.ndarray,
    current_step: int,
    frame_step: int,
    axis_length: float,
    title: str,
) -> go.Figure:
    """Create the complete synchronized bimanual 3D figure."""
    left_positions = left[:, :3, 3]
    right_positions = right[:, :3, 3]

    frame_indices = np.arange(0, len(left), frame_step)
    if frame_indices[-1] != len(left) - 1:
        frame_indices = np.append(frame_indices, len(left) - 1)

    figure = go.Figure()

    # Two paths use separate color palettes.
    add_gradient_path(
        figure,
        left_positions,
        start_color=(82, 195, 239),
        end_color=(49, 95, 229),
        name="Left EEF",
    )
    add_gradient_path(
        figure,
        right_positions,
        start_color=(244, 155, 69),
        end_color=(229, 80, 141),
        name="Right EEF",
    )

    # Sparse static EEF frames along both trajectories.
    add_eef_frames(
        figure, left, frame_indices, axis_length, opacity=0.58,
        line_width=4, show_legend=True,
    )
    add_eef_frames(
        figure, right, frame_indices, axis_length, opacity=0.58,
        line_width=4, show_legend=False,
    )

    # Larger frames highlight the selected synchronized step.
    selected = np.array([current_step])
    add_eef_frames(
        figure, left, selected, axis_length * 1.25, opacity=1.0,
        line_width=7, show_legend=False,
    )
    add_eef_frames(
        figure, right, selected, axis_length * 1.25, opacity=1.0,
        line_width=7, show_legend=False,
    )

    add_arm_endpoints(
        figure, left_positions, prefix="L", colors=("#54b9ec", "#315fe5")
    )
    add_arm_endpoints(
        figure, right_positions, prefix="R", colors=("#f39a47", "#e5508d")
    )

    # Current left/right EEF positions.
    figure.add_trace(
        go.Scatter3d(
            x=[left_positions[current_step, 0], right_positions[current_step, 0]],
            y=[left_positions[current_step, 1], right_positions[current_step, 1]],
            z=[left_positions[current_step, 2], right_positions[current_step, 2]],
            mode="markers",
            name=f"Current step {current_step}",
            marker=dict(
                size=11,
                color=["#318fec", "#e865a6"],
                line=dict(color="white", width=3),
            ),
            hoverinfo="skip",
        )
    )

    figure.update_layout(
        title=dict(
            text=(
                f"{title}<br>"
                f"<sup>synchronized step {current_step} / {len(left) - 1}</sup>"
            ),
            x=0.02,
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, b=0, t=90),
        legend=dict(
            orientation="h",
            x=1.0,
            xanchor="right",
            y=1.03,
            yanchor="bottom",
        ),
        scene=dict(
            xaxis=dict(
                title="X / m",
                showbackground=False,
                gridcolor="#e8ecef",
            ),
            yaxis=dict(
                title="Y / m",
                showbackground=False,
                gridcolor="#e8ecef",
            ),
            zaxis=dict(
                title="Z / m",
                showbackground=False,
                gridcolor="#e8ecef",
            ),
            # Preserve the actual X/Y/Z metric proportions.
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.55, z=1.05)),
        ),
        font=dict(
            family="Arial, sans-serif",
            color="#202124",
            size=14,
        ),
    )
    return figure


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save two synchronized absolute EEF trajectories as PNG"
    )
    parser.add_argument("left", nargs="?", type=Path, help="left-arm trajectory")
    parser.add_argument("right", nargs="?", type=Path, help="right-arm trajectory")
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("bimanual_eef_trajectory.png"),
    )
    parser.add_argument("--left-npz-key")
    parser.add_argument("--right-npz-key")
    parser.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=-1,
        help="highlighted synchronized step; -1 selects the midpoint",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        help="draw one static EEF frame every K poses",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        help="local EEF coordinate-axis length in meters",
    )
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="PNG resolution multiplier",
    )
    parser.add_argument(
        "--title",
        default="Bimanual absolute EEF trajectories · world frame",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if (args.left is None) != (args.right is None):
        raise ValueError(
            "Provide both LEFT and RIGHT trajectories, or neither for the demo"
        )

    if args.left is None:
        left, right = demo_bimanual()
    else:
        left_raw = load_array(args.left, args.left_npz_key)
        right_raw = load_array(args.right, args.right_npz_key)
        left, left_timestamps = as_transforms(left_raw, args.quat_order)
        right, right_timestamps = as_transforms(right_raw, args.quat_order)

        if len(left_timestamps) == len(right_timestamps) and not np.allclose(
            left_timestamps, right_timestamps, atol=1e-6
        ):
            print("Warning: timestamps differ; synchronizing by array index")

    if len(left) != len(right):
        raise ValueError(
            f"Synchronized trajectory lengths differ: left={len(left)}, right={len(right)}"
        )

    current_step = len(left) // 2 if args.step == -1 else args.step
    if not 0 <= current_step < len(left):
        raise ValueError(f"--step must be in [0, {len(left) - 1}]")

    frame_step = (
        max(1, int(np.ceil(len(left) / 10)))
        if args.frame_step is None
        else args.frame_step
    )
    all_positions = np.concatenate(
        (left[:, :3, 3], right[:, :3, 3]), axis=0
    )
    axis_length = (
        automatic_axis_length(all_positions)
        if args.axis_length is None
        else args.axis_length
    )

    if frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if axis_length <= 0:
        raise ValueError("--axis-length must be > 0")
    if args.width <= 0 or args.height <= 0 or args.scale <= 0:
        raise ValueError("--width, --height and --scale must be positive")

    figure = make_figure(
        left=left,
        right=right,
        current_step=current_step,
        frame_step=frame_step,
        axis_length=axis_length,
        title=args.title,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.write_image(
            args.output,
            width=args.width,
            height=args.height,
            scale=args.scale,
        )
    except Exception as exc:
        raise RuntimeError(
            "PNG export failed. Ensure Kaleido and Chrome/Chromium are available: "
            "pip install -U kaleido"
        ) from exc

    print(
        f"Wrote {args.output.resolve()} | poses={len(left)}, "
        f"step={current_step}, frame_step={frame_step}, "
        f"axis_length={axis_length:.4g} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
