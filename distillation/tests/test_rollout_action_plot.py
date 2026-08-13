import numpy as np

from distillation.rollout_visualization import write_action_plot

def rotation_to_quaternion_xyzw(rotations: np.ndarray) -> np.ndarray:
    """Convert [N,3,3] matrices to xyzw quaternions for demo generation."""
    quaternions = np.empty((len(rotations), 4), dtype=np.float64)
    for i, matrix in enumerate(rotations):
        trace = np.trace(matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            quaternions[i] = (
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
                0.25 * s,
            )
        else:
            diagonal = np.diag(matrix)
            axis = int(np.argmax(diagonal))
            if axis == 0:
                s = np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
                quaternions[i] = (0.25*s, (matrix[0,1]+matrix[1,0])/s, (matrix[0,2]+matrix[2,0])/s, (matrix[2,1]-matrix[1,2])/s)
            elif axis == 1:
                s = np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
                quaternions[i] = ((matrix[0,1]+matrix[1,0])/s, 0.25*s, (matrix[1,2]+matrix[2,1])/s, (matrix[0,2]-matrix[2,0])/s)
            else:
                s = np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
                quaternions[i] = ((matrix[0,2]+matrix[2,0])/s, (matrix[1,2]+matrix[2,1])/s, 0.25*s, (matrix[1,0]-matrix[0,1])/s)
    return quaternions

def euler_xyz_to_matrices(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    matrices = np.empty((len(roll), 3, 3), dtype=np.float64)
    for i, (rx, ry, rz) in enumerate(zip(roll, pitch, yaw)):
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        matrices[i] = np.array(
            [
                [cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                [-sy, cy*sx, cy*cx],
            ]
        )
    return matrices


def demo_bimanual(n: int = 140) -> tuple[np.ndarray, np.ndarray]:
    """Generate two synchronized, spatially separated [N,7] EEF trajectories."""
    u = np.linspace(0.0, 1.0, n)
    left_xyz = np.column_stack(
        (
            -0.32 + 0.46*u + 0.035*np.sin(2*np.pi*u),
            0.22 + 0.10*np.sin(np.pi*u) + 0.025*np.sin(4*np.pi*u),
            0.20 + 0.24*np.sin(np.pi*u),
        )
    )
    right_xyz = np.column_stack(
        (
            -0.30 + 0.49*u - 0.035*np.sin(2*np.pi*u),
            -0.22 - 0.10*np.sin(np.pi*u) - 0.025*np.sin(4*np.pi*u),
            0.20 + 0.22*np.sin(np.pi*u + 0.08),
        )
    )
    left_rot = euler_xyz_to_matrices(
        0.18*np.sin(2*np.pi*u), -0.45+0.55*u, -0.55+1.25*u
    )
    right_rot = euler_xyz_to_matrices(
        -0.18*np.sin(2*np.pi*u), -0.42+0.50*u, 0.55-1.25*u
    )
    left = np.column_stack((left_xyz, rotation_to_quaternion_xyzw(left_rot)))
    right = np.column_stack((right_xyz, rotation_to_quaternion_xyzw(right_rot)))
    return left, right

def test_action_plot_separates_gt_and_prediction(tmp_path) -> None:
    left, right = demo_bimanual()
    gt = np.zeros((len(left), 16), dtype=np.float32)
    gt[:, :7] = left
    gt[:, 8:15] = right

    prediction_start = 3
    pred = gt.copy()
    rng = np.random.default_rng(0)
    position_columns = [0, 1, 2, 8, 9, 10]
    pred[prediction_start:, position_columns] += rng.normal(
        scale=0.003,
        size=(len(pred) - prediction_start, len(position_columns)),
    )
    output = tmp_path / "actions.html"

    write_action_plot(
        output,
        gt,
        pred,
        np.arange(len(gt), dtype=np.float32),
        fps=10.0,
        prediction_start=prediction_start,
        step=500,
    )

    html = output.read_text(encoding="utf-8")
    assert "GT Left trajectory" in html
    assert "Prediction Left trajectory" in html
    assert "GT time (s)" in html
    assert "Prediction time (s)" in html
    assert "Prediction starts" in html
    assert "Ground Truth" in html
    assert '"scene":"scene2"' in html
