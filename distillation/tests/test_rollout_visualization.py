from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from distillation.configs.consistency_distillation import (
    consistency_distillation_cfg,
)
from distillation.configs.self_gradient_forcing_dmd import (
    self_gradient_forcing_dmd_cfg,
)
from distillation.rollout_visualization import (
    RolloutVisualizer,
    absolute_action_trajectories,
    compose_comparison_frame,
    rollout_output_dir,
    rollout_sample_dir,
    should_visualize_rollout,
    write_action_plot,
)
from distillation.trainer.base import DistillationTrainerBase


def test_rollout_interval_and_output_path(tmp_path) -> None:
    assert not should_visualize_rollout(0, 500)
    assert not should_visualize_rollout(499, 500)
    assert should_visualize_rollout(500, 500)
    assert not should_visualize_rollout(500, 0)
    assert rollout_output_dir(tmp_path, 500) == (
        tmp_path / "rollouts" / "step_00000500"
    )
    assert rollout_sample_dir(tmp_path, 500, 2) == (
        tmp_path / "rollouts" / "step_00000500" / "sample_002"
    )
    assert consistency_distillation_cfg.distill.rollout_visualization_interval == 500
    assert self_gradient_forcing_dmd_cfg.distill.rollout_visualization_interval == 500


def test_nonzero_rank_runs_rollout_without_writing(tmp_path) -> None:
    calls = []
    model = nn.Linear(1, 1)
    model.train()
    config = SimpleNamespace(
        rank=1,
        train_seed=4,
        save_root=str(tmp_path),
        distill=SimpleNamespace(rollout_visualization_interval=5),
    )
    visualizer = RolloutVisualizer(
        config=config,
        device=torch.device("cpu"),
        rollout=lambda batch: calls.append(batch) or object(),
        rollout_model=lambda: model,
        get_vae=lambda: (_ for _ in ()).throw(AssertionError("rank 1 decoded")),
    )

    batch = {"marker": torch.tensor([1])}
    assert visualizer.maybe_run(5, batch) is None
    assert len(calls) == 1
    assert calls[0] is batch
    assert model.training
    assert not (tmp_path / "rollouts").exists()


def test_train_reuses_the_training_batch_for_rollout() -> None:
    trainer = DistillationTrainerBase.__new__(DistillationTrainerBase)
    batch = {"sample": torch.tensor([3])}
    seen = []
    trainer.config = SimpleNamespace(num_steps=1)
    trainer.step = 0
    trainer.optimizer_step = 0
    trainer.gradient_accumulation_steps = 1
    trainer.save_interval = 100
    trainer._last_checkpoint_step = 1
    trainer._get_next_batch = lambda: batch

    def train_step(value, batch_index):
        assert value is batch
        assert batch_index == 0
        trainer.optimizer_step += 1
        processed_batch = {"processed": value["sample"]}
        return {
            "optimizer_step_event": True,
            "batch": processed_batch,
        }

    trainer._train_step = train_step
    trainer.rollout_visualizer = SimpleNamespace(
        maybe_run=lambda step, value: seen.append((step, value))
    )
    trainer.save_checkpoint = lambda: (_ for _ in ()).throw(
        AssertionError("unexpected checkpoint")
    )

    trainer.train()

    assert seen == [(1, {"processed": batch["sample"]})]


def test_comparison_video_frame_is_left_prediction_right_gt() -> None:
    prediction = np.full((64, 80, 3), 17, dtype=np.uint8)
    ground_truth = np.full((64, 80, 3), 231, dtype=np.uint8)
    frame = compose_comparison_frame(
        prediction,
        ground_truth,
        view_name="head",
        step=12,
        is_prediction=True,
    )

    assert frame.shape == (64, 160, 3)
    assert np.all(frame[-1, :80] == 17)
    assert np.all(frame[-1, 80:] == 231)


def _identity_relative(position: tuple[float, float, float]) -> torch.Tensor:
    arm = torch.tensor([*position, 1, 0, 0, 0, 1, 0, 0], dtype=torch.float32)
    return torch.cat((arm, arm))


def test_action_trajectory_data_is_converted_to_absolute_coordinates() -> None:
    frames = 3
    gt_relative = torch.stack(
        [_identity_relative((1.0, 0.0, 0.0)) for _ in range(frames)], dim=1
    )[:, :, None, None]
    references = torch.zeros(16, frames, 1, 1)
    references[0] = 10.0
    references[3 + 3] = 1.0
    references[8] = 20.0
    references[8 + 3 + 3] = 1.0
    batch = {
        "actions": gt_relative[None],
        "action_valid_mask": torch.ones_like(gt_relative[None], dtype=torch.bool),
        "action_reference_states": references[None],
        "action_q01": -torch.ones(1, 20),
        "action_q99": torch.ones(1, 20),
    }
    prediction = _identity_relative((3.0, 0.0, 0.0))[:, None, None, None, None]

    gt, pred, times, prediction_start = absolute_action_trajectories(
        batch,
        prediction,
        (2,),
    )

    assert gt.shape == pred.shape == (3, 16)
    assert np.allclose(gt[:, 0], 11.0)
    assert np.allclose(gt[:, 8], 21.0)
    assert np.allclose(pred[:2, 0], 11.0)
    assert np.isclose(pred[2, 0], 13.0)
    assert np.array_equal(times, np.arange(3, dtype=np.float32))
    assert prediction_start == 2


def test_action_plot_writes_interactive_html(tmp_path) -> None:
    action = np.zeros((3, 16), dtype=np.float32)
    action[:, 6] = 1.0
    action[:, 14] = 1.0
    output = tmp_path / "actions.html"

    write_action_plot(
        output,
        action,
        action.copy(),
        np.arange(3, dtype=np.float32),
        fps=10.0,
        prediction_start=2,
        step=500,
    )

    html = output.read_text(encoding="utf-8")
    assert "plotly.js" in html
    assert "GT Left trajectory" in html
    assert "Prediction Left trajectory" in html
    assert "GT time (s)" in html
    assert "Prediction time (s)" in html
