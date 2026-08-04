from types import SimpleNamespace

import pytest
import torch

from distillation.configs.consistency_distillation import consistency_distillation_cfg
from distillation.pipeline.consistency_training import ConsistencyTrainingPipeline
from distillation.pipeline.utils import replace_va_streams
from distillation.scheduler import (
    consistency_prediction,
    sample_consistency_timesteps,
)
from distillation.schema import VAPrediction, VATimesteps
from distillation.self_rollout.provider import resolve_ground_truth_provider


class _Scheduler:
    timesteps = torch.tensor([10.0, 0.0])
    sigmas = torch.tensor([1.0, 0.0])


def test_consistency_stage_defaults_enable_cfg_normalization_and_periodic_rollout() -> None:
    config = consistency_distillation_cfg
    assert config.cfg_prob == 0.0
    assert (config.distill.cfg_min, config.distill.cfg_max) == (2.0, 10.0)
    assert config.distill.sigma_data == 0.5
    assert config.distill.rollout_interval == 500
    assert config.distill.rollout_video_num_steps == 2
    assert config.distill.rollout_action_num_steps == 2
    assert config.distill.rollout_horizon_frames == 3
    assert config.distill.rollout_gt_mode == "none"


def test_consistency_provider_mode_requires_explicit_provider() -> None:
    with pytest.raises(ValueError, match="explicit ground_truth_provider"):
        resolve_ground_truth_provider("provider", {})


def test_consistency_provider_mode_forwards_provider() -> None:
    provider = object()
    assert resolve_ground_truth_provider("provider", {}, provider) is provider


def test_video_consistency_prediction_uses_flash_wam_boundary_scaling() -> None:
    noisy = torch.full((1, 1, 2, 1, 1, 1), 2.0)
    flow = torch.ones_like(noisy)
    timesteps = torch.tensor([[10.0, 0.0]])

    prediction = consistency_prediction(
        flow,
        noisy,
        timesteps,
        _Scheduler(),
        sigma_data=0.5,
    )

    expected_noisy_frame = 0.2 * 2.0 + (0.5 / (1.25**0.5)) * 1.0
    torch.testing.assert_close(
        prediction[:, :, 0],
        torch.full_like(prediction[:, :, 0], expected_noisy_frame),
    )
    torch.testing.assert_close(prediction[:, :, 1], noisy[:, :, 1])


def test_consistency_timestep_pair_always_moves_to_lower_noise() -> None:
    scheduler = SimpleNamespace(
        timesteps=torch.tensor([1000.0, 750.0, 500.0, 250.0, 0.0]),
    )
    mask = torch.tensor([[True, True, True, False]])
    torch.manual_seed(7)

    current, following = sample_consistency_timesteps(
        scheduler,
        num_steps=2,
        shape=mask.shape,
        device=torch.device("cpu"),
        mask=mask,
    )

    assert torch.all(following[mask] < current[mask])
    assert torch.equal(current[~mask], torch.zeros_like(current[~mask]))
    assert torch.equal(following[~mask], torch.zeros_like(following[~mask]))


def test_replace_va_streams_builds_complete_native_forward_contract() -> None:
    clean = VAPrediction(
        video=torch.zeros(1, 2, 2, 1, 1, 1),
        action=torch.zeros(1, 3, 2, 1, 1),
    )
    noisy = VAPrediction(clean.video + 1, clean.action + 2)
    timesteps = VATimesteps(
        video=torch.tensor([[1000.0, 0.0]]),
        action=torch.tensor([[500.0, 0.0]]),
    )
    base_input = {
        "latent_dict": {"text_emb": torch.ones(1, 1, 1)},
        "action_dict": {"text_emb": torch.ones(1, 1, 1)},
        "geometry_dict": {"rgb": torch.ones(1, 2, 1)},
        "chunk_size": 4,
        "window_size": 4,
    }

    model_input = replace_va_streams(base_input, noisy, clean, timesteps)

    assert model_input["latent_dict"]["targets"].shape == clean.video.shape
    assert model_input["action_dict"]["targets"].shape == clean.action.shape
    assert model_input["action_dict"]["noisy_latents"] is noisy.action
    assert torch.equal(
        model_input["action_dict"]["cond_timesteps"],
        torch.zeros_like(timesteps.action),
    )
    assert model_input["geometry_dict"] is base_input["geometry_dict"]


class _Teacher:
    def __init__(self):
        self.text_conditions = []

    def __call__(self, input_dict, *, mode):
        assert mode == "train"
        text = input_dict["latent_dict"]["text_emb"]
        self.text_conditions.append(text.detach().clone())
        value = 4.0 if bool(text.any()) else 1.0
        return {
            "latent_pred": torch.full_like(
                input_dict["latent_dict"]["noisy_latents"],
                value,
            ),
            "action_pred": torch.full_like(
                input_dict["action_dict"]["noisy_latents"],
                value,
            ),
        }


def test_teacher_cfg_guides_video_only_and_accepts_empty_embedding_from_batch() -> None:
    pipeline = ConsistencyTrainingPipeline.__new__(ConsistencyTrainingPipeline)
    pipeline.teacher = _Teacher()
    pipeline.cfg_min = 3.0
    pipeline.cfg_max = 3.0
    pipeline.trainer = SimpleNamespace(
        _get_empty_text_emb=lambda: torch.full((1, 1), -1.0)
    )
    input_dict = {
        "latent_dict": {
            "text_emb": torch.ones(2, 1, 1),
            "noisy_latents": torch.zeros(2, 1, 1, 1, 1, 1),
        },
        "action_dict": {
            "text_emb": torch.ones(2, 1, 1),
            "noisy_latents": torch.zeros(2, 1, 1, 1, 1),
        },
    }
    batch = {"empty_text_emb": torch.zeros(2, 1, 1)}

    flow, scale = pipeline._teacher_cfg_flow(input_dict, batch)

    assert scale == 3.0
    assert len(pipeline.teacher.text_conditions) == 2
    assert bool(pipeline.teacher.text_conditions[0].all())
    assert not bool(pipeline.teacher.text_conditions[1].any())
    assert torch.all(flow.video == 10.0)
    assert torch.all(flow.action == 4.0)


class _FlowModel(torch.nn.Module):
    def __init__(self, scale: float, *, trainable: bool):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor(scale),
            requires_grad=trainable,
        )
        self.calls = 0

    def forward(self, input_dict, *, mode):
        assert mode == "train"
        # 模拟原生 MOT forward 的结构要求：即使 loss 在 pipeline 外计算，
        # action targets 仍必须存在，供输出 reshape 读取 shape。
        assert "targets" in input_dict["action_dict"]
        self.calls += 1
        return {
            "latent_pred": torch.ones_like(
                input_dict["latent_dict"]["noisy_latents"]
            )
            * self.scale,
            "action_pred": torch.ones_like(
                input_dict["action_dict"]["noisy_latents"]
            )
            * self.scale,
        }


def test_stage2_synthetic_batch_reaches_loss_with_student_only_gradient() -> None:
    scheduler = SimpleNamespace(
        timesteps=torch.tensor([10.0, 0.0]),
        sigmas=torch.tensor([1.0, 0.0]),
        training_target=lambda clean, noise, timesteps: noise - clean,
    )
    video_mask = torch.tensor([[False, True]])
    action_mask = video_mask[:, None, :, None, None]

    class _SyntheticTrainer:
        train_scheduler_latent = scheduler
        train_scheduler_action = scheduler

        @staticmethod
        def _prepare_joint_input_dict(batch, *, add_noise):
            assert add_noise is False
            return {
                "latent_dict": {
                    "latent": batch["latents"],
                    "text_emb": batch["text_emb"],
                    "video_latent_loss_mask": video_mask,
                },
                "action_dict": {
                    "latent": batch["actions"],
                    "text_emb": batch["text_emb"],
                    "action_loss_mask": action_mask,
                },
                "geometry_dict": {"rgb": torch.ones(1, 2, 1)},
                "chunk_size": 1,
                "window_size": 4,
            }

        @staticmethod
        def _get_empty_text_emb():
            return torch.zeros(1, 1, 1)

    student = _FlowModel(0.25, trainable=True)
    teacher = _FlowModel(0.5, trainable=False)
    ema = _FlowModel(0.75, trainable=False)
    config = SimpleNamespace(
        video_loss_weight=1.0,
        action_loss_weight=1.0,
        distill=SimpleNamespace(
            video_num_steps=2,
            action_num_steps=2,
            cfg_min=1.0,
            cfg_max=1.0,
            sigma_data=0.5,
            action_aware_weight=0.01,
        ),
    )
    pipeline = ConsistencyTrainingPipeline(
        config=config,
        trainer=_SyntheticTrainer(),
        device=torch.device("cpu"),
        student=student,
        teacher=teacher,
        ema_student=ema,
    )
    batch = {
        "latents": torch.zeros(1, 1, 2, 1, 1, 1),
        "actions": torch.zeros(1, 1, 2, 1, 1),
        "text_emb": torch.ones(1, 1, 1),
        "empty_text_emb": torch.zeros(1, 1, 1),
    }

    torch.manual_seed(3)
    loss, metrics = pipeline.compute_loss(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert student.calls == 1
    assert teacher.calls == 2  # conditional + unconditional CFG
    assert ema.calls == 1
    assert student.scale.grad is not None
    assert teacher.scale.grad is None
    assert ema.scale.grad is None
    assert "distill/consistency_training_loss" in metrics
