from types import SimpleNamespace

import torch

from distillation.model.objectives import dmd_surrogate_loss
from distillation.pipeline.self_gradient_forcing_training import (
    SelfGradientForcingTrainingPipeline,
)
from distillation.schema import (
    DMDUpdateSchedule,
    ReplayContext,
    VALossWeights,
    VAMasks,
    VAPrediction,
    VATimesteps,
)


class _Scheduler:
    num_train_timesteps = 2
    timesteps = torch.tensor([1000.0, 0.0])
    # 训练 scheduler 的最低离散 sigma 可以很小但不是严格 0；condition 是否
    # 保持 clean 必须靠 mask，而不是靠 timestep 数值 0。
    sigmas = torch.tensor([1.0, 0.1])


class _Trainer:
    train_scheduler_latent = _Scheduler()
    train_scheduler_action = _Scheduler()

    def __init__(self):
        self.batches = []

    def _prepare_joint_input_dict(self, batch, *, add_noise):
        assert add_noise is False
        self.batches.append(batch)
        return {
            "latent_dict": {
                "latent": batch["latents"],
                "video_latent_loss_mask": batch["video_latent_loss_mask"],
            },
            "action_dict": {
                "latent": batch["actions"],
                "action_loss_mask": batch["action_loss_mask"],
            },
            "geometry_dict": {"rgb": batch["geometry_rgb"]},
        }


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))
        self.calls = []

    def forward(self, input_dict, *, mode):
        self.calls.append((input_dict, mode, torch.is_grad_enabled()))
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


def test_replay_uses_recorded_batch_and_one_train_student_forward() -> None:
    trainer = _Trainer()
    student = _Model()
    real_score = _Model()
    fake_score = _Model()
    pipeline = SelfGradientForcingTrainingPipeline.__new__(
        SelfGradientForcingTrainingPipeline
    )
    pipeline.trainer = trainer
    pipeline.device = torch.device("cpu")
    pipeline.student = student
    pipeline.real_score = real_score
    pipeline.fake_score = fake_score
    pipeline.loss_weights = VALossWeights()
    pipeline.normalizer_eps = 1e-6
    pipeline.flow_target_eps = 1e-6
    pipeline.score_t_min = 0
    pipeline.score_t_max = 1

    generated = VAPrediction(
        video=torch.full((1, 1, 4, 1, 1, 1), 2.0),
        action=torch.full((1, 1, 4, 1, 1), 3.0),
    )
    noisy = VAPrediction(
        video=generated.video + 1.0,
        action=generated.action + 1.0,
    )
    video_mask = torch.tensor([[False, True, False, True]])
    action_mask = video_mask[:, None, :, None, None]
    recorded_batch = {
        "latents": generated.video,
        "actions": generated.action,
        "video_latent_loss_mask": video_mask,
        "action_loss_mask": action_mask,
        "geometry_rgb": torch.full((1, 4, 1), 9.0),
    }
    context = ReplayContext(
        batch=recorded_batch,
        timesteps=VATimesteps(
            video=torch.tensor([[0.0, 1000.0, 0.0, 1000.0]]),
            action=torch.tensor([[0.0, 1000.0, 0.0, 1000.0]]),
        ),
        noisy=noisy,
        generated=generated,
        masks=VAMasks(video_mask, action_mask),
    )

    loss, metrics = pipeline.replay_and_score(context)

    assert loss.requires_grad
    assert "distill/sgf_total_loss" in metrics
    assert len(student.calls) == 1
    assert student.calls[0][1:] == ("train", True)
    assert len(real_score.calls) == len(fake_score.calls) == 1
    assert real_score.calls[0][1:] == ("train", False)
    assert fake_score.calls[0][1:] == ("train", False)
    assert all(batch is recorded_batch for batch in trainer.batches)
    assert torch.all(student.calls[0][0]["geometry_dict"]["rgb"] == 9)
    score_input = real_score.calls[0][0]
    condition_frames = ~video_mask
    torch.testing.assert_close(
        score_input["latent_dict"]["latent"][:, :, condition_frames[0]],
        generated.video[:, :, condition_frames[0]],
    )
    torch.testing.assert_close(
        score_input["action_dict"]["latent"][:, :, condition_frames[0]],
        generated.action[:, :, condition_frames[0]],
    )


def test_dmd_surrogate_gradient_moves_student_toward_real_score() -> None:
    student_video = torch.zeros(1, 1, 1, 1, 1, 1, requires_grad=True)
    student = VAPrediction(
        video=student_video,
        action=torch.zeros(1, 1, 1, 1, 1, requires_grad=True),
    )
    fake = VAPrediction(
        video=torch.zeros_like(student.video),
        action=torch.zeros_like(student.action),
    )
    real = VAPrediction(
        video=torch.ones_like(student.video),
        action=torch.zeros_like(student.action),
    )
    masks = VAMasks(
        video=torch.tensor([[True]]),
        action=torch.zeros(1, 1, 1, 1, 1, dtype=torch.bool),
    )

    loss, _ = dmd_surrogate_loss(student, fake, real, masks)
    loss.backward()

    # 当前 student=0、real=1、fake=0。负梯度意味着 optimizer 的梯度下降
    # 会增大 student x0，也就是沿 real-score 指示的方向移动。
    assert student_video.grad.item() < 0


def test_dmd_update_schedule_is_four_fake_steps_then_one_student() -> None:
    schedule = DMDUpdateSchedule(fake_score_steps=4)
    assert [schedule.optimizer_for_step(step) for step in range(10)] == [
        "fake_score",
        "fake_score",
        "fake_score",
        "fake_score",
        "student",
        "fake_score",
        "fake_score",
        "fake_score",
        "fake_score",
        "student",
    ]


def test_fake_score_path_builds_gradient_only_for_fake_model() -> None:
    trainer = _Trainer()
    student = _Model()
    real_score = _Model()
    fake_score = _Model()
    pipeline = SelfGradientForcingTrainingPipeline.__new__(
        SelfGradientForcingTrainingPipeline
    )
    pipeline.trainer = trainer
    pipeline.device = torch.device("cpu")
    pipeline.student = student
    pipeline.real_score = real_score
    pipeline.fake_score = fake_score
    pipeline.loss_weights = VALossWeights()
    pipeline.normalizer_eps = 1e-6
    pipeline.flow_target_eps = 1e-6
    pipeline.score_t_min = 0
    pipeline.score_t_max = 1

    video_mask = torch.tensor([[False, True]])
    action_mask = video_mask[:, None, :, None, None]
    generated = VAPrediction(
        video=torch.full((1, 1, 2, 1, 1, 1), 2.0),
        action=torch.full((1, 1, 2, 1, 1), 3.0),
    )
    recorded_batch = {
        "latents": generated.video,
        "actions": generated.action,
        "video_latent_loss_mask": video_mask,
        "action_loss_mask": action_mask,
        "geometry_rgb": torch.ones(1, 2, 1),
    }
    context = ReplayContext(
        batch=recorded_batch,
        timesteps=VATimesteps(
            video=torch.zeros(1, 2),
            action=torch.zeros(1, 2),
        ),
        noisy=generated,
        generated=generated,
        masks=VAMasks(video_mask, action_mask),
    )

    torch.manual_seed(5)
    loss, metrics = pipeline.fake_score_step(context)
    loss.backward()

    assert torch.isfinite(loss)
    assert fake_score.calls[0][1:] == ("train", True)
    assert fake_score.scale.grad is not None
    assert not student.calls
    assert not real_score.calls
    assert "distill/fake_score_total_loss" in metrics
