import sys
from types import ModuleType, SimpleNamespace

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
from distillation.self_rollout import SelfRolloutRecorder


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
                "text_emb": batch["text_emb"],
            },
            "action_dict": {
                "latent": batch["actions"],
                "action_loss_mask": batch["action_loss_mask"],
                "text_emb": batch["text_emb"],
            },
            "geometry_dict": {"rgb": batch["geometry_rgb"]},
        }

    def _get_empty_text_emb(self):
        return torch.zeros(1, 1, 1)


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


class _TextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_dict, *, mode):
        self.calls.append((input_dict, mode, torch.is_grad_enabled()))
        text = input_dict["latent_dict"]["text_emb"].mean()
        return {
            "latent_pred": torch.ones_like(
                input_dict["latent_dict"]["noisy_latents"]
            )
            * text,
            "action_pred": torch.ones_like(
                input_dict["action_dict"]["noisy_latents"]
            )
            * (text + 5),
        }


def test_sgf_teacher_cfg_guides_video_only() -> None:
    pipeline = SelfGradientForcingTrainingPipeline.__new__(
        SelfGradientForcingTrainingPipeline
    )
    pipeline.trainer = _Trainer()
    pipeline.real_score = _TextModel()
    pipeline.teacher_cfg_min = 3.0
    pipeline.teacher_cfg_max = 3.0
    input_dict = {
        "latent_dict": {
            "text_emb": torch.full((1, 1, 1), 2.0),
            "noisy_latents": torch.zeros(1, 1, 1, 1, 1, 1),
        },
        "action_dict": {
            "text_emb": torch.full((1, 1, 1), 2.0),
            "noisy_latents": torch.zeros(1, 1, 1, 1, 1),
        },
    }

    with torch.no_grad():
        flow, scale = pipeline._teacher_cfg_flow(input_dict, {})

    assert scale == 3.0
    torch.testing.assert_close(flow.video, torch.full_like(flow.video, 6.0))
    torch.testing.assert_close(flow.action, torch.full_like(flow.action, 7.0))
    assert len(pipeline.real_score.calls) == 2
    assert all(call[1:] == ("train", False) for call in pipeline.real_score.calls)


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
    pipeline.teacher_cfg_min = 2.0
    pipeline.teacher_cfg_max = 2.0

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
        "text_emb": torch.ones(1, 1, 1),
    }
    teacher_batch = {
        **recorded_batch,
        "latents": torch.full_like(generated.video, 20.0),
        "actions": torch.full_like(generated.action, 30.0),
        "geometry_rgb": torch.full((1, 4, 1), 5.0),
    }
    context = ReplayContext(
        student_batch=recorded_batch,
        teacher_batch=teacher_batch,
        rollout_timesteps=VATimesteps(
            video=torch.tensor([[0.0, 1000.0, 0.0, 1000.0]]),
            action=torch.tensor([[0.0, 1000.0, 0.0, 1000.0]]),
        ),
        rollout_noisy=noisy,
        pred_clean=generated,
        teacher_clean=VAPrediction(
            teacher_batch["latents"],
            teacher_batch["actions"],
        ),
        masks=VAMasks(video_mask, action_mask),
    )

    loss, metrics = pipeline.replay_and_score(context)

    assert loss.requires_grad
    assert "distill/sgf_total_loss" in metrics
    assert len(student.calls) == 1
    assert student.calls[0][1:] == ("train", True)
    assert len(real_score.calls) == 3
    assert len(fake_score.calls) == 1
    assert all(call[1:] == ("train", False) for call in real_score.calls)
    assert fake_score.calls[0][1:] == ("train", False)
    assert trainer.batches[0] is recorded_batch
    assert trainer.batches[1] is teacher_batch
    assert trainer.batches[2] is recorded_batch
    assert torch.all(student.calls[0][0]["geometry_dict"]["rgb"] == 9)
    teacher_input = real_score.calls[0][0]
    assert torch.all(teacher_input["geometry_dict"]["rgb"] == 5)
    torch.testing.assert_close(
        teacher_input["latent_dict"]["latent"],
        teacher_batch["latents"],
    )
    torch.testing.assert_close(
        teacher_input["latent_dict"]["noisy_latents"][:, :, video_mask[0]],
        noisy.video[:, :, video_mask[0]],
    )
    score_input = real_score.calls[2][0]
    condition_frames = ~video_mask
    torch.testing.assert_close(
        score_input["latent_dict"]["latent"][:, :, condition_frames[0]],
        generated.video[:, :, condition_frames[0]],
    )
    torch.testing.assert_close(
        score_input["action_dict"]["latent"][:, :, condition_frames[0]],
        generated.action[:, :, condition_frames[0]],
    )
    assert metrics["distill/sgf_teacher_cfg_scale"].item() == 2.0


def test_build_replay_context_replaces_only_target_predictions() -> None:
    pipeline = SelfGradientForcingTrainingPipeline.__new__(
        SelfGradientForcingTrainingPipeline
    )
    pipeline.rollout_horizon_frames = 3
    frames = 8
    latents = torch.arange(frames, dtype=torch.float32).reshape(
        1, 1, frames, 1, 1, 1
    )
    actions = torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1)
    geometry = torch.arange(frames, dtype=torch.float32).reshape(1, frames, 1)
    action_valid = torch.ones_like(actions, dtype=torch.bool)
    action_valid[:, :, 6] = False
    batch = {
        "latents": latents,
        "actions": actions,
        "geometry_rgb": geometry,
        "video_latent_valid_mask": torch.ones(1, frames, dtype=torch.bool),
        "action_valid_mask": action_valid,
        "video_latent_loss_mask": torch.ones(1, frames, dtype=torch.bool),
        "action_loss_mask": torch.ones_like(actions, dtype=torch.bool),
    }
    rollout = SimpleNamespace(
        pred_latents=latents + 100,
        pred_actions=actions + 200,
        pred_geometry_rgb=geometry + 300,
    )
    recorder = SelfRolloutRecorder(video_step=0, action_step=0)
    for frame_id in (5, 6, 7):
        recorder.observe(
            "video",
            frame_id=frame_id,
            step_index=0,
            timestep=torch.tensor(700.0),
            sample=torch.full((1, 1, 1, 1, 1, 1), float(frame_id + 10)),
        )
        recorder.observe(
            "action",
            frame_id=frame_id,
            step_index=0,
            timestep=torch.tensor(400.0),
            sample=torch.full((1, 1, 1, 1, 1), float(frame_id + 20)),
        )

    context = pipeline._build_replay_context(
        batch,
        rollout,
        recorder,
        history_frames=4,
    )

    torch.testing.assert_close(context.pred_clean.video[:, :, :5], latents[:, :, :5])
    torch.testing.assert_close(
        context.pred_clean.video[:, :, 5:8],
        rollout.pred_latents[:, :, 5:8],
    )
    torch.testing.assert_close(context.pred_clean.action[:, :, :5], actions[:, :, :5])
    torch.testing.assert_close(
        context.teacher_clean.video,
        latents,
    )
    torch.testing.assert_close(
        context.teacher_batch["geometry_rgb"],
        geometry,
    )
    torch.testing.assert_close(
        context.student_batch["geometry_rgb"][:, 5:8],
        rollout.pred_geometry_rgb[:, 5:8],
    )
    assert context.masks.video.tolist() == [[False, False, False, False, False, True, True, True]]
    assert not context.masks.action[:, :, 6].any()
    assert torch.all(context.rollout_timesteps.video[:, 5:8] == 700)
    assert context.rollout_timesteps.action[0, 6] == 0
    assert context.rollout_noisy.video[0, 0, 5, 0, 0, 0] == 15
    torch.testing.assert_close(
        context.rollout_noisy.action[:, :, 6],
        context.pred_clean.action[:, :, 6],
    )


def test_generate_context_uses_one_self_rollout_and_not_legacy_inference(
    monkeypatch,
) -> None:
    pipeline = SelfGradientForcingTrainingPipeline.__new__(
        SelfGradientForcingTrainingPipeline
    )
    pipeline.config = SimpleNamespace()
    pipeline.trainer = _Trainer()
    pipeline.student = object()
    pipeline.device = torch.device("cpu")
    pipeline.rollout_video_num_steps = 2
    pipeline.rollout_action_num_steps = 2
    pipeline.rollout_horizon_frames = 3
    batch = {
        "latents": torch.zeros(1, 1, 8, 1, 1, 1),
        "actions": torch.zeros(1, 1, 8, 1, 1),
        "geometry_rgb": torch.zeros(1, 8, 1),
        "video_latent_valid_mask": torch.ones(1, 8, dtype=torch.bool),
        "action_valid_mask": torch.ones(1, 1, 8, 1, 1, dtype=torch.bool),
        "video_latent_loss_mask": torch.zeros(1, 8, dtype=torch.bool),
        "action_loss_mask": torch.zeros(1, 1, 8, 1, 1, dtype=torch.bool),
    }
    calls = []

    def fake_self_rollout(local_batch, **kwargs):
        calls.append((local_batch, kwargs))
        recorder = kwargs["recorder"]
        for frame_id in (5, 6, 7):
            recorder.observe(
                "video",
                frame_id=frame_id,
                step_index=recorder.record_steps["video"],
                timestep=torch.tensor(600.0),
                sample=torch.ones(1, 1, 1, 1, 1, 1) * frame_id,
            )
            recorder.observe(
                "action",
                frame_id=frame_id,
                step_index=recorder.record_steps["action"],
                timestep=torch.tensor(300.0),
                sample=torch.ones(1, 1, 1, 1, 1) * frame_id,
            )
        return SimpleNamespace(
            pred_latents=local_batch["latents"] + 10,
            pred_actions=local_batch["actions"] + 20,
            pred_geometry_rgb=local_batch["geometry_rgb"] + 30,
        )

    monkeypatch.setattr(
        "distillation.pipeline.self_gradient_forcing_training.self_rollout",
        fake_self_rollout,
    )
    mot_spec_module = ModuleType("wan_va.mot_spec")
    mot_spec_module.mot_spec_from_config = lambda _config: SimpleNamespace(
        history_latent_frames=4
    )
    monkeypatch.setitem(sys.modules, "wan_va.mot_spec", mot_spec_module)

    context = pipeline.generate_and_record_context(batch)

    assert len(calls) == 1
    assert calls[0][0] is batch
    assert calls[0][1]["transformer"] is pipeline.student
    assert calls[0][1]["rollout_frames"] == 3
    assert context.masks.video.tolist() == [[False, False, False, False, False, True, True, True]]
    assert "inference.mot_inference" not in sys.modules


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
    pipeline.teacher_cfg_min = 2.0
    pipeline.teacher_cfg_max = 2.0

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
        "text_emb": torch.ones(1, 1, 1),
    }
    context = ReplayContext(
        student_batch=recorded_batch,
        teacher_batch=recorded_batch,
        rollout_timesteps=VATimesteps(
            video=torch.zeros(1, 2),
            action=torch.zeros(1, 2),
        ),
        rollout_noisy=generated,
        pred_clean=generated,
        teacher_clean=generated,
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
