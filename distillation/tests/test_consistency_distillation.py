from types import SimpleNamespace

import torch
from torch import nn

from distillation.model.autoregressive_mot import (
    AutoregressiveVAMOTTransformer3DModel,
)
from distillation.model.consistency import ConsistencyModel
from distillation.model.wan_wrapper import WanDiffusionWrapper
from wan_va.modules.model_va_mot import VAMOTTransformer3DModel
from wan_va.modules.mot_attention import build_x_metadata


class _TrackingModel(nn.Module):
    def __init__(self, scale: float, *, trainable: bool) -> None:
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(scale),
            requires_grad=trainable,
        )
        self.calls = []

    def forward(self, input_dict, *, mode: str):
        assert mode == "train"
        self.calls.append(input_dict)
        return {
            "latent_pred": input_dict["latent_dict"]["noisy_latents"] * self.scale,
            "action_pred": input_dict["action_dict"]["noisy_latents"] * self.scale,
        }


def _config():
    return SimpleNamespace(
        video_loss_weight=1.0,
        action_loss_weight=1.0,
        snr_shift=1.0,
        action_snr_shift=1.0,
        distill=SimpleNamespace(
            ema_decay=0.99,
            student_init=None,
            resume_from=None,
            video_num_steps=2,
            action_num_steps=2,
            cfg_min=2.0,
            cfg_max=2.0,
            reuse_teacher_noise=False,
            sigma_data=0.5,
            action_aware_weight=0.01,
            generation_shape={"history_frames": 0},
            rollout_denoising_step_list=SimpleNamespace(
                video=[1000, 500],
                action=[1000, 500],
            ),
            rollout_horizon_frames=1,
            rollout_num_frame_per_block=1,
            rollout_per_rank_exit_step=True,
        ),
    )


def _batch_and_input():
    video = torch.tensor([[[[[[1.0]]], [[[2.0]]]]]])
    action = torch.tensor([[[[[1.0]], [[2.0]]]]])
    video_mask = torch.tensor([[False, True]])
    action_mask = torch.tensor([[[[[False]], [[True]]]]])
    text = torch.ones(1, 1, 1)
    batch = {
        "latents": video,
        "actions": action,
        "text_emb": text,
    }
    base_input = {
        "latent_dict": {
            "latent": video,
            "text_emb": text,
            "video_latent_loss_mask": video_mask,
            "video_latent_valid_mask": torch.ones_like(video_mask),
        },
        "action_dict": {
            "latent": action,
            "text_emb": text,
            "action_loss_mask": action_mask,
            "action_valid_mask": torch.ones_like(action_mask),
        },
        "stream_ids": torch.zeros(1, 1, dtype=torch.long),
        "chunk_size": 1,
        "window_size": 4,
    }
    return batch, base_input


def test_consistency_step_preserves_forward_and_gradient_boundaries(
    monkeypatch,
) -> None:
    student = _TrackingModel(0.1, trainable=True)
    teacher = _TrackingModel(0.2, trainable=False)
    ema = _TrackingModel(0.3, trainable=False)
    loaded_models = iter([student, teacher, ema])
    loaded_paths = []

    def fake_load_model(checkpoint_path, config, *, autoregressive):
        del config, autoregressive
        loaded_paths.append(checkpoint_path)
        return next(loaded_models)

    monkeypatch.setattr(
        WanDiffusionWrapper,
        "_load_model",
        staticmethod(fake_load_model),
    )
    model = ConsistencyModel(
        config=_config(),
        device=torch.device("cpu"),
        student_init="student-checkpoint",
        teacher_checkpoint="teacher-checkpoint",
    )
    assert loaded_paths == [
        "student-checkpoint",
        "teacher-checkpoint",
        "student-checkpoint",
    ]
    assert model.generate.model is student
    assert model.teacher.model is teacher
    assert model.target.model is ema
    batch, base_input = _batch_and_input()

    torch.manual_seed(7)
    result = model.compute_step(
        batch,
        base_input=base_input,
        empty_text_emb=torch.zeros(1, 1, 1),
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert len(student.calls) == 1
    assert len(teacher.calls) == 2
    assert len(ema.calls) == 1
    assert student.scale.grad is not None
    assert teacher.scale.grad is None
    assert ema.scale.grad is None

    student_input = student.calls[0]
    assert student_input["latent_dict"]["timesteps"][0, 0] == 0
    assert student_input["action_dict"]["timesteps"][0, 0] == 0
    assert torch.equal(
        student_input["latent_dict"]["noisy_latents"][:, :, :1],
        batch["latents"][:, :, :1],
    )
    assert torch.equal(
        student_input["action_dict"]["noisy_latents"][:, :, :1],
        batch["actions"][:, :, :1],
    )


@torch.inference_mode()
def test_consistency_teacher_renoise_can_reuse_or_resample_noise(
    monkeypatch,
) -> None:
    for reuse_teacher_noise in (False, True):
        student = _TrackingModel(0.1, trainable=True)
        teacher = _TrackingModel(0.2, trainable=False)
        ema = _TrackingModel(0.3, trainable=False)
        loaded_models = iter([student, teacher, ema])
        monkeypatch.setattr(
            WanDiffusionWrapper,
            "_load_model",
            staticmethod(lambda *args, **kwargs: next(loaded_models)),
        )
        config = _config()
        config.distill.reuse_teacher_noise = reuse_teacher_noise
        model = ConsistencyModel(
            config=config,
            device=torch.device("cpu"),
            student_init="student-checkpoint",
            teacher_checkpoint="teacher-checkpoint",
        )
        noises = []

        def track_noise(original_samples, noise, timestep, t_dim=2):
            del original_samples, timestep, t_dim
            noises.append(noise)
            return noise

        model.train_scheduler_latent.add_noise = track_noise
        model.train_scheduler_action.add_noise = track_noise
        batch, base_input = _batch_and_input()
        model.compute_step(
            batch,
            base_input=base_input,
            empty_text_emb=torch.zeros(1, 1, 1),
        )

        assert (noises[0] is noises[2]) is reuse_teacher_noise
        assert (noises[1] is noises[3]) is reuse_teacher_noise


def test_consistency_rollout_delegates_to_shared_pipeline() -> None:
    model = ConsistencyModel.__new__(ConsistencyModel)
    model.device = torch.device("cpu")
    model.rollout_horizon_frames = 3
    model.rollout_history_frames = 4
    sentinel = object()
    calls = []

    class _Pipeline:
        def generate(self, batch, **kwargs):
            calls.append((batch, kwargs))
            return sentinel

    model.pipeline = _Pipeline()
    text_emb = torch.ones(1, 1, 1)
    batch = {"text_emb": torch.zeros_like(text_emb)}

    assert model.rollout(batch, text_emb=text_emb) is sentinel
    rollout_batch, kwargs = calls[0]
    assert rollout_batch["text_emb"] is text_emb
    assert kwargs == {
        "rollout_frames": 3,
        "history_frames": 4,
        "device": torch.device("cpu"),
    }


def test_autoregressive_train_metadata_uses_segmented_target_order(
    monkeypatch,
) -> None:
    metadata = build_x_metadata(
        batch_size=1,
        video_tokens_per_frame=1,
        action_tokens_per_frame=1,
        num_frames=8,
        chunk_size=4,
        window_size=16,
        device=torch.device("cpu"),
    )
    diagnostics = {"source": "native"}
    monkeypatch.setattr(
        VAMOTTransformer3DModel,
        "_prepare_metadata",
        lambda self, input_dict, prepared: (metadata, diagnostics),
    )
    model = AutoregressiveVAMOTTransformer3DModel.__new__(
        AutoregressiveVAMOTTransformer3DModel
    )
    model.history_frames = 4
    model.chunk_size = 4

    rewritten, returned_diagnostics = model._prepare_metadata({}, None)

    assert returned_diagnostics is diagnostics
    expected_video = torch.tensor([0, 0, 0, 0, 2, 4, 6, 8])
    expected_action = expected_video + 1
    frame_ids = rewritten.frame_ids
    assert torch.equal(
        rewritten.order_ids[rewritten.stream_ids == 0],
        expected_video[frame_ids[rewritten.stream_ids == 0]],
    )
    assert torch.equal(
        rewritten.order_ids[rewritten.stream_ids == 1],
        expected_action[frame_ids[rewritten.stream_ids == 1]],
    )
