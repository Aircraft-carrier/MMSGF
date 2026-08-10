import json
from types import SimpleNamespace

import pytest
import torch
from easydict import EasyDict
from torch import nn

from distillation.configs.runtime_dataset import apply_distillation_runtime_overrides
from distillation.model import objectives
from distillation.model.dmd import SGFDMDModel
from distillation.pipeline.self_gradient_forcing_training import (
    SelfGradientForcingTrainingPipeline,
)
from distillation.diffusion_utils import sample_interval_timesteps
from distillation.schema import (
    DMDUpdateSchedule,
    DenoisyInterval,
    ReplayContext,
    VADenoisySelection,
    VALossWeights,
    VAMasks,
    VAPrediction,
    VATimesteps,
)
from distillation.self_rollout.transitions import build_sgf_rollout_schedule
from distillation.train import parse_args


def test_video_action_intervals_are_independent() -> None:
    video = SelfGradientForcingTrainingPipeline._interval((1000.0, 833.0), 0)
    action = SelfGradientForcingTrainingPipeline._interval((1000.0, 500.0), 1)

    assert video == DenoisyInterval(0, 1000, 833)
    assert action == DenoisyInterval(1, 500, 0)


def test_cli_parses_independent_video_action_lists() -> None:
    args = parse_args(
        [
            "--method",
            "self_gradient_forcing_dmd",
            "--save-root",
            "/tmp/sgf-test",
            "--video-denoisy-step-list",
            "1000,833",
            "--action-denoisy-step-list",
            "1000,500,250",
        ]
    )

    assert args.video_denoisy_step_list == [1000.0, 833.0]
    assert args.action_denoisy_step_list == [1000.0, 500.0, 250.0]


def test_runtime_dataset_override_populates_wan22_base_paths(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    payload = {
        "mot_manifest_path": str(meta / "manifest.jsonl"),
        "empty_emb_path": str(root / "empty.pt"),
        "text_emb_cache_path": str(root / "text.pt"),
        "action_cache_manifest_path": "",
        "norm_stat": {"q01": [0], "q99": [1]},
        "norm_stats_by_task": {"default": {"q01": [0], "q99": [1]}},
    }
    (meta / "mot_config.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("MOT_DATASET_ROOT", str(root))
    config = EasyDict()

    apply_distillation_runtime_overrides(config)

    assert config.dataset_path == str(root.resolve())
    assert config.mot_config_path == str((meta / "mot_config.json").resolve())
    assert config.mot_manifest_path == payload["mot_manifest_path"]
    assert config.action_cache_manifest_path is None


def test_sgf_schedule_rejects_non_descending_modality() -> None:
    config = SimpleNamespace(
        distill=SimpleNamespace(
            denoisy_step_list=SimpleNamespace(
                video=[1000, 833],
                action=[500, 750],
            )
        )
    )
    scheduler = SimpleNamespace(num_train_timesteps=1000)

    with pytest.raises(ValueError, match="strictly descending"):
        build_sgf_rollout_schedule(
            config,
            video_scheduler=scheduler,
            action_scheduler=scheduler,
        )


def test_interval_timestep_sampling_respects_mask_and_bounds() -> None:
    interval = DenoisyInterval(exit_id=0, denoisy_from=700, denoisy_to=300)
    mask = torch.tensor([[True, False, True], [False, True, False]])

    timesteps = sample_interval_timesteps(
        interval,
        tuple(mask.shape),
        torch.device("cpu"),
        mask,
    )

    assert torch.equal(timesteps[~mask], torch.zeros_like(timesteps[~mask]))
    assert bool((timesteps[mask] >= 300).all())
    assert bool((timesteps[mask] <= 700).all())


def test_student_objective_is_dmd_only_and_has_expected_direction() -> None:
    assert not hasattr(objectives, "replay_target_loss")
    student_video = torch.zeros(1, 1, 1, 1, 1, 1, requires_grad=True)
    student_action = torch.zeros(1, 1, 1, 1, 1, requires_grad=True)
    student = VAPrediction(student_video, student_action)
    fake = VAPrediction(
        torch.zeros_like(student_video),
        torch.zeros_like(student_action),
    )
    real = VAPrediction(
        torch.ones_like(student_video),
        torch.ones_like(student_action),
    )
    masks = VAMasks(
        video=torch.ones(1, 1, dtype=torch.bool),
        action=torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
    )

    loss, metrics = objectives.dmd_surrogate_loss(
        student,
        fake,
        real,
        masks,
        VALossWeights(),
    )
    loss.backward()

    assert set(metrics) == {
        "distill/dmd_video_loss",
        "distill/dmd_action_loss",
        "distill/dmd_total_loss",
    }
    assert student_video.grad.item() < 0
    assert student_action.grad.item() < 0


def test_resume_rejects_changed_denoisy_lists() -> None:
    model = SGFDMDModel.__new__(SGFDMDModel)
    model.update_schedule = DMDUpdateSchedule(fake_score_steps=4)
    model.pipeline = SimpleNamespace(
        sgf_schedule=SimpleNamespace(
            video_steps=(1000.0, 833.0),
            action_steps=(1000.0, 500.0),
        )
    )

    with pytest.raises(ValueError, match="do not match checkpoint"):
        model.load_state_dict(
            {
                "fake_score_update_ratio": 4,
                "video_denoisy_step_list": (1000.0, 750.0),
                "action_denoisy_step_list": (1000.0, 500.0),
            }
        )


class _TinyScheduler:
    def __init__(self, shift: float) -> None:
        del shift
        self.num_train_timesteps = 1000
        self.timesteps = torch.tensor([1000.0, 500.0, 0.0])
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])


class _TinyScore(nn.Module):
    def __init__(self, scale: float, *, trainable: bool) -> None:
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(scale),
            requires_grad=trainable,
        )

    def forward(self, input_dict, *, mode: str):
        assert mode == "train"
        return {
            "latent_pred": input_dict["latent_dict"]["noisy_latents"] * self.scale,
            "action_pred": input_dict["action_dict"]["noisy_latents"] * self.scale,
        }


class _TinyTrainer:
    def __init__(self) -> None:
        self.train_scheduler_latent = _TinyScheduler(1)
        self.train_scheduler_action = _TinyScheduler(1)

    def _prepare_joint_input_dict(self, batch, *, add_noise: bool):
        assert not add_noise
        return {
            "latent_dict": {"text_emb": batch["text_emb"]},
            "action_dict": {"text_emb": batch["text_emb"]},
            "stream_ids": batch["stream_ids"],
        }

    def _get_empty_text_emb(self):
        return torch.zeros(1, 1, 1)


def _tiny_pipeline_and_context():
    config = SimpleNamespace(
        video_loss_weight=1.0,
        action_loss_weight=1.0,
        distill=SimpleNamespace(
            dmd_normalizer_eps=1e-6,
            rollout_horizon_frames=1,
            rollout_masked_attn_backend="dense",
            teacher_cfg_min=1.0,
            teacher_cfg_max=1.0,
            denoisy_step_list=SimpleNamespace(
                video=[1000, 500],
                action=[1000, 500],
            ),
        ),
    )
    trainer = _TinyTrainer()
    student = _TinyScore(0.1, trainable=True)
    real = _TinyScore(0.2, trainable=False)
    fake = _TinyScore(0.3, trainable=True)
    pipeline = SelfGradientForcingTrainingPipeline(
        config,
        trainer,
        torch.device("cpu"),
        student,
        real,
        fake,
    )
    final_clean = VAPrediction(
        video=torch.zeros(1, 1, 2, 1, 1, 1),
        action=torch.zeros(1, 1, 2, 1, 1),
    )
    noisy = VAPrediction(
        video=final_clean.video.clone(),
        action=final_clean.action.clone(),
    )
    noisy.video[:, :, 1] = 1
    noisy.action[:, :, 1] = 1
    masks = VAMasks(
        video=torch.tensor([[False, True]]),
        action=torch.tensor([[[[[False]], [[True]]]]]),
    )
    selection = VADenoisySelection(
        video=DenoisyInterval(0, 1000, 500),
        action=DenoisyInterval(1, 500, 0),
    )
    context = ReplayContext(
        replay_batch={
            "text_emb": torch.ones(1, 1, 1),
            "stream_ids": torch.zeros(1, 1, dtype=torch.long),
        },
        rollout_timesteps=VATimesteps(
            video=torch.tensor([[0.0, 1000.0]]),
            action=torch.tensor([[0.0, 500.0]]),
        ),
        rollout_noisy=noisy,
        final_clean_context=final_clean,
        masks=masks,
        denoisy_selection=selection,
    )
    return pipeline, context, student, real, fake


def test_pipeline_student_and_fake_steps_isolate_gradients() -> None:
    pipeline, context, student, real, fake = _tiny_pipeline_and_context()

    student_loss, student_metrics = pipeline.replay_and_score(context)
    student_loss.backward()
    assert student.scale.grad is not None
    assert fake.scale.grad is None
    assert real.scale.grad is None
    assert "distill/sgf_total_loss" in student_metrics
    assert not any("replay" in key for key in student_metrics)

    student.scale.grad = None
    fake_loss, _ = pipeline.fake_score_step(context)
    fake_loss.backward()
    assert student.scale.grad is None
    assert fake.scale.grad is not None
    assert real.scale.grad is None
