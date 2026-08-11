import json
from dataclasses import replace
from inspect import signature
from types import SimpleNamespace

import pytest
import torch
from easydict import EasyDict
from torch import nn

from distillation.configs.runtime_dataset import apply_distillation_runtime_overrides
from distillation.model import objectives
from distillation.model.consistency import (
    ConsistencyBaseModel,
    ConsistencyModel,
    ConsistencyTrainingModel,
)
from distillation.model.dmd import (
    BaseModel,
    SGFDMDModel,
    SelfGradientForcingModel,
    warp_denoisy_progress,
)
from distillation.model.wan_wrapper import (
    WanDiffusionWrapper as _RealWrapper,
)
from distillation.pipeline import SelfGradientForcingTrainingPipeline
from distillation.schema import (
    DMDUpdateSchedule,
    DenoisyInterval,
    ReplayContext,
    VALossWeights,
    VADenoisySelection,
    VAMasks,
    VAPair,
    VATimesteps,
)
from distillation.train import parse_args
from wan_va.utils.scheduler import FlowMatchScheduler


def test_training_models_keep_behavior_in_model_modules() -> None:
    assert issubclass(SelfGradientForcingModel, BaseModel)
    assert issubclass(SGFDMDModel, SelfGradientForcingModel)
    assert issubclass(ConsistencyTrainingModel, ConsistencyBaseModel)
    assert issubclass(ConsistencyModel, ConsistencyTrainingModel)

    for method in {"_initialize_models", "_prepare_input"}:
        assert method in BaseModel.__dict__
    for method in {"run_generator", "record", "replay"}:
        assert method in SelfGradientForcingModel.__dict__
    for method in {
        "_get_timestep",
        "_warp_timestep",
        "_get_va_timesteps",
        "_add_dmd_noise",
        "_empty_text_condition",
        "_compute_kl_grad",
        "compute_distribution_matching_loss",
        "generator_loss",
        "critic_loss",
    }:
        assert method in SGFDMDModel.__dict__
    assert BaseModel.__module__ == "distillation.model.dmd"
    assert ConsistencyModel.__module__ == "distillation.model.consistency"


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


def test_interval_progress_sampling_respects_mask_and_half_open_bounds() -> None:
    interval = DenoisyInterval(exit_id=0, denoisy_from=700, denoisy_to=300)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    model = SGFDMDModel.__new__(SGFDMDModel)
    model.device = torch.device("cpu")
    model.ts_schedule = True
    model.ts_schedule_max = True
    model.min_score_timestep = 0
    model.num_train_timestep = 1000

    progress = model._get_timestep(
        interval,
        tuple(mask.shape),
        mask,
    )

    assert torch.equal(progress[~mask], torch.zeros_like(progress[~mask]))
    assert bool((progress[mask] >= 300).all())
    assert bool((progress[mask] < 700).all())


def test_interval_progress_is_uniform_across_frames_per_batch_item() -> None:
    mask = torch.ones(4, 8, dtype=torch.bool)
    model = SGFDMDModel.__new__(SGFDMDModel)
    model.device = torch.device("cpu")
    model.ts_schedule = True
    model.ts_schedule_max = True
    model.min_score_timestep = 0
    model.num_train_timestep = 1000

    progress = model._get_timestep(
        DenoisyInterval(exit_id=1, denoisy_from=750, denoisy_to=500),
        tuple(mask.shape),
        mask,
    )

    assert torch.equal(progress, progress[:, :1].expand_as(progress))


@pytest.mark.parametrize(
    ("ts_schedule", "ts_schedule_max", "expected_low", "expected_high"),
    [
        (False, False, 100, 1000),
        (True, False, 300, 1000),
        (False, True, 100, 700),
        (True, True, 300, 700),
    ],
)
def test_dmd_schedule_switches_select_linear_sampling_bounds(
    ts_schedule,
    ts_schedule_max,
    expected_low,
    expected_high,
) -> None:
    model = SGFDMDModel.__new__(SGFDMDModel)
    model.device = torch.device("cpu")
    model.ts_schedule = ts_schedule
    model.ts_schedule_max = ts_schedule_max
    model.min_score_timestep = 100
    model.num_train_timestep = 1000
    mask = torch.ones(32, 16, dtype=torch.bool)

    progress = model._get_timestep(
        DenoisyInterval(exit_id=0, denoisy_from=700, denoisy_to=300),
        tuple(mask.shape),
        mask,
    )

    assert bool((progress >= expected_low).all())
    assert bool((progress < expected_high).all())


def test_linear_progress_warps_through_each_modality_scheduler() -> None:
    video_scheduler = FlowMatchScheduler(
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
    )
    action_scheduler = FlowMatchScheduler(
        shift=1.0,
        sigma_min=0.0,
        extra_one_step=True,
    )
    video_scheduler.set_timesteps(1000, training=True)
    action_scheduler.set_timesteps(1000, training=True)
    progress = torch.tensor([1000, 750, 500, 250, 0])

    torch.testing.assert_close(
        warp_denoisy_progress(
            progress,
            video_scheduler,
        ),
        torch.tensor([1000.0, 937.5, 833.3333, 625.0, 0.0]),
        rtol=1e-5,
        atol=1e-4,
    )
    torch.testing.assert_close(
        warp_denoisy_progress(
            progress,
            action_scheduler,
        ),
        progress.float(),
    )


def test_dmd_losses_are_explicit() -> None:
    assert not hasattr(objectives, "replay_target_loss")
    assert callable(objectives.dmd_surrogate_loss)
    assert "_compute_kl_grad" in SGFDMDModel.__dict__
    assert "compute_distribution_matching_loss" in SGFDMDModel.__dict__


def test_dmd_surrogate_loss_is_half_masked_mse() -> None:
    x0 = VAPair(
        video=torch.zeros(1, 1, 1, 1, 1, 1),
        action=torch.zeros(1, 1, 1, 1, 1),
    )
    target = VAPair(
        video=torch.ones(1, 1, 1, 1, 1, 1),
        action=torch.ones(1, 1, 1, 1, 1),
    )
    masks = VAMasks(
        video=torch.ones(1, 1, dtype=torch.bool),
        action=torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
    )

    loss, metrics = objectives.dmd_surrogate_loss(
        x0,
        target,
        masks,
        VALossWeights(video=1.0, action=1.0),
    )

    # Per-stream MSE = 1, weighted total = 2, surrogate = 0.5 * 2 = 1.
    assert torch.allclose(loss, torch.tensor(1.0))
    assert torch.allclose(metrics["distill/dmd_video_loss"], torch.tensor(0.5))
    assert torch.allclose(metrics["distill/dmd_action_loss"], torch.tensor(0.5))
    assert torch.allclose(metrics["distill/dmd_total_loss"], torch.tensor(1.0))


def test_resume_rejects_changed_denoisy_lists() -> None:
    model = SGFDMDModel.__new__(SGFDMDModel)
    model.update_schedule = DMDUpdateSchedule(fake_score_steps=4)
    model.video_denoising_step_list = (1000.0, 833.0)
    model.action_denoising_step_list = (1000.0, 500.0)

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


def _tiny_model_and_context(monkeypatch):
    config = SimpleNamespace(
        video_loss_weight=1.0,
        action_loss_weight=1.0,
        snr_shift=1.0,
        action_snr_shift=1.0,
        distill=SimpleNamespace(
            fake_score_update_ratio=4,
            dmd_normalizer_eps=1e-6,
            ts_schedule=True,
            ts_schedule_max=True,
            min_score_timestep=0,
            dmd_timestep_min=20,
            dmd_timestep_max=980,
            rollout_horizon_frames=1,
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

    sources = iter([student, real, fake])

    def fake_load_model(checkpoint_path, config, *, autoregressive):
        del checkpoint_path, config, autoregressive
        return next(sources)

    monkeypatch.setattr(
        _RealWrapper,
        "_load_model",
        staticmethod(fake_load_model),
    )
    model = SGFDMDModel(
        config=config,
        student_init="/tmp/tiny-student",
        device=torch.device("cpu"),
        real_score_checkpoint="/tmp/tiny-real",
        fake_score_init="/tmp/tiny-fake",
    )
    final_clean = VAPair(
        video=torch.zeros(1, 1, 2, 1, 1, 1),
        action=torch.zeros(1, 1, 2, 1, 1),
    )
    noisy = VAPair(
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
        exit_timesteps=VATimesteps(
            video=torch.tensor([[0.0, 1000.0]]),
            action=torch.tensor([[0.0, 500.0]]),
        ),
        noisy_at_t=noisy,
        clean_hat=final_clean,
        masks=masks,
        denoisy_selection=selection,
    )
    return model, context, trainer, student, real, fake


def test_dmd_model_loads_models_from_paths_only(monkeypatch) -> None:
    model, _context, _trainer, _student, _real, _fake = _tiny_model_and_context(
        monkeypatch
    )

    assert not hasattr(model, "text_encoder")
    assert not hasattr(model, "vae")
    assert model.generator.video_scheduler.timesteps.device.type == "cpu"


def test_dmd_model_hierarchy_and_steps_isolate_gradients(monkeypatch) -> None:
    model, context, trainer, student, real, fake = _tiny_model_and_context(
        monkeypatch
    )
    assert isinstance(model, SelfGradientForcingModel)
    assert isinstance(model, BaseModel)
    model._run_generator = lambda _batch: context
    base_input = trainer._prepare_joint_input_dict(
        {
            "text_emb": torch.ones(1, 1, 1),
            "stream_ids": torch.zeros(1, 1, dtype=torch.long),
        },
        add_noise=False,
    )
    empty_text_emb = trainer._get_empty_text_emb()

    student_loss, student_metrics = model.generator_loss(
        {},
        base_input=base_input,
        empty_text_emb=empty_text_emb,
    )
    student_loss.backward()
    assert student.scale.grad is not None
    assert fake.scale.grad is None
    assert real.scale.grad is None
    assert "distill/sgf_total_loss" in student_metrics
    assert not any("replay" in key for key in student_metrics)

    student.scale.grad = None
    fake_loss, _ = model.critic_loss({}, base_input=base_input)
    fake_loss.backward()
    assert student.scale.grad is None
    assert fake.scale.grad is not None
    assert real.scale.grad is None


def test_generator_loss_uses_batch_empty_text_embedding(monkeypatch) -> None:
    model, context, _trainer, _student, _real, _fake = _tiny_model_and_context(
        monkeypatch
    )
    model._run_generator = lambda _batch: context
    batch_empty_text_emb = torch.full((1, 1, 1), 2.0)
    captured = {}

    def capture_kl_grad(
        _score_input,
        _noisy,
        estimated_clean,
        _timesteps,
        _masks,
        empty_text_emb,
    ):
        captured["empty_text_emb"] = empty_text_emb
        return VAPair(
            video=torch.zeros_like(estimated_clean.video),
            action=torch.zeros_like(estimated_clean.action),
        ), {}

    model._compute_kl_grad = capture_kl_grad
    model.generator_loss(
        {"empty_text_emb": batch_empty_text_emb},
        base_input={
            "latent_dict": {"text_emb": torch.ones(1, 1, 1)},
            "action_dict": {"text_emb": torch.ones(1, 1, 1)},
            "stream_ids": torch.zeros(1, 1, dtype=torch.long),
        },
        empty_text_emb=torch.zeros(1, 1, 1),
    )

    assert captured["empty_text_emb"] is batch_empty_text_emb


def test_generator_loss_handles_all_false_action_mask(monkeypatch) -> None:
    model, context, trainer, student, _real, _fake = _tiny_model_and_context(
        monkeypatch
    )
    context = replace(
        context,
        masks=VAMasks(
            video=context.masks.video,
            action=torch.zeros_like(context.masks.action),
        ),
    )
    model._run_generator = lambda _batch: context
    base_input = trainer._prepare_joint_input_dict(
        {
            "text_emb": torch.ones(1, 1, 1),
            "stream_ids": torch.zeros(1, 1, dtype=torch.long),
        },
        add_noise=False,
    )

    loss, metrics = model.generator_loss(
        {},
        base_input=base_input,
        empty_text_emb=trainer._get_empty_text_emb(),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.equal(
        metrics["distill/dmd_action_gradient_norm"],
        torch.zeros_like(metrics["distill/dmd_action_gradient_norm"]),
    )
    assert student.scale.grad is not None
    assert torch.isfinite(student.scale.grad)


@pytest.mark.parametrize("optimizer_name", ["generator", "fake_score"])
def test_rollout_and_replay_share_text_condition(
    monkeypatch,
    optimizer_name,
) -> None:
    model, context, trainer, _student, _real, _fake = _tiny_model_and_context(
        monkeypatch
    )
    raw_text = torch.ones(1, 1, 1)
    effective_text = torch.full_like(raw_text, 2.0)
    batch = {
        "text_emb": raw_text,
        "stream_ids": torch.zeros(1, 1, dtype=torch.long),
    }
    base_input = trainer._prepare_joint_input_dict(batch, add_noise=False)
    base_input["latent_dict"]["text_emb"] = effective_text
    base_input["action_dict"]["text_emb"] = effective_text
    captured = {}

    def capture_rollout(rollout_batch):
        captured["text_emb"] = rollout_batch["text_emb"]
        return context

    model._run_generator = capture_rollout
    model.compute_step(
        batch,
        optimizer_name,
        base_input=base_input,
        empty_text_emb=trainer._get_empty_text_emb(),
    )

    assert torch.equal(captured["text_emb"], effective_text)


def test_dmd_model_does_not_own_resume_loading() -> None:
    assert "resume_from" not in signature(BaseModel.__init__).parameters
    assert not hasattr(SGFDMDModel, "_resume_method_state")


def test_add_dmd_noise_supports_batch_gt_one(monkeypatch) -> None:
    """Per-sample [B,F] timesteps must broadcast on the frame axis, not crash."""
    from distillation.model.wan_wrapper import (
        broadcast_frame_values,
        sigmas_for_timesteps,
    )

    model, _context, _trainer, _student, _real, _fake = _tiny_model_and_context(
        monkeypatch
    )
    clean = VAPair(
        video=torch.zeros(3, 1, 2, 1, 1, 1),
        action=torch.zeros(3, 1, 2, 1, 1),
    )
    masks = VAMasks(
        video=torch.tensor([[True, False], [True, True], [False, True]]),
        action=torch.tensor(
            [
                [[[[True]], [[False]]]],
                [[[[True]], [[True]]]],
                [[[[False]], [[True]]]],
            ]
        ),
    )
    timesteps = VATimesteps(
        video=torch.tensor(
            [[500.0, 0.0], [500.0, 500.0], [0.0, 500.0]],
        ),
        action=torch.tensor(
            [[500.0, 0.0], [500.0, 500.0], [0.0, 500.0]],
        ),
    )

    noisy, noise = model._add_dmd_noise(clean, timesteps, masks)

    assert noisy.video.shape == clean.video.shape
    assert noisy.action.shape == clean.action.shape
    video_mask = masks.video[:, None, :, None, None, None]
    sigma_video = broadcast_frame_values(
        sigmas_for_timesteps(
            model.generator.video_scheduler,
            timesteps.video,
            dtype=torch.float32,
        ),
        noisy.video,
    )
    expected_video = torch.where(
        video_mask,
        (1.0 - sigma_video) * clean.video + sigma_video * noise.video,
        clean.video,
    )
    assert torch.equal(
        noisy.video,
        expected_video,
    )
    sigma_action = broadcast_frame_values(
        sigmas_for_timesteps(
            model.generator.action_scheduler,
            timesteps.action,
            dtype=torch.float32,
        ),
        noisy.action,
    )
    expected_action = torch.where(
        masks.action,
        (1.0 - sigma_action) * clean.action + sigma_action * noise.action,
        clean.action,
    )
    assert torch.equal(
        noisy.action,
        expected_action,
    )


def test_critic_loss_score_input_can_use_gt_clean(monkeypatch) -> None:
    """score_input_use_gt_clean=True feeds GT clean as the score condition."""
    model, context, _trainer, _student, _real, _fake = _tiny_model_and_context(
        monkeypatch
    )
    gt_video = torch.randn(1, 1, 2, 1, 1, 1)
    gt_action = torch.randn(1, 1, 2, 1, 1)
    base_input = {
        "latent_dict": {
            "latent": gt_video,
            "text_emb": torch.ones(1, 1, 1),
        },
        "action_dict": {
            "latent": gt_action,
            "text_emb": torch.ones(1, 1, 1),
        },
        "stream_ids": torch.zeros(1, 1, dtype=torch.long),
    }
    model._run_generator = lambda _batch: context
    model.score_input_use_gt_clean = True
    original_fake = model.fake_score
    captured = {}

    def spy_score(input_dict, noisy, timesteps):
        captured["latent"] = input_dict["latent_dict"]["latent"]
        captured["action_latent"] = input_dict["action_dict"]["latent"]
        return original_fake(input_dict, noisy, timesteps)

    model.fake_score = spy_score
    model.critic_loss({}, base_input=base_input)

    assert torch.equal(captured["latent"], gt_video)
    assert torch.equal(captured["action_latent"], gt_action)


def test_generator_loss_score_input_can_use_gt_clean(monkeypatch) -> None:
    """score_input_use_gt_clean=True also feeds GT clean in the generator loss."""
    model, context, _trainer, _student, _real, _fake = _tiny_model_and_context(
        monkeypatch
    )
    gt_video = torch.randn(1, 1, 2, 1, 1, 1)
    gt_action = torch.randn(1, 1, 2, 1, 1)
    base_input = {
        "latent_dict": {
            "latent": gt_video,
            "text_emb": torch.ones(1, 1, 1),
        },
        "action_dict": {
            "latent": gt_action,
            "text_emb": torch.ones(1, 1, 1),
        },
        "stream_ids": torch.zeros(1, 1, dtype=torch.long),
    }
    model._run_generator = lambda _batch: context
    model.score_input_use_gt_clean = True
    original_fake = model.fake_score
    captured = {}

    def spy_score(input_dict, noisy, timesteps):
        captured["latent"] = input_dict["latent_dict"]["latent"]
        captured["action_latent"] = input_dict["action_dict"]["latent"]
        return original_fake(input_dict, noisy, timesteps)

    model.fake_score = spy_score
    model.generator_loss(
        {},
        base_input=base_input,
        empty_text_emb=torch.zeros(1, 1, 1),
    )

    assert torch.equal(captured["latent"], gt_video)
    assert torch.equal(captured["action_latent"], gt_action)
