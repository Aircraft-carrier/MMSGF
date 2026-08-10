from types import SimpleNamespace

import pytest
import torch
from torch import nn

from distillation.model.autoregressive_types import AutoregressiveModelOutput
from distillation.model.wan_wrapper import WanDiffusionWrapper
from distillation.diffusion_utils import renoise_x0
from distillation.schema import DenoisyInterval
from distillation.self_rollout.engine import self_rollout
from distillation.self_rollout.recorder import SelfRolloutRecorder
from distillation.self_rollout.transitions import RolloutSchedulers, SGFRolloutSchedule


class _InferenceScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([2.0, 1.0])
        self.calls = 0

    def step(self, model_output, timestep, sample):
        self.calls += 1
        return sample + model_output


class _SigmaScheduler:
    def __init__(self, timesteps, sigmas) -> None:
        self.timesteps = torch.tensor(timesteps, dtype=torch.float32)
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.num_train_timesteps = 1000


class _RolloutTransformer(nn.Module):
    def forward_autoregressive(self, request):
        raise AssertionError("the public forward boundary must be used")

    def forward(self, request, *, mode: str):
        assert mode == "self_rollout"
        if request.operation in {"predict_video", "predict_action"}:
            return AutoregressiveModelOutput(
                prediction=torch.zeros_like(request.payload["sample"])
            )
        return AutoregressiveModelOutput()

    def assert_video_commit(self, *args, **kwargs):
        return None

    def assert_action_commit(self, *args, **kwargs):
        return None


def _batch():
    return {
        "latents": torch.zeros(1, 1, 3, 1, 1, 1),
        "actions": torch.zeros(1, 1, 3, 1, 1),
        "video_latent_valid_mask": torch.ones(1, 3, dtype=torch.bool),
        "action_valid_mask": torch.ones(1, 1, 3, 1, 1, dtype=torch.bool),
        "stream_ids": torch.zeros(1, 1, dtype=torch.long),
        "text_emb": torch.zeros(1, 1, 1),
    }


def _config():
    return SimpleNamespace(
        seed=11,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        distill=SimpleNamespace(
            generation_shape={
                "profile_name": "segmented_history_va_v1",
                "history_frames": 1,
                "order_mode": "segmented",
                "chunk_size": 1,
                "window_size": 16,
            }
        ),
    )


def _spec():
    return SimpleNamespace(
        history_latent_frames=1,
        latent_frames_per_action_chunk_per_view=1,
    )


def test_inference_is_backward_compatible_default_mode() -> None:
    video_scheduler = _InferenceScheduler()
    action_scheduler = _InferenceScheduler()
    result = self_rollout(
        _batch(),
        transformer=_RolloutTransformer(),
        config=_config(),
        spec=_spec(),
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        rollout_frames=1,
        schedulers=RolloutSchedulers(video_scheduler, action_scheduler),
    )

    generator = torch.Generator().manual_seed(11)
    expected_video = torch.randn((1, 1, 1, 1, 1, 1), generator=generator)
    torch.testing.assert_close(result.pred_latents[:, :, 2:3], expected_video)
    assert video_scheduler.calls == 2
    assert action_scheduler.calls == 2
    assert result.diagnostics["transition_mode"] == "inference"


def test_inference_rejects_sgf_only_arguments() -> None:
    interval = DenoisyInterval(exit_id=0, denoisy_from=1000, denoisy_to=0)
    with pytest.raises(ValueError, match="does not accept SGF"):
        self_rollout(
            _batch(),
            transformer=_RolloutTransformer(),
            config=_config(),
            spec=_spec(),
            device=torch.device("cpu"),
            empty_text_emb=torch.zeros(1, 1, 1),
            rollout_frames=1,
            schedulers=RolloutSchedulers(
                _InferenceScheduler(),
                _InferenceScheduler(),
            ),
            recorder=SelfRolloutRecorder(video=interval, action=interval),
        )


def test_sgf_renoise_uses_different_lists_and_exit_ids() -> None:
    video_scheduler = _SigmaScheduler([1000, 500], [1.0, 0.5])
    action_scheduler = _SigmaScheduler([1000], [1.0])
    model = _RolloutTransformer()
    wrapper = WanDiffusionWrapper(
        model,
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
    )
    video_interval = DenoisyInterval(0, 1000, 500)
    action_interval = DenoisyInterval(0, 1000, 0)
    recorder = SelfRolloutRecorder(video=video_interval, action=action_interval)
    generator = torch.Generator().manual_seed(7)

    result = self_rollout(
        _batch(),
        transformer=model,
        config=_config(),
        spec=_spec(),
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        rollout_frames=1,
        transition_mode="sgf_renoise",
        sgf_schedule=SGFRolloutSchedule(
            video_steps=(1000.0, 500.0),
            action_steps=(1000.0,),
            video_scheduler=video_scheduler,
            action_scheduler=action_scheduler,
        ),
        diffusion_wrapper=wrapper,
        recorder=recorder,
        generator=generator,
    )

    expected_generator = torch.Generator().manual_seed(7)
    initial_video = torch.randn((1, 1, 1, 1, 1, 1), generator=expected_generator)
    fresh_video = torch.randn((1, 1, 1, 1, 1, 1), generator=expected_generator)
    expected_video = 0.5 * initial_video + 0.5 * fresh_video
    torch.testing.assert_close(result.pred_latents[:, :, 2:3], expected_video)
    torch.testing.assert_close(recorder.video[2].sample, initial_video)
    assert recorder.video[2].exit_id == 0
    assert recorder.action[2].denoisy_to == 0
    assert result.diagnostics["transition_mode"] == "sgf_renoise"


def test_renoise_consumes_fresh_noise_each_call() -> None:
    scheduler = _SigmaScheduler([500], [0.5])
    x0 = torch.zeros(1, 1, 1, 1, 1, 1)
    timestep = torch.tensor([[500.0]])
    generator = torch.Generator().manual_seed(3)

    first, first_noise = renoise_x0(x0, timestep, scheduler, generator=generator)
    second, second_noise = renoise_x0(x0, timestep, scheduler, generator=generator)

    assert not torch.equal(first_noise, second_noise)
    torch.testing.assert_close(first, 0.5 * first_noise)
    torch.testing.assert_close(second, 0.5 * second_noise)
