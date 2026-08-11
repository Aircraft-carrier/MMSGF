from types import SimpleNamespace

import torch
import pytest
from torch import nn

from distillation.model.autoregressive_types import (
    AutoregressiveModelOutput,
)
from distillation.pipeline import KVCache
from distillation.model.wan_wrapper import (
    WanDiffusionWrapper,
)
from distillation.schema import VAPrediction, VATimesteps


def _scheduler(timestep: float, sigma: float):
    return SimpleNamespace(
        timesteps=torch.tensor([timestep], dtype=torch.float32),
        sigmas=torch.tensor([sigma], dtype=torch.float32),
    )


class _JointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_dict, *, mode: str):
        assert mode == "train"
        return {
            "latent_pred": input_dict["video_velocity"] * self.scale,
            "action_pred": input_dict["action_velocity"] * self.scale,
        }


class _ARModel(nn.Module):
    def forward(self, request, *, mode):
        assert mode == "self_rollout"
        if request.operation in {"predict_video", "predict_action"}:
            return AutoregressiveModelOutput(prediction=request.payload["sample"])
        return AutoregressiveModelOutput()


def test_wrapper_ar_generation_returns_clean_x0(monkeypatch) -> None:
    config = SimpleNamespace(snr_shift=1.0, action_snr_shift=1.0)

    def fake_load_model(checkpoint_path, config, *, autoregressive):
        del checkpoint_path, config, autoregressive
        return _ARModel()

    monkeypatch.setattr(
        WanDiffusionWrapper,
        "_load_model",
        staticmethod(fake_load_model),
    )
    wrapper = WanDiffusionWrapper(
        config=config,
        checkpoint_path="dummy",
        autoregressive=True,
    )
    cache = KVCache()
    noisy = torch.ones(1, 1, 1, 1, 1, 1)
    actions = torch.ones(1, 1, 1, 1, 1)
    stream_ids = torch.zeros(1, 1, dtype=torch.long)
    text_emb = torch.zeros(1, 1, 1)

    x0_video = wrapper.generate_video(
        noisy,
        torch.tensor(1000.0),
        frame_ids=[0],
        stream_ids=stream_ids,
        cache=cache,
        text_emb=text_emb,
    )
    x0_action = wrapper.generate_action(
        actions,
        torch.tensor(1000.0),
        frame_ids=[0],
        cache=cache,
        text_emb=text_emb,
    )

    assert x0_video.shape == noisy.shape
    assert x0_action.shape == actions.shape
    # sigma=1 at t=1000, so x0 = xt - 1 * flow = 0 for the identity model.
    torch.testing.assert_close(x0_video, torch.zeros_like(x0_video), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x0_action, torch.zeros_like(x0_action), atol=1e-5, rtol=1e-5)
    wrapper.commit_video(noisy, frame_ids=[0], stream_ids=stream_ids, cache=cache, text_emb=text_emb)
    wrapper.commit_action(actions, frame_ids=[0], cache=cache, text_emb=text_emb)


def test_forward_returns_flow_x0_and_preserves_student_gradient(monkeypatch) -> None:
    model = _JointModel()
    config = SimpleNamespace(snr_shift=1.0, action_snr_shift=1.0)

    def fake_load_model(checkpoint_path, config, *, autoregressive):
        del checkpoint_path, config, autoregressive
        return model

    monkeypatch.setattr(
        WanDiffusionWrapper,
        "_load_model",
        staticmethod(fake_load_model),
    )
    wrapper = WanDiffusionWrapper(
        config=config,
        checkpoint_path="dummy",
    )
    noisy = VAPrediction(
        video=torch.ones(1, 1, 1, 1, 1, 1),
        action=torch.ones(1, 1, 1, 1, 1),
    )
    input_dict = {
        "video_velocity": torch.full_like(noisy.video, 2.0),
        "action_velocity": torch.full_like(noisy.action, 3.0),
    }
    timesteps = VATimesteps(
        video=torch.tensor([[500.0]]),
        action=torch.tensor([[500.0]]),
    )

    output = wrapper(input_dict, noisy, timesteps)
    (output.x0.video.sum() + output.x0.action.sum()).backward()

    assert model.scale.grad is not None
    torch.testing.assert_close(model.scale.grad, torch.tensor(-2.5))
    torch.testing.assert_close(
        output.velocity.video,
        torch.full_like(noisy.video, 2.0),
    )
    torch.testing.assert_close(
        output.velocity.action,
        torch.full_like(noisy.action, 3.0),
    )
    torch.testing.assert_close(output.x0.video, torch.zeros_like(noisy.video))
    torch.testing.assert_close(output.x0.action, torch.full_like(noisy.action, -0.5))


def test_wrapper_initializes_video_and_action_schedulers(monkeypatch) -> None:
    config = SimpleNamespace(snr_shift=5.0, action_snr_shift=1.0)

    def fake_load_model(checkpoint_path, config, *, autoregressive):
        del checkpoint_path, config, autoregressive
        return nn.Identity()

    monkeypatch.setattr(
        WanDiffusionWrapper,
        "_load_model",
        staticmethod(fake_load_model),
    )
    wrapper = WanDiffusionWrapper(config=config, checkpoint_path="dummy")

    assert len(wrapper.video_scheduler.timesteps) == 1000
    assert len(wrapper.action_scheduler.timesteps) == 1000
    assert wrapper.video_scheduler.shift == 5.0
    assert wrapper.action_scheduler.shift == 1.0


@pytest.mark.parametrize(
    ("autoregressive", "architecture"),
    [(True, "autoregressive_va_mot_v1"), (False, "va_mot_v1")],
)
def test_wrapper_loads_autoregressive_or_bidirectional_model_in_place(
    monkeypatch,
    tmp_path,
    autoregressive: bool,
    architecture: str,
) -> None:
    from distillation.model import wan_wrapper

    calls = []

    class LoadedModel(nn.Identity):
        pass

    class ModelClass:
        @classmethod
        def from_pretrained(cls, checkpoint_path, **kwargs):
            calls.append((checkpoint_path, kwargs))
            return LoadedModel()

    monkeypatch.setattr(
        wan_wrapper.WanDiffusionWrapper,
        "_model_class",
        staticmethod(lambda _autoregressive: ModelClass),
    )
    checkpoint = tmp_path / "checkpoint"
    transformer = checkpoint / "transformer"
    transformer.mkdir(parents=True)
    (checkpoint / "_SUCCESS").touch()
    (checkpoint / "checkpoint_metadata.json").write_text(
        '{"model_architecture": "' + architecture + '"}',
        encoding="utf-8",
    )
    (transformer / "config.json").touch()
    (transformer / "diffusion_pytorch_model.safetensors").touch()
    config = SimpleNamespace(
        snr_shift=5.0,
        action_snr_shift=1.0,
        param_dtype=torch.bfloat16,
        distill=SimpleNamespace(
            model_architecture="autoregressive_va_mot_v1",
            generation_shape={"history_frames": 4},
        ),
    )
    wrapper = WanDiffusionWrapper(
        config=config,
        checkpoint_path=checkpoint,
        autoregressive=autoregressive,
    )

    assert isinstance(wrapper.model, LoadedModel)
    assert calls == [(transformer, {"torch_dtype": torch.bfloat16})]
    assert not hasattr(wrapper.model, "unshard")


def test_wrapper_exposes_only_joint_forward_conversion(monkeypatch) -> None:
    config = SimpleNamespace(snr_shift=1.0, action_snr_shift=1.0)
    def fake_load_model(checkpoint_path, config, *, autoregressive):
        del checkpoint_path, config, autoregressive
        return nn.Identity()

    monkeypatch.setattr(
        WanDiffusionWrapper,
        "_load_model",
        staticmethod(fake_load_model),
    )
    wrapper = WanDiffusionWrapper(
        config=config,
        checkpoint_path="dummy",
    )

    assert not hasattr(wrapper, "predict_velocity")
    assert not hasattr(wrapper, "velocity_to_x0")
    assert not hasattr(wrapper, "x0_to_velocity")


def test_wrapper_does_not_register_borrowed_model(monkeypatch) -> None:
    config = SimpleNamespace(snr_shift=1.0, action_snr_shift=1.0)
    def fake_load_model(checkpoint_path, config, *, autoregressive):
        del checkpoint_path, config, autoregressive
        return _JointModel()

    monkeypatch.setattr(
        WanDiffusionWrapper,
        "_load_model",
        staticmethod(fake_load_model),
    )
    wrapper = WanDiffusionWrapper(
        config=config,
        checkpoint_path="dummy",
    )

    assert not isinstance(wrapper, nn.Module)
    assert not hasattr(wrapper, "state_dict")
