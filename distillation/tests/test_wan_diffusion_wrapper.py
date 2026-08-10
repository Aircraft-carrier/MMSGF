from types import SimpleNamespace

import torch
from torch import nn

from distillation.model.wan_wrapper import WanDiffusionWrapper
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


def test_velocity_to_x0_hand_calculation() -> None:
    wrapper = WanDiffusionWrapper(
        nn.Identity(),
        video_scheduler=_scheduler(750, 0.75),
        action_scheduler=_scheduler(750, 0.75),
    )
    noisy = torch.tensor([[[[[[5.0]]]]]])
    velocity = torch.tensor([[[[[[4.0]]]]]])
    timestep = torch.tensor([[750.0]])

    x0 = wrapper.velocity_to_x0(
        velocity,
        noisy,
        timestep,
        modality="video",
    )

    torch.testing.assert_close(x0, torch.full_like(x0, 2.0))


def test_video_and_action_use_different_scheduler_sigmas() -> None:
    wrapper = WanDiffusionWrapper(
        nn.Identity(),
        video_scheduler=_scheduler(500, 0.8),
        action_scheduler=_scheduler(500, 0.25),
    )
    video = torch.ones(1, 1, 1, 1, 1, 1)
    action = torch.ones(1, 1, 1, 1, 1)
    timestep = torch.tensor([[500.0]])

    video_x0 = wrapper.velocity_to_x0(video, video * 5, timestep, modality="video")
    action_x0 = wrapper.velocity_to_x0(
        action,
        action * 5,
        timestep,
        modality="action",
    )

    torch.testing.assert_close(video_x0, torch.full_like(video_x0, 4.2))
    torch.testing.assert_close(action_x0, torch.full_like(action_x0, 4.75))


def test_predict_joint_preserves_student_gradient() -> None:
    model = _JointModel()
    wrapper = WanDiffusionWrapper(
        model,
        video_scheduler=_scheduler(500, 0.5),
        action_scheduler=_scheduler(500, 0.5),
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

    output = wrapper.predict_joint(input_dict, noisy, timesteps)
    (output.x0.video.sum() + output.x0.action.sum()).backward()

    assert model.scale.grad is not None
    torch.testing.assert_close(model.scale.grad, torch.tensor(-2.5))


def test_wrapper_does_not_register_borrowed_model() -> None:
    wrapper = WanDiffusionWrapper(
        _JointModel(),
        video_scheduler=_scheduler(1, 1),
        action_scheduler=_scheduler(1, 1),
    )

    assert not isinstance(wrapper, nn.Module)
    assert not hasattr(wrapper, "state_dict")

