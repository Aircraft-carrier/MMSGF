"""Few-step scheduler construction for self rollout."""
from __future__ import annotations

from dataclasses import dataclass

from wan_va.utils.scheduler import FlowMatchScheduler


@dataclass(frozen=True, slots=True)
class RolloutSchedulers:
    video: FlowMatchScheduler
    action: FlowMatchScheduler


def build_rollout_schedulers(config, *, video_num_steps: int, action_num_steps: int) -> RolloutSchedulers:
    if int(video_num_steps) <= 0 or int(action_num_steps) <= 0:
        raise ValueError("video_num_steps and action_num_steps must be positive")
    video = FlowMatchScheduler(
        shift=float(config.snr_shift),
        sigma_min=0.0,
        extra_one_step=True,
    )
    video.set_timesteps(int(video_num_steps))
    action = FlowMatchScheduler(
        shift=float(config.action_snr_shift),
        sigma_min=0.0,
        extra_one_step=True,
    )
    action.set_timesteps(int(action_num_steps))
    return RolloutSchedulers(video=video, action=action)
