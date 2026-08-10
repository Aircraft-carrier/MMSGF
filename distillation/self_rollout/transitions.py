"""Transition modes and schedule construction for self rollout."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from wan_va.utils.scheduler import FlowMatchScheduler


@dataclass(frozen=True, slots=True)
class RolloutSchedulers:
    video: FlowMatchScheduler
    action: FlowMatchScheduler


RolloutTransitionMode = Literal["inference", "sgf_renoise"]


@dataclass(frozen=True, slots=True)
class SGFRolloutSchedule:
    video_steps: tuple[float, ...]
    action_steps: tuple[float, ...]
    video_scheduler: FlowMatchScheduler
    action_scheduler: FlowMatchScheduler


def normalize_rollout_transition_mode(value: str) -> RolloutTransitionMode:
    mode = str(value)
    if mode not in {"inference", "sgf_renoise"}:
        raise ValueError(f"unsupported rollout transition mode: {mode!r}")
    return mode


def _validated_steps(values, *, name: str, maximum: float) -> tuple[float, ...]:
    steps = tuple(float(value) for value in values)
    if not steps:
        raise ValueError(f"{name} must contain at least one timestep")
    for index, value in enumerate(steps):
        if not math.isfinite(value) or not 0.0 < value <= maximum:
            raise ValueError(
                f"{name}[{index}] must be finite and in (0,{maximum}], got {value}"
            )
    if any(left <= right for left, right in zip(steps, steps[1:])):
        raise ValueError(f"{name} must be strictly descending, got {steps}")
    return steps


def build_sgf_rollout_schedule(
    config,
    *,
    video_scheduler: FlowMatchScheduler,
    action_scheduler: FlowMatchScheduler,
) -> SGFRolloutSchedule:
    values = config.distill.denoisy_step_list
    maximum = float(video_scheduler.num_train_timesteps)
    if float(action_scheduler.num_train_timesteps) != maximum:
        raise ValueError("video and action train schedulers must share timestep range")
    return SGFRolloutSchedule(
        video_steps=_validated_steps(
            values.video,
            name="distill.denoisy_step_list.video",
            maximum=maximum,
        ),
        action_steps=_validated_steps(
            values.action,
            name="distill.denoisy_step_list.action",
            maximum=maximum,
        ),
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
    )


def build_rollout_schedulers(
    config,
    *,
    video_num_steps: int,
    action_num_steps: int,
) -> RolloutSchedulers:
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
