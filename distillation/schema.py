"""Shared distillation data structures."""
from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass(frozen=True, slots=True)
class VAPair:
    """一对同阶段 V/A tensor；video 6-D，action 5-D，二者共享 B/F。"""

    video: torch.Tensor
    action: torch.Tensor


@dataclass(frozen=True, slots=True)
class VADiffusionOutput:
    """Flow-matching velocity and the corresponding clean V/A prediction."""

    velocity: VAPair
    x0: VAPair


@dataclass(frozen=True, slots=True)
class VATimesteps:
    """V/A 各自的 frame timestep，均为 ``[B,F]``，不假设 sigma 相同。"""

    video: torch.Tensor
    action: torch.Tensor


@dataclass(frozen=True, slots=True)
class VAMasks:
    """Video frame mask ``[B,F]`` 与 action element mask ``[B,Ca,F,N,1]``。"""

    video: torch.Tensor
    action: torch.Tensor


@dataclass(frozen=True, slots=True)
class VALossWeights:
    video: float = 1.0
    action: float = 1.0


@dataclass(frozen=True, slots=True)
class DenoisyInterval:
    """One rollout exit and its candidate DMD bounds in linear progress d.

    ``denoisy_from`` and ``denoisy_to`` are not network timesteps. The DMD
    schedule switches decide whether each bound is active; sampled ``d`` is
    only then mapped through the modality-specific scheduler.
    """

    exit_id: int
    denoisy_from: float
    denoisy_to: float


@dataclass(frozen=True, slots=True)
class VADenoisySelection:
    """Independent video and action rollout exits."""

    video: DenoisyInterval
    action: DenoisyInterval


@dataclass(frozen=True, slots=True)
class DMDUpdateSchedule:
    """用已完成 optimizer_step 决定下一窗口更新谁。

    ``fake_score_steps=4`` 时 step 0..4 为 fake,fake,fake,fake,generator，之后
    重复。gradient accumulation 期间 optimizer_step 不变，因此一个累积窗口
    不会在中途切换模型或 optimizer。
    """

    fake_score_steps: int

    def optimizer_for_step(self, step: int) -> Literal["generator", "fake_score"]:
        cycle_step = step % (self.fake_score_steps + 1)
        return "generator" if cycle_step == self.fake_score_steps else "fake_score"


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    loss: torch.Tensor
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReplayContext:
    """Stage3 no-grad record 与有梯度 replay 之间的完整边界。

    ``noisy_at_t`` 是 V/A 各自 exit 的真实 sampler state；
    ``clean_hat`` 是 GT history/anchor 加完整 rollout 的最终 x0。
    所有 record tensor 都已 detach，梯度只在 replay 时重新建立。
    """

    exit_timesteps: VATimesteps
    noisy_at_t: VAPair
    clean_hat: VAPair
    masks: VAMasks
    denoisy_selection: VADenoisySelection
