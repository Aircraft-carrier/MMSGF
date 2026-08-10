"""Shared distillation data structures."""
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass(frozen=True, slots=True)
class VAPrediction:
    """一对同阶段 V/A tensor；video 6-D，action 5-D，二者共享 B/F。"""

    video: torch.Tensor
    action: torch.Tensor


@dataclass(frozen=True, slots=True)
class VADiffusionOutput:
    """Flow-matching velocity and the corresponding clean V/A prediction."""

    velocity: VAPrediction
    x0: VAPrediction


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

    def frame_mask(self) -> torch.Tensor:
        # score noise 以 frame 为单位采样：只要 video 或 action 任一 token 在该
        # frame 受监督，就给它非零 timestep；两者都无效时返回 False。
        video = self.video.reshape(self.video.shape[0], -1)
        return video | self.action.any(dim=(1, 3, 4))


@dataclass(frozen=True, slots=True)
class VALossWeights:
    video: float = 1.0
    action: float = 1.0


@dataclass(frozen=True, slots=True)
class DenoisyInterval:
    """One modality's sampled rollout exit and DMD timestep interval."""

    exit_id: int
    denoisy_from: float
    denoisy_to: float

    def __post_init__(self) -> None:
        if int(self.exit_id) < 0:
            raise ValueError("exit_id must be non-negative")
        denoisy_from = float(self.denoisy_from)
        denoisy_to = float(self.denoisy_to)
        if (
            not math.isfinite(denoisy_from)
            or not math.isfinite(denoisy_to)
            or not 0.0 <= denoisy_to < denoisy_from
        ):
            raise ValueError(
                "denoisy interval requires 0 <= denoisy_to < denoisy_from, "
                f"got from={denoisy_from}, to={denoisy_to}"
            )


@dataclass(frozen=True, slots=True)
class VADenoisySelection:
    """Independent video and action rollout exits."""

    video: DenoisyInterval
    action: DenoisyInterval


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    format_version: int
    checkpoint_type: str
    model_architecture: str
    has_full_state: bool
    distill_method: str
    exported_model: str
    step: int
    optimizer_step: int
    generation_profile: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DMDUpdateSchedule:
    """用已完成 optimizer_step 决定下一窗口更新谁。

    ``fake_score_steps=4`` 时 step 0..4 为 fake,fake,fake,fake,student，之后
    重复。gradient accumulation 期间 optimizer_step 不变，因此一个累积窗口
    不会在中途切换模型或 optimizer。
    """

    fake_score_steps: int

    def optimizer_for_step(self, step: int) -> Literal["student", "fake_score"]:
        cycle_step = step % (self.fake_score_steps + 1)
        return "student" if cycle_step == self.fake_score_steps else "fake_score"


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    loss: torch.Tensor
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReplayContext:
    """Stage3 no-grad record 与有梯度 replay 之间的完整边界。

    ``rollout_noisy`` 是 V/A 各自 exit 的真实 sampler state；
    ``final_clean_context`` 是 GT history/anchor 加完整 rollout 的最终 x0。
    所有 record tensor 都已 detach，梯度只在 replay 时重新建立。
    """

    replay_batch: dict[str, Any]
    rollout_timesteps: VATimesteps
    rollout_noisy: VAPrediction
    final_clean_context: VAPrediction
    masks: VAMasks
    denoisy_selection: VADenoisySelection

    @property
    def batch(self) -> dict[str, Any]:
        """Compatibility alias for callers that consume the replay batch."""

        return self.replay_batch

    @property
    def student_batch(self) -> dict[str, Any]:
        """Compatibility alias for the former field name."""

        return self.replay_batch

    @property
    def timesteps(self) -> VATimesteps:
        return self.rollout_timesteps

    @property
    def noisy(self) -> VAPrediction:
        return self.rollout_noisy

    @property
    def generated(self) -> VAPrediction:
        return self.final_clean_context

    @property
    def pred_clean(self) -> VAPrediction:
        """Compatibility alias for the former final-clean field name."""

        return self.final_clean_context
