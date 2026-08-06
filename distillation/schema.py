"""Shared distillation data structures."""
from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass(frozen=True, slots=True)
class VAPrediction:
    """一对同阶段 V/A tensor；video 6-D，action 5-D，二者共享 B/F。"""

    video: torch.Tensor
    action: torch.Tensor


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
class CheckpointMetadata:
    format_version: int
    checkpoint_type: str
    vggto_attention_topology: str
    optimization_composition: str
    has_full_state: bool
    distill_method: str
    exported_model: str
    step: int
    optimizer_step: int
    generation_profile: dict[str, Any]
    model_architecture: str = "autoregressive_mot_v1"


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

    Student batch 使用预测 target V/A/G，teacher batch 保持 GT V/A/G；
    ``rollout_noisy`` 是 rollout 某个真实 denoise step 的 sampler state，
    ``pred_clean`` 是最终预测 clean，``teacher_clean`` 是 dataset GT clean。
    所有 record tensor 都已 detach，梯度只在 replay 时重新建立。
    """

    student_batch: dict[str, Any]
    teacher_batch: dict[str, Any]
    rollout_timesteps: VATimesteps
    rollout_noisy: VAPrediction
    pred_clean: VAPrediction
    teacher_clean: VAPrediction
    masks: VAMasks

    @property
    def batch(self) -> dict[str, Any]:
        """Compatibility alias for the legacy student replay batch."""

        return self.student_batch

    @property
    def timesteps(self) -> VATimesteps:
        return self.rollout_timesteps

    @property
    def noisy(self) -> VAPrediction:
        return self.rollout_noisy

    @property
    def generated(self) -> VAPrediction:
        return self.pred_clean
