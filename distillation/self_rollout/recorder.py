"""Optional denoise-state recording for distillation self rollout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from distillation.schema import DenoisyInterval


@dataclass(frozen=True, slots=True)
class RecordedDenoiseState:
    sample: torch.Tensor
    timestep: torch.Tensor
    exit_id: int
    denoisy_from: float
    denoisy_to: float


class SelfRolloutRecorder:
    """Record one real scheduler input state per generated frame and modality."""

    def __init__(
        self,
        *,
        video: DenoisyInterval,
        action: DenoisyInterval,
    ) -> None:
        self.intervals = {
            "video": video,
            "action": action,
        }
        self.video: dict[int, RecordedDenoiseState] = {}
        self.action: dict[int, RecordedDenoiseState] = {}

    def validate(self, *, video_num_steps: int, action_num_steps: int) -> None:
        limits = {
            "video": int(video_num_steps),
            "action": int(action_num_steps),
        }
        for modality, limit in limits.items():
            step = self.intervals[modality].exit_id
            if step >= limit:
                raise ValueError(
                    f"{modality}_exit_id must be in [0,{limit}), got {step}"
                )

    def observe(
        self,
        modality: Literal["video", "action"],
        *,
        frame_id: int,
        step_index: int,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> None:
        interval = self.intervals[modality]
        if int(step_index) != interval.exit_id:
            return
        records = self.video if modality == "video" else self.action
        frame_id = int(frame_id)
        if frame_id in records:
            raise RuntimeError(
                f"duplicate {modality} denoise record for frame {frame_id}"
            )
        records[frame_id] = RecordedDenoiseState(
            sample=sample.detach().clone(),
            timestep=torch.as_tensor(timestep, device=sample.device).detach().clone(),
            exit_id=interval.exit_id,
            denoisy_from=interval.denoisy_from,
            denoisy_to=interval.denoisy_to,
        )

    def require_frames(self, frame_ids: list[int]) -> None:
        expected = set(map(int, frame_ids))
        for modality, records in (("video", self.video), ("action", self.action)):
            missing = sorted(expected - set(records))
            if missing:
                raise RuntimeError(
                    f"self rollout did not record {modality} frames {missing}"
                )
