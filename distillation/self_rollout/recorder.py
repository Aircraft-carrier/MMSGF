"""Optional denoise-state recording for distillation self rollout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, slots=True)
class RecordedDenoiseState:
    sample: torch.Tensor
    timestep: torch.Tensor


class SelfRolloutRecorder:
    """Record one real scheduler input state per generated frame and modality."""

    def __init__(self, *, video_step: int, action_step: int) -> None:
        self.record_steps = {
            "video": int(video_step),
            "action": int(action_step),
        }
        if any(step < 0 for step in self.record_steps.values()):
            raise ValueError("record steps must be non-negative")
        self.video: dict[int, RecordedDenoiseState] = {}
        self.action: dict[int, RecordedDenoiseState] = {}

    def validate(self, *, video_num_steps: int, action_num_steps: int) -> None:
        limits = {
            "video": int(video_num_steps),
            "action": int(action_num_steps),
        }
        for modality, limit in limits.items():
            step = self.record_steps[modality]
            if step >= limit:
                raise ValueError(
                    f"{modality}_step must be in [0,{limit}), got {step}"
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
        if int(step_index) != self.record_steps[modality]:
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
        )

    def require_frames(self, frame_ids: list[int]) -> None:
        expected = set(map(int, frame_ids))
        for modality, records in (("video", self.video), ("action", self.action)):
            missing = sorted(expected - set(records))
            if missing:
                raise RuntimeError(
                    f"self rollout did not record {modality} frames {missing}"
                )
