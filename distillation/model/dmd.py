"""Self-gradient-forcing DMD model state."""
from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from distillation.schema import DMDUpdateSchedule, TrainingStepResult
from distillation.model.factory import build_frozen_transformer, build_trainable_transformer
from distillation.pipeline.self_gradient_forcing_training import SelfGradientForcingTrainingPipeline


class SGFDMDModel:
    """Own stage3 student/real-score/fake-score and optimizer routing.

    student 是 stage2 checkpoint 导出的 EMA；real-score/fake-score 都使用
    双向 VAMOT 模型和原始 wan_va mask。real-score 在 DMD 分支估计 real
    distribution 并永久冻结，fake 有独立 AdamW。每个
    microstep 先 no-grad 生成 ReplayContext，再根据 optimizer_step 只建立一条
    梯度路径。
    """

    def __init__(
        self,
        config: Any,
        trainer: Any,
        student: nn.Module,
        device: torch.device,
        real_score_checkpoint: str | None,
        fake_score_init: str | None,
    ):
        self.update_schedule = DMDUpdateSchedule(
            fake_score_steps=int(config.distill.fake_score_update_ratio)
        )

        self.real_score = build_frozen_transformer(
            real_score_checkpoint,
            config,
            device,
            install_distillation_profile=False,
            validate_distillation_profile=False,
            autoregressive=False,
        )
        self.fake_score = build_trainable_transformer(
            fake_score_init,
            config,
            device,
            install_distillation_profile=False,
            validate_distillation_profile=False,
            autoregressive=False,
        )
        if type(self.real_score) is not type(self.fake_score):
            raise TypeError(
                "DMD real-score and fake-score must use the same bidirectional model class"
            )

        self.pipeline = SelfGradientForcingTrainingPipeline(
            config=config,
            trainer=trainer,
            device=device,
            student=student,
            real_score=self.real_score,
            fake_score=self.fake_score,
        )

    def optimizer_for_step(self, optimizer_step: int) -> Literal["student", "fake_score"]:
        return self.update_schedule.optimizer_for_step(optimizer_step)

    def compute_step(
        self,
        batch: dict,
        optimizer_name: Literal["student", "fake_score"],
    ) -> TrainingStepResult:
        context = self.pipeline.generate_and_record_context(batch)
        if optimizer_name == "student":
            loss, metrics = self.pipeline.replay_and_score(context)
        else:
            loss, metrics = self.pipeline.fake_score_step(context)
        return TrainingStepResult(loss=loss, metrics=metrics)

    def state_dict(self) -> dict[str, Any]:
        return {
            "fake_score_update_ratio": self.update_schedule.fake_score_steps,
            "video_denoisy_step_list": tuple(self.pipeline.sgf_schedule.video_steps),
            "action_denoisy_step_list": tuple(self.pipeline.sgf_schedule.action_steps),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ratio = int(
            state.get(
                "fake_score_update_ratio",
                self.update_schedule.fake_score_steps,
            )
        )
        expected_video = tuple(self.pipeline.sgf_schedule.video_steps)
        expected_action = tuple(self.pipeline.sgf_schedule.action_steps)
        saved_video = tuple(state.get("video_denoisy_step_list", expected_video))
        saved_action = tuple(state.get("action_denoisy_step_list", expected_action))
        if saved_video != expected_video or saved_action != expected_action:
            raise ValueError(
                "SGF denoisy step lists do not match checkpoint: "
                f"saved video/action={saved_video}/{saved_action}, "
                f"configured={expected_video}/{expected_action}"
            )
        self.update_schedule = DMDUpdateSchedule(fake_score_steps=ratio)
