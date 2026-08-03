"""Consistency distillation model state."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from distillation.model.factory import build_frozen_transformer
from distillation.model.utils import update_ema
from distillation.pipeline.consistency_training import ConsistencyTrainingPipeline
from distillation.schema import TrainingStepResult


class ConsistencyModel:
    """Own the three stage2 model roles and the small resume-only method state.

    ``student`` 由父 trainer 从 ``student_init/transformer`` 加载并可训练；
    ``teacher`` 从 autoregressive checkpoint 加载、冻结；``ema_student`` 初始与
    student 来源相同，之后只由 EMA hook 更新。跨阶段 export 选 EMA，而同阶段
    resume 的 raw student/optimizer/EMA 全部由 checkpoint DCP 保存。
    """

    def __init__(
        self,
        config: Any,
        trainer: Any,
        student: nn.Module,
        device: torch.device,
        teacher_checkpoint: str | None,
    ):
        self.ema_decay = float(config.distill.ema_decay)
        self.teacher = build_frozen_transformer(teacher_checkpoint, config, device)
        ema_init = config.distill.student_init or config.distill.resume_from
        self.ema_student = build_frozen_transformer(
            ema_init,
            config,
            device,
        )

        self.pipeline = ConsistencyTrainingPipeline(
            config=config,
            trainer=trainer,
            device=device,
            student=student,
            teacher=self.teacher,
            ema_student=self.ema_student,
        )

    def compute_step(
        self,
        batch: dict,
    ) -> TrainingStepResult:
        loss, metrics = self.pipeline.compute_loss(batch)
        return TrainingStepResult(loss=loss, metrics=metrics)

    @torch.no_grad()
    def after_student_step(self, student: nn.Module) -> None:
        update_ema(self.ema_student, student, decay=self.ema_decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "ema_decay": self.ema_decay,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.ema_decay = float(state.get("ema_decay", self.ema_decay))
