"""Consistency distillation trainer."""
from __future__ import annotations

from functools import partial
from typing import Any

from distillation.configs import CONSISTENCY_DISTILLATION
from distillation.model.utils import freeze_model, set_trainable
from distillation.trainer.base import DistillationTrainerBase, OptimizationTarget
from wan_va.train_mot import (
    apply_ac_mot,
    apply_mot_parameter_ownership,
    shard_mot_model,
)
from wan_va.distributed.util import _configure_model


class ConsistencyTrainer(DistillationTrainerBase):
    """Load the consistency model from paths, wrap it, and run EMA updates."""

    method = CONSISTENCY_DISTILLATION

    def __init__(self, config: Any):
        super().__init__(config)
        self.optimizer = self._build_optimizer(config, self.model.student)
        self.lr_scheduler = self._build_lr_scheduler(config, self.optimizer)
        if self._resume_from is not None:
            self.load_checkpoint(self._resume_from)

    def _build_method_model(self, config: Any):
        from distillation.model.consistency import ConsistencyModel

        student_init = getattr(config.distill, "student_init", None)
        if student_init is None:
            raise ValueError(
                "consistency_distillation requires distill.student_init"
            )
        teacher_checkpoint = getattr(config.distill, "teacher_checkpoint", None)
        if teacher_checkpoint is None:
            raise ValueError(
                "consistency_distillation requires distill.teacher_checkpoint"
            )
        return ConsistencyModel(
            config=config,
            device=self.device,
            student_init=str(student_init),
            teacher_checkpoint=str(teacher_checkpoint),
            resume_from=self._resume_from,
        )

    def _wrap_method_models(self) -> None:
        student = set_trainable(self.model.student)
        apply_mot_parameter_ownership(student)
        apply_ac_mot(student)
        student = _configure_model(
            model=student,
            shard_fn=partial(shard_mot_model),
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        student.train()

        teacher = freeze_model(self.model.teacher)
        teacher = _configure_model(
            model=teacher,
            shard_fn=partial(shard_mot_model),
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )

        ema_student = freeze_model(self.model.ema_student)
        ema_student = _configure_model(
            model=ema_student,
            shard_fn=partial(shard_mot_model),
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )
        self.model.attach_wrapped_models(
            student=student,
            teacher=teacher,
            ema_student=ema_student,
        )

    def _trainable_model(self) -> Any:
        return self.model.student

    def _optimization_target(self) -> OptimizationTarget:
        return OptimizationTarget(
            name="generator",
            optimizer=self.optimizer,
            model=self.model.student,
        )

    def _after_optimizer_step(self, target: OptimizationTarget) -> None:
        self.lr_scheduler.step()
        self.model.after_student_step(self.model.student)
