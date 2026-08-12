"""Consistency distillation trainer."""
from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)

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
        self.optimizer = self._build_optimizer(config, self.model.generate.model)
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
        )

    def _wrap_method_models(self) -> None:
        generate = set_trainable(self.model.generate.model)
        apply_mot_parameter_ownership(generate)
        apply_ac_mot(generate)
        generate = _configure_model(
            model=generate,
            shard_fn=partial(shard_mot_model),
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        generate.train()

        teacher = freeze_model(self.model.teacher.model)
        teacher = _configure_model(
            model=teacher,
            shard_fn=partial(shard_mot_model),
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )

        target = freeze_model(self.model.target.model)
        target = _configure_model(
            model=target,
            shard_fn=partial(shard_mot_model),
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )
        self.model.attach_wrapped_models(
            generate=generate,
            teacher=teacher,
            target=target,
        )

    def _trainable_model(self) -> Any:
        return self.model.generate.model

    def _export_model(self) -> Any:
        return self.model.target.model

    def _optimization_target(self) -> OptimizationTarget:
        return OptimizationTarget(
            name="generator",
            optimizer=self.optimizer,
            model=self.model.generate.model,
        )

    def _after_optimizer_step(self, target: OptimizationTarget) -> None:
        self.lr_scheduler.step()
        self.model.after_student_step(self.model.generate.model)

    def _extra_save_state(self, state: dict[str, Any]) -> None:
        state["ema_student"] = get_model_state_dict(
            self.model.target.model,
            options=self._checkpoint_options(),
        )

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        set_model_state_dict(
            self.model.target.model,
            state["ema_student"],
            options=self._checkpoint_options(),
        )

    @torch.no_grad()
    def rollout(self, batch: dict):
        """Run consistency-stage EMA inference through the shared AR pipeline."""
        batch = self.convert_input_format(batch)
        batch = self._materialize_batch_latents(batch)
        base_input = self._prepare_joint_input_dict(batch, add_noise=False)
        return self.model.rollout(
            batch,
            text_emb=base_input["latent_dict"]["text_emb"],
        )
