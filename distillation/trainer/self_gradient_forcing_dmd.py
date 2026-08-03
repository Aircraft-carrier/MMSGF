"""Self-gradient-forcing DMD trainer."""
from __future__ import annotations

from typing import Any

import torch

from distillation.configs import SELF_GRADIENT_FORCING_DMD
from distillation.schema import TrainingStepResult
from distillation.trainer.base import DistillationTrainerBase, OptimizationTarget
from wan_va.train_mot import (
    _configure_adamw_foreach,
    build_mot_param_groups,
)


class SelfGradientForcingDMDTrainer(DistillationTrainerBase):
    method = SELF_GRADIENT_FORCING_DMD

    def __init__(self, config: Any):
        super().__init__(config)
        self.fake_score_optimizer = self._build_fake_score_optimizer(config)
        if self._resume_from is not None:
            self.checkpoint_io.load(self, self._resume_from)

    def _build_method_model(self, config: Any):
        from distillation.model.dmd import SGFDMDModel

        real_score_checkpoint = getattr(
            config.distill,
            "real_score_checkpoint",
            None,
        )
        if real_score_checkpoint is None:
            raise ValueError(
                "self_gradient_forcing_dmd requires distill.real_score_checkpoint"
            )
        fake_score_init = (
            getattr(config.distill, "fake_score_init", None)
            or getattr(config.distill, "resume_from", None)
        )
        if fake_score_init is None:
            raise ValueError(
                "self_gradient_forcing_dmd requires distill.fake_score_init "
                "for a fresh run"
            )
        return SGFDMDModel(
            config=config,
            trainer=self,
            student=self.transformer,
            device=self.device,
            real_score_checkpoint=str(real_score_checkpoint),
            fake_score_init=str(fake_score_init),
        )

    def _build_fake_score_optimizer(self, config: Any) -> torch.optim.Optimizer:
        param_groups = build_mot_param_groups(
            self.method_model.fake_score,
            base_lr=config.learning_rate,
            vggto_lr_multiplier=config.vggto_lr_multiplier,
            weight_decay=config.weight_decay,
        )
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
        )
        _configure_adamw_foreach(optimizer)
        return optimizer

    def _optimization_target(self) -> OptimizationTarget:
        name = self.method_model.optimizer_for_step(self.optimizer_step)
        if name == "student":
            return super()._optimization_target()
        return OptimizationTarget(
            name="fake_score",
            optimizer=self.fake_score_optimizer,
            model=self.method_model.fake_score,
        )

    def _compute_training_step(
        self,
        batch: dict,
        target: OptimizationTarget,
    ) -> TrainingStepResult:
        return self.method_model.compute_step(batch, target.name)

    def _after_optimizer_step(self, target: OptimizationTarget) -> None:
        if target.name == "student":
            super()._after_optimizer_step(target)
