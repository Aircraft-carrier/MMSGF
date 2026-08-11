"""Self-gradient-forcing DMD trainer."""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from distillation.configs import SELF_GRADIENT_FORCING_DMD
from distillation.model.utils import freeze_model, set_trainable
from distillation.schema import TrainingStepResult
from distillation.trainer.base import DistillationTrainerBase, OptimizationTarget
from wan_va.train_mot import (
    _configure_adamw_foreach,
    apply_ac_mot,
    apply_mot_parameter_ownership,
    shard_mot_model,
)
from wan_va.distributed.util import _configure_model

if TYPE_CHECKING:
    from distillation.model.dmd import SGFDMDModel


class SelfGradientForcingDMDTrainer(DistillationTrainerBase):
    """Load the DMD model, wrap/resume its models, and own its optimizers."""

    method = SELF_GRADIENT_FORCING_DMD
    model: "SGFDMDModel"

    def __init__(self, config: Any):
        super().__init__(config)
        self._build_optimizer(config)
        self.lr_scheduler = self._build_lr_scheduler(config, self.optimizer)
        if self._resume_from is not None:
            self.load_checkpoint(self._resume_from)

    def _build_optimizer(self, config: Any) -> None:
        """Build both the generator and fake-score optimizers."""
        self.optimizer = super()._build_optimizer(
            config,
            self.model.generator.model,
        )
        self.fake_score_optimizer = super()._build_optimizer(
            config,
            self.model.fake_score.model,
        )

    def _build_method_model(self, config: Any) -> "SGFDMDModel":
        from distillation.model.dmd import SGFDMDModel

        student_init = getattr(config.distill, "student_init", None)
        if student_init is None:
            raise ValueError(
                "self_gradient_forcing_dmd requires distill.student_init"
            )
        real_score_checkpoint = getattr(
            config.distill,
            "real_score_checkpoint",
            None,
        )
        if real_score_checkpoint is None:
            raise ValueError(
                "self_gradient_forcing_dmd requires distill.real_score_checkpoint"
            )
        # A resume checkpoint publicly exports the AR generator, not the private
        # bidirectional fake-score.  Construct the fake-score skeleton from an
        # explicit bidirectional export (or the real-score source) and restore
        # its exact parameters from the resume checkpoint afterwards.
        fake_score_init = (
            getattr(config.distill, "fake_score_init", None)
            or real_score_checkpoint
        )
        return SGFDMDModel(
            config=config,
            device=self.device,
            student_init=str(student_init),
            real_score_checkpoint=str(real_score_checkpoint),
            fake_score_init=str(fake_score_init),
            resume_from=self._resume_from,
        )

    def _wrap_method_models(self) -> None:
        generator = self.model.generator.model
        apply_mot_parameter_ownership(generator)
        apply_ac_mot(generator)
        generator = _configure_model(
            model=generator,
            shard_fn=partial(shard_mot_model),
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        generator.train()

        real_score = freeze_model(self.model.real_score.model)
        real_score = _configure_model(
            model=real_score,
            shard_fn=partial(shard_mot_model),
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )

        fake_score = set_trainable(self.model.fake_score.model)
        apply_mot_parameter_ownership(fake_score)
        apply_ac_mot(fake_score)
        fake_score = _configure_model(
            model=fake_score,
            shard_fn=partial(shard_mot_model),
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        fake_score.train()
        self.model.attach_wrapped_models(
            generator=generator,
            real_score=real_score,
            fake_score=fake_score,
        )

    def _optimization_target(self) -> OptimizationTarget:
        name = self.model.optimizer_for_step(self.optimizer_step)
        if name == "generator":
            return OptimizationTarget(
                name="generator",
                optimizer=self.optimizer,
                model=self.model.generator.model,
            )
        return OptimizationTarget(
            name="fake_score",
            optimizer=self.fake_score_optimizer,
            model=self.model.fake_score.model,
        )

    def _compute_training_step(
        self,
        batch: dict,
        base_input: dict,
        empty_text_emb: torch.Tensor,
        target: OptimizationTarget,
    ) -> TrainingStepResult:
        return self.model.compute_step(
            batch,
            target.name,
            base_input=base_input,
            empty_text_emb=empty_text_emb,
        )

    def _after_optimizer_step(self, target: OptimizationTarget) -> None:
        if target.name == "generator":
            self.lr_scheduler.step()

    def _extra_save_state(self, state: dict[str, Any]) -> None:
        state["optimizer"] = self.optimizer.state_dict()
        state["lr_scheduler"] = self.lr_scheduler.state_dict()
        state["fake_score"] = self._full_model_state(
            self.model.fake_score.model
        )
        state["fake_score_optimizer"] = self.fake_score_optimizer.state_dict()

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self._restore_full_model_state(
            self.model.fake_score.model,
            state["fake_score"],
        )
        self.fake_score_optimizer.load_state_dict(state["fake_score_optimizer"])
        _configure_adamw_foreach(self.optimizer)
        _configure_adamw_foreach(self.fake_score_optimizer)
