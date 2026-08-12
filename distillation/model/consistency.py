"""Consistency trajectory construction, loss computation, and model state."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from distillation.diffusion_utils import (
    consistency_prediction,
    flow_step,
    sample_consistency_timesteps,
)
from distillation.model.objectives import action_aware_loss, consistency_loss
from distillation.model.utils import (
    add_noise_to_va,
    replace_text_condition,
    replace_va_streams,
    update_ema,
)
from distillation.model.wan_wrapper import WanDiffusionWrapper
from distillation.pipeline import SelfGradientForcingTrainingPipeline
from distillation.schema import (
    TrainingStepResult,
    VALossWeights,
    VAMasks,
    VAPair,
    VATimesteps,
)


class ConsistencyBaseModel:
    """Own the student, frozen teacher, EMA target, and stage2 configuration."""

    def __init__(
        self,
        config: Any,
        device: torch.device,
        student_init: str | None,
        teacher_checkpoint: str | None,
        *,
        resume_from: str | None = None,
        student: nn.Module | None = None,
        teacher: nn.Module | None = None,
        ema_student: nn.Module | None = None,
    ) -> None:
        self.device = device
        self.ema_decay = float(config.distill.ema_decay)
        if student is None and student_init is None:
            raise ValueError("student_init path is required")
        self.student_wrapper = WanDiffusionWrapper(
            config=config,
            checkpoint_path=student_init,
            autoregressive=True,
            model=student,
        )
        ema_init = config.distill.student_init or config.distill.resume_from
        self.teacher_wrapper = WanDiffusionWrapper(
            config=config,
            checkpoint_path=teacher_checkpoint,
            autoregressive=True,
            model=teacher,
        )
        self.ema_wrapper = WanDiffusionWrapper(
            config=config,
            checkpoint_path=ema_init,
            autoregressive=True,
            model=ema_student,
        )
        self.student = self.student_wrapper.model
        self.teacher = self.teacher_wrapper.model
        self.ema_student = self.ema_wrapper.model
        self.train_scheduler_latent = self.student_wrapper.video_scheduler
        self.train_scheduler_action = self.student_wrapper.action_scheduler

        rollout_steps = config.distill.rollout_denoising_step_list
        self.rollout_history_frames = int(
            config.distill.generation_shape["history_frames"]
        )
        self.rollout_horizon_frames = int(config.distill.rollout_horizon_frames)
        self.pipeline = SelfGradientForcingTrainingPipeline(
            rollout_steps,
            self.ema_wrapper,
            num_frame_per_block=int(config.distill.rollout_num_frame_per_block),
            per_rank_exit_step=bool(config.distill.rollout_per_rank_exit_step),
        )

        self.video_num_steps = int(config.distill.video_num_steps)
        self.action_num_steps = int(config.distill.action_num_steps)
        self.cfg_min = float(config.distill.cfg_min)
        self.cfg_max = float(config.distill.cfg_max)
        self.sigma_data = float(config.distill.sigma_data)
        self.action_aware_weight = float(config.distill.action_aware_weight)
        self.loss_weights = VALossWeights(
            video=float(config.video_loss_weight),
            action=float(config.action_loss_weight),
        )

    def attach_wrapped_models(
        self,
        *,
        student: nn.Module | None = None,
        teacher: nn.Module | None = None,
        ema_student: nn.Module | None = None,
    ) -> None:
        """Install trainer-wrapped (FSDP/AC) models back into this model."""
        if student is not None:
            self.student = student
            self.student_wrapper.model = student
        if teacher is not None:
            self.teacher = teacher
            self.teacher_wrapper.model = teacher
        if ema_student is not None:
            self.ema_student = ema_student
            self.ema_wrapper.model = ema_student

    def _empty_text_condition(
        self,
        batch: dict,
        text_emb: torch.Tensor,
        empty_text_emb: torch.Tensor,
    ) -> torch.Tensor:
        empty = batch.get("empty_text_emb")
        if empty is None:
            empty = empty_text_emb
        return empty.to(
            device=text_emb.device,
            dtype=text_emb.dtype,
        ).expand_as(text_emb)

    def _sample_timesteps(
        self,
        masks: VAMasks,
    ) -> tuple[VATimesteps, VATimesteps]:
        shape = masks.video.shape
        video, next_video = sample_consistency_timesteps(
            self.train_scheduler_latent,
            self.video_num_steps,
            shape,
            self.device,
            masks.video,
        )
        action_mask = masks.action.any(dim=(1, 3, 4))
        action, next_action = sample_consistency_timesteps(
            self.train_scheduler_action,
            self.action_num_steps,
            shape,
            self.device,
            action_mask,
        )
        return VATimesteps(video, action), VATimesteps(next_video, next_action)


class ConsistencyTrainingModel(ConsistencyBaseModel):
    """Build one teacher trajectory and compute the student consistency loss."""

    def _predict_consistency(
        self,
        wrapper: WanDiffusionWrapper,
        input_dict: dict,
        noisy: VAPair,
        timesteps: VATimesteps,
    ) -> VAPair:
        output = wrapper(input_dict, noisy, timesteps)
        return VAPair(
            video=consistency_prediction(
                output.velocity.video,
                noisy.video,
                timesteps.video,
                self.train_scheduler_latent,
                sigma_data=self.sigma_data,
            ),
            action=output.x0.action,
        )

    def _teacher_cfg_flow(
        self,
        input_dict: dict,
        batch: dict,
        empty_text_emb: torch.Tensor,
        noisy: VAPair,
        timesteps: VATimesteps,
    ) -> tuple[VAPair, float]:
        conditioned = self.teacher_wrapper(input_dict, noisy, timesteps)
        text_emb = input_dict["latent_dict"]["text_emb"]
        empty_text_emb = self._empty_text_condition(
            batch,
            text_emb,
            empty_text_emb,
        )
        unconditioned = self.teacher_wrapper(
            replace_text_condition(input_dict, empty_text_emb),
            noisy,
            timesteps,
        )
        cfg_scale = self.cfg_min + torch.rand(()).item() * (
            self.cfg_max - self.cfg_min
        )
        video = unconditioned.velocity.video + cfg_scale * (
            conditioned.velocity.video - unconditioned.velocity.video
        )
        return VAPair(video, conditioned.velocity.action), cfg_scale

    def compute_loss(
        self,
        batch: dict,
        *,
        base_input: dict,
        empty_text_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Construct one adjacent consistency trajectory and its scalar loss."""
        bsz, _, frames = batch["latents"].shape[:3]
        masks = VAMasks(
            video=base_input["latent_dict"]["video_latent_loss_mask"].reshape(
                bsz,
                frames,
            ),
            action=base_input["action_dict"]["action_loss_mask"],
        )
        timesteps, next_timesteps = self._sample_timesteps(masks)

        clean = VAPair(batch["latents"], batch["actions"])
        noise = VAPair(
            torch.randn_like(clean.video),
            torch.randn_like(clean.action),
        )
        noisy = add_noise_to_va(
            clean,
            noise,
            timesteps,
            masks,
            self.train_scheduler_latent,
            self.train_scheduler_action,
        )

        with torch.no_grad():
            teacher_input = replace_va_streams(
                base_input,
                noisy,
                clean,
                timesteps,
            )
            teacher_flow, cfg_scale = self._teacher_cfg_flow(
                teacher_input,
                batch,
                empty_text_emb,
                noisy,
                timesteps,
            )
            next_noisy = VAPair(
                flow_step(
                    teacher_flow.video,
                    noisy.video,
                    timesteps.video,
                    next_timesteps.video,
                    self.train_scheduler_latent,
                ),
                flow_step(
                    teacher_flow.action,
                    noisy.action,
                    timesteps.action,
                    next_timesteps.action,
                    self.train_scheduler_action,
                ),
            )
            ema_input = replace_va_streams(
                base_input,
                next_noisy,
                clean,
                next_timesteps,
            )
            target_consistency = self._predict_consistency(
                self.ema_wrapper,
                ema_input,
                next_noisy,
                next_timesteps,
            )

        student_input = replace_va_streams(base_input, noisy, clean, timesteps)
        student_out = self.student_wrapper(student_input, noisy, timesteps)
        student_flow = student_out.velocity
        student_consistency = VAPair(
            video=consistency_prediction(
                student_flow.video,
                noisy.video,
                timesteps.video,
                self.train_scheduler_latent,
                sigma_data=self.sigma_data,
            ),
            action=student_out.x0.action,
        )

        consistency, metrics = consistency_loss(
            student_consistency,
            target_consistency,
            masks,
            self.loss_weights,
        )
        action_target = self.train_scheduler_action.training_target(
            clean.action,
            noise.action,
            timesteps.action,
        )
        aware, aware_metrics = action_aware_loss(
            student_flow.action,
            action_target,
            masks.action,
        )
        loss = consistency + self.action_aware_weight * aware

        metrics.update(aware_metrics)
        metrics["distill/action_aware_weighted_loss"] = (
            self.action_aware_weight * aware.detach()
        )
        metrics["distill/consistency_training_loss"] = loss.detach()
        metrics["distill/teacher_cfg_scale"] = loss.detach().new_tensor(cfg_scale)
        metrics["distill/consistency_video_t_mean"] = timesteps.video.float().mean()
        metrics["distill/consistency_action_t_mean"] = timesteps.action.float().mean()
        metrics["distill/consistency_video_t_next_mean"] = (
            next_timesteps.video.float().mean()
        )
        metrics["distill/consistency_action_t_next_mean"] = (
            next_timesteps.action.float().mean()
        )
        return loss, metrics


class ConsistencyModel(ConsistencyTrainingModel):
    """Expose trainer-facing step, EMA, and checkpoint state contracts."""

    def compute_step(
        self,
        batch: dict,
        *,
        base_input: dict,
        empty_text_emb: torch.Tensor,
    ) -> TrainingStepResult:
        loss, metrics = self.compute_loss(
            batch,
            base_input=base_input,
            empty_text_emb=empty_text_emb,
        )
        return TrainingStepResult(loss=loss, metrics=metrics)

    @torch.no_grad()
    def rollout(
        self,
        batch: dict,
        *,
        text_emb: torch.Tensor | None = None,
    ):
        rollout_batch = dict(batch)
        if text_emb is not None:
            rollout_batch["text_emb"] = text_emb
        return self.pipeline.generate(
            rollout_batch,
            rollout_frames=self.rollout_horizon_frames,
            history_frames=self.rollout_history_frames,
            device=self.device,
        )

    @torch.no_grad()
    def after_student_step(self, student: nn.Module) -> None:
        update_ema(self.ema_student, student, decay=self.ema_decay)

    def state_dict(self) -> dict[str, Any]:
        return {"ema_decay": self.ema_decay}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.ema_decay = float(state.get("ema_decay", self.ema_decay))
