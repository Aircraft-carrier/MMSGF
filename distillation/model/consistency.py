"""Consistency trajectory construction, loss computation, and model state."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from distillation.model.dmd import update_ema
from distillation.model.utils import (
    add_noise,
    apply_va_mask,
    broadcast_frame_values,
    randn_like_va,
    replace_text_condition,
    replace_va_streams,
    sigmas_for_timesteps,
    va_loss,
)
from distillation.model.wan_wrapper import WanDiffusionWrapper
from distillation.pipeline import SelfGradientForcingTrainingPipeline
from distillation.schema import (
    TrainingStepResult,
    VADiffusionOutput,
    VALossWeights,
    VAMasks,
    VAPair,
    VATimesteps,
)

from wan_va.utils.scheduler import FlowMatchScheduler


def consistency_loss(
    student_x0: VAPair,
    target_x0: VAPair,
    masks: VAMasks,
    weights: VALossWeights = VALossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the masked weighted V/A consistency objective."""
    return va_loss(
        student_x0,
        target_x0,
        masks,
        weights,
        "consistency",
    )


def action_aware_loss(
    student_flow: torch.Tensor,
    target_flow: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress student action flow against the exact flow-matching target."""
    loss = F.mse_loss(
        student_flow.float(),
        target_flow.float().detach(),
        reduction="none",
    )
    mask = mask.expand_as(loss)
    loss = torch.where(mask, loss, 0).sum() / mask.sum().clamp_min(1)
    return loss, {"distill/action_aware_loss": loss.detach()}


def sample_consistency_timesteps(
    scheduler: FlowMatchScheduler,
    num_steps: int,
    shape: tuple[int, int],
    device: torch.device,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    schedule = scheduler.timesteps.to(device)
    stride = len(schedule) // num_steps
    timestep_ids = torch.randint(len(schedule), shape, device=device)
    next_ids = (timestep_ids + stride).clamp_max(len(schedule) - 1)
    timesteps = torch.where(mask, schedule[timestep_ids], 0)
    next_timesteps = torch.where(mask, schedule[next_ids], 0)
    return timesteps, next_timesteps


class ConsistencyBaseModel:
    """Own the student, frozen teacher, EMA target, and stage2 configuration."""

    def __init__(
        self,
        config: Any,
        device: torch.device,
        student_init: str | None,
        teacher_checkpoint: str | None,
    ) -> None:
        self.config = config
        self.device = device
        self.ema_decay = float(config.distill.ema_decay)
        self._initialize_models(
            student_init=student_init,
            teacher_checkpoint=teacher_checkpoint,
        )

        self.train_scheduler_latent = self.generate.video_scheduler
        self.train_scheduler_action = self.generate.action_scheduler

        rollout_steps = config.distill.rollout_denoising_step_list
        self.rollout_history_frames = int(
            config.distill.generation_shape["history_frames"]
        )
        self.rollout_horizon_frames = int(config.distill.rollout_horizon_frames)
        self.pipeline = SelfGradientForcingTrainingPipeline(
            rollout_steps,
            self.target,
            num_frame_per_block=int(config.distill.rollout_num_frame_per_block),
            per_rank_exit_step=bool(config.distill.rollout_per_rank_exit_step),
        )

        self.video_num_steps = int(config.distill.video_num_steps)
        self.action_num_steps = int(config.distill.action_num_steps)
        self.cfg_min = float(config.distill.cfg_min)
        self.cfg_max = float(config.distill.cfg_max)
        self.reuse_teacher_noise = bool(config.distill.reuse_teacher_noise)
        self.sigma_data = float(config.distill.sigma_data)
        self.action_aware_weight = float(config.distill.action_aware_weight)
        self.loss_weights = VALossWeights(
            video=float(config.video_loss_weight),
            action=float(config.action_loss_weight),
        )

    def _initialize_models(
        self,
        *,
        student_init: str,
        teacher_checkpoint: str,
    ) -> None:
        """Load the trainable, teacher, and EMA-target wrappers from paths."""
        self.generate = WanDiffusionWrapper(
            config=self.config,
            checkpoint_path=student_init,
            autoregressive=True,
        )
        self.teacher = WanDiffusionWrapper(
            config=self.config,
            checkpoint_path=teacher_checkpoint,
            autoregressive=True,
        )
        self.target = WanDiffusionWrapper(
            config=self.config,
            checkpoint_path=student_init,
            autoregressive=True,
        )

    def attach_wrapped_models(
        self,
        *,
        generate: nn.Module | None = None,
        teacher: nn.Module | None = None,
        target: nn.Module | None = None,
    ) -> None:
        """Install trainer-wrapped (FSDP/AC) models back into this model."""
        self.generate.model = generate
        self.teacher.model = teacher
        if target is not None:
            self.target.model = target

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


class ConsistencyTrainingModel(ConsistencyBaseModel):
    """Build one teacher trajectory and compute the student consistency loss."""

    def _predict_consistency(
        self,
        output: VADiffusionOutput,
        noisy: VAPair,
        timesteps: VATimesteps,
    ) -> VAPair:
        sigma = sigmas_for_timesteps(
            self.train_scheduler_latent,
            timesteps.video,
            dtype=noisy.video.dtype,
        )
        sigma = broadcast_frame_values(sigma, noisy.video)
        sigma_data = noisy.video.new_tensor(self.sigma_data)
        denominator = sigma.square() + sigma_data.square()
        c_skip = sigma_data.square() / denominator
        c_out = sigma * sigma_data / denominator.sqrt()
        video = c_skip * noisy.video + c_out * output.x0.video
        return VAPair(
            video=video,
            action=output.x0.action,
        )

    def _teacher_cfg(
        self,
        input_dict: dict,
        batch: dict,
        empty_text_emb: torch.Tensor,
        noisy: VAPair,
        timesteps: VATimesteps,
    ) -> tuple[VAPair, float]:
        conditioned = self.teacher(input_dict, noisy, timesteps)
        text_emb = input_dict["latent_dict"]["text_emb"]
        empty_text_emb = self._empty_text_condition(
            batch,
            text_emb,
            empty_text_emb,
        )
        unconditioned = self.teacher(
            replace_text_condition(input_dict, empty_text_emb),
            noisy,
            timesteps,
        )
        cfg_scale = self.cfg_min + torch.rand(()).item() * (
            self.cfg_max - self.cfg_min
        )
        video = unconditioned.x0.video + cfg_scale * (
            conditioned.x0.video - unconditioned.x0.video
        )
        teacher_x0 = VAPair(video=video, action=conditioned.x0.action)
        return teacher_x0, cfg_scale

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
        shape = masks.video.shape
        video_timesteps, next_video_timesteps = sample_consistency_timesteps(
            self.train_scheduler_latent,
            self.video_num_steps,
            shape,
            self.device,
            masks.video,
        )
        action_mask = masks.action.any(dim=(1, 3, 4))
        action_timesteps, next_action_timesteps = sample_consistency_timesteps(
            self.train_scheduler_action,
            self.action_num_steps,
            shape,
            self.device,
            action_mask,
        )
        timesteps = VATimesteps(
            video=video_timesteps,
            action=action_timesteps,
        )
        next_timesteps = VATimesteps(
            video=next_video_timesteps,
            action=next_action_timesteps,
        )

        clean = VAPair(batch["latents"], batch["actions"])
        noise = randn_like_va(clean)

        noisy_video = add_noise(
            clean.video,
            noise.video,
            timesteps.video,
            self.train_scheduler_latent,
        )
        noisy_action = add_noise(
            clean.action,
            noise.action,
            timesteps.action,
            self.train_scheduler_action,
        )

        noisy = apply_va_mask(
            VAPair(video=noisy_video, action=noisy_action),
            clean,
            masks,
        )

        with torch.no_grad():
            teacher_input = replace_va_streams(
                base_input,
                noisy,
                clean,
                timesteps,
            )
            teacher_x0, cfg_scale = self._teacher_cfg(
                teacher_input,
                batch,
                empty_text_emb,
                noisy,
                timesteps,
            )
            transition_noise = (
                noise
                if self.reuse_teacher_noise
                else randn_like_va(clean)
            )
            next_noisy_video = add_noise(
                teacher_x0.video,
                transition_noise.video,
                next_timesteps.video,
                self.train_scheduler_latent,
            )
            next_noisy_action = add_noise(
                teacher_x0.action,
                transition_noise.action,
                next_timesteps.action,
                self.train_scheduler_action,
            )
            next_noisy = apply_va_mask(
                VAPair(video=next_noisy_video, action=next_noisy_action),
                clean,
                masks,
            )
            ema_input = replace_va_streams(
                base_input,
                next_noisy,
                clean,
                next_timesteps,
            )
            target_output = self.target(
                ema_input,
                next_noisy,
                next_timesteps,
            )
            target_consistency = self._predict_consistency(
                target_output,
                next_noisy,
                next_timesteps,
            )

        student_input = replace_va_streams(base_input, noisy, clean, timesteps)
        student_out = self.generate(student_input, noisy, timesteps)
        student_flow = student_out.velocity
        student_consistency = self._predict_consistency(
            student_out,
            noisy,
            timesteps,
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
        update_ema(self.target.model, student, decay=self.ema_decay)

    def state_dict(self) -> dict[str, Any]:
        return {"ema_decay": self.ema_decay}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.ema_decay = float(state.get("ema_decay", self.ema_decay))
