"""Self-gradient-forcing trajectory and DMD objectives."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from distillation.model.objectives import dmd_surrogate_loss, fake_score_flow_loss
from distillation.model.utils import temporary_masked_attention_backend
from distillation.model.wan_wrapper import WanDiffusionWrapper
from distillation.pipeline.utils import (
    add_noise_to_va,
    replace_text_condition,
    replace_va_streams,
)
from distillation.diffusion_utils import sample_interval_timesteps
from distillation.schema import (
    DenoisyInterval,
    ReplayContext,
    VADenoisySelection,
    VALossWeights,
    VAMasks,
    VAPrediction,
    VATimesteps,
)
from distillation.self_rollout import (
    SelfRolloutRecorder,
    build_sgf_rollout_schedule,
    self_rollout,
)


class SelfGradientForcingTrainingPipeline:
    """Stage3 final-clean SGF rollout followed by DMD-only optimization.

    Video and action use independent denoisy step lists and exit ids.  The
    no-grad rollout always completes every configured step and commits only its
    final clean x0.  Exit states are replayed once through the student; that
    replay x0 is the generator sample used by both the student DMD and
    fake-score regression paths.
    """

    def __init__(
        self,
        config: Any,
        trainer: Any,
        device: torch.device,
        student: nn.Module,
        real_score: nn.Module,
        fake_score: nn.Module,
    ) -> None:
        self.config = config
        self.trainer = trainer
        self.device = device
        self.student = student
        self.real_score = real_score
        self.fake_score = fake_score
        self.loss_weights = VALossWeights(
            video=float(config.video_loss_weight),
            action=float(config.action_loss_weight),
        )
        self.normalizer_eps = float(config.distill.dmd_normalizer_eps)
        if self.normalizer_eps <= 0.0:
            raise ValueError("distill.dmd_normalizer_eps must be positive")

        self.rollout_horizon_frames = int(config.distill.rollout_horizon_frames)
        if self.rollout_horizon_frames <= 0:
            raise ValueError("distill.rollout_horizon_frames must be positive")
        self.rollout_backend = str(
            getattr(config.distill, "rollout_masked_attn_backend", "dense")
        )

        self.teacher_cfg_min = float(config.distill.teacher_cfg_min)
        self.teacher_cfg_max = float(config.distill.teacher_cfg_max)
        if self.teacher_cfg_min < 0.0 or self.teacher_cfg_max < self.teacher_cfg_min:
            raise ValueError(
                "SGF real-score CFG requires 0 <= teacher_cfg_min <= "
                f"teacher_cfg_max, got {self.teacher_cfg_min} and "
                f"{self.teacher_cfg_max}"
            )

        self.sgf_schedule = build_sgf_rollout_schedule(
            config,
            video_scheduler=trainer.train_scheduler_latent,
            action_scheduler=trainer.train_scheduler_action,
        )
        self.student_wrapper = WanDiffusionWrapper(
            student,
            video_scheduler=trainer.train_scheduler_latent,
            action_scheduler=trainer.train_scheduler_action,
        )
        self.real_wrapper = WanDiffusionWrapper(
            real_score,
            video_scheduler=trainer.train_scheduler_latent,
            action_scheduler=trainer.train_scheduler_action,
        )
        self.fake_wrapper = WanDiffusionWrapper(
            fake_score,
            video_scheduler=trainer.train_scheduler_latent,
            action_scheduler=trainer.train_scheduler_action,
        )

    @staticmethod
    def _interval(steps: tuple[float, ...], exit_id: int) -> DenoisyInterval:
        denoisy_from = steps[exit_id]
        denoisy_to = steps[exit_id + 1] if exit_id + 1 < len(steps) else 0.0
        return DenoisyInterval(
            exit_id=int(exit_id),
            denoisy_from=float(denoisy_from),
            denoisy_to=float(denoisy_to),
        )

    def _sample_exit_ids(self) -> VADenoisySelection:
        video_exit_id = int(
            torch.randint(
                len(self.sgf_schedule.video_steps),
                (),
                device=self.device,
            ).item()
        )
        action_exit_id = int(
            torch.randint(
                len(self.sgf_schedule.action_steps),
                (),
                device=self.device,
            ).item()
        )
        return VADenoisySelection(
            video=self._interval(self.sgf_schedule.video_steps, video_exit_id),
            action=self._interval(self.sgf_schedule.action_steps, action_exit_id),
        )

    def _sample_dmd_timesteps(
        self,
        selection: VADenoisySelection,
        masks: VAMasks,
    ) -> VATimesteps:
        shape = tuple(masks.video.shape)
        return VATimesteps(
            video=sample_interval_timesteps(
                selection.video,
                shape,
                self.device,
                masks.video,
            ),
            action=sample_interval_timesteps(
                selection.action,
                shape,
                self.device,
                masks.action.any(dim=(1, 3, 4)),
            ),
        )

    def _add_dmd_noise(
        self,
        clean: VAPrediction,
        timesteps: VATimesteps,
        masks: VAMasks,
    ) -> tuple[VAPrediction, VAPrediction]:
        noise = VAPrediction(
            video=torch.randn_like(clean.video),
            action=torch.randn_like(clean.action),
        )
        noisy = add_noise_to_va(
            clean,
            noise,
            timesteps,
            masks,
            self.trainer.train_scheduler_latent,
            self.trainer.train_scheduler_action,
        )
        return noisy, noise

    def _prepare_input(
        self,
        batch: dict,
        noisy: VAPrediction,
        clean: VAPrediction,
        timesteps: VATimesteps,
    ) -> dict:
        input_dict = self.trainer._prepare_joint_input_dict(batch, add_noise=False)
        return replace_va_streams(input_dict, noisy, clean, timesteps)

    def _empty_text_condition(self, batch: dict, text_emb: torch.Tensor) -> torch.Tensor:
        empty = batch.get("empty_text_emb")
        if empty is None:
            empty = self.trainer._get_empty_text_emb()
        empty = empty.to(device=text_emb.device, dtype=text_emb.dtype)
        while empty.ndim < text_emb.ndim:
            empty = empty.unsqueeze(0)
        if empty.shape[0] == 1 and text_emb.shape[0] > 1:
            empty = empty.expand(text_emb.shape[0], *empty.shape[1:])
        if empty.shape != text_emb.shape:
            raise ValueError(
                f"empty text embedding shape {tuple(empty.shape)} does not match "
                f"text embedding {tuple(text_emb.shape)}"
            )
        return empty

    def _real_score_cfg_x0(
        self,
        input_dict: dict,
        batch: dict,
        noisy: VAPrediction,
        timesteps: VATimesteps,
    ) -> tuple[VAPrediction, float]:
        conditioned = self.real_wrapper.predict_velocity(input_dict)
        text_emb = input_dict["latent_dict"]["text_emb"]
        unconditioned = self.real_wrapper.predict_velocity(
            replace_text_condition(
                input_dict,
                self._empty_text_condition(batch, text_emb),
            )
        )
        cfg_scale = self.teacher_cfg_min + torch.rand(()).item() * (
            self.teacher_cfg_max - self.teacher_cfg_min
        )
        velocity = VAPrediction(
            video=unconditioned.video
            + cfg_scale * (conditioned.video - unconditioned.video),
            action=conditioned.action,
        )
        return (
            VAPrediction(
                video=self.real_wrapper.velocity_to_x0(
                    velocity.video,
                    noisy.video,
                    timesteps.video,
                    modality="video",
                ),
                action=self.real_wrapper.velocity_to_x0(
                    velocity.action,
                    noisy.action,
                    timesteps.action,
                    modality="action",
                ),
            ),
            cfg_scale,
        )

    @staticmethod
    def _masked_noisy(
        noisy: VAPrediction,
        clean: VAPrediction,
        masks: VAMasks,
    ) -> VAPrediction:
        return VAPrediction(
            video=torch.where(
                masks.video[:, None, :, None, None, None],
                noisy.video,
                clean.video,
            ),
            action=torch.where(masks.action, noisy.action, clean.action),
        )

    @staticmethod
    def _frame_timestep(
        timestep: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        timestep = torch.as_tensor(timestep, device=device, dtype=torch.float32)
        if timestep.numel() == 1:
            return timestep.reshape(1).expand(batch_size)
        if timestep.numel() == batch_size:
            return timestep.reshape(batch_size)
        raise ValueError(
            f"recorded timestep must be scalar or have {batch_size} values, got "
            f"{tuple(timestep.shape)}"
        )

    @staticmethod
    def _validate_record_interval(record, expected: DenoisyInterval, modality: str) -> None:
        observed = (
            int(record.exit_id),
            float(record.denoisy_from),
            float(record.denoisy_to),
        )
        wanted = (
            int(expected.exit_id),
            float(expected.denoisy_from),
            float(expected.denoisy_to),
        )
        if observed != wanted:
            raise RuntimeError(
                f"recorded {modality} denoisy interval {observed} does not match {wanted}"
            )

    def _build_replay_context(
        self,
        batch: dict,
        rollout,
        recorder: SelfRolloutRecorder,
        selection: VADenoisySelection,
        *,
        history_frames: int,
    ) -> ReplayContext:
        frames = int(batch["latents"].shape[2])
        generated_start = int(history_frames) + 1
        generated_end = generated_start + self.rollout_horizon_frames
        if generated_end > frames:
            raise ValueError(
                f"SGF rollout target [{generated_start},{generated_end}) exceeds {frames} frames"
            )
        generated_ids = list(range(generated_start, generated_end))
        recorder.require_frames(generated_ids)

        generated_frames = torch.zeros(
            (batch["latents"].shape[0], frames),
            dtype=torch.bool,
            device=batch["latents"].device,
        )
        generated_frames[:, generated_start:generated_end] = True
        video_mask = batch["video_latent_valid_mask"].bool() & generated_frames
        action_mask = batch["action_valid_mask"].bool() & generated_frames[
            :, None, :, None, None
        ]
        masks = VAMasks(video=video_mask, action=action_mask)

        replay_batch = dict(batch)
        replay_batch["latents"] = batch["latents"].clone()
        replay_batch["actions"] = batch["actions"].clone()
        replay_batch["latents"][:, :, generated_start:generated_end] = (
            rollout.pred_latents[:, :, generated_start:generated_end]
        )
        replay_batch["actions"][:, :, generated_start:generated_end] = (
            rollout.pred_actions[:, :, generated_start:generated_end]
        )
        replay_batch["video_latent_loss_mask"] = video_mask
        replay_batch["action_loss_mask"] = action_mask

        final_clean = VAPrediction(
            video=replay_batch["latents"].detach(),
            action=replay_batch["actions"].detach(),
        )
        noisy_video = final_clean.video.clone()
        noisy_action = final_clean.action.clone()
        video_timesteps = torch.zeros(
            (final_clean.video.shape[0], frames),
            dtype=torch.float32,
            device=final_clean.video.device,
        )
        action_timesteps = torch.zeros_like(video_timesteps)

        for frame_id in generated_ids:
            video_record = recorder.video[frame_id]
            action_record = recorder.action[frame_id]
            self._validate_record_interval(video_record, selection.video, "video")
            self._validate_record_interval(action_record, selection.action, "action")
            noisy_video[:, :, frame_id : frame_id + 1] = video_record.sample
            noisy_action[:, :, frame_id : frame_id + 1] = action_record.sample
            video_timesteps[:, frame_id] = self._frame_timestep(
                video_record.timestep,
                batch_size=final_clean.video.shape[0],
                device=final_clean.video.device,
            )
            action_timesteps[:, frame_id] = self._frame_timestep(
                action_record.timestep,
                batch_size=final_clean.video.shape[0],
                device=final_clean.video.device,
            )

        rollout_noisy = self._masked_noisy(
            VAPrediction(noisy_video, noisy_action),
            final_clean,
            masks,
        )
        video_timesteps = torch.where(video_mask, video_timesteps, 0)
        action_timesteps = torch.where(
            action_mask.any(dim=(1, 3, 4)),
            action_timesteps,
            0,
        )
        return ReplayContext(
            replay_batch=replay_batch,
            rollout_timesteps=VATimesteps(video_timesteps, action_timesteps),
            rollout_noisy=rollout_noisy,
            final_clean_context=final_clean,
            masks=masks,
            denoisy_selection=selection,
        )

    @torch.no_grad()
    def generate_and_record_context(self, batch: dict) -> ReplayContext:
        """Run a complete final-clean rollout and capture independent V/A exits."""

        from wan_va.mot_spec import mot_spec_from_config

        selection = self._sample_exit_ids()
        recorder = SelfRolloutRecorder(
            video=selection.video,
            action=selection.action,
        )
        spec = mot_spec_from_config(self.config)
        with temporary_masked_attention_backend(self.student, self.rollout_backend):
            rollout = self_rollout(
                batch,
                transformer=self.student,
                config=self.config,
                spec=spec,
                device=self.device,
                empty_text_emb=self.trainer._get_empty_text_emb(),
                rollout_frames=self.rollout_horizon_frames,
                transition_mode="sgf_renoise",
                sgf_schedule=self.sgf_schedule,
                diffusion_wrapper=self.student_wrapper,
                recorder=recorder,
            )
        return self._build_replay_context(
            batch,
            rollout,
            recorder,
            selection,
            history_frames=int(spec.history_latent_frames),
        )

    def _replay_student_x0(
        self,
        context: ReplayContext,
        *,
        requires_grad: bool,
    ) -> VAPrediction:
        replay_input = self._prepare_input(
            context.replay_batch,
            context.rollout_noisy,
            context.final_clean_context,
            context.rollout_timesteps,
        )
        grad_context = nullcontext() if requires_grad else torch.no_grad()
        with grad_context:
            estimated = self.student_wrapper.predict_joint(
                replay_input,
                context.rollout_noisy,
                context.rollout_timesteps,
            ).x0
        return VAPrediction(
            video=torch.where(
                context.masks.video[:, None, :, None, None, None],
                estimated.video,
                context.final_clean_context.video,
            ),
            action=torch.where(
                context.masks.action,
                estimated.action,
                context.final_clean_context.action,
            ),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.float()
        mask = mask.to(device=values.device, dtype=torch.bool)
        return torch.where(mask, values, 0).sum() / mask.sum().clamp_min(1)

    def _selection_metrics(
        self,
        reference: torch.Tensor,
        selection: VADenoisySelection,
        timesteps: VATimesteps,
        masks: VAMasks,
    ) -> dict[str, torch.Tensor]:
        value = reference.detach()
        return {
            "distill/sgf_video_exit_id": value.new_tensor(selection.video.exit_id),
            "distill/sgf_action_exit_id": value.new_tensor(selection.action.exit_id),
            "distill/sgf_video_denoisy_from": value.new_tensor(
                selection.video.denoisy_from
            ),
            "distill/sgf_video_denoisy_to": value.new_tensor(
                selection.video.denoisy_to
            ),
            "distill/sgf_action_denoisy_from": value.new_tensor(
                selection.action.denoisy_from
            ),
            "distill/sgf_action_denoisy_to": value.new_tensor(
                selection.action.denoisy_to
            ),
            "distill/dmd_video_t_mean": self._masked_mean(
                timesteps.video,
                masks.video,
            ).detach(),
            "distill/dmd_action_t_mean": self._masked_mean(
                timesteps.action,
                masks.action.any(dim=(1, 3, 4)),
            ).detach(),
        }

    def replay_and_score(
        self,
        context: ReplayContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Replay the student once and optimize only the DMD surrogate."""

        student_x0 = self._replay_student_x0(context, requires_grad=True)
        score_timesteps = self._sample_dmd_timesteps(
            context.denoisy_selection,
            context.masks,
        )
        score_noisy, _ = self._add_dmd_noise(
            student_x0,
            score_timesteps,
            context.masks,
        )
        score_input = self._prepare_input(
            context.replay_batch,
            score_noisy,
            student_x0,
            score_timesteps,
        )
        with torch.no_grad():
            real_x0, cfg_scale = self._real_score_cfg_x0(
                score_input,
                context.replay_batch,
                score_noisy,
                score_timesteps,
            )
            fake_x0 = self.fake_wrapper.predict_joint(
                score_input,
                score_noisy,
                score_timesteps,
            ).x0

        loss, metrics = dmd_surrogate_loss(
            student_x0,
            fake_x0,
            real_x0,
            context.masks,
            self.loss_weights,
            self.normalizer_eps,
        )
        metrics.update(
            self._selection_metrics(
                loss,
                context.denoisy_selection,
                score_timesteps,
                context.masks,
            )
        )
        metrics["distill/sgf_real_cfg_scale"] = loss.detach().new_tensor(cfg_scale)
        metrics["distill/sgf_total_loss"] = loss.detach()
        return loss, metrics

    def fake_score_step(
        self,
        context: ReplayContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Train fake-score on a detached no-grad exit replay sample."""

        generated_x0 = self._replay_student_x0(context, requires_grad=False)
        score_timesteps = self._sample_dmd_timesteps(
            context.denoisy_selection,
            context.masks,
        )
        score_noisy, noise = self._add_dmd_noise(
            generated_x0,
            score_timesteps,
            context.masks,
        )
        score_input = self._prepare_input(
            context.replay_batch,
            score_noisy,
            generated_x0,
            score_timesteps,
        )
        fake_velocity = self.fake_wrapper.predict_velocity(score_input)
        exact_target = VAPrediction(
            video=noise.video - generated_x0.video,
            action=noise.action - generated_x0.action,
        )
        loss, metrics = fake_score_flow_loss(
            fake_velocity,
            exact_target,
            context.masks,
            self.loss_weights,
        )
        metrics.update(
            self._selection_metrics(
                loss,
                context.denoisy_selection,
                score_timesteps,
                context.masks,
            )
        )
        return loss, metrics
