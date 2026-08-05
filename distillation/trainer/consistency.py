"""Consistency distillation trainer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from distillation.configs import CONSISTENCY_DISTILLATION
from distillation.model.utils import (
    temporary_fsdp_unshard,
    temporary_masked_attention_backend,
)
from distillation.trainer.base import DistillationTrainerBase, OptimizationTarget
from wan_va.mot_spec import mot_spec_from_config


class ConsistencyTrainer(DistillationTrainerBase):
    method = CONSISTENCY_DISTILLATION

    def __init__(self, config: Any):
        super().__init__(config)
        if self._resume_from is not None:
            self.checkpoint_io.load(self, self._resume_from)

    def _build_method_model(self, config: Any):
        from distillation.model.consistency import ConsistencyModel

        teacher_checkpoint = getattr(config.distill, "teacher_checkpoint", None)
        if teacher_checkpoint is None:
            raise ValueError(
                "consistency_distillation requires distill.teacher_checkpoint"
            )
        return ConsistencyModel(
            config=config,
            trainer=self,
            student=self.transformer,
            device=self.device,
            teacher_checkpoint=str(teacher_checkpoint),
        )

    def _after_optimizer_step(self, target: OptimizationTarget) -> None:
        super()._after_optimizer_step(target)
        self.method_model.after_student_step(self.transformer)

    @torch.no_grad()
    def _decode_rollout_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self._get_train_vae()
        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype
        batch_size, channels, frames, views, height, width = latents.shape
        per_view = latents.permute(0, 3, 1, 2, 4, 5).reshape(
            batch_size * views, channels, frames, height, width
        ).to(device=vae_device, dtype=vae_dtype)
        mean = torch.tensor(
            vae.config.latents_mean,
            device=vae_device,
            dtype=vae_dtype,
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            vae.config.latents_std,
            device=vae_device,
            dtype=vae_dtype,
        ).view(1, -1, 1, 1, 1)
        decoded = vae.decode(per_view * std + mean, return_dict=False)[0]
        decoded = (decoded.float() * 0.5 + 0.5).clamp(0, 1)
        decoded = decoded.permute(0, 2, 1, 3, 4).reshape(
            batch_size,
            views,
            decoded.shape[2],
            decoded.shape[1],
            decoded.shape[3],
            decoded.shape[4],
        ).permute(0, 2, 1, 3, 4, 5).contiguous()
        return decoded.to(device=latents.device, dtype=latents.dtype)

    @torch.no_grad()
    def _run_rollout(self, batch: dict, ground_truth_provider=None):
        from distillation.self_rollout import (
            resolve_ground_truth_provider,
            self_rollout,
        )

        gt_mode = str(getattr(self.config.distill, "rollout_gt_mode", "none"))
        ground_truth_provider = resolve_ground_truth_provider(
            gt_mode,
            batch,
            ground_truth_provider,
        )

        rollout_backend = getattr(self.config.distill, "rollout_masked_attn_backend", "dense")
        with temporary_masked_attention_backend(
            self.method_model.ema_student,
            rollout_backend,
        ), temporary_fsdp_unshard(self.method_model.ema_student):
            return self_rollout(
                batch,
                transformer=self.method_model.ema_student,
                config=self.config,
                spec=mot_spec_from_config(self.config),
                device=self.device,
                empty_text_emb=self._get_empty_text_emb(),
                decode_latents_to_rgb_views=self._decode_rollout_latents,
                video_num_steps=int(self.config.distill.rollout_video_num_steps),
                action_num_steps=int(self.config.distill.rollout_action_num_steps),
                rollout_frames=int(self.config.distill.rollout_horizon_frames),
                ground_truth_provider=ground_truth_provider,
                replacement_policy=str(
                    getattr(
                        self.config.distill,
                        "rollout_replacement_policy",
                        "require_ground_truth",
                    )
                ),
            )

    @torch.no_grad()
    def rollout(self, batch: dict, *, ground_truth_provider=None):
        """Run the configured EMA-student rollout for a caller-provided batch."""
        batch = self.convert_input_format(batch)
        batch = self._materialize_batch_latents(batch)
        return self._run_rollout(
            batch,
            ground_truth_provider=ground_truth_provider,
        )

    @torch.no_grad()
    def _maybe_run_training_rollout(self, batch: dict, completed_step: int) -> None:
        interval = int(getattr(self.config.distill, "rollout_interval", 0))
        from distillation.self_rollout import (
            rollout_artifact_due,
            save_rollout_artifacts,
        )

        if not rollout_artifact_due(completed_step, interval):
            return

        rollout = self._run_rollout(batch)
        if self.config.rank != 0:
            return
        output_dir = Path(self.config.save_root) / "rollouts" / f"step_{completed_step:08d}"
        eval_cfg = getattr(self.config, "eval_cfg", None)
        fps = int(getattr(eval_cfg, "inference_video_fps", 10)) if eval_cfg is not None else 10
        artifacts = save_rollout_artifacts(
            rollout,
            batch,
            decode_latents_to_rgb_views=self._decode_rollout_latents,
            output_dir=output_dir,
            video_fps=fps,
        )
        if bool(getattr(self.config, "enable_wandb", False)) and getattr(
            self,
            "wandb",
            None,
        ) is not None:
            self.wandb.log(
                {
                    "rollout/video_target_vs_generated": self.wandb.Video(
                        artifacts["video"],
                        fps=fps,
                        format="mp4",
                    ),
                    "rollout/actions": self.wandb.Image(artifacts["actions"]),
                },
                step=completed_step,
            )
