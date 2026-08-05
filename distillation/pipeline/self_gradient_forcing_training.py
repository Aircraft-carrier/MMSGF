"""Self-gradient-forcing trajectory."""
from typing import Any

import torch
import torch.nn as nn

from distillation.schema import (
    ReplayContext,
    VALossWeights,
    VAMasks,
    VAPrediction,
    VATimesteps,
)
from distillation.model.objectives import (
    dmd_surrogate_loss,
    fake_score_flow_loss,
    replay_target_loss,
)
from distillation.model.utils import (
    temporary_fsdp_unshard,
    temporary_masked_attention_backend,
)
from distillation.pipeline.utils import (
    add_noise_to_va,
    replace_text_condition,
    replace_va_streams,
)
from distillation.self_rollout import SelfRolloutRecorder, self_rollout
from distillation.scheduler import (
    broadcast_frame_values,
    flow_to_x0,
    sigmas_for_timesteps,
)


class SelfGradientForcingTrainingPipeline:
    """Stage3 self-gradient-forcing + DMD 的逐步模拟。

    标准窗口 ``F=8`` 可写成：

    ``H0 H1 H2 H3 | T0 T1 T2 T3``

    ``H0..H3`` 是 GT history，``T0`` 是 GT target anchor；真正 replay 的位置
    只有 ``T1..T3``。stage3 先用 student 在 no_grad 下执行一次增量
    ``self_rollout``，逐帧生成 V→G→A，并在随机 denoise step 记录真实
    noisy/timestep；再选择以下互斥路径之一：

    - student update：一次有梯度 replay + frozen real/fake score + DMD loss；
    - fake-score update：对 detached generated sample 人工加噪，回归精确 flow。

    ReplayContext 中 video/action 分开保存 timestep，因为 video 例如 25 个
    inference steps、action 例如 50 个 steps，同一个 record_step 索引并不代表
    相同 sigma。所有 score loss 仍使用各自 scheduler 做 sigma lookup。
    """
    def __init__(
        self,
        config: Any,
        trainer: Any,
        device: torch.device,
        student: nn.Module,
        real_score: nn.Module,
        fake_score: nn.Module,
    ):
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
        self.flow_target_eps = float(config.distill.flow_target_eps)
        self.score_t_min = int(config.distill.score_timestep_min)
        self.score_t_max = min(
            int(config.distill.score_timestep_max),
            trainer.train_scheduler_latent.num_train_timesteps,
        )
        if self.score_t_min >= self.score_t_max:
            raise ValueError("score timestep range is empty")
        self.rollout_video_num_steps = int(config.distill.rollout_video_num_steps)
        self.rollout_action_num_steps = int(config.distill.rollout_action_num_steps)
        self.rollout_horizon_frames = int(config.distill.rollout_horizon_frames)
        if min(
            self.rollout_video_num_steps,
            self.rollout_action_num_steps,
            self.rollout_horizon_frames,
        ) <= 0:
            raise ValueError("SGF rollout steps and horizon must be positive")
        self.teacher_cfg_min = float(config.distill.teacher_cfg_min)
        self.teacher_cfg_max = float(config.distill.teacher_cfg_max)
        if self.teacher_cfg_min < 0.0 or self.teacher_cfg_max < self.teacher_cfg_min:
            raise ValueError(
                "SGF teacher CFG requires 0 <= teacher_cfg_min <= teacher_cfg_max, got "
                f"{self.teacher_cfg_min} and {self.teacher_cfg_max}"
            )

    def _sample_timesteps(
        self,
        values: VAPrediction,
        masks: VAMasks,
    ) -> torch.Tensor:
        timesteps = torch.randint(
            self.score_t_min,
            self.score_t_max,
            (values.video.shape[0], values.video.shape[2]),
            device=self.device,
        )
        return torch.where(masks.frame_mask(), timesteps, 0)

    def _add_noise(
        self,
        clean: VAPrediction,
        timesteps: torch.Tensor,
        masks: VAMasks,
    ) -> VAPrediction:
        noise = VAPrediction(
            torch.randn_like(clean.video),
            torch.randn_like(clean.action),
        )
        return add_noise_to_va(
            clean,
            noise,
            VATimesteps(timesteps, timesteps),
            masks,
            self.trainer.train_scheduler_latent,
            self.trainer.train_scheduler_action,
        )

    def _predict_x0(
        self,
        model: nn.Module,
        input_dict: dict,
        noisy: VAPrediction,
        timesteps: VATimesteps,
    ) -> VAPrediction:
        out = model(input_dict, mode="train")
        return VAPrediction(
            flow_to_x0(
                out["latent_pred"],
                noisy.video,
                timesteps.video,
                self.trainer.train_scheduler_latent,
            ),
            flow_to_x0(
                out["action_pred"],
                noisy.action,
                timesteps.action,
                self.trainer.train_scheduler_action,
            ),
        )

    def _prepare_input(
        self,
        batch: dict,
        noisy: VAPrediction,
        clean: VAPrediction,
        timesteps: VATimesteps,
    ) -> dict:
        input_dict = self.trainer._prepare_joint_input_dict(batch, add_noise=False)
        return replace_va_streams(
            input_dict,
            noisy,
            clean,
            timesteps,
        )

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

    def _teacher_cfg_flow(
        self,
        input_dict: dict,
        batch: dict,
    ) -> tuple[VAPrediction, float]:
        conditioned = self.real_score(input_dict, mode="train")
        text_emb = input_dict["latent_dict"]["text_emb"]
        unconditioned = self.real_score(
            replace_text_condition(
                input_dict,
                self._empty_text_condition(batch, text_emb),
            ),
            mode="train",
        )
        cfg_scale = self.teacher_cfg_min + torch.rand(()).item() * (
            self.teacher_cfg_max - self.teacher_cfg_min
        )
        video = unconditioned["latent_pred"] + cfg_scale * (
            conditioned["latent_pred"] - unconditioned["latent_pred"]
        )
        return VAPrediction(video, conditioned["action_pred"]), cfg_scale

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

    def _build_replay_context(
        self,
        batch: dict,
        rollout,
        recorder: SelfRolloutRecorder,
        *,
        history_frames: int,
    ) -> ReplayContext:
        frames = int(batch["latents"].shape[2])
        anchor = int(history_frames)
        generated_start = anchor + 1
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
        video_mask = (
            batch["video_latent_valid_mask"].to(dtype=torch.bool)
            & generated_frames
        )
        action_mask = (
            batch["action_valid_mask"].to(dtype=torch.bool)
            & generated_frames[:, None, :, None, None]
        )

        student_batch = dict(batch)
        for key in ("latents", "actions", "geometry_rgb"):
            student_batch[key] = batch[key].clone()
        student_batch["latents"][:, :, generated_start:generated_end] = (
            rollout.pred_latents[:, :, generated_start:generated_end]
        )
        student_batch["actions"][:, :, generated_start:generated_end] = (
            rollout.pred_actions[:, :, generated_start:generated_end]
        )
        student_batch["geometry_rgb"][:, generated_start:generated_end] = (
            rollout.pred_geometry_rgb[:, generated_start:generated_end]
        )
        student_batch["video_latent_loss_mask"] = video_mask
        student_batch["action_loss_mask"] = action_mask

        teacher_batch = dict(batch)
        teacher_batch["video_latent_loss_mask"] = video_mask
        teacher_batch["action_loss_mask"] = action_mask

        pred_clean = VAPrediction(
            student_batch["latents"].detach(),
            student_batch["actions"].detach(),
        )
        teacher_clean = VAPrediction(
            batch["latents"].detach(),
            batch["actions"].detach(),
        )
        noisy_video = pred_clean.video.clone()
        noisy_action = pred_clean.action.clone()
        video_timesteps = torch.zeros(
            (pred_clean.video.shape[0], frames),
            dtype=torch.float32,
            device=pred_clean.video.device,
        )
        action_timesteps = torch.zeros_like(video_timesteps)
        for frame_id in generated_ids:
            video_record = recorder.video[frame_id]
            action_record = recorder.action[frame_id]
            noisy_video[:, :, frame_id : frame_id + 1] = video_record.sample
            noisy_action[:, :, frame_id : frame_id + 1] = action_record.sample
            video_timesteps[:, frame_id] = self._frame_timestep(
                video_record.timestep,
                batch_size=pred_clean.video.shape[0],
                device=pred_clean.video.device,
            )
            action_timesteps[:, frame_id] = self._frame_timestep(
                action_record.timestep,
                batch_size=pred_clean.video.shape[0],
                device=pred_clean.video.device,
            )

        masks = VAMasks(video_mask, action_mask)
        rollout_noisy = self._masked_noisy(
            VAPrediction(noisy_video, noisy_action),
            pred_clean,
            masks,
        )
        video_timesteps = torch.where(video_mask, video_timesteps, 0)
        action_timesteps = torch.where(
            action_mask.any(dim=(1, 3, 4)),
            action_timesteps,
            0,
        )
        return ReplayContext(
            student_batch=student_batch,
            teacher_batch=teacher_batch,
            rollout_timesteps=VATimesteps(video_timesteps, action_timesteps),
            rollout_noisy=rollout_noisy,
            pred_clean=pred_clean,
            teacher_clean=teacher_clean,
            masks=masks,
        )

    def _flow_targets(
        self,
        noisy: VAPrediction,
        clean: VAPrediction,
        timesteps: VATimesteps,
    ) -> VAPrediction:
        """从已知 noisy/clean pair 精确反解 flow-matching target。

        由 ``x_t=(1-sigma)*x0+sigma*noise`` 可得
        ``(x_t-x0)/sigma=noise-x0``。例如 ``x0=2, noise=6, sigma=.75``，
        ``x_t=5``，反解得到 ``(5-2)/.75=4``。condition/padding 位置不参与
        loss；若 sigma 数值为 0，则明确返回 0，避免除零产生 NaN。
        """
        video_sigma = broadcast_frame_values(
            sigmas_for_timesteps(
                self.trainer.train_scheduler_latent,
                timesteps.video,
                dtype=noisy.video.dtype,
            ),
            noisy.video,
        )
        action_sigma = broadcast_frame_values(
            sigmas_for_timesteps(
                self.trainer.train_scheduler_action,
                timesteps.action,
                dtype=noisy.action.dtype,
            ),
            noisy.action,
        )

        def target(noisy_value, clean_value, sigma):
            return torch.where(
                sigma > 0,
                (noisy_value - clean_value) / sigma.clamp_min(self.flow_target_eps),
                torch.zeros_like(noisy_value),
            )

        return VAPrediction(
            target(noisy.video, clean.video, video_sigma),
            target(noisy.action, clean.action, action_sigma),
        )

    @torch.no_grad()
    def _decode_rollout_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self.trainer._get_train_vae()
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
    def generate_and_record_context(self, batch: dict) -> ReplayContext:
        """Generate only T1..T3 and record real intermediate V/A sampler states."""
        from wan_va.mot_spec import mot_spec_from_config

        video_record_step = int(
            torch.randint(
                self.rollout_video_num_steps,
                (1,),
                device=self.device,
            ).item()
        )
        action_record_step = int(
            torch.randint(
                self.rollout_action_num_steps,
                (1,),
                device=self.device,
            ).item()
        )
        recorder = SelfRolloutRecorder(
            video_step=video_record_step,
            action_step=action_record_step,
        )
        spec = mot_spec_from_config(self.config)
        rollout_backend = getattr(self.config.distill, "rollout_masked_attn_backend", "dense")
        with temporary_masked_attention_backend(
            self.student,
            rollout_backend,
        ), temporary_fsdp_unshard(self.student):
            rollout = self_rollout(
                batch,
                transformer=self.student,
                config=self.config,
                spec=spec,
                device=self.device,
                empty_text_emb=self.trainer._get_empty_text_emb(),
                decode_latents_to_rgb_views=self._decode_rollout_latents,
                video_num_steps=self.rollout_video_num_steps,
                action_num_steps=self.rollout_action_num_steps,
                rollout_frames=self.rollout_horizon_frames,
                recorder=recorder,
            )
        return self._build_replay_context(
            batch,
            rollout,
            recorder,
            history_frames=int(spec.history_latent_frames),
        )

    def replay_and_score(
        self,
        context: ReplayContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """执行 student optimizer 对应的一次 replay + DMD 更新。

        数据流可按下列顺序人工核对：

        ``recorded noisy/t_context``
        ``-> student_flow (唯一有梯度的大模型 forward)``
        ``-> GT-clean teacher CFG flow at the same rollout state``
        ``-> student_x0=noisy-sigma_context*student_flow``
        ``-> 重新采样 score_t/noise``
        ``-> frozen real_x0 与 fake_x0``
        ``-> dmd_target``
        ``-> replay_loss + dmd_loss``

        real/fake 接收完全相同的 score_noisy 和条件，差值才具有可比较意义；
        两者 forward 均在 no_grad 中，DMD gradient 只回到 student replay graph。
        """
        noisy = context.rollout_noisy
        generated = context.pred_clean
        replay_input = self._prepare_input(
            context.student_batch,
            noisy,
            generated,
            context.rollout_timesteps,
        )
        # 1) Student clean/geometry 使用 rollout prediction；noisy 与 timestep
        #    来自同一次 rollout 的真实 sampler state，而不是重新随机合成。
        student_out = self.student(replay_input, mode="train")
        student_flow = VAPrediction(
            student_out["latent_pred"],
            student_out["action_pred"],
        )
        teacher_noisy = self._masked_noisy(
            noisy,
            context.teacher_clean,
            context.masks,
        )
        teacher_input = self._prepare_input(
            context.teacher_batch,
            teacher_noisy,
            context.teacher_clean,
            context.rollout_timesteps,
        )
        with torch.no_grad():
            target_flow, cfg_scale = self._teacher_cfg_flow(
                teacher_input,
                context.teacher_batch,
            )
        replay_loss, replay_metrics = replay_target_loss(
            student_flow,
            target_flow,
            context.masks,
            self.loss_weights,
        )

        # 2) 把 student replay flow 转成可微 x0。masked target 部分保留 graph；
        #    condition 部分随后用 generated clean 覆盖，避免 scheduler 最小 sigma
        #    非零时把 anchor 悄悄改动。
        estimated_x0 = VAPrediction(
            flow_to_x0(
                student_flow.video,
                noisy.video,
                context.rollout_timesteps.video,
                self.trainer.train_scheduler_latent,
            ),
            flow_to_x0(
                student_flow.action,
                noisy.action,
                context.rollout_timesteps.action,
                self.trainer.train_scheduler_action,
            ),
        )
        # ``rollout_timesteps`` uses numeric 0 for anchors/padding, but the MOT
        # training scheduler's last sigma is small rather than exactly zero.
        # Applying ``x0 = noisy - sigma * flow`` to those locations would subtly
        # modify H0/T0 and invalid action tokens.  They become conditioning for the
        # following score-model pass, so preserve the recorded generated values
        # exactly outside the replay mask.  Only supervised positions retain the
        # differentiable replay estimate and therefore the student gradient path.
        student_x0 = VAPrediction(
            video=torch.where(
                context.masks.video[:, None, :, None, None, None],
                estimated_x0.video,
                generated.video,
            ),
            action=torch.where(
                context.masks.action,
                estimated_x0.action,
                generated.action,
            ),
        )
        score_t = self._sample_timesteps(
            student_x0,
            context.masks,
        )
        # 3) score timestep 是新的随机变量，与 rollout record timestep 无关。
        #    video/action 使用同一 nominal [B,F] t，但各自 scheduler shift 可把它
        #    映射成不同 sigma；add_noise_to_va 仍逐 modality 计算。
        score_noisy = self._add_noise(
            student_x0,
            score_t,
            context.masks,
        )
        score_input = self._prepare_input(
            context.student_batch,
            score_noisy,
            student_x0,
            VATimesteps(score_t, score_t),
        )
        with torch.no_grad():
            # 4) Real/fake score models 只产生 stop-gradient clean estimates。
            #    student update 不应该在这两个模型上留下 grad 或 optimizer state。
            real_x0 = self._predict_x0(
                self.real_score,
                score_input,
                score_noisy,
                VATimesteps(score_t, score_t),
            )
            fake_x0 = self._predict_x0(
                self.fake_score,
                score_input,
                score_noisy,
                VATimesteps(score_t, score_t),
            )

        dmd_loss, dmd_metrics = dmd_surrogate_loss(
            student_x0,
            fake_x0,
            real_x0,
            context.masks,
            self.loss_weights,
            self.normalizer_eps,
        )
        # 5) replay 保持 student 能复现自己的多步 rollout；DMD 再把该分布朝
        #    real-score、远离 fake-score 的方向移动。当前实现使用等权相加。
        total = replay_loss + dmd_loss
        return total, {
            **replay_metrics,
            **dmd_metrics,
            "distill/sgf_teacher_cfg_scale": total.detach().new_tensor(cfg_scale),
            "distill/sgf_total_loss": total.detach(),
        }

    def fake_score_step(
        self,
        context: ReplayContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """执行 fake-score optimizer 对应的一次精确 flow regression。

        ``context.pred_clean`` 已 detach。重新采样 ``score_t`` 和 Gaussian noise
        后，noisy/clean/sigma 都已知，因此 target flow 可直接反解，不需要
        real-score forward。只有 ``fake_score(...)`` 位于 autograd graph；
        student 只在前面的 trajectory record 中以 no_grad 方式出现。
        """
        generated = context.pred_clean
        timesteps = self._sample_timesteps(
            generated,
            context.masks,
        )
        noisy = self._add_noise(
            generated,
            timesteps,
            context.masks,
        )
        score_input = self._prepare_input(
            context.student_batch,
            noisy,
            generated,
            VATimesteps(timesteps, timesteps),
        )
        fake_out = self.fake_score(score_input, mode="train")
        target_flow = self._flow_targets(
            noisy,
            generated,
            VATimesteps(timesteps, timesteps),
        )
        return fake_score_flow_loss(
            VAPrediction(fake_out["latent_pred"], fake_out["action_pred"]),
            target_flow,
            context.masks,
            self.loss_weights,
        )
