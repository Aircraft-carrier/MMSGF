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
from distillation.pipeline.utils import (
    add_noise_to_va,
    replace_va_streams,
)
from distillation.rollout import record_sgf_trajectory
from distillation.scheduler import (
    broadcast_frame_values,
    flow_to_x0,
    sigmas_for_timesteps,
)


class SelfGradientForcingTrainingPipeline:
    """Stage3 self-gradient-forcing + DMD 的逐步模拟。

    标准窗口 ``F=8`` 可写成：

    ``H0 H1 H2 H3 | T0 T1 T2 T3``

    ``H0`` 与 ``T0`` 是每个 chunk 的 clean anchor；真正 replay 的位置是
    ``H1..H3`` 和 ``T1..T3``。stage3 先用 student 在 no_grad 下完整生成两次
    V→G→A trajectory，并在随机 denoise step 记录 noisy/timestep；再选择以下
    互斥路径之一：

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
        """在 no_grad 下生成 self-forcing 数据，不使用真实 target 作 replay GT。

        输入 batch 仍是 dataset 的 clean 8-frame window。假设随机抽到
        ``video_record_step=7``、``action_record_step=19``：

        1. 以 H0 为 anchor，生成 H1..H3；记录 video 第 7 步、action 第 19 步
           的 noisy sample，同时继续 denoise 得到 final generated history；
        2. 把 generated H1..H3 及其 decoded/recomputed geometry 写回工作 batch；
        3. 以 T0 为第二个 anchor，在已经生成的 history 条件下生成 T1..T3，
           并在相同 modality record step 记录第二组 noisy sample；
        4. 返回 generated=[H0,pH1..pH3,T0,pT1..pT3]，mask 为
           [0,1,1,1,0,1,1,1]，anchor/padding 保持 clean。

        这一步不保留 student graph，控制显存；真正梯度在 replay 中重建。
        """
        from wan_va.mot_spec import mot_spec_from_config

        video_record_step = int(
            torch.randint(
                int(self.config.num_inference_steps),
                (1,),
                device=self.device,
            ).item()
        )
        action_record_step = int(
            torch.randint(
                int(self.config.action_num_inference_steps),
                (1,),
                device=self.device,
            ).item()
        )
        return record_sgf_trajectory(
            batch,
            transformer=self.student,
            config=self.config,
            spec=mot_spec_from_config(self.config),
            device=self.device,
            empty_text_emb=self.trainer._get_empty_text_emb(),
            decode_latents_to_rgb_views=self._decode_rollout_latents,
            video_record_step=video_record_step,
            action_record_step=action_record_step,
        )

    def replay_and_score(
        self,
        context: ReplayContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """执行 student optimizer 对应的一次 replay + DMD 更新。

        数据流可按下列顺序人工核对：

        ``recorded noisy/t_context``
        ``-> student_flow (唯一有梯度的大模型 forward)``
        ``-> replay_target=(noisy-generated)/sigma_context``
        ``-> student_x0=noisy-sigma_context*student_flow``
        ``-> 重新采样 score_t/noise``
        ``-> frozen real_x0 与 fake_x0``
        ``-> dmd_target``
        ``-> replay_loss + dmd_loss``

        real/fake 接收完全相同的 score_noisy 和条件，差值才具有可比较意义；
        两者 forward 均在 no_grad 中，DMD gradient 只回到 student replay graph。
        """
        noisy = context.noisy
        generated = context.generated
        replay_input = self._prepare_input(
            context.batch,
            noisy,
            generated,
            context.timesteps,
        )
        # 1) recorded_batch 包含 generated V/A/G；noisy 与 timestep 来自 rollout
        #    中实际记录的 sampler state，而不是重新随机合成。
        student_out = self.student(replay_input, mode="train")
        student_flow = VAPrediction(
            student_out["latent_pred"],
            student_out["action_pred"],
        )
        target_flow = self._flow_targets(noisy, generated, context.timesteps)
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
                context.timesteps.video,
                self.trainer.train_scheduler_latent,
            ),
            flow_to_x0(
                student_flow.action,
                noisy.action,
                context.timesteps.action,
                self.trainer.train_scheduler_action,
            ),
        )
        # ``context.timesteps`` uses numeric 0 for anchors/padding, but the MOT
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
            context.batch,
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
            "distill/sgf_total_loss": total.detach(),
        }

    def fake_score_step(
        self,
        context: ReplayContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """执行 fake-score optimizer 对应的一次精确 flow regression。

        ``context.generated`` 已 detach。重新采样 ``score_t`` 和 Gaussian noise
        后，noisy/clean/sigma 都已知，因此 target flow 可直接反解，不需要
        real-score forward。只有 ``fake_score(...)`` 位于 autograd graph；
        student 只在前面的 trajectory record 中以 no_grad 方式出现。
        """
        generated = context.generated
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
            context.batch,
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
