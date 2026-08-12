"""Self-gradient-forcing generation, replay, and DMD optimization."""
from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any, Literal

import torch
import torch.nn as nn

from distillation.model.utils import (
    apply_va_mask,
    randn_like_va,
    replace_text_condition,
    replace_va_streams,
    va_loss,
)
from distillation.model.wan_wrapper import (
    WanDiffusionWrapper,
)
from distillation.pipeline import SelfGradientForcingTrainingPipeline
from distillation.schema import (
    DMDUpdateSchedule,
    DenoisyInterval,
    ReplayContext,
    TrainingStepResult,
    VADenoisySelection,
    VALossWeights,
    VAMasks,
    VAPair,
    VATimesteps,
)
from wan_va.utils.scheduler import FlowMatchScheduler


def dmd_surrogate_loss(
    generator_x0: VAPair,
    target_x0: VAPair,
    masks: VAMasks,
    weights: VALossWeights = VALossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute ``0.5 *`` masked weighted V/A MSE for the DMD surrogate.

    The factor of one half makes the surrogate derivative equal the normalized
    KL direction instead of twice that direction.
    """
    loss, metrics = va_loss(generator_x0, target_x0, masks, weights, "dmd")
    return 0.5 * loss, {name: 0.5 * value for name, value in metrics.items()}


def fake_score_flow_loss(
    fake_flow: VAPair,
    target_flow: VAPair,
    masks: VAMasks,
    weights: VALossWeights = VALossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress fake-score V/A flow against the exact generated-data flow."""
    return va_loss(
        fake_flow,
        target_flow,
        masks,
        weights,
        "fake_score",
    )


@torch.no_grad()
def update_ema(
    ema_model: nn.Module,
    student_model: nn.Module,
    decay: float,
) -> None:
    for ema_param, student_param in zip(
        ema_model.parameters(),
        student_model.parameters(),
        strict=True,
    ):
        ema_param.mul_(decay).add_(student_param, alpha=1.0 - decay)


def warp_denoisy_progress(
    progress: torch.Tensor,
    scheduler: FlowMatchScheduler,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Map linear denoising progress ``d`` to the scheduler's network timestep.

    ``d`` uses the reference SGF coordinate ``[0,T]`` with ``d=T`` at the
    noisiest end and ``d=0`` at the clean end.
    """
    total = int(scheduler.num_train_timesteps)
    shift = float(scheduler.shift)
    progress_f = torch.as_tensor(
        progress,
        device=progress.device,
        dtype=torch.float32,
    )

    # Stage 1/3: Linear progress d -> linear sigma = d / T.
    #
    # Example (T=1000):
    #   d=1000 -> sigma=1.0
    #   d=250  -> sigma=0.25
    #   d=0    -> sigma=0.0
    sigma = progress_f / float(total)

    # Stage 2/3: Apply the FlowMatchScheduler rational shift.
    #
    #                    shift * sigma
    #   warped_sigma = ---------------------
    #                   1 + (shift-1)*sigma
    #
    # Example (shift=5, sigma=0.25): 1.25 / 2 = 0.625.
    warped_sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)

    # Stage 3/3: Warped sigma back to the network timestep t = sigma * T.
    #
    # Example: 0.625 * 1000 = 625.
    warped_timestep = warped_sigma * float(total)

    # Pin the two exact endpoints to avoid floating-point drift:
    #   d=0 -> t=0, d=T -> t=T.
    warped_timestep = torch.where(
        progress_f == 0,
        torch.zeros_like(warped_timestep),
        warped_timestep,
    )
    warped_timestep = torch.where(
        progress_f == total,
        torch.full_like(warped_timestep, float(total)),
        warped_timestep,
    )
    network_timestep = warped_timestep.to(dtype=dtype)
    return network_timestep


class BaseModel:
    """Own the stage3 model roles and shared timestep/noise operations."""

    def __init__(
        self,
        config: Any,
        device: torch.device,
        student_init: str | None,
        real_score_checkpoint: str | None,
        fake_score_init: str | None,
    ) -> None:
        self.config = config
        self.device = device
        self._initialize_models(
            student_init=student_init,
            real_score_checkpoint=real_score_checkpoint,
            fake_score_init=fake_score_init,
        )
        
        self.loss_weights = VALossWeights(
            video=float(config.video_loss_weight),
            action=float(config.action_loss_weight),
        )
        self.normalizer_eps = float(config.distill.dmd_normalizer_eps)

        # DMD first samples in the linear denoising-progress coordinate d, then
        # warps d through each modality's training scheduler.  Clamp limits are
        # expressed in the resulting network-timestep coordinate t.
        total_timesteps = int(self.generator.video_scheduler.num_train_timesteps)
        self.num_train_timestep = total_timesteps
        self.ts_schedule = bool(config.distill.ts_schedule)
        self.ts_schedule_max = bool(config.distill.ts_schedule_max)
        self.min_score_timestep = float(config.distill.min_score_timestep)
        self.min_step = float(config.distill.dmd_timestep_min)
        self.max_step = float(config.distill.dmd_timestep_max)

        self.teacher_cfg_min = float(config.distill.teacher_cfg_min)
        self.teacher_cfg_max = float(config.distill.teacher_cfg_max)

        self.video_denoising_step_list = tuple(
            int(value) for value in config.distill.denoisy_step_list.video
        )
        self.action_denoising_step_list = tuple(
            int(value) for value in config.distill.denoisy_step_list.action
        )

    def _initialize_models(
        self,
        *,
        student_init: str | None,
        real_score_checkpoint: str | None,
        fake_score_init: str | None,
    ) -> None:
        """Load the generator/real/fake wrappers from checkpoint paths only."""
        if student_init is None:
            raise ValueError("student_init path is required")
        # 自回归学生：负责 SGF rollout 和最终生成
        self.generator = WanDiffusionWrapper(
            config=self.config,
            checkpoint_path=student_init,
            autoregressive=True,
        )
        # 冻结的双向教师：近似真实数据分布
        self.real_score = WanDiffusionWrapper(
            config=self.config,
            checkpoint_path=real_score_checkpoint,
            autoregressive=False,
        )
        # 可训练的双向 score：近似 generator 当前分布
        self.fake_score = WanDiffusionWrapper(
            config=self.config,
            checkpoint_path=fake_score_init,
            autoregressive=False,
        )

    def attach_wrapped_models(
        self,
        *,
        generator: nn.Module | None = None,
        real_score: nn.Module | None = None,
        fake_score: nn.Module | None = None,
    ) -> None:
        """Install trainer-wrapped (FSDP/AC) models back into the wrappers."""
        # Trainer 完成 FSDP、activation checkpoint 等包装后，
        # 把包装后的 nn.Module 重新放回 wrapper。
        if generator is not None:
            self.generator.model = generator
        if real_score is not None:
            self.real_score.model = real_score
        if fake_score is not None:
            self.fake_score.model = fake_score

    def _prepare_input(
        self,
        noisy: VAPair,
        clean: VAPair,
        timesteps: VATimesteps,
        base_input: dict,
    ) -> dict:
        """Replace the clean V/A streams in base_input with noisy/clean/timesteps.

        base_input(trainer 准备好,model 只消费):
        {
            "latent_dict": {
                "latent":                  [B,Cv,F,V,H,W],  # GT clean video latents
                "text_emb":                [B,L,D],         # 文本条件嵌入(L=512 pad)
                "video_latent_loss_mask":  [B,F],           # 训练监督 mask(帧级)
                "video_latent_valid_mask": [B,F],           # 有效帧 mask(attention token 布局)
            },
            "action_dict": {
                "latent":             [B,Ca,F,N,1],         # GT clean action
                "text_emb":           [B,L,D],              # 文本条件嵌入
                "action_loss_mask":   [B,Ca,F,N,1],         # 训练监督 mask(token 级)
                "action_valid_mask":  [B,Ca,F,N,1],         # 有效 token mask
            },
            "stream_ids":  [B,V],                           # 视角流标识(0 左手腕/1 头/2 右手腕)
            "chunk_size":  int,                             # 每个 chunk 的 latent 帧数
            "window_size": int,                             # 注意力窗口大小
        }

        prepared_input(模型 forward 实际消费):
        prepared_input = {
            **base_input,
            "latent_dict": {
                "noisy_latents":  [B,Cv,F,V,H,W],  # 加噪后的 video latent
                "latent":         [B,Cv,F,V,H,W],  # clean video latent(GT 或预测 x0)
                "targets":        [B,Cv,F,V,H,W],  # 形状占位,loss 在模型外计算
                "timesteps":      [B,F],           # video 网络 timestep
                "cond_timesteps": [B,F],           # clean 条件帧 timestep(恒 0)
            },
            "action_dict": {
                "noisy_latents":  [B,Ca,F,N,1],    # 加噪后的 action
                "latent":         [B,Ca,F,N,1],    # clean action
                "targets":        [B,Ca,F,N,1],    # 形状占位
                "timesteps":      [B,F],           # action 网络 timestep
                "cond_timesteps": [B,F],           # clean 条件帧 timestep(恒 0)
            },
        }
        """
        return replace_va_streams(base_input, noisy, clean, timesteps)

class SelfGradientForcingModel(BaseModel):
    """Run the SGF generator, record exit states, and replay the generator."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rollout_horizon_frames = int(
            self.config.distill.rollout_horizon_frames
        )
        self.pipeline = SelfGradientForcingTrainingPipeline(
            denoising_step_list=self.config.distill.denoisy_step_list,
            generator=self.generator,
            num_frame_per_block=1,
            per_rank_exit_step=True,
        )

    def record(
        self,
        batch: dict,
        rollout,
    ) -> ReplayContext:
        """Build the detached boundary between rollout and gradient replay.
        VADenoisySelection = {
            "video" = {
                "exit_id"                              # video 退出采样的 id = 2 
                "denoisy_from"                         # video 线性去噪区间起点 500 ([1000, 750, 500, 250] )
                "denoisy_to"                           # video 线性去噪区间终点 250 
            },
            "action" = {
                "exit_id"
                "denoisy_from"
                "denoisy_to"
            }
        }
        ReplayContext = {
            "exit_timesteps":    VATimesteps,         # exit 时刻网络 timestep,video/action 各 [B,F]
            "noisy_at_t":        VAPair,              # exit 时刻加噪输入(video [B,Cv,F,V,H,W],action [B,Ca,F,N,1])
            "clean_hat":         VAPair,              # 全轨迹 clean(生成区间=rollout x0,其余=GT)
            "masks":             VAMasks,             # video [B,F],action [B,Ca,F,N,1],只覆盖生成区间
            "denoisy_selection": VADenoisySelection,  # 各模态 exit interval(线性 d 坐标)
        }
        """
        frames = int(batch["latents"].shape[2])
        # 阶段 1/4：确定生成区间
        generated_ids = list(rollout.pred_frame_ids)
        generated_start = generated_ids[0]
        generated_end = generated_ids[-1] + 1
        # 保存 video/action 各自的退出位置和线性去噪区间
        selection = VADenoisySelection(
            video=DenoisyInterval(
                exit_id=rollout.video_exit_id,
                denoisy_from=rollout.video_denoised_timestep_from,
                denoisy_to=rollout.video_denoised_timestep_to,
            ),
            action=DenoisyInterval(
                exit_id=rollout.action_exit_id,
                denoisy_from=rollout.action_denoised_timestep_from,
                denoisy_to=rollout.action_denoised_timestep_to,
            ),
        )
        
        # 阶段 2/4：只监督“有效且属于生成区间”的位置
        # generated_frames: [B,F]
        generated_frames = torch.zeros(
            (batch["latents"].shape[0], frames),
            dtype=torch.bool,
            device=batch["latents"].device,
        )
        generated_frames[:, generated_start:generated_end] = True
        # video_mask:  [B,F]
        # action_mask: [B,Ca,F,N,1]
        video_mask = batch["video_latent_valid_mask"].bool() & generated_frames
        action_mask = batch["action_valid_mask"].bool() & generated_frames[
            :, None, :, None, None
        ]
        masks = VAMasks(video=video_mask, action=action_mask)
        
        # 阶段 3/4：构造最终 clean 上下文
        # 历史/条件区间保留 GT，生成区间换成 rollout 最终预测。
        final_clean_video = batch["latents"].clone()
        final_clean_action = batch["actions"].clone()
        final_clean_video[:, :, generated_start:generated_end] = (
            rollout.predicted_clean.video
        )
        final_clean_action[:, :, generated_start:generated_end] = (
            rollout.predicted_clean.action
        )

        clean_hat = VAPair(
            video=final_clean_video.detach(),
            action=final_clean_action.detach(),
        )
        
        # 阶段 4/4：构造各模态退出时的 noisy 状态
        noisy_video = clean_hat.video.clone()
        noisy_action = clean_hat.action.clone()
        noisy_video[:, :, generated_start:generated_end] = rollout.noisy_at_t.video
        noisy_action[:, :, generated_start:generated_end] = rollout.noisy_at_t.action
        noisy_at_t = apply_va_mask(
            VAPair(noisy_video, noisy_action),
            clean_hat,
            masks,
        )
        # video/action 的退出 timestep 
        video_timesteps = torch.zeros(
            (clean_hat.video.shape[0], frames),
            dtype=torch.float32,
            device=clean_hat.video.device,
        )
        action_timesteps = torch.zeros_like(video_timesteps)
        video_timesteps[:, generated_start:generated_end] = rollout.video_exit_timestep
        action_timesteps[:, generated_start:generated_end] = rollout.action_exit_timestep
        video_timesteps = torch.where(video_mask, video_timesteps, 0)
        action_timesteps = torch.where(
            action_mask.any(dim=(1, 3, 4)),
            action_timesteps,
            0,
        )
        exit_timesteps = VATimesteps(video_timesteps, action_timesteps)
        replay_context = ReplayContext(
            exit_timesteps=exit_timesteps,
            noisy_at_t=noisy_at_t,
            clean_hat=clean_hat,
            masks=masks,
            denoisy_selection=selection,
        )
        return replay_context

    @torch.no_grad()
    def _run_generator(self, batch: dict) -> ReplayContext:
        """Run the complete final-clean SGF rollout and record independent exits."""
        from wan_va.mot_spec import mot_spec_from_config

        spec = mot_spec_from_config(self.config)
        rollout = self.pipeline.generate(
            batch,
            rollout_frames=self.rollout_horizon_frames,
            history_frames=int(spec.history_latent_frames),
            device=self.device,
        )
        return self.record(batch, rollout)

    @staticmethod
    def _rollout_batch(batch: dict, base_input: dict) -> dict:
        """Use the replay text condition for the corresponding SGF rollout."""
        text_emb = base_input["latent_dict"]["text_emb"]
        if batch.get("text_emb") is text_emb:
            return batch
        rollout_batch = dict(batch)
        rollout_batch["text_emb"] = text_emb
        return rollout_batch

    def replay(
        self,
        context: ReplayContext,
        *,
        requires_grad: bool,
        base_input: dict,
    ) -> VAPair:
        """Replay recorded exit states once through the generator."""
        # context.noisy_at_t.video -> replay_input["latent_dict"]["noisy_latents"]
        # context.noisy_at_t.action -> replay_input["action_dict"]["noisy_latents"]
        # context.clean_hat.video -> replay_input["latent_dict"]["latent"]
        # context.clean_hat.action -> replay_input["action_dict"]["latent"]
        replay_input = self._prepare_input(
            context.noisy_at_t,
            context.clean_hat,
            context.exit_timesteps,
            base_input,
        )
        grad_context = nullcontext() if requires_grad else torch.no_grad()
        with grad_context:
            estimated = self.generator(
                replay_input,
                context.noisy_at_t,
                context.exit_timesteps,
            ).x0
        replayed_x0 = apply_va_mask(
            estimated,
            context.clean_hat,
            context.masks,
        )
        return replayed_x0

class SGFDMDModel(SelfGradientForcingModel):
    """Compute DMD generator/fake-score losses on top of SGF replay."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.update_schedule = DMDUpdateSchedule(
            fake_score_steps=int(self.config.distill.fake_score_update_ratio)
        )
        self.score_input_use_gt_clean = bool(
            getattr(self.config.distill, "score_input_use_gt_clean", False)
        )

    def _get_timestep(
        self,
        interval: DenoisyInterval,
        shape: tuple[int, int],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample linear progress using the configured exit-bound switches."""
        # Stage 1/4: Choose the interval in linear progress d.  The four cases
        # are [min_score,T), [to,T), [min_score,from), and [to,from) for
        # (ts_schedule, ts_schedule_max) equal to (F,F), (T,F), (F,T), and
        # (T,T), respectively.
        denoisy_to = (
            interval.denoisy_to
            if self.ts_schedule
            else self.min_score_timestep
        )
        denoisy_from = (
            interval.denoisy_from
            if self.ts_schedule_max
            else self.num_train_timestep
        )
        # Sampling must happen before scheduler warping; uniform sampling from
        # the warped table would define a different probability density.
        # One linear progress per batch item in [to, from), broadcast across
        # frames to reproduce the reference sampler's uniform_timestep=True
        # behavior; masked frames are restored to numerical zero.
        low = int(math.ceil(float(denoisy_to)))
        high = int(math.ceil(float(denoisy_from)))
        progress = torch.randint(low, high, (shape[0], 1), device=self.device).expand(
            shape
        )
        return torch.where(mask, progress, 0)

    def _warp_timestep(
        self,
        progress: torch.Tensor,
        scheduler,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convert linear progress ``d`` to a clamped network timestep ``t``."""
        timestep = warp_denoisy_progress(
            progress,
            scheduler,
        )
        timestep = timestep.clamp(self.min_step, self.max_step)
        return torch.where(
            mask,
            timestep,
            0,
        )

    def _get_va_timesteps(
        self,
        selection: VADenoisySelection,
        masks: VAMasks,
    ) -> tuple[VATimesteps, VATimesteps]:
        """Return linear progress and warped network timesteps for both streams.

        progress(线性去噪进度 d,mask 帧为 0):
        VATimesteps = {
            "video":  [B,F],  # 每个 batch 样本采一个 d,广播到所有帧
            "action": [B,F],
        }

        timesteps(scheduler warp + clamp 后的网络时间 t,真正用于加噪/模型):
        VATimesteps = {
            "video":  [B,F],  # video/action 各自 scheduler shift 独立
            "action": [B,F],
        }
        """
        shape = tuple(masks.video.shape)
        video_mask = masks.video
        action_mask = masks.action.any(dim=(1, 3, 4))
        progress = VATimesteps(
            video=self._get_timestep(
                selection.video,
                shape,
                video_mask,
            ),
            action=self._get_timestep(
                selection.action,
                shape,
                action_mask,
            ),
        )
        # Video and action can map the same d to different t because their
        # schedulers use independent shift values.
        timesteps = VATimesteps(
            video=self._warp_timestep(
                progress.video,
                self.generator.video_scheduler,
                video_mask,
            ),
            action=self._warp_timestep(
                progress.action,
                self.generator.action_scheduler,
                action_mask,
            ),
        )
        return progress, timesteps

    def _add_dmd_noise(
        self,
        clean: VAPair,
        timesteps: VATimesteps,
        masks: VAMasks,
    ) -> tuple[VAPair, VAPair]:
        """Add DMD noise: x_t = (1-sigma) * x0 + sigma * noise.

        输入/输出 VAPair:
        VAPair = {
            "video":  [B,Cv,F,V,H,W],  # video latents(V=视角数,H/W=空间)
            "action": [B,Ca,F,N,1],    # action(N=每帧 action token 数)
        }

        sigma 按 timesteps [B,F] 逐帧查 scheduler 表；mask 外位置恢复 clean。
        """
        noise = randn_like_va(clean)

        noisy_video = self.generator.video_scheduler.add_noise(
            clean.video,
            noise.video,
            timesteps.video,
        )
        noisy_action = self.generator.action_scheduler.add_noise(
            clean.action,
            noise.action,
            timesteps.action,
        )

        noisy = apply_va_mask(
            VAPair(video=noisy_video, action=noisy_action),
            clean,
            masks,
        )
        return noisy, noise

    def _score_clean(
        self,
        generated_x0: VAPair,
        base_input: dict,
    ) -> VAPair:
        """Return the score condition clean stream under the GT-clean option."""
        if not self.score_input_use_gt_clean:
            return generated_x0
        score_clean = VAPair(
            video=base_input["latent_dict"]["latent"],
            action=base_input["action_dict"]["latent"],
        )
        return score_clean

    def _empty_text_condition(
        self,
        text_emb: torch.Tensor,
        empty_text_emb: torch.Tensor,
    ) -> torch.Tensor:
        return empty_text_emb.to(
            device=text_emb.device,
            dtype=text_emb.dtype,
        ).expand_as(text_emb)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.float()
        mask = mask.expand_as(values)
        return torch.where(mask, values, 0).sum() / mask.sum().clamp_min(1)

    def _selection_metrics(
        self,
        reference: torch.Tensor,
        selection: VADenoisySelection,
        progress: VATimesteps,
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
            "distill/dmd_video_d_mean": self._masked_mean(
                progress.video,
                masks.video,
            ).detach(),
            "distill/dmd_action_d_mean": self._masked_mean(
                progress.action,
                masks.action.any(dim=(1, 3, 4)),
            ).detach(),
            "distill/dmd_video_t_mean": self._masked_mean(
                timesteps.video,
                masks.video,
            ).detach(),
            "distill/dmd_action_t_mean": self._masked_mean(
                timesteps.action,
                masks.action.any(dim=(1, 3, 4)),
            ).detach(),
        }

    def _compute_kl_grad(
        self,
        score_input: dict,
        noisy: VAPair,
        estimated_clean: VAPair,
        timesteps: VATimesteps,
        masks: VAMasks,
        empty_text_emb: torch.Tensor,
    ) -> tuple[VAPair, dict[str, torch.Tensor]]:
        """Compute real/fake score predictions and the normalized KL gradient.

        This mirrors the reference DMD implementation: fake score first, real
        conditional/unconditional score second, then ``fake_x0 - real_x0`` and
        per-sample normalization. Video uses CFG while action remains
        conditional, matching the repository's joint V/A contract.
        """
        fake_output = self.fake_score(score_input, noisy, timesteps)
        real_conditional = self.real_score(score_input, noisy, timesteps)

        text_emb = score_input["latent_dict"]["text_emb"]
        unconditional_input = replace_text_condition(
            score_input,
            self._empty_text_condition(text_emb, empty_text_emb),
        )
        real_unconditional = self.real_score(
            unconditional_input,
            noisy,
            timesteps,
        )
        cfg_scale = self.teacher_cfg_min + torch.rand(()).item() * (
            self.teacher_cfg_max - self.teacher_cfg_min
        )
        real_x0 = VAPair(
            video=real_unconditional.x0.video
            + cfg_scale
            * (real_conditional.x0.video - real_unconditional.x0.video),
            action=real_conditional.x0.action,
        )

        def normalized_gradient(generator, fake, real, mask):
            mask = mask.expand_as(generator)
            gradient = fake - real
            error = torch.where(mask, (generator - real).abs(), 0)
            valid = mask.flatten(1).sum(1)
            normalizer = error.flatten(1).sum(1) / valid.clamp_min(1)
            normalizer = normalizer.clamp_min(self.normalizer_eps)
            normalizer = normalizer.reshape(-1, *([1] * (generator.ndim - 1)))
            normalized = gradient / normalizer
            return torch.where(mask, normalized, torch.zeros_like(normalized))

        gradient = VAPair(
            video=normalized_gradient(
                estimated_clean.video,
                fake_output.x0.video,
                real_x0.video,
                masks.video[:, None, :, None, None, None],
            ),
            action=normalized_gradient(
                estimated_clean.action,
                fake_output.x0.action,
                real_x0.action,
                masks.action,
            ),
        )
        return gradient, {
            "distill/dmd_video_gradient_norm": self._masked_mean(
                gradient.video.abs(),
                masks.video[:, None, :, None, None, None],
            ).detach(),
            "distill/dmd_action_gradient_norm": self._masked_mean(
                gradient.action.abs(),
                masks.action,
            ).detach(),
            "distill/sgf_real_cfg_scale": gradient.video.detach().new_tensor(
                cfg_scale
            ),
        }

    def compute_distribution_matching_loss(
        self,
        context: ReplayContext,
        generator_x0: VAPair,
        *,
        base_input: dict,
        empty_text_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score a replayed generator sample against real and fake distributions."""
        with torch.no_grad():
            score_progress, score_timesteps = self._get_va_timesteps(
                context.denoisy_selection,
                context.masks,
            )
            score_noisy, _ = self._add_dmd_noise(
                generator_x0,
                score_timesteps,
                context.masks,
            )
            score_input = self._prepare_input(
                score_noisy,
                self._score_clean(generator_x0, base_input),
                score_timesteps,
                base_input,
            )
            kl_grad, metrics = self._compute_kl_grad(
                score_input,
                score_noisy,
                generator_x0,
                score_timesteps,
                context.masks,
                empty_text_emb,
            )
        dmd_target = VAPair(
            video=generator_x0.video - kl_grad.video,
            action=generator_x0.action - kl_grad.action,
        )
        loss, dmd_metrics = dmd_surrogate_loss(
            generator_x0,
            dmd_target,
            context.masks,
            self.loss_weights,
        )
        metrics.update(dmd_metrics)
        metrics.update(
            self._selection_metrics(
                loss,
                context.denoisy_selection,
                score_progress,
                score_timesteps,
                context.masks,
            )
        )
        metrics["distill/sgf_total_loss"] = loss.detach()
        return loss, metrics

    def generator_loss(
        self,
        batch: dict,
        *,
        base_input: dict,
        empty_text_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run SGF generation, replay with gradients, and compute DMD loss."""
        batch_empty_text_emb = batch.get("empty_text_emb")
        if batch_empty_text_emb is not None:
            empty_text_emb = batch_empty_text_emb
        context = self._run_generator(self._rollout_batch(batch, base_input))
        generator_x0 = self.replay(
            context,
            requires_grad=True,
            base_input=base_input,
        )
        return self.compute_distribution_matching_loss(
            context,
            generator_x0,
            base_input=base_input,
            empty_text_emb=empty_text_emb,
        )

    def critic_loss(
        self,
        batch: dict,
        *,
        base_input: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run SGF generation and train the fake score on generated samples."""
        context = self._run_generator(self._rollout_batch(batch, base_input))
        generated_x0 = self.replay(
            context,
            requires_grad=False,
            base_input=base_input,
        )
        score_progress, score_timesteps = self._get_va_timesteps(
            context.denoisy_selection,
            context.masks,
        )
        score_noisy, noise = self._add_dmd_noise(
            generated_x0,
            score_timesteps,
            context.masks,
        )
        score_input = self._prepare_input(
            score_noisy,
            self._score_clean(generated_x0, base_input),
            score_timesteps,
            base_input,
        )
        fake_output = self.fake_score(
            score_input,
            score_noisy,
            score_timesteps,
        )
        exact_target = VAPair(
            video=noise.video - generated_x0.video,
            action=noise.action - generated_x0.action,
        )
        loss, metrics = fake_score_flow_loss(
            fake_output.velocity,
            exact_target,
            context.masks,
            self.loss_weights,
        )
        metrics.update(
            self._selection_metrics(
                loss,
                context.denoisy_selection,
                score_progress,
                score_timesteps,
                context.masks,
            )
        )
        return loss, metrics

    def optimizer_for_step(
        self,
        optimizer_step: int,
    ) -> Literal["generator", "fake_score"]:
        return self.update_schedule.optimizer_for_step(optimizer_step)

    def compute_step(
        self,
        batch: dict,
        optimizer_name: Literal["generator", "fake_score"],
        *,
        base_input: dict,
        empty_text_emb: torch.Tensor,
    ) -> TrainingStepResult:
        if optimizer_name == "generator":
            loss, metrics = self.generator_loss(
                batch,
                base_input=base_input,
                empty_text_emb=empty_text_emb,
            )
        else:
            loss, metrics = self.critic_loss(batch, base_input=base_input)
        step_result = TrainingStepResult(loss=loss, metrics=metrics)
        return step_result

    def state_dict(self) -> dict[str, Any]:
        return {
            "fake_score_update_ratio": self.update_schedule.fake_score_steps,
            "video_denoisy_step_list": tuple(self.video_denoising_step_list),
            "action_denoisy_step_list": tuple(self.action_denoising_step_list),
            "ts_schedule": self.ts_schedule,
            "ts_schedule_max": self.ts_schedule_max,
            "min_score_timestep": self.min_score_timestep,
            "dmd_timestep_min": self.min_step,
            "dmd_timestep_max": self.max_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ratio = int(
            state.get(
                "fake_score_update_ratio",
                self.update_schedule.fake_score_steps,
            )
        )
        expected_video = tuple(self.video_denoising_step_list)
        expected_action = tuple(self.action_denoising_step_list)
        saved_video = tuple(state.get("video_denoisy_step_list", expected_video))
        saved_action = tuple(state.get("action_denoisy_step_list", expected_action))
        if saved_video != expected_video or saved_action != expected_action:
            raise ValueError(
                "SGF denoisy step lists do not match checkpoint: "
                f"saved video/action={saved_video}/{saved_action}, "
                f"configured={expected_video}/{expected_action}"
            )
        expected_sampling = {
            "ts_schedule": self.ts_schedule,
            "ts_schedule_max": self.ts_schedule_max,
            "min_score_timestep": self.min_score_timestep,
            "dmd_timestep_min": self.min_step,
            "dmd_timestep_max": self.max_step,
        }
        for name, expected in expected_sampling.items():
            saved = state.get(name, expected)
            if saved != expected:
                raise ValueError(
                    f"DMD sampling setting {name} does not match checkpoint: "
                    f"saved={saved}, configured={expected}"
                )
        self.update_schedule = DMDUpdateSchedule(fake_score_steps=ratio)
