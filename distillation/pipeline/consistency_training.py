"""Consistency distillation trajectory."""
from typing import Any

import torch
import torch.nn as nn

from distillation.model.objectives import action_aware_loss, consistency_loss
from distillation.pipeline.utils import (
    add_noise_to_va,
    replace_text_condition,
    replace_va_streams,
)
from distillation.scheduler import (
    consistency_prediction,
    flow_step,
    flow_to_x0,
    sample_consistency_timesteps,
)
from distillation.schema import VALossWeights, VAMasks, VAPrediction, VATimesteps


class ConsistencyTrainingPipeline:
    """Stage2 consistency distillation 的单个 microstep 数据流。

    真实 batch 的主要结构（空间尺寸和 view 数由 dataset 决定）：

    - video latent ``[B,48,8,V,Hl,Wl]``；8 帧通常是 4 history + 4 target；
    - action ``[B,20,8,16,1]``；20 是 action channel，16 是每 latent frame
      打包的 action token 数；
    - video loss mask ``[B,8]``；action loss mask 与 action 同形；
    - text/stream_ids 不在本类重建，原样复用 MOTTrainer 的输入。

    为便于人工手算，可以把它缩小成 ``B=1,Cv=1,F=2,Ca=1``：frame0 是
    condition，frame1 是监督位置。假设 frame1 ``x0=2, noise=6, sigma=0.75``，
    则 ``x_t=(1-0.75)*2+0.75*6=5``，精确 flow 为 ``noise-x0=4``。
    teacher 预测 flow 后把 5 推进到较低 sigma 的相邻状态；EMA 在相邻状态
    给 target，student 在原状态给 prediction，最后只在 frame1 计算 loss。

    这里的“train mode”是 MOT 模型的 joint V/A/G forward route；teacher 与
    EMA 模型对象仍然是 eval、requires_grad=False，且调用位于 no_grad 中。
    """
    def __init__(
        self,
        config: Any,
        trainer: Any,
        device: torch.device,
        student: nn.Module,
        teacher: nn.Module,
        ema_student: nn.Module,
    ):
        self.trainer = trainer
        self.device = device
        self.student = student
        self.teacher = teacher
        self.ema_student = ema_student
        self.video_num_steps = int(config.distill.video_num_steps)
        self.action_num_steps = int(config.distill.action_num_steps)
        self.cfg_min = float(config.distill.cfg_min)
        self.cfg_max = float(config.distill.cfg_max)
        if self.cfg_min < 0.0 or self.cfg_max < self.cfg_min:
            raise ValueError(
                "Consistency teacher CFG requires 0 <= cfg_min <= cfg_max, got "
                f"{self.cfg_min} and {self.cfg_max}"
            )
        self.sigma_data = float(config.distill.sigma_data)
        if self.sigma_data <= 0.0:
            raise ValueError(f"sigma_data must be positive, got {self.sigma_data}")
        self.action_aware_weight = float(config.distill.action_aware_weight)
        self.loss_weights = VALossWeights(
            video=float(config.video_loss_weight),
            action=float(config.action_loss_weight),
        )

    def _predict_consistency(
        self,
        model: nn.Module,
        input_dict: dict,
        noisy: VAPrediction,
        timesteps: VATimesteps,
    ) -> VAPrediction:
        out = model(input_dict, mode="train")
        flow = VAPrediction(out["latent_pred"], out["action_pred"])
        return VAPrediction(
            video=consistency_prediction(
                flow.video,
                noisy.video,
                timesteps.video,
                self.trainer.train_scheduler_latent,
                sigma_data=self.sigma_data,
            ),
            action=flow_to_x0(
                flow.action,
                noisy.action,
                timesteps.action,
                self.trainer.train_scheduler_action,
            ),
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
        conditioned = self.teacher(input_dict, mode="train")
        text_emb = input_dict["latent_dict"]["text_emb"]
        empty_text_emb = self._empty_text_condition(batch, text_emb)
        unconditioned = self.teacher(
            replace_text_condition(input_dict, empty_text_emb),
            mode="train",
        )
        cfg_scale = self.cfg_min + torch.rand(()).item() * (
            self.cfg_max - self.cfg_min
        )
        video = unconditioned["latent_pred"] + cfg_scale * (
            conditioned["latent_pred"] - unconditioned["latent_pred"]
        )
        return VAPrediction(video, conditioned["action_pred"]), cfg_scale

    def _sample_timesteps(
        self,
        masks: VAMasks,
    ) -> tuple[VATimesteps, VATimesteps]:
        shape = masks.video.shape
        video, next_video = sample_consistency_timesteps(
            self.trainer.train_scheduler_latent,
            self.video_num_steps,
            shape,
            self.device,
            masks.video,
        )
        action_mask = masks.action.any(dim=(1, 3, 4))
        action, next_action = sample_consistency_timesteps(
            self.trainer.train_scheduler_action,
            self.action_num_steps,
            shape,
            self.device,
            action_mask,
        )
        return VATimesteps(video, action), VATimesteps(next_video, next_action)

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """把一个已 materialize 的 batch 推到 stage2 标量 loss。

        假设一条样本的 video mask 为 ``[0,0,0,0,0,1,1,1]``：前四帧是
        history，第 4 号 target anchor 也保持 clean，只监督最后三帧。一次
        可能的 timestep 采样为 ``t=[0,0,0,0,0,900,620,300]``，相邻低噪声
        状态为 ``t_next=[0,0,0,0,0,400,120,0]``（具体值取决于 scheduler
        离散表）。mask=0 的位置即使数值 timestep 写成 0，也会再次通过 mask
        强制保留 clean，不能把“数字 0”误解成 scheduler 一定有严格 sigma=0。

        返回的 ``loss`` 保留 student graph；metrics 全部 detach，只用于日志。
        """
        trainer = self.trainer

        # 1) 复用原生 input builder 做 dtype、mask、text、stream id 和
        #    attention-window 校验；不让它生成 AR noise，因为下面马上会生成
        #    consistency 专用的同一条 (x_t -> x_t_next) trajectory。
        base_input = trainer._prepare_joint_input_dict(batch, add_noise=False)
        bsz, _, frames = batch["latents"].shape[:3]

        # 2) 只从 input builder 读回规范化后的 bool mask。video 是 frame 级，
        #    action 是 element 级；action 某 frame 任一 token 有效时该 frame 才
        #    采样 action timestep。
        masks = VAMasks(
            video=base_input["latent_dict"]["video_latent_loss_mask"].reshape(
                bsz,
                frames,
            ),
            action=base_input["action_dict"]["action_loss_mask"],
        )
        timesteps, next_timesteps = self._sample_timesteps(masks)

        # 3) clean 与 noise 完全同形，但 video/action 分别采样随机噪声。
        #    例如 action [1,20,8,16,1] 有 2560 个值；mask 可以只选择其中一部分，
        #    未选择的 token 在 add_noise_to_va 中逐元素恢复为原 clean action。
        clean = VAPrediction(batch["latents"], batch["actions"])
        noise = VAPrediction(
            torch.randn_like(clean.video),
            torch.randn_like(clean.action),
        )
        noisy = add_noise_to_va(
            clean,
            noise,
            timesteps,
            masks,
            trainer.train_scheduler_latent,
            trainer.train_scheduler_action,
        )

        with torch.no_grad():
            # 4) Frozen teacher 在同一个 x_t 上做 conditional/unconditional forward。
            #    只对 video 做 CFG：v = v_u + scale*(v_c-v_u)；action 直接用 v_c。
            teacher_input = replace_va_streams(
                base_input,
                noisy,
                clean,
                timesteps,
            )
            teacher_flow, cfg_scale = self._teacher_cfg_flow(teacher_input, batch)

            # 5) 不调用原生 scheduler.step，因为它会把整个 [B,F] timestep tensor
            #    压成单个全局索引。flow_step 按每个 frame 的 sigma 差执行：
            #    x_next = x_t + (sigma_next - sigma_t) * teacher_flow。
            next_noisy = VAPrediction(
                flow_step(
                    teacher_flow.video,
                    noisy.video,
                    timesteps.video,
                    next_timesteps.video,
                    trainer.train_scheduler_latent,
                ),
                flow_step(
                    teacher_flow.action,
                    noisy.action,
                    timesteps.action,
                    next_timesteps.action,
                    trainer.train_scheduler_action,
                ),
            )

            # 6) EMA student 在 x_next/t_next 产生 stop-gradient target。
            #    video 使用 consistency boundary scaling；action 转为线性 x0。
            ema_input = replace_va_streams(
                base_input,
                next_noisy,
                clean,
                next_timesteps,
            )
            target_consistency = self._predict_consistency(
                self.ema_student,
                ema_input,
                next_noisy,
                next_timesteps,
            )

        # 7) 只有这一遍 student forward 在 autograd graph 中。输入仍是原始 x_t/t，
        #    所以它学习让同一 consistency function 在相邻 trajectory 点保持一致。
        student_input = replace_va_streams(base_input, noisy, clean, timesteps)
        student_out = self.student(student_input, mode="train")
        student_flow = VAPrediction(
            student_out["latent_pred"],
            student_out["action_pred"],
        )
        student_consistency = VAPrediction(
            video=consistency_prediction(
                student_flow.video,
                noisy.video,
                timesteps.video,
                trainer.train_scheduler_latent,
                sigma_data=self.sigma_data,
            ),
            action=flow_to_x0(
                student_flow.action,
                noisy.action,
                timesteps.action,
                trainer.train_scheduler_action,
            ),
        )

        # 8) 主 loss 在有效 V/A mask 内比较 student 与 EMA consistency output。
        #    loss helper 先用 FP32 做 elementwise MSE，再按 frame/mask 归一，避免
        #    view 数、空间分辨率或 padding token 数改变单样本权重。
        consistency, metrics = consistency_loss(
            student_consistency,
            target_consistency,
            masks,
            self.loss_weights,
        )

        # 9) action-aware 辅助项直接回归精确 flow=noise-clean。它不是第二次
        #    forward，只是复用 student_action_flow，默认乘 0.01 防止 action 的
        #    consistency x0 参数化在高噪声处训练信号过弱。
        action_target = trainer.train_scheduler_action.training_target(
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
