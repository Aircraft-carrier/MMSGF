"""Shared distillation trainer runtime."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from distillation.checkpoint import DistillationCheckpointIO
from distillation.mask_profile import (
    install_order_profile,
    validate_checkpoint_generation_profile,
)
from distillation.schema import TrainingStepResult
from wan_va.train_mot import MOTTrainer


@dataclass(frozen=True, slots=True)
class OptimizationTarget:
    name: Literal["student", "fake_score"]
    optimizer: torch.optim.Optimizer
    model: torch.nn.Module


class DistillationTrainerBase(MOTTrainer):
    """Stage2/3 shared runtime from dataloader batch to optimizer/checkpoint.

    数据准备有两条合法入口：

    1. cached latent batch 已含 ``latents [B,48,8,V,Hl,Wl]``，直接使用；
    2. RGB batch 含 ``vae_rgb_history`` 与 ``vae_rgb_target``，调用父类 streaming
       VAE 逐 batch/逐 view 编码，再沿 frame 轴拼成同样的 8-frame latent。

    ``actions [B,20,8,16,1]``、V/A loss/valid mask、text embedding、geometry
    RGB/point/mask 和 ``stream_ids [B,V]`` 都继续来自原生 MOT dataset/collate。
    ``convert_input_format`` 先递归搬 tensor 到当前 rank device；
    ``_materialize_batch_latents`` 只在缺少 cached latent 时调用 VAE。蒸馏代码
    不伪造真实 RGB/geometry schema，也不复制 dataset 逻辑。

    本类只接管 stage-specific loss、双 optimizer/EMA hook 和 checkpoint；外层
    epoch/step、view-aware sampler、日志、NaN 同步仍复用 ``MOTTrainer.train``。
    """

    method: str

    def _load_transformer(self):
        # MOTTrainer applies activation checkpoint wrappers immediately after
        # this hook. Install the instance-local attention policy first so each
        # wrapper preserves the patched block forward instead of being bypassed.
        model = super()._load_transformer()
        install_order_profile(model, self.config.distill.generation_shape)
        return model

    def __init__(self, config: Any):
        self._resume_from = getattr(config.distill, "resume_from", None)
        student_init = getattr(config.distill, "student_init", None)

        # MOTTrainer only understands its top-level ``initialize_from`` and
        # ``resume_from`` fields.  Distillation CLI arguments live under
        # ``config.distill``; copy them before parent construction so the actual
        # trainable student starts from the requested previous-stage export.
        #
        # Fresh stage2 example:
        #   distill.student_init = stage1/checkpoint_step_2000
        #   -> initialize_from    = the same checkpoint
        #   -> parent loads stage1/transformer into ``self.transformer``
        #
        # Resume example:
        #   distill.resume_from = stage2/checkpoint_step_4000
        #   -> parent first materializes a model with the resume export layout;
        #   -> DistillationCheckpointIO later restores raw student/optimizer/EMA.
        # Clear the native env-derived resume field in both cases; otherwise an
        # unrelated MOT_RESUME_FROM takes precedence over ``student_init``.
        config.resume_from = None
        if self._resume_from is not None:
            config.initialize_from = str(Path(self._resume_from))
        elif student_init is not None:
            validate_checkpoint_generation_profile(
                student_init,
                config.distill.generation_shape,
            )
            config.initialize_from = str(Path(student_init))
        else:
            raise ValueError(
                f"{self.method} requires distill.student_init for a fresh run"
            )

        super().__init__(config)
        install_order_profile(self.transformer, config.distill.generation_shape)

        self.method_model = self._build_method_model(config)
        self.checkpoint_io = DistillationCheckpointIO()

    @staticmethod
    def _local_metric_value(value: Any) -> Any:
        try:
            from torch.distributed.tensor import DTensor
        except Exception:
            DTensor = ()
        if torch.is_tensor(value) and isinstance(value, DTensor):
            return value.to_local()
        return value

    def _aggregate_log_records(self, records: list[dict[str, Any]]) -> dict[str, float]:
        local_records = [
            {key: self._local_metric_value(value) for key, value in record.items()}
            for record in records
        ]
        return super()._aggregate_log_records(local_records)

    def _build_method_model(self, config: Any):
        raise NotImplementedError

    def _prepare_joint_input_dict(self, batch_dict: dict, *, add_noise=True) -> dict:
        """Reuse native field validation but apply the configured distill mask profile.

        ``MOTWindowSpec`` describes physical packing (8 latent frames, 16 action
        tokens/frame); ``generation_shape`` describes how those frames receive causal
        order ids. Distillation uses the segmented profile: the first half keeps
        chunk order, while the second half advances one frame at a time. Stage1,
        stage2 and stage3 must use the same values, or a checkpoint would be trained
        under one attention graph while metadata advertises another.
        """
        input_dict = super()._prepare_joint_input_dict(
            batch_dict,
            add_noise=add_noise,
        )
        shape = self.config.distill.generation_shape
        input_dict["chunk_size"] = int(shape["chunk_size"])
        input_dict["window_size"] = int(shape["window_size"])
        return input_dict

    def _optimization_target(self) -> OptimizationTarget:
        return OptimizationTarget("student", self.optimizer, self.transformer)

    def _compute_training_step(
        self,
        batch: dict,
        target: OptimizationTarget,
    ) -> TrainingStepResult:
        return self.method_model.compute_step(batch)

    def _after_optimizer_step(self, target: OptimizationTarget) -> None:
        self.lr_scheduler.step()

    def _maybe_run_training_rollout(self, batch: dict, completed_step: int) -> None:
        return None

    def _loss_metrics(
        self,
        loss: torch.Tensor,
        metrics: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        zero = loss.detach().new_zeros(())
        video_loss = sum(
            (value for key, value in metrics.items() if key.endswith("_video_loss")),
            zero,
        )
        action_loss = sum(
            (value for key, value in metrics.items() if key.endswith("_action_loss")),
            zero,
        )
        video_weight = float(self.config.video_loss_weight)
        action_weight = float(self.config.action_loss_weight)
        return {
            "total_loss_raw": loss.detach(),
            "latent_loss_raw": video_loss,
            "action_loss_raw": action_loss,
            "weighted_video_loss_raw": video_weight * video_loss,
            "weighted_action_loss_raw": action_weight * action_loss,
            "video_loss_weight": zero.new_tensor(video_weight),
            "action_loss_weight": zero.new_tensor(action_weight),
        }

    @staticmethod
    def _data_metrics(batch: dict, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = batch["latents"].shape[0]
        has_pointcloud = batch.get("has_pointcloud")
        if has_pointcloud is None:
            has_pointcloud = batch["geometry_group_valid_mask"].reshape(batch_size, -1).any(1)
        has_pointcloud = torch.as_tensor(has_pointcloud, device=reference.device, dtype=torch.bool)

        skip_count = batch.get("dataset_skip_count")
        if skip_count is None:
            skip_count = reference.new_zeros(batch_size)
        skip_count = torch.as_tensor(skip_count, device=reference.device)

        video_loss_mask = batch["video_latent_loss_mask"]
        video_valid_mask = batch["video_latent_valid_mask"]
        action_loss_mask = batch["action_loss_mask"]
        action_valid_mask = batch["action_valid_mask"]
        return {
            "data_pointcloud_samples": has_pointcloud.sum().float(),
            "data_pure_samples": (~has_pointcloud).sum().float(),
            "data_dataset_skip_count": skip_count.sum().float(),
            "data_local_samples": reference.new_tensor(batch_size, dtype=torch.float32),
            "data_native_views": reference.new_tensor(batch["latents"].shape[3], dtype=torch.float32),
            "data_video_supervised_num": video_loss_mask.sum().float(),
            "data_video_supervised_den": reference.new_tensor(video_loss_mask.numel()),
            "data_video_valid_num": video_valid_mask.sum().float(),
            "data_video_valid_den": reference.new_tensor(video_valid_mask.numel()),
            "data_action_supervised_num": action_loss_mask.sum().float(),
            "data_action_supervised_den": reference.new_tensor(action_loss_mask.numel()),
            "data_action_valid_num": action_valid_mask.sum().float(),
            "data_action_valid_den": reference.new_tensor(action_valid_mask.numel()),
        }

    @staticmethod
    def _active_nonfinite_grad_locations(
        model: torch.nn.Module,
        max_items: int = 20,
    ) -> list[dict[str, Any]]:
        locations = []
        for name, parameter in model.named_parameters():
            grad = parameter.grad
            if grad is None or bool(torch.isfinite(grad).all()):
                continue
            locations.append(
                {
                    "kind": "grad",
                    "name": name,
                    "num_nonfinite": int((~torch.isfinite(grad)).sum().item()),
                }
            )
            if len(locations) == max_items:
                break
        return locations

    def _train_step(
        self,
        batch,
        batch_idx,
        *,
        collect_detailed_metrics: bool = False,
        measure_performance: bool = False,
    ):
        """Run one microstep without requiring a stage-specific parent train loop.

        输入可能是 RGB batch 或 cached-latent batch。先递归搬到当前 device，再
        materialize ``latents``；stage2/stage3 从此处开始看到统一结构。之后：

        1. ``_optimization_target`` 选择 student 或 fake-score；
        2. stage-specific pipeline 返回一个保留 graph 的标量 loss；
        3. loss 除以 gradient_accumulation_steps 后 backward；
        4. 仅累积窗口最后一个 microstep 做 clip/finite check/optimizer.step；
        5. 成功 student step 才推进 student LR scheduler；stage2 还更新 EMA；
        6. 父 train loop 看到 optimizer_step_event 后才增加 optimizer_step。

        例如 accumulation=2、stage3 当前选择 fake-score：batch_idx=0 只累积 grad，
        batch_idx=1 才 clip 和 step；两个 microstep 的 optimizer_step 相同，所以
        都走 fake-score 路径，不会出现第一半 fake、第二半 student 的混合窗口。
        """
        batch = self.convert_input_format(batch)
        batch = self._materialize_batch_latents(batch)
        target = self._optimization_target()
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if hasattr(target.model, "set_requires_gradient_sync"):
            target.model.set_requires_gradient_sync(should_sync)

        result = self._compute_training_step(batch, target)
        loss = result.loss
        metrics = result.metrics

        local_loss_finite = torch.isfinite(loss).all()
        global_loss_bad = self._distributed_any(not local_loss_finite)
        out = {
            "loss": loss,
            **metrics,
            **self._loss_metrics(loss, metrics),
            **self._data_metrics(batch, loss),
            "nonfinite_loss_event": bool(global_loss_bad),
            "nonfinite_grad_event": False,
            "optimizer_step_event": False,
            "grad_clip_event": torch.zeros((), device=self.device),
            "grad_clip_count": torch.zeros((), device=self.device),
            "skipped_step": False,
            "should_log": True,
            "_phase_timings": {},
        }
        if global_loss_bad:
            # 任一 rank loss 非有限时，所有 rank 都丢弃本累积窗口的梯度，避免
            # FSDP collective/optimizer state 在不同 rank 上发生分叉。
            self._skip_optimizer_step_for_nan(
                trigger_stage="loss",
                losses=out,
                total_norm=None,
                loss_finite=local_loss_finite,
                grad_finite=None,
                global_has_nan=True,
                rank_has_nan=not local_loss_finite,
                nan_locations=[],
                batch=batch,
            )
            target.optimizer.zero_grad(set_to_none=True)
            out["skipped_step"] = True
            out["total_norm"] = torch.tensor(float("nan"), device=self.device)
            return {key: value.detach() if torch.is_tensor(value) else value for key, value in out.items()}

        inv_accum = 1.0 / float(self.gradient_accumulation_steps)
        (loss * inv_accum).backward()

        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(
                target.model.parameters(),
                self.config.distill.max_grad_norm,
            )
            local_grad_finite = bool(torch.isfinite(total_norm).all())
            global_grad_bad = self._distributed_any(not local_grad_finite)
            if global_grad_bad:
                out["nonfinite_grad_event"] = True
                self._skip_optimizer_step_for_nan(
                    trigger_stage="grad",
                    losses=out,
                    total_norm=total_norm,
                    loss_finite=local_loss_finite,
                    grad_finite=local_grad_finite,
                    global_has_nan=True,
                    rank_has_nan=not local_grad_finite,
                    nan_locations=(
                        self._active_nonfinite_grad_locations(target.model)
                        if not local_grad_finite
                        else []
                    ),
                    batch=batch,
                )
                target.optimizer.zero_grad(set_to_none=True)
                out["skipped_step"] = True
            else:
                # 顺序很重要：先更新参数，再执行 stage hook。stage2 的 EMA 因而
                # 读取本次新 student 参数；fake-score step 不会误推进 student LR。
                target.optimizer.step()
                self._after_optimizer_step(target)
                target.optimizer.zero_grad(set_to_none=True)
                out["optimizer_step_event"] = True
                out["grad_clip_event"] = (
                    total_norm > self.config.distill.max_grad_norm
                ).to(dtype=torch.float32)
                out["grad_clip_count"] = torch.ones((), device=self.device)
            out["total_norm"] = total_norm.detach()
        else:
            out["should_log"] = False
            out["total_norm"] = torch.tensor(float("nan"), device=self.device)

        if out["optimizer_step_event"]:
            self._maybe_run_training_rollout(batch, self.step + 1)
        return {key: value.detach() if torch.is_tensor(value) else value for key, value in out.items()}

    def save_checkpoint(self) -> Path:
        return self.checkpoint_io.save(self)

    def load_checkpoint(self, checkpoint_root: str | Path) -> None:
        self.checkpoint_io.load(self, Path(checkpoint_root))

    def get_method_state_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "method_model_state_dict": self.method_model.state_dict(),
        }

    def load_method_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("method") != self.method:
            raise ValueError(
                f"Checkpoint method {state.get('method')} does not match "
                f"current method {self.method}"
            )
        if "method_model_state_dict" in state:
            self.method_model.load_state_dict(state["method_model_state_dict"])
