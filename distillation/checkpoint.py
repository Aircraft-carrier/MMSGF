"""Distillation checkpoint IO."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)

from distillation.configs import CONSISTENCY_DISTILLATION, SELF_GRADIENT_FORCING_DMD
from distillation.mask_profile import generation_profile_contract
from distillation.schema import CheckpointMetadata
from wan_va.checkpoint_retention import prune_successful_checkpoints

MOT_DCP_DIR_NAME = "distributed_state"
CHECKPOINT_FORMAT_VERSION = 3


def find_latest_successful_checkpoint(checkpoint_root: Path) -> Path | None:
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        if path.is_dir() and (path / "_SUCCESS").exists():
            name = path.name
            if name.startswith("checkpoint_step_"):
                try:
                    step = int(name[len("checkpoint_step_") :])
                    candidates.append((step, path))
                except ValueError:
                    continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


class DistillationCheckpointIO:
    def save(self, trainer: Any) -> Path:
        """Atomically save resume state plus the model consumed by the next stage.

        发布后的目录结构为：

        ``checkpoint_step_N/``
        ``  transformer/``              下一阶段 ``from_pretrained`` 的 export
        ``  distributed_state/``        DCP sharded model/optimizer states
        ``  training_state.pt``         step、LR scheduler、RNG、method state
        ``  checkpoint_metadata.json``  MOT 兼容字段 + distillation 字段
        ``  _SUCCESS``                  最后写入的发布完成标记

        stage2 的 ``transformer/`` 是 EMA student，但 DCP 的 ``model`` 是 raw
        student，并额外含 ``ema_student``；这样“交给 stage3 的稳定模型”和
        “stage2 精确续训状态”可以同时满足。stage3 export 是 student，DCP 另外
        保存 fake-score 与其 optimizer。teacher/real-score 可从配置来源重建，
        不重复写入 checkpoint。
        """
        checkpoint_dir = trainer.save_dir / f"checkpoint_step_{trainer.step}"
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
        temp_dir = trainer._create_checkpoint_temp_dir(checkpoint_dir)

        local_error = None
        local_exception = None
        try:
            self._save_dcp_state(trainer, temp_dir / MOT_DCP_DIR_NAME)
            export_model = (
                trainer.model.ema_student
                if trainer.method == CONSISTENCY_DISTILLATION
                else trainer.transformer
            )
            transformer_export = get_model_state_dict(
                export_model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            rng_states = trainer._collect_rng_states()
            if trainer.config.rank == 0:
                trainer._write_transformer_export(temp_dir, transformer_export)
                torch.save(
                    self._build_training_state(trainer, rng_states),
                    temp_dir / "training_state.pt",
                )
                self._write_metadata(temp_dir, trainer)
        except Exception as exc:
            local_exception = exc
            local_error = f"{type(exc).__name__}: {exc}"

        error = self._collect_error(local_error, trainer)
        if error is not None:
            if trainer.config.rank == 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(
                f"Failed to save distillation checkpoint: {error}"
            ) from local_exception

        publish_error, publish_exception = self._publish_checkpoint(
            temp_dir,
            checkpoint_dir,
            trainer,
        )
        error = self._collect_error(publish_error, trainer)
        if error is not None:
            if trainer.config.rank == 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(
                f"Failed to publish distillation checkpoint: {error}"
            ) from publish_exception

        retention_error, retention_exception = None, None
        if trainer.config.rank == 0:
            try:
                removed = prune_successful_checkpoints(
                    trainer.save_dir,
                    max_checkpoints=trainer.max_checkpoints,
                )
                if removed:
                    logging.getLogger(__name__).info(
                        "Removed old distillation checkpoints: "
                        + ", ".join(p.name for p in removed)
                    )
            except Exception as exc:
                retention_exception = exc
                retention_error = f"{type(exc).__name__}: {exc}"
        error = self._collect_error(retention_error, trainer)
        if error is not None:
            raise RuntimeError(
                f"Distillation checkpoint step {trainer.step} saved, but retention failed: {error}"
            ) from retention_exception

        trainer._last_checkpoint_step = int(trainer.step)
        return checkpoint_dir

    def load(self, trainer: Any, checkpoint_root: Path) -> None:
        """Restore a same-method run after models/optimizers have been constructed.

        加载顺序不能只读取 ``transformer/``：该 export 是跨阶段推理/初始化用，
        不一定等于训练中的 raw model（stage2 就导出 EMA）。这里先验证 method 和
        format，再用 DCP 恢复 raw student + optimizer + EMA/fake-score，最后恢复
        step、student LR scheduler、每个 rank 的 RNG 与 sampler offset。

        RNG 恢复意味着 resume 后的 timestep/noise/CFG/update schedule 能从保存点
        继续，而不是重复或跳过随机轨迹。``optimizer_step`` 也必须恢复，因为
        stage3 用它决定下一次是 fake-score 还是 student update。
        """
        checkpoint_root = Path(checkpoint_root)
        training_state = torch.load(
            checkpoint_root / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        metadata = CheckpointMetadata(
            **json.loads(
                (checkpoint_root / "checkpoint_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        if metadata.distill_method != trainer.method:
            raise ValueError(
                f"Checkpoint method {metadata.distill_method} does not match "
                f"trainer method {trainer.method}"
            )
        expected_architecture = str(
            getattr(
                trainer.config.distill,
                "model_architecture",
                "autoregressive_va_mot_v1",
            )
        )
        if getattr(metadata, "model_architecture", expected_architecture) != expected_architecture:
            raise ValueError(
                "Checkpoint model architecture does not match trainer: "
                f"checkpoint={getattr(metadata, 'model_architecture', None)}, "
                f"expected={expected_architecture}"
            )
        if metadata.format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format version {metadata.format_version}"
            )
        expected_profile = generation_profile_contract(
            trainer.config.distill.generation_shape
        )
        if metadata.generation_profile != expected_profile:
            raise ValueError(
                "Checkpoint generation profile does not match trainer: "
                f"checkpoint={metadata.generation_profile}, expected={expected_profile}"
            )

        state = self._dcp_state_dict(trainer)
        dcp.load(
            state,
            checkpoint_id=checkpoint_root / MOT_DCP_DIR_NAME,
            process_group=getattr(trainer, "checkpoint_process_group", None),
        )
        self._set_dcp_state_dict(trainer, state)

        trainer.step = int(training_state["step"])
        trainer.optimizer_step = int(training_state["optimizer_step"])
        trainer.skipped_step_total = int(training_state.get("skipped_step_total", 0))
        trainer.nonfinite_loss_total = int(training_state.get("nonfinite_loss_total", 0))
        trainer.nonfinite_grad_total = int(training_state.get("nonfinite_grad_total", 0))
        trainer.dataset_skip_total = int(training_state.get("dataset_skip_total", 0))
        trainer.lr_scheduler.load_state_dict(training_state["lr_scheduler_state_dict"])
        if "rng_states_by_rank" in training_state:
            rank = int(getattr(trainer.config, "rank", 0))
            trainer._restore_rng_state(training_state["rng_states_by_rank"][rank])
        if "method_state_dict" in training_state:
            trainer.load_method_state_dict(training_state["method_state_dict"])
        trainer._set_sampler_resume_offset()
        trainer._last_checkpoint_step = trainer.step
        trainer.performance_jsonl_start_step = trainer.step

        if dist.is_initialized():
            group = getattr(trainer, "checkpoint_process_group", None)
            dist.barrier(group=group)

    def _save_dcp_state(self, trainer: Any, dcp_path: Path) -> None:
        dcp.save(
            self._dcp_state_dict(trainer),
            checkpoint_id=dcp_path,
            process_group=getattr(trainer, "checkpoint_process_group", None),
        )

    def _dcp_state_dict(self, trainer: Any) -> dict[str, Any]:
        # Stage2 resume bundle:
        #   raw student + student AdamW + EMA student
        # Stage3 resume bundle:
        #   student + student AdamW + fake-score + fake-score AdamW
        # 冻结的 teacher/real-score 没有 optimizer，也不会随训练变化。
        options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=True)
        if trainer.method == CONSISTENCY_DISTILLATION:
            model_state, optimizer_state = get_state_dict(
                trainer.transformer,
                trainer.optimizer,
                options=options,
            )
            ema_state = get_model_state_dict(
                trainer.model.ema_student,
                options=options,
            )
            return {
                "model": model_state,
                "optimizer": optimizer_state,
                "ema_student": ema_state,
            }
        if trainer.method == SELF_GRADIENT_FORCING_DMD:
            model_state, optimizer_state = get_state_dict(
                trainer.transformer,
                trainer.optimizer,
                options=options,
            )
            fake_state, fake_optimizer_state = get_state_dict(
                trainer.model.fake_score.model,
                trainer.fake_score_optimizer,
                options=options,
            )
            return {
                "model": model_state,
                "optimizer": optimizer_state,
                "fake_score": fake_state,
                "fake_score_optimizer": fake_optimizer_state,
            }
        raise ValueError(f"Unsupported distillation method {trainer.method}")

    def _set_dcp_state_dict(
        self,
        trainer: Any,
        state: dict[str, Any],
    ) -> None:
        options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
            strict=True,
        )
        set_state_dict(
            trainer.transformer,
            trainer.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=options,
        )
        if trainer.method == CONSISTENCY_DISTILLATION:
            set_model_state_dict(
                trainer.model.ema_student,
                state["ema_student"],
                options=options,
            )
        else:
            set_state_dict(
                trainer.model.fake_score.model,
                trainer.fake_score_optimizer,
                model_state_dict=state["fake_score"],
                optim_state_dict=state["fake_score_optimizer"],
                options=options,
            )
        from wan_va.train_mot import _configure_adamw_foreach

        _configure_adamw_foreach(trainer.optimizer)
        if getattr(trainer, "fake_score_optimizer", None) is not None:
            _configure_adamw_foreach(trainer.fake_score_optimizer)

    def _build_training_state(
        self,
        trainer: Any,
        rng_states_by_rank: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        base_state = trainer._trainer_state(rng_states_by_rank)
        base_state["method_state_dict"] = trainer.get_method_state_dict()
        return base_state

    def _write_metadata(self, temp_dir: Path, trainer: Any) -> None:
        # 先调用 MOT 原生 writer，保证这个 distillation export 可以直接作为
        # 下一阶段的 ``student_init`` / ``fake_score_init``。尤其是 stage3 加载
        # stage2 时，``_validate_transformer_checkpoint_layout`` 会严格检查
        # checkpoint_type 和纯 VA model architecture。
        trainer._write_checkpoint_metadata(temp_dir, has_full_state=True)
        base_metadata = json.loads(
            (temp_dir / "checkpoint_metadata.json").read_text(encoding="utf-8")
        )
        shape = trainer.config.distill.generation_shape
        metadata = CheckpointMetadata(
            format_version=int(base_metadata["format_version"]),
            checkpoint_type=str(base_metadata["checkpoint_type"]),
            model_architecture=str(
                getattr(
                    trainer.config.distill,
                    "model_architecture",
                    "autoregressive_va_mot_v1",
                )
            ),
            has_full_state=bool(base_metadata["has_full_state"]),
            distill_method=trainer.method,
            exported_model=(
                "ema_student"
                if trainer.method == CONSISTENCY_DISTILLATION
                else "student"
            ),
            step=int(trainer.step),
            optimizer_step=int(trainer.optimizer_step),
            generation_profile=generation_profile_contract(shape),
        )
        (temp_dir / "checkpoint_metadata.json").write_text(
            json.dumps(asdict(metadata), indent=2),
            encoding="utf-8",
        )

    def _collect_error(self, local_error: str | None, trainer: Any) -> str | None:
        if not dist.is_initialized():
            return local_error
        group = getattr(trainer, "checkpoint_process_group", None)
        errors = [None] * dist.get_world_size(group)
        if group is None:
            dist.all_gather_object(errors, local_error)
        else:
            dist.all_gather_object(errors, local_error, group=group)
        return next((error for error in errors if error is not None), None)

    def _publish_checkpoint(
        self,
        temp_dir: Path,
        checkpoint_dir: Path,
        trainer: Any,
    ) -> tuple[str | None, Exception | None]:
        if trainer.config.rank != 0:
            return None, None
        try:
            (temp_dir / "_SUCCESS").write_text("", encoding="utf-8")
            temp_dir.replace(checkpoint_dir)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}", exc
        return None, None
