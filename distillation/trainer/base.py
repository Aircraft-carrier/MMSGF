"""Simplified standalone distillation trainer base (data + common prep).

Deliberately does NOT inherit ``MOTTrainer``. The base owns device setup, MOT
data loading, VAE/text initialization, latent preprocessing, the microstep loop
and checkpoint save/load. Subclasses own schedulers, optimizers, model
initialization from paths (``_build_method_model``) and FSDP/AC wrapping
(``_wrap_method_models``), which hands wrapped models back to the method model
via ``attach_wrapped_models``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Literal

import numpy as np
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
from torch.utils.data import DataLoader
from safetensors.torch import save_file

from distillation.mask_profile import generation_profile_contract
from distillation.schema import TrainingStepResult
from wan_va.dataset.mot_dataset import validate_mot_batch_for_forward
from wan_va.modules.utils import WanVAEStreamingWrapper, load_vae
from wan_va.mot_spec import mot_spec_from_config
from wan_va.train_mot import (
    _apply_cfg_text_dropout,
    _build_mot_train_sampler,
    _configure_adamw_foreach,
    _mot_dataloader_kwargs,
    _move_to_device,
    _seed_mot_training,
    build_mot_param_groups,
    build_mot_train_dataset,
)
from wan_va.utils import warmup_constant_lambda


@dataclass(frozen=True, slots=True)
class OptimizationTarget:
    name: Literal["generator", "fake_score"]
    optimizer: torch.optim.Optimizer
    model: torch.nn.Module


class DistillationTrainerBase:
    """Common trainer preparation: data loading and preprocessing.

    Subclass contract:

    - ``_build_method_model(config)``: construct the method model from paths.
    - ``_wrap_method_models()``: FSDP/AC wrap and ``attach_wrapped_models``.
    - ``_optimization_target`` / ``_compute_training_step`` / ``_after_optimizer_step``.
    """

    method: str

    def __init__(self, config: Any) -> None:
        self.config = config
        self._resume_from = getattr(config.distill, "resume_from", None)
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.step = 0
        self.optimizer_step = 0
        self.gradient_accumulation_steps = int(
            getattr(config, "gradient_accumulation_steps", 1)
        )
        self.save_interval = int(config.save_interval)
        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._last_checkpoint_step = None
        self.train_vae = None
        self.empty_text_emb = None

        _seed_mot_training(int(getattr(config, "train_seed", 42)))
        self._build_dataloader(config)

        self.model = self._build_method_model(config)
        self._wrap_method_models()

    # Data loading ---------------------------------------------------------

    def _build_dataloader(self, config: Any) -> None:
        train_dataset = build_mot_train_dataset(config)
        train_sampler = _build_mot_train_sampler(train_dataset, config)
        self.train_loader = DataLoader(
            train_dataset,
            **_mot_dataloader_kwargs(config, train_sampler),
        )
        self.train_loader_iter = iter(self.train_loader)

    def _get_next_batch(self) -> dict[str, Any]:
        try:
            return next(self.train_loader_iter)
        except StopIteration:
            self.train_loader_iter = iter(self.train_loader)
            return next(self.train_loader_iter)

    # Model hooks ----------------------------------------------------------

    def _build_method_model(self, config: Any):
        raise NotImplementedError

    def _wrap_method_models(self) -> None:
        """FSDP/AC wrap and hand models back via ``attach_wrapped_models``."""

    def _trainable_model(self) -> torch.nn.Module:
        """Trainable model persisted under the checkpoint ``generator`` key."""
        return self.model.generator.model

    def _optimization_target(self) -> OptimizationTarget:
        raise NotImplementedError

    def _compute_training_step(
        self,
        batch: dict,
        base_input: dict,
        empty_text_emb: torch.Tensor,
        target: OptimizationTarget,
    ) -> TrainingStepResult:
        return self.model.compute_step(
            batch,
            base_input=base_input,
            empty_text_emb=empty_text_emb,
        )

    def _after_optimizer_step(self, target: OptimizationTarget) -> None:
        return None

    # Optimizer ------------------------------------------------------------

    def _build_optimizer(
        self,
        config: Any,
        model: torch.nn.Module,
    ) -> torch.optim.Optimizer:
        """Build a MOT-style AdamW optimizer for one trainable model.

        Subclasses may override this to customize lr/groups per role; the SGF
        trainer reuses it for both the generator and the fake-score.
        """
        param_groups = build_mot_param_groups(
            model,
            base_lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
        )
        _configure_adamw_foreach(optimizer)
        return optimizer

    def _build_lr_scheduler(
        self,
        config: Any,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Build the shared warmup-then-constant LR scheduler."""
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_constant_lambda(
                step,
                warmup_steps=config.warmup_steps,
            ),
        )

    # Preprocessing --------------------------------------------------------

    def convert_input_format(self, input_dict):
        return _move_to_device(
            input_dict,
            self.device,
            non_blocking=(self.device.type == "cuda"),
        )

    def _get_train_vae(self):
        if self.train_vae is None:
            self.train_vae = load_vae(
                str(Path(self.config.wan22_pretrained_model_name_or_path) / "vae"),
                torch_dtype=self.dtype,
                torch_device=self.device,
            ).eval()
            self.train_vae.requires_grad_(False)
        return self.train_vae

    def _get_empty_text_emb(self):
        if self.empty_text_emb is None:
            empty_path = Path(getattr(self.config, "empty_emb_path"))
            if not empty_path.is_file():
                raise FileNotFoundError(empty_path)
            self.empty_text_emb = torch.load(
                empty_path,
                map_location="cpu",
                weights_only=False,
            )
        return self.empty_text_emb

    @torch.no_grad()
    def _encode_one_view_latent(self, frames: torch.Tensor) -> torch.Tensor:
        spec = mot_spec_from_config(self.config)
        if frames.shape[0] != spec.vae_input_frame_count:
            raise ValueError(
                f"MOT VAE input must have {spec.vae_input_frame_count} frames, "
                f"got {frames.shape[0]}"
            )
        vae = self._get_train_vae()
        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype
        wrapper = WanVAEStreamingWrapper(vae)
        video = frames.permute(1, 0, 2, 3)[None].to(
            device=vae_device,
            dtype=vae_dtype,
        ) * 2.0 - 1.0
        chunks = [wrapper.encode_chunk(video[:, :, :1])]
        for latent_idx in range(1, spec.latent_frames_per_action_chunk_per_view):
            start = 1 + spec.vae_temporal_factor * (latent_idx - 1)
            end = start + spec.vae_temporal_factor
            chunks.append(wrapper.encode_chunk(video[:, :, start:end]))
        enc = torch.cat(chunks, dim=2)
        mu, _logvar = torch.chunk(enc, 2, dim=1)
        mean = torch.tensor(
            vae.config.latents_mean,
            device=mu.device,
            dtype=mu.dtype,
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            vae.config.latents_std,
            device=mu.device,
            dtype=mu.dtype,
        ).view(1, -1, 1, 1, 1)
        return ((mu - mean) / std)[0].to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _encode_vae_rgb(self, vae_rgb: torch.Tensor) -> torch.Tensor:
        batch_size, _time, views = vae_rgb.shape[:3]
        per_batch = []
        for batch_idx in range(batch_size):
            per_view = [
                self._encode_one_view_latent(vae_rgb[batch_idx, :, view_idx])
                for view_idx in range(views)
            ]
            per_batch.append(torch.stack(per_view, dim=2))
        return torch.stack(per_batch, dim=0)

    @torch.no_grad()
    def _materialize_batch_latents(self, batch_dict):
        if "latents" in batch_dict:
            return batch_dict
        if (
            "vae_rgb_history" not in batch_dict
            or "vae_rgb_target" not in batch_dict
        ):
            raise KeyError(
                "MOT batch must contain either latents or vae_rgb_history/target"
            )
        out = dict(batch_dict)
        history = self._encode_vae_rgb(batch_dict["vae_rgb_history"])
        target = self._encode_vae_rgb(batch_dict["vae_rgb_target"])
        out["latents"] = torch.cat([history, target], dim=2)
        return out

    def _prepare_joint_input_dict(self, batch_dict: dict, *, add_noise: bool = True):
        # Distillation installs its own noisy trajectories via
        # ``replace_va_streams``, so only the clean-field layout is supported.
        if add_noise:
            raise NotImplementedError(
                "simplified distillation trainer only prepares add_noise=False inputs"
            )
        spec = mot_spec_from_config(self.config)
        video_latent_loss_mask = batch_dict["video_latent_loss_mask"].to(
            device=batch_dict["latents"].device,
            dtype=torch.bool,
        )
        video_latent_valid_mask = batch_dict["video_latent_valid_mask"].to(
            device=batch_dict["latents"].device,
            dtype=torch.bool,
        )
        action_loss_mask = batch_dict["action_loss_mask"].to(
            device=batch_dict["actions"].device,
            dtype=torch.bool,
        )
        action_valid_mask = batch_dict["action_valid_mask"].to(
            device=batch_dict["actions"].device,
            dtype=torch.bool,
        )
        text_emb = _apply_cfg_text_dropout(
            batch_dict["text_emb"],
            batch_dict.get("empty_text_emb")
            if batch_dict.get("empty_text_emb") is not None
            else self._get_empty_text_emb(),
            float(getattr(self.config, "cfg_prob", 0.0)),
            training=True,
        )
        latent_dict = {
            "latent": batch_dict["latents"],
            "text_emb": text_emb,
            "video_latent_loss_mask": video_latent_loss_mask,
            "video_latent_valid_mask": video_latent_valid_mask,
        }
        action_dict = {
            "latent": batch_dict["actions"],
            "text_emb": text_emb,
            "action_loss_mask": action_loss_mask,
            "action_valid_mask": action_valid_mask,
        }
        validate_mot_batch_for_forward(
            batch_dict,
            action_sequence_length=spec.action_sequence_length,
        )
        return {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "stream_ids": batch_dict["stream_ids"],
            "chunk_size": spec.latent_frames_per_action_chunk_per_view,
            "window_size": spec.attention_window_size,
        }

    # Step loop ------------------------------------------------------------

    def _distributed_any(self, value: bool) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return value
        tensor = torch.tensor(int(bool(value)), device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(tensor.item())

    def _train_step(self, batch: dict, batch_idx: int) -> dict[str, Any]:
        batch = self.convert_input_format(batch)
        batch = self._materialize_batch_latents(batch)
        base_input = self._prepare_joint_input_dict(batch, add_noise=False)
        empty_text_emb = self._get_empty_text_emb()
        target = self._optimization_target()
        result = self._compute_training_step(
            batch,
            base_input,
            empty_text_emb,
            target,
        )
        loss = result.loss
        metrics = result.metrics
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        out = {
            "loss": loss,
            **metrics,
            "optimizer_step_event": False,
            "skipped_step": False,
            "should_log": True,
        }
        if bool(self._distributed_any(not torch.isfinite(loss).all())):
            target.optimizer.zero_grad(set_to_none=True)
            out["skipped_step"] = True
            return out
        (loss / float(self.gradient_accumulation_steps)).backward()
        if should_sync:
            torch.nn.utils.clip_grad_norm_(
                target.model.parameters(),
                self.config.distill.max_grad_norm,
            )
            target.optimizer.step()
            self._after_optimizer_step(target)
            target.optimizer.zero_grad(set_to_none=True)
            self.optimizer_step += 1
            out["optimizer_step_event"] = True
        else:
            out["should_log"] = False
        return out

    def train(self) -> None:
        total = int(self.config.num_steps)
        microstep = int(self.step)
        while self.optimizer_step < total:
            result = self._train_step(
                self._get_next_batch(),
                microstep % self.gradient_accumulation_steps,
            )
            microstep += 1
            self.step = microstep
            if (
                result["optimizer_step_event"]
                and self.optimizer_step > 0
                and self.optimizer_step % self.save_interval == 0
                and self._last_checkpoint_step != self.optimizer_step
            ):
                self.save_checkpoint()
        if self._last_checkpoint_step != self.optimizer_step:
            self.save_checkpoint()

    # Checkpoint -----------------------------------------------------------

    def _full_model_state(self, model: torch.nn.Module) -> dict[str, Any]:
        return get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    def _restore_full_model_state(
        self,
        model: torch.nn.Module,
        state: dict[str, Any],
    ) -> None:
        set_model_state_dict(
            model,
            state,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    @staticmethod
    def _checkpoint_options() -> StateDictOptions:
        return StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
            strict=True,
        )

    def _checkpoint_state_dict(self) -> dict[str, Any]:
        model, optimizer = get_state_dict(
            self._trainable_model(),
            self.optimizer,
            options=self._checkpoint_options(),
        )
        state = {"generator": model, "optimizer": optimizer}
        self._extra_save_state(state)
        return state

    def _restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        set_state_dict(
            self._trainable_model(),
            self.optimizer,
            model_state_dict=state["generator"],
            optim_state_dict=state["optimizer"],
            options=self._checkpoint_options(),
        )
        self._restore_extra_state(state)
        _configure_adamw_foreach(self.optimizer)

    def _export_model(self) -> torch.nn.Module:
        return self._trainable_model()

    def _write_transformer_export(
        self,
        checkpoint_dir: Path,
        model: torch.nn.Module,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        transformer_dir = checkpoint_dir / "transformer"
        transformer_dir.mkdir(parents=True)
        config_dict = dict(model.config)
        config_dict.pop("_name_or_path", None)
        config_dict["generation_shape"] = dict(
            self.config.distill.generation_shape
        )
        (transformer_dir / "config.json").write_text(
            json.dumps(config_dict, indent=2) + "\n",
            encoding="utf-8",
        )
        save_file(
            {name: value.contiguous() for name, value in state_dict.items()},
            transformer_dir / "diffusion_pytorch_model.safetensors",
        )

    def _write_checkpoint_metadata(self, checkpoint_dir: Path) -> None:
        metadata = {
            "format_version": 3,
            "checkpoint_type": "mot_training",
            "model_architecture": str(self.config.distill.model_architecture),
            "has_full_state": True,
            "distill_method": self.method,
            "exported_model": (
                "ema_student"
                if self.method == "consistency_distillation"
                else "student"
            ),
            "step": int(self.step),
            "optimizer_step": int(self.optimizer_step),
            "generation_profile": generation_profile_contract(
                self.config.distill.generation_shape
            ),
        }
        (checkpoint_dir / "checkpoint_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _capture_rng_state(self) -> dict[str, Any]:
        cuda_state = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state(self.device)
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": cuda_state,
        }

    def _collect_rng_states(self) -> list[dict[str, Any]]:
        local_state = self._capture_rng_state()
        if not (dist.is_available() and dist.is_initialized()):
            return [local_state]
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local_state)
        return states

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if state["torch_cuda"] is not None:
            torch.cuda.set_rng_state(state["torch_cuda"], self.device)

    def _set_sampler_resume_offset(self) -> None:
        loader = getattr(self, "train_loader", None)
        sampler = getattr(loader, "batch_sampler", None)
        if hasattr(sampler, "set_start_step"):
            sampler.set_start_step(self.step)
            self.train_loader_iter = iter(loader)

    def _extra_save_state(self, state: dict[str, Any]) -> None:
        """Subclass hook for extra state (e.g. fake-score + its optimizer)."""

    def _restore_extra_state(self, state: dict[str, Any]) -> None:
        """Subclass hook for extra state (e.g. fake-score + its optimizer)."""

    def save_checkpoint(self) -> Path:
        checkpoint_dir = self.save_dir / f"checkpoint_step_{self.optimizer_step:08d}"
        export_model = self._export_model()
        dcp_state = self._checkpoint_state_dict()
        temp_dir = self.save_dir / f".{checkpoint_dir.name}.tmp"
        if int(getattr(self.config, "rank", 0)) == 0:
            if checkpoint_dir.exists():
                raise FileExistsError(checkpoint_dir)
            if temp_dir.exists():
                raise FileExistsError(temp_dir)
            temp_dir.mkdir(parents=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        dcp.save(
            dcp_state,
            checkpoint_id=temp_dir / "distributed_state",
        )
        export_state = self._full_model_state(export_model)
        training_state = {
            "step": self.step,
            "optimizer_step": self.optimizer_step,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "rng_states_by_rank": self._collect_rng_states(),
            "method_state_dict": self.get_method_state_dict(),
        }
        if int(getattr(self.config, "rank", 0)) == 0:
            torch.save(training_state, temp_dir / "training_state.pt")
            self._write_transformer_export(temp_dir, export_model, export_state)
            self._write_checkpoint_metadata(temp_dir)
            (temp_dir / "_SUCCESS").write_text("", encoding="utf-8")
            temp_dir.replace(checkpoint_dir)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self._last_checkpoint_step = self.optimizer_step
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_root: str | Path) -> None:
        checkpoint_root = Path(checkpoint_root)
        metadata = json.loads(
            (checkpoint_root / "checkpoint_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        if metadata.get("distill_method") != self.method:
            raise ValueError(
                f"Checkpoint method {metadata.get('distill_method')} does not "
                f"match current method {self.method}"
            )
        expected_profile = generation_profile_contract(
            self.config.distill.generation_shape
        )
        if metadata.get("generation_profile") != expected_profile:
            raise ValueError(
                "Checkpoint generation profile does not match current config"
            )
        training_state = torch.load(
            checkpoint_root / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        dcp_state = self._checkpoint_state_dict()
        dcp.load(
            dcp_state,
            checkpoint_id=checkpoint_root / "distributed_state",
        )
        self._restore_checkpoint_state(dcp_state)
        self.step = int(training_state["step"])
        self.optimizer_step = int(training_state["optimizer_step"])
        self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
        self.load_method_state_dict(training_state["method_state_dict"])
        rank = int(getattr(self.config, "rank", 0))
        self._restore_rng_state(training_state["rng_states_by_rank"][rank])
        self._set_sampler_resume_offset()
        self._last_checkpoint_step = self.optimizer_step

    def get_method_state_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "method_model_state_dict": self.model.state_dict(),
        }

    def load_method_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("method") != self.method:
            raise ValueError(
                f"Checkpoint method {state.get('method')} does not match "
                f"current method {self.method}"
            )
        if "method_model_state_dict" in state:
            self.model.load_state_dict(state["method_model_state_dict"])
