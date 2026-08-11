"""Wan Video+Action model loading and diffusion forward adapter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from distillation.model.autoregressive_types import AutoregressiveModelRequest
from distillation.schema import VADiffusionOutput, VAPair, VATimesteps
from wan_va.utils.scheduler import FlowMatchScheduler


def sigmas_for_timesteps(
    scheduler: FlowMatchScheduler,
    timesteps: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    scheduler_timesteps = scheduler.timesteps.to(timesteps.device).reshape(
        -1,
        *([1] * timesteps.ndim),
    )
    indices = (scheduler_timesteps - timesteps.unsqueeze(0)).abs().argmin(dim=0)
    return scheduler.sigmas.to(timesteps.device)[indices].to(dtype=dtype)


def broadcast_frame_values(
    values: torch.Tensor,
    sample: torch.Tensor,
) -> torch.Tensor:
    return values.reshape(
        values.shape[0],
        1,
        values.shape[1],
        *([1] * (sample.ndim - 3)),
    )
    
class WanDiffusionWrapper:
    """Load or borrow a Wan model and return its joint flow and x0 predictions.

    Model placement, trainability, activation checkpointing, and sharding are
    deliberately trainer-owned. The trainer assigns the processed module back
    to ``wrapper.model`` without changing this adapter's checkpoint ownership.
    """

    def __init__(
        self,
        *,
        config: Any | None = None,
        checkpoint_path: str | None = None,
        autoregressive: bool = True,
    ) -> None:
        self.model = self._load_model(
            checkpoint_path,
            config,
            autoregressive=autoregressive,
        )

        video_scheduler, action_scheduler = self._initialize_schedulers(config)
        self.video_scheduler = video_scheduler
        self.action_scheduler = action_scheduler

    @staticmethod
    def _load_model(
        checkpoint_path: str | Path,
        config: Any,
        *,
        autoregressive: bool,
    ) -> nn.Module:
        """Load a completed transformer export without post-processing it."""
        checkpoint_path = Path(checkpoint_path)
        metadata_path = checkpoint_path / "checkpoint_metadata.json"
        transformer_path = checkpoint_path / "transformer"
        required = (
            checkpoint_path / "_SUCCESS",
            metadata_path,
            transformer_path / "config.json",
            transformer_path / "diffusion_pytorch_model.safetensors",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Distillation initialization requires a completed transformer "
                "export; missing: " + ", ".join(missing)
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_architecture = (
            str(
                getattr(
                    config.distill,
                    "model_architecture",
                    "autoregressive_va_mot_v1",
                )
            )
            if autoregressive
            else "va_mot_v1"
        )
        if metadata.get("model_architecture") != expected_architecture:
            raise ValueError(
                "Checkpoint model architecture does not match distillation model: "
                f"checkpoint={metadata.get('model_architecture')!r}, "
                f"expected={expected_architecture!r}"
            )
        model_cls = WanDiffusionWrapper._model_class(autoregressive)
        model = model_cls.from_pretrained(
            transformer_path,
            torch_dtype=config.param_dtype,
        )
        masked_attn_backend = getattr(config, "masked_attn_backend", None)
        if masked_attn_backend is not None:
            model.masked_attn_backend = str(masked_attn_backend)
        return model

    @staticmethod
    def _model_class(autoregressive: bool):
        if autoregressive:
            from distillation.model.autoregressive_mot import (
                AutoregressiveVAMOTTransformer3DModel,
            )

            return AutoregressiveVAMOTTransformer3DModel

        from wan_va.modules.model_va_mot import VAMOTTransformer3DModel

        return VAMOTTransformer3DModel

    @staticmethod
    def _initialize_schedulers(
        config: Any,
    ) -> tuple[FlowMatchScheduler, FlowMatchScheduler]:
        """Create the independent training schedules owned by this wrapper."""
        # Stage 1/2: Build the video schedule with the video SNR warp.
        # extra_one_step=True keeps 1000 update states while retaining a clean
        # terminal transition in FlowMatchScheduler.step().
        video = FlowMatchScheduler(
            shift=float(config.snr_shift),
            sigma_min=0.0,
            extra_one_step=True,
        )
        # training=True also prepares timestep sampling weights used by native
        # MOT training; timesteps/sigmas themselves remain one-dimensional [S].
        video.set_timesteps(1000, training=True)

        # Stage 2/2: Build an independent action table because action can use a
        # different SNR shift even when both modalities have S=1000 states.
        action = FlowMatchScheduler(
            shift=float(config.action_snr_shift),
            sigma_min=0.0,
            extra_one_step=True,
        )
        action.set_timesteps(1000, training=True)
        return video, action

    @staticmethod
    def _convert_flow_pred_to_x0(
        scheduler,
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Convert ``flow=noise-x0`` to x0 using FP64 arithmetic.

        This follows the reference wrapper's conversion, adapted from flattened
        video samples to this repository's ``[B,C,F,...]`` V/A tensors.
        """
        # Stage 1/3: Preserve the public output dtype, but perform the affine
        # conversion in FP64 to reduce cancellation error near low noise levels.
        original_dtype = flow_pred.dtype
        flow_fp64 = flow_pred.to(dtype=torch.float64)
        xt_fp64 = xt.to(dtype=torch.float64)

        # Stage 2/3: Nearest scheduler lookup returns [B,F]. Broadcasting expands
        # it to [B,1,F,1,...] without copying values across channel/spatial axes.
        sigma = sigmas_for_timesteps(
            scheduler,
            timestep,
            dtype=torch.float64,
        )
        sigma = broadcast_frame_values(sigma, flow_fp64)

        # Stage 3/3: Shape is preserved: [B,C,F,...] -> [B,C,F,...].
        # Example: x_t=5, sigma=0.75, flow=4 gives x0=5-0.75*4=2.
        return (xt_fp64 - sigma * flow_fp64).to(dtype=original_dtype)

    @staticmethod
    def _frame_timesteps(
        timestep: float | int | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast a scalar/per-frame timestep to ``[B,F]`` for x0 conversion."""
        timestep = torch.as_tensor(timestep, dtype=torch.float32, device=sample.device)
        batch_size, _channels, frames = sample.shape[:3]
        if timestep.ndim == 0:
            return timestep.expand(batch_size, frames)
        if timestep.ndim == 1:
            if timestep.numel() == 1:
                return timestep.expand(batch_size, frames)
            if timestep.numel() == frames:
                return timestep[None].expand(batch_size, -1)
        if tuple(timestep.shape) != (batch_size, frames):
            raise ValueError(
                f"timesteps must broadcast to [{batch_size},{frames}], "
                f"got {tuple(timestep.shape)}"
            )
        return timestep

    def forward(
        self,
        input_dict: dict[str, Any],
        noisy: VAPair,
        timesteps: VATimesteps,
    ) -> VADiffusionOutput:
        # Stage 1/3: One backbone call predicts both modality flows. No detach or
        # no_grad is introduced here; the caller controls gradient construction.
        output = self.model(input_dict, mode="train")
        velocity = VAPair(
            video=output["latent_pred"],
            action=output["action_pred"],
        )

        # Stage 2/3: Convert each flow with its own scheduler and [B,F] table.
        # Both conversions preserve their input tensor shapes and output dtype.
        x0 = VAPair(
            video=self._convert_flow_pred_to_x0(
                self.video_scheduler,
                velocity.video,
                noisy.video,
                timesteps.video,
            ),
            action=self._convert_flow_pred_to_x0(
                self.action_scheduler,
                velocity.action,
                noisy.action,
                timesteps.action,
            ),
        )

        # Stage 3/3: Return flow and x0 together so DMD can train on flow while
        # generator/score comparisons consume the corresponding clean estimate.
        return VADiffusionOutput(velocity=velocity, x0=x0)

    def generate_video(
        self,
        noisy: torch.Tensor,
        timestep: float | int | torch.Tensor,
        frame_ids,
        stream_ids: torch.Tensor,
        *,
        cache,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one video block through the AR model and return clean x0."""
        timestep = self._frame_timesteps(timestep, noisy)
        request = AutoregressiveModelRequest(
            operation="predict_video",
            payload={
                "sample": noisy,
                "timestep": timestep,
                "frame_ids": frame_ids,
                "stream_ids": stream_ids,
                "cache": cache,
                "text_emb": text_emb,
            },
        )
        output = self.model(request, mode="self_rollout")
        return self._convert_flow_pred_to_x0(
            self.video_scheduler,
            output.prediction,
            noisy,
            timestep,
        )

    def generate_action(
        self,
        noisy: torch.Tensor,
        timestep: float | int | torch.Tensor,
        frame_ids,
        *,
        cache,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one action block through the AR model and return clean x0."""
        timestep = self._frame_timesteps(timestep, noisy)
        request = AutoregressiveModelRequest(
            operation="predict_action",
            payload={
                "sample": noisy,
                "timestep": timestep,
                "frame_ids": frame_ids,
                "cache": cache,
                "text_emb": text_emb,
            },
        )
        output = self.model(request, mode="self_rollout")
        return self._convert_flow_pred_to_x0(
            self.action_scheduler,
            output.prediction,
            noisy,
            timestep,
        )

    def commit_video(
        self,
        latents: torch.Tensor,
        frame_ids,
        stream_ids: torch.Tensor,
        *,
        cache,
        text_emb: torch.Tensor,
    ) -> None:
        """Commit clean video K/V into the incremental cache."""
        request = AutoregressiveModelRequest(
            operation="commit_video",
            payload={
                "latents": latents,
                "frame_ids": frame_ids,
                "stream_ids": stream_ids,
                "cache": cache,
                "text_emb": text_emb,
            },
        )
        self.model(request, mode="self_rollout")

    def commit_action(
        self,
        actions: torch.Tensor,
        frame_ids,
        *,
        cache,
        text_emb: torch.Tensor,
    ) -> None:
        """Commit clean action K/V into the incremental cache."""
        request = AutoregressiveModelRequest(
            operation="commit_action",
            payload={
                "actions": actions,
                "frame_ids": frame_ids,
                "cache": cache,
                "text_emb": text_emb,
            },
        )
        self.model(request, mode="self_rollout")

    def __call__(
        self,
        input_dict: dict[str, Any],
        noisy: VAPair,
        timesteps: VATimesteps,
    ) -> VADiffusionOutput:
        # Match nn.Module call ergonomics without registering model parameters on
        # this serialization-free adapter.
        return self.forward(input_dict, noisy, timesteps)
