"""Cached bidirectional VA-MOT inference for online RoboTwin observations."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from distillation.eval.infer_pipeline import (
    STREAM_IDS,
    OnlineMOTWindowBuilder,
    StreamingVAECodec,
    TextEmbedder,
)
from distillation.eval.protocol import OnlineObservation
from distillation.model.autoregressive_mot import AutoregressiveVAMOTTransformer3DModel
from distillation.pipeline.cache import KVCache
from wan_va.dataset.mot_dataset import relative_20d_to_absolute_actions
from wan_va.modules.utils import load_vae
from wan_va.mot_spec import MOTWindowSpec, mot_spec_from_config
from wan_va.utils.scheduler import FlowMatchScheduler


CHECKPOINT_ARCHITECTURE = "va_mot_v1"
TRANSFORMER_WEIGHTS = "diffusion_pytorch_model.safetensors"
TARGET_FRAME_IDS = (4, 5, 6, 7)
ACTION_TARGET_FRAME_IDS = (5, 6, 7)


class BidirectionalMOTInferencePipeline:
    def __init__(
        self,
        *,
        model,
        codec: StreamingVAECodec,
        text_embedder,
        empty_text_emb: torch.Tensor,
        norm_stats_by_task: dict[str, Any],
        spec: MOTWindowSpec,
        device: torch.device,
        dtype: torch.dtype,
        video_num_steps: int,
        action_num_steps: int,
        guidance_scale: float,
        video_snr_shift: float,
        action_snr_shift: float,
        execution_action_count: int = 48,
    ) -> None:
        self.model = model
        self.codec = codec
        self.text_embedder = text_embedder
        self.empty_text_emb = empty_text_emb
        self.norm_stats_by_task = norm_stats_by_task
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.video_num_steps = int(video_num_steps)
        self.action_num_steps = int(action_num_steps)
        self.execution_action_count = int(execution_action_count)
        if not 1 <= self.execution_action_count <= 48:
            raise ValueError("execution_action_count must be between 1 and 48")
        self.guidance_scale = float(guidance_scale)
        self.video_scheduler = FlowMatchScheduler(
            shift=video_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.builder: OnlineMOTWindowBuilder | None = None
        self.text_emb: torch.Tensor | None = None
        self.task_key: str | None = None
        self.seed = 0

    def reset(self, *, task_name: str, instruction: str, seed: int) -> None:
        prefix = f"{task_name}-"
        matches = [key for key in self.norm_stats_by_task if key.startswith(prefix)]
        if len(matches) != 1:
            raise KeyError(
                f"RoboTwin task {task_name!r} matched {len(matches)} normalization keys"
            )
        self.task_key = matches[0]
        self.builder = OnlineMOTWindowBuilder(self.norm_stats_by_task[self.task_key])
        self.text_emb = self.text_embedder(instruction).to(self.device)
        self.seed = int(seed)

    def _assemble_batch(
        self, window, history: torch.Tensor, anchor: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if tuple(history.shape[2:4]) != (4, 3) or tuple(anchor.shape[2:4]) != (1, 3):
            raise ValueError(
                "online VAE output must contain four history and one anchor latent"
            )
        generated = torch.zeros(
            *anchor.shape[:2], 3, *anchor.shape[3:], device=self.device, dtype=self.dtype
        )
        latents = torch.cat([history, anchor, generated], dim=2)
        actions = torch.zeros(
            (1, 20, 8, 16, 1), device=self.device, dtype=self.dtype
        )
        actions[:, :, :4] = window.history_actions.to(self.device, self.dtype)
        action_loss_mask = torch.zeros_like(actions, dtype=torch.bool)
        action_loss_mask[:, :, 5:8] = True
        action_valid_mask = torch.zeros_like(actions, dtype=torch.bool)
        action_valid_mask[:, :, :4] = window.action_valid.to(self.device)
        action_valid_mask[:, :, 5:8] = True
        video_valid = torch.cat(
            [
                window.video_valid.to(self.device),
                torch.ones((1, 4), device=self.device, dtype=torch.bool),
            ],
            dim=1,
        )
        video_loss = torch.zeros_like(video_valid)
        video_loss[:, 5:8] = True
        return {
            "latents": latents,
            "actions": actions,
            "action_loss_mask": action_loss_mask,
            "action_valid_mask": action_valid_mask,
            "video_latent_valid_mask": video_valid,
            "video_latent_loss_mask": video_loss,
            "stream_ids": STREAM_IDS.to(self.device),
            "text_emb": self.text_emb,
        }

    def _empty_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        empty = self.empty_text_emb.to(device=self.device, dtype=text_emb.dtype)
        if empty.ndim == text_emb.ndim - 1:
            empty = empty.unsqueeze(0)
        if empty.shape != text_emb.shape:
            raise ValueError(
                f"empty text embedding {tuple(empty.shape)} does not match {tuple(text_emb.shape)}"
            )
        return empty

    def _history_cache(
        self, batch: dict[str, torch.Tensor], text_emb: torch.Tensor
    ) -> KVCache:
        cache = KVCache()
        self.model.commit_video(
            batch["latents"][:, :, :4],
            frame_ids=range(4),
            stream_ids=batch["stream_ids"],
            text_emb=text_emb,
            cache=cache,
            token_valid_mask=batch["video_latent_valid_mask"][:, :4],
        )
        self.model.commit_action(
            batch["actions"][:, :, :4],
            frame_ids=range(4),
            text_emb=text_emb,
            cache=cache,
            token_valid_mask=batch["action_valid_mask"][:, :, :4],
        )
        return cache

    def _generate_video(
        self,
        batch: dict[str, torch.Tensor],
        *,
        text_emb: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, KVCache]:
        cache = self._history_cache(batch, text_emb)
        empty_text = self._empty_text(text_emb)
        unconditional_cache = (
            self._history_cache(batch, empty_text) if self.guidance_scale != 1.0 else None
        )
        target_template = batch["latents"][:, :, 4:8]
        sample = torch.randn(
            tuple(target_template[:, :, 1:].shape),
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        self.video_scheduler.set_timesteps(self.video_num_steps)
        for timestep in self.video_scheduler.timesteps.to(self.device):
            target = target_template.clone()
            target[:, :, 1:] = sample
            timesteps = torch.zeros((1, 4), device=self.device)
            timesteps[:, 1:] = timestep
            conditional = self.model.predict_video(
                target,
                frame_ids=TARGET_FRAME_IDS,
                timestep=timesteps,
                stream_ids=batch["stream_ids"],
                text_emb=text_emb,
                cache=cache,
            )
            prediction = conditional
            if unconditional_cache is not None:
                unconditional = self.model.predict_video(
                    target,
                    frame_ids=TARGET_FRAME_IDS,
                    timestep=timesteps,
                    stream_ids=batch["stream_ids"],
                    text_emb=empty_text,
                    cache=unconditional_cache,
                )
                prediction = unconditional + self.guidance_scale * (
                    conditional - unconditional
                )
            sample = self.video_scheduler.step(
                prediction[:, :, 1:], timestep, sample
            )
        predicted = target_template.clone()
        predicted[:, :, 1:] = sample
        self.model.commit_video(
            predicted,
            frame_ids=TARGET_FRAME_IDS,
            stream_ids=batch["stream_ids"],
            text_emb=text_emb,
            cache=cache,
        )
        cache.assert_no_transactions()
        if unconditional_cache is not None:
            unconditional_cache.assert_no_transactions()
        return predicted, cache

    def _generate_action(
        self,
        *,
        cache: KVCache,
        text_emb: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        sample = torch.randn(
            (1, 20, 3, 16, 1),
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        self.action_scheduler.set_timesteps(self.action_num_steps)
        for timestep in self.action_scheduler.timesteps.to(self.device):
            prediction = self.model.predict_action(
                sample,
                frame_ids=ACTION_TARGET_FRAME_IDS,
                timestep=timestep,
                text_emb=text_emb,
                cache=cache,
            )
            sample = self.action_scheduler.step(prediction, timestep, sample)
        cache.assert_no_transactions()
        return sample

    # 阶段 1/2：一个训练窗口有 8 个 latent 时间位置
    #
    # latent frame:
    #             0    1    2    3  |  4    5    6    7
    #             -----history----- | anchor + future target
    #
    # 视频：
    #   0:4 = 4 个历史视频 latent
    #   4   = 当前时刻 anchor
    #   5:8 = 3 个未来视频 latent
    #
    # 动作：
    #   1:4 = 有效历史动作
    #   4   = anchor 对齐位置，不预测动作
    #   5:8 = 未来动作预测目标
    # 阶段 2/2：简化到“一帧一个 token”后的可见集合
    #
    # 未来 noisy video token NV5 可以看到：
    # visible_to_NV5 = [
    #     "NV4", "NV5", "NV6", "NV7",  # 整个 noisy target video chunk
    #     "CV0", "CV1", "CV2", "CV3",  # 历史 clean video
    #     "CA1", "CA2", "CA3",         # 历史 clean action
    # ]

    # # 未来 noisy action token NA5 可以看到：
    # visible_to_NA5 = [
    #     "CV0", "CV1", "CV2", "CV3",
    #     "CV4", "CV5", "CV6", "CV7",  # 完整 clean video
    #     "NA5", "NA6", "NA7",         # 整个 noisy action chunk
    #     "CA1", "CA2", "CA3",         # 历史 clean action
    # ]
    @torch.no_grad()
    def infer(
        self,
        *,
        observations: list[OnlineObservation],
        executed_actions: list[np.ndarray],
        request_id: int,
        return_video: bool,
    ) -> dict[str, Any]:
        if self.builder is None or self.text_emb is None or self.task_key is None:
            raise RuntimeError("reset must be called before inference")
        started = time.perf_counter()
        candidate = self.builder.clone()
        candidate.append(observations, executed_actions)
        window = candidate.build()
        window_ms = (time.perf_counter() - started) * 1000

        vae_started = time.perf_counter()
        history = self.codec.encode_history(window.history_rgb).to(self.device, self.dtype)
        anchor = self.codec.encode_anchor(window.anchor_rgb).to(self.device, self.dtype)
        batch = self._assemble_batch(window, history, anchor)
        vae_ms = (time.perf_counter() - vae_started) * 1000

        generator = torch.Generator(device=self.device).manual_seed(
            self.seed + int(request_id)
        )
        video_started = time.perf_counter()
        predicted_video, cache = self._generate_video(
            batch, text_emb=self.text_emb, generator=generator
        )
        video_ms = (time.perf_counter() - video_started) * 1000
        action_started = time.perf_counter()
        predicted_action = self._generate_action(
            cache=cache, text_emb=self.text_emb, generator=generator
        )
        action_ms = (time.perf_counter() - action_started) * 1000

        normalized = (
            predicted_action[0, :, :, :, 0]
            .permute(1, 2, 0)
            .reshape(48, 20)
            .float()
            .cpu()
            .numpy()
        )
        stats = self.norm_stats_by_task[self.task_key]
        q01 = np.asarray(stats["q01"], dtype=np.float32).reshape(20)
        q99 = np.asarray(stats["q99"], dtype=np.float32).reshape(20)
        relative = (normalized + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01
        references = np.broadcast_to(window.anchor_state, (48, 16))
        absolute = relative_20d_to_absolute_actions(references, relative)
        decoded = self.codec.decode_video(predicted_video) if return_video else None
        response = {
            "observation_step": candidate.last_step,
            "actions": absolute[: self.execution_action_count].astype(np.float32).tolist(),
            "action_type": "ee",
            "predicted_video": decoded,
            "timings_ms": {
                "window": window_ms,
                "vae_encode": vae_ms,
                "video_diffusion": video_ms,
                "action_diffusion": action_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        }
        self.builder = candidate
        return response


def _load_spec(mot_config: dict[str, Any]) -> MOTWindowSpec:
    config = SimpleNamespace(
        action_chunk_size=int(mot_config["action_chunk_size"]),
        video_downsample_ratio=int(mot_config["video_downsample_ratio"]),
        vae_temporal_factor=int(mot_config.get("vae_temporal_factor", 4)),
    )
    spec = mot_spec_from_config(config)
    observed = (
        spec.history_latent_frames,
        spec.target_latent_frames,
        spec.action_per_frame,
    )
    if observed != (4, 4, 16):
        raise ValueError(
            "bidirectional evaluation requires history=4, target=4, action_per_frame=16"
        )
    return spec


def load_pipeline(
    *,
    checkpoint_root: str | Path,
    dataset_root: str | Path,
    model_root: str | Path,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    video_num_steps: int = 25,
    action_num_steps: int = 50,
    guidance_scale: float = 5.0,
    video_snr_shift: float = 5.0,
    action_snr_shift: float = 1.0,
    execution_action_count: int = 48,
) -> BidirectionalMOTInferencePipeline:
    checkpoint_root = Path(checkpoint_root)
    metadata = json.loads(
        (checkpoint_root / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("model_architecture") != CHECKPOINT_ARCHITECTURE:
        raise ValueError("evaluation requires a va_mot_v1 checkpoint")
    transformer_root = checkpoint_root / "transformer"
    for path in (
        transformer_root / "config.json",
        transformer_root / TRANSFORMER_WEIGHTS,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset_root = Path(dataset_root)
    mot_config = json.loads(
        (dataset_root / "meta" / "mot_config.json").read_text(encoding="utf-8")
    )
    spec = _load_spec(mot_config)
    model_root = Path(model_root)
    for directory in ("vae", "tokenizer", "text_encoder"):
        path = model_root / directory
        if not path.is_dir():
            raise FileNotFoundError(path)

    torch_device = torch.device(device)
    # The cached runtime is parameter-compatible with VA-MOT and supplies the
    # commit/predict operations reused by distillation eval.  Checkpoint identity
    # remains fixed-window and was validated above.
    model = AutoregressiveVAMOTTransformer3DModel.from_pretrained(
        transformer_root, torch_dtype=dtype
    ).to(torch_device).eval().requires_grad_(False)
    vae = load_vae(model_root / "vae", torch_dtype=dtype, torch_device=torch_device)
    vae.eval().requires_grad_(False)
    empty_text_emb = torch.load(
        dataset_root / "empty_emb.pt", map_location="cpu", weights_only=False
    )
    return BidirectionalMOTInferencePipeline(
        model=model,
        codec=StreamingVAECodec(vae, device=torch_device, dtype=dtype),
        text_embedder=TextEmbedder(model_root, torch_device, dtype),
        empty_text_emb=empty_text_emb,
        norm_stats_by_task=mot_config["norm_stats_by_task"],
        spec=spec,
        device=torch_device,
        dtype=dtype,
        video_num_steps=video_num_steps,
        action_num_steps=action_num_steps,
        guidance_scale=guidance_scale,
        video_snr_shift=video_snr_shift,
        action_snr_shift=action_snr_shift,
        execution_action_count=execution_action_count,
    )
