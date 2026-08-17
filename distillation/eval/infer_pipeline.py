"""Online window construction and native autoregressive MOT Euler inference."""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from distillation.model.autoregressive_mot import (
    AutoregressiveVAMOTTransformer3DModel,
)
from distillation.pipeline.cache import KVCache
from wan_va.dataset.mot_dataset import (
    _latent_valid_mask_from_sampled_frames,
    absolute_actions_to_relative_20d,
    relative_20d_to_absolute_actions,
)
from wan_va.modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)
from wan_va.utils.scheduler import FlowMatchScheduler

from .protocol import CAMERA_KEYS, OnlineObservation, encode_jpeg


HISTORY_RAW_FRAMES = 13
HISTORY_LATENT_FRAMES = 4
ACTION_TOKENS_PER_FRAME = 16
ACTION_HISTORY = 48
STREAM_IDS = torch.tensor([[1, 0, 2]], dtype=torch.long)
MOT_IMAGE_SIZE = (480, 640)


@dataclass(frozen=True, slots=True)
class OnlineWindow:
    # [13, 3, 3, H, W]
    # 13 个历史时刻，3 个相机，RGB 三通道
    history_rgb: torch.Tensor

    # [1, 3, 3, H, W]
    # 当前时刻的 anchor 图像
    anchor_rgb: torch.Tensor

    # 历史视频 latent 的有效位置
    video_valid: torch.Tensor

    # [1, 20, 4, 16, 1]
    # 历史动作，以 20D latent action 表示
    history_actions: torch.Tensor

    # 历史动作的有效位置
    action_valid: torch.Tensor

    # 当前 anchor 对应的真实机器人状态
    anchor_state: np.ndarray


class OnlineMOTWindowBuilder:
    """Keep raw episode data and reproduce MotTrainData history packing."""

    def __init__(self, norm_stats: dict[str, Any]) -> None:
        # q01/q99 用于把 20D action latent 映射到模型常用的 [-1, 1] 范围
        self.q01 = np.asarray(norm_stats["q01"], dtype=np.float32).reshape(1, 1, 20)
        self.q99 = np.asarray(norm_stats["q99"], dtype=np.float32).reshape(1, 1, 20)
        self.observations: deque[OnlineObservation] = deque(maxlen=50)
        self.actions: deque[tuple[int, np.ndarray]] = deque(maxlen=48)

    def clone(self) -> "OnlineMOTWindowBuilder":
        clone = OnlineMOTWindowBuilder(
            {"q01": self.q01.reshape(-1), "q99": self.q99.reshape(-1)}
        )
        clone.observations.extend(self.observations)
        clone.actions.extend((step, action.copy()) for step, action in self.actions)
        return clone

    @property
    def last_step(self) -> int:
        return -1 if not self.observations else int(self.observations[-1].step)

    def append(
        self,
        observations: list[OnlineObservation],
        executed_actions: list[np.ndarray],
    ) -> None:
        # 阶段 1/3：处理初始观测
        if not self.observations:
            # 第一次调用只能给 step=0 的一张观测
            if len(observations) != 1 or executed_actions:
                raise ValueError("the first request requires one observation and no actions")
            if observations[0].step != 0:
                raise ValueError("the first observation step must be 0")
            self.observations.append(observations[0])
            return
        # 阶段 2/3：处理后续观测和已经执行的动作
        if len(observations) != len(executed_actions) or not observations:
            raise ValueError("new observations and executed actions must be non-empty pairs")
        expected = self.last_step + 1
        for observation, action in zip(observations, executed_actions):
            if observation.step != expected:
                raise ValueError(
                    f"expected observation step {expected}, got {observation.step}"
                )
            absolute = np.asarray(action, dtype=np.float32)
            if absolute.shape != (16,) or not np.isfinite(absolute).all():
                raise ValueError("each executed action must contain 16 finite values")
            # 到达 observation step=expected 之前执行的是 action step=expected-1
            self.actions.append((expected - 1, absolute.copy()))
            self.observations.append(observation)
            expected += 1

    def _observation_map(self) -> dict[int, OnlineObservation]:
        # deque -> {时间步: observation}
        return {observation.step: observation for observation in self.observations}

    @staticmethod
    def _rgb(observation: OnlineObservation) -> np.ndarray:
        # 按 CAMERA_KEYS 取出三个相机图像
        #
        # 每张 observation.images[key] 假设为 [H, W, 3]
        # stack 后得到 [3, H, W, 3]
        return np.stack([observation.images[key] for key in CAMERA_KEYS], axis=0)

    def build(self) -> OnlineWindow:
        if not self.observations:
            raise ValueError("cannot build an online window without observations")
        current = self.last_step
        observations = self._observation_map()
        first = min(observations)
        # 阶段 1/4：采样历史视频帧
        #
        # 当前时间为 current 时：
        # history_ids = [current-49, current-45, ..., current-1]
        #
        # 一共 13 帧，间隔为 4
        history_ids = [current - 49 + 4 * index for index in range(HISTORY_RAW_FRAMES)]
        # 如果历史不够长，就用最早观测重复填充
        padded_ids = [max(first, frame_id) for frame_id in history_ids]
        if any(frame_id not in observations for frame_id in padded_ids):
            raise ValueError("ring buffer does not cover the requested history images")
        # [13, 3, H, W, 3]
        history_rgb = np.stack([self._rgb(observations[index]) for index in padded_ids])
        # [1 , 3, H, W, 3]
        anchor_rgb = self._rgb(observations[current])[None]
        # 标记哪些 history frame 是真实存在的，
        # 哪些是因为历史不足而被 padding 的
        raw_valid = torch.tensor(
            [frame_id >= first for frame_id in history_ids], dtype=torch.bool
        )
        # 把原始 frame 的有效性转换成 latent frame 的有效性
        # [B, T] -> [1, 4]
        video_valid = _latent_valid_mask_from_sampled_frames(
            raw_valid,
            latent_frames_per_action_chunk_per_view=HISTORY_LATENT_FRAMES,
        )[None]
        
        # 阶段 2/4：准备动作历史
        #
        # 4 个 latent frame × 每帧 16 个 action token × 20D
        # [T, Na, D] = [4, 16, 20]
        packed = np.zeros((HISTORY_LATENT_FRAMES, ACTION_TOKENS_PER_FRAME, 20), np.float32)
        valid = np.zeros_like(packed, dtype=bool)
        action_map = {step: action for step, action in self.actions}
        # 只使用最近 48 个动作
        start = current - ACTION_HISTORY
        # relative action 需要一个参考状态
        reference_step = max(0, start)
        if reference_step not in observations:
            raise ValueError("ring buffer does not contain the action reference state")
        reference = observations[reference_step].state[None]
        # 阶段 3/4：把动作放入对应的 latent frame/token
        for action_offset in range(ACTION_HISTORY):
            action_step = start + action_offset
            # 某些动作可能因为 episode 开头而不存在
            if action_step not in action_map:
                continue
            latent_index = 1 + action_offset // ACTION_TOKENS_PER_FRAME
            token_index = action_offset % ACTION_TOKENS_PER_FRAME
            packed[latent_index, token_index] = absolute_actions_to_relative_20d(
                reference, action_map[action_step][None]
            )[0]
            valid[latent_index, token_index] = True
        packed = (packed - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0
        packed = np.clip(packed, -1.5, 1.5) * valid
        
        # [latent_frame, action_token, action_dim]
        # [4, 16, 20]
        #
        # transpose(2,0,1):
        # [4,16,20] -> [20,4,16]
        #
        # 再增加 batch 和末尾 singleton 维度：
        # [20,4,16] -> [1,20,4,16,1]
        history_actions = torch.from_numpy(packed.transpose(2, 0, 1)[None, ..., None])
        action_valid = torch.from_numpy(valid.transpose(2, 0, 1)[None, ..., None])
        return OnlineWindow(
            history_rgb=torch.from_numpy(history_rgb).permute(0, 1, 4, 2, 3).float() / 255.0,
            anchor_rgb=torch.from_numpy(anchor_rgb).permute(0, 1, 4, 2, 3).float() / 255.0,
            video_valid=video_valid,
            history_actions=history_actions.float(),
            action_valid=action_valid.bool(),
            anchor_state=observations[current].state.copy(),
        )


class StreamingVAECodec:
    def __init__(self, vae, *, device: torch.device, dtype: torch.dtype) -> None:
        self.vae = vae
        self.device = device
        self.dtype = dtype

    def _normalize_latent(self, encoded: torch.Tensor) -> torch.Tensor:
        mu, _logvar = torch.chunk(encoded, 2, dim=1)
        mean = torch.tensor(
            self.vae.config.latents_mean, device=mu.device, dtype=mu.dtype
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            self.vae.config.latents_std, device=mu.device, dtype=mu.dtype
        ).view(1, -1, 1, 1, 1)
        return ((mu - mean) / std).to(dtype=self.dtype)

    def _video(self, frames: torch.Tensor) -> torch.Tensor:
        if tuple(frames.shape[-2:]) != MOT_IMAGE_SIZE:
            frames = F.interpolate(
                frames,
                size=MOT_IMAGE_SIZE,
                mode="bilinear",
                align_corners=False,
            )
        return frames.permute(1, 0, 2, 3)[None].to(self.device, self.vae.dtype) * 2 - 1

    @torch.no_grad()
    def encode_history(self, rgb: torch.Tensor) -> torch.Tensor:
        if tuple(rgb.shape[:2]) != (HISTORY_RAW_FRAMES, 3):
            raise ValueError("history RGB must be [13,3,3,H,W]")
        views = []
        for view in range(3):
            wrapper = WanVAEStreamingWrapper(self.vae)
            video = self._video(rgb[:, view])
            chunks = [wrapper.encode_chunk(video[:, :, :1])]
            for index in range(1, HISTORY_LATENT_FRAMES):
                start = 1 + 4 * (index - 1)
                chunks.append(wrapper.encode_chunk(video[:, :, start : start + 4]))
            views.append(self._normalize_latent(torch.cat(chunks, dim=2))[0])
        return torch.stack(views, dim=2)[None]

    @torch.no_grad()
    def encode_anchor(self, rgb: torch.Tensor) -> torch.Tensor:
        if tuple(rgb.shape[:2]) != (1, 3):
            raise ValueError("anchor RGB must be [1,3,3,H,W]")
        views = []
        for view in range(3):
            wrapper = WanVAEStreamingWrapper(self.vae)
            encoded = wrapper.encode_chunk(self._video(rgb[:, view]))
            views.append(self._normalize_latent(encoded)[0])
        return torch.stack(views, dim=2)[None]

    @torch.no_grad()
    def decode_one(self, latent: torch.Tensor) -> list[str]:
        batch, channels, frames, views, height, width = latent.shape
        packed = latent.permute(0, 3, 1, 2, 4, 5).reshape(
            batch * views, channels, frames, height, width
        ).to(self.device, self.vae.dtype)
        mean = torch.tensor(
            self.vae.config.latents_mean, device=self.device, dtype=packed.dtype
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            self.vae.config.latents_std, device=self.device, dtype=packed.dtype
        ).view(1, -1, 1, 1, 1)
        decoded = self.vae.decode(packed * std + mean, return_dict=False)[0]
        images = (decoded[:, :, 0].float() * 0.5 + 0.5).clamp(0, 1)
        return [
            encode_jpeg((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            for image in images
        ]

    @torch.no_grad()
    def decode_video(self, latent: torch.Tensor) -> dict[str, Any]:
        batch, channels, frames, views, height, width = latent.shape
        if batch != 1 or views != len(CAMERA_KEYS):
            raise ValueError("generated video requires one sample with three views")
        packed = latent.permute(0, 3, 1, 2, 4, 5).reshape(
            batch * views, channels, frames, height, width
        ).to(self.device, self.vae.dtype)
        mean = torch.tensor(
            self.vae.config.latents_mean, device=self.device, dtype=packed.dtype
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            self.vae.config.latents_std, device=self.device, dtype=packed.dtype
        ).view(1, -1, 1, 1, 1)
        decoded = self.vae.decode(packed * std + mean, return_dict=False)[0]
        decoded = (decoded.float() * 0.5 + 0.5).clamp(0, 1)
        raw_frames = decoded.shape[2]
        decoded_channels = decoded.shape[1]
        decoded = decoded.permute(0, 2, 1, 3, 4).reshape(
            batch,
            views,
            raw_frames,
            decoded_channels,
            decoded.shape[3],
            decoded.shape[4],
        ).permute(0, 2, 1, 3, 4, 5)
        generated = decoded[0, 1:]
        return {
            "fps": 10,
            "camera_keys": list(CAMERA_KEYS),
            "frames": [
                [
                    encode_jpeg(
                        (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    )
                    for image in frame
                ]
                for frame in generated
            ],
        }


class AutoregressiveMOTInferencePipeline:
    def __init__(
        self,
        *,
        model,
        codec: StreamingVAECodec,
        text_embedder,
        norm_stats_by_task: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        video_num_steps: int,
        action_num_steps: int,
        video_snr_shift: float,
        action_snr_shift: float,
        prediction_chunks: int = 1,
    ) -> None:
        self.model: AutoregressiveVAMOTTransformer3DModel = model
        self.codec = codec
        self.text_embedder = text_embedder
        self.norm_stats_by_task = norm_stats_by_task
        self.device = device
        self.dtype = dtype
        self.video_num_steps = int(video_num_steps)
        self.action_num_steps = int(action_num_steps)
        self.prediction_chunks = int(prediction_chunks)
        if not 1 <= self.prediction_chunks <= 3:
            raise ValueError("prediction_chunks must be between 1 and 3")
        self.video_scheduler = FlowMatchScheduler(
            shift=video_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.builder: OnlineMOTWindowBuilder | None = None
        self.text_emb: torch.Tensor | None = None
        self.task_name: str | None = None
        self.seed = 0

    def reset(self, *, task_name: str, instruction: str, seed: int) -> None:
        if task_name not in self.norm_stats_by_task:
            raise KeyError(f"unknown RoboTwin task {task_name!r}")
        self.builder = OnlineMOTWindowBuilder(self.norm_stats_by_task[task_name])
        self.text_emb = self.text_embedder(instruction).to(self.device)
        self.task_name = task_name
        self.seed = int(seed)

    def _euler_video(self, cache, text_emb, generator, frame_id: int) -> torch.Tensor:
        scheduler = self.video_scheduler
        scheduler.set_timesteps(self.video_num_steps)
        # _target_video_shape 由 infer() 根据 anchor latent 设置
        sample = torch.randn(
            self._target_video_shape,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        stream_ids = STREAM_IDS.to(self.device)
        
        # 从纯噪声开始做 Euler flow matching
        for timestep in scheduler.timesteps.to(self.device):
            flow = self.model.predict_video(
                sample,
                timestep=timestep,
                frame_ids=[frame_id],
                stream_ids=stream_ids,
                text_emb=text_emb,
                cache=cache,
            )
            # sample_next = EulerStep(sample, flow, timestep)
            sample = scheduler.step(flow, timestep, sample)
        return sample

    def _euler_action(self, cache, text_emb, generator, frame_id: int) -> torch.Tensor:
        scheduler = self.action_scheduler
        scheduler.set_timesteps(self.action_num_steps)
        sample = torch.randn(
            (1, 20, 1, ACTION_TOKENS_PER_FRAME, 1),
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        for timestep in scheduler.timesteps.to(self.device):
            flow = self.model.predict_action(
                sample,
                timestep=timestep,
                frame_ids=[frame_id],
                text_emb=text_emb,
                cache=cache,
            )
            sample = scheduler.step(flow, timestep, sample)
        return sample

    @torch.no_grad()
    def infer(
        self,
        *,
        observations: list[OnlineObservation],
        executed_actions: list[np.ndarray],
        request_id: int,
        return_video: bool,
    ) -> dict[str, Any]:
        if self.builder is None or self.text_emb is None or self.task_name is None:
            raise RuntimeError("reset must be called before inference")
        candidate = self.builder.clone()
        candidate.append(observations, executed_actions)
        started = time.perf_counter()
        window = candidate.build()
        window_ms = (time.perf_counter() - started) * 1000

        vae_started = time.perf_counter()
        history = self.codec.encode_history(window.history_rgb).to(self.device, self.dtype)
        anchor = self.codec.encode_anchor(window.anchor_rgb).to(self.device, self.dtype)
        self._target_video_shape = tuple(anchor.shape)
        vae_ms = (time.perf_counter() - vae_started) * 1000

        cache_started = time.perf_counter()
        cache = KVCache()
        text_emb = self.text_emb.to(self.device)
        stream_ids = STREAM_IDS.to(self.device)
        self.model.commit_video(
            history,
            frame_ids=[0, 1, 2, 3],
            stream_ids=stream_ids,
            text_emb=text_emb,
            cache=cache,
            token_valid_mask=window.video_valid.to(self.device),
        )
        self.model.commit_action(
            window.history_actions.to(self.device, self.dtype),
            frame_ids=[0, 1, 2, 3],
            text_emb=text_emb,
            cache=cache,
            token_valid_mask=window.action_valid.to(self.device),
        )
        self.model.commit_video(
            anchor,
            frame_ids=[4],
            stream_ids=stream_ids,
            text_emb=text_emb,
            cache=cache,
        )
        # Frame 4 has no valid action tokens in the training packing, so no
        # anchor action K/V may be committed to the incremental inference cache.
        cache_ms = (time.perf_counter() - cache_started) * 1000

        generator = torch.Generator(device=self.device).manual_seed(self.seed + int(request_id))
        predicted_videos = []
        predicted_actions = []
        video_ms = 0.0
        action_ms = 0.0
        for chunk_index in range(self.prediction_chunks):
            frame_id = 5 + chunk_index
            video_started = time.perf_counter()
            predicted_video = self._euler_video(
                cache, text_emb, generator, frame_id
            )
            video_ms += (time.perf_counter() - video_started) * 1000
            self.model.commit_video(
                predicted_video,
                frame_ids=[frame_id],
                stream_ids=stream_ids,
                text_emb=text_emb,
                cache=cache,
            )
            action_started = time.perf_counter()
            predicted_action = self._euler_action(
                cache, text_emb, generator, frame_id
            )
            action_ms += (time.perf_counter() - action_started) * 1000
            self.model.commit_action(
                predicted_action,
                frame_ids=[frame_id],
                text_emb=text_emb,
                cache=cache,
            )
            predicted_videos.append(predicted_video)
            predicted_actions.append(predicted_action)
        cache.assert_no_transactions()

        normalized = np.concatenate(
            [
                action[0, :, 0, :, 0].transpose(0, 1).float().cpu().numpy()
                for action in predicted_actions
            ]
        )
        stats = self.norm_stats_by_task[self.task_name]
        q01 = np.asarray(stats["q01"], np.float32)
        q99 = np.asarray(stats["q99"], np.float32)
        relative = (normalized + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01
        references = np.broadcast_to(window.anchor_state, (normalized.shape[0], 16))
        absolute = relative_20d_to_absolute_actions(references, relative)
        decoded_videos = (
            [self.codec.decode_one(video) for video in predicted_videos]
            if return_video
            else None
        )
        response = {
            "observation_step": candidate.last_step,
            "actions": absolute.astype(np.float32).tolist(),
            "action_type": "ee",
            "predicted_video": (
                decoded_videos[0]
                if decoded_videos is not None and self.prediction_chunks == 1
                else decoded_videos
            ),
            "timings_ms": {
                "window": window_ms,
                "vae": vae_ms,
                "cache": cache_ms,
                "video_euler": video_ms,
                "action_euler": action_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        }
        self.builder = candidate
        return response


class TextEmbedder:
    def __init__(self, model_root: Path, device: torch.device, dtype: torch.dtype) -> None:
        self.tokenizer = load_tokenizer(str(model_root / "tokenizer"))
        self.encoder = load_text_encoder(
            str(model_root / "text_encoder"), torch_dtype=dtype, torch_device=device
        ).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, text: str) -> torch.Tensor:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        inputs = self.tokenizer(
            [prompt_clean(text)],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.device)
        mask = inputs.attention_mask.to(self.device)
        return self.encoder(input_ids, mask).last_hidden_state


def load_pipeline(
    *,
    checkpoint_root: str | Path,
    dataset_root: str | Path,
    model_root: str | Path,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    video_num_steps: int = 2,
    action_num_steps: int = 4,
    video_snr_shift: float = 5.0,
    action_snr_shift: float = 1.0,
    prediction_chunks: int = 1,
) -> AutoregressiveMOTInferencePipeline:
    checkpoint_root = Path(checkpoint_root)
    metadata = json.loads((checkpoint_root / "checkpoint_metadata.json").read_text())
    if metadata.get("model_architecture") != "autoregressive_va_mot_v1":
        raise ValueError("evaluation requires an autoregressive_va_mot_v1 checkpoint")
    torch_device = torch.device(device)
    model = AutoregressiveVAMOTTransformer3DModel.from_pretrained(
        checkpoint_root / "transformer", torch_dtype=dtype
    ).to(torch_device).eval().requires_grad_(False)
    model_root = Path(model_root)
    vae = load_vae(model_root / "vae", torch_dtype=dtype, torch_device=torch_device)
    vae.eval().requires_grad_(False)
    mot_config = json.loads(
        (Path(dataset_root) / "meta" / "mot_config.json").read_text()
    )
    return AutoregressiveMOTInferencePipeline(
        model=model,
        codec=StreamingVAECodec(vae, device=torch_device, dtype=dtype),
        text_embedder=TextEmbedder(model_root, torch_device, dtype),
        norm_stats_by_task=mot_config["norm_stats_by_task"],
        device=torch_device,
        dtype=dtype,
        video_num_steps=video_num_steps,
        action_num_steps=action_num_steps,
        video_snr_shift=video_snr_shift,
        action_snr_shift=action_snr_shift,
        prediction_chunks=prediction_chunks,
    )
