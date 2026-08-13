"""Shared incremental pipeline base: cache ownership and history construction."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .cache import KVCache

if TYPE_CHECKING:
    from distillation.model.common.wan_wrapper import WanDiffusionWrapper


class BasePipeline:
    """Own one KV cache and the history-frame commit helpers.

    The base class is reused as the historical-cache owner so future pipelines
    can build on the same commit order (video then action per frame range).
    """

    def __init__(
        self,
        generator: "WanDiffusionWrapper",
        *,
        num_frame_per_block: int = 3,
    ) -> None:
        self.generator = generator
        self.cache = KVCache()
        self.num_frame_per_block = int(num_frame_per_block)

    def reset_cache(self) -> None:
        self.cache = KVCache()

    def build_history_cache(
        self,
        batch: dict[str, Any],
        *,
        history_frames: int,
        device: torch.device,
    ) -> None:
        """Commit history and anchor K/V in causal order.

        Order matters for cache attention: video history, then action history,
        then the clean anchor video, then the anchor action.
        """
        history_frames = int(history_frames)
        history_ids = list(range(history_frames))
        stream_ids = batch["stream_ids"].to(device=device)
        text_emb = batch["text_emb"].to(device=device)
        latents = batch["latents"].to(device=device)
        actions = batch["actions"].to(device=device)
        video_valid = batch.get("video_latent_valid_mask")
        if video_valid is not None:
            video_valid = video_valid.to(device=device, dtype=torch.bool)
        action_valid = batch.get("action_valid_mask")
        if action_valid is not None:
            action_valid = action_valid.to(device=device, dtype=torch.bool)
        self.generator.commit_video(
            latents[:, :, :history_frames],
            frame_ids=history_ids,
            stream_ids=stream_ids,
            cache=self.cache,
            text_emb=text_emb,
            token_valid_mask=(
                None if video_valid is None else video_valid[:, :history_frames]
            ),
        )
        self.generator.commit_action(
            actions[:, :, :history_frames],
            frame_ids=history_ids,
            cache=self.cache,
            text_emb=text_emb,
            token_valid_mask=(
                None
                if action_valid is None
                else action_valid[:, :, :history_frames]
            ),
        )
        self.generator.commit_video(
            latents[:, :, history_frames : history_frames + 1],
            frame_ids=[history_frames],
            stream_ids=stream_ids,
            cache=self.cache,
            text_emb=text_emb,
        )
        self.generator.commit_action(
            actions[:, :, history_frames : history_frames + 1],
            frame_ids=[history_frames],
            cache=self.cache,
            text_emb=text_emb,
        )
