"""Shared fixed-window protocol for MOT training and inference."""

from dataclasses import dataclass

from .dataset.mot_dataset import (
    MOT_HISTORY_CHUNKS,
    MOT_TARGET_CHUNKS,
    mot_action_per_frame,
    mot_action_sequence_length,
    mot_latent_frames_per_action_chunk_per_view,
    mot_sampled_video_frames_per_action_chunk_per_view,
)


@dataclass(frozen=True)
class MOTWindowSpec:
    history_chunks: int
    target_chunks: int
    action_chunk_size: int
    video_downsample_ratio: int
    vae_temporal_factor: int
    sampled_video_frames_per_action_chunk_per_view: int
    latent_frames_per_action_chunk_per_view: int
    history_latent_frames: int
    target_latent_frames: int
    total_latent_frames: int
    vae_input_frame_count: int
    action_sequence_length: int
    action_per_frame: int
    target_actions_per_sample: int
    attention_window_size: int


def derive_mot_window_spec(
    *,
    action_chunk_size: int,
    video_downsample_ratio: int,
    vae_temporal_factor: int = 4,
) -> MOTWindowSpec:
    history_chunks = MOT_HISTORY_CHUNKS
    target_chunks = MOT_TARGET_CHUNKS
    action_chunk_size = int(action_chunk_size)
    video_downsample_ratio = int(video_downsample_ratio)
    vae_temporal_factor = int(vae_temporal_factor)
    for name, value in (
        ("history_chunks", history_chunks),
        ("target_chunks", target_chunks),
        ("action_chunk_size", action_chunk_size),
        ("video_downsample_ratio", video_downsample_ratio),
        ("vae_temporal_factor", vae_temporal_factor),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    sampled_video_frames_per_view = mot_sampled_video_frames_per_action_chunk_per_view(
        action_chunk_size,
        video_downsample_ratio,
    )
    latent_frames_per_view = mot_latent_frames_per_action_chunk_per_view(
        action_chunk_size,
        video_downsample_ratio,
        vae_temporal_factor,
    )
    action_per_frame = mot_action_per_frame(video_downsample_ratio, vae_temporal_factor)
    expected_action_chunk_size = (latent_frames_per_view - 1) * action_per_frame
    if action_chunk_size != expected_action_chunk_size:
        raise ValueError(
            "action_chunk_size must match latent-frame action packing: "
            f"got {action_chunk_size}, expected {expected_action_chunk_size}"
        )
    history_latent_frames = history_chunks * latent_frames_per_view
    target_latent_frames = target_chunks * latent_frames_per_view
    total_latent_frames = history_latent_frames + target_latent_frames
    return MOTWindowSpec(
        history_chunks=history_chunks,
        target_chunks=target_chunks,
        action_chunk_size=action_chunk_size,
        video_downsample_ratio=video_downsample_ratio,
        vae_temporal_factor=vae_temporal_factor,
        sampled_video_frames_per_action_chunk_per_view=sampled_video_frames_per_view,
        latent_frames_per_action_chunk_per_view=latent_frames_per_view,
        history_latent_frames=history_latent_frames,
        target_latent_frames=target_latent_frames,
        total_latent_frames=total_latent_frames,
        vae_input_frame_count=sampled_video_frames_per_view,
        action_sequence_length=mot_action_sequence_length(action_chunk_size),
        action_per_frame=action_per_frame,
        target_actions_per_sample=target_chunks * action_chunk_size,
        attention_window_size=2 * (history_chunks + target_chunks),
    )


def mot_spec_from_config(config) -> MOTWindowSpec:
    return derive_mot_window_spec(
        action_chunk_size=getattr(config, "action_chunk_size"),
        video_downsample_ratio=getattr(config, "video_downsample_ratio"),
        vae_temporal_factor=getattr(config, "vae_temporal_factor", 4),
    )
