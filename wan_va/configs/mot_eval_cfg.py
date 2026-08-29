"""Configuration shared by automatic CPU and manual GPU MOT evaluation."""

from pathlib import Path

from easydict import EasyDict


def make_mot_eval_cfg(
    *,
    dataset_root: str | Path,
    wan22_model_root: str | Path,
    mode: str = "video",
) -> EasyDict:
    mode = str(mode).strip().lower()
    if mode not in {"video", "full"}:
        raise ValueError(f"MOT evaluation mode must be video or full, got {mode!r}")
    return EasyDict(
        mode=mode,
        device="cuda:0",
        masked_attn_backend="auto",
        dataset_root=str(dataset_root),
        wan22_model_root=str(wan22_model_root),
        output_root=None,
        seed=42,
        num_inference_steps=25,
        action_num_inference_steps=50,
        guidance_scale=5.0,
        action_guidance_scale=1.0,
        inference_video_fps=10,
    )
