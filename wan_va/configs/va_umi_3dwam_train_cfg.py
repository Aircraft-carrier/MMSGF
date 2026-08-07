# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Training config for the UMI 3DWA/MOT post-training path.

This config intentionally mirrors the official `robotwin_train` post-training
defaults wherever the training semantics are shared: optimizer hyperparameters,
FSDP dtype, and step-based checkpointing.  The MOT-specific fields describe the
fixed-window raw-RGB training protocol.

The MOT dataset builder writes `meta/mot_config.json` next to the prepared
LeRobot-style dataset.  That file is the source of truth for camera keys, action
channel mapping, action normalization statistics, and the MOT manifest path.
Reading it here prevents silent drift between the generated dataset and the
training config.
"""

import json
import os
from pathlib import Path

from easydict import EasyDict

from .mot_eval_cfg import make_mot_eval_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_VIDEO_DOWNSAMPLE_RATIO = os.getenv("MOT_VIDEO_DOWNSAMPLE_RATIO")
_DEFAULT_VIDEO_DOWNSAMPLE_RATIO = int(_ENV_VIDEO_DOWNSAMPLE_RATIO or "4")
_DEFAULT_DATASET_ROOT = _REPO_ROOT / "data" / "0619_train_50subset"
_DATASET_ROOT = Path(os.getenv("MOT_DATASET_ROOT", str(_DEFAULT_DATASET_ROOT)))
_MOT_CONFIG_PATH = _DATASET_ROOT / "meta" / "mot_config.json"


def _default_mot_config() -> dict:
    return {
        "format": "umi_mot_real_train",
        "mot_manifest_path": str(_DATASET_ROOT / "meta" / "mot_final_training_manifest.jsonl"),
        "empty_emb_path": str(_DATASET_ROOT / "empty_emb.pt"),
        "text_emb_cache_path": str(_DATASET_ROOT / "text_emb_cache.pt"),
        "obs_cam_keys": ["observation.images.robot_0", "observation.images.robot_1"],
        "action_dim": 20,
        "action_representation": "relative_to_chunk_reference_state_rot6d_cols",
        "video_downsample_ratio": _DEFAULT_VIDEO_DOWNSAMPLE_RATIO,
        "vae_temporal_factor": 4,
        "norm_stat": {"q01": [0.0] * 20, "q99": [1.0] * 20},
        "action_chunk_size": 48,
        "action_sequence_length": 49,
        "action_per_frame": 16,
        "action_frames": 8,
        "sampled_video_frames_per_action_chunk_per_view": 13,
        "latent_frames_per_action_chunk_per_view": 4,
        "norm_stats_by_task": {
            "default": {"q01": [0.0] * 20, "q99": [1.0] * 20},
        },
    }


_MOT_CONFIG_EXISTS = _MOT_CONFIG_PATH.is_file()
_mot_config = json.loads(_MOT_CONFIG_PATH.read_text()) if _MOT_CONFIG_EXISTS else _default_mot_config()
if _MOT_CONFIG_EXISTS and not _mot_config.get("norm_stats_by_task"):
    raise ValueError(
        "Selected MOT dataset is missing norm_stats_by_task. "
        f"Regenerate it from raw data: {_MOT_CONFIG_PATH}"
    )
_DATASET_VIDEO_DOWNSAMPLE_RATIO = int(_mot_config.get("video_downsample_ratio", _DEFAULT_VIDEO_DOWNSAMPLE_RATIO))
if _ENV_VIDEO_DOWNSAMPLE_RATIO is not None and int(_ENV_VIDEO_DOWNSAMPLE_RATIO) != _DATASET_VIDEO_DOWNSAMPLE_RATIO:
    raise ValueError(
        "MOT_VIDEO_DOWNSAMPLE_RATIO does not match the selected dataset config: "
        f"env={_ENV_VIDEO_DOWNSAMPLE_RATIO}, dataset={_DATASET_VIDEO_DOWNSAMPLE_RATIO}, "
        f"mot_config={_MOT_CONFIG_PATH}"
    )
_ACTION_CACHE_MANIFEST = os.getenv(
    "MOT_ACTION_CACHE_MANIFEST",
    _mot_config.get("action_cache_manifest_path") or "",
)
def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def _prefer_local_path(relative_path: str, fallback: str) -> str:
    local = _DATASET_ROOT / relative_path
    return str(local if local.exists() else Path(fallback))


va_umi_3dwam_train_cfg = EasyDict(__name__="Config: UMI 3DWA MOT train")
va_umi_3dwam_train_cfg.update(va_robotwin_train_cfg)
va_umi_3dwam_train_cfg.pop("batch_size", None)

# Source checkpoints used only when neither checkpoint input is provided.
va_umi_3dwam_train_cfg.wan22_pretrained_model_name_or_path = os.getenv(
    "WAN22_PRETRAINED_MODEL_PATH",
    "/workspace/cache/huggingface_cache/hub/"
    "models--robbyant--lingbot-va-base/snapshots/"
    "68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c",
)
va_umi_3dwam_train_cfg.lingbot_transformer_path = (
    va_umi_3dwam_train_cfg.wan22_pretrained_model_name_or_path + "/transformer"
)
va_umi_3dwam_train_cfg.init_model_from_lingbot = _env_flag(
    "INIT_MODEL_FROM_LINGBOT",
    True,
)
va_umi_3dwam_train_cfg.wan22_transformer_path = os.path.join(
    os.getenv("WAN22_DIFFUSERS_MODEL_PATH", "/workspace/model/wan2_2_diffusers"),
    "transformer",
)
va_umi_3dwam_train_cfg.eval_with_cpu = _env_flag("MOT_EVAL_WITH_CPU")
va_umi_3dwam_train_cfg.eval_cfg = make_mot_eval_cfg(
    dataset_root=_DATASET_ROOT,
    wan22_model_root=va_umi_3dwam_train_cfg.wan22_pretrained_model_name_or_path,
    mode=os.getenv("MOT_EVAL_MODE", "video"),
)
# Dataset fields are read from the generated MOT dataset metadata.
va_umi_3dwam_train_cfg.dataset_path = str(_DATASET_ROOT)
va_umi_3dwam_train_cfg.mot_config_path = str(_MOT_CONFIG_PATH)
va_umi_3dwam_train_cfg.mot_manifest_path = _prefer_local_path(
    "meta/mot_final_training_manifest.jsonl",
    _mot_config["mot_manifest_path"],
)
va_umi_3dwam_train_cfg.empty_emb_path = _prefer_local_path("empty_emb.pt", _mot_config["empty_emb_path"])
va_umi_3dwam_train_cfg.text_emb_cache_path = _prefer_local_path(
    "text_emb_cache.pt",
    _mot_config.get("text_emb_cache_path", va_umi_3dwam_train_cfg.empty_emb_path),
)
va_umi_3dwam_train_cfg.action_cache_manifest_path = (
    _prefer_local_path("cache/actions/action_cache_manifest.jsonl", _ACTION_CACHE_MANIFEST)
    if _ACTION_CACHE_MANIFEST
    else None
)
va_umi_3dwam_train_cfg.obs_cam_keys = _mot_config["obs_cam_keys"]
va_umi_3dwam_train_cfg.action_dim = int(_mot_config["action_dim"])
if va_umi_3dwam_train_cfg.action_dim != 20:
    raise ValueError(f"MOT relative-action training requires action_dim=20, got {_mot_config['action_dim']}")
va_umi_3dwam_train_cfg.vae_temporal_factor = int(_mot_config.get("vae_temporal_factor", 4))
va_umi_3dwam_train_cfg.video_downsample_ratio = _DATASET_VIDEO_DOWNSAMPLE_RATIO
va_umi_3dwam_train_cfg.action_chunk_size = int(_mot_config["action_chunk_size"])
va_umi_3dwam_train_cfg.action_sequence_length = int(_mot_config["action_sequence_length"])
va_umi_3dwam_train_cfg.action_per_frame = int(_mot_config["action_per_frame"])
va_umi_3dwam_train_cfg.action_frames = int(_mot_config["action_frames"])
va_umi_3dwam_train_cfg.sampled_video_frames_per_action_chunk_per_view = int(_mot_config["sampled_video_frames_per_action_chunk_per_view"])
va_umi_3dwam_train_cfg.latent_frames_per_action_chunk_per_view = int(_mot_config["latent_frames_per_action_chunk_per_view"])
va_umi_3dwam_train_cfg.norm_stat = _mot_config["norm_stat"]
va_umi_3dwam_train_cfg.norm_stats_by_task = _mot_config.get("norm_stats_by_task")
va_umi_3dwam_train_cfg.env_type = "none"
va_umi_3dwam_train_cfg.height = 224
va_umi_3dwam_train_cfg.width = 224

# MOT-specific training controls.
va_umi_3dwam_train_cfg.model_type = "va_mot"
va_umi_3dwam_train_cfg.dataset_type = "mot_real_lerobot"
va_umi_3dwam_train_cfg.num_steps = int(os.getenv("MOT_NUM_STEPS", "1000000"))
va_umi_3dwam_train_cfg.save_interval = int(os.getenv("MOT_SAVE_INTERVAL", "2000"))
va_umi_3dwam_train_cfg.max_checkpoints = 3
va_umi_3dwam_train_cfg.resume_from = os.getenv("MOT_RESUME_FROM") or None
va_umi_3dwam_train_cfg.initialize_from = os.getenv("MOT_INITIALIZE_FROM") or None
va_umi_3dwam_train_cfg.save_full_state = True
va_umi_3dwam_train_cfg.enable_wandb = True
va_umi_3dwam_train_cfg.load_worker = 8
va_umi_3dwam_train_cfg.video_noisy_cond_prob = 0.5
va_umi_3dwam_train_cfg.masked_attn_backend = "fa4"
va_umi_3dwam_train_cfg.init_noise_seed = 42
va_umi_3dwam_train_cfg.train_seed = int(os.getenv("MOT_TRAIN_SEED", "42"))
va_umi_3dwam_train_cfg.sampler_seed = 42
va_umi_3dwam_train_cfg.log_interval = int(os.getenv("MOT_LOG_INTERVAL", "100"))
va_umi_3dwam_train_cfg.performance_jsonl_enabled = False
va_umi_3dwam_train_cfg.performance_jsonl_interval = 1
va_umi_3dwam_train_cfg.performance_jsonl_max_steps = 500
# Main training ranks disable automatic cyclic GC after workers start; 0 disables scheduled full GC.
va_umi_3dwam_train_cfg.gc_interval = 3500
va_umi_3dwam_train_cfg.dataloader_pin_memory = True
va_umi_3dwam_train_cfg.dataloader_prefetch_factor = 2
# Limits apply to each MotTrainData instance in every persistent worker.
va_umi_3dwam_train_cfg.video_decoder_cache_size = int(os.getenv("MOT_VIDEO_DECODER_CACHE_SIZE", "256"))
va_umi_3dwam_train_cfg.action_cache_size = int(os.getenv("MOT_ACTION_CACHE_SIZE", "2176"))
va_umi_3dwam_train_cfg.memory_jsonl_enabled = False
va_umi_3dwam_train_cfg.memory_jsonl_interval = 100
va_umi_3dwam_train_cfg.memory_smaps_interval = 500
va_umi_3dwam_train_cfg.video_loss_weight = 1.0
va_umi_3dwam_train_cfg.action_loss_weight = 1.0
va_umi_3dwam_train_cfg.wandb_name = "umi_subset3k_newmot_stage2_video_only_0724"
va_umi_3dwam_train_cfg.wandb_mode = "offline"
va_umi_3dwam_train_cfg.action_norm_method = "quantiles"

va_umi_3dwam_train_cfg.max_views_per_gpu = int(
    os.getenv("MOT_MAX_VIEWS_PER_GPU", "3")
)
