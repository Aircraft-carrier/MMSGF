# Copyright 2024-2025 The Alibaba Wan Team Authors.
"""Standalone WAN2.2 Video+Action MOT training configuration.

This file intentionally does not inherit from any other local configuration.
Fill in the empty dataset/model paths before launching training.
"""

import os
from pathlib import Path

import torch
from easydict import EasyDict


va_wan22_train_cfg = EasyDict(__name__="Config: standalone WAN2.2 MOT train")

# ---------------------------------------------------------------------------
# Paths: fill these in before starting a run.
# ---------------------------------------------------------------------------
va_wan22_train_cfg.dataset_path = ""
va_wan22_train_cfg.mot_config_path = ""
va_wan22_train_cfg.mot_manifest_path = ""
va_wan22_train_cfg.empty_emb_path = ""
va_wan22_train_cfg.text_emb_cache_path = ""
va_wan22_train_cfg.action_cache_manifest_path = None

# WAN2.2 initialization.  `init_model_from_lingbot=False` selects the
# diffusers WAN2.2 transformer below; the VAE is loaded from the model root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WAN22_MODEL_ROOT = _REPO_ROOT / "playground" / "Pretrained_models" / "Wan2.2-TI2V-5B"
_WAN22_MODEL_ROOT = Path(
    os.getenv("WAN22_PRETRAINED_MODEL_PATH", str(_DEFAULT_WAN22_MODEL_ROOT))
).expanduser()

# This is a base Wan2.2 transformer, not a saved MOT checkpoint.  Training
# therefore constructs a VAMOTTransformer3DModel from its video backbone and
# initializes the action expert locally.
va_wan22_train_cfg.init_model_from_lingbot = False
va_wan22_train_cfg.wan22_pretrained_model_name_or_path = str(_WAN22_MODEL_ROOT)
va_wan22_train_cfg.wan22_transformer_path = os.getenv(
    "WAN22_TRANSFORMER_PATH", str(_WAN22_MODEL_ROOT)
)
va_wan22_train_cfg.lingbot_transformer_path = ""
# Keep both unset for a fresh MOT run: the base WAN2.2 video backbone above is
# converted into a MOT model and its action blocks are initialized from video.
va_wan22_train_cfg.resume_from = None
va_wan22_train_cfg.initialize_from = None

# ---------------------------------------------------------------------------
# Dataset / fixed-window MOT protocol.
# These values must match meta/mot_config.json generated for the dataset.
# ---------------------------------------------------------------------------
va_wan22_train_cfg.dataset_type = "mot_real_lerobot"
va_wan22_train_cfg.model_type = "va_mot"
va_wan22_train_cfg.env_type = "none"
va_wan22_train_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_wan22_train_cfg.action_dim = 20
va_wan22_train_cfg.action_representation = "relative_to_chunk_reference_state_rot6d_cols"
va_wan22_train_cfg.action_chunk_size = 48
va_wan22_train_cfg.action_sequence_length = 49
va_wan22_train_cfg.action_per_frame = 16
va_wan22_train_cfg.action_frames = 8
va_wan22_train_cfg.video_downsample_ratio = 4
va_wan22_train_cfg.vae_temporal_factor = 4
va_wan22_train_cfg.sampled_video_frames_per_action_chunk_per_view = 13
va_wan22_train_cfg.latent_frames_per_action_chunk_per_view = 4
va_wan22_train_cfg.height = 256
va_wan22_train_cfg.width = 320
va_wan22_train_cfg.norm_stat = {
    "q01": [
        -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
        -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
        -0.11975435614585876, -1, -1, -1, -1
    ] + [0.] * 16,
    "q99": [
        0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
        1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
        1, 1
    ] + [0.] * 14 + [1.0, 1.0],
}

va_wan22_train_cfg.norm_stats_by_task = {
    "default": va_wan22_train_cfg.norm_stat,
}
va_wan22_train_cfg.action_norm_method = "quantiles"

# ---------------------------------------------------------------------------
# Model/runtime defaults.
# ---------------------------------------------------------------------------
va_wan22_train_cfg.param_dtype = torch.bfloat16
va_wan22_train_cfg.patch_size = (1, 2, 2)
va_wan22_train_cfg.enable_offload = False
va_wan22_train_cfg.masked_attn_backend = "fa4"
va_wan22_train_cfg.attn_window = 72
va_wan22_train_cfg.frame_chunk_size = 2
va_wan22_train_cfg.init_noise_seed = 42

# ---------------------------------------------------------------------------
# Optimizer and training schedule.
# ---------------------------------------------------------------------------
va_wan22_train_cfg.learning_rate = 1e-5
va_wan22_train_cfg.beta1 = 0.9
va_wan22_train_cfg.beta2 = 0.95
va_wan22_train_cfg.weight_decay = 0.1
va_wan22_train_cfg.warmup_steps = 10
va_wan22_train_cfg.gradient_accumulation_steps = 1
va_wan22_train_cfg.num_steps = 50000 
va_wan22_train_cfg.snr_shift = 5.0
va_wan22_train_cfg.action_snr_shift = 1.0
va_wan22_train_cfg.cfg_prob = 0.1
va_wan22_train_cfg.video_noisy_cond_prob = 0.5
va_wan22_train_cfg.video_loss_weight = 1.0
va_wan22_train_cfg.action_loss_weight = 1.0

# ---------------------------------------------------------------------------
# Data loading, checkpointing, logging, and reproducibility.
# ---------------------------------------------------------------------------
va_wan22_train_cfg.load_worker = 8
va_wan22_train_cfg.max_views_per_gpu = 3
va_wan22_train_cfg.dataloader_pin_memory = True
va_wan22_train_cfg.dataloader_prefetch_factor = 2
va_wan22_train_cfg.video_decoder_cache_size = 256
va_wan22_train_cfg.action_cache_size = 2176
va_wan22_train_cfg.save_root = "./train_out"
va_wan22_train_cfg.save_interval = 2000
va_wan22_train_cfg.max_checkpoints = 3
va_wan22_train_cfg.save_full_state = True
va_wan22_train_cfg.enable_wandb = False
va_wan22_train_cfg.wandb_name = "va_wan22_mot_train"
va_wan22_train_cfg.wandb_mode = "offline"
va_wan22_train_cfg.log_interval = 100
va_wan22_train_cfg.train_seed = 42
va_wan22_train_cfg.sampler_seed = 42
va_wan22_train_cfg.gc_interval = 3500
va_wan22_train_cfg.performance_jsonl_enabled = False
va_wan22_train_cfg.performance_jsonl_interval = 1
va_wan22_train_cfg.performance_jsonl_max_steps = 500
va_wan22_train_cfg.memory_jsonl_enabled = False
va_wan22_train_cfg.memory_jsonl_interval = 100
va_wan22_train_cfg.memory_smaps_interval = 500
va_wan22_train_cfg.eval_with_cpu = False
va_wan22_train_cfg.eval_cfg = None
