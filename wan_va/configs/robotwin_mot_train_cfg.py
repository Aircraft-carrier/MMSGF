"""Original bidirectional MOT training configuration for prepared RoboTwin data."""

import copy
import json
import os
from pathlib import Path

from .va_wan22_train_cfg import va_wan22_train_cfg


robotwin_mot_train_cfg = copy.deepcopy(va_wan22_train_cfg)
robotwin_mot_train_cfg.__name__ = "Config: RoboTwin original VA-MOT train"

_dataset_root_value = os.getenv("MOT_DATASET_ROOT")
if not _dataset_root_value:
    raise ValueError("MOT_DATASET_ROOT must point to a prepared RoboTwin dataset")

_dataset_root = Path(_dataset_root_value).expanduser().resolve()
_mot_config_path = _dataset_root / "meta" / "mot_config.json"
if not _mot_config_path.is_file():
    raise FileNotFoundError(f"Missing RoboTwin MOT config: {_mot_config_path}")

_mot_config = json.loads(_mot_config_path.read_text(encoding="utf-8"))


def _local_or_configured(local_path: Path, key: str):
    return str(local_path if local_path.is_file() else Path(_mot_config[key]))


robotwin_mot_train_cfg.dataset_path = str(_dataset_root)
robotwin_mot_train_cfg.mot_config_path = str(_mot_config_path)
robotwin_mot_train_cfg.mot_manifest_path = _local_or_configured(
    _dataset_root / "meta" / "mot_final_training_manifest.jsonl",
    "mot_manifest_path",
)
robotwin_mot_train_cfg.empty_emb_path = _local_or_configured(
    _dataset_root / "empty_emb.pt",
    "empty_emb_path",
)
robotwin_mot_train_cfg.text_emb_cache_path = _local_or_configured(
    _dataset_root / "text_emb_cache.pt",
    "text_emb_cache_path",
)

_action_cache_manifest = _mot_config.get("action_cache_manifest_path")
if _action_cache_manifest:
    _local_action_cache_manifest = (
        _dataset_root / "cache" / "actions" / "action_cache_manifest.jsonl"
    )
    robotwin_mot_train_cfg.action_cache_manifest_path = str(
        _local_action_cache_manifest
        if _local_action_cache_manifest.is_file()
        else Path(_action_cache_manifest)
    )

for _name in (
    "obs_cam_keys",
    "action_dim",
    "action_representation",
    "video_downsample_ratio",
    "vae_temporal_factor",
    "action_chunk_size",
    "action_sequence_length",
    "action_per_frame",
    "action_frames",
    "sampled_video_frames_per_action_chunk_per_view",
    "latent_frames_per_action_chunk_per_view",
    "norm_stat",
    "norm_stats_by_task",
):
    if _name in _mot_config:
        robotwin_mot_train_cfg[_name] = _mot_config[_name]

# Keep the native bidirectional model and masked-attention training path.
robotwin_mot_train_cfg.model_type = "va_mot"
