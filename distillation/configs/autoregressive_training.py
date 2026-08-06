"""Autoregressive training configuration."""
import copy

from easydict import EasyDict

from distillation.configs.runtime_dataset import apply_distillation_runtime_overrides
from wan_va.configs import VA_CONFIGS

autoregressive_training_cfg = EasyDict(copy.deepcopy(VA_CONFIGS["umi_3dwam_train"]))
apply_distillation_runtime_overrides(autoregressive_training_cfg)
autoregressive_training_cfg.distill = EasyDict(
    method="autoregressive_training",
    model_architecture="autoregressive_mot_v1",
    generation_shape={
        "profile_name": "segmented_history_strict_geometry_v1",
        "order_mode": "segmented",
        "history_frames": 4,
        "chunk_size": 4,
        "window_size": 16,
    },
    student_init=None,
    resume_from=None,
)
