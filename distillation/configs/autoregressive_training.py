"""Autoregressive training configuration."""
import copy

from easydict import EasyDict

from wan_va.configs import VA_CONFIGS

autoregressive_training_cfg = EasyDict(copy.deepcopy(VA_CONFIGS["umi_3dwam_train"]))
autoregressive_training_cfg.distill = EasyDict(
    method="autoregressive_training",
    generation_shape={"order_mode": "segmented", "chunk_size": 4, "window_size": 16},
    student_init=None,
    resume_from=None,
)
