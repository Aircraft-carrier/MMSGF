"""Distillation configs."""
from distillation.configs.autoregressive_training import autoregressive_training_cfg
from distillation.configs.consistency_distillation import consistency_distillation_cfg
from distillation.configs.self_gradient_forcing_dmd import self_gradient_forcing_dmd_cfg

AUTOREGRESSIVE_TRAINING = "autoregressive_training"
CONSISTENCY_DISTILLATION = "consistency_distillation"
SELF_GRADIENT_FORCING_DMD = "self_gradient_forcing_dmd"
CONFIGS = {
    AUTOREGRESSIVE_TRAINING: autoregressive_training_cfg,
    CONSISTENCY_DISTILLATION: consistency_distillation_cfg,
    SELF_GRADIENT_FORCING_DMD: self_gradient_forcing_dmd_cfg,
}
METHODS = tuple(CONFIGS)

__all__ = [
    "AUTOREGRESSIVE_TRAINING",
    "CONFIGS",
    "CONSISTENCY_DISTILLATION",
    "METHODS",
    "SELF_GRADIENT_FORCING_DMD",
]
