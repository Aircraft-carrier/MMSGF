"""Distillation models."""
from __future__ import annotations


def __getattr__(name: str):
    if name == "WanDiffusionWrapper":
        from distillation.model.common import wan_wrapper
        return getattr(wan_wrapper, name)
    if name in {
        "AutoregressiveVAMOTTransformer3DModel",
        "AutoregressiveVAMOTBlock",
    }:
        from distillation.model.autoregressive_mot import (
            AutoregressiveVAMOTBlock,
            AutoregressiveVAMOTTransformer3DModel,
        )
        return {
            "AutoregressiveVAMOTTransformer3DModel": AutoregressiveVAMOTTransformer3DModel,
            "AutoregressiveVAMOTBlock": AutoregressiveVAMOTBlock,
        }[name]
    if name in {
        "action_aware_loss",
        "ConsistencyBaseModel",
        "ConsistencyTrainingModel",
        "ConsistencyModel",
        "consistency_loss",
    }:
        from distillation.model import consistency
        return getattr(consistency, name)
    if name in {
        "BaseModel",
        "dmd_surrogate_loss",
        "fake_score_flow_loss",
        "SelfGradientForcingModel",
        "SGFDMDModel",
        "update_ema",
    }:
        from distillation.model import dmd
        return getattr(dmd, name)
    if name in {
        "add_noise",
        "add_noise_to_va",
        "apply_va_mask",
        "broadcast_frame_values",
        "freeze_model",
        "randn_like_va",
        "replace_text_condition",
        "replace_va_streams",
        "set_trainable",
        "sigmas_for_timesteps",
        "va_loss",
    }:
        from distillation.model.common import utils
        return getattr(utils, name)
    if name in {
        "load_transformer_export",
        "build_frozen_transformer",
        "build_trainable_transformer",
    }:
        from distillation.model.common import factory
        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AutoregressiveVAMOTTransformer3DModel",
    "AutoregressiveVAMOTBlock",
    "ConsistencyModel",
    "ConsistencyBaseModel",
    "ConsistencyTrainingModel",
    "BaseModel",
    "SGFDMDModel",
    "SelfGradientForcingModel",
    "action_aware_loss",
    "add_noise",
    "add_noise_to_va",
    "apply_va_mask",
    "broadcast_frame_values",
    "build_frozen_transformer",
    "build_trainable_transformer",
    "freeze_model",
    "randn_like_va",
    "load_transformer_export",
    "set_trainable",
    "sigmas_for_timesteps",
    "replace_text_condition",
    "replace_va_streams",
    "update_ema",
    "va_loss",
    "consistency_loss",
    "dmd_surrogate_loss",
    "fake_score_flow_loss",
    "WanDiffusionWrapper",
]
