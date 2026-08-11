"""Distillation models."""
from __future__ import annotations


def __getattr__(name: str):
    if name == "WanDiffusionWrapper":
        from distillation.model import wan_wrapper
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
        "ConsistencyBaseModel",
        "ConsistencyTrainingModel",
        "ConsistencyModel",
    }:
        from distillation.model import consistency
        return getattr(consistency, name)
    if name in {"BaseModel", "SelfGradientForcingModel", "SGFDMDModel"}:
        from distillation.model import dmd
        return getattr(dmd, name)
    if name in {
        "add_noise_to_va",
        "freeze_model",
        "mask_clean_targets",
        "replace_text_condition",
        "replace_va_streams",
        "set_trainable",
        "update_ema",
    }:
        from distillation.model import utils
        return getattr(utils, name)
    if name in {
        "load_transformer_export",
        "build_frozen_transformer",
        "build_trainable_transformer",
    }:
        from distillation.model import factory
        return getattr(factory, name)
    if name in {
        "action_aware_loss",
        "consistency_loss",
        "dmd_surrogate_loss",
        "fake_score_flow_loss",
    }:
        from distillation.model import objectives
        return getattr(objectives, name)
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
    "add_noise_to_va",
    "build_frozen_transformer",
    "build_trainable_transformer",
    "freeze_model",
    "mask_clean_targets",
    "load_transformer_export",
    "set_trainable",
    "replace_text_condition",
    "replace_va_streams",
    "update_ema",
    "consistency_loss",
    "dmd_surrogate_loss",
    "fake_score_flow_loss",
    "WanDiffusionWrapper",
]
