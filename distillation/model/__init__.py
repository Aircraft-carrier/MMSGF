"""Distillation models."""
from __future__ import annotations


def __getattr__(name: str):
    if name in {"ConsistencyModel", "SGFDMDModel"}:
        if name == "ConsistencyModel":
            from distillation.model.consistency import ConsistencyModel
            return ConsistencyModel
        from distillation.model.dmd import SGFDMDModel
        return SGFDMDModel
    if name in {
        "freeze_model",
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
        "replay_target_loss",
    }:
        from distillation.model import objectives
        return getattr(objectives, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConsistencyModel",
    "SGFDMDModel",
    "action_aware_loss",
    "build_frozen_transformer",
    "build_trainable_transformer",
    "freeze_model",
    "load_transformer_export",
    "set_trainable",
    "update_ema",
    "consistency_loss",
    "dmd_surrogate_loss",
    "fake_score_flow_loss",
    "replay_target_loss",
]
