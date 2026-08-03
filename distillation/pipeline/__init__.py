"""Distillation pipelines."""
from __future__ import annotations

from distillation.pipeline.utils import (
    VAPrediction,
    add_noise_to_va,
    hide_target_geometry,
    mask_clean_targets,
    replace_va_streams,
)


def __getattr__(name: str):
    if name == "ConsistencyTrainingPipeline":
        from distillation.pipeline.consistency_training import ConsistencyTrainingPipeline
        return ConsistencyTrainingPipeline
    if name == "SelfGradientForcingTrainingPipeline":
        from distillation.pipeline.self_gradient_forcing_training import SelfGradientForcingTrainingPipeline
        return SelfGradientForcingTrainingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConsistencyTrainingPipeline",
    "SelfGradientForcingTrainingPipeline",
    "VAPrediction",
    "add_noise_to_va",
    "hide_target_geometry",
    "mask_clean_targets",
    "replace_va_streams",
]
