"""Distillation-owned incremental MOT rollout."""

from .attention import TokenMetadataBatch, build_cache_visibility, segmented_orders
from .artifacts import rollout_artifact_due, save_rollout_artifacts
from .cache import KVSegment, SelfRolloutKVCache
from .provider import (
    GroundTruthProvider,
    GroundTruthStep,
    OfflineGroundTruthProvider,
    resolve_ground_truth_provider,
)
from .recorder import RecordedDenoiseState, SelfRolloutRecorder
from .result import RolloutResult
from .state import CacheSource, RolloutPhase, RolloutState


def self_rollout(*args, **kwargs):
    # Keep heavy model adapter imports lazy so cache/mask unit tests do not load
    # optional model attention extensions.
    from .engine import self_rollout as _self_rollout

    return _self_rollout(*args, **kwargs)

__all__ = [
    "CacheSource",
    "GroundTruthProvider",
    "GroundTruthStep",
    "KVSegment",
    "OfflineGroundTruthProvider",
    "RecordedDenoiseState",
    "RolloutPhase",
    "RolloutResult",
    "RolloutState",
    "SelfRolloutKVCache",
    "SelfRolloutRecorder",
    "TokenMetadataBatch",
    "build_cache_visibility",
    "segmented_orders",
    "resolve_ground_truth_provider",
    "rollout_artifact_due",
    "save_rollout_artifacts",
    "self_rollout",
]
