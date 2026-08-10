"""Distillation-owned incremental MOT rollout."""

from .artifacts import rollout_artifact_due, save_rollout_artifacts
from .provider import (
    GroundTruthProvider,
    GroundTruthStep,
    OfflineGroundTruthProvider,
    resolve_ground_truth_provider,
)
from .recorder import RecordedDenoiseState, SelfRolloutRecorder
from .result import RolloutResult
from .state import CacheSource, RolloutPhase, RolloutState
from .transitions import (
    RolloutTransitionMode,
    SGFRolloutSchedule,
    build_sgf_rollout_schedule,
)


def self_rollout(*args, **kwargs):
    # Keep heavy model adapter imports lazy so cache-policy unit tests do not
    # load optional model attention extensions.
    from .engine import self_rollout as _self_rollout

    return _self_rollout(*args, **kwargs)

__all__ = [
    "CacheSource",
    "GroundTruthProvider",
    "GroundTruthStep",
    "OfflineGroundTruthProvider",
    "RecordedDenoiseState",
    "RolloutPhase",
    "RolloutResult",
    "RolloutState",
    "RolloutTransitionMode",
    "SGFRolloutSchedule",
    "SelfRolloutRecorder",
    "resolve_ground_truth_provider",
    "rollout_artifact_due",
    "save_rollout_artifacts",
    "self_rollout",
    "build_sgf_rollout_schedule",
]
