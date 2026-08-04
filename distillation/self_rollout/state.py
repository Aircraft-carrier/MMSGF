"""Semantic rollout state, checkpoints, and prediction logs."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import torch

from .attention import STREAM_ACTION, STREAM_GEOMETRY, STREAM_VIDEO
from .cache import KVCacheSnapshot, SelfRolloutKVCache


class RolloutPhase(IntEnum):
    LATENT = 0
    GEOMETRY = 1
    ACTION = 2


class CacheSource(IntEnum):
    HISTORY = 0
    ANCHOR = 1
    PREDICTED = 2
    GROUND_TRUTH = 3


@dataclass(slots=True)
class SemanticFrameState:
    frame_id: int
    video_latent: torch.Tensor | None = None
    geometry_rgb: torch.Tensor | None = None
    geometry_state: Any | None = None
    action: torch.Tensor | None = None
    video_source: CacheSource | None = None
    geometry_source: CacheSource | None = None
    action_source: CacheSource | None = None
    video_version: int = 0
    geometry_version: int = 0
    action_version: int = 0


@dataclass(slots=True)
class PredictionLog:
    video: dict[int, torch.Tensor] = field(default_factory=dict)
    geometry: dict[int, torch.Tensor] = field(default_factory=dict)
    action: dict[int, torch.Tensor] = field(default_factory=dict)
    replacements: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RolloutSnapshot:
    mot_cache: KVCacheSnapshot
    geometry_snapshot: Any
    semantic_frames: dict[int, SemanticFrameState]
    next_transaction_id: int
    generator_state: torch.Tensor | None


class RolloutState:
    def __init__(
        self,
        *,
        mot_cache: SelfRolloutKVCache | None = None,
        geometry_cache: Any | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.mot_cache = mot_cache or SelfRolloutKVCache()
        self.geometry_cache = geometry_cache
        self.semantic_frames: dict[int, SemanticFrameState] = {}
        self.predictions = PredictionLog()
        self.generator = generator
        self._next_transaction_id = 0
        self.checkpoints: dict[int, RolloutSnapshot] = {}
        self.phase_checkpoints: dict[tuple[int, RolloutPhase], RolloutSnapshot] = {}

    def new_transaction_id(self) -> int:
        transaction_id = self._next_transaction_id
        self._next_transaction_id += 1
        return transaction_id

    def frame(self, frame_id: int) -> SemanticFrameState:
        frame_id = int(frame_id)
        return self.semantic_frames.setdefault(frame_id, SemanticFrameState(frame_id))

    def snapshot(self) -> RolloutSnapshot:
        geometry_snapshot = None
        if self.geometry_cache is not None:
            geometry_snapshot = self.geometry_cache.snapshot()
        generator_state = None
        if self.generator is not None:
            generator_state = self.generator.get_state().clone()
        return RolloutSnapshot(
            mot_cache=self.mot_cache.snapshot(),
            geometry_snapshot=geometry_snapshot,
            semantic_frames={
                frame_id: copy.copy(frame)
                for frame_id, frame in self.semantic_frames.items()
            },
            next_transaction_id=self._next_transaction_id,
            generator_state=generator_state,
        )

    def restore(self, snapshot: RolloutSnapshot) -> None:
        self.mot_cache.restore(snapshot.mot_cache)
        if self.geometry_cache is not None:
            self.geometry_cache.restore(snapshot.geometry_snapshot)
        self.semantic_frames = {
            frame_id: copy.copy(frame)
            for frame_id, frame in snapshot.semantic_frames.items()
        }
        self._next_transaction_id = snapshot.next_transaction_id
        if self.generator is not None and snapshot.generator_state is not None:
            self.generator.set_state(snapshot.generator_state)

    def save_checkpoint_before(self, frame_id: int) -> None:
        snapshot = self.snapshot()
        self.checkpoints[int(frame_id)] = snapshot
        self.phase_checkpoints[(int(frame_id), RolloutPhase.LATENT)] = snapshot

    def save_phase_checkpoint(self, frame_id: int, phase: RolloutPhase) -> None:
        self.phase_checkpoints[(int(frame_id), RolloutPhase(phase))] = self.snapshot()

    def restore_phase_checkpoint(self, frame_id: int, phase: RolloutPhase) -> None:
        key = (int(frame_id), RolloutPhase(phase))
        try:
            snapshot = self.phase_checkpoints[key]
        except KeyError as exc:
            raise KeyError(
                f"no checkpoint exists for frame={frame_id}, phase={RolloutPhase(phase).name}"
            ) from exc
        self.restore(snapshot)
        self.phase_checkpoints = {
            checkpoint_key: checkpoint
            for checkpoint_key, checkpoint in self.phase_checkpoints.items()
            if checkpoint_key[0] < key[0]
            or (
                checkpoint_key[0] == key[0]
                and checkpoint_key[1] <= key[1]
            )
        }
        self.checkpoints = {
            checkpoint_frame: checkpoint
            for checkpoint_frame, checkpoint in self.checkpoints.items()
            if checkpoint_frame <= frame_id
        }

    def delete_predicted(self, frame_id: int, phase: RolloutPhase) -> None:
        """Delete one predicted phase and every downstream derived cache."""

        frame_id = int(frame_id)
        phase = RolloutPhase(phase)
        frame = self.semantic_frames.get(frame_id)
        if frame is None:
            raise KeyError(f"frame {frame_id} has no semantic state")
        source = {
            RolloutPhase.LATENT: frame.video_source,
            RolloutPhase.GEOMETRY: frame.geometry_source,
            RolloutPhase.ACTION: frame.action_source,
        }[phase]
        if source != CacheSource.PREDICTED:
            raise ValueError(
                f"frame {frame_id} {phase.name.lower()} is not predicted: {source}"
            )
        self.restore_phase_checkpoint(frame_id, phase)

    def restore_before(self, frame_id: int) -> None:
        try:
            snapshot = self.checkpoints[int(frame_id)]
        except KeyError as exc:
            raise KeyError(f"no checkpoint exists before frame {frame_id}") from exc
        self.restore(snapshot)

    def truncate_from(self, frame_id: int) -> None:
        frame_id = int(frame_id)
        self.mot_cache.truncate_from(frame_id)
        if self.geometry_cache is not None:
            self.geometry_cache.truncate_from(frame_id)
        self.semantic_frames = {
            index: frame
            for index, frame in self.semantic_frames.items()
            if index < frame_id
        }
        self.checkpoints = {
            index: checkpoint
            for index, checkpoint in self.checkpoints.items()
            if index <= frame_id
        }
        self.phase_checkpoints = {
            key: checkpoint
            for key, checkpoint in self.phase_checkpoints.items()
            if key[0] < frame_id
        }

    def assert_cache_versions(self) -> None:
        """Reject committed K/V derived from stale semantic frame versions."""

        self.mot_cache.assert_no_transactions()
        caches = [("mot", self.mot_cache)]
        if self.geometry_cache is not None:
            relation_cache = getattr(self.geometry_cache, "relation_cache", None)
            if relation_cache is not None:
                relation_cache.assert_no_transactions()
                caches.append(("geometry_relation", relation_cache))

        for cache_name, cache in caches:
            for layer_id, segment in cache.committed_segments():
                metadata = segment.metadata
                for stream_id, component in (
                    (STREAM_VIDEO, "video"),
                    (STREAM_GEOMETRY, "geometry"),
                    (STREAM_ACTION, "action"),
                ):
                    stream_mask = metadata.stream_ids == stream_id
                    if not bool(stream_mask.any().item()):
                        continue
                    for frame_id in metadata.frame_ids[stream_mask].unique().tolist():
                        frame = self.semantic_frames.get(int(frame_id))
                        if frame is None:
                            raise RuntimeError(
                                f"{cache_name} layer {layer_id} contains frame {frame_id} "
                                "without semantic state"
                            )
                        expected_version = getattr(frame, f"{component}_version")
                        expected_source = getattr(frame, f"{component}_source")
                        frame_mask = stream_mask & (metadata.frame_ids == int(frame_id))
                        versions = metadata.version_ids[frame_mask]
                        sources = metadata.source_ids[frame_mask]
                        if not bool((versions == int(expected_version)).all().item()):
                            raise RuntimeError(
                                f"stale {component} K/V in {cache_name} layer {layer_id}, "
                                f"frame {frame_id}: cache versions={versions.unique().tolist()}, "
                                f"semantic version={expected_version}"
                            )
                        if expected_source is None or not bool(
                            (sources == int(expected_source)).all().item()
                        ):
                            raise RuntimeError(
                                f"stale {component} source in {cache_name} layer {layer_id}, "
                                f"frame {frame_id}: cache sources={sources.unique().tolist()}, "
                                f"semantic source={expected_source}"
                            )

    def public_view(self) -> dict[str, Any]:
        return {
            "frames": self.semantic_frames,
            "predictions": self.predictions,
        }
