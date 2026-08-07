"""CPU-only phase trace for the AR MoT self-rollout path.

No checkpoint, dataset, GPU, or production model is required.  Video/action
inputs use the production UMI MOT packing, while only model depth and hidden
width are reduced for CPU execution.  This file also renders one grouped
rectangular attention-mask PNG for every rollout phase.

Run as a test::

    PYTHONPATH=. pytest -q -s \
        distillation/tests/test_mot_incremental_adapter_ipdb_trace.py

Run the printable trace directly::

    PYTHONPATH=. python \
        distillation/tests/test_mot_incremental_adapter_ipdb_trace.py

Enter ipdb immediately before the first AR rollout phase::

    MOT_AR_IPDB=1 PYTHONPATH=. python \
        distillation/tests/test_mot_incremental_adapter_ipdb_trace.py
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

import distillation.self_rollout.attention as attention_module
from distillation.model.autoregressive_mot import (
    AutoregressiveThreeDVAMOTTransformer3DModel,
)
from distillation.model.autoregressive_types import AutoregressiveProfile
from distillation.self_rollout.attention import (
    NOISE_CLEAN,
    NOISE_GEOMETRY,
    NOISE_NOISY,
    STREAM_ACTION,
    STREAM_GEOMETRY,
    STREAM_VIDEO,
    TokenMetadataBatch,
    build_token_metadata,
)
from distillation.self_rollout.cache import KVSegment
from distillation.self_rollout.state import CacheSource, RolloutState


STREAM_NAMES = {
    STREAM_VIDEO: "video",
    STREAM_ACTION: "action",
    STREAM_GEOMETRY: "geometry",
}
NOISE_NAMES = {
    NOISE_NOISY: "noisy",
    NOISE_CLEAN: "clean",
    NOISE_GEOMETRY: "geometry",
}

# Production UMI MOT packing.  The real model receives [B,48,8,V,14,14]
# cached latents and [B,20,8,16,1] actions.  History is frames 0..3, frame 4
# is the known anchor, and frames 5..7 are generated autoregressively.
BATCH_SIZE = 1
VIDEO_CHANNELS = 48
TOTAL_FRAMES = 8
HISTORY_FRAMES = 4
ANCHOR_FRAME = HISTORY_FRAMES
ROLLOUT_FRAME_IDS = tuple(range(ANCHOR_FRAME + 1, TOTAL_FRAMES))
VIEWS = 2
LATENT_HEIGHT = 14
LATENT_WIDTH = 14
VIDEO_PATCH_SIZE = 2
VIDEO_TOKENS_PER_FRAME = (
    VIEWS
    * (LATENT_HEIGHT // VIDEO_PATCH_SIZE)
    * (LATENT_WIDTH // VIDEO_PATCH_SIZE)
)
ACTION_CHANNELS = 20
ACTION_PER_FRAME = 16
ACTION_WIDTH = 1
GEOMETRY_TOKENS_PER_FRAME = 1
HIDDEN_DIM = 2
DEFAULT_MASK_DIR = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "mot_incremental_adapter_phases"
)


class _TextEmbedder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.linear_1.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear_1(value)


class _Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.heads = 1
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.norm_k = nn.Identity()
        nn.init.eye_(self.to_k.weight)
        nn.init.eye_(self.to_v.weight)


class _SubBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn1 = _Attention(dim)
        self.attn2 = _Attention(dim)


class _TinyMOTBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.video_block = _SubBlock(dim)
        self.action_block = _SubBlock(dim)
        self.geometry = None

    @staticmethod
    def _modulation(block, conditioning):
        del block, conditioning
        return None

    @staticmethod
    def _self_qkv(block, hidden, modulation, rotary):
        del block, modulation, rotary
        values = hidden[:, :, None, :]
        return values, values, values

    @staticmethod
    def _attention_output(attention, output):
        del attention
        return output.flatten(2, 3)

    @staticmethod
    def _finish_block(
        block,
        hidden,
        self_output,
        modulation,
        text_key,
        text_value,
    ):
        del block, modulation, text_key, text_value
        return hidden + self_output


class _ZeroRope(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            positions.shape[0],
            positions.shape[-1],
            self.dim,
            device=positions.device,
        )


class _TinyMOTModel(nn.Module):
    """Deterministic model with production packing and tiny hidden width."""

    def __init__(self, dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        if dim != HIDDEN_DIM:
            raise ValueError(f"the trace model requires hidden dim {HIDDEN_DIM}")
        self.weight = nn.Parameter(torch.zeros(()))
        self.mot_blocks = nn.ModuleList([_TinyMOTBlock(dim)])
        self.rope = _ZeroRope(dim)
        self.action_rope = _ZeroRope(dim)
        self.condition_embedder = SimpleNamespace(
            text_embedder=_TextEmbedder(dim)
        )

    @staticmethod
    def _embed_video(latents: torch.Tensor, stream_ids: torch.Tensor):
        batch, channels, frames, views, height, width = latents.shape
        expected = (
            BATCH_SIZE,
            VIDEO_CHANNELS,
            VIEWS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
        )
        if (batch, channels, views, height, width) != expected:
            raise ValueError(
                "video must use production [B,48,F,2,14,14] packing, "
                f"got {tuple(latents.shape)}"
            )
        if tuple(stream_ids.shape) != (batch, views):
            raise ValueError("stream_ids must be [B,V]")

        h_tokens = height // VIDEO_PATCH_SIZE
        w_tokens = width // VIDEO_PATCH_SIZE
        patches = latents.reshape(
            batch,
            channels,
            frames,
            views,
            h_tokens,
            VIDEO_PATCH_SIZE,
            w_tokens,
            VIDEO_PATCH_SIZE,
        ).permute(0, 2, 3, 4, 6, 1, 5, 7)
        patch_mean = patches.reshape(
            batch,
            frames * views * h_tokens * w_tokens,
            -1,
        ).mean(dim=-1, keepdim=True)
        hidden = torch.cat([patch_mean, patch_mean * 0.5 + 0.25], dim=-1)
        return hidden, (
            batch,
            frames,
            views,
            height,
            width,
            h_tokens,
            w_tokens,
        )

    @staticmethod
    def _embed_action(actions: torch.Tensor) -> torch.Tensor:
        expected = (
            BATCH_SIZE,
            ACTION_CHANNELS,
            actions.shape[2],
            ACTION_PER_FRAME,
            ACTION_WIDTH,
        )
        if tuple(actions.shape) != expected:
            raise ValueError(
                "action must use production [B,20,F,16,1] packing, "
                f"got {tuple(actions.shape)}"
            )
        tokens = actions.permute(0, 2, 3, 4, 1).reshape(
            actions.shape[0],
            -1,
            ACTION_CHANNELS,
        )
        token_mean = tokens.mean(dim=-1, keepdim=True)
        return torch.cat([token_mean, token_mean * 0.5 + 0.5], dim=-1)

    @staticmethod
    def _time_embed_repeated(
        timesteps: torch.Tensor,
        repeats_per_frame: int,
        dtype: torch.dtype,
        *,
        action: bool,
    ):
        del action
        length = timesteps.shape[1] * repeats_per_frame
        device = timesteps.device
        return (
            torch.zeros(timesteps.shape[0], length, 2, dtype=dtype, device=device),
            torch.zeros(timesteps.shape[0], length, 6, 2, dtype=dtype, device=device),
        )

    @staticmethod
    def _final_video(hidden, timesteps, shape):
        del timesteps
        batch, frames, views, height, width, h_tokens, w_tokens = shape
        patches = hidden.mean(dim=-1).reshape(
            batch,
            frames,
            views,
            h_tokens,
            w_tokens,
        )
        spatial = patches.repeat_interleave(VIDEO_PATCH_SIZE, dim=-2)
        spatial = spatial.repeat_interleave(VIDEO_PATCH_SIZE, dim=-1)
        if spatial.shape[-2:] != (height, width):
            raise RuntimeError("tiny video projection produced the wrong spatial shape")
        return spatial[:, None].expand(
            batch,
            VIDEO_CHANNELS,
            frames,
            views,
            height,
            width,
        ).contiguous()

    @staticmethod
    def _final_action(hidden, timesteps, action_shape):
        del timesteps
        batch, channels, frames, slots, width = action_shape
        values = hidden.mean(dim=-1).reshape(batch, frames, slots, width)
        return values[:, None].expand(
            batch,
            channels,
            frames,
            slots,
            width,
        ).contiguous()


@dataclass(frozen=True)
class _AttentionEvent:
    phase: str
    query: TokenMetadataBatch
    key: TokenMetadataBatch
    mask: torch.Tensor


class _VisibilityRecorder:
    """Record the dense reference implied by maskless cache selection."""

    def __init__(self) -> None:
        self.phase = "unassigned"
        self.events: list[_AttentionEvent] = []
        self._real_build = attention_module.build_cache_selection

    def __call__(
        self,
        query: TokenMetadataBatch,
        key: TokenMetadataBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_valid, key_valid = self._real_build(query, key)
        mask = query_valid[:, :, None] & key_valid[:, None, :]
        self.events.append(
            _AttentionEvent(
                phase=self.phase,
                query=query,
                key=key,
                mask=mask.detach().clone(),
            )
        )
        return query_valid, key_valid


@dataclass(frozen=True)
class _TraceResult:
    state: RolloutState
    events: list[_AttentionEvent]
    video_predictions: dict[int, torch.Tensor]
    action_predictions: dict[int, torch.Tensor]
    temporary_cache_counts: dict[str, tuple[int, int]]


def _video(values: list[float]) -> torch.Tensor:
    """Return production-shaped synthetic [1,48,F,2,14,14] latents."""

    frames = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1, 1, 1)
    return frames.expand(
        BATCH_SIZE,
        VIDEO_CHANNELS,
        len(values),
        VIEWS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    ).clone()


def _action(values: list[float]) -> torch.Tensor:
    """Return production-shaped synthetic [1,20,F,16,1] actions."""

    frames = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1, 1)
    return frames.expand(
        BATCH_SIZE,
        ACTION_CHANNELS,
        len(values),
        ACTION_PER_FRAME,
        ACTION_WIDTH,
    ).clone()


def _commit_geometry_groups_stub(
    state: RolloutState,
    model: _TinyMOTModel,
    *,
    frame_ids: list[int] | tuple[int, ...],
    source: CacheSource,
    version_id: int = 1,
) -> None:
    """Commit one mutually visible geometry transaction without constructing VGGTO."""

    frame_ids = tuple(int(frame_id) for frame_id in frame_ids)
    transaction_id = state.new_transaction_id()
    for layer_id in range(len(model.mot_blocks)):
        metadata = build_token_metadata(
            batch_size=1,
            frame_ids=torch.tensor(frame_ids),
            tokens_per_frame=GEOMETRY_TOKENS_PER_FRAME,
            stream_id=STREAM_GEOMETRY,
            noise_id=NOISE_GEOMETRY,
            history_frames=HISTORY_FRAMES,
            chunk_size=HISTORY_FRAMES,
            device=torch.device("cpu"),
            committed=False,
            transaction_id=transaction_id,
            source_id=int(source),
            version_id=version_id,
        )
        key = torch.full(
            (
                BATCH_SIZE,
                len(frame_ids) * GEOMETRY_TOKENS_PER_FRAME,
                1,
                HIDDEN_DIM,
            ),
            float(frame_ids[-1]),
        )
        state.mot_cache.append_transaction(
            layer_id,
            transaction_id,
            KVSegment(
                key,
                key + 0.5,
                metadata,
                stream_id=STREAM_GEOMETRY,
            ),
        )
        materialized_key, _value, key_metadata = state.mot_cache.materialize(
            layer_id,
            transaction_id=transaction_id,
            stream_id=STREAM_GEOMETRY,
        )
        del materialized_key
        attention_module.build_cache_selection(
            metadata,
            key_metadata,
        )
    state.mot_cache.commit_transaction(transaction_id, source_id=int(source))


def _attach_ar_mot_methods(model: _TinyMOTModel) -> _TinyMOTModel:
    model.generation_profile = AutoregressiveProfile(
        profile_name="segmented_history_strict_geometry_v1",
        profile_version=2,
        order_mode="segmented",
        history_frames=HISTORY_FRAMES,
        chunk_size=HISTORY_FRAMES,
        window_size=16,
        geometry_relation="segmented_order_causal",
        x_to_g_relation="strict_order",
    )
    for name in ("_normalize_frame_ids", "_normalize_timesteps"):
        setattr(
            model,
            name,
            getattr(AutoregressiveThreeDVAMOTTransformer3DModel, name),
        )
    for name in (
        "_text",
        "_stream_axes",
        "_video_rotary",
        "_action_rotary",
        "_video_input",
        "_action_input",
        "_run_stream",
        "_run_transaction",
        "_assert_clean_commit",
        "predict_video",
        "predict_action",
        "commit_video",
        "commit_action",
        "assert_video_commit",
        "assert_action_commit",
    ):
        setattr(
            model,
            name,
            MethodType(
                getattr(AutoregressiveThreeDVAMOTTransformer3DModel, name),
                model,
            ),
        )
    return model


def _run_trace() -> _TraceResult:
    model = _attach_ar_mot_methods(_TinyMOTModel())
    adapter = model
    state = RolloutState()
    recorder = _VisibilityRecorder()
    text_emb = torch.ones(BATCH_SIZE, 1, HIDDEN_DIM)
    stream_ids = torch.tensor([[0, 1]], dtype=torch.long)
    video_predictions: dict[int, torch.Tensor] = {}
    action_predictions: dict[int, torch.Tensor] = {}
    temporary_cache_counts: dict[str, tuple[int, int]] = {}

    if os.getenv("MOT_AR_IPDB"):
        import ipdb

        ipdb.set_trace()

    with patch.object(
        attention_module,
        "build_cache_selection",
        new=recorder,
    ):
        recorder.phase = "history.commit_video"
        adapter.commit_video(
            _video([0.0, 1.0, 2.0, 3.0]),
            frame_ids=list(range(HISTORY_FRAMES)),
            stream_ids=stream_ids,
            text_emb=text_emb,
            state=state,
            source=CacheSource.HISTORY,
            version_id=1,
        )
        recorder.phase = "history.commit_geometry"
        _commit_geometry_groups_stub(
            state,
            model,
            frame_ids=tuple(range(HISTORY_FRAMES)),
            source=CacheSource.HISTORY,
        )
        recorder.phase = "history.commit_action"
        adapter.commit_action(
            _action([10.0, 11.0, 12.0, 13.0]),
            frame_ids=list(range(HISTORY_FRAMES)),
            text_emb=text_emb,
            state=state,
            source=CacheSource.HISTORY,
            version_id=1,
        )

        recorder.phase = "anchor.commit_video"
        adapter.commit_video(
            _video([4.0]),
            frame_ids=[ANCHOR_FRAME],
            stream_ids=stream_ids,
            text_emb=text_emb,
            state=state,
            source=CacheSource.ANCHOR,
            version_id=1,
        )
        recorder.phase = "anchor.commit_geometry"
        _commit_geometry_groups_stub(
            state,
            model,
            frame_ids=[ANCHOR_FRAME],
            source=CacheSource.ANCHOR,
        )
        recorder.phase = "anchor.commit_action"
        adapter.commit_action(
            _action([14.0]),
            frame_ids=[ANCHOR_FRAME],
            text_emb=text_emb,
            state=state,
            source=CacheSource.ANCHOR,
            version_id=1,
        )

        for frame_id in ROLLOUT_FRAME_IDS:
            video_phase = f"rollout.frame{frame_id}.predict_video"
            before = state.mot_cache.committed_token_count(0)
            recorder.phase = video_phase
            video_prediction = adapter.predict_video(
                _video([0.0]),
                timestep=500,
                frame_id=frame_id,
                stream_ids=stream_ids,
                text_emb=text_emb,
                state=state,
            )
            after = state.mot_cache.committed_token_count(0)
            temporary_cache_counts[video_phase] = (before, after)
            video_predictions[frame_id] = video_prediction

            recorder.phase = f"rollout.frame{frame_id}.commit_video"
            adapter.commit_video(
                video_prediction,
                frame_ids=[frame_id],
                stream_ids=stream_ids,
                text_emb=text_emb,
                state=state,
                source=CacheSource.PREDICTED,
                version_id=1,
            )

            recorder.phase = f"rollout.frame{frame_id}.commit_geometry"
            _commit_geometry_groups_stub(
                state,
                model,
                frame_ids=[frame_id],
                source=CacheSource.PREDICTED,
            )

            action_phase = f"rollout.frame{frame_id}.predict_action"
            before = state.mot_cache.committed_token_count(0)
            recorder.phase = action_phase
            action_prediction = adapter.predict_action(
                _action([0.0]),
                timestep=500,
                frame_id=frame_id,
                text_emb=text_emb,
                state=state,
            )
            after = state.mot_cache.committed_token_count(0)
            temporary_cache_counts[action_phase] = (before, after)
            action_predictions[frame_id] = action_prediction

            recorder.phase = f"rollout.frame{frame_id}.commit_action"
            adapter.commit_action(
                action_prediction,
                frame_ids=[frame_id],
                text_emb=text_emb,
                state=state,
                source=CacheSource.PREDICTED,
                version_id=1,
            )

    return _TraceResult(
        state=state,
        events=recorder.events,
        video_predictions=video_predictions,
        action_predictions=action_predictions,
        temporary_cache_counts=temporary_cache_counts,
    )


def _select_event(result: _TraceResult, phase: str) -> _AttentionEvent:
    matches = [event for event in result.events if event.phase == phase]
    if len(matches) != 1:
        raise AssertionError(f"expected one event for {phase!r}, got {len(matches)}")
    return matches[0]


def _assert_trace(result: _TraceResult) -> None:
    expected_phases = [
        "history.commit_video",
        "history.commit_geometry",
        "history.commit_action",
        "anchor.commit_video",
        "anchor.commit_geometry",
        "anchor.commit_action",
    ]
    for frame_id in ROLLOUT_FRAME_IDS:
        expected_phases.extend(
            [
                f"rollout.frame{frame_id}.predict_video",
                f"rollout.frame{frame_id}.commit_video",
                f"rollout.frame{frame_id}.commit_geometry",
                f"rollout.frame{frame_id}.predict_action",
                f"rollout.frame{frame_id}.commit_action",
            ]
        )
    assert [event.phase for event in result.events] == expected_phases

    for event in result.events:
        query_streams = set(event.query.stream_ids.flatten().tolist())
        if query_streams <= {STREAM_VIDEO, STREAM_ACTION}:
            assert event.mask.all(), event.phase
            continue
        assert query_streams == {STREAM_GEOMETRY}
        visible_columns = event.mask.any(dim=1)[0]
        assert visible_columns.any()
        assert (
            event.key.stream_ids[0, visible_columns] == STREAM_GEOMETRY
        ).all(), event.phase

    assert result.temporary_cache_counts
    assert all(
        before == after
        for before, after in result.temporary_cache_counts.values()
    )
    result.state.mot_cache.assert_no_transactions()

    history_video = _select_event(result, "history.commit_video")
    history_geometry = _select_event(result, "history.commit_geometry")
    history_action = _select_event(result, "history.commit_action")
    assert history_video.query.seq_len == HISTORY_FRAMES * VIDEO_TOKENS_PER_FRAME
    assert history_geometry.query.seq_len == HISTORY_FRAMES
    history_geometry_keys = history_geometry.key.stream_ids == STREAM_GEOMETRY
    assert history_geometry_keys.sum().item() == HISTORY_FRAMES
    assert history_geometry.mask[:, :, history_geometry_keys[0]].all()
    assert history_action.query.seq_len == HISTORY_FRAMES * ACTION_PER_FRAME
    assert _select_event(result, "anchor.commit_video").query.seq_len == VIDEO_TOKENS_PER_FRAME
    assert _select_event(result, "anchor.commit_action").query.seq_len == ACTION_PER_FRAME

    for frame_id in ROLLOUT_FRAME_IDS:
        predicted_video = _select_event(
            result,
            f"rollout.frame{frame_id}.predict_video",
        )
        assert predicted_video.query.seq_len == VIDEO_TOKENS_PER_FRAME
        assert predicted_video.query.noise_ids.unique().tolist() == [NOISE_NOISY]
        current_video = (
            (predicted_video.key.frame_ids == frame_id)
            & (predicted_video.key.stream_ids == STREAM_VIDEO)
            & ~predicted_video.key.committed_ids
        )
        assert current_video.sum().item() == VIDEO_TOKENS_PER_FRAME
        assert predicted_video.mask[:, :, current_video[0]].all()

        predicted_action = _select_event(
            result,
            f"rollout.frame{frame_id}.predict_action",
        )
        assert predicted_action.query.seq_len == ACTION_PER_FRAME
        assert predicted_action.query.noise_ids.unique().tolist() == [NOISE_NOISY]
        committed_current_video = (
            (predicted_action.key.frame_ids == frame_id)
            & (predicted_action.key.stream_ids == STREAM_VIDEO)
            & predicted_action.key.committed_ids
        )
        committed_current_geometry = (
            (predicted_action.key.frame_ids == frame_id)
            & (predicted_action.key.stream_ids == STREAM_GEOMETRY)
            & predicted_action.key.committed_ids
        )
        assert committed_current_video.sum().item() == VIDEO_TOKENS_PER_FRAME
        assert committed_current_geometry.sum().item() == GEOMETRY_TOKENS_PER_FRAME
        assert predicted_action.mask[:, :, committed_current_video[0]].all()
        assert predicted_action.mask[:, :, committed_current_geometry[0]].all()

        assert result.video_predictions[frame_id].shape == (
            BATCH_SIZE,
            VIDEO_CHANNELS,
            1,
            VIEWS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
        )
        assert result.action_predictions[frame_id].shape == (
            BATCH_SIZE,
            ACTION_CHANNELS,
            1,
            ACTION_PER_FRAME,
            ACTION_WIDTH,
        )

    _key, _value, committed = result.state.mot_cache.materialize(0)
    x_tokens = (committed.stream_ids == STREAM_VIDEO) | (
        committed.stream_ids == STREAM_ACTION
    )
    assert committed.committed_ids.all()
    assert (committed.transaction_ids == -1).all()
    assert (committed.noise_ids[x_tokens] == NOISE_CLEAN).all()
    assert committed.seq_len == TOTAL_FRAMES * (
        VIDEO_TOKENS_PER_FRAME
        + ACTION_PER_FRAME
        + GEOMETRY_TOKENS_PER_FRAME
    )


def test_mot_incremental_adapter_across_rollout_phases(tmp_path: Path) -> None:
    result = _run_trace()
    _assert_trace(result)
    mask_paths = _render_phase_masks(result.events, tmp_path)
    assert len(mask_paths) == len(result.events)
    for mask_path in mask_paths:
        with Image.open(mask_path) as image:
            assert image.format == "PNG"
            assert image.width > 100
            assert image.height > 100


def _metadata_counts(metadata: TokenMetadataBatch) -> Counter[str]:
    counts: Counter[str] = Counter()
    for stream, noise, frame, committed, transaction in zip(
        metadata.stream_ids[0].tolist(),
        metadata.noise_ids[0].tolist(),
        metadata.frame_ids[0].tolist(),
        metadata.committed_ids[0].tolist(),
        metadata.transaction_ids[0].tolist(),
    ):
        lifetime = "committed" if committed else f"tx={transaction}"
        label = (
            f"{STREAM_NAMES[int(stream)]}/"
            f"{NOISE_NAMES[int(noise)]}/"
            f"frame={int(frame)}/"
            f"{lifetime}"
        )
        counts[label] += 1
    return counts


def _short_token_label(metadata: TokenMetadataBatch, index: int) -> str:
    stream = {
        STREAM_VIDEO: "V",
        STREAM_ACTION: "A",
        STREAM_GEOMETRY: "G",
    }[int(metadata.stream_ids[0, index].item())]
    noise = {
        NOISE_NOISY: "N",
        NOISE_CLEAN: "C",
        NOISE_GEOMETRY: "G",
    }[int(metadata.noise_ids[0, index].item())]
    frame = int(metadata.frame_ids[0, index].item())
    if bool(metadata.committed_ids[0, index].item()):
        lifetime = "C"
    else:
        lifetime = f"T{int(metadata.transaction_ids[0, index].item())}"
    return f"{stream}-{noise}-F{frame}-{lifetime}"


def _metadata_groups(
    metadata: TokenMetadataBatch,
) -> list[tuple[str, list[int]]]:
    grouped: dict[str, list[int]] = {}
    for index in range(metadata.seq_len):
        label = _short_token_label(metadata, index)
        grouped.setdefault(label, []).append(index)
    return list(grouped.items())


def _cell_color(visible_fraction: float) -> tuple[int, int, int]:
    if visible_fraction <= 0.0:
        return (255, 255, 255)
    if visible_fraction >= 1.0:
        return (37, 99, 235)
    return (245, 158, 11)


def _render_phase_mask(event: _AttentionEvent, output_path: Path) -> Path:
    """Render a Q-group by K-group visibility matrix for one phase."""

    query_groups = _metadata_groups(event.query)
    key_groups = _metadata_groups(event.key)
    cell_width = 52
    cell_height = 38
    left_margin = 130
    top_margin = 170
    right_margin = 30
    bottom_margin = 60
    image = Image.new(
        "RGB",
        (
            left_margin + len(key_groups) * cell_width + right_margin,
            top_margin + len(query_groups) * cell_height + bottom_margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((15, 15), event.phase, fill=(17, 24, 39), font=font)
    draw.text(
        (15, 35),
        f"raw mask [B,Q,K] = {tuple(event.mask.shape)}",
        fill=(55, 65, 81),
        font=font,
    )
    draw.text(
        (15, 52),
        "label = stream-noise-frame-lifetime; C=committed, Tn=transaction",
        fill=(55, 65, 81),
        font=font,
    )
    draw.text(
        (15, 69),
        "blue=fully visible, white=masked, orange=partially visible",
        fill=(55, 65, 81),
        font=font,
    )
    draw.text((15, top_margin - 18), "Query (Q)", fill=(17, 24, 39), font=font)
    draw.text((left_margin, 90), "Key (K)", fill=(17, 24, 39), font=font)

    for column, (label, _indices) in enumerate(key_groups):
        label_image = Image.new("RGBA", (130, 16), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_image)
        label_draw.text((0, 1), label, fill=(17, 24, 39), font=font)
        rotated = label_image.rotate(90, expand=True)
        x = left_margin + column * cell_width + (cell_width - rotated.width) // 2
        image.paste(rotated, (x, top_margin - rotated.height - 4), rotated)

    for row, (query_label, query_indices) in enumerate(query_groups):
        y = top_margin + row * cell_height
        draw.text((8, y + 12), query_label, fill=(17, 24, 39), font=font)
        query_selector = torch.tensor(query_indices, dtype=torch.long)
        for column, (_key_label, key_indices) in enumerate(key_groups):
            key_selector = torch.tensor(key_indices, dtype=torch.long)
            selected = event.mask[0].index_select(0, query_selector)
            selected = selected.index_select(1, key_selector)
            fraction = float(selected.float().mean().item())
            x = left_margin + column * cell_width
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                fill=_cell_color(fraction),
                outline=(156, 163, 175),
            )
            text_color = (255, 255, 255) if fraction >= 1.0 else (17, 24, 39)
            draw.text(
                (x + 18, y + 12),
                f"{fraction:.1f}",
                fill=text_color,
                font=font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _render_phase_masks(
    events: list[_AttentionEvent],
    output_dir: Path,
) -> list[Path]:
    paths = []
    for index, event in enumerate(events, start=1):
        filename = event.phase.replace(".", "_").replace("/", "_")
        paths.append(
            _render_phase_mask(
                event,
                output_dir / f"{index:02d}_{filename}.png",
            )
        )
    return paths


def main() -> None:
    result = _run_trace()
    _assert_trace(result)
    mask_dir = Path(os.getenv("MOT_AR_MASK_DIR", str(DEFAULT_MASK_DIR)))
    mask_paths = _render_phase_masks(result.events, mask_dir)

    print("AR MoT CPU rollout trace: PASS")
    print(
        "production video shape: "
        f"[{BATCH_SIZE},{VIDEO_CHANNELS},{TOTAL_FRAMES},{VIEWS},"
        f"{LATENT_HEIGHT},{LATENT_WIDTH}]"
    )
    print(
        "production action shape: "
        f"[{BATCH_SIZE},{ACTION_CHANNELS},{TOTAL_FRAMES},"
        f"{ACTION_PER_FRAME},{ACTION_WIDTH}]"
    )
    print("Each event below is one real visibility-mask call.")
    for event in result.events:
        print(f"\n[{event.phase}]")
        print(f"  query tokens: {_metadata_counts(event.query)}")
        print(f"  key tokens:   {_metadata_counts(event.key)}")
        print(
            "  mask shape:   "
            f"{tuple(event.mask.shape)}, "
            f"visible={int(event.mask.sum().item())}/{event.mask.numel()}"
        )

    print("\nTemporary prediction cache checks:")
    for phase, (before, after) in result.temporary_cache_counts.items():
        print(f"  {phase}: {before} -> {after}")
    print(
        "  final committed tokens in layer 0: "
        f"{result.state.mot_cache.committed_token_count(0)}"
    )
    for frame_id in ROLLOUT_FRAME_IDS:
        print(
            f"  frame {frame_id} video prediction: "
            f"{tuple(result.video_predictions[frame_id].shape)}"
        )
        print(
            f"  frame {frame_id} action prediction: "
            f"{tuple(result.action_predictions[frame_id].shape)}"
        )
    print(f"\nSaved {len(mask_paths)} phase masks to: {mask_dir}")


if __name__ == "__main__":
    main()
