from __future__ import annotations

import torch

from distillation.self_rollout.attention import (
    NOISE_CLEAN,
    NOISE_GEOMETRY,
    NOISE_NOISY,
    SOURCE_HISTORY,
    SOURCE_PREDICTED,
    STREAM_ACTION,
    STREAM_GEOMETRY,
    STREAM_VIDEO,
    TokenMetadataBatch,
    build_cache_visibility,
    build_token_metadata,
    incremental_attention,
    segmented_orders,
)
from distillation.self_rollout.cache import KVSegment, SelfRolloutKVCache
from distillation.self_rollout.geometry_cache import GeometryRolloutCache
from distillation.self_rollout.state import CacheSource, RolloutPhase, RolloutState
from distillation.mask_profile import _segmented_frame_orders


def _meta(
    frame: int,
    *,
    stream: int,
    noise: int,
    committed: bool,
    transaction_id: int = -1,
    source: int = SOURCE_HISTORY,
    version: int = 1,
) -> TokenMetadataBatch:
    return build_token_metadata(
        batch_size=1,
        frame_ids=torch.tensor([frame]),
        tokens_per_frame=1,
        stream_id=stream,
        noise_id=noise,
        history_frames=4,
        chunk_size=4,
        device=torch.device("cpu"),
        committed=committed,
        transaction_id=transaction_id,
        source_id=source,
        version_id=version,
    )


def _segment(meta: TokenMetadataBatch, value: float) -> KVSegment:
    tensor = torch.full((1, meta.seq_len, 1, 2), value)
    return KVSegment(tensor, tensor + 10, meta)


def test_segmented_orders_are_stable_beyond_default_window() -> None:
    frames = torch.arange(10)
    orders = segmented_orders(frames, history_frames=4, chunk_size=4)
    assert orders.tolist() == [0, 0, 0, 0, 2, 4, 6, 8, 10, 12]
    assert segmented_orders(frames[:8], history_frames=4, chunk_size=4).tolist() == orders[:8].tolist()
    training_video, training_action = _segmented_frame_orders(
        num_frames=10,
        history_frames=4,
        chunk_size=4,
        device=torch.device("cpu"),
    )
    assert training_video.tolist() == orders.tolist()
    assert training_action.tolist() == [value + 1 for value in orders.tolist()]


def test_cache_transaction_commit_discard_truncate_and_restore() -> None:
    cache = SelfRolloutKVCache()
    history = _meta(0, stream=STREAM_VIDEO, noise=NOISE_CLEAN, committed=True)
    predicted = _meta(
        4,
        stream=STREAM_VIDEO,
        noise=NOISE_NOISY,
        committed=False,
        transaction_id=7,
        source=SOURCE_PREDICTED,
    )
    cache.append_committed(0, _segment(history, 1))
    cache.append_transaction(0, 7, _segment(predicted, 2))
    key, value, metadata = cache.materialize(0, transaction_id=7)
    assert key.shape[1] == value.shape[1] == metadata.seq_len == 2

    snapshot = cache.snapshot()
    cache.discard_transaction(7)
    assert cache.materialize(0)[2].seq_len == 1
    cache.restore(snapshot)
    cache.commit_transaction(7, source_id=SOURCE_PREDICTED)
    assert cache.materialize(0)[2].committed_ids.all()
    assert cache.has_frame(4)
    cache.truncate_from(4)
    assert not cache.has_frame(4)
    assert cache.has_frame(0)


def test_truncate_splits_a_multi_frame_prefill_segment() -> None:
    cache = SelfRolloutKVCache()
    metadata = build_token_metadata(
        batch_size=1,
        frame_ids=torch.tensor([0, 1, 2]),
        tokens_per_frame=1,
        stream_id=STREAM_VIDEO,
        noise_id=NOISE_CLEAN,
        history_frames=3,
        chunk_size=3,
        device=torch.device("cpu"),
        committed=True,
        source_id=SOURCE_HISTORY,
    )
    cache.append_committed(0, _segment(metadata, 1))
    cache.truncate_from(2)
    _key, _value, remaining = cache.materialize(0)
    assert remaining.frame_ids.tolist() == [[0, 1]]


def test_geometry_visibility_uses_commit_order_and_stays_geometry_only() -> None:
    query = _meta(
        5,
        stream=STREAM_GEOMETRY,
        noise=NOISE_GEOMETRY,
        committed=False,
        transaction_id=10,
    )
    keys = TokenMetadataBatch.cat(
        [
            _meta(4, stream=STREAM_GEOMETRY, noise=NOISE_GEOMETRY, committed=True),
            _meta(5, stream=STREAM_GEOMETRY, noise=NOISE_GEOMETRY, committed=True),
            _meta(
                5,
                stream=STREAM_GEOMETRY,
                noise=NOISE_GEOMETRY,
                committed=False,
                transaction_id=10,
            ),
            _meta(
                5,
                stream=STREAM_GEOMETRY,
                noise=NOISE_GEOMETRY,
                committed=False,
                transaction_id=11,
            ),
            _meta(4, stream=STREAM_VIDEO, noise=NOISE_CLEAN, committed=True),
        ]
    )
    mask = build_cache_visibility(query, keys, window_size=16)[0, 0]
    assert mask.tolist() == [True, True, True, False, False]


def test_multi_frame_transaction_is_one_current_commit_unit() -> None:
    frames = torch.arange(4)
    clean = build_token_metadata(
        batch_size=1,
        frame_ids=frames,
        tokens_per_frame=1,
        stream_id=STREAM_VIDEO,
        noise_id=NOISE_CLEAN,
        history_frames=4,
        chunk_size=4,
        device=torch.device("cpu"),
        committed=False,
        transaction_id=7,
        source_id=SOURCE_HISTORY,
        version_id=1,
    )
    mask = build_cache_visibility(clean, clean, window_size=16)
    assert mask.shape == (1, 4, 4)
    assert mask.all()

    first_frame = clean.slice(0, 1)
    framewise_mask = build_cache_visibility(first_frame, first_frame, window_size=16)
    assert framewise_mask.shape == (1, 1, 1)


def test_video_and_action_read_all_committed_kv_and_their_own_transaction() -> None:
    current_nv = _meta(
        5,
        stream=STREAM_VIDEO,
        noise=NOISE_NOISY,
        committed=False,
        transaction_id=11,
    )
    current_nv_key = current_nv
    committed_video = _meta(5, stream=STREAM_VIDEO, noise=NOISE_CLEAN, committed=True)
    committed_geometry = _meta(5, stream=STREAM_GEOMETRY, noise=NOISE_GEOMETRY, committed=True)
    current_na = _meta(
        5,
        stream=STREAM_ACTION,
        noise=NOISE_NOISY,
        committed=False,
        transaction_id=12,
    )
    committed_action = _meta(6, stream=STREAM_ACTION, noise=NOISE_CLEAN, committed=True)
    other_transaction = _meta(
        5,
        stream=STREAM_VIDEO,
        noise=NOISE_NOISY,
        committed=False,
        transaction_id=99,
    )
    video_keys = TokenMetadataBatch.cat(
        [current_nv_key, committed_video, committed_geometry, committed_action, other_transaction]
    )
    video_mask = build_cache_visibility(current_nv, video_keys, window_size=16)[0, 0]
    assert video_mask.tolist() == [True, True, True, True, False]

    action_keys = TokenMetadataBatch.cat(
        [current_na, committed_video, committed_geometry, committed_action, other_transaction]
    )
    action_mask = build_cache_visibility(current_na, action_keys, window_size=16)[0, 0]
    assert action_mask.tolist() == [True, True, True, True, False]


def test_visibility_ignores_frame_window_but_rejects_other_transactions() -> None:
    query = _meta(
        9,
        stream=STREAM_ACTION,
        noise=NOISE_NOISY,
        committed=False,
        transaction_id=20,
    )
    old = _meta(0, stream=STREAM_VIDEO, noise=NOISE_CLEAN, committed=True)
    other_tx = _meta(
        9,
        stream=STREAM_ACTION,
        noise=NOISE_NOISY,
        committed=False,
        transaction_id=21,
    )
    mask = build_cache_visibility(
        query,
        TokenMetadataBatch.cat([old, other_tx]),
        window_size=4,
    )[0, 0]
    assert mask.tolist() == [True, False]


def test_incremental_attention_handles_empty_or_fully_masked_keys() -> None:
    query = torch.randn(1, 2, 1, 4)
    empty = torch.empty(1, 0, 1, 4)
    out = incremental_attention(
        query,
        empty,
        empty,
        torch.zeros(1, 2, 0, dtype=torch.bool),
    )
    torch.testing.assert_close(out, torch.zeros_like(query))

    key = torch.randn(1, 1, 1, 4)
    out = incremental_attention(
        query,
        key,
        key,
        torch.zeros(1, 2, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(out, torch.zeros_like(query))


def _state_with_complete_predicted_frame(frame_id: int = 4) -> RolloutState:
    geometry_cache = GeometryRolloutCache()
    state = RolloutState(geometry_cache=geometry_cache)
    state.save_checkpoint_before(frame_id)

    video = _meta(
        frame_id,
        stream=STREAM_VIDEO,
        noise=NOISE_CLEAN,
        committed=True,
        source=SOURCE_PREDICTED,
    )
    state.mot_cache.append_committed(0, _segment(video, 1))
    frame = state.frame(frame_id)
    frame.video_latent = torch.ones(1)
    frame.video_source = CacheSource.PREDICTED
    frame.video_version = 1
    state.save_phase_checkpoint(frame_id, RolloutPhase.GEOMETRY)

    geometry = _meta(
        frame_id,
        stream=STREAM_GEOMETRY,
        noise=NOISE_GEOMETRY,
        committed=True,
        source=SOURCE_PREDICTED,
    )
    state.mot_cache.append_committed(0, _segment(geometry, 2))
    geometry_cache.relation_cache.append_committed(1, _segment(geometry, 3))
    geometry_cache.frames[frame_id] = object()
    frame.geometry_rgb = torch.ones(1)
    frame.geometry_state = geometry_cache.frames[frame_id]
    frame.geometry_source = CacheSource.PREDICTED
    frame.geometry_version = 1
    state.save_phase_checkpoint(frame_id, RolloutPhase.ACTION)

    action = _meta(
        frame_id,
        stream=STREAM_ACTION,
        noise=NOISE_CLEAN,
        committed=True,
        source=SOURCE_PREDICTED,
    )
    state.mot_cache.append_committed(0, _segment(action, 4))
    frame.action = torch.ones(1)
    frame.action_source = CacheSource.PREDICTED
    frame.action_version = 1
    state.assert_cache_versions()
    return state


def test_phase_deletion_removes_exact_component_and_downstream_caches() -> None:
    latent_state = _state_with_complete_predicted_frame()
    latent_state.delete_predicted(4, RolloutPhase.LATENT)
    assert 4 not in latent_state.semantic_frames
    assert not latent_state.mot_cache.has_frame(4)
    assert not latent_state.geometry_cache.relation_cache.has_frame(4)
    assert 4 not in latent_state.geometry_cache.frames

    geometry_state = _state_with_complete_predicted_frame()
    geometry_state.delete_predicted(4, RolloutPhase.GEOMETRY)
    frame = geometry_state.frame(4)
    assert frame.video_latent is not None
    assert frame.geometry_rgb is None and frame.action is None
    assert geometry_state.mot_cache.has_frame(4)
    assert not geometry_state.geometry_cache.relation_cache.has_frame(4)
    assert 4 not in geometry_state.geometry_cache.frames
    metadata = geometry_state.mot_cache.materialize(0)[2]
    assert set(metadata.stream_ids.flatten().tolist()) == {STREAM_VIDEO}

    action_state = _state_with_complete_predicted_frame()
    action_state.delete_predicted(4, RolloutPhase.ACTION)
    frame = action_state.frame(4)
    assert frame.video_latent is not None and frame.geometry_rgb is not None
    assert frame.action is None
    metadata = action_state.mot_cache.materialize(0)[2]
    assert set(metadata.stream_ids.flatten().tolist()) == {
        STREAM_VIDEO,
        STREAM_GEOMETRY,
    }
    assert action_state.geometry_cache.relation_cache.has_frame(4)


def test_cache_version_assertion_rejects_stale_committed_kv() -> None:
    state = RolloutState()
    metadata = _meta(
        4,
        stream=STREAM_VIDEO,
        noise=NOISE_CLEAN,
        committed=True,
        source=SOURCE_PREDICTED,
        version=1,
    )
    state.mot_cache.append_committed(0, _segment(metadata, 1))
    frame = state.frame(4)
    frame.video_latent = torch.ones(1)
    frame.video_source = CacheSource.PREDICTED
    frame.video_version = 2

    try:
        state.assert_cache_versions()
    except RuntimeError as exc:
        assert "stale video K/V" in str(exc)
    else:
        raise AssertionError("stale K/V version was not rejected")


def test_canonical_barrier_requires_one_clean_stream_per_layer() -> None:
    cache = SelfRolloutKVCache()
    noisy = _meta(
        4,
        stream=STREAM_VIDEO,
        noise=NOISE_NOISY,
        committed=True,
        source=SOURCE_PREDICTED,
        version=2,
    )
    clean = _meta(
        4,
        stream=STREAM_VIDEO,
        noise=NOISE_CLEAN,
        committed=True,
        source=SOURCE_PREDICTED,
        version=2,
    )
    cache.append_committed(0, _segment(clean, 1))
    cache.append_committed(1, _segment(noisy, 3))

    try:
        cache.assert_committed_frame(
            range(3),
            frame_id=4,
            stream_id=STREAM_VIDEO,
            noise_ids=(NOISE_CLEAN,),
            source_id=SOURCE_PREDICTED,
            version_id=2,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "layer 1 missing noise_ids" in message
        assert "layer 2 missing noise_ids" in message
    else:
        raise AssertionError("incomplete per-layer canonical cache was accepted")
