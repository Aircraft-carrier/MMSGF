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


def test_geometry_visibility_is_strictly_earlier_and_committed() -> None:
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
                3,
                stream=STREAM_GEOMETRY,
                noise=NOISE_GEOMETRY,
                committed=False,
                transaction_id=10,
            ),
            _meta(6, stream=STREAM_GEOMETRY, noise=NOISE_GEOMETRY, committed=True),
            _meta(4, stream=STREAM_VIDEO, noise=NOISE_CLEAN, committed=True),
        ]
    )
    mask = build_cache_visibility(query, keys, window_size=16)[0, 0]
    assert mask.tolist() == [True, False, False, False, False]


def test_history_block_prefill_matches_full_policy_and_framewise_prefill_does_not() -> None:
    frames = torch.arange(4)
    noisy = build_token_metadata(
        batch_size=1,
        frame_ids=frames,
        tokens_per_frame=1,
        stream_id=STREAM_VIDEO,
        noise_id=NOISE_NOISY,
        history_frames=4,
        chunk_size=4,
        device=torch.device("cpu"),
        committed=False,
        transaction_id=7,
        source_id=SOURCE_HISTORY,
        version_id=1,
    )
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
    block = TokenMetadataBatch.cat([noisy, clean])
    full_history_mask = build_cache_visibility(block, block, window_size=16)

    # The incremental history prefill is one transaction containing every
    # same-order history frame, so its rectangular mask is exactly the full
    # training-policy history submatrix.
    incremental_history_mask = build_cache_visibility(block, block, window_size=16)
    assert torch.equal(incremental_history_mask, full_history_mask)
    assert full_history_mask[0, 0, 3]

    # A frame-at-a-time prefill of H0 would expose only H0 and lose the allowed
    # H0->H1/H2/H3 same-order relation, proving why history must be one block.
    first_frame = TokenMetadataBatch.cat([noisy.slice(0, 1), clean.slice(0, 1)])
    framewise_mask = build_cache_visibility(first_frame, first_frame, window_size=16)
    assert framewise_mask.shape[-1] == 2
    assert full_history_mask.shape[-1] == 8


def test_current_video_and_action_follow_no_leak_and_phase_order() -> None:
    current_nv = _meta(
        5,
        stream=STREAM_VIDEO,
        noise=NOISE_NOISY,
        committed=False,
        transaction_id=11,
    )
    current_nv_key = current_nv
    current_cv = _meta(5, stream=STREAM_VIDEO, noise=NOISE_CLEAN, committed=True)
    current_g = _meta(5, stream=STREAM_GEOMETRY, noise=NOISE_GEOMETRY, committed=True)
    current_na = _meta(
        5,
        stream=STREAM_ACTION,
        noise=NOISE_NOISY,
        committed=False,
        transaction_id=12,
    )
    future_cv = _meta(6, stream=STREAM_VIDEO, noise=NOISE_CLEAN, committed=True)
    video_keys = TokenMetadataBatch.cat([current_nv_key, current_cv, current_g, future_cv])
    video_mask = build_cache_visibility(current_nv, video_keys, window_size=16)[0, 0]
    assert video_mask.tolist() == [True, False, False, False]

    action_keys = TokenMetadataBatch.cat([current_na, current_cv, current_g, future_cv])
    action_mask = build_cache_visibility(current_na, action_keys, window_size=16)[0, 0]
    assert action_mask.tolist() == [True, True, True, False]


def test_visibility_respects_window_and_other_transactions() -> None:
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
    assert mask.tolist() == [False, False]


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


def test_canonical_barrier_rejects_missing_layer_or_clean_stream() -> None:
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
    cache.append_committed(0, _segment(noisy, 1))
    cache.append_committed(0, _segment(clean, 2))
    cache.append_committed(1, _segment(noisy, 3))

    try:
        cache.assert_committed_frame(
            range(3),
            frame_id=4,
            stream_id=STREAM_VIDEO,
            noise_ids=(NOISE_NOISY, NOISE_CLEAN),
            source_id=SOURCE_PREDICTED,
            version_id=2,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "layer 1 missing noise_ids" in message
        assert "layer 2 missing noise_ids" in message
    else:
        raise AssertionError("incomplete per-layer canonical cache was accepted")
