from unittest.mock import patch

import torch

from distillation.self_rollout.attention import (
    NOISE_CLEAN,
    STREAM_ACTION,
    STREAM_VIDEO,
    build_cache_selection,
    build_cache_visibility,
    build_token_metadata,
    incremental_attention,
    indexed_attention,
)
from distillation.self_rollout.cache import KVSegment, SelfRolloutKVCache


def _metadata(*, stream_id: int, tokens: int):
    return build_token_metadata(
        batch_size=1,
        frame_ids=torch.tensor([0]),
        tokens_per_frame=tokens,
        stream_id=stream_id,
        noise_id=NOISE_CLEAN,
        history_frames=1,
        chunk_size=1,
        device=torch.device("cpu"),
        committed=True,
    )


def test_indexed_attention_matches_dense_mask_with_ragged_validity() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(2, 4, 2, 3, generator=generator)
    key = torch.randn(2, 5, 2, 3, generator=generator)
    value = torch.randn(2, 5, 2, 3, generator=generator)
    query_valid = torch.tensor([[True, False, True, True], [False, True, True, False]])
    key_valid = torch.tensor([[True, False, True, False, True], [False] * 5])
    dense_mask = query_valid[:, :, None] & key_valid[:, None, :]

    expected = incremental_attention(query, key, value, dense_mask)
    actual, visible = indexed_attention(
        query, key, value, query_valid=query_valid, key_valid=key_valid
    )
    torch.testing.assert_close(actual, expected)
    assert torch.equal(visible, dense_mask.any(dim=-1))


def test_indexed_attention_uses_maskless_sdpa_for_full_validity() -> None:
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 5, 2, 4)
    value = torch.randn(1, 5, 2, 4)
    real_sdpa = torch.nn.functional.scaled_dot_product_attention
    seen_masks = []

    def record_sdpa(*args, **kwargs):
        seen_masks.append(kwargs.get("attn_mask"))
        return real_sdpa(*args, **kwargs)

    with patch("torch.nn.functional.scaled_dot_product_attention", side_effect=record_sdpa):
        output, visible = indexed_attention(
            query,
            key,
            value,
            query_valid=torch.ones(1, 3, dtype=torch.bool),
            key_valid=torch.ones(1, 5, dtype=torch.bool),
        )
    assert output.shape == query.shape
    assert visible.all()
    assert seen_masks == [None]


def test_cache_materialize_filters_video_and_action_segments() -> None:
    cache = SelfRolloutKVCache()
    video = torch.full((1, 2, 1, 1), 1.0)
    action = torch.full((1, 3, 1, 1), 2.0)
    for values, stream in ((video, STREAM_VIDEO), (action, STREAM_ACTION)):
        cache.append_committed(
            0,
            KVSegment(
                values,
                values + 0.25,
                _metadata(stream_id=stream, tokens=values.shape[1]),
                stream_id=stream,
            ),
        )

    key, value, metadata = cache.materialize(0, stream_id=STREAM_ACTION)
    assert torch.equal(key, action)
    assert torch.equal(value, action + 0.25)
    assert (metadata.stream_ids == STREAM_ACTION).all()


def test_va_visibility_reads_committed_and_current_transaction() -> None:
    query = build_token_metadata(
        batch_size=1,
        frame_ids=torch.tensor([1]),
        tokens_per_frame=2,
        stream_id=STREAM_ACTION,
        noise_id=NOISE_CLEAN,
        history_frames=1,
        chunk_size=1,
        device=torch.device("cpu"),
        valid_ids=torch.tensor([[True, False]]),
        committed=False,
        transaction_id=3,
    )
    committed = _metadata(stream_id=STREAM_VIDEO, tokens=2)
    current = query
    key = type(query).cat([committed, current])
    visibility = build_cache_visibility(query, key, window_size=16)
    query_valid, key_valid = build_cache_selection(query, key)

    assert torch.equal(visibility, query_valid[:, :, None] & key_valid[:, None, :])
