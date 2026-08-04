from unittest.mock import patch

import torch

from distillation.self_rollout.attention import (
    NOISE_CLEAN,
    NOISE_GEOMETRY,
    STREAM_GEOMETRY,
    STREAM_VIDEO,
    build_cache_selection,
    build_cache_visibility,
    build_token_metadata,
    incremental_attention,
    indexed_attention,
)
from distillation.self_rollout.cache import KVSegment, SelfRolloutKVCache


def test_indexed_attention_matches_dense_mask_with_ragged_validity() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(2, 4, 2, 3, generator=generator)
    key = torch.randn(2, 5, 2, 3, generator=generator)
    value = torch.randn(2, 5, 2, 3, generator=generator)
    query_valid = torch.tensor(
        [[True, False, True, True], [False, True, True, False]]
    )
    key_valid = torch.tensor(
        [[True, False, True, False, True], [False, False, False, False, False]]
    )
    dense_mask = query_valid[:, :, None] & key_valid[:, None, :]

    expected = incremental_attention(query, key, value, dense_mask)
    actual, visible = indexed_attention(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
    )

    torch.testing.assert_close(actual, expected)
    assert torch.equal(visible, dense_mask.any(dim=-1))


def test_indexed_attention_uses_maskless_sdpa_for_full_validity() -> None:
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 5, 2, 4)
    value = torch.randn(1, 5, 2, 4)
    query_valid = torch.ones(1, 3, dtype=torch.bool)
    key_valid = torch.ones(1, 5, dtype=torch.bool)
    real_sdpa = torch.nn.functional.scaled_dot_product_attention
    seen_masks: list[torch.Tensor | None] = []

    def record_sdpa(*args, **kwargs):
        seen_masks.append(kwargs.get("attn_mask"))
        return real_sdpa(*args, **kwargs)

    with patch(
        "torch.nn.functional.scaled_dot_product_attention",
        side_effect=record_sdpa,
    ):
        output, visible = indexed_attention(
            query,
            key,
            value,
            query_valid=query_valid,
            key_valid=key_valid,
        )

    assert output.shape == query.shape
    assert visible.all()
    assert seen_masks == [None]


def _metadata(*, stream_id: int, tokens: int):
    noise_id = NOISE_GEOMETRY if stream_id == STREAM_GEOMETRY else NOISE_CLEAN
    return build_token_metadata(
        batch_size=1,
        frame_ids=torch.tensor([0]),
        tokens_per_frame=tokens,
        stream_id=stream_id,
        noise_id=noise_id,
        history_frames=1,
        chunk_size=1,
        device=torch.device("cpu"),
        committed=True,
    )


def test_cache_materialize_filters_stream_segments_before_concatenation() -> None:
    cache = SelfRolloutKVCache()
    video = torch.full((1, 2, 1, 1), 1.0)
    geometry = torch.full((1, 3, 1, 1), 2.0)
    cache.append_committed(
        0,
        KVSegment(
            video,
            video + 0.25,
            _metadata(stream_id=STREAM_VIDEO, tokens=2),
            stream_id=STREAM_VIDEO,
        ),
    )
    cache.append_committed(
        0,
        KVSegment(
            geometry,
            geometry + 0.25,
            _metadata(stream_id=STREAM_GEOMETRY, tokens=3),
            stream_id=STREAM_GEOMETRY,
        ),
    )

    key, value, metadata = cache.materialize(0, stream_id=STREAM_GEOMETRY)

    assert key.shape == (1, 3, 1, 1)
    assert torch.equal(key, geometry)
    assert torch.equal(value, geometry + 0.25)
    assert (metadata.stream_ids == STREAM_GEOMETRY).all()


def test_cache_selection_is_one_dimensional_and_matches_selected_cache() -> None:
    query = build_token_metadata(
        batch_size=2,
        frame_ids=torch.tensor([1]),
        tokens_per_frame=3,
        stream_id=STREAM_GEOMETRY,
        noise_id=NOISE_GEOMETRY,
        history_frames=1,
        chunk_size=1,
        device=torch.device("cpu"),
        valid_ids=torch.tensor([[True, False, True], [False, True, True]]),
        committed=False,
        transaction_id=4,
    )
    key = build_token_metadata(
        batch_size=2,
        frame_ids=torch.tensor([0, 1]),
        tokens_per_frame=2,
        stream_id=STREAM_GEOMETRY,
        noise_id=NOISE_GEOMETRY,
        history_frames=1,
        chunk_size=1,
        device=torch.device("cpu"),
        valid_ids=torch.tensor(
            [[True, True, False, True], [True, False, True, True]]
        ),
        committed=True,
    )

    query_valid, key_valid = build_cache_selection(query, key)

    assert query_valid.shape == (2, 3)
    assert key_valid.shape == (2, 4)
    assert torch.equal(query_valid, query.valid_ids)
    assert torch.equal(key_valid, key.valid_ids)


def test_geometry_stream_selection_matches_dense_mixed_cache_attention() -> None:
    generator = torch.Generator().manual_seed(11)
    cache = SelfRolloutKVCache()
    video = torch.randn(1, 2, 1, 4, generator=generator)
    geometry = torch.randn(1, 3, 1, 4, generator=generator)
    cache.append_committed(
        0,
        KVSegment(
            video,
            torch.randn(1, 2, 1, 4, generator=generator),
            _metadata(stream_id=STREAM_VIDEO, tokens=2),
            stream_id=STREAM_VIDEO,
        ),
    )
    cache.append_committed(
        0,
        KVSegment(
            geometry,
            torch.randn(1, 3, 1, 4, generator=generator),
            _metadata(stream_id=STREAM_GEOMETRY, tokens=3),
            stream_id=STREAM_GEOMETRY,
        ),
    )
    transaction_id = 5
    current_metadata = build_token_metadata(
        batch_size=1,
        frame_ids=torch.tensor([1]),
        tokens_per_frame=2,
        stream_id=STREAM_GEOMETRY,
        noise_id=NOISE_GEOMETRY,
        history_frames=1,
        chunk_size=1,
        device=torch.device("cpu"),
        valid_ids=torch.tensor([[True, False]]),
        committed=False,
        transaction_id=transaction_id,
    )
    query = torch.randn(1, 2, 1, 4, generator=generator)
    current_key = torch.randn(1, 2, 1, 4, generator=generator)
    current_value = torch.randn(1, 2, 1, 4, generator=generator)
    cache.append_transaction(
        0,
        transaction_id,
        KVSegment(
            current_key,
            current_value,
            current_metadata,
            stream_id=STREAM_GEOMETRY,
        ),
    )

    dense_key, dense_value, dense_metadata = cache.materialize(
        0,
        transaction_id=transaction_id,
    )
    dense_mask = build_cache_visibility(
        current_metadata,
        dense_metadata,
        window_size=16,
    )
    expected = incremental_attention(query, dense_key, dense_value, dense_mask)

    selected_key, selected_value, selected_metadata = cache.materialize(
        0,
        transaction_id=transaction_id,
        stream_id=STREAM_GEOMETRY,
    )
    query_valid, key_valid = build_cache_selection(
        current_metadata,
        selected_metadata,
    )
    actual, visible = indexed_attention(
        query,
        selected_key,
        selected_value,
        query_valid=query_valid,
        key_valid=key_valid,
    )

    torch.testing.assert_close(actual, expected)
    assert torch.equal(visible, dense_mask.any(dim=-1))
    assert selected_key.shape[1] == 5
    assert dense_key.shape[1] == 7
