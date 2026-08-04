from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from distillation.self_rollout.mot_adapter import MOTIncrementalAdapter
from distillation.self_rollout.state import CacheSource, RolloutState


class _TextEmbedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.linear_1.weight)

    def forward(self, value):
        return self.linear_1(value)


class _Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.heads = 1
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.norm_k = nn.Identity()
        self.attn1_marker = True
        nn.init.eye_(self.to_k.weight)
        nn.init.eye_(self.to_v.weight)


class _SubBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn1 = _Attention(dim)
        self.attn2 = _Attention(dim)


class _MOTBlock(nn.Module):
    def __init__(self, dim: int):
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
    def _attention_output(attn, output):
        del attn
        return output.flatten(2, 3)

    @staticmethod
    def _finish_block(block, hidden, self_output, modulation, text_key, text_value):
        del block, modulation, text_key, text_value
        return hidden + self_output


class _Rope(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, positions):
        return torch.zeros(
            positions.shape[0],
            positions.shape[-1],
            self.dim,
            device=positions.device,
        )


class _TinyModel(nn.Module):
    def __init__(self, dim: int = 2):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.mot_blocks = nn.ModuleList([_MOTBlock(dim), _MOTBlock(dim)])
        self.rope = _Rope(dim)
        self.action_rope = _Rope(dim)
        self.condition_embedder = SimpleNamespace(text_embedder=_TextEmbedder(dim))

    def _embed_video(self, latents, stream_ids):
        del stream_ids
        batch, channels, frames, views, height, width = latents.shape
        assert height == width == 1
        hidden = latents[:, :, :, :, 0, 0].permute(0, 2, 3, 1).reshape(
            batch, frames * views, channels
        )
        return hidden, (batch, frames, views, height, width, 1, 1)

    @staticmethod
    def _embed_action(actions):
        return actions[:, :, :, :, 0].permute(0, 2, 3, 1).reshape(
            actions.shape[0], -1, actions.shape[1]
        )

    @staticmethod
    def _time_embed_repeated(timesteps, repeats_per_frame, dtype, *, action):
        del action
        length = timesteps.shape[1] * repeats_per_frame
        return (
            torch.zeros(timesteps.shape[0], length, 2, dtype=dtype),
            torch.zeros(timesteps.shape[0], length, 6, 2, dtype=dtype),
        )

    @staticmethod
    def _final_video(hidden, timesteps, shape):
        del timesteps
        batch, frames, views, _height, _width, _ht, _wt = shape
        return hidden.reshape(batch, frames, views, hidden.shape[-1]).permute(
            0, 3, 1, 2
        )[:, :, :, :, None, None]

    @staticmethod
    def _final_action(hidden, timesteps, action_shape):
        del timesteps
        batch, channels, frames, slots, width = action_shape
        return hidden.reshape(batch, frames, slots, width, channels).permute(
            0, 4, 1, 2, 3
        )


class _FailingMOTBlock(_MOTBlock):
    @staticmethod
    def _finish_block(block, hidden, self_output, modulation, text_key, text_value):
        del block, hidden, self_output, modulation, text_key, text_value
        raise RuntimeError("injected layer failure")


def test_mot_adapter_prediction_is_temporary_and_canonical_commit_is_persistent() -> None:
    model = _TinyModel()
    adapter = MOTIncrementalAdapter(
        model,
        history_frames=2,
        chunk_size=2,
        window_size=16,
    )
    state = RolloutState()
    text = torch.ones(1, 1, 2)
    stream_ids = torch.zeros(1, 1, dtype=torch.long)
    history = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1, 1, 1).expand(-1, 2, -1, -1, -1, -1)

    adapter.commit_video(
        history,
        frame_ids=[0, 1],
        stream_ids=stream_ids,
        text_emb=text,
        state=state,
        source=CacheSource.HISTORY,
        version_id=1,
    )
    assert state.mot_cache.committed_token_count(0) == 4
    before = state.mot_cache.snapshot()
    output = adapter.predict_video(
        torch.ones(1, 2, 1, 1, 1, 1),
        timestep=500,
        frame_id=2,
        stream_ids=stream_ids,
        text_emb=text,
        state=state,
    )
    assert output.shape == (1, 2, 1, 1, 1, 1)
    state.mot_cache.assert_no_transactions()
    assert state.mot_cache.committed_token_count(0) == 4
    assert state.mot_cache.snapshot().committed.keys() == before.committed.keys()

    adapter.commit_video(
        output,
        frame_ids=[2],
        stream_ids=stream_ids,
        text_emb=text,
        state=state,
        source=CacheSource.PREDICTED,
        version_id=1,
    )
    assert state.mot_cache.committed_token_count(0) == 6
    adapter.assert_video_commit(
        state,
        frame_id=2,
        source=CacheSource.PREDICTED,
        version_id=1,
    )


def test_action_commit_uses_na_and_ca_and_respects_token_validity() -> None:
    adapter = MOTIncrementalAdapter(
        _TinyModel(),
        history_frames=2,
        chunk_size=2,
        window_size=16,
    )
    state = RolloutState()
    actions = torch.ones(1, 2, 1, 2, 1)
    valid = torch.tensor([[[[[True], [False]]], [[[True], [False]]]]])
    adapter.commit_action(
        actions,
        frame_ids=[2],
        text_emb=torch.ones(1, 1, 2),
        state=state,
        source=CacheSource.ANCHOR,
        version_id=1,
        valid_mask=valid,
    )
    _key, _value, metadata = state.mot_cache.materialize(0)
    assert metadata.seq_len == 4
    assert metadata.valid_ids.tolist() == [[True, False, True, False]]
    adapter.assert_action_commit(
        state,
        frame_id=2,
        source=CacheSource.ANCHOR,
        version_id=1,
    )


def test_video_commit_failure_discards_every_layer_transaction() -> None:
    model = _TinyModel()
    model.mot_blocks[1] = _FailingMOTBlock(2)
    adapter = MOTIncrementalAdapter(
        model,
        history_frames=2,
        chunk_size=2,
        window_size=16,
    )
    state = RolloutState()

    try:
        adapter.commit_video(
            torch.ones(1, 2, 1, 1, 1, 1),
            frame_ids=[2],
            stream_ids=torch.zeros(1, 1, dtype=torch.long),
            text_emb=torch.ones(1, 1, 2),
            state=state,
            source=CacheSource.PREDICTED,
            version_id=1,
        )
    except RuntimeError as exc:
        assert "injected layer failure" in str(exc)
    else:
        raise AssertionError("injected canonical commit failure did not propagate")

    state.mot_cache.assert_no_transactions()
    assert state.mot_cache.layer_ids == ()


def test_cond_and_uncond_predictions_never_share_temporary_kv() -> None:
    adapter = MOTIncrementalAdapter(
        _TinyModel(),
        history_frames=2,
        chunk_size=2,
        window_size=16,
    )
    state = RolloutState()
    sample = torch.ones(1, 2, 1, 1, 1, 1)
    common = dict(
        timestep=500,
        frame_id=2,
        stream_ids=torch.zeros(1, 1, dtype=torch.long),
        state=state,
    )

    adapter.predict_video(sample, text_emb=torch.ones(1, 1, 2), **common)
    state.mot_cache.assert_no_transactions()
    assert state.mot_cache.layer_ids == ()
    adapter.predict_video(sample, text_emb=torch.zeros(1, 1, 2), **common)
    state.mot_cache.assert_no_transactions()
    assert state.mot_cache.layer_ids == ()


def test_mot_adapter_fails_fast_instead_of_falling_back_to_full_inference() -> None:
    try:
        MOTIncrementalAdapter(
            object(),
            history_frames=2,
            chunk_size=2,
            window_size=16,
        )
    except TypeError as exc:
        assert "ThreeDVAMOT-compatible" in str(exc)
    else:
        raise AssertionError("unsupported MoT topology did not fail fast")
