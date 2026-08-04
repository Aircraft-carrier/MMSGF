from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from distillation.self_rollout.geometry_cache import (
    GeometryIncrementalAdapter,
    GeometryRolloutCache,
)
from distillation.self_rollout.state import CacheSource, RolloutState


class _Zero(nn.Module):
    def forward(self, value):
        return torch.zeros_like(value)


class _Raise(nn.Module):
    def forward(self, value):
        del value
        raise RuntimeError("injected geometry layer failure")


class _Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.num_heads = 1
        self.use_qk_norm = False
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Identity()
        with torch.no_grad():
            self.qkv.weight.copy_(torch.cat([torch.eye(dim)] * 3))
            self.proj.weight.copy_(torch.eye(dim))

    @staticmethod
    def apply_rope(query, key, rope):
        del rope
        return query, key


class _RelationBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.Identity()
        self.attn = _Attention(dim)
        self.ls1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.mlp = _Zero()
        self.ls2 = nn.Identity()


class _JointGeometry(nn.Module):
    def qkv_project(self, hidden, rotary):
        del rotary
        values = hidden[:, :, None, :]
        return values, values, values

    @staticmethod
    def attn_delta(output):
        return output.flatten(2, 3)

    @staticmethod
    def ffn_delta(hidden):
        return torch.zeros_like(hidden)


class _VGGTO(nn.Module):
    depth = 2
    register_attention_indices = {0}
    cached_layer_set = {1}
    patch_start_idx = 1

    def __init__(self):
        super().__init__()
        self.inter_frame_blocks = nn.ModuleList([nn.Identity(), _RelationBlock(2)])

    @staticmethod
    def encode_grouped(rgb, slot_valid_mask=None):
        del slot_valid_mask
        batch, groups, slots, views = rgb.shape[:4]
        base = rgb[:, :, :, :, 0, 0, 0].reshape(batch, groups * slots * views)
        tokens = torch.stack([base, base + 1], dim=-1)[:, :, None].expand(-1, -1, 2, -1).clone()
        return SimpleNamespace(
            tokens=tokens,
            patch_hw=(1, 1),
            image_hw=(1, 1),
            patch_token_start=1,
        )

    @staticmethod
    def run_frame_block(tokens, patch_hw, layer_id):
        del patch_hw
        return tokens + float(layer_id)

    @staticmethod
    def _run_cross_view_block(
        frame_tokens,
        patch_hw,
        layer_id,
        *,
        groups,
        group_size,
        views,
        slot_valid_mask,
    ):
        del patch_hw, layer_id, slot_valid_mask
        return frame_tokens.reshape(
            frame_tokens.shape[0],
            groups,
            group_size,
            views,
            frame_tokens.shape[2],
            frame_tokens.shape[3],
        )

    @staticmethod
    def _full_token_rope(patch_hw, images_per_sequence, device):
        del patch_hw
        return (
            torch.zeros(images_per_sequence * 2, 2, device=device),
            torch.ones(images_per_sequence * 2, 2, device=device),
        )


class _Rope(nn.Module):
    def forward(self, grid):
        return torch.zeros(grid.shape[0], grid.shape[-1], 2, device=grid.device)


class _MOTBlock(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vggto = _VGGTO()
        self.rope = _Rope()
        self.mot_blocks = nn.ModuleList(
            [_MOTBlock(_JointGeometry()), _MOTBlock(None)]
        )


def _rgb(value: float) -> torch.Tensor:
    return torch.full((1, 1, 2, 1, 3, 1, 1), value)


def test_geometry_adapter_commits_both_joint_and_relation_caches_and_truncates() -> None:
    adapter = GeometryIncrementalAdapter(
        _Model(),
        history_frames=2,
        chunk_size=2,
        window_size=16,
    )
    state = RolloutState(geometry_cache=GeometryRolloutCache())
    first = adapter.encode_and_commit(
        _rgb(1),
        frame_id=0,
        slot_valid_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        state=state,
        source=CacheSource.HISTORY,
        version_id=1,
    )
    second = adapter.encode_and_commit(
        _rgb(2),
        frame_id=1,
        slot_valid_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        state=state,
        source=CacheSource.HISTORY,
        version_id=1,
    )

    assert first.final_tokens.shape == second.final_tokens.shape == (1, 2, 2, 2)
    assert torch.isfinite(first.final_tokens).all()
    adapter.assert_geometry_commit(
        state,
        frame_id=0,
        source=CacheSource.HISTORY,
        version_id=1,
    )
    adapter.assert_geometry_commit(
        state,
        frame_id=1,
        source=CacheSource.HISTORY,
        version_id=1,
    )
    assert state.mot_cache.has_frame(0)
    assert state.mot_cache.has_frame(1)
    assert state.geometry_cache.relation_cache.has_frame(0)
    assert state.geometry_cache.relation_cache.has_frame(1)
    assert set(state.geometry_cache.frames) == {0, 1}
    state.truncate_from(1)
    assert state.mot_cache.has_frame(0) and not state.mot_cache.has_frame(1)
    assert state.geometry_cache.relation_cache.has_frame(0)
    assert not state.geometry_cache.relation_cache.has_frame(1)
    assert set(state.geometry_cache.frames) == {0}


def test_geometry_failure_rolls_back_joint_relation_and_frame_state() -> None:
    model = _Model()
    model.vggto.inter_frame_blocks[1].mlp = _Raise()
    adapter = GeometryIncrementalAdapter(
        model,
        history_frames=2,
        chunk_size=2,
        window_size=16,
    )
    state = RolloutState(geometry_cache=GeometryRolloutCache())

    try:
        adapter.encode_and_commit(
            _rgb(1),
            frame_id=0,
            slot_valid_mask=torch.ones(1, 1, 2, dtype=torch.bool),
            state=state,
            source=CacheSource.PREDICTED,
            version_id=1,
        )
    except RuntimeError as exc:
        assert "injected geometry layer failure" in str(exc)
    else:
        raise AssertionError("injected geometry failure did not propagate")

    state.mot_cache.assert_no_transactions()
    state.geometry_cache.relation_cache.assert_no_transactions()
    assert state.mot_cache.layer_ids == ()
    assert state.geometry_cache.relation_cache.layer_ids == ()
    assert state.geometry_cache.frames == {}


def test_geometry_adapter_fails_fast_on_unsupported_topology() -> None:
    try:
        GeometryIncrementalAdapter(
            object(),
            history_frames=2,
            chunk_size=2,
            window_size=16,
        )
    except TypeError as exc:
        assert "ThreeDVAMOT attributes" in str(exc)
    else:
        raise AssertionError("unsupported geometry topology did not fail fast")
