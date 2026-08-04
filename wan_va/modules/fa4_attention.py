"""
FlashAttention-4 custom-mask helpers for 3DVA_MOT.

This module is intentionally import-safe on machines that do not yet have
`flash-attn-4` installed.  The public dataclasses and backend-dispatch hooks are
used by config loading, while the CuTe/FA4 imports happen lazily inside the
actual FA4 execution path.  That keeps the existing FlexAttention and SDPA paths
usable, but makes `backend="fa4"` fail fast with a clear environment error
instead of silently falling back.

The implementation follows the FA4 custom-mask pattern from the local
flash-attention CuTe examples: the attention kernel receives a CuTe `mask_mod`,
and block-sparse metadata is generated from the same mask with PyTorch
FlexAttention's `create_block_mask`.  The attention math is therefore FA4, while
Flex is only used to describe which 128x128 blocks are visited in forward and
backward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import torch
from torch.nn.attention.flex_attention import create_block_mask


STREAM_VIDEO = 0
STREAM_ACTION = 1
STREAM_GEOMETRY = 2
NOISE_NOISY = 0
NOISE_CLEAN = 1


@dataclass(frozen=True)
class ChunkCausalMaskSpec:
    """Structured VGGTO chunk-causal mask used by the FA4 backend."""

    frames: int
    frames_per_chunk: int
    tokens_per_frame: int
    backend: str = "fa4"
    token_valid_ids: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class _FA4Runtime:
    flash_attn_func: Any
    BlockSparseTensorsTorch: Any
    cutlass: Any
    cute: Any
    utils: Any


@lru_cache(maxsize=1)
def _load_fa4_runtime() -> _FA4Runtime:
    try:
        import cutlass
        import cutlass.cute as cute
        from flash_attn.cute import utils
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
        from flash_attn.cute.interface import flash_attn_func
    except Exception as exc:
        raise RuntimeError(
            "FlashAttention-4 backend requested, but flash-attn-4 CuTe dependencies "
            "are not importable. Install the local package with "
            "`pip install -e dependencies/flash-attention/flash_attn/cute[dev]`."
        ) from exc
    return _FA4Runtime(flash_attn_func, BlockSparseTensorsTorch, cutlass, cute, utils)


def _require_h100_cuda(q: torch.Tensor) -> None:
    if q.device.type != "cuda":
        raise RuntimeError("FlashAttention-4 backend requires CUDA tensors on H100/SM90; got non-CUDA input.")
    major = torch.cuda.get_device_capability(q.device)[0]
    if major != 9:
        raise RuntimeError(
            f"FlashAttention-4 custom-mask training is enabled only for H100/SM90 in this code path; got SM{major}x."
        )


def validate_fa4_training_environment(device: Optional[torch.device] = None) -> None:
    """Fail fast for configs that explicitly request FA4 custom-mask training."""

    _load_fa4_runtime()
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("FlashAttention-4 backend requires CUDA tensors on H100/SM90; got non-CUDA device.")
    major = torch.cuda.get_device_capability(device)[0]
    if major != 9:
        raise RuntimeError(
            f"FlashAttention-4 custom-mask training is enabled only for H100/SM90 in this code path; got SM{major}x."
        )


def chunk_causal_mask_from_spec(batch_size: int, spec: ChunkCausalMaskSpec, device: torch.device) -> torch.Tensor:
    seq_len = spec.frames * spec.tokens_per_frame
    frame_chunk_ids = torch.arange(spec.frames, device=device) // spec.frames_per_chunk
    token_chunk_ids = frame_chunk_ids.repeat_interleave(spec.tokens_per_frame)
    allow = token_chunk_ids[None, :] <= token_chunk_ids[:, None]
    mask = allow[None, None].expand(batch_size, 1, seq_len, seq_len)
    if spec.token_valid_ids is None:
        return mask

    token_valid = spec.token_valid_ids.to(device=device, dtype=torch.bool)
    if tuple(token_valid.shape) != (batch_size, seq_len):
        raise RuntimeError(f"ChunkCausalMaskSpec token_valid_ids must be [{batch_size},{seq_len}], got {tuple(token_valid.shape)}")
    valid_mask = token_valid[:, None, :, None] & token_valid[:, None, None, :]
    mask = mask & valid_mask
    eye = torch.eye(seq_len, dtype=torch.bool, device=device)[None, None]
    invalid_queries = ~token_valid[:, None, :, None]
    return torch.where(invalid_queries, eye, mask)


@lru_cache(maxsize=1)
def _cute_chunk_causal_mask():
    rt = _load_fa4_runtime()
    cutlass = rt.cutlass
    cute = rt.cute

    @cute.jit
    def _mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
        del batch, head, seqlen_info, aux_tensors
        tokens_per_chunk = cutlass.Int32(aux_scalars[0])
        return (kv_idx // tokens_per_chunk) <= (q_idx // tokens_per_chunk)

    return _mask


@lru_cache(maxsize=1)
def _cute_chunk_causal_valid_mask():
    rt = _load_fa4_runtime()
    cutlass = rt.cutlass
    cute = rt.cute
    utils = rt.utils

    @cute.jit
    def _mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
        token_valid_ids = aux_tensors[0]
        tokens_per_chunk = cutlass.Int32(aux_scalars[0])
        b = batch[0]
        h = head[0]
        q = q_idx[0]
        k = kv_idx[0]
        q_valid = utils.scalar_to_ssa(token_valid_ids[b, h, q], cutlass.Int32)
        k_valid = utils.scalar_to_ssa(token_valid_ids[b, h, k], cutlass.Int32)
        chunk_causal = (kv_idx // tokens_per_chunk) <= (q_idx // tokens_per_chunk)
        valid_pair = (q_valid != cutlass.Int32(0)) & (k_valid != cutlass.Int32(0))
        invalid_query_self = (q_valid == cutlass.Int32(0)) & (q_idx == kv_idx)
        return (chunk_causal & valid_pair) | invalid_query_self

    return _mask


@lru_cache(maxsize=1)
def _cute_mot_mask():
    rt = _load_fa4_runtime()
    cutlass = rt.cutlass
    cute = rt.cute
    utils = rt.utils

    @cute.jit
    def _mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
        seq_ids = aux_tensors[0]
        order_ids = aux_tensors[1]
        stream_ids = aux_tensors[2]
        noise_ids = aux_tensors[3]
        token_valid_ids = aux_tensors[4]
        b = batch[0]
        h = head[0]
        q = q_idx[0]
        k = kv_idx[0]
        window = cutlass.Int32(aux_scalars[0])

        q_seq = utils.scalar_to_ssa(seq_ids[b, h, q], cutlass.Int32)
        k_seq = utils.scalar_to_ssa(seq_ids[b, h, k], cutlass.Int32)
        q_order = utils.scalar_to_ssa(order_ids[b, h, q], cutlass.Int32)
        k_order = utils.scalar_to_ssa(order_ids[b, h, k], cutlass.Int32)
        q_stream = utils.scalar_to_ssa(stream_ids[b, h, q], cutlass.Int32)
        k_stream = utils.scalar_to_ssa(stream_ids[b, h, k], cutlass.Int32)
        q_noise = utils.scalar_to_ssa(noise_ids[b, h, q], cutlass.Int32)
        k_noise = utils.scalar_to_ssa(noise_ids[b, h, k], cutlass.Int32)
        q_valid = utils.scalar_to_ssa(token_valid_ids[b, h, q], cutlass.Int32)
        k_valid = utils.scalar_to_ssa(token_valid_ids[b, h, k], cutlass.Int32)

        valid = (q_seq == k_seq) & (q_seq >= cutlass.Int32(0))
        valid = valid & (q_valid != cutlass.Int32(0)) & (k_valid != cutlass.Int32(0))
        valid = valid & ((q_order - k_order) <= window) & ((k_order - q_order) <= window)

        x_query = (q_stream == cutlass.Int32(STREAM_VIDEO)) | (q_stream == cutlass.Int32(STREAM_ACTION))
        x_key = (k_stream == cutlass.Int32(STREAM_VIDEO)) | (k_stream == cutlass.Int32(STREAM_ACTION))
        g_query = q_stream == cutlass.Int32(STREAM_GEOMETRY)
        g_key = k_stream == cutlass.Int32(STREAM_GEOMETRY)

        clean_to_clean = (
            (q_noise == cutlass.Int32(NOISE_CLEAN))
            & (k_noise == cutlass.Int32(NOISE_CLEAN))
            & (k_order <= q_order)
        )
        noisy_to_clean = (
            (q_noise == cutlass.Int32(NOISE_NOISY))
            & (k_noise == cutlass.Int32(NOISE_CLEAN))
            & (k_order < q_order)
        )
        noisy_to_noisy = (
            (q_noise == cutlass.Int32(NOISE_NOISY))
            & (k_noise == cutlass.Int32(NOISE_NOISY))
            & (k_order == q_order)
        )
        x_to_x = x_query & x_key & (clean_to_clean | noisy_to_clean | noisy_to_noisy)

        # G is self-contained and uses the same chunk-causal clock as VA.
        g_to_g = g_query & g_key & (k_order <= q_order)
        clean_to_g = x_query & g_key & (q_noise == cutlass.Int32(NOISE_CLEAN)) & (k_order < q_order)
        noisy_to_g = x_query & g_key & (q_noise == cutlass.Int32(NOISE_NOISY)) & (k_order < q_order)
        return valid & (x_to_x | g_to_g | clean_to_g | noisy_to_g)

    return _mask


def _mot_aux_tensors(
    meta: Any,
    num_heads: int,
) -> list[torch.Tensor]:
    token_valid_ids = getattr(meta, "token_valid_ids", None)
    if token_valid_ids is None:
        token_valid_ids = torch.ones_like(meta.seq_ids, dtype=torch.bool)
    tensors = [meta.seq_ids, meta.order_ids, meta.stream_ids, meta.noise_ids, token_valid_ids]
    return [t[:, None, :].expand(-1, num_heads, -1).contiguous().to(dtype=torch.int32) for t in tensors]


def _mot_flex_mask_mod(meta: Any):
    def mask_mod(b, h, q_idx, kv_idx):
        q_stream = meta.stream_ids[b, q_idx]
        k_stream = meta.stream_ids[b, kv_idx]
        valid = (meta.seq_ids[b, q_idx] == meta.seq_ids[b, kv_idx]) & (meta.seq_ids[b, q_idx] >= 0)
        valid = valid & ((meta.order_ids[b, q_idx] - meta.order_ids[b, kv_idx]).abs() <= meta.window_size)
        if getattr(meta, "token_valid_ids", None) is not None:
            valid = valid & meta.token_valid_ids[b, q_idx] & meta.token_valid_ids[b, kv_idx]

        x_query = (q_stream == STREAM_VIDEO) | (q_stream == STREAM_ACTION)
        x_key = (k_stream == STREAM_VIDEO) | (k_stream == STREAM_ACTION)
        g_query = q_stream == STREAM_GEOMETRY
        g_key = k_stream == STREAM_GEOMETRY

        q_noise = meta.noise_ids[b, q_idx]
        k_noise = meta.noise_ids[b, kv_idx]
        q_order = meta.order_ids[b, q_idx]
        k_order = meta.order_ids[b, kv_idx]
        clean_to_clean = (q_noise == NOISE_CLEAN) & (k_noise == NOISE_CLEAN) & (k_order <= q_order)
        noisy_to_clean = (q_noise == NOISE_NOISY) & (k_noise == NOISE_CLEAN) & (k_order < q_order)
        noisy_to_noisy = (q_noise == NOISE_NOISY) & (k_noise == NOISE_NOISY) & (k_order == q_order)
        x_to_x = x_query & x_key & (clean_to_clean | noisy_to_clean | noisy_to_noisy)
        g_to_g = g_query & g_key & (k_order <= q_order)
        clean_to_g = x_query & g_key & (q_noise == NOISE_CLEAN) & (k_order < q_order)
        noisy_to_g = x_query & g_key & (q_noise == NOISE_NOISY) & (k_order < q_order)
        return valid & (x_to_x | g_to_g | clean_to_g | noisy_to_g)

    return mask_mod


def _chunk_causal_flex_mask_mod(tokens_per_chunk: int, token_valid_ids: Optional[torch.Tensor] = None):
    def mask_mod(b, h, q_idx, kv_idx):
        chunk_causal = (kv_idx // tokens_per_chunk) <= (q_idx // tokens_per_chunk)
        if token_valid_ids is None:
            return chunk_causal
        q_valid = token_valid_ids[b, q_idx]
        k_valid = token_valid_ids[b, kv_idx]
        return (chunk_causal & q_valid & k_valid) | (~q_valid & (q_idx == kv_idx))

    return mask_mod


def _block_sparse_from_flex(
    mask_mod,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
    block_size=(128, 128),
    mask_only: bool = False,
):
    rt = _load_fa4_runtime()
    bm = create_block_mask(
        mask_mod,
        batch_size,
        num_heads,
        seq_len,
        seq_len,
        device=device,
        BLOCK_SIZE=block_size,
    )
    (
        _seq_q,
        _seq_k,
        kv_mask_cnt,
        kv_mask_idx,
        full_kv_cnt,
        full_kv_idx,
        q_mask_cnt,
        q_mask_idx,
        full_q_cnt,
        full_q_idx,
        *_,
    ) = bm.as_tuple()
    if mask_only:
        kv_mask_cnt, kv_mask_idx = _merge_ordered_blocks(kv_mask_cnt, kv_mask_idx, full_kv_cnt, full_kv_idx)
        q_mask_cnt, q_mask_idx = _merge_ordered_blocks(q_mask_cnt, q_mask_idx, full_q_cnt, full_q_idx)
        full_kv_cnt, full_kv_idx = _empty_ordered_blocks_like(kv_mask_cnt, kv_mask_idx)
        full_q_cnt, full_q_idx = _empty_ordered_blocks_like(q_mask_cnt, q_mask_idx)
    fwd = rt.BlockSparseTensorsTorch(
        mask_block_cnt=kv_mask_cnt,
        mask_block_idx=kv_mask_idx,
        full_block_cnt=full_kv_cnt,
        full_block_idx=full_kv_idx,
        block_size=block_size,
    )
    bwd = rt.BlockSparseTensorsTorch(
        mask_block_cnt=q_mask_cnt,
        mask_block_idx=q_mask_idx,
        full_block_cnt=full_q_cnt,
        full_block_idx=full_q_idx,
        block_size=block_size,
    )
    return fwd, bwd


_BLOCK_MASK_HEADS = 1
_FA4_BLOCK_SIZE = (128, 128)
_MOT_BLOCK_CACHE: dict[tuple, tuple[Any, Any]] = {}


def _merge_ordered_blocks(
    mask_cnt: torch.Tensor,
    mask_idx: torch.Tensor,
    full_cnt: Optional[torch.Tensor],
    full_idx: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if full_cnt is None or full_idx is None:
        return mask_cnt, mask_idx
    merged_cnt = torch.empty_like(mask_cnt)
    flat_mask_cnt = mask_cnt.reshape(-1)
    flat_mask_idx = mask_idx.reshape(-1, mask_idx.shape[-1])
    flat_full_cnt = full_cnt.reshape(-1)
    flat_full_idx = full_idx.reshape(-1, full_idx.shape[-1])
    merged_rows = []
    max_blocks = 1
    for row in range(flat_mask_cnt.numel()):
        mask_n = int(flat_mask_cnt[row].item())
        full_n = int(flat_full_cnt[row].item())
        merged = torch.cat([flat_mask_idx[row, 0:mask_n], flat_full_idx[row, 0:full_n]]).unique(sorted=True)
        merged_rows.append(merged)
        max_blocks = max(max_blocks, int(merged.numel()))
    merged_idx = torch.empty(*mask_idx.shape[:-1], max_blocks, dtype=mask_idx.dtype, device=mask_idx.device)
    flat_merged_cnt = merged_cnt.reshape(-1)
    flat_merged_idx = merged_idx.reshape(-1, max_blocks)
    for row, merged in enumerate(merged_rows):
        count = int(merged.numel())
        flat_merged_cnt[row] = count
        if count:
            flat_merged_idx[row, 0:count] = merged
        if count < max_blocks:
            fill = int(merged[0].item()) if count else 0
            flat_merged_idx[row, count:] = fill
    return merged_cnt, merged_idx


def _empty_ordered_blocks_like(mask_cnt: torch.Tensor, mask_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros_like(mask_cnt), torch.zeros(
        *mask_idx.shape[:-1],
        1,
        dtype=mask_idx.dtype,
        device=mask_idx.device,
    )


def _mot_block_sparse(meta: Any, num_heads: int):
    # The MoT mask is head-independent. FA4 broadcasts one-head block-sparse
    # metadata to all QKV heads, avoiding large temporary Flex masks for L~10k.
    del num_heads
    block_size = _FA4_BLOCK_SIZE
    semantic_key = getattr(meta, "cache_key", None)
    mask_only = False
    if semantic_key is not None:
        key = (semantic_key, str(meta.device), _BLOCK_MASK_HEADS, block_size)
        cache = _MOT_BLOCK_CACHE
    else:
        structure_key = getattr(meta, "structure_cache_key", None)
        if structure_key is not None:
            key = ("mask_only", structure_key, str(meta.device), _BLOCK_MASK_HEADS, block_size)
            cache = _MOT_BLOCK_CACHE
            mask_only = True
        else:
            cache = getattr(meta, "_fa4_block_sparse_cache", None)
            if cache is None:
                cache = {}
                setattr(meta, "_fa4_block_sparse_cache", cache)
            key = (meta.batch_size, meta.seq_len, meta.window_size, str(meta.device), _BLOCK_MASK_HEADS, block_size)
    if key not in cache:
        mask_meta = meta
        if mask_only:
            mask_meta = meta.__class__(
                meta.seq_ids,
                meta.order_ids,
                meta.stream_ids,
                meta.noise_ids,
                meta.window_size,
                frame_ids=meta.frame_ids,
                token_valid_ids=None,
                cache_key=meta.cache_key,
                structure_cache_key=getattr(meta, "structure_cache_key", None),
            )
        cache[key] = _block_sparse_from_flex(
            _mot_flex_mask_mod(mask_meta),
            batch_size=meta.batch_size,
            num_heads=_BLOCK_MASK_HEADS,
            seq_len=meta.seq_len,
            device=meta.device,
            block_size=block_size,
            mask_only=mask_only,
        )
    return cache[key]


_CHUNK_BLOCK_CACHE: dict[tuple, tuple[Any, Any]] = {}


def _chunk_block_sparse(batch_size: int, num_heads: int, seq_len: int, spec: ChunkCausalMaskSpec, device: torch.device):
    del num_heads
    tokens_per_chunk = spec.frames_per_chunk * spec.tokens_per_frame
    mask_only = spec.token_valid_ids is not None
    key = (
        "mask_only" if mask_only else "exact",
        batch_size,
        _BLOCK_MASK_HEADS,
        seq_len,
        spec.frames,
        spec.frames_per_chunk,
        spec.tokens_per_frame,
        device,
        _FA4_BLOCK_SIZE,
    )
    if spec.token_valid_ids is not None:
        if key not in _CHUNK_BLOCK_CACHE:
            _CHUNK_BLOCK_CACHE[key] = _block_sparse_from_flex(
                _chunk_causal_flex_mask_mod(tokens_per_chunk),
                batch_size=batch_size,
                num_heads=_BLOCK_MASK_HEADS,
                seq_len=seq_len,
                device=device,
                block_size=_FA4_BLOCK_SIZE,
                mask_only=True,
            )
        return _CHUNK_BLOCK_CACHE[key]
    if key not in _CHUNK_BLOCK_CACHE:
        _CHUNK_BLOCK_CACHE[key] = _block_sparse_from_flex(
            _chunk_causal_flex_mask_mod(tokens_per_chunk),
            batch_size=batch_size,
            num_heads=_BLOCK_MASK_HEADS,
            seq_len=seq_len,
            device=device,
            block_size=_FA4_BLOCK_SIZE,
        )
    return _CHUNK_BLOCK_CACHE[key]


def fa4_attention_from_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    meta: Any,
) -> torch.Tensor:
    """Run VA-G MoT attention with FA4 custom mask.

    Inputs follow the local model convention `[B, L, H, D]`.  This is the FA4
    migration point for the register-only VA-G joint attention mask discussed
    in review; it consumes the same metadata as dense/Flex backends.
    """

    _require_h100_cuda(q)
    rt = _load_fa4_runtime()
    batch_size, seq_len, num_heads, head_dim = q.shape
    aux_tensors = _mot_aux_tensors(meta, num_heads)
    block_sparse_fwd, block_sparse_bwd = _mot_block_sparse(meta, num_heads)
    out, _ = rt.flash_attn_func(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        softmax_scale=1.0 / math.sqrt(head_dim),
        pack_gqa=False,
        score_mod=None,
        score_mod_bwd=None,
        mask_mod=_cute_mot_mask(),
        aux_tensors=aux_tensors,
        aux_scalars=[rt.cutlass.Int32(meta.window_size)],
        block_sparse_tensors=block_sparse_fwd,
        block_sparse_tensors_bwd=block_sparse_bwd,
        return_lse=False,
    )
    return out


def fa4_full_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Run ordinary unmasked full attention with FlashAttention-4.

    Inputs and outputs use the local `[B, L, H, D]` convention.  Cross-view
    VGGTO attention forms one independent timestamp sequence per batch row, so
    valid rows need no custom CuTe mask or block-sparse metadata.
    """

    _require_h100_cuda(q)
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise RuntimeError(
            "FlashAttention-4 full attention requires matching [B,L,H,D] Q/K/V; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    rt = _load_fa4_runtime()
    head_dim = q.shape[-1]
    out, _ = rt.flash_attn_func(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        softmax_scale=1.0 / math.sqrt(head_dim),
        pack_gqa=False,
        return_lse=False,
    )
    return out


def fa4_chunk_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spec: ChunkCausalMaskSpec) -> torch.Tensor:
    """Run VGGTO chunk-causal attention with FA4 custom mask."""

    _require_h100_cuda(q)
    rt = _load_fa4_runtime()
    batch_size, seq_len, num_heads, head_dim = q.shape
    expected_len = spec.frames * spec.tokens_per_frame
    if seq_len != expected_len:
        raise RuntimeError(
            f"FlashAttention-4 VGGTO chunk-causal mask expected L={expected_len} "
            f"from frames={spec.frames}, tokens_per_frame={spec.tokens_per_frame}; got L={seq_len}."
        )
    tokens_per_chunk = spec.frames_per_chunk * spec.tokens_per_frame
    aux_tensors = None
    mask_mod = _cute_chunk_causal_mask()
    if spec.token_valid_ids is not None:
        token_valid_ids = spec.token_valid_ids.to(device=q.device, dtype=torch.int32)
        if tuple(token_valid_ids.shape) != (batch_size, seq_len):
            raise RuntimeError(f"ChunkCausalMaskSpec token_valid_ids must be [{batch_size},{seq_len}], got {tuple(token_valid_ids.shape)}")
        aux_tensors = [token_valid_ids[:, None, :].expand(-1, num_heads, -1).contiguous()]
        mask_mod = _cute_chunk_causal_valid_mask()
    block_sparse_fwd, block_sparse_bwd = _chunk_block_sparse(batch_size, num_heads, seq_len, spec, q.device)
    out, _ = rt.flash_attn_func(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        softmax_scale=1.0 / math.sqrt(head_dim),
        pack_gqa=False,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        aux_scalars=[rt.cutlass.Int32(tokens_per_chunk)],
        block_sparse_tensors=block_sparse_fwd,
        block_sparse_tensors_bwd=block_sparse_bwd,
        return_lse=False,
    )
    return out
