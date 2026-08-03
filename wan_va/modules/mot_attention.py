"""
MoT attention mask utilities for 3DVA_MOT.

This file owns the visibility of multi-modal tokens.
It deliberately separates "token order in the packed sequence" from "what a
query is allowed to read".  The physical order used by the model is convenient
for splitting tensors, but all semantic safety comes from the metadata here:

    noisy_video, clean_video, G, noisy_action, clean_action, pad

The important invariants encoded in this file are:

* G is a self-contained geometry stream.  A G query can only read G keys; it can
  never read LingBot video, action, or text tokens. G->G uses the same
  chunk-causal clock as VA, so geometry is bidirectional inside one generated
  chunk and cannot read future chunks.
* LingBot VA/action tokens may read G, but only according to the block-wise
  no-leak policy.  
  Noisy video cannot read current/future G.  
  Noisy action can read current G because that preserves the original LingBot inverse-dynamics
  semantics where action denoising can observe the current visual condition.
* View id is not a causal axis. Multi-view tokens at the same latent frame/block
  are governed by stream/noise rules, not by a fake camera ordering. Video uses
  `t -> view -> patch`; unpooled G uses `t -> slot -> view -> register`. The
  corresponding `*_tokens_per_frame` must include every token inside one `t`, or
  the per-frame `order_ids` below will no longer match physical timestamps.
* Dense masks are kept as a readable reference and fallback.  The actual
  training path uses FA4 or FlexAttention BlockMask generated from the same
  metadata rules, so every backend has the same visibility contract.

关键 pair 的实际结果：

  NV1 -> G1 : False   noisy video 不能看当前 G, 防泄露
  NV1 -> G0 : True    noisy video 可以看过去 G
  NA1 -> G1 : True    code NA1 是物理 a0, 可以看 v0/v1 对应的 G0/G1
  NA1 -> G2 : False   NA1 不能看再下一帧 G2
  NA1 -> CV1: True    code NA1 可以看 clean video V1
  NA1 -> CA0: True    CA0 是 dataset padding/condition action slot
  NA1 -> CA1: False   noisy action 不能看自己的 clean action target
  NV2 -> CA1: True    code CA1 是物理 a0, 对当前 query 来说已经完成
  NV2 -> CA2: False   code CA2 是物理 a1, 需要当前 query 才能反推
  G(H)  -> NV(H): False   G query 不能看 video/action
  G(H)  -> G(T) : False   history G 不能看 target chunk
  G(T0) -> G(T3): True    target chunk 内的 G 双向可见
  G(T)  -> G(H) : True    target G 可以看 history chunk
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from .fa4_attention import fa4_attention_from_meta


STREAM_VIDEO = 0
STREAM_ACTION = 1
STREAM_GEOMETRY = 2
STREAM_PAD = -1

NOISE_NOISY = 0
NOISE_CLEAN = 1
NOISE_GEOMETRY = 2
NOISE_PAD = -1


@dataclass
class MOTMaskMetadata:
    """Flat per-token metadata used to decide visibility.

    `seq_ids` isolates samples in a packed batch. `order_ids` is the block
    clock used by causal masking. Video/G use even order `2 * block_id`; action
    uses odd order `2 * block_id + 1`, matching LingBot's original video-then-
    action MDP structure. `stream_ids` and `noise_ids` encode semantic token
    type, so no-leak rules never depend on fragile physical tensor offsets.
    """

    seq_ids: torch.Tensor
    order_ids: torch.Tensor
    stream_ids: torch.Tensor
    noise_ids: torch.Tensor
    window_size: int
    frame_ids: Optional[torch.Tensor] = None
    token_valid_ids: Optional[torch.Tensor] = None
    cache_key: Optional[tuple] = None
    structure_cache_key: Optional[tuple] = None

    @property
    def device(self):
        return self.seq_ids.device

    @property
    def batch_size(self):
        return self.seq_ids.shape[0]

    @property
    def seq_len(self):
        return self.seq_ids.shape[1]


def _none_if_all_valid(token_valid_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if token_valid_ids is None:
        return None
    if bool(token_valid_ids.all().item()):
        return None
    return token_valid_ids


def _same_batch(meta: MOTMaskMetadata, q_idx, kv_idx):
    return (meta.seq_ids[:, q_idx] == meta.seq_ids[:, kv_idx]) & (meta.seq_ids[:, q_idx] >= 0)


def _window(meta: MOTMaskMetadata, q_idx, kv_idx):
    return (meta.order_ids[:, q_idx] - meta.order_ids[:, kv_idx]).abs() <= meta.window_size


def _token_valid(meta: MOTMaskMetadata, q_idx, kv_idx):
    if meta.token_valid_ids is None:
        return True
    return meta.token_valid_ids[:, q_idx] & meta.token_valid_ids[:, kv_idx]


def _x_to_x(meta: MOTMaskMetadata, q_idx, kv_idx):
    """LingBot-compatible noisy/clean rule for VA/action tokens.

    This intentionally matches `FlexAttnFunc._get_mask_mod` in
    `wan_va/modules/model.py` and the MoT mask. Video/G tokens
    get even order `2 * block_id`; action tokens get odd order `2 * block_id + 1`.
    The original dataset pads one action slot at the beginning, so code action
    `A_i` represents the physical transition `v_{i-1} -> v_i` (`A0` is
    padding/condition context). With that offset, the order rule implements
    inverse dynamics: code `NA1` can read `CV0/CV1`, while `NV1` cannot read
    code `CA1`.

    clean  -> clean:  k_order <= q_order
    noisy  -> clean:  k_order <  q_order
    noisy  -> noisy:  k_order == q_order
    """

    q_noise = meta.noise_ids[:, q_idx]
    k_noise = meta.noise_ids[:, kv_idx]
    q_order = meta.order_ids[:, q_idx]
    k_order = meta.order_ids[:, kv_idx]

    clean_to_clean = (q_noise == NOISE_CLEAN) & (k_noise == NOISE_CLEAN) & (k_order <= q_order)
    noisy_to_clean = (q_noise == NOISE_NOISY) & (k_noise == NOISE_CLEAN) & (k_order < q_order)
    noisy_to_noisy = (q_noise == NOISE_NOISY) & (k_noise == NOISE_NOISY) & (k_order == q_order)
    return clean_to_clean | noisy_to_clean | noisy_to_noisy


def _g_to_g(meta: MOTMaskMetadata, q_idx, kv_idx):
    """G is self-contained and chunk-causal, matching the VA generation unit."""

    return meta.order_ids[:, kv_idx] <= meta.order_ids[:, q_idx]


def _x_to_g(meta: MOTMaskMetadata, q_idx, kv_idx):
    """Visibility from LingBot VA/action queries into VGGTO geometry keys.

    G shares clean video's position in the LingBot causal order. Therefore noisy
    queries can read only strict-past G by order id, while clean queries can read
    current/past G. Because action order is odd and video/G order is even, code
    `NA1` reads `G0/G1`; this is the same dataset-offset inverse-dynamics
    convention described in `_x_to_x`.
    """

    q_noise = meta.noise_ids[:, q_idx]
    q_order = meta.order_ids[:, q_idx]
    k_order = meta.order_ids[:, kv_idx]

    clean_to_g = (q_noise == NOISE_CLEAN) & (k_order < q_order)
    noisy_to_g = (q_noise == NOISE_NOISY) & (k_order < q_order)
    return clean_to_g | noisy_to_g


def build_dense_mot_mask(meta: MOTMaskMetadata) -> torch.Tensor:
    """Build a dense `[B, Q, K]` boolean mask from canonical metadata.

    Dense masks are intentionally explicit and easy to inspect. The model
    training path uses FA4 or FlexAttention; this dense form is retained as a
    compact reference for exact semantic checks.
    """

    q_idx = torch.arange(meta.seq_len, device=meta.device)[:, None]
    kv_idx = torch.arange(meta.seq_len, device=meta.device)[None, :]

    q_stream = meta.stream_ids[:, q_idx]
    k_stream = meta.stream_ids[:, kv_idx]

    valid = _same_batch(meta, q_idx, kv_idx) & _window(meta, q_idx, kv_idx) & _token_valid(meta, q_idx, kv_idx)
    x_query = (q_stream == STREAM_VIDEO) | (q_stream == STREAM_ACTION)
    x_key = (k_stream == STREAM_VIDEO) | (k_stream == STREAM_ACTION)
    g_query = q_stream == STREAM_GEOMETRY
    g_key = k_stream == STREAM_GEOMETRY

    allow = torch.zeros((meta.batch_size, meta.seq_len, meta.seq_len), dtype=torch.bool, device=meta.device)
    allow = allow | (x_query & x_key & _x_to_x(meta, q_idx, kv_idx))
    allow = allow | (g_query & g_key & _g_to_g(meta, q_idx, kv_idx))
    allow = allow | (x_query & g_key & _x_to_g(meta, q_idx, kv_idx))
    return allow & valid


def build_x_metadata(
    *,
    batch_size: int,
    video_tokens_per_frame: int,
    action_tokens_per_frame: int,
    num_frames: int,
    chunk_size: int,
    window_size: int,
    device: torch.device,
    token_valid_ids: Optional[torch.Tensor] = None,
) -> MOTMaskMetadata:
    # X-only MOT layers use explicit-view tokenization without geometry tokens.
    # Joint MOT layers use single-copy G in the physical order:
    # noisy video, clean video, G, noisy action, clean action.
    # Explicit-view contract: callers must pack each stream as
    #   frame0: view0 tokens, view1 tokens, ...; frame1: view0 tokens, ...
    # and include all views in `*_tokens_per_frame`. View itself is intentionally
    # not represented in the causal metadata, so all views from the same frame
    # receive the same order id and are treated as simultaneous observations.

    # TODO： 本质上没用
    seq = torch.arange(batch_size, device=device)

    video_seq = seq[:, None, None].expand(-1, num_frames, video_tokens_per_frame).reshape(batch_size, -1)
    action_seq = seq[:, None, None].expand(-1, num_frames, action_tokens_per_frame).reshape(batch_size, -1)
    video_order = (torch.arange(num_frames, device=device) // chunk_size * 2)[:, None].expand(-1, video_tokens_per_frame).reshape(1, -1).expand(batch_size, -1)
    action_order = (torch.arange(num_frames, device=device) // chunk_size * 2 + 1)[:, None].expand(-1, action_tokens_per_frame).reshape(1, -1).expand(batch_size, -1)
    video_frame = torch.arange(num_frames, device=device)[:, None].expand(-1, video_tokens_per_frame).reshape(1, -1).expand(batch_size, -1)
    action_frame = torch.arange(num_frames, device=device)[:, None].expand(-1, action_tokens_per_frame).reshape(1, -1).expand(batch_size, -1)

    seq_ids = torch.cat([video_seq, video_seq, action_seq, action_seq], dim=1)
    order_ids = torch.cat([video_order, video_order, action_order, action_order], dim=1)
    frame_ids = torch.cat([video_frame, video_frame, action_frame, action_frame], dim=1)
    stream_ids = torch.cat([
        torch.full_like(video_order, STREAM_VIDEO),
        torch.full_like(video_order, STREAM_VIDEO),
        torch.full_like(action_order, STREAM_ACTION),
        torch.full_like(action_order, STREAM_ACTION),
    ], dim=1)
    noise_ids = torch.cat([
        torch.full_like(video_order, NOISE_NOISY),
        torch.full_like(video_order, NOISE_CLEAN),
        torch.full_like(action_order, NOISE_NOISY),
        torch.full_like(action_order, NOISE_CLEAN),
    ], dim=1)
    if token_valid_ids is not None:
        token_valid_ids = token_valid_ids.to(device=device, dtype=torch.bool)
        if tuple(token_valid_ids.shape) != tuple(seq_ids.shape):
            raise ValueError(f"token_valid_ids shape {tuple(token_valid_ids.shape)} does not match metadata {tuple(seq_ids.shape)}")
        token_valid_ids = _none_if_all_valid(token_valid_ids)
    structure_cache_key = (
        "x_only",
        batch_size,
        video_tokens_per_frame,
        action_tokens_per_frame,
        num_frames,
        chunk_size,
        window_size,
    )
    cache_key = None
    if token_valid_ids is None:
        cache_key = structure_cache_key
    return MOTMaskMetadata(
        seq_ids,
        order_ids,
        stream_ids,
        noise_ids,
        window_size,
        frame_ids=frame_ids,
        token_valid_ids=token_valid_ids,
        cache_key=cache_key,
        structure_cache_key=structure_cache_key,
    )


def build_mot_metadata(
    *,
    batch_size: int,
    video_tokens_per_frame: int,
    geometry_tokens_per_frame: int,
    action_tokens_per_frame: int,
    num_frames: int,
    chunk_size: int,
    window_size: int,
    device: torch.device,
    token_valid_ids: Optional[torch.Tensor] = None,
) -> MOTMaskMetadata:
    # TODO： 本质上没用
    seq = torch.arange(batch_size, device=device)

    video_seq = seq[:, None, None].expand(-1, num_frames, video_tokens_per_frame).reshape(batch_size, -1)
    geometry_seq = seq[:, None, None].expand(-1, num_frames, geometry_tokens_per_frame).reshape(batch_size, -1)
    action_seq = seq[:, None, None].expand(-1, num_frames, action_tokens_per_frame).reshape(batch_size, -1)

    video_order = (torch.arange(num_frames, device=device) // chunk_size * 2)[:, None].expand(-1, video_tokens_per_frame).reshape(1, -1).expand(batch_size, -1)
    geometry_order = (torch.arange(num_frames, device=device) // chunk_size * 2)[:, None].expand(-1, geometry_tokens_per_frame).reshape(1, -1).expand(batch_size, -1)
    action_order = (torch.arange(num_frames, device=device) // chunk_size * 2 + 1)[:, None].expand(-1, action_tokens_per_frame).reshape(1, -1).expand(batch_size, -1)
    video_frame = torch.arange(num_frames, device=device)[:, None].expand(-1, video_tokens_per_frame).reshape(1, -1).expand(batch_size, -1)
    geometry_frame = torch.arange(num_frames, device=device)[:, None].expand(-1, geometry_tokens_per_frame).reshape(1, -1).expand(batch_size, -1)
    action_frame = torch.arange(num_frames, device=device)[:, None].expand(-1, action_tokens_per_frame).reshape(1, -1).expand(batch_size, -1)

    seq_ids = torch.cat([video_seq, video_seq, geometry_seq, action_seq, action_seq], dim=1)
    order_ids = torch.cat([video_order, video_order, geometry_order, action_order, action_order], dim=1)
    frame_ids = torch.cat([video_frame, video_frame, geometry_frame, action_frame, action_frame], dim=1)
    stream_ids = torch.cat([
        torch.full_like(video_order, STREAM_VIDEO),
        torch.full_like(video_order, STREAM_VIDEO),
        torch.full_like(geometry_order, STREAM_GEOMETRY),
        torch.full_like(action_order, STREAM_ACTION),
        torch.full_like(action_order, STREAM_ACTION),
    ], dim=1)
    noise_ids = torch.cat([
        torch.full_like(video_order, NOISE_NOISY),
        torch.full_like(video_order, NOISE_CLEAN),
        torch.full_like(geometry_order, NOISE_GEOMETRY),
        torch.full_like(action_order, NOISE_NOISY),
        torch.full_like(action_order, NOISE_CLEAN),
    ], dim=1)
    if token_valid_ids is not None:
        token_valid_ids = token_valid_ids.to(device=device, dtype=torch.bool)
        if tuple(token_valid_ids.shape) != tuple(seq_ids.shape):
            raise ValueError(f"token_valid_ids shape {tuple(token_valid_ids.shape)} does not match metadata {tuple(seq_ids.shape)}")
        token_valid_ids = _none_if_all_valid(token_valid_ids)
    structure_cache_key = (
        "mot",
        batch_size,
        video_tokens_per_frame,
        geometry_tokens_per_frame,
        action_tokens_per_frame,
        num_frames,
        chunk_size,
        window_size,
    )
    cache_key = None
    if token_valid_ids is None:
        cache_key = structure_cache_key
    return MOTMaskMetadata(
        seq_ids,
        order_ids,
        stream_ids,
        noise_ids,
        window_size,
        frame_ids=frame_ids,
        token_valid_ids=token_valid_ids,
        cache_key=cache_key,
        structure_cache_key=structure_cache_key,
    )


def build_geometry_metadata(
    *,
    batch_size: int,
    geometry_tokens_per_frame: int,
    num_frames: int,
    chunk_size: int,
    window_size: int,
    device: torch.device,
    token_valid_ids: Optional[torch.Tensor] = None,
) -> MOTMaskMetadata:
    seq = torch.arange(batch_size, device=device)
    geometry_seq = seq[:, None, None].expand(-1, num_frames, geometry_tokens_per_frame).reshape(batch_size, -1)
    geometry_order = (torch.arange(num_frames, device=device) // chunk_size * 2)[:, None].expand(-1, geometry_tokens_per_frame).reshape(1, -1).expand(batch_size, -1)
    geometry_frame = torch.arange(num_frames, device=device)[:, None].expand(-1, geometry_tokens_per_frame).reshape(1, -1).expand(batch_size, -1)
    if token_valid_ids is not None:
        token_valid_ids = token_valid_ids.to(device=device, dtype=torch.bool)
        if tuple(token_valid_ids.shape) != tuple(geometry_seq.shape):
            raise ValueError(
                f"token_valid_ids shape {tuple(token_valid_ids.shape)} "
                f"does not match geometry metadata {tuple(geometry_seq.shape)}"
            )
        token_valid_ids = _none_if_all_valid(token_valid_ids)
    structure_cache_key = (
        "geometry",
        batch_size,
        geometry_tokens_per_frame,
        num_frames,
        chunk_size,
        window_size,
    )
    return MOTMaskMetadata(
        seq_ids=geometry_seq,
        order_ids=geometry_order,
        stream_ids=torch.full_like(geometry_order, STREAM_GEOMETRY),
        noise_ids=torch.full_like(geometry_order, NOISE_GEOMETRY),
        window_size=window_size,
        frame_ids=geometry_frame,
        token_valid_ids=token_valid_ids,
        cache_key=structure_cache_key if token_valid_ids is None else None,
        structure_cache_key=structure_cache_key,
    )


def dense_attention(
    q,
    k,
    v,
    allow_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Small explicit attention wrapper kept as the dense-mask reference.

    `allow_mask` is boolean allowed-form, matching PyTorch SDPA semantics. This
    path is intentionally retained so fallback attention can share the same
    inspectable dense mask.
    """

    attn_mask = None if allow_mask is None else allow_mask[:, None, :, :]
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
    ).transpose(1, 2)


class FlexMOTAttention(nn.Module):
    """High-performance MoT attention backend for the training path.

    This mirrors LingBot's original FlexAttention setup: compiled
    `create_block_mask` builds a sparse BlockMask, and compiled `flex_attention`
    consumes `[B, H, L, D]` tensors. The mask policy still lives in
    `MOTMaskMetadata`; dense fallback and training therefore use the same source
    of truth.
    """

    flex_attn: ClassVar[Callable] = torch.compile(flex_attention, dynamic=True)
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)


def apply_wan_rotary(x: torch.Tensor, freqs: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply LingBot/Wan complex RoPE to `[B, L, H, D]` tensors.

    Geometry patch tokens already have joint coordinates mapped into the LingBot token
    grid before this call. Register tokens already use special `(0, 0)` spatial
    coordinates, so this function remains a pure numerical transform.
    """

    if freqs is None:
        return x
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_complex * freqs).flatten(-2)
    return out.to(x.dtype)


def create_flex_mot_block_mask(meta: MOTMaskMetadata, compile: Optional[bool] = None) -> BlockMask:
    """Create a FlexAttention BlockMask from the same metadata rules.

    Dense `build_dense_mot_mask` is the readable reference. This helper is the
    training mask path and must stay semantically identical to the dense rules.
    By default we compile on CUDA and use the unfused FlexAttention path on
    CPU, because CPU Inductor compilation is not part of the training path and
    can fail independently of mask correctness.
    """

    if compile is None:
        compile = meta.device.type == "cuda"

    def mask_mod(b, h, q_idx, kv_idx):
        q_stream = meta.stream_ids[b, q_idx]
        k_stream = meta.stream_ids[b, kv_idx]
        valid = (meta.seq_ids[b, q_idx] == meta.seq_ids[b, kv_idx]) & (meta.seq_ids[b, q_idx] >= 0)
        valid = valid & ((meta.order_ids[b, q_idx] - meta.order_ids[b, kv_idx]).abs() <= meta.window_size)
        if meta.token_valid_ids is not None:
            valid = valid & meta.token_valid_ids[b, q_idx] & meta.token_valid_ids[b, kv_idx]

        x_query = (q_stream == STREAM_VIDEO) | (q_stream == STREAM_ACTION)
        x_key = (k_stream == STREAM_VIDEO) | (k_stream == STREAM_ACTION)
        g_query = q_stream == STREAM_GEOMETRY
        g_key = k_stream == STREAM_GEOMETRY

        q_noise = meta.noise_ids[b, q_idx]
        k_noise = meta.noise_ids[b, kv_idx]
        q_order = meta.order_ids[b, q_idx]
        k_order = meta.order_ids[b, kv_idx]

        # Same LingBot order/noise rules as _x_to_x, written inline because FlexAttention
        # mask_mod must be scalar-friendly. The action inverse-dynamics offset is encoded
        # by odd action order ids plus the dataset's padded A0 slot, not by a special branch here.
        clean_to_clean = (q_noise == NOISE_CLEAN) & (k_noise == NOISE_CLEAN) & (k_order <= q_order)
        noisy_to_clean = (q_noise == NOISE_NOISY) & (k_noise == NOISE_CLEAN) & (k_order < q_order)
        noisy_to_noisy = (q_noise == NOISE_NOISY) & (k_noise == NOISE_NOISY) & (k_order == q_order)
        x_to_x = x_query & x_key & (clean_to_clean | noisy_to_clean | noisy_to_noisy)

        g_to_g = g_query & g_key & (k_order <= q_order)
        clean_to_g = x_query & g_key & (q_noise == NOISE_CLEAN) & (k_order <= q_order)
        noisy_to_g = x_query & g_key & (q_noise == NOISE_NOISY) & (k_order < q_order)
        return valid & (x_to_x | g_to_g | clean_to_g | noisy_to_g)

    create = FlexMOTAttention.compiled_create_block_mask if compile else create_block_mask
    return create(partial(mask_mod), meta.batch_size, 1, meta.seq_len, meta.seq_len, device=meta.device, _compile=compile)


def flex_attention_from_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    meta: MOTMaskMetadata,
    *,
    block_mask: Optional[BlockMask] = None,
    compile: Optional[bool] = None,
) -> torch.Tensor:
    """Run MoT attention with FlexAttention using canonical metadata.

    Inputs and output use the local model convention `[B, L, H, D]`. Internally
    FlexAttention expects `[B, H, L, D]`, so this wrapper is the only place where
    that layout conversion happens. Actual CUDA training uses compiled
    FlexAttention, while dense attention remains available as a dense fallback.
    CPU defaults to the unfused FlexAttention path.
    """

    if compile is None:
        compile = q.device.type == "cuda"

    if block_mask is None:
        block_mask = create_flex_mot_block_mask(meta, compile=compile)

    fn = FlexMOTAttention.flex_attn if compile else flex_attention
    q_bhld = q.transpose(1, 2).contiguous()
    k_bhld = k.transpose(1, 2).contiguous()
    v_bhld = v.transpose(1, 2).contiguous()
    out = fn(
        q_bhld,
        k_bhld,
        v_bhld,
        score_mod=None,
        block_mask=block_mask,
        kernel_options={
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        },
    )
    return out.transpose(1, 2)


def attention_from_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    meta: MOTMaskMetadata,
    *,
    backend: str = "flex",
    block_mask: Optional[BlockMask] = None,
    compile: Optional[bool] = None,
) -> torch.Tensor:
    """Dispatch masked MoT attention to the requested backend.

    Review decision for FA4 migration: dense masks remain the readable reference,
    Flex remains the non-FA4 training fallback, and `backend="fa4"` must really
    enter FlashAttention-4 custom-mask kernels or fail fast with a clear error.
    """

    if backend == "flex":
        out = flex_attention_from_meta(
            q,
            k,
            v,
            meta,
            block_mask=block_mask,
            compile=compile,
        )
    elif backend == "fa4":
        out = fa4_attention_from_meta(q, k, v, meta)
    elif backend == "dense":
        out = dense_attention(q, k, v, build_dense_mot_mask(meta))
    else:
        raise ValueError(f"Unsupported masked attention backend: {backend}")
    if meta.token_valid_ids is not None:
        out = out * meta.token_valid_ids[:, :, None, None].to(device=out.device, dtype=out.dtype)
    return out



class DenseMaskedAttention(nn.Module):
    def forward(self, q, k, v, meta: Optional[MOTMaskMetadata]):
        mask = build_dense_mot_mask(meta) if meta is not None else None
        return dense_attention(q, k, v, mask)
