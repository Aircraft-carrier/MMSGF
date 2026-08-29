"""Canonical attention metadata and backends for Video+Action MoT."""

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
STREAM_PAD = -1

NOISE_NOISY = 0
NOISE_CLEAN = 1
NOISE_PAD = -1


@dataclass
class MOTMaskMetadata:
    """Flat per-token metadata used by every masked-attention backend."""

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
    if token_valid_ids is None or bool(token_valid_ids.all().item()):
        return None
    return token_valid_ids


def _same_batch(meta: MOTMaskMetadata, q_idx, kv_idx):
    return (meta.seq_ids[:, q_idx] == meta.seq_ids[:, kv_idx]) & (
        meta.seq_ids[:, q_idx] >= 0
    )


def _window(meta: MOTMaskMetadata, q_idx, kv_idx):
    return (
        meta.order_ids[:, q_idx] - meta.order_ids[:, kv_idx]
    ).abs() <= meta.window_size


def _token_valid(meta: MOTMaskMetadata, q_idx, kv_idx):
    if meta.token_valid_ids is None:
        return True
    return meta.token_valid_ids[:, q_idx] & meta.token_valid_ids[:, kv_idx]


def _va_visibility(meta: MOTMaskMetadata, q_idx, kv_idx):
    q_noise = meta.noise_ids[:, q_idx]
    k_noise = meta.noise_ids[:, kv_idx]
    q_order = meta.order_ids[:, q_idx]
    k_order = meta.order_ids[:, kv_idx]
    clean_to_clean = (
        (q_noise == NOISE_CLEAN)
        & (k_noise == NOISE_CLEAN)
        & (k_order <= q_order)
    )
    noisy_to_clean = (
        (q_noise == NOISE_NOISY)
        & (k_noise == NOISE_CLEAN)
        & (k_order < q_order)
    )
    noisy_to_noisy = (
        (q_noise == NOISE_NOISY)
        & (k_noise == NOISE_NOISY)
        & (k_order == q_order)
    )
    return clean_to_clean | noisy_to_clean | noisy_to_noisy


def build_dense_mot_mask(meta: MOTMaskMetadata) -> torch.Tensor:
    q_idx = torch.arange(meta.seq_len, device=meta.device)[:, None]
    kv_idx = torch.arange(meta.seq_len, device=meta.device)[None, :]
    valid = (
        _same_batch(meta, q_idx, kv_idx)
        & _window(meta, q_idx, kv_idx)
        & _token_valid(meta, q_idx, kv_idx)
    )
    return valid & _va_visibility(meta, q_idx, kv_idx)


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
    """Pack noisy/clean video and noisy/clean action metadata."""

    seq = torch.arange(batch_size, device=device)
    video_seq = seq[:, None, None].expand(
        -1, num_frames, video_tokens_per_frame
    ).reshape(batch_size, -1)
    action_seq = seq[:, None, None].expand(
        -1, num_frames, action_tokens_per_frame
    ).reshape(batch_size, -1)
    frame = torch.arange(num_frames, device=device)
    video_order = (frame // chunk_size * 2)[:, None].expand(
        -1, video_tokens_per_frame
    ).reshape(1, -1).expand(batch_size, -1)
    action_order = (frame // chunk_size * 2 + 1)[:, None].expand(
        -1, action_tokens_per_frame
    ).reshape(1, -1).expand(batch_size, -1)
    video_frame = frame[:, None].expand(-1, video_tokens_per_frame).reshape(
        1, -1
    ).expand(batch_size, -1)
    action_frame = frame[:, None].expand(-1, action_tokens_per_frame).reshape(
        1, -1
    ).expand(batch_size, -1)
    seq_ids = torch.cat([video_seq, video_seq, action_seq, action_seq], dim=1)
    order_ids = torch.cat(
        [video_order, video_order, action_order, action_order], dim=1
    )
    frame_ids = torch.cat(
        [video_frame, video_frame, action_frame, action_frame], dim=1
    )
    stream_ids = torch.cat(
        [
            torch.full_like(video_order, STREAM_VIDEO),
            torch.full_like(video_order, STREAM_VIDEO),
            torch.full_like(action_order, STREAM_ACTION),
            torch.full_like(action_order, STREAM_ACTION),
        ],
        dim=1,
    )
    noise_ids = torch.cat(
        [
            torch.full_like(video_order, NOISE_NOISY),
            torch.full_like(video_order, NOISE_CLEAN),
            torch.full_like(action_order, NOISE_NOISY),
            torch.full_like(action_order, NOISE_CLEAN),
        ],
        dim=1,
    )
    if token_valid_ids is not None:
        token_valid_ids = token_valid_ids.to(device=device, dtype=torch.bool)
        if tuple(token_valid_ids.shape) != tuple(seq_ids.shape):
            raise ValueError(
                f"token_valid_ids shape {tuple(token_valid_ids.shape)} "
                f"does not match metadata {tuple(seq_ids.shape)}"
            )
        token_valid_ids = _none_if_all_valid(token_valid_ids)
    structure_cache_key = (
        "va",
        batch_size,
        video_tokens_per_frame,
        action_tokens_per_frame,
        num_frames,
        chunk_size,
        window_size,
    )
    return MOTMaskMetadata(
        seq_ids=seq_ids,
        order_ids=order_ids,
        stream_ids=stream_ids,
        noise_ids=noise_ids,
        window_size=window_size,
        frame_ids=frame_ids,
        token_valid_ids=token_valid_ids,
        cache_key=structure_cache_key if token_valid_ids is None else None,
        structure_cache_key=structure_cache_key,
    )


def dense_attention(q, k, v, allow_mask: Optional[torch.Tensor]) -> torch.Tensor:
    attn_mask = None if allow_mask is None else allow_mask[:, None, :, :]
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
    ).transpose(1, 2)


class FlexMOTAttention(nn.Module):
    flex_attn: ClassVar[Callable] = torch.compile(flex_attention, dynamic=True)
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)


def apply_wan_rotary(x: torch.Tensor, freqs: Optional[torch.Tensor]) -> torch.Tensor:
    if freqs is None:
        return x
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(*x.shape[:-1], -1, 2)
    )
    return torch.view_as_real(x_complex * freqs).flatten(-2).to(x.dtype)


def create_flex_mot_block_mask(
    meta: MOTMaskMetadata,
    compile: Optional[bool] = None,
) -> BlockMask:
    if compile is None:
        compile = meta.device.type == "cuda"

    def mask_mod(b, h, q_idx, kv_idx):
        del h
        valid = (meta.seq_ids[b, q_idx] == meta.seq_ids[b, kv_idx]) & (
            meta.seq_ids[b, q_idx] >= 0
        )
        valid = valid & (
            (meta.order_ids[b, q_idx] - meta.order_ids[b, kv_idx]).abs()
            <= meta.window_size
        )
        if meta.token_valid_ids is not None:
            valid = (
                valid
                & meta.token_valid_ids[b, q_idx]
                & meta.token_valid_ids[b, kv_idx]
            )
        q_noise = meta.noise_ids[b, q_idx]
        k_noise = meta.noise_ids[b, kv_idx]
        q_order = meta.order_ids[b, q_idx]
        k_order = meta.order_ids[b, kv_idx]
        clean_to_clean = (
            (q_noise == NOISE_CLEAN)
            & (k_noise == NOISE_CLEAN)
            & (k_order <= q_order)
        )
        noisy_to_clean = (
            (q_noise == NOISE_NOISY)
            & (k_noise == NOISE_CLEAN)
            & (k_order < q_order)
        )
        noisy_to_noisy = (
            (q_noise == NOISE_NOISY)
            & (k_noise == NOISE_NOISY)
            & (k_order == q_order)
        )
        return valid & (clean_to_clean | noisy_to_clean | noisy_to_noisy)

    create = FlexMOTAttention.compiled_create_block_mask if compile else create_block_mask
    return create(
        partial(mask_mod),
        meta.batch_size,
        1,
        meta.seq_len,
        meta.seq_len,
        device=meta.device,
        _compile=compile,
    )


def flex_attention_from_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    meta: MOTMaskMetadata,
    *,
    block_mask: Optional[BlockMask] = None,
    compile: Optional[bool] = None,
) -> torch.Tensor:
    if compile is None:
        compile = q.device.type == "cuda"
    if block_mask is None:
        block_mask = create_flex_mot_block_mask(meta, compile=compile)
    fn = FlexMOTAttention.flex_attn if compile else flex_attention
    out = fn(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
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
    if backend == "flex":
        out = flex_attention_from_meta(
            q, k, v, meta, block_mask=block_mask, compile=compile
        )
    elif backend == "fa4":
        out = fa4_attention_from_meta(q, k, v, meta)
    elif backend == "dense":
        out = dense_attention(q, k, v, build_dense_mot_mask(meta))
    else:
        raise ValueError(f"Unsupported masked attention backend: {backend}")
    if meta.token_valid_ids is not None:
        out = out * meta.token_valid_ids[:, :, None, None].to(
            device=out.device, dtype=out.dtype
        )
    return out


class DenseMaskedAttention(nn.Module):
    def forward(self, q, k, v, meta: Optional[MOTMaskMetadata]):
        mask = build_dense_mot_mask(meta) if meta is not None else None
        return dense_attention(q, k, v, mask)
