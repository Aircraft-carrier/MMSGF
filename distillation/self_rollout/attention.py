"""Distillation-owned attention metadata and visibility for ``self_rollout``.

This module intentionally has no dependency on ``wan_va.modules`` so cache and
mask semantics can be tested on CPU without importing optional FlashAttention
extensions.  The integer constants mirror ``wan_va.modules.mot_attention``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch


STREAM_VIDEO = 0
STREAM_ACTION = 1
STREAM_GEOMETRY = 2

NOISE_NOISY = 0
NOISE_CLEAN = 1
NOISE_GEOMETRY = 2

SOURCE_HISTORY = 0
SOURCE_ANCHOR = 1
SOURCE_PREDICTED = 2
SOURCE_GROUND_TRUTH = 3

_METADATA_TENSOR_FIELDS = (
    "seq_ids",
    "frame_ids",
    "order_ids",
    "stream_ids",
    "noise_ids",
    "valid_ids",
    "committed_ids",
    "transaction_ids",
    "source_ids",
    "version_ids",
)


@dataclass(frozen=True, slots=True)
class TokenMetadataBatch:
    """Per-token metadata for rectangular incremental attention."""

    seq_ids: torch.Tensor
    frame_ids: torch.Tensor
    order_ids: torch.Tensor
    stream_ids: torch.Tensor
    noise_ids: torch.Tensor
    valid_ids: torch.Tensor
    committed_ids: torch.Tensor
    transaction_ids: torch.Tensor
    source_ids: torch.Tensor
    version_ids: torch.Tensor
    runtime_cache: dict[str, object] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        shape = tuple(self.seq_ids.shape)
        if len(shape) != 2:
            raise ValueError(f"token metadata must be [B,L], got {shape}")
        for name in (
            "frame_ids",
            "order_ids",
            "stream_ids",
            "noise_ids",
            "valid_ids",
            "committed_ids",
            "transaction_ids",
            "source_ids",
            "version_ids",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} does not match {shape}"
                )
            if value.device != self.seq_ids.device:
                raise ValueError(f"{name} must be on {self.seq_ids.device}")
        if self.valid_ids.dtype != torch.bool:
            raise TypeError("valid_ids must be bool")
        if self.committed_ids.dtype != torch.bool:
            raise TypeError("committed_ids must be bool")

    @property
    def batch_size(self) -> int:
        return int(self.seq_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.seq_ids.shape[1])

    @property
    def device(self) -> torch.device:
        return self.seq_ids.device

    def as_committed(self, *, source_id: int | None = None) -> "TokenMetadataBatch":
        source_ids = self.source_ids
        if source_id is not None:
            source_ids = torch.full_like(source_ids, int(source_id))
        return replace(
            self,
            committed_ids=torch.ones_like(self.committed_ids),
            transaction_ids=torch.full_like(self.transaction_ids, -1),
            source_ids=source_ids,
        )

    def slice(self, start: int, end: int) -> "TokenMetadataBatch":
        return TokenMetadataBatch(
            **{
                name: getattr(self, name)[:, start:end]
                for name in _METADATA_TENSOR_FIELDS
            }
        )

    def select(self, columns: torch.Tensor) -> "TokenMetadataBatch":
        columns = torch.as_tensor(columns, device=self.device)
        if columns.dtype == torch.bool:
            if columns.ndim != 1 or columns.numel() != self.seq_len:
                raise ValueError("boolean metadata selector must be [L]")
        elif columns.dtype != torch.long:
            columns = columns.to(dtype=torch.long)
        return TokenMetadataBatch(
            **{
                name: getattr(self, name)[:, columns]
                for name in _METADATA_TENSOR_FIELDS
            }
        )

    @classmethod
    def empty(cls, batch_size: int, device: torch.device) -> "TokenMetadataBatch":
        long = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        boolean = torch.empty((batch_size, 0), dtype=torch.bool, device=device)
        return cls(
            seq_ids=long,
            frame_ids=long.clone(),
            order_ids=long.clone(),
            stream_ids=long.clone(),
            noise_ids=long.clone(),
            valid_ids=boolean,
            committed_ids=boolean.clone(),
            transaction_ids=long.clone(),
            source_ids=long.clone(),
            version_ids=long.clone(),
        )

    @classmethod
    def cat(cls, parts: list["TokenMetadataBatch"]) -> "TokenMetadataBatch":
        if not parts:
            raise ValueError("TokenMetadataBatch.cat requires at least one part")
        batch_size = parts[0].batch_size
        device = parts[0].device
        if any(part.batch_size != batch_size or part.device != device for part in parts):
            raise ValueError("all metadata parts must share batch size and device")
        return cls(
            **{
                name: torch.cat([getattr(part, name) for part in parts], dim=1)
                for name in _METADATA_TENSOR_FIELDS
            }
        )


def segmented_orders(
    frame_ids: torch.Tensor,
    *,
    history_frames: int,
    chunk_size: int,
) -> torch.Tensor:
    """Return stable video/geometry orders for arbitrary rollout horizons."""

    if history_frames <= 0:
        raise ValueError(f"history_frames must be positive, got {history_frames}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if bool((frame_ids < 0).any().item()):
        raise ValueError("frame_ids must be non-negative")
    history_segments = (int(history_frames) + int(chunk_size) - 1) // int(chunk_size)
    history_order = torch.div(frame_ids, int(chunk_size), rounding_mode="floor") * 2
    target_order = (
        2 * history_segments
        + 2 * (frame_ids - int(history_frames))
    )
    return torch.where(frame_ids < int(history_frames), history_order, target_order)


def build_token_metadata(
    *,
    batch_size: int,
    frame_ids: torch.Tensor,
    tokens_per_frame: int,
    stream_id: int,
    noise_id: int,
    history_frames: int,
    chunk_size: int,
    device: torch.device,
    valid_ids: torch.Tensor | None = None,
    committed: bool,
    transaction_id: int = -1,
    source_id: int = SOURCE_PREDICTED,
    version_id: int = 0,
) -> TokenMetadataBatch:
    """Build flat metadata for one homogeneous stream segment."""

    if batch_size <= 0 or tokens_per_frame <= 0:
        raise ValueError("batch_size and tokens_per_frame must be positive")
    frame_ids = torch.as_tensor(frame_ids, dtype=torch.long, device=device)
    if frame_ids.ndim == 1:
        frame_ids = frame_ids[None].expand(batch_size, -1)
    if frame_ids.ndim != 2 or frame_ids.shape[0] != batch_size:
        raise ValueError(
            f"frame_ids must be [F] or [B,F] with B={batch_size}, got {tuple(frame_ids.shape)}"
        )
    expanded_frames = frame_ids.repeat_interleave(tokens_per_frame, dim=1)
    base_order = segmented_orders(
        expanded_frames,
        history_frames=history_frames,
        chunk_size=chunk_size,
    )
    order_ids = base_order + (1 if int(stream_id) == STREAM_ACTION else 0)
    shape = expanded_frames.shape
    seq_ids = torch.arange(batch_size, device=device)[:, None].expand(shape)
    if valid_ids is None:
        valid_ids = torch.ones(shape, dtype=torch.bool, device=device)
    else:
        valid_ids = torch.as_tensor(valid_ids, dtype=torch.bool, device=device)
        if valid_ids.ndim == 2 and valid_ids.shape[1] == frame_ids.shape[1]:
            valid_ids = valid_ids.repeat_interleave(tokens_per_frame, dim=1)
        if tuple(valid_ids.shape) != tuple(shape):
            raise ValueError(
                f"valid_ids must match frames or tokens {shape}, got {tuple(valid_ids.shape)}"
            )
    return TokenMetadataBatch(
        seq_ids=seq_ids,
        frame_ids=expanded_frames,
        order_ids=order_ids,
        stream_ids=torch.full(shape, int(stream_id), dtype=torch.long, device=device),
        noise_ids=torch.full(shape, int(noise_id), dtype=torch.long, device=device),
        valid_ids=valid_ids,
        committed_ids=torch.full(shape, bool(committed), dtype=torch.bool, device=device),
        transaction_ids=torch.full(
            shape,
            -1 if committed else int(transaction_id),
            dtype=torch.long,
            device=device,
        ),
        source_ids=torch.full(shape, int(source_id), dtype=torch.long, device=device),
        version_ids=torch.full(shape, int(version_id), dtype=torch.long, device=device),
    )


def from_mot_metadata(meta, *, committed: bool = True) -> TokenMetadataBatch:
    """Convert an existing square MOT metadata object into this policy format."""

    cached = getattr(meta, "_distillation_policy_metadata", None)
    if cached is not None and bool(committed):
        return cached
    if meta.frame_ids is None:
        raise ValueError("strict geometry policy requires MOT frame_ids")
    valid = meta.seq_ids >= 0
    if meta.token_valid_ids is not None:
        valid = valid & meta.token_valid_ids.to(device=meta.device, dtype=torch.bool)
    shape = meta.seq_ids.shape
    converted = TokenMetadataBatch(
        seq_ids=meta.seq_ids,
        frame_ids=meta.frame_ids,
        order_ids=meta.order_ids,
        stream_ids=meta.stream_ids,
        noise_ids=meta.noise_ids,
        valid_ids=valid,
        committed_ids=torch.full(shape, bool(committed), dtype=torch.bool, device=meta.device),
        transaction_ids=torch.full(shape, -1, dtype=torch.long, device=meta.device),
        source_ids=torch.full(shape, SOURCE_HISTORY, dtype=torch.long, device=meta.device),
        version_ids=torch.zeros(shape, dtype=torch.long, device=meta.device),
    )
    if committed:
        meta._distillation_policy_metadata = converted
    return converted


def build_cache_visibility(
    query: TokenMetadataBatch,
    key: TokenMetadataBatch,
    *,
    window_size: int,
) -> torch.Tensor:
    """Build the canonical rectangular ``[B,Q,K]`` allowed-form mask."""

    if query.batch_size != key.batch_size:
        raise ValueError("query/key batch sizes must match")
    if query.device != key.device:
        raise ValueError("query/key metadata must be on the same device")
    if window_size < 0:
        raise ValueError(f"window_size must be non-negative, got {window_size}")
    if key.seq_len == 0:
        return torch.zeros(
            (query.batch_size, query.seq_len, 0),
            dtype=torch.bool,
            device=query.device,
        )

    q_seq = query.seq_ids[:, :, None]
    k_seq = key.seq_ids[:, None, :]
    q_frame = query.frame_ids[:, :, None]
    k_frame = key.frame_ids[:, None, :]
    q_order = query.order_ids[:, :, None]
    k_order = key.order_ids[:, None, :]
    q_stream = query.stream_ids[:, :, None]
    k_stream = key.stream_ids[:, None, :]
    q_noise = query.noise_ids[:, :, None]
    k_noise = key.noise_ids[:, None, :]

    base = (
        (q_seq == k_seq)
        & (q_seq >= 0)
        & query.valid_ids[:, :, None]
        & key.valid_ids[:, None, :]
        & ((q_order - k_order).abs() <= int(window_size))
    )

    g_query = q_stream == STREAM_GEOMETRY
    g_key = k_stream == STREAM_GEOMETRY
    geometry_relation = (
        g_query
        & g_key
        & key.committed_ids[:, None, :]
        & (k_frame < q_frame)
    )

    x_query = (q_stream == STREAM_VIDEO) | (q_stream == STREAM_ACTION)
    x_key = (k_stream == STREAM_VIDEO) | (k_stream == STREAM_ACTION)
    same_transaction = (
        (query.transaction_ids[:, :, None] >= 0)
        & (
            query.transaction_ids[:, :, None]
            == key.transaction_ids[:, None, :]
        )
    )
    readable = key.committed_ids[:, None, :] | same_transaction

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
    x_relation = x_query & x_key & readable & (
        clean_to_clean | noisy_to_clean | noisy_to_noisy
    )
    x_to_g = x_query & g_key & readable & (k_order < q_order)
    return base & (geometry_relation | x_relation | x_to_g)


def incremental_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allow_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply rectangular SDPA using local ``[B,L,H,D]`` tensor layout."""

    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    expected = (query.shape[0], query.shape[1], key.shape[1])
    if tuple(allow_mask.shape) != expected:
        raise ValueError(
            f"allow_mask must be {expected}, got {tuple(allow_mask.shape)}"
        )
    if key.shape[1] == 0:
        return torch.zeros_like(query)

    has_visible_key = allow_mask.any(dim=-1)
    safe_mask = allow_mask
    if not bool(has_visible_key.all().item()):
        safe_mask = allow_mask.clone()
        missing = (~has_visible_key).nonzero(as_tuple=False)
        safe_mask[missing[:, 0], missing[:, 1], 0] = True
    output = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=safe_mask[:, None],
    ).transpose(1, 2)
    return torch.where(has_visible_key[:, :, None, None], output, torch.zeros_like(output))
