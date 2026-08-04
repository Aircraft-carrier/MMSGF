"""Layered committed/transaction K/V caches for self rollout."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention import TokenMetadataBatch


@dataclass(frozen=True, slots=True)
class KVSegment:
    key: torch.Tensor
    value: torch.Tensor
    metadata: TokenMetadataBatch

    def __post_init__(self) -> None:
        if self.key.shape != self.value.shape:
            raise ValueError("K/V shapes must match")
        if self.key.ndim != 4:
            raise ValueError(f"K/V must be [B,L,H,D], got {tuple(self.key.shape)}")
        if self.key.shape[:2] != (
            self.metadata.batch_size,
            self.metadata.seq_len,
        ):
            raise ValueError("K/V leading dimensions must match metadata")
        if self.key.device != self.metadata.device:
            raise ValueError("K/V and metadata must share a device")


@dataclass(frozen=True, slots=True)
class KVCacheSnapshot:
    committed: dict[int, tuple[KVSegment, ...]]
    transactions: dict[int, dict[int, tuple[KVSegment, ...]]]


class SelfRolloutKVCache:
    """Append-only per-layer cache with explicit transaction lifetimes."""

    def __init__(self) -> None:
        self._committed: dict[int, list[KVSegment]] = {}
        self._transactions: dict[int, dict[int, list[KVSegment]]] = {}

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(set(self._committed) | set(self._transactions)))

    def committed_segments(self) -> tuple[tuple[int, KVSegment], ...]:
        """Return an immutable diagnostic view of all committed segments."""

        return tuple(
            (layer_id, segment)
            for layer_id in sorted(self._committed)
            for segment in self._committed[layer_id]
        )

    def append_committed(self, layer_id: int, segment: KVSegment) -> None:
        if not bool(segment.metadata.committed_ids.all().item()):
            raise ValueError("committed segment metadata must be fully committed")
        if bool((segment.metadata.transaction_ids >= 0).any().item()):
            raise ValueError("committed segment cannot carry a transaction id")
        self._committed.setdefault(int(layer_id), []).append(segment)

    def append_transaction(
        self,
        layer_id: int,
        transaction_id: int,
        segment: KVSegment,
    ) -> None:
        transaction_id = int(transaction_id)
        if transaction_id < 0:
            raise ValueError("transaction_id must be non-negative")
        if bool(segment.metadata.committed_ids.any().item()):
            raise ValueError("transaction segment cannot contain committed tokens")
        if not bool((segment.metadata.transaction_ids == transaction_id).all().item()):
            raise ValueError("transaction metadata id mismatch")
        by_layer = self._transactions.setdefault(transaction_id, {})
        by_layer.setdefault(int(layer_id), []).append(segment)

    def materialize(
        self,
        layer_id: int,
        *,
        transaction_id: int | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        heads: int | None = None,
        head_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, TokenMetadataBatch]:
        parts = list(self._committed.get(int(layer_id), ()))
        if transaction_id is not None:
            parts.extend(
                self._transactions.get(int(transaction_id), {}).get(int(layer_id), ())
            )
        if parts:
            return (
                torch.cat([part.key for part in parts], dim=1),
                torch.cat([part.value for part in parts], dim=1),
                TokenMetadataBatch.cat([part.metadata for part in parts]),
            )
        if None in (batch_size, device, dtype, heads, head_dim):
            raise ValueError("empty cache materialization requires tensor shape arguments")
        empty = torch.empty(
            (int(batch_size), 0, int(heads), int(head_dim)),
            device=device,
            dtype=dtype,
        )
        return empty, empty.clone(), TokenMetadataBatch.empty(int(batch_size), device)

    def discard_transaction(self, transaction_id: int) -> None:
        self._transactions.pop(int(transaction_id), None)

    def commit_transaction(self, transaction_id: int, *, source_id: int) -> None:
        by_layer = self._transactions.pop(int(transaction_id), None)
        if not by_layer:
            raise ValueError(f"transaction {transaction_id} is empty")
        for layer_id, segments in by_layer.items():
            for segment in segments:
                self.append_committed(
                    layer_id,
                    KVSegment(
                        segment.key,
                        segment.value,
                        segment.metadata.as_committed(source_id=source_id),
                    ),
                )

    def truncate_from(self, frame_id: int) -> None:
        frame_id = int(frame_id)

        def truncate(segment: KVSegment) -> KVSegment | None:
            keep = (segment.metadata.frame_ids < frame_id).all(dim=0)
            if not bool(keep.any().item()):
                return None
            if bool(keep.all().item()):
                return segment
            return KVSegment(
                segment.key[:, keep],
                segment.value[:, keep],
                segment.metadata.select(keep),
            )

        def truncate_segments(segments: list[KVSegment]) -> list[KVSegment]:
            out = []
            for segment in segments:
                kept = truncate(segment)
                if kept is not None:
                    out.append(kept)
            return out

        self._committed = {
            layer: truncate_segments(segments)
            for layer, segments in self._committed.items()
        }
        self._committed = {
            layer: segments for layer, segments in self._committed.items() if segments
        }
        for transaction_id in list(self._transactions):
            by_layer = self._transactions[transaction_id]
            kept = {
                layer: truncate_segments(segments)
                for layer, segments in by_layer.items()
            }
            kept = {layer: segments for layer, segments in kept.items() if segments}
            if kept:
                self._transactions[transaction_id] = kept
            else:
                del self._transactions[transaction_id]

    def snapshot(self) -> KVCacheSnapshot:
        return KVCacheSnapshot(
            committed={
                layer: tuple(segments) for layer, segments in self._committed.items()
            },
            transactions={
                tx: {layer: tuple(segments) for layer, segments in by_layer.items()}
                for tx, by_layer in self._transactions.items()
            },
        )

    def restore(self, snapshot: KVCacheSnapshot) -> None:
        self._committed = {
            layer: list(segments) for layer, segments in snapshot.committed.items()
        }
        self._transactions = {
            tx: {layer: list(segments) for layer, segments in by_layer.items()}
            for tx, by_layer in snapshot.transactions.items()
        }

    def assert_no_transactions(self) -> None:
        if self._transactions:
            raise RuntimeError(
                f"active cache transactions remain: {sorted(self._transactions)}"
            )

    def has_frame(self, frame_id: int, *, committed_only: bool = True) -> bool:
        containers = [self._committed]
        if not committed_only:
            containers.extend(self._transactions.values())
        for container in containers:
            for segments in container.values():
                if any(
                    bool((segment.metadata.frame_ids == int(frame_id)).any().item())
                    for segment in segments
                ):
                    return True
        return False

    def committed_token_count(self, layer_id: int) -> int:
        return sum(
            segment.metadata.seq_len
            for segment in self._committed.get(int(layer_id), ())
        )

    def assert_committed_frame(
        self,
        layer_ids: tuple[int, ...] | range,
        *,
        frame_id: int,
        stream_id: int,
        noise_ids: tuple[int, ...],
        source_id: int,
        version_id: int,
    ) -> None:
        """Assert that every expected layer has the canonical frame streams."""

        self.assert_no_transactions()
        expected_noise = set(int(value) for value in noise_ids)
        missing: list[str] = []
        for layer_id in layer_ids:
            seen_noise: set[int] = set()
            for segment in self._committed.get(int(layer_id), ()):
                metadata = segment.metadata
                selected = (
                    (metadata.frame_ids == int(frame_id))
                    & (metadata.stream_ids == int(stream_id))
                    & (metadata.source_ids == int(source_id))
                    & (metadata.version_ids == int(version_id))
                    & metadata.committed_ids
                )
                if bool(selected.any().item()):
                    seen_noise.update(
                        int(value)
                        for value in metadata.noise_ids[selected].unique().tolist()
                    )
            absent = sorted(expected_noise - seen_noise)
            if absent:
                missing.append(f"layer {int(layer_id)} missing noise_ids={absent}")
        if missing:
            raise RuntimeError(
                f"incomplete canonical cache for frame={int(frame_id)}, "
                f"stream={int(stream_id)}, source={int(source_id)}, "
                f"version={int(version_id)}: " + "; ".join(missing)
            )
