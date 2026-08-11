from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class KVSegment:
    key: torch.Tensor    # K [B,L,H,D]
    value: torch.Tensor  # V [B,L,H,D]

    def __post_init__(self) -> None:
        # 契约：同一段 K/V 必须同形且统一为 4 维 [B,L,H,D]；
        # 违反说明调用方传入了错误的注意力张量，尽早失败。
        if self.key.shape != self.value.shape:
            raise ValueError("K/V shapes must match")
        if self.key.ndim != 4:
            raise ValueError(f"K/V must be [B,L,H,D], got {tuple(self.key.shape)}")


class KVCache:
    """Append-only per-layer K/V cache with predict/commit transaction separation.

    Committed segments are permanent history. Prediction transactions hold the
    current denoising step's K/V only for the duration of one model call and are
    discarded afterwards, so noisy intermediate states never enter the cache.
    """

    def __init__(self) -> None:
        self._committed: dict[int, list[KVSegment]] = {}
        self._transactions: dict[int, dict[int, list[KVSegment]]] = {}
        self._next_transaction_id = 0

    def new_transaction_id(self) -> int:
        transaction_id = self._next_transaction_id
        self._next_transaction_id += 1
        return transaction_id

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(set(self._committed) | set(self._transactions)))

    def append(
        self,
        layer_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        transaction_id: int | None = None,
    ) -> None:
        layer_id = int(layer_id)
        segment = KVSegment(key, value)
        if transaction_id is None:
            self._committed.setdefault(layer_id, []).append(segment)
        else:
            transaction_id = int(transaction_id)
            by_layer = self._transactions.setdefault(transaction_id, {})
            by_layer.setdefault(layer_id, []).append(segment)

    def materialize(
        self,
        layer_id: int,
        *,
        transaction_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return committed K/V plus the optional transaction, in append order."""
        layer_id = int(layer_id)
        parts = list(self._committed.get(layer_id, ()))
        if transaction_id is not None:
            parts.extend(
                self._transactions.get(int(transaction_id), {}).get(layer_id, ())
            )
        if not parts:
            raise ValueError(f"no cached K/V for layer {layer_id}")
        return (
            torch.cat([part.key for part in parts], dim=1),
            torch.cat([part.value for part in parts], dim=1),
        )

    def commit(self, transaction_id: int) -> None:
        """Promote a prediction transaction's clean K/V to committed history."""
        transaction_id = int(transaction_id)
        by_layer = self._transactions.pop(transaction_id, None)
        if by_layer is None:
            raise ValueError(f"transaction {transaction_id} is empty or unknown")
        for layer_id, segments in by_layer.items():
            self._committed.setdefault(layer_id, []).extend(segments)

    def discard(self, transaction_id: int) -> None:
        self._transactions.pop(int(transaction_id), None)

    def committed_token_count(self, layer_id: int) -> int:
        return sum(
            segment.key.shape[1]
            for segment in self._committed.get(int(layer_id), ())
        )

    def assert_no_transactions(self) -> None:
        if self._transactions:
            raise RuntimeError(
                f"active cache transactions remain: {sorted(self._transactions)}"
            )
