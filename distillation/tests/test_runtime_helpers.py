"""Regression tests for distillation-only runtime state helpers."""
from __future__ import annotations

import pytest
from torch import nn

from distillation.model.utils import (
    temporary_fsdp_unshard,
    temporary_masked_attention_backend,
)


class _BackendModule(nn.Module):
    def __init__(self, backend: str | None = "fa4") -> None:
        super().__init__()
        if backend is not None:
            self.masked_attn_backend = backend


class _ShardModule(nn.Module):
    def __init__(self, events: list[str], name: str, *, fail: bool = False) -> None:
        super().__init__()
        self.events = events
        self.name = name
        self.fail = fail

    def unshard(self):
        self.events.append(f"unshard:{self.name}")
        if self.fail:
            raise RuntimeError(f"failed:{self.name}")

    def reshard(self):
        self.events.append(f"reshard:{self.name}")


def test_temporary_backend_restores_target_attribute() -> None:
    model = _BackendModule("fa4")

    with temporary_masked_attention_backend(model, "dense"):
        assert model.masked_attn_backend == "dense"

    assert model.masked_attn_backend == "fa4"


def test_temporary_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported masked attention backend"):
        with temporary_masked_attention_backend(_BackendModule(), "bogus"):
            pass


def test_temporary_fsdp_unshard_reshards_only_modules_unsharded_successfully() -> None:
    events: list[str] = []
    root = _ShardModule(events, "root")
    root.child = _ShardModule(events, "child")

    with temporary_fsdp_unshard(root):
        assert events == ["unshard:root", "unshard:child"]

    assert events == [
        "unshard:root",
        "unshard:child",
        "reshard:child",
        "reshard:root",
    ]


def test_temporary_fsdp_unshard_cleans_up_after_partial_failure() -> None:
    events: list[str] = []
    root = _ShardModule(events, "root")
    root.child = _ShardModule(events, "child", fail=True)

    with pytest.raises(RuntimeError, match="failed:child"):
        with temporary_fsdp_unshard(root):
            raise AssertionError("the context body must not run")

    assert events == ["unshard:root", "unshard:child", "reshard:root"]
