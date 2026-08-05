"""Model state helpers."""
from contextlib import contextmanager

import torch
import torch.nn as nn


_MISSING = object()


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval().requires_grad_(False)
    return model


def set_trainable(model: nn.Module) -> nn.Module:
    model.train().requires_grad_(True)
    return model


@contextmanager
def temporary_masked_attention_backend(model: nn.Module, backend: str | None):
    if backend is None:
        yield
        return

    if str(backend) not in {"dense", "fa4", "flex"}:
        raise ValueError(f"unsupported masked attention backend: {backend!r}")

    modules = []
    for module in (model, getattr(model, "vggto", None)):
        if module is not None and all(module is not existing for existing in modules):
            modules.append(module)
    previous = [
        (module, getattr(module, "masked_attn_backend", _MISSING))
        for module in modules
    ]
    try:
        for module, _old_backend in previous:
            module.masked_attn_backend = str(backend)
        yield
    finally:
        for module, old_backend in previous:
            if old_backend is _MISSING:
                if hasattr(module, "masked_attn_backend"):
                    delattr(module, "masked_attn_backend")
            else:
                module.masked_attn_backend = old_backend


@contextmanager
def temporary_fsdp_unshard(model: nn.Module):
    """Temporarily materialize FSDP2 DTensor parameters for internal calls.

    Distillation self-rollout intentionally calls MOT/VGGTO submodules directly
    to maintain an incremental KV cache.  That bypasses FSDP2 root pre-forward
    hooks, so FSDP parameters can remain as DTensors while rollout inputs are
    local tensors.  Use this only around no-grad rollout sections, then reshard
    immediately before returning to normal training forwards.
    """

    try:
        from torch.distributed._composable.fsdp import FSDPModule
    except Exception:
        FSDPModule = ()

    fsdp_modules: list[nn.Module] = []
    for module in model.modules():
        is_fsdp_module = isinstance(module, FSDPModule) if FSDPModule else False
        if is_fsdp_module or (
            callable(getattr(module, "unshard", None))
            and callable(getattr(module, "reshard", None))
        ):
            fsdp_modules.append(module)
    unsharded: list[nn.Module] = []
    try:
        for module in fsdp_modules:
            module.unshard()
            unsharded.append(module)
        yield
    finally:
        for module in reversed(unsharded):
            module.reshard()


@torch.no_grad()
def update_ema(ema_model: nn.Module, student_model: nn.Module, decay: float) -> None:
    for ema_param, student_param in zip(
        ema_model.parameters(),
        student_model.parameters(),
        strict=True,
    ):
        ema_param.mul_(decay).add_(student_param, alpha=1.0 - decay)
