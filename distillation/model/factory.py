"""Distillation model loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import torch

from distillation.model.utils import freeze_model, set_trainable
from distillation.model.autoregressive_mot import (
    AutoregressiveVAMOTTransformer3DModel,
)
from wan_va.modules.model_va_mot import VAMOTTransformer3DModel


def load_transformer_export(
    checkpoint_path: str | Path,
    config: Any,
    *,
    autoregressive: bool = True,
) -> AutoregressiveVAMOTTransformer3DModel:
    """Load only a published cross-stage ``transformer/`` export.

    checkpoint root 必须有 ``_SUCCESS``、MOT-compatible metadata、config 和
    safetensors。这里不会读取 DCP optimizer state；跨方法初始化只消费
    export。
    """
    checkpoint_path = Path(checkpoint_path)
    required = (
        checkpoint_path / "_SUCCESS",
        checkpoint_path / "checkpoint_metadata.json",
        checkpoint_path / "transformer" / "config.json",
        checkpoint_path / "transformer" / "diffusion_pytorch_model.safetensors",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Distillation initialization requires a completed transformer export; "
            "missing: " + ", ".join(missing)
        )
    metadata = json.loads(required[1].read_text(encoding="utf-8"))
    expected_architecture = (
        str(getattr(config.distill, "model_architecture", "autoregressive_va_mot_v1"))
        if autoregressive
        else "va_mot_v1"
    )
    if metadata.get("model_architecture") != expected_architecture:
        raise ValueError(
            "Checkpoint model architecture does not match distillation model: "
            f"checkpoint={metadata.get('model_architecture')!r}, "
            f"expected={expected_architecture!r}"
        )
    transformer_path = checkpoint_path / "transformer"
    model_cls = (
        AutoregressiveVAMOTTransformer3DModel
        if autoregressive
        else VAMOTTransformer3DModel
    )
    model = model_cls.from_pretrained(
        transformer_path,
        torch_dtype=config.param_dtype,
    )
    masked_attn_backend = getattr(config, "masked_attn_backend", None)
    if masked_attn_backend is not None:
        model.masked_attn_backend = str(masked_attn_backend)
    return model


def build_frozen_transformer(
    checkpoint_path: str | Path,
    config: Any,
    device: torch.device,
    *,
    autoregressive: bool = True,
) -> AutoregressiveVAMOTTransformer3DModel:
    """Load a frozen transformer."""
    model = load_transformer_export(
        checkpoint_path,
        config,
        autoregressive=autoregressive,
    )
    return _configure_distillation_model(
        model,
        config,
        device,
        trainable=False,
    )


def build_trainable_transformer(
    checkpoint_path: str | Path,
    config: Any,
    device: torch.device,
    *,
    autoregressive: bool = True,
) -> AutoregressiveVAMOTTransformer3DModel:
    model = load_transformer_export(
        checkpoint_path,
        config,
        autoregressive=autoregressive,
    )
    return _configure_distillation_model(
        model,
        config,
        device,
        trainable=True,
    )


def _configure_distillation_model(
    model: VAMOTTransformer3DModel,
    config: Any,
    device: torch.device,
    *,
    trainable: bool,
) -> VAMOTTransformer3DModel:
    from functools import partial

    from wan_va.distributed.util import _configure_model
    from wan_va.train_mot import (
        apply_ac_mot,
        apply_mot_parameter_ownership,
        shard_mot_model,
    )

    if trainable:
        set_trainable(model)
        apply_mot_parameter_ownership(model)
        apply_ac_mot(model)
    else:
        freeze_model(model)
    configured = _configure_model(
        model=model,
        shard_fn=partial(shard_mot_model),
        param_dtype=config.param_dtype,
        device=device,
        eval_mode=not trainable,
    )
    return configured
