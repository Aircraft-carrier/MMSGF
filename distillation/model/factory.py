"""Distillation model loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import torch

from distillation.model.utils import freeze_model, set_trainable
from distillation.mask_profile import validate_checkpoint_generation_profile
from distillation.model.autoregressive_mot import (
    AutoregressiveVAMOTTransformer3DModel,
)
from wan_va.modules.model_va_mot import VAMOTTransformer3DModel


def load_transformer_export(
    checkpoint_path: str | Path,
    config: Any,
    *,
    validate_distillation_profile: bool = True,
    autoregressive: bool = True,
) -> AutoregressiveVAMOTTransformer3DModel:
    """Load only a published cross-stage ``transformer/`` export.

    checkpoint root 必须有 ``_SUCCESS``、MOT-compatible metadata、config 和
    safetensors。这里不会读取 DCP optimizer state;同方法 resume 由
    ``DistillationCheckpointIO.load`` 负责，跨方法初始化只消费 export。
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
    if validate_distillation_profile:
        validate_checkpoint_generation_profile(
            checkpoint_path,
            config.distill.generation_shape,
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
    if autoregressive:
        model.configure_generation_profile(config.distill.generation_shape)
    masked_attn_backend = getattr(config, "masked_attn_backend", None)
    if masked_attn_backend is not None:
        model.masked_attn_backend = str(masked_attn_backend)
    return model


def build_frozen_transformer(
    checkpoint_path: str | Path,
    config: Any,
    device: torch.device,
    *,
    install_distillation_profile: bool = True,
    validate_distillation_profile: bool = True,
    autoregressive: bool = True,
) -> AutoregressiveVAMOTTransformer3DModel:
    """Load a frozen transformer.

    Stage2 teachers and EMA targets use the default distillation profile checks.
    Stage3's real-score teacher intentionally disables both switches so the
    source teacher checkpoint keeps its original wan_va attention mask.
    """
    model = load_transformer_export(
        checkpoint_path,
        config,
        validate_distillation_profile=validate_distillation_profile,
        autoregressive=autoregressive,
    )
    return _configure_distillation_model(
        model,
        config,
        device,
        trainable=False,
        install_distillation_profile=install_distillation_profile,
    )


def build_trainable_transformer(
    checkpoint_path: str | Path,
    config: Any,
    device: torch.device,
    *,
    install_distillation_profile: bool = True,
    validate_distillation_profile: bool = True,
    autoregressive: bool = True,
) -> AutoregressiveVAMOTTransformer3DModel:
    model = load_transformer_export(
        checkpoint_path,
        config,
        validate_distillation_profile=validate_distillation_profile,
        autoregressive=autoregressive,
    )
    return _configure_distillation_model(
        model,
        config,
        device,
        trainable=True,
        install_distillation_profile=install_distillation_profile,
    )


def _configure_distillation_model(
    model: VAMOTTransformer3DModel,
    config: Any,
    device: torch.device,
    *,
    trainable: bool,
    install_distillation_profile: bool = True,
) -> VAMOTTransformer3DModel:
    from functools import partial

    from wan_va.distributed.util import _configure_model
    from wan_va.train_mot import (
        apply_ac_mot,
        apply_mot_parameter_ownership,
        shard_mot_model,
    )

    if install_distillation_profile and hasattr(model, "configure_generation_profile"):
        model.configure_generation_profile(config.distill.generation_shape)
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
