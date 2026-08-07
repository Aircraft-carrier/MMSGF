"""Distillation model loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from distillation.model.utils import freeze_model, set_trainable
from distillation.mask_profile import validate_checkpoint_generation_profile
from distillation.model.autoregressive_mot import (
    AutoregressiveThreeDVAMOTTransformer3DModel,
)
from wan_va.modules.model_3dva_mot import ThreeDVAMOTTransformer3DModel


def load_transformer_export(
    checkpoint_path: str | Path,
    config: Any,
    *,
    validate_distillation_profile: bool = True,
    autoregressive: bool = True,
) -> AutoregressiveThreeDVAMOTTransformer3DModel:
    """Load only a published cross-stage ``transformer/`` export.

    checkpoint root 必须有 ``_SUCCESS``、MOT-compatible metadata、config 和
    safetensors。这里不会读取 DCP optimizer state;同方法 resume 由
    ``DistillationCheckpointIO.load`` 负责，跨方法初始化只消费 export。
    """
    from wan_va.train_mot import MOTTrainer

    checkpoint_path = Path(checkpoint_path)
    MOTTrainer._validate_transformer_checkpoint_layout(checkpoint_path)
    if validate_distillation_profile:
        validate_checkpoint_generation_profile(
            checkpoint_path,
            config.distill.generation_shape,
        )
    transformer_path = checkpoint_path / "transformer"
    model_cls = (
        AutoregressiveThreeDVAMOTTransformer3DModel
        if autoregressive
        else ThreeDVAMOTTransformer3DModel
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
        if hasattr(model, "vggto"):
            model.vggto.masked_attn_backend = str(masked_attn_backend)
    return model


def build_frozen_transformer(
    checkpoint_path: str | Path,
    config: Any,
    device: torch.device,
    *,
    install_distillation_profile: bool = True,
    validate_distillation_profile: bool = True,
    autoregressive: bool = True,
) -> AutoregressiveThreeDVAMOTTransformer3DModel:
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
) -> AutoregressiveThreeDVAMOTTransformer3DModel:
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
    model: ThreeDVAMOTTransformer3DModel,
    config: Any,
    device: torch.device,
    *,
    trainable: bool,
    install_distillation_profile: bool = True,
) -> ThreeDVAMOTTransformer3DModel:
    from functools import partial

    from wan_va.distributed.util import _configure_model
    from wan_va.train_mot import (
        apply_ac_mot,
        apply_ac_vggto,
        apply_mot_parameter_ownership,
        shard_mot_model,
    )

    execution_route = getattr(config, "execution_route", "joint")
    if autoregressive and hasattr(model, "configure_generation_profile"):
        model.configure_generation_profile(config.distill.generation_shape)
    if trainable:
        set_trainable(model)
        apply_mot_parameter_ownership(model, config.optimization_composition)
        apply_ac_mot(model, execution_route=execution_route)
        apply_ac_vggto(model)
    else:
        freeze_model(model)
    configured = _configure_model(
        model=model,
        shard_fn=partial(
            shard_mot_model,
            execution_route=execution_route,
        ),
        param_dtype=config.param_dtype,
        device=device,
        eval_mode=not trainable,
    )
    return configured
