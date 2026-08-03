"""Distillation model loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from distillation.model.utils import freeze_model, set_trainable
from distillation.mask_profile import install_order_profile
from wan_va.modules.model_3dva_mot import ThreeDVAMOTTransformer3DModel


def load_transformer_export(
    checkpoint_path: str | Path,
    config: Any,
) -> ThreeDVAMOTTransformer3DModel:
    """Load only a published cross-stage ``transformer/`` export.

    checkpoint root 必须有 ``_SUCCESS``、MOT-compatible metadata、config 和
    safetensors。这里不会读取 DCP optimizer state;同方法 resume 由
    ``DistillationCheckpointIO.load`` 负责，跨方法初始化只消费 export。
    """
    from wan_va.train_mot import MOTTrainer

    checkpoint_path = Path(checkpoint_path)
    MOTTrainer._validate_transformer_checkpoint_layout(checkpoint_path)
    transformer_path = checkpoint_path / "transformer"
    return ThreeDVAMOTTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=config.param_dtype,
    )


def build_frozen_transformer(
    checkpoint_path: str | Path,
    config: Any,
    device: torch.device,
) -> ThreeDVAMOTTransformer3DModel:
    model = load_transformer_export(checkpoint_path, config)
    return _configure_distillation_model(model, config, device, trainable=False)


def build_trainable_transformer(
    checkpoint_path: str | Path,
    config: Any,
    device: torch.device,
) -> ThreeDVAMOTTransformer3DModel:
    model = load_transformer_export(checkpoint_path, config)
    return _configure_distillation_model(model, config, device, trainable=True)


def _configure_distillation_model(
    model: ThreeDVAMOTTransformer3DModel,
    config: Any,
    device: torch.device,
    *,
    trainable: bool,
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
    if trainable:
        set_trainable(model)
        apply_mot_parameter_ownership(model, config.optimization_composition)
        apply_ac_mot(model, execution_route=execution_route)
        apply_ac_vggto(model)
    else:
        freeze_model(model)
    install_order_profile(model, config.distill.generation_shape)
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
