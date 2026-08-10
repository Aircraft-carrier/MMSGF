#!/usr/bin/env python3
"""Load a WAN2.2 video backbone using the RoboTwin MOT training path.

Example:
    python tests/robotwin_weight_loading.py \
        --checkpoint playground/Pretrained_models/Wan2.2-TI2V-5B

The default ``video-backbone`` mode mirrors a fresh training run with
``init_model_from_lingbot=False`` and ``initialize_from=None``: it loads the
base WAN2.2 transformer, creates the Video+Action MOT model, then initializes
the action blocks from the video blocks.  Use ``mot-checkpoint`` only for an
already exported MOT checkpoint that contains the action weights.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.modules.model_va_mot import VAMOTTransformer3DModel


def transformer_directory(checkpoint: Path) -> Path:
    """Accept a model root, a checkpoint directory, or its transformer directory."""

    checkpoint = checkpoint.resolve()
    transformer_path = checkpoint / "transformer"
    if transformer_path.is_dir():
        return transformer_path
    if (checkpoint / "config.json").is_file():
        return checkpoint
    raise FileNotFoundError(
        f"No transformer weights found at {checkpoint}. Expected either "
        "<checkpoint>/transformer/ or a transformer directory with config.json."
    )


def mot_training_overrides() -> dict[str, object]:
    """Match the new-model initialization parameters in ``train_mot.py``."""

    return {
        "max_num_views": 3,
        "action_dim": 20,
        "action_hidden_dim": 768,
        "action_ffn_dim": 3072,
        "attn_mode": "torch",
        "num_layers": 30,
        "masked_attn_backend": "fa4",
        "init_noise_seed": 42,
    }


def load_robotwin_transformer(
    checkpoint: Path,
    *,
    dtype: torch.dtype,
    load_mode: str,
) -> VAMOTTransformer3DModel:
    """Load either a base WAN2.2 video backbone or a complete MOT export."""

    transformer_path = transformer_directory(checkpoint)
    if load_mode == "video-backbone":
        model, _report = VAMOTTransformer3DModel.from_video_backbone(
            str(transformer_path),
            init_model_from_lingbot=False,
            config_overrides=mot_training_overrides(),
        )
        return model.to(dtype=dtype)

    return VAMOTTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=os.getenv("WAN22_PRETRAINED_MODEL_PATH"),
        required=os.getenv("WAN22_PRETRAINED_MODEL_PATH") is None,
        help="WAN2.2 model root or MOT checkpoint. Defaults to WAN22_PRETRAINED_MODEL_PATH.",
    )
    parser.add_argument(
        "--load-mode",
        choices=("video-backbone", "mot-checkpoint"),
        default="video-backbone",
        help="Use video-backbone to mirror fresh MOT training (default).",
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cpu", help="Optional device after loading, e.g. cuda:0.")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    model = load_robotwin_transformer(
        args.checkpoint,
        dtype=dtype,
        load_mode=args.load_mode,
    ).to(args.device)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        "\n".join(
            (
                f"transformer_dir: {transformer_directory(args.checkpoint)}",
                f"load_mode: {args.load_mode}",
                f"device: {next(model.parameters()).device}",
                f"dtype: {next(model.parameters()).dtype}",
                f"parameters: {parameter_count}",
                f"trainable_parameters: {trainable_count}",
            )
        )
    )


if __name__ == "__main__":
    main()
