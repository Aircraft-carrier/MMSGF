#!/usr/bin/env python3
"""Interactively load a saved RoboTwin MOT transformer checkpoint.

Example:
    python -m ipdb tests/robotwin_weight_loading.py \
        --checkpoint /path/to/checkpoint_or_transformer
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
    """Accept either a checkpoint directory or its transformer subdirectory."""

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


def load_robotwin_transformer(checkpoint: Path, *, dtype: torch.dtype) -> VAMOTTransformer3DModel:
    """Load the complete transformer export saved by MOT training."""

    return VAMOTTransformer3DModel.from_pretrained(
        transformer_directory(checkpoint),
        torch_dtype=dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=os.getenv("MOT_INITIALIZE_FROM"),
        required=os.getenv("MOT_INITIALIZE_FROM") is None,
        help="Checkpoint directory or its transformer subdirectory. Defaults to MOT_INITIALIZE_FROM.",
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cpu", help="Optional device after loading, e.g. cuda:0.")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    model = load_robotwin_transformer(args.checkpoint, dtype=dtype).to(args.device)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        "\n".join(
            (
                f"transformer_dir: {transformer_directory(args.checkpoint)}",
                f"device: {next(model.parameters()).device}",
                f"dtype: {next(model.parameters()).dtype}",
                f"parameters: {parameter_count}",
                f"trainable_parameters: {trainable_count}",
            )
        )
    )


if __name__ == "__main__":
    main()
