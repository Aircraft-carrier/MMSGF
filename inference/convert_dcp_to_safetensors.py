import argparse
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

'''
    python inference/convert_dcp_to_safetensors.py train_logs/umi_mot_full_pretrain_8nodes_0714_2nd_try/checkpoints/checkpoint_step_63000
'''


DCP_DIR_NAME = "distributed_state"
MODEL_PREFIX = "model."
SAFETENSORS_NAME = "diffusion_pytorch_model.safetensors"


def _checkpoint_paths(checkpoint_dir: Path) -> tuple[Path, Path]:
    dcp_dir = checkpoint_dir / DCP_DIR_NAME
    config_path = checkpoint_dir / "transformer" / "config.json"
    required = (
        checkpoint_dir / "_SUCCESS",
        dcp_dir / ".metadata",
        config_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete MOT DCP checkpoint; missing: " + ", ".join(missing)
        )
    return dcp_dir, config_path


def _allocate_model_state(dcp_dir: Path) -> tuple[dict[str, torch.Tensor], int]:
    metadata = FileSystemReader(dcp_dir).read_metadata()
    model_state = {}
    total_bytes = 0
    unsupported = []

    for key, storage_metadata in metadata.state_dict_metadata.items():
        if not key.startswith(MODEL_PREFIX):
            continue
        if not isinstance(storage_metadata, TensorStorageMetadata):
            unsupported.append(key)
            continue

        name = key[len(MODEL_PREFIX) :]
        tensor = torch.empty(
            tuple(storage_metadata.size),
            dtype=storage_metadata.properties.dtype,
        )
        model_state[name] = tensor
        total_bytes += math.prod(storage_metadata.size) * tensor.element_size()

    if unsupported:
        raise TypeError(
            "Safetensors cannot store non-tensor model entries: "
            + ", ".join(unsupported)
        )
    if not model_state:
        raise ValueError(f"No {MODEL_PREFIX} entries found in {dcp_dir / '.metadata'}")
    return model_state, total_bytes


def convert_dcp_to_safetensors(
    checkpoint_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    dcp_dir, source_config = _checkpoint_paths(checkpoint_dir)
    output_dir = (
        Path(output_dir) if output_dir is not None else checkpoint_dir / "transformer"
    )
    output_path = output_dir / SAFETENSORS_NAME
    output_config = output_dir / "config.json"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing safetensors file: {output_path}"
        )
    copy_config = source_config.resolve() != output_config.resolve()
    if copy_config and output_config.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing transformer config: {output_config}"
        )

    model_state, total_bytes = _allocate_model_state(dcp_dir)
    print(
        f"Loading {len(model_state)} model tensors from {dcp_dir} "
        f"({total_bytes / 1024**3:.2f} GiB on CPU)"
    )
    dcp.load({"model": model_state}, checkpoint_id=dcp_dir, no_dist=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = output_dir / f".{SAFETENSORS_NAME}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        save_file(model_state, temp_path)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)

    if copy_config:
        shutil.copy2(source_config, output_config)
    print(f"Saved transformer export to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a MOT DCP checkpoint model to safetensors"
    )
    parser.add_argument(
        "checkpoint_dir",
        help="Checkpoint root containing distributed_state/",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output transformer directory (default: CHECKPOINT_DIR/transformer)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing transformer export",
    )
    args = parser.parse_args()
    convert_dcp_to_safetensors(
        args.checkpoint_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
