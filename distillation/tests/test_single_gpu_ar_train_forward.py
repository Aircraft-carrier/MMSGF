"""Single-GPU H100 smoke test: one real autoregressive train forward (V/A/G).

Loads the autoregressive MOT transformer from a checkpoint export and runs
exactly one joint training step through the same code path as
``MOTTrainer._train_step``::

    convert_input_format -> _materialize_batch_latents
    -> _prepare_joint_input_dict -> model(input_dict, mode="train")
    -> compute_loss -> loss.backward()

FSDP is intentionally not enabled; the point is to debug the forward/backward
on one H100 without sharding.  Activation checkpointing mirrors the real
training path and can be disabled with ``--no-activation-checkpoint``.

The synthetic batch follows the production UMI MOT packing (see
``test_mot_incremental_adapter_ipdb_trace.py`` for the same contract):

- latents               [B, 48, 8, V, 14, 14]
- actions               [B, 20, 8, 16, 1]
- geometry_rgb          [B, 8, 4, V, 3, 224, 224]
- geometry_pts3d        [B, 8, 4, V, 224, 224, 3]
- video/action masks    per real training masks; frames 0..3 history,
                        frame 4 anchor, frames 5..7 supervised targets

The dataset normally yields ``vae_rgb_history``/``vae_rgb_target`` and the
trainer VAE-encodes them; this script synthesizes the post-VAE ``latents``
directly so no VAE weights are loaded.  All other keys and shapes match the
batch produced by ``_materialize_batch_latents``.

Usage (run from the repository root on the H100 host)::

    PYTHONPATH=. python distillation/tests/test_single_gpu_ar_train_forward.py \
        --checkpoint models/uni3dwam_video_only_step62000

    # or point directly at a transformer export
    PYTHONPATH=. python distillation/tests/test_single_gpu_ar_train_forward.py \
        --transformer /path/to/checkpoint_step_1000/transformer

The default checkpoint is ``$DISTILL_STUDENT_INIT`` (then
``$MOT_INITIALIZE_FROM``), falling back to
``<repo>/models/uni3dwam_video_only_step62000``.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from distillation.model.autoregressive_mot import (
    AutoregressiveThreeDVAMOTTransformer3DModel,
)
from distillation.trainer.autoregressive import AutoregressiveTrainer
from wan_va.train_mot import (
    apply_ac_mot,
    apply_ac_vggto,
    optimization_branches,
    validate_mot_training_config,
)
from utils import FlowMatchScheduler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSFORMER = REPO_ROOT / "models" / "uni3dwam_video_only_step62000" / "transformer"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transformer",
        type=str,
        default=None,
        help="Transformer export dir containing config.json + diffusion_pytorch_model.safetensors.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint root that contains a transformer/ export.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float32", "float16"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--views", type=int, default=2, choices=(2, 3))
    parser.add_argument(
        "--latent-size",
        type=int,
        default=14,
        help="VAE latent spatial size; production MOT data uses 14.",
    )
    parser.add_argument(
        "--text-seq-len",
        type=int,
        default=8,
        help="Synthetic text embedding sequence length.",
    )
    parser.add_argument(
        "--optimization-composition",
        type=str,
        default="vag",
        choices=("vag", "va", "v"),
        help="Composition for compute_loss; 'vag' exercises all three losses.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=("fa4", "flex", "dense"),
        help="Override the masked-attention backend; defaults to the checkpoint config.",
    )
    parser.add_argument("--no-activation-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def resolve_transformer_dir(args: argparse.Namespace) -> Path:
    raw: Path
    if args.transformer:
        raw = Path(args.transformer)
    elif args.checkpoint:
        raw = Path(args.checkpoint)
    else:
        env = os.environ.get("DISTILL_STUDENT_INIT") or os.environ.get(
            "MOT_INITIALIZE_FROM"
        )
        raw = Path(env) if env else DEFAULT_TRANSFORMER
    candidate = raw / "transformer" if (raw / "transformer").is_dir() else raw
    for name in ("config.json", "diffusion_pytorch_model.safetensors"):
        if not (candidate / name).is_file():
            raise FileNotFoundError(
                f"missing {candidate / name}; pass --checkpoint or --transformer"
            )
    return candidate


def build_config(args: argparse.Namespace, device: torch.device) -> Any:
    from distillation.configs.autoregressive_training import (
        autoregressive_training_cfg,
    )

    config = copy.deepcopy(autoregressive_training_cfg)
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.device = device
    config.seed = args.seed
    config.optimization_composition = args.optimization_composition
    config.enable_wandb = False
    validate_mot_training_config(config)
    return config


def load_model(
    transformer_dir: Path,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    *,
    backend: str | None,
    activation_checkpoint: bool,
) -> torch.nn.Module:
    model = AutoregressiveThreeDVAMOTTransformer3DModel.from_pretrained(
        str(transformer_dir),
        torch_dtype=dtype,
    )
    model.to(device=device, dtype=dtype)
    if backend is not None:
        model.masked_attn_backend = backend
        model.vggto.masked_attn_backend = backend
    if activation_checkpoint:
        apply_ac_mot(model, execution_route="joint")
        apply_ac_vggto(model)
    model.vggto.keep_heads_fp32_()
    model.train()
    model.requires_grad_(True)
    model.configure_generation_profile(config.distill.generation_shape)
    return model


def build_trainer_shell(
    config: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> AutoregressiveTrainer:
    """A minimal MOTTrainer whose methods prepare the real training input.

    ``MOTTrainer.__init__`` additionally builds the dataset, FSDP, optimizer,
    and wandb; none of that is needed for one forward.  Only the attributes
    consumed by ``convert_input_format`` / ``_materialize_batch_latents`` /
    ``_prepare_joint_input_dict`` / ``compute_loss`` are attached.
    """

    trainer = AutoregressiveTrainer.__new__(AutoregressiveTrainer)
    trainer.config = config
    trainer.device = device
    trainer.dtype = config.param_dtype
    trainer.optimization_composition = config.optimization_composition
    trainer.optimization_branches = optimization_branches(
        config.optimization_composition
    )
    trainer.execution_route = "joint"
    trainer.gradient_accumulation_steps = int(config.gradient_accumulation_steps)
    trainer.train_scheduler_latent = FlowMatchScheduler(
        shift=float(config.snr_shift),
        sigma_min=0.0,
        extra_one_step=True,
    )
    trainer.train_scheduler_latent.set_timesteps(1000, training=True)
    trainer.train_scheduler_action = FlowMatchScheduler(
        shift=float(config.action_snr_shift),
        sigma_min=0.0,
        extra_one_step=True,
    )
    trainer.train_scheduler_action.set_timesteps(1000, training=True)
    trainer.transformer = model
    return trainer


def build_synthetic_batch(
    *,
    batch_size: int,
    views: int,
    latent_size: int,
    text_seq_len: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """One batch with the exact keys/shapes of a production MOT sample."""

    generator = torch.Generator().manual_seed(seed)
    frames = 8
    batch = {
        # Post-VAE latents.  Real training materializes these from
        # vae_rgb_history/vae_rgb_target; here they are synthesized directly.
        "latents": torch.randn(
            batch_size,
            48,
            frames,
            views,
            latent_size,
            latent_size,
            generator=generator,
        ),
        "actions": torch.rand(
            batch_size, 20, frames, 16, 1, generator=generator
        )
        * 2.0
        - 1.0,
        "geometry_rgb": torch.rand(
            batch_size,
            frames,
            4,
            views,
            3,
            224,
            224,
            generator=generator,
        ),
        "geometry_pts3d": torch.randn(
            batch_size,
            frames,
            4,
            views,
            224,
            224,
            3,
            generator=generator,
        ),
        "geometry_point_valid_mask": torch.ones(
            batch_size, frames, 4, views, 224, 224, dtype=torch.bool
        ),
        "geometry_group_valid_mask": torch.ones(
            batch_size, frames, 4, dtype=torch.bool
        ),
        "stream_ids": torch.arange(views, dtype=torch.long).expand(
            batch_size, -1
        ),
        # Frames 0..3 history, frame 4 anchor, frames 5..7 supervised targets.
        "video_latent_loss_mask": torch.zeros(
            batch_size, frames, dtype=torch.bool
        ),
        "video_latent_valid_mask": torch.ones(
            batch_size, frames, dtype=torch.bool
        ),
        "action_loss_mask": torch.zeros(
            batch_size, 20, frames, 16, 1, dtype=torch.bool
        ),
        "action_valid_mask": torch.ones(
            batch_size, 20, frames, 16, 1, dtype=torch.bool
        ),
        "text_emb": torch.randn(
            batch_size, text_seq_len, 4096, generator=generator
        )
        * 0.1,
        "empty_text_emb": torch.zeros(1, text_seq_len, 4096),
        "has_pointcloud": torch.ones(batch_size, dtype=torch.bool),
        "dataset_skip_count": torch.zeros(batch_size, dtype=torch.int64),
        "action_q01": torch.zeros(batch_size, 20),
        "action_q99": torch.ones(batch_size, 20),
    }
    batch["video_latent_loss_mask"][:, 5:] = True
    batch["action_loss_mask"][:, :, 5:] = True
    return batch


def run_single_train_forward(
    trainer: AutoregressiveTrainer,
    model: torch.nn.Module,
    device: torch.device,
    *,
    batch_size: int,
    views: int,
    latent_size: int,
    text_seq_len: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, float]:
    batch = build_synthetic_batch(
        batch_size=batch_size,
        views=views,
        latent_size=latent_size,
        text_seq_len=text_seq_len,
        seed=seed,
    )
    batch = trainer.convert_input_format(batch)
    batch = trainer._materialize_batch_latents(batch)
    input_dict = trainer._prepare_joint_input_dict(batch)

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model(input_dict, mode="train")
    losses = trainer.compute_loss(input_dict, output)
    losses["loss"].backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    return losses, output, grad_norm, elapsed


def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("this test is designed for a single CUDA (H100) GPU")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }[args.dtype]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    transformer_dir = resolve_transformer_dir(args)
    config = build_config(args, device)
    print(f"[ar-train] checkpoint transformer: {transformer_dir}")
    print(f"[ar-train] device={device} dtype={args.dtype} "
          f"views={args.views} composition={config.optimization_composition}")

    model = load_model(
        transformer_dir,
        config,
        device,
        dtype,
        backend=args.backend,
        activation_checkpoint=not args.no_activation_checkpoint,
    )
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"[ar-train] loaded {n_params / 1e9:.2f}B parameters")

    trainer = build_trainer_shell(config, model, device)
    losses, output, grad_norm, elapsed = run_single_train_forward(
        trainer,
        model,
        device,
        batch_size=args.batch_size,
        views=args.views,
        latent_size=args.latent_size,
        text_seq_len=args.text_seq_len,
        seed=args.seed,
    )

    print("[ar-train] forward+loss+backward completed")
    for name in (
        "loss",
        "latent_loss",
        "action_loss",
        "depth_loss",
        "point_loss",
    ):
        value = losses.get(name)
        if value is not None:
            print(f"[ar-train] {name}={_scalar(value):.6f}")
    print(f"[ar-train] grad_norm={_scalar(grad_norm):.4f}")
    print(f"[ar-train] pred latent {tuple(output['latent_pred'].shape)} "
          f"action {tuple(output['action_pred'].shape)} "
          f"depth {tuple(output['depth'].shape)}")
    print(f"[ar-train] forward+backward time={elapsed:.3f}s")
    print(
        f"[ar-train] peak cuda memory="
        f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f}GB"
    )

    if not bool(torch.isfinite(losses["loss"])):
        print("[ar-train] FAIL: non-finite loss", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
