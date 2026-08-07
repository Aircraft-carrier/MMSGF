"""Single-GPU H100 smoke test: one full self rollout for debugging.

Loads the autoregressive MOT transformer from a checkpoint export and runs
``distillation.self_rollout.self_rollout`` once, mirroring the consistency
trainer's rollout path::

    resolve_ground_truth_provider -> self_rollout(
        batch, transformer, config, spec, device, empty_text_emb,
        decode_latents_to_rgb_views, video_num_steps, action_num_steps,
        rollout_frames, ground_truth_provider, replacement_policy)

The synthetic batch uses the production UMI MOT packing: latents
``[B,48,8,V,14,14]``, actions ``[B,20,8,16,1]``, geometry RGB
``[B,8,4,V,3,224,224]``; frames 0..3 are history, frame 4 the known anchor,
and frames 5..7 the generated targets (``--rollout-frames`` controls how many
of them are rolled out).

The real rollout decodes predicted latents with the Wan VAE; this script
replaces that with a deterministic bilinear-upsampled mock (no VAE weights are
loaded).  Ground-truth replacement defaults to ``offline`` so the batch GT
replays the video/geometry/action commits after each prediction, which also
exercises the replacement machinery.  Use ``--gt-mode none`` to run pure
prediction instead.

Usage (run from the repository root on the H100 host)::

    PYTHONPATH=. python distillation/tests/test_single_gpu_self_rollout.py \
        --checkpoint models/uni3dwam_video_only_step62000

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
from typing import Any, Callable

import torch
import torch.nn.functional as F

from distillation.model.autoregressive_mot import (
    AutoregressiveThreeDVAMOTTransformer3DModel,
)
from distillation.model.utils import temporary_masked_attention_backend
from distillation.self_rollout import (
    resolve_ground_truth_provider,
    self_rollout,
)
from wan_va.mot_spec import mot_spec_from_config


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
    parser.add_argument("--rollout-frames", type=int, default=3)
    parser.add_argument("--video-steps", type=int, default=2)
    parser.add_argument("--action-steps", type=int, default=2)
    parser.add_argument(
        "--gt-mode",
        type=str,
        default="offline",
        choices=("none", "offline"),
        help="Ground-truth replacement mode for the rollout.",
    )
    parser.add_argument(
        "--replacement-policy",
        type=str,
        default="require_ground_truth",
        choices=("require_ground_truth", "recompute_predicted"),
    )
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
    config.distill.rollout_video_num_steps = args.video_steps
    config.distill.rollout_action_num_steps = args.action_steps
    config.distill.rollout_horizon_frames = args.rollout_frames
    config.distill.rollout_gt_mode = args.gt_mode
    config.distill.rollout_replacement_policy = args.replacement_policy
    config.distill.rollout_masked_attn_backend = "dense"
    return config


def load_model(
    transformer_dir: Path,
    config: Any,
    device: torch.device,
) -> torch.nn.Module:
    model = AutoregressiveThreeDVAMOTTransformer3DModel.from_pretrained(
        str(transformer_dir),
        torch_dtype=torch.bfloat16,
    )
    model.to(device=device, dtype=torch.bfloat16)
    model.configure_generation_profile(config.distill.generation_shape)
    model.eval()
    model.requires_grad_(False)
    return model


def build_synthetic_batch(
    *,
    device: torch.device,
    batch_size: int,
    views: int,
    latent_size: int,
    text_seq_len: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """One batch matching the self-rollout input contract and production data."""

    generator = torch.Generator().manual_seed(seed)
    frames = 8
    batch = {
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
    }
    batch["video_latent_loss_mask"][:, 5:] = True
    batch["action_loss_mask"][:, :, 5:] = True
    return {key: value.to(device) for key, value in batch.items()}


def build_mock_decode(
    *,
    vae_temporal_factor: int,
    rgb_size: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Deterministic stand-in for the Wan VAE decoder.

    The engine calls ``decode_latents_to_rgb_views(latents)`` with
    ``[B,C,F,V,H,W]`` and expects ``[B,1+vae*(F-1),V,3,H,W]`` RGB in [0,1].
    """

    def decode_latents_to_rgb_views(
        latents: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _channels, latent_frames, views, height, width = (
            latents.shape
        )
        mono = latents.mean(dim=1, keepdim=True).clamp(0, 1)
        parts = [mono[:, :, 0:1]]
        for index in range(1, latent_frames):
            frame = mono[:, :, index : index + 1]
            parts.append(
                frame.expand(
                    -1,
                    -1,
                    vae_temporal_factor,
                    -1,
                    -1,
                    -1,
                ).reshape(
                    batch_size,
                    1,
                    vae_temporal_factor,
                    views,
                    height,
                    width,
                )
            )
        rgb = torch.cat(parts, dim=2)  # [B,1,T,V,H,W]
        rgb = rgb.expand(-1, 3, -1, -1, -1, -1)  # [B,3,T,V,H,W]
        rgb = rgb.permute(0, 2, 3, 1, 4, 5)  # [B,T,V,3,H,W]
        flat = rgb.reshape(
            batch_size * rgb.shape[1] * views,
            3,
            height,
            width,
        )
        up = F.interpolate(
            flat,
            size=(rgb_size, rgb_size),
            mode="bilinear",
            align_corners=False,
        )
        return up.reshape(
            batch_size,
            rgb.shape[1],
            views,
            3,
            rgb_size,
            rgb_size,
        )

    return decode_latents_to_rgb_views


def run_self_rollout(
    model: torch.nn.Module,
    config: Any,
    device: torch.device,
    *,
    batch_size: int,
    views: int,
    latent_size: int,
    text_seq_len: int,
    rollout_frames: int,
    video_steps: int,
    action_steps: int,
    seed: int,
) -> tuple[Any, float]:
    batch = build_synthetic_batch(
        device=device,
        batch_size=batch_size,
        views=views,
        latent_size=latent_size,
        text_seq_len=text_seq_len,
        seed=seed,
    )
    ground_truth_provider = resolve_ground_truth_provider(
        str(config.distill.rollout_gt_mode),
        batch,
    )

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), temporary_masked_attention_backend(
        model,
        str(config.distill.rollout_masked_attn_backend),
    ):
        result = self_rollout(
            batch,
            transformer=model,
            config=config,
            spec=mot_spec_from_config(config),
            device=device,
            empty_text_emb=batch["empty_text_emb"],
            decode_latents_to_rgb_views=build_mock_decode(
                vae_temporal_factor=int(config.vae_temporal_factor),
                rgb_size=int(config.height),
            ),
            video_num_steps=video_steps,
            action_num_steps=action_steps,
            rollout_frames=rollout_frames,
            ground_truth_provider=ground_truth_provider,
            replacement_policy=str(
                config.distill.rollout_replacement_policy
            ),
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return result, elapsed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("this test is designed for a single CUDA (H100) GPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    transformer_dir = resolve_transformer_dir(args)
    config = build_config(args, device)
    print(f"[self-rollout] checkpoint transformer: {transformer_dir}")
    print(
        f"[self-rollout] device={device} views={args.views} "
        f"rollout_frames={args.rollout_frames} video_steps={args.video_steps} "
        f"action_steps={args.action_steps} gt_mode={args.gt_mode}"
    )

    model = load_model(transformer_dir, config, device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"[self-rollout] loaded {n_params / 1e9:.2f}B parameters")

    result, elapsed = run_self_rollout(
        model,
        config,
        device,
        batch_size=1,
        views=args.views,
        latent_size=args.latent_size,
        text_seq_len=args.text_seq_len,
        rollout_frames=args.rollout_frames,
        video_steps=args.video_steps,
        action_steps=args.action_steps,
        seed=args.seed,
    )

    diagnostics = result.diagnostics
    print("[self-rollout] rollout completed")
    print(
        f"[self-rollout] pred latents {tuple(result.pred_latents.shape)} "
        f"target latents {tuple(result.target_latents.shape)}"
    )
    print(
        f"[self-rollout] pred actions {tuple(result.pred_actions.shape)} "
        f"target actions {tuple(result.target_actions.shape)}"
    )
    print(
        f"[self-rollout] pred geometry {tuple(result.pred_geometry_rgb.shape)} "
        f"target geometry {tuple(result.target_geometry_rgb.shape)}"
    )
    print(
        f"[self-rollout] frame sources: "
        f"{ {frame_id: sources for frame_id, sources in diagnostics['sources'].items()} }"
    )
    replacements = diagnostics.get("replacements", [])
    print(
        f"[self-rollout] replacements: {len(replacements)} "
        f"({[item['frame_id'] for item in replacements]})"
    )
    print(
        f"[self-rollout] cache tokens: "
        f"{diagnostics.get('mot_cache_tokens', {})}"
    )
    print(f"[self-rollout] rollout time={elapsed:.3f}s")
    print(
        f"[self-rollout] peak cuda memory="
        f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f}GB"
    )

    if not bool(torch.isfinite(result.pred_latents).all()):
        print("[self-rollout] FAIL: non-finite predicted latents", file=sys.stderr)
        return 1
    if not bool(torch.isfinite(result.pred_actions).all()):
        print("[self-rollout] FAIL: non-finite predicted actions", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
