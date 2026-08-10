"""Distillation training entrypoint."""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist

from distillation.configs import (
    AUTOREGRESSIVE_TRAINING,
    CONFIGS,
    CONSISTENCY_DISTILLATION,
    METHODS,
    SELF_GRADIENT_FORCING_DMD,
)
from wan_va.distributed.util import init_distributed


def _parse_timestep_list(value: str) -> list[float]:
    parts = value.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "timestep list must be a comma-separated sequence of numbers"
        )
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid timestep list {value!r}"
        ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a distilled MOT transformer.")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--save-root", required=True, type=str,
                        help="Directory for checkpoints and logs.")
    parser.add_argument("--student-init", default=None, type=str,
                        help="Checkpoint root used to initialize the student transformer.")
    parser.add_argument("--teacher-checkpoint", default=None, type=str,
                        help="Teacher checkpoint for consistency distillation.")
    parser.add_argument("--real-score-checkpoint", default=None, type=str,
                        help="Frozen real-score checkpoint for DMD.")
    parser.add_argument("--fake-score-init", default=None, type=str,
                        help="Trainable fake-score initialization for DMD.")
    parser.add_argument("--resume-from", default=None, type=str,
                        help="Checkpoint root to resume the selected method.")
    parser.add_argument("--rollout-interval", default=None, type=int,
                        help="Run a consistency rollout every N completed training steps; <=0 disables it.")
    parser.add_argument("--rollout-video-num-steps", default=None, type=int,
                        help="Video denoising steps used by each training rollout.")
    parser.add_argument("--rollout-action-num-steps", default=None, type=int,
                        help="Action denoising steps used by each training rollout.")
    parser.add_argument(
        "--video-denoisy-step-list",
        default=None,
        type=_parse_timestep_list,
        help="Comma-separated SGF video timesteps, strictly descending.",
    )
    parser.add_argument(
        "--action-denoisy-step-list",
        default=None,
        type=_parse_timestep_list,
        help="Comma-separated SGF action timesteps, strictly descending.",
    )
    parser.add_argument("--rollout-horizon-frames", default=None, type=int,
                        help="Logical target frames generated after the clean target anchor.")
    parser.add_argument(
        "--rollout-gt-mode",
        default=None,
        choices=("none", "offline", "provider"),
        help="Choose no GT replacement, offline batch GT, or an explicit provider API.",
    )
    parser.add_argument(
        "--rollout-replacement-policy",
        default=None,
        choices=("require_ground_truth", "recompute_predicted"),
        help="How to rebuild downstream action state after a partial GT replacement.",
    )
    parser.add_argument("--cfg-min", default=None, type=float,
                        help="Minimum teacher video CFG scale for consistency distillation.")
    parser.add_argument("--cfg-max", default=None, type=float,
                        help="Maximum teacher video CFG scale for consistency distillation.")
    parser.add_argument("--teacher-cfg-min", default=None, type=float,
                        help="Minimum frozen-teacher video CFG scale for SGF.")
    parser.add_argument("--teacher-cfg-max", default=None, type=float,
                        help="Maximum frozen-teacher video CFG scale for SGF.")
    parser.add_argument("--sigma-data", default=None, type=float,
                        help="Flash-WAM video consistency boundary scaling sigma_data.")
    return parser.parse_args(argv)


def _trainer_class(method: str):
    from distillation.trainer.autoregressive import AutoregressiveTrainer
    from distillation.trainer.consistency import ConsistencyTrainer
    from distillation.trainer.self_gradient_forcing_dmd import (
        SelfGradientForcingDMDTrainer,
    )

    return {
        AUTOREGRESSIVE_TRAINING: AutoregressiveTrainer,
        CONSISTENCY_DISTILLATION: ConsistencyTrainer,
        SELF_GRADIENT_FORCING_DMD: SelfGradientForcingDMDTrainer,
    }[method]


def run(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    init_distributed(world_size, local_rank, rank)

    config = copy.deepcopy(CONFIGS[args.method])
    config.save_root = str(Path(args.save_root).resolve())
    for name in (
        "student_init",
        "teacher_checkpoint",
        "real_score_checkpoint",
        "fake_score_init",
        "resume_from",
    ):
        value = getattr(args, name)
        if value:
            config.distill[name] = Path(value)
    for name in (
        "rollout_interval",
        "rollout_video_num_steps",
        "rollout_action_num_steps",
        "rollout_horizon_frames",
    ):
        value = getattr(args, name)
        if value is not None:
            config.distill[name] = int(value)
    if args.method == SELF_GRADIENT_FORCING_DMD:
        if (
            args.rollout_video_num_steps is not None
            or args.rollout_action_num_steps is not None
        ):
            raise ValueError(
                "self_gradient_forcing_dmd uses --video/--action-denoisy-step-list "
                "instead of rollout num steps"
            )
        if args.video_denoisy_step_list is not None:
            config.distill.denoisy_step_list.video = list(
                args.video_denoisy_step_list
            )
        if args.action_denoisy_step_list is not None:
            config.distill.denoisy_step_list.action = list(
                args.action_denoisy_step_list
            )
    elif (
        args.video_denoisy_step_list is not None
        or args.action_denoisy_step_list is not None
    ):
        raise ValueError("denoisy step list flags are only valid for SGF DMD")
    for name in ("cfg_min", "cfg_max", "sigma_data"):
        value = getattr(args, name)
        if value is not None:
            config.distill[name] = float(value)
    for name in ("teacher_cfg_min", "teacher_cfg_max"):
        value = getattr(args, name)
        if value is not None:
            config.distill[name] = float(value)
    for name in ("rollout_gt_mode", "rollout_replacement_policy"):
        value = getattr(args, name)
        if value is not None:
            config.distill[name] = str(value)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    config.device = torch.device(f"cuda:{local_rank}")

    trainer = _trainer_class(args.method)(config)

    try:
        trainer.train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    try:
        run(parse_args())
        return 0
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception("Distillation training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
