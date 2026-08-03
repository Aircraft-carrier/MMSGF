"""Distillation workflow."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from distillation.checkpoint import find_latest_successful_checkpoint
from distillation.configs import (
    AUTOREGRESSIVE_TRAINING,
    CONSISTENCY_DISTILLATION,
    SELF_GRADIENT_FORCING_DMD,
    METHODS,
)

PIPELINE_FORMAT_VERSION = 2
PIPELINE_STATE_FILE = "pipeline_state.json"
_DISTILL_ENV_KEYS = (
    "DISTILL_STUDENT_INIT",
    "DISTILL_TEACHER_CHECKPOINT",
    "DISTILL_REAL_SCORE_CHECKPOINT",
    "DISTILL_FAKE_SCORE_INIT",
    "DISTILL_RESUME_FROM",
)


@dataclass(slots=True)
class WorkflowState:
    format_version: int = PIPELINE_FORMAT_VERSION
    completed: dict[str, str] = field(default_factory=dict)
    running_method: str | None = None
    failed_method: str | None = None

    @classmethod
    def load(cls, path: Path) -> "WorkflowState":
        if not path.is_file():
            return cls()
        state = cls(**json.loads(path.read_text(encoding="utf-8")))
        if state.format_version != PIPELINE_FORMAT_VERSION:
            raise ValueError(f"Unsupported pipeline state: {path}")
        return state


@dataclass(frozen=True, slots=True)
class MethodRun:
    name: str
    student_init: Path
    resume_from: Path | None = None
    teacher_checkpoint: Path | None = None
    real_score_checkpoint: Path | None = None
    fake_score_init: Path | None = None

    def kwargs(self) -> dict:
        return {
            "student_init": self.student_init,
            "resume_from": self.resume_from,
            "teacher_checkpoint": self.teacher_checkpoint,
            "real_score_checkpoint": self.real_score_checkpoint,
            "fake_score_init": self.fake_score_init,
        }


def _default_python_bin() -> str:
    python = os.environ.get("PYTHON_BIN", sys.executable)
    if os.path.isfile(python) and os.access(python, os.X_OK):
        return python
    return sys.executable


def _run_method(
    method: str,
    save_root: Path,
    common_env: dict[str, str],
    *,
    student_init: Path | None = None,
    teacher_checkpoint: Path | None = None,
    real_score_checkpoint: Path | None = None,
    fake_score_init: Path | None = None,
    resume_from: Path | None = None,
) -> Path:
    script = (
        Path(__file__).resolve().parents[1]
        / "1shell"
        / "distill"
        / f"train_distill_{method}_4gpu.sh"
    )
    if not script.is_file():
        raise FileNotFoundError(f"Method launcher not found: {script}")

    env = {**os.environ, **common_env}
    for key in _DISTILL_ENV_KEYS:
        env.pop(key, None)
    env["DISTILL_SAVE_ROOT"] = str(save_root)
    if student_init is not None:
        env["DISTILL_STUDENT_INIT"] = str(student_init)
    if teacher_checkpoint is not None:
        env["DISTILL_TEACHER_CHECKPOINT"] = str(teacher_checkpoint)
    if real_score_checkpoint is not None:
        env["DISTILL_REAL_SCORE_CHECKPOINT"] = str(real_score_checkpoint)
    if fake_score_init is not None:
        env["DISTILL_FAKE_SCORE_INIT"] = str(fake_score_init)
    if resume_from is not None:
        env["DISTILL_RESUME_FROM"] = str(resume_from)
    result = subprocess.run([str(script)], env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Distillation method {method} failed with exit code {result.returncode}"
        )

    checkpoint_root = save_root / "checkpoints"
    latest = find_latest_successful_checkpoint(checkpoint_root)
    if latest is None:
        raise RuntimeError(
            f"Distillation method {method} produced no _SUCCESS checkpoint in "
            f"{checkpoint_root}"
        )
    return latest


def _save_pipeline_state(pipeline_root: Path, state: WorkflowState) -> None:
    state_file = pipeline_root / PIPELINE_STATE_FILE
    temp_file = state_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    temp_file.replace(state_file)


def _run_and_record(
    run: MethodRun,
    pipeline_root: Path,
    state: WorkflowState,
    common_env: dict[str, str],
) -> Path:
    if run.name in state.completed:
        return Path(state.completed[run.name])

    state.running_method = run.name
    state.failed_method = None
    _save_pipeline_state(pipeline_root, state)
    try:
        checkpoint = _run_method(
            run.name,
            pipeline_root / run.name,
            common_env,
            **run.kwargs(),
        )
    except Exception:
        state.running_method = None
        state.failed_method = run.name
        _save_pipeline_state(pipeline_root, state)
        raise

    state.completed[run.name] = str(checkpoint)
    state.running_method = None
    state.failed_method = None
    _save_pipeline_state(pipeline_root, state)
    return checkpoint


def run_pipeline(args: argparse.Namespace) -> None:
    if bool(args.resume_from) != bool(args.resume_method):
        raise ValueError("--resume-from and --resume-method must be provided together")

    pipeline_root = Path(args.pipeline_root).resolve()
    pipeline_root.mkdir(parents=True, exist_ok=True)
    state = WorkflowState.load(pipeline_root / PIPELINE_STATE_FILE)

    common_env = {
        "NGPU": str(args.ngpu),
        "MASTER_PORT": str(args.master_port),
        "PYTHON_BIN": _default_python_bin(),
    }
    resume_from = Path(args.resume_from) if args.resume_from else None

    autoregressive_checkpoint = _run_and_record(
        MethodRun(
            name=AUTOREGRESSIVE_TRAINING,
            student_init=Path(args.student_init),
            resume_from=(
                resume_from
                if args.resume_method == AUTOREGRESSIVE_TRAINING
                else None
            ),
        ),
        pipeline_root,
        state,
        common_env,
    )
    consistency_checkpoint = _run_and_record(
        MethodRun(
            name=CONSISTENCY_DISTILLATION,
            student_init=autoregressive_checkpoint,
            teacher_checkpoint=autoregressive_checkpoint,
            resume_from=(
                resume_from
                if args.resume_method == CONSISTENCY_DISTILLATION
                else None
            ),
        ),
        pipeline_root,
        state,
        common_env,
    )
    final_checkpoint = _run_and_record(
        MethodRun(
            name=SELF_GRADIENT_FORCING_DMD,
            student_init=consistency_checkpoint,
            real_score_checkpoint=autoregressive_checkpoint,
            fake_score_init=autoregressive_checkpoint,
            resume_from=(
                resume_from
                if args.resume_method == SELF_GRADIENT_FORCING_DMD
                else None
            ),
        ),
        pipeline_root,
        state,
        common_env,
    )
    print(f"Pipeline complete. Final checkpoint: {final_checkpoint}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the distillation workflow.")
    parser.add_argument("--pipeline-root", required=True, type=str,
                        help="Root directory for the full pipeline run.")
    parser.add_argument("--student-init", required=True, type=str,
                        help="Initial transformer checkpoint for autoregressive training")
    parser.add_argument("--ngpu", default=4, type=int,
                        help="Number of GPUs per training job.")
    parser.add_argument("--master-port", default=29561, type=int,
                        help="Master port for torchrun.")
    parser.add_argument("--resume-from", default=None, type=str,
                        help="Checkpoint path for --resume-method")
    parser.add_argument(
        "--resume-method",
        default=None,
        choices=METHODS,
        help="Training method resumed from --resume-from",
    )
    return parser.parse_args(argv)


def main() -> int:
    try:
        run_pipeline(parse_args())
        return 0
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception("Distillation workflow failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
