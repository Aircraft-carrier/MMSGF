"""Persistent CPU checkpoint-evaluation queue and external worker."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

from wan_va.checkpoint_retention import EVAL_IN_PROGRESS_MARKER


PENDING_SUFFIX = ".pending.json"
RUNNING_SUFFIX = ".running.json"
FAILED_SUFFIX = ".failed.json"
WORKER_READY_MARKER = "_WORKER_READY"
WORKER_LOCK_FILE = "_WORKER_LOCK"
DEFAULT_STOP_MARKER = "_STOP"
_CHECKPOINT_STEP_PATTERN = re.compile(r"^checkpoint_step_(\d+)")


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temp_path.write_text(
            json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _job_path(queue_dir: Path, checkpoint_name: str, suffix: str) -> Path:
    return queue_dir / f"{checkpoint_name}{suffix}"


class CheckpointEvaluationQueue:
    """Publish evaluation jobs without creating a process in a CUDA rank."""

    def __init__(
        self,
        *,
        enabled: bool,
        eval_cfg: Mapping,
        output_root: str | Path,
        queue_dir: str | Path,
    ) -> None:
        self.enabled = bool(enabled)
        self.eval_cfg = dict(eval_cfg)
        self.output_root = Path(output_root)
        self.queue_dir = Path(queue_dir)

    def enqueue(self, checkpoint_dir: str | Path) -> bool:
        if not self.enabled:
            return False

        checkpoint_dir = Path(checkpoint_dir).resolve()
        if not (checkpoint_dir / "_SUCCESS").is_file():
            raise FileNotFoundError(f"Cannot evaluate incomplete checkpoint: {checkpoint_dir}")

        output_dir = (self.output_root / checkpoint_dir.name).resolve()
        marker = checkpoint_dir / EVAL_IN_PROGRESS_MARKER
        if (output_dir / "_SUCCESS").is_file():
            marker.unlink(missing_ok=True)
            return False

        pending_path = _job_path(self.queue_dir, checkpoint_dir.name, PENDING_SUFFIX)
        running_path = _job_path(self.queue_dir, checkpoint_dir.name, RUNNING_SUFFIX)
        if pending_path.is_file() or running_path.is_file():
            marker.touch(exist_ok=True)
            return False

        cpu_cfg = dict(self.eval_cfg)
        cpu_cfg["device"] = "cpu"
        cpu_cfg["masked_attn_backend"] = "dense"
        payload = {
            "checkpoint_root": str(checkpoint_dir),
            "output_dir": str(output_dir),
            "eval_cfg": cpu_cfg,
            "created_at_unix": time.time(),
        }

        marker.touch(exist_ok=True)
        try:
            _write_json_atomic(pending_path, payload)
        except Exception:
            marker.unlink(missing_ok=True)
            raise
        return True


def _state_path(path: Path, old_suffix: str, new_suffix: str) -> Path:
    if not path.name.endswith(old_suffix):
        raise ValueError(f"Expected job path ending in {old_suffix}: {path}")
    return path.with_name(path.name[: -len(old_suffix)] + new_suffix)


def process_pending_job(
    pending_path: str | Path,
    *,
    python_executable: str = sys.executable,
    repo_root: str | Path | None = None,
) -> bool:
    """Run one CPU evaluation job and persist its terminal state."""

    pending_path = Path(pending_path)
    running_path = _state_path(pending_path, PENDING_SUFFIX, RUNNING_SUFFIX)
    failed_path = _state_path(pending_path, PENDING_SUFFIX, FAILED_SUFFIX)
    pending_path.replace(running_path)
    payload = json.loads(running_path.read_text(encoding="utf-8"))
    checkpoint_dir = Path(payload["checkpoint_root"])
    output_dir = Path(payload["output_dir"])
    marker = checkpoint_dir / EVAL_IN_PROGRESS_MARKER
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(python_executable),
        "-m",
        "inference.mot_chunk_infer",
        "--checkpoint-root",
        str(checkpoint_dir),
        "--eval-cfg-json",
        json.dumps(payload["eval_cfg"], sort_keys=True),
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    returncode = None
    error = None
    try:
        with (output_dir / "eval.log").open("ab") as log_handle:
            result = subprocess.run(
                command,
                cwd=str(repo_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                check=False,
            )
        returncode = int(result.returncode)
        if returncode == 0 and not (output_dir / "_SUCCESS").is_file():
            error = "evaluation exited successfully without writing _SUCCESS"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        marker.unlink(missing_ok=True)

    if returncode == 0 and error is None:
        running_path.unlink(missing_ok=True)
        return True

    failure = dict(payload)
    failure["returncode"] = returncode
    failure["error"] = error or f"evaluation process exited with return code {returncode}"
    failure["failed_at_unix"] = time.time()
    _write_json_atomic(failed_path, failure)
    running_path.unlink(missing_ok=True)
    return False


def _job_sort_key(path: Path) -> tuple[int, str]:
    match = _CHECKPOINT_STEP_PATTERN.match(path.name)
    step = int(match.group(1)) if match is not None else sys.maxsize
    return step, path.name


def _recover_running_jobs(queue_dir: Path) -> None:
    for running_path in sorted(queue_dir.glob(f"*{RUNNING_SUFFIX}"), key=_job_sort_key):
        pending_path = _state_path(running_path, RUNNING_SUFFIX, PENDING_SUFFIX)
        if pending_path.exists():
            raise FileExistsError(f"Both pending and running jobs exist: {pending_path}, {running_path}")
        running_path.replace(pending_path)


def run_evaluation_worker(
    queue_dir: str | Path,
    *,
    stop_file: str | Path | None = None,
    poll_interval: float = 1.0,
) -> None:
    """Process queued evaluations sequentially until stopped and drained."""

    queue_dir = Path(queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    stop_file = Path(stop_file) if stop_file is not None else queue_dir / DEFAULT_STOP_MARKER
    ready_path = queue_dir / WORKER_READY_MARKER
    lock_path = queue_dir / WORKER_LOCK_FILE
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another CPU evaluation worker owns {queue_dir}") from exc

        _recover_running_jobs(queue_dir)
        ready_path.touch(exist_ok=True)
        print(f"CPU evaluation worker ready: queue_dir={queue_dir}", flush=True)
        try:
            while True:
                pending_jobs = sorted(queue_dir.glob(f"*{PENDING_SUFFIX}"), key=_job_sort_key)
                if pending_jobs:
                    pending_path = pending_jobs[0]
                    print(f"Starting CPU evaluation job: {pending_path.name}", flush=True)
                    succeeded = process_pending_job(pending_path)
                    status = "succeeded" if succeeded else "failed"
                    print(f"CPU evaluation job {status}: {pending_path.name}", flush=True)
                    continue
                if stop_file.is_file():
                    print("CPU evaluation queue drained; worker exiting", flush=True)
                    return
                time.sleep(float(poll_interval))
        finally:
            ready_path.unlink(missing_ok=True)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the external MOT CPU evaluation worker.")
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--stop-file", default=None)
    args = parser.parse_args()
    run_evaluation_worker(args.queue_dir, stop_file=args.stop_file)


if __name__ == "__main__":
    main()
