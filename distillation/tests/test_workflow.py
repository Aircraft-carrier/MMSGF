"""Smoke tests for the distillation workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from distillation.workflow import run_pipeline


def _make_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "_SUCCESS").write_text("")
    return path


def test_pipeline_runs_all_methods(tmp_path: Path):
    pipeline_root = tmp_path / "pipeline"
    student_init = tmp_path / "init"
    student_init.mkdir()

    autoregressive = tmp_path / "autoregressive"
    consistency = tmp_path / "consistency"
    sgf_dmd = tmp_path / "sgf_dmd"

    with patch("distillation.workflow._run_method") as mock_run:
        mock_run.side_effect = [
            _make_checkpoint(autoregressive),
            _make_checkpoint(consistency),
            _make_checkpoint(sgf_dmd),
        ]
        args = argparse.Namespace(
            pipeline_root=str(pipeline_root),
            student_init=str(student_init),
            ngpu=1,
            master_port=12345,
            generation_profile="1/16",
            resume_from=None,
            resume_method=None,
        )
        run_pipeline(args)

    assert mock_run.call_count == 3
    autoregressive_call, consistency_call, sgf_call = mock_run.call_args_list
    assert autoregressive_call.args[:2] == (
        "autoregressive_training",
        pipeline_root / "autoregressive_training",
    )
    assert autoregressive_call.kwargs["student_init"] == student_init
    assert consistency_call.kwargs["student_init"] == autoregressive
    assert consistency_call.kwargs["teacher_checkpoint"] == autoregressive
    assert sgf_call.kwargs["student_init"] == consistency
    assert sgf_call.kwargs["real_score_checkpoint"] == autoregressive
    assert sgf_call.kwargs["fake_score_init"] == autoregressive
    assert (pipeline_root / "pipeline_state.json").is_file()


def test_pipeline_stops_on_failure(tmp_path: Path):
    pipeline_root = tmp_path / "pipeline"
    student_init = tmp_path / "init"
    student_init.mkdir()

    with patch("distillation.workflow._run_method") as mock_run:
        mock_run.side_effect = RuntimeError("autoregressive training failed")
        args = argparse.Namespace(
            pipeline_root=str(pipeline_root),
            student_init=str(student_init),
            ngpu=1,
            master_port=12345,
            generation_profile="1/16",
            resume_from=None,
            resume_method=None,
        )
        with pytest.raises(RuntimeError, match="autoregressive training failed"):
            run_pipeline(args)

    state = json.loads((pipeline_root / "pipeline_state.json").read_text())
    assert state["failed_method"] == "autoregressive_training"


def test_pipeline_resumes_from_completed(tmp_path: Path):
    pipeline_root = tmp_path / "pipeline"
    pipeline_root.mkdir(parents=True, exist_ok=True)
    student_init = tmp_path / "init"
    student_init.mkdir()

    autoregressive = _make_checkpoint(tmp_path / "autoregressive")
    consistency = tmp_path / "consistency"
    sgf_dmd = tmp_path / "sgf_dmd"

    state = {
        "format_version": 2,
        "completed": {"autoregressive_training": str(autoregressive)},
        "running_method": None,
        "failed_method": None,
    }
    (pipeline_root / "pipeline_state.json").write_text(json.dumps(state))

    with patch("distillation.workflow._run_method") as mock_run:
        mock_run.side_effect = [
            _make_checkpoint(consistency),
            _make_checkpoint(sgf_dmd),
        ]
        args = argparse.Namespace(
            pipeline_root=str(pipeline_root),
            student_init=str(student_init),
            ngpu=1,
            master_port=12345,
            generation_profile="1/16",
            resume_from=None,
            resume_method=None,
        )
        run_pipeline(args)

    assert mock_run.call_count == 2


def test_pipeline_passes_autoregressive_resume_checkpoint(tmp_path: Path):
    pipeline_root = tmp_path / "pipeline"
    student_init = tmp_path / "init"
    resume_from = _make_checkpoint(tmp_path / "resume")
    student_init.mkdir()

    with patch("distillation.workflow._run_method") as mock_run:
        mock_run.side_effect = [
            _make_checkpoint(tmp_path / "autoregressive"),
            _make_checkpoint(tmp_path / "consistency"),
            _make_checkpoint(tmp_path / "sgf_dmd"),
        ]
        run_pipeline(argparse.Namespace(
            pipeline_root=str(pipeline_root),
            student_init=str(student_init),
            ngpu=1,
            master_port=12345,
            generation_profile="1/16",
            resume_from=str(resume_from),
            resume_method="autoregressive_training",
        ))

    assert mock_run.call_args_list[0].kwargs["resume_from"] == resume_from
