"""Checkpoint tests that need no model weights, GPU, or real dataset."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillation.checkpoint import DistillationCheckpointIO
from distillation.configs import (
    CONSISTENCY_DISTILLATION,
    SELF_GRADIENT_FORCING_DMD,
)


class _MetadataTrainer:
    method = CONSISTENCY_DISTILLATION
    step = 12
    optimizer_step = 10
    config = SimpleNamespace(
        distill=SimpleNamespace(
            generation_shape={
                "profile_name": "segmented_history_strict_geometry_v1",
                "order_mode": "segmented",
                "history_frames": 4,
                "chunk_size": 4,
                "window_size": 16,
            },
        )
    )

    @staticmethod
    def _write_checkpoint_metadata(path: Path, *, has_full_state: bool) -> None:
        (path / "checkpoint_metadata.json").write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "checkpoint_type": "mot_training",
                    "vggto_attention_topology": "synchronized_multiview_v1",
                    "optimization_composition": "va",
                    "has_full_state": has_full_state,
                }
            ),
            encoding="utf-8",
        )


def test_distillation_metadata_remains_valid_next_stage_initialization(tmp_path: Path) -> None:
    DistillationCheckpointIO()._write_metadata(tmp_path, _MetadataTrainer())

    metadata = json.loads(
        (tmp_path / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    # 这三个字段正是 MOT 基础 loader 在读取下一阶段 transformer export 前
    # 校验的兼容契约。这里不导入完整 MOTTrainer，因其会连带要求真实数据栈
    # 的 pyarrow 等依赖，违背本测试“无实际训练环境”的目的。
    assert metadata["format_version"] == 2
    assert metadata["checkpoint_type"] == "mot_training"
    assert metadata["vggto_attention_topology"] == "synchronized_multiview_v1"
    assert metadata["optimization_composition"] == "va"
    assert metadata["distill_method"] == CONSISTENCY_DISTILLATION
    assert metadata["exported_model"] == "ema_student"
    assert metadata["has_full_state"] is True
    assert metadata["generation_profile"] == {
        "profile_name": "segmented_history_strict_geometry_v1",
        "profile_version": 1,
        "order_mode": "segmented",
        "history_frames": 4,
        "chunk_size": 4,
        "window_size": 16,
        "geometry_relation": "strict_frame_history",
        "x_to_g_relation": "strict_order",
    }


@pytest.mark.parametrize(
    ("method", "expected_keys"),
    [
        (
            CONSISTENCY_DISTILLATION,
            {"model", "optimizer", "ema_student"},
        ),
        (
            SELF_GRADIENT_FORCING_DMD,
            {"model", "optimizer", "fake_score", "fake_score_optimizer"},
        ),
    ],
)
def test_dcp_resume_bundle_contains_every_mutable_training_role(
    monkeypatch,
    method: str,
    expected_keys: set[str],
) -> None:
    student = object()
    student_optimizer = object()
    ema_student = object()
    fake_score = object()
    fake_optimizer = object()

    def fake_get_state_dict(model, optimizer, *, options):
        return {"owner": model}, {"owner": optimizer}

    monkeypatch.setattr(
        "distillation.checkpoint.get_state_dict",
        fake_get_state_dict,
    )
    monkeypatch.setattr(
        "distillation.checkpoint.get_model_state_dict",
        lambda model, *, options: {"owner": model},
    )
    trainer = SimpleNamespace(
        method=method,
        transformer=student,
        optimizer=student_optimizer,
        method_model=SimpleNamespace(
            ema_student=ema_student,
            fake_score=fake_score,
        ),
        fake_score_optimizer=fake_optimizer,
    )

    state = DistillationCheckpointIO()._dcp_state_dict(trainer)

    assert set(state) == expected_keys
    assert state["model"]["owner"] is student
    assert state["optimizer"]["owner"] is student_optimizer
    if method == CONSISTENCY_DISTILLATION:
        assert state["ema_student"]["owner"] is ema_student
    else:
        assert state["fake_score"]["owner"] is fake_score
        assert state["fake_score_optimizer"]["owner"] is fake_optimizer
