from argparse import Namespace
from distillation import workflow
from distillation.configs import (
    AUTOREGRESSIVE_TRAINING,
    CONSISTENCY_DISTILLATION,
    SELF_GRADIENT_FORCING_DMD,
)


def test_workflow_connects_all_distillation_checkpoint_roles(
    tmp_path,
    monkeypatch,
) -> None:
    base_checkpoint = tmp_path / "base_bidir"
    ar_checkpoint = tmp_path / "ar"
    consistency_checkpoint = tmp_path / "consistency"
    final_checkpoint = tmp_path / "sgf"
    outputs = iter([ar_checkpoint, consistency_checkpoint, final_checkpoint])
    runs = []

    def record(run, pipeline_root, state, common_env):
        del pipeline_root, state, common_env
        runs.append(run)
        return next(outputs)

    monkeypatch.setattr(workflow, "_run_and_record", record)
    workflow.run_pipeline(
        Namespace(
            pipeline_root=str(tmp_path / "pipeline"),
            student_init=str(base_checkpoint),
            ngpu=4,
            master_port=29561,
            resume_from=None,
            resume_method=None,
        )
    )

    assert [run.name for run in runs] == [
        AUTOREGRESSIVE_TRAINING,
        CONSISTENCY_DISTILLATION,
        SELF_GRADIENT_FORCING_DMD,
    ]
    assert runs[0].student_init == base_checkpoint
    assert runs[1].student_init == ar_checkpoint
    assert runs[1].teacher_checkpoint == ar_checkpoint
    assert runs[2].student_init == consistency_checkpoint
    assert runs[2].real_score_checkpoint == base_checkpoint
    assert runs[2].fake_score_init == base_checkpoint
