import numpy as np
import torch

from distillation.eval.infer_pipeline import (
    AutoregressiveMOTInferencePipeline,
    OnlineMOTWindowBuilder,
)
from distillation.eval.protocol import CAMERA_KEYS, OnlineObservation


def _observation(step: int) -> OnlineObservation:
    image = np.full((4, 5, 3), step, dtype=np.uint8)
    return OnlineObservation(
        step=step,
        images={key: image.copy() for key in CAMERA_KEYS},
        state=np.array(
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            dtype=np.float32,
        ),
    )


def _stats():
    return {"q01": [-1.0] * 20, "q99": [1.0] * 20}


def test_online_window_matches_early_episode_video_and_action_layout() -> None:
    builder = OnlineMOTWindowBuilder(_stats())
    builder.append([_observation(0)], [])
    first = builder.build()
    assert first.video_valid.tolist() == [[False, False, False, False]]
    assert not bool(first.action_valid.any())

    observations = [_observation(step) for step in range(1, 17)]
    actions = [observation.state.copy() for observation in observations]
    builder.append(observations, actions)
    window = builder.build()

    assert window.video_valid.tolist() == [[False, False, False, True]]
    assert window.history_rgb.shape == (13, 3, 3, 4, 5)
    assert window.anchor_rgb.shape == (1, 3, 3, 4, 5)
    assert window.history_actions.shape == (1, 20, 4, 16, 1)
    assert window.action_valid[:, :, 3].all()
    assert not window.action_valid[:, :, :3].any()
    torch.testing.assert_close(window.history_actions[:, :, :3], torch.zeros_like(window.history_actions[:, :, :3]))


def test_online_window_clone_only_commits_after_success() -> None:
    builder = OnlineMOTWindowBuilder(_stats())
    builder.append([_observation(0)], [])
    candidate = builder.clone()
    candidate.append([_observation(1)], [_observation(1).state])
    assert builder.last_step == 0
    assert candidate.last_step == 1


def test_online_window_rejects_non_contiguous_steps() -> None:
    builder = OnlineMOTWindowBuilder(_stats())
    builder.append([_observation(0)], [])
    try:
        builder.append([_observation(2)], [_observation(2).state])
    except ValueError as exc:
        assert "expected observation step 1" in str(exc)
    else:
        raise AssertionError("non-contiguous steps were accepted")


class _Codec:
    def encode_history(self, rgb):
        return torch.zeros(1, 1, 4, 3, 1, 1)

    def encode_anchor(self, rgb):
        return torch.zeros(1, 1, 1, 3, 1, 1)

    def decode_one(self, latent):
        return [str(int(latent[0, 0, 0, 0, 0, 0]))]


class _Model:
    def __init__(self):
        self.commits = []

    def commit_video(self, latent, *, frame_ids, **kwargs):
        self.commits.append(("video", tuple(frame_ids)))

    def commit_action(self, action, *, frame_ids, **kwargs):
        self.commits.append(("action", tuple(frame_ids)))


class _Pipeline(AutoregressiveMOTInferencePipeline):
    def _euler_video(self, cache, text_emb, generator, frame_id):
        return torch.full(self._target_video_shape, frame_id, dtype=self.dtype)

    def _euler_action(self, cache, text_emb, generator, frame_id):
        return torch.zeros(1, 20, 1, 16, 1, dtype=self.dtype)


def _pipeline(prediction_chunks=1):
    return _Pipeline(
        model=_Model(),
        codec=_Codec(),
        text_embedder=lambda instruction: torch.zeros(1, 1, 1),
        norm_stats_by_task={"task": _stats()},
        device=torch.device("cpu"),
        dtype=torch.float32,
        video_num_steps=1,
        action_num_steps=1,
        video_snr_shift=1.0,
        action_snr_shift=1.0,
        prediction_chunks=prediction_chunks,
    )


def test_infer_commits_each_predicted_chunk_before_the_next() -> None:
    pipeline = _pipeline(prediction_chunks=2)
    pipeline.reset(task_name="task", instruction="do it", seed=0)

    response = pipeline.infer(
        observations=[_observation(0)],
        executed_actions=[],
        request_id=0,
        return_video=True,
    )

    assert pipeline.model.commits[-4:] == [
        ("video", (5,)),
        ("action", (5,)),
        ("video", (6,)),
        ("action", (6,)),
    ]
    assert ("action", (4,)) not in pipeline.model.commits
    assert len(response["actions"]) == 32
    assert response["predicted_video"] == [["5"], ["6"]]


def test_prediction_chunks_defaults_to_one_and_rejects_values_outside_one_to_three() -> None:
    pipeline = _pipeline()
    pipeline.reset(task_name="task", instruction="do it", seed=0)
    response = pipeline.infer(
        observations=[_observation(0)],
        executed_actions=[],
        request_id=0,
        return_video=True,
    )
    assert len(response["actions"]) == 16
    assert response["predicted_video"] == ["5"]

    for prediction_chunks in (0, 4):
        try:
            _pipeline(prediction_chunks)
        except ValueError as exc:
            assert "between 1 and 3" in str(exc)
        else:
            raise AssertionError(f"prediction_chunks={prediction_chunks} was accepted")
