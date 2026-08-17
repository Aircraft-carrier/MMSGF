import json

import numpy as np
import pytest
import torch

from distillation.eval.protocol import CAMERA_KEYS, OnlineObservation
from inference.eval.pipeline import (
    ACTION_TARGET_FRAME_IDS,
    TARGET_FRAME_IDS,
    BidirectionalMOTInferencePipeline,
    _load_spec,
    load_pipeline,
)


def _stats():
    return {"q01": [-1.0] * 20, "q99": [1.0] * 20}


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


class _Codec:
    def __init__(self):
        self.decoded_shape = None

    def encode_history(self, _rgb):
        return torch.zeros(1, 1, 4, 3, 1, 1)

    def encode_anchor(self, _rgb):
        return torch.ones(1, 1, 1, 3, 1, 1)

    def decode_one(self, _latent):
        return ["video"]

    def decode_video(self, latent):
        self.decoded_shape = tuple(latent.shape)
        return {
            "fps": 10,
            "camera_keys": list(CAMERA_KEYS),
            "frames": [["video"] * 3],
        }


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls = []
        self.video_inputs = []
        self.fail_action = False

    def commit_video(self, latents, *, frame_ids, cache, **_kwargs):
        self.calls.append(
            ("commit_video", tuple(frame_ids), id(cache), tuple(latents.shape))
        )

    def commit_action(self, actions, *, frame_ids, cache, **_kwargs):
        self.calls.append(
            ("commit_action", tuple(frame_ids), id(cache), tuple(actions.shape))
        )

    def predict_video(self, sample, *, frame_ids, cache, **_kwargs):
        self.calls.append(
            ("predict_video", tuple(frame_ids), id(cache), tuple(sample.shape))
        )
        self.video_inputs.append(sample.detach().clone())
        return torch.zeros_like(sample)

    def predict_action(self, sample, *, frame_ids, cache, **_kwargs):
        if self.fail_action:
            raise RuntimeError("action failed")
        self.calls.append(("predict_action", tuple(frame_ids), id(cache), tuple(sample.shape)))
        return torch.zeros_like(sample)


def _pipeline(model=None, *, video_num_steps=1, execution_action_count=48):
    return BidirectionalMOTInferencePipeline(
        model=model or _Model(),
        codec=_Codec(),
        text_embedder=lambda _text: torch.zeros(1, 1, 1),
        empty_text_emb=torch.zeros(1, 1, 1),
        norm_stats_by_task={"task-clean": _stats()},
        spec=_load_spec(
            {
                "action_chunk_size": 48,
                "video_downsample_ratio": 4,
                "vae_temporal_factor": 4,
            }
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
        video_num_steps=video_num_steps,
        action_num_steps=1,
        guidance_scale=1.0,
        video_snr_shift=1.0,
        action_snr_shift=1.0,
        execution_action_count=execution_action_count,
    )


def test_online_batch_has_fixed_eight_frame_layout_and_masks() -> None:
    pipeline = _pipeline()
    pipeline.reset(task_name="task", instruction="do it", seed=0)
    candidate = pipeline.builder.clone()
    candidate.append([_observation(0)], [])
    window = candidate.build()
    batch = pipeline._assemble_batch(
        window,
        pipeline.codec.encode_history(window.history_rgb),
        pipeline.codec.encode_anchor(window.anchor_rgb),
    )

    assert batch["latents"].shape == (1, 1, 8, 3, 1, 1)
    assert batch["actions"].shape == (1, 20, 8, 16, 1)
    assert batch["video_latent_valid_mask"].tolist() == [
        [False, False, False, False, True, True, True, True]
    ]
    assert not batch["action_valid_mask"][:, :, :5].any()
    assert batch["action_valid_mask"][:, :, 5:8].all()
    assert batch["action_loss_mask"][:, :, 5:8].all()
    torch.testing.assert_close(
        batch["latents"][:, :, 4], torch.ones_like(batch["latents"][:, :, 4])
    )
    torch.testing.assert_close(
        batch["latents"][:, :, 5:8], torch.zeros_like(batch["latents"][:, :, 5:8])
    )


def test_infer_caches_history_then_predicts_whole_target_chunks() -> None:
    pipeline = _pipeline()
    pipeline.reset(task_name="task", instruction="do it", seed=7)
    response = pipeline.infer(
        observations=[_observation(0)],
        executed_actions=[],
        request_id=0,
        return_video=False,
    )

    assert len(response["actions"]) == 48
    assert all(len(action) == 16 for action in response["actions"])
    assert response["predicted_video"] is None
    calls = pipeline.model.calls
    assert [call[:2] for call in calls] == [
        ("commit_video", (0, 1, 2, 3)),
        ("commit_action", (0, 1, 2, 3)),
        ("predict_video", TARGET_FRAME_IDS),
        ("commit_video", TARGET_FRAME_IDS),
        ("predict_action", ACTION_TARGET_FRAME_IDS),
    ]
    assert len({call[2] for call in calls}) == 1
    assert calls[2][3][2] == 4
    assert calls[4][3][2] == 3


def test_infer_can_execute_only_the_first_action_frame() -> None:
    pipeline = _pipeline(execution_action_count=16)
    pipeline.reset(task_name="task", instruction="do it", seed=7)
    response = pipeline.infer(
        observations=[_observation(0)],
        executed_actions=[],
        request_id=0,
        return_video=False,
    )

    assert len(response["actions"]) == 16
    assert all(len(action) == 16 for action in response["actions"])


def test_infer_returns_the_full_generated_video_chunk_when_requested() -> None:
    pipeline = _pipeline()
    pipeline.reset(task_name="task", instruction="do it", seed=7)
    response = pipeline.infer(
        observations=[_observation(0)],
        executed_actions=[],
        request_id=0,
        return_video=True,
    )

    assert response["predicted_video"]["fps"] == 10
    assert pipeline.codec.decoded_shape == (1, 1, 4, 3, 1, 1)


def test_seed_and_request_id_control_target_noise() -> None:
    inputs = []
    for request_id in (0, 0, 1):
        pipeline = _pipeline()
        pipeline.reset(task_name="task", instruction="do it", seed=11)
        pipeline.infer(
            observations=[_observation(0)],
            executed_actions=[],
            request_id=request_id,
            return_video=False,
        )
        inputs.append(pipeline.model.video_inputs[0][:, :, 1:])
    torch.testing.assert_close(inputs[0], inputs[1])
    assert not torch.equal(inputs[0], inputs[2])


def test_every_video_forward_restores_the_clean_anchor() -> None:
    pipeline = _pipeline(video_num_steps=2)
    pipeline.reset(task_name="task", instruction="do it", seed=0)
    pipeline.infer(
        observations=[_observation(0)],
        executed_actions=[],
        request_id=0,
        return_video=False,
    )
    assert len(pipeline.model.video_inputs) == 2
    for target in pipeline.model.video_inputs:
        torch.testing.assert_close(target[:, :, :1], torch.ones_like(target[:, :, :1]))


def test_failed_request_does_not_commit_online_history() -> None:
    model = _Model()
    model.fail_action = True
    pipeline = _pipeline(model)
    pipeline.reset(task_name="task", instruction="do it", seed=0)
    with pytest.raises(RuntimeError, match="action failed"):
        pipeline.infer(
            observations=[_observation(0)],
            executed_actions=[],
            request_id=0,
            return_video=False,
        )
    assert pipeline.builder.last_step == -1


def test_reset_requires_one_normalization_prefix_match() -> None:
    pipeline = _pipeline()
    with pytest.raises(KeyError, match="matched 0"):
        pipeline.reset(task_name="missing", instruction="do it", seed=0)
    pipeline.norm_stats_by_task["task-random"] = _stats()
    with pytest.raises(KeyError, match="matched 2"):
        pipeline.reset(task_name="task", instruction="do it", seed=0)


def test_load_pipeline_rejects_autoregressive_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps({"model_architecture": "autoregressive_va_mot_v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="va_mot_v1 checkpoint"):
        load_pipeline(
            checkpoint_root=checkpoint,
            dataset_root=tmp_path,
            model_root=tmp_path,
            device="cpu",
        )
