from __future__ import annotations

from types import SimpleNamespace

import torch

from distillation.rollout import autoregressive_rollout
from distillation.self_rollout.engine import self_rollout
from distillation.self_rollout.geometry_cache import EncodedGeometryFrame
from distillation.self_rollout.provider import GroundTruthStep, OfflineGroundTruthProvider
from distillation.self_rollout.recorder import SelfRolloutRecorder


class _Scheduler:
    def __init__(self):
        self.timesteps = torch.tensor([1.0, 0.5])

    def step(self, model_output, timestep, sample):
        del timestep, sample
        return model_output


class _FakeMOTAdapter:
    def __init__(self):
        self.events = []

    def commit_video(self, latents, *, frame_ids, source, **kwargs):
        del kwargs
        self.events.append(("video", tuple(frame_ids), source.name, latents.detach().clone()))

    def commit_action(self, actions, *, frame_ids, source, **kwargs):
        del kwargs
        self.events.append(("action", tuple(frame_ids), source.name, actions.detach().clone()))

    def predict_video(self, sample, *, frame_id, **kwargs):
        del kwargs
        return torch.full_like(sample, float(frame_id))

    def predict_action(self, sample, *, frame_id, **kwargs):
        del kwargs
        return torch.full_like(sample, float(frame_id + 10))


class _NoiseRecordingMOTAdapter(_FakeMOTAdapter):
    def __init__(self):
        super().__init__()
        self.video_noise = {}

    def predict_video(self, sample, *, frame_id, **kwargs):
        self.video_noise.setdefault(frame_id, sample.detach().clone())
        return super().predict_video(sample, frame_id=frame_id, **kwargs)


class _ContinuationAwareMOTAdapter(_FakeMOTAdapter):
    def predict_video(self, sample, *, frame_id, state, **kwargs):
        del kwargs
        if frame_id == 3:
            return torch.full_like(sample, 103.0)
        previous = state.frame(frame_id - 1).video_latent
        if previous is None:
            raise RuntimeError("previous continuation video is missing")
        return previous + 1.0


class _FakeGeometryAdapter:
    def __init__(self):
        self.events = []

    def encode_and_commit(self, rgb, *, frame_id, state, source, **kwargs):
        del kwargs
        self.events.append((frame_id, source.name, rgb.detach().clone()))
        encoded = EncodedGeometryFrame(
            frame_id=frame_id,
            rgb=rgb,
            final_tokens=torch.empty(0),
            patch_hw=(1, 1),
            image_hw=(1, 1),
            patch_token_start=1,
            cached_outputs=[],
            layer_registers={},
        )
        state.geometry_cache.frames[frame_id] = encoded
        return encoded


def _batch(frames: int = 6) -> dict:
    return {
        "latents": torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1, 1),
        "actions": torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1),
        "geometry_rgb": torch.arange(frames, dtype=torch.float32).reshape(
            1, frames, 1, 1, 1, 1, 1
        ).expand(-1, -1, 4, -1, 3, -1, -1).clone(),
        "geometry_group_valid_mask": torch.ones(1, frames, 4, dtype=torch.bool),
        "video_latent_valid_mask": torch.ones(1, frames, dtype=torch.bool),
        "action_valid_mask": torch.ones(1, 1, frames, 1, 1, dtype=torch.bool),
        "action_loss_mask": torch.ones(1, 1, frames, 1, 1, dtype=torch.bool),
        "stream_ids": torch.zeros(1, 1, dtype=torch.long),
        "text_emb": torch.ones(1, 1, 1),
    }


def _decode(latents: torch.Tensor) -> torch.Tensor:
    batch, _channels, frames, views, _height, _width = latents.shape
    values = [latents[:, 0, 0, :, 0, 0][:, None, :, None, None, None]]
    for frame in range(1, frames):
        value = latents[:, 0, frame, :, 0, 0][:, None, :, None, None, None]
        values.append(value.expand(-1, 4, -1, 3, 1, 1))
    first = values[0].expand(-1, -1, -1, 3, 1, 1)
    return torch.cat([first, *values[1:]], dim=1)


def _config():
    return SimpleNamespace(
        distill=SimpleNamespace(
            generation_shape={"order_mode": "segmented", "chunk_size": 2, "window_size": 16}
        ),
        vae_temporal_factor=4,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        seed=3,
    )


def _spec():
    return SimpleNamespace(
        history_latent_frames=2,
        latent_frames_per_action_chunk_per_view=3,
    )


def test_self_rollout_prefills_anchor_then_runs_latent_geometry_action() -> None:
    mot = _FakeMOTAdapter()
    geometry = _FakeGeometryAdapter()
    result = self_rollout(
        _batch(),
        transformer=object(),
        config=_config(),
        spec=_spec(),
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        decode_latents_to_rgb_views=_decode,
        video_num_steps=2,
        action_num_steps=2,
        rollout_frames=2,
        mot_adapter=mot,
        geometry_adapter=geometry,
        schedulers=SimpleNamespace(video=_Scheduler(), action=_Scheduler()),
    )

    assert [(kind, ids, source) for kind, ids, source, _ in mot.events] == [
        ("video", (0, 1), "HISTORY"),
        ("action", (0, 1), "HISTORY"),
        ("video", (2,), "ANCHOR"),
        ("action", (2,), "ANCHOR"),
        ("video", (3,), "PREDICTED"),
        ("action", (3,), "PREDICTED"),
        ("video", (4,), "PREDICTED"),
        ("action", (4,), "PREDICTED"),
    ]
    assert [event[:2] for event in geometry.events] == [
        (0, "HISTORY"),
        (1, "HISTORY"),
        (2, "ANCHOR"),
        (3, "PREDICTED"),
        (4, "PREDICTED"),
    ]
    torch.testing.assert_close(result.pred_latents[:, :, 3], torch.full((1, 1, 1, 1, 1), 3.0))
    torch.testing.assert_close(result.pred_latents[:, :, 4], torch.full((1, 1, 1, 1, 1), 4.0))
    torch.testing.assert_close(result.pred_actions[:, :, 3], torch.full((1, 1, 1, 1), 13.0))
    assert result.diagnostics["profile"] == "segmented_history_strict_geometry_v1"


def test_self_rollout_records_scheduler_inputs_without_changing_predictions() -> None:
    recorder = SelfRolloutRecorder(video_step=1, action_step=1)
    result = self_rollout(
        _batch(),
        transformer=object(),
        config=_config(),
        spec=_spec(),
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        decode_latents_to_rgb_views=_decode,
        video_num_steps=2,
        action_num_steps=2,
        rollout_frames=2,
        mot_adapter=_FakeMOTAdapter(),
        geometry_adapter=_FakeGeometryAdapter(),
        schedulers=SimpleNamespace(video=_Scheduler(), action=_Scheduler()),
        recorder=recorder,
    )

    assert set(recorder.video) == {3, 4}
    assert set(recorder.action) == {3, 4}
    torch.testing.assert_close(
        recorder.video[3].sample,
        torch.full((1, 1, 1, 1, 1, 1), 3.0),
    )
    torch.testing.assert_close(
        recorder.action[3].sample,
        torch.full((1, 1, 1, 1, 1), 13.0),
    )
    assert recorder.video[3].timestep.item() == 0.5
    assert recorder.action[3].timestep.item() == 0.5
    torch.testing.assert_close(
        result.pred_latents[:, :, 3],
        torch.full((1, 1, 1, 1, 1), 3.0),
    )
    torch.testing.assert_close(
        result.pred_actions[:, :, 3],
        torch.full((1, 1, 1, 1), 13.0),
    )


def test_gt_replacement_changes_continuation_source_but_preserves_predictions() -> None:
    batch = _batch()
    result = self_rollout(
        batch,
        transformer=object(),
        config=_config(),
        spec=_spec(),
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        decode_latents_to_rgb_views=_decode,
        video_num_steps=2,
        action_num_steps=2,
        rollout_frames=1,
        ground_truth_provider=OfflineGroundTruthProvider(batch),
        mot_adapter=_FakeMOTAdapter(),
        geometry_adapter=_FakeGeometryAdapter(),
        schedulers=SimpleNamespace(video=_Scheduler(), action=_Scheduler()),
    )

    # Artifact prediction is retained even though continuation was replaced by GT.
    torch.testing.assert_close(result.pred_latents[:, :, 3], torch.full((1, 1, 1, 1, 1), 3.0))
    torch.testing.assert_close(result.pred_actions[:, :, 3], torch.full((1, 1, 1, 1), 13.0))
    assert result.diagnostics["sources"][3] == {
        "video": "ground_truth",
        "geometry": "ground_truth",
        "action": "ground_truth",
    }
    assert result.diagnostics["replacements"][0]["frame_id"] == 3
    assert result.diagnostics["versions"][3] == {
        "video": 2,
        "geometry": 2,
        "action": 2,
    }


def test_action_only_gt_replacement_keeps_predicted_video_and_geometry() -> None:
    batch = _batch()
    result = self_rollout(
        batch,
        transformer=object(),
        config=_config(),
        spec=_spec(),
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        decode_latents_to_rgb_views=_decode,
        video_num_steps=2,
        action_num_steps=2,
        rollout_frames=1,
        ground_truth_provider=OfflineGroundTruthProvider(
            batch,
            replace=frozenset({"action"}),
        ),
        mot_adapter=_FakeMOTAdapter(),
        geometry_adapter=_FakeGeometryAdapter(),
        schedulers=SimpleNamespace(video=_Scheduler(), action=_Scheduler()),
    )

    assert result.diagnostics["sources"][3] == {
        "video": "predicted",
        "geometry": "predicted",
        "action": "ground_truth",
    }
    torch.testing.assert_close(
        result.pred_actions[:, :, 3],
        torch.full((1, 1, 1, 1), 13.0),
    )


def test_geometry_only_gt_replacement_recomputes_action_from_phase_checkpoint() -> None:
    batch = _batch()
    result = self_rollout(
        batch,
        transformer=object(),
        config=_config(),
        spec=_spec(),
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        decode_latents_to_rgb_views=_decode,
        video_num_steps=2,
        action_num_steps=2,
        rollout_frames=1,
        ground_truth_provider=OfflineGroundTruthProvider(
            batch,
            replace=frozenset({"geometry"}),
        ),
        replacement_policy="recompute_predicted",
        mot_adapter=_FakeMOTAdapter(),
        geometry_adapter=_FakeGeometryAdapter(),
        schedulers=SimpleNamespace(video=_Scheduler(), action=_Scheduler()),
    )

    assert result.diagnostics["sources"][3] == {
        "video": "predicted",
        "geometry": "ground_truth",
        "action": "predicted",
    }
    assert result.diagnostics["versions"][3] == {
        "video": 1,
        "geometry": 2,
        "action": 2,
    }


def test_replacement_continuation_reads_gt_semantic_state() -> None:
    batch = _batch()
    result = self_rollout(
        batch,
        transformer=object(),
        config=_config(),
        spec=_spec(),
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        decode_latents_to_rgb_views=_decode,
        video_num_steps=2,
        action_num_steps=2,
        rollout_frames=2,
        ground_truth_provider=OfflineGroundTruthProvider(batch),
        mot_adapter=_ContinuationAwareMOTAdapter(),
        geometry_adapter=_FakeGeometryAdapter(),
        schedulers=SimpleNamespace(video=_Scheduler(), action=_Scheduler()),
    )

    # Frame 3 predicts 103 but is replaced by GT=3 for continuation. Frame 4
    # therefore predicts 4; reading the stale prediction would have produced 104.
    torch.testing.assert_close(
        result.pred_latents[:, :, 3],
        torch.full((1, 1, 1, 1, 1), 103.0),
    )
    torch.testing.assert_close(
        result.pred_latents[:, :, 4],
        torch.full((1, 1, 1, 1, 1), 4.0),
    )


def test_action_only_replacement_preserves_next_video_noise_sequence() -> None:
    batch = _batch()
    baseline = _NoiseRecordingMOTAdapter()
    replaced = _NoiseRecordingMOTAdapter()
    common = dict(
        batch=batch,
        transformer=object(),
        config=_config(),
        spec=_spec(),
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        decode_latents_to_rgb_views=_decode,
        video_num_steps=2,
        action_num_steps=2,
        rollout_frames=2,
        geometry_adapter=_FakeGeometryAdapter(),
        schedulers=SimpleNamespace(video=_Scheduler(), action=_Scheduler()),
    )
    self_rollout(**common, mot_adapter=baseline)
    self_rollout(
        **common,
        mot_adapter=replaced,
        ground_truth_provider=OfflineGroundTruthProvider(
            batch,
            replace=frozenset({"action"}),
        ),
    )

    torch.testing.assert_close(replaced.video_noise[4], baseline.video_noise[4])


class _VideoActionOnlyProvider:
    def maybe_get(self, *, frame_id, predicted_action, state):
        del predicted_action, state
        return GroundTruthStep(
            frame_id=frame_id,
            video_latent=torch.zeros(1, 1, 1, 1, 1, 1),
            action=torch.zeros(1, 1, 1, 1, 1),
        )


def test_require_ground_truth_rejects_missing_downstream_geometry() -> None:
    try:
        self_rollout(
            _batch(),
            transformer=object(),
            config=_config(),
            spec=_spec(),
            device=torch.device("cpu"),
            empty_text_emb=torch.zeros(1, 1, 1),
            decode_latents_to_rgb_views=_decode,
            video_num_steps=2,
            action_num_steps=2,
            rollout_frames=1,
            ground_truth_provider=_VideoActionOnlyProvider(),
            mot_adapter=_FakeMOTAdapter(),
            geometry_adapter=_FakeGeometryAdapter(),
            schedulers=SimpleNamespace(video=_Scheduler(), action=_Scheduler()),
        )
    except ValueError as exc:
        assert "needs gt.geometry_rgb" in str(exc)
    else:
        raise AssertionError("missing downstream GT geometry was not rejected")


class _ActionWithValidityProvider:
    def maybe_get(self, *, frame_id, predicted_action, state):
        del predicted_action, state
        return GroundTruthStep(
            frame_id=frame_id,
            action=torch.full((1, 1, 1, 1, 1), 99.0),
            action_valid=torch.zeros(1, 1, 1, 1, 1, dtype=torch.bool),
        )


def test_provider_gt_updates_target_and_working_valid_mask() -> None:
    result = self_rollout(
        _batch(),
        transformer=object(),
        config=_config(),
        spec=_spec(),
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        decode_latents_to_rgb_views=_decode,
        video_num_steps=2,
        action_num_steps=2,
        rollout_frames=1,
        ground_truth_provider=_ActionWithValidityProvider(),
        mot_adapter=_FakeMOTAdapter(),
        geometry_adapter=_FakeGeometryAdapter(),
        schedulers=SimpleNamespace(video=_Scheduler(), action=_Scheduler()),
    )

    torch.testing.assert_close(
        result.target_actions[:, :, 3],
        torch.full((1, 1, 1, 1), 99.0),
    )
    assert not bool(result.action_valid_mask[:, :, 3].any())
    torch.testing.assert_close(
        result.pred_actions[:, :, 3],
        torch.full((1, 1, 1, 1), 13.0),
    )


class _ExplicitFutureFrameProvider:
    def __init__(self, batch):
        self.batch = batch
        self.calls = []

    def maybe_get(self, *, frame_id, predicted_action, state):
        del predicted_action, state
        self.calls.append(frame_id)
        if frame_id != 3:
            raise AssertionError("pending GT should be consumed without another provider call")
        return GroundTruthStep(
            frame_id=4,
            action=self.batch["actions"][:, :, 4:5].clone(),
            caused_by_action_frame_id=3,
        )


def test_provider_frame_id_is_used_verbatim_without_implicit_offset() -> None:
    batch = _batch()
    provider = _ExplicitFutureFrameProvider(batch)
    result = self_rollout(
        batch,
        transformer=object(),
        config=_config(),
        spec=_spec(),
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        decode_latents_to_rgb_views=_decode,
        video_num_steps=2,
        action_num_steps=2,
        rollout_frames=2,
        ground_truth_provider=provider,
        mot_adapter=_FakeMOTAdapter(),
        geometry_adapter=_FakeGeometryAdapter(),
        schedulers=SimpleNamespace(video=_Scheduler(), action=_Scheduler()),
    )

    assert provider.calls == [3]
    assert result.diagnostics["sources"][3]["action"] == "predicted"
    assert result.diagnostics["sources"][4]["action"] == "ground_truth"
    assert result.diagnostics["replacements"][0]["frame_id"] == 4
    assert result.diagnostics["replacements"][0]["caused_by_action_frame_id"] == 3


def test_self_rollout_rejects_legacy_chunk_pairs_with_frame_horizon() -> None:
    config = _config()
    config.distill.rollout_chunk_pairs = 1
    config.distill.rollout_horizon_frames = 2
    try:
        self_rollout(
            _batch(),
            transformer=object(),
            config=config,
            spec=_spec(),
            device=torch.device("cpu"),
            empty_text_emb=torch.zeros(1, 1, 1),
            decode_latents_to_rgb_views=_decode,
            video_num_steps=2,
            action_num_steps=2,
            rollout_frames=1,
            mot_adapter=_FakeMOTAdapter(),
            geometry_adapter=_FakeGeometryAdapter(),
            schedulers=SimpleNamespace(video=_Scheduler(), action=_Scheduler()),
        )
    except ValueError as exc:
        assert "legacy fixed-window" in str(exc)
    else:
        raise AssertionError("conflicting rollout settings were not rejected")


def test_self_rollout_matches_deterministic_full_window_result_contract(
    monkeypatch,
) -> None:
    batch = _batch(frames=4)
    spec = SimpleNamespace(
        history_latent_frames=2,
        latent_frames_per_action_chunk_per_view=2,
        target_latent_frames=2,
        total_latent_frames=4,
    )

    def fake_full_window(local_batch, frame_count, **kwargs):
        del frame_count, kwargs
        pred_latents = local_batch["latents"].clone()
        pred_actions = local_batch["actions"].clone()
        pred_geometry = local_batch["geometry_rgb"].clone()
        pred_latents[:, :, 3:4] = 3.0
        pred_actions[:, :, 3:4] = 13.0
        pred_geometry[:, 3:4] = 3.0
        return SimpleNamespace(
            pred_latents=pred_latents,
            pred_actions=pred_actions,
            action_geometry_rgb=pred_geometry,
        )

    monkeypatch.setattr("distillation.rollout.run_mot_inference", fake_full_window)
    legacy = autoregressive_rollout(
        batch,
        transformer=object(),
        config=_config(),
        spec=spec,
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        decode_latents_to_rgb_views=_decode,
        video_num_steps=2,
        action_num_steps=2,
        chunk_pairs=1,
    )
    incremental = self_rollout(
        batch,
        transformer=object(),
        config=_config(),
        spec=spec,
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        decode_latents_to_rgb_views=_decode,
        video_num_steps=2,
        action_num_steps=2,
        rollout_frames=1,
        mot_adapter=_FakeMOTAdapter(),
        geometry_adapter=_FakeGeometryAdapter(),
        schedulers=SimpleNamespace(video=_Scheduler(), action=_Scheduler()),
    )

    assert incremental.pred_latents.shape == legacy.pred_latents.shape
    assert incremental.pred_actions.shape == legacy.pred_actions.shape
    assert incremental.pred_geometry_rgb.shape == legacy.pred_geometry_rgb.shape
    torch.testing.assert_close(incremental.pred_latents, legacy.pred_latents)
    torch.testing.assert_close(incremental.pred_actions, legacy.pred_actions)
    torch.testing.assert_close(incremental.pred_geometry_rgb, legacy.pred_geometry_rgb)
    assert incremental.diagnostics["sources"][3] == {
        "video": "predicted",
        "geometry": "predicted",
        "action": "predicted",
    }
