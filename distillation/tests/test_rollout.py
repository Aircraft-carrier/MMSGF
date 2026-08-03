import json
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from distillation.rollout import (
    autoregressive_rollout,
    record_sgf_trajectory,
    save_rollout_artifacts,
)


@dataclass(frozen=True)
class _SGFSpec:
    history_chunks: int = 1
    target_chunks: int = 1
    latent_frames_per_action_chunk_per_view: int = 4
    history_latent_frames: int = 4
    target_latent_frames: int = 4
    total_latent_frames: int = 8
    attention_window_size: int = 4


def _batch() -> dict:
    frames = 6
    return {
        "latents": torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1, 1),
        "actions": torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1),
        "action_loss_mask": torch.ones(1, 1, frames, 1, 1, dtype=torch.bool),
        "action_valid_mask": torch.ones(1, 1, frames, 1, 1, dtype=torch.bool),
        "video_latent_valid_mask": torch.ones(1, frames, dtype=torch.bool),
        "text_emb": torch.ones(1, 1, 1),
        "geometry_rgb": torch.arange(frames, dtype=torch.float32).reshape(
            1, frames, 1, 1, 1, 1, 1
        ),
        "geometry_group_valid_mask": torch.ones(1, frames, 1, dtype=torch.bool),
        "stream_ids": torch.zeros(1, 1, dtype=torch.long),
    }


def _sgf_batch() -> dict:
    frames = 8
    return {
        "latents": torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1, 1),
        "actions": torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1),
        "action_loss_mask": torch.zeros(1, 1, frames, 1, 1, dtype=torch.bool),
        "action_valid_mask": torch.ones(1, 1, frames, 1, 1, dtype=torch.bool),
        "video_latent_loss_mask": torch.zeros(1, frames, dtype=torch.bool),
        "video_latent_valid_mask": torch.ones(1, frames, dtype=torch.bool),
        "text_emb": torch.ones(1, 1, 1),
        "empty_text_emb": torch.zeros(1, 1, 1),
        "geometry_rgb": torch.arange(frames, dtype=torch.float32).reshape(
            1, frames, 1, 1, 1, 1, 1
        ),
        "geometry_group_valid_mask": torch.ones(1, frames, 1, dtype=torch.bool),
        "stream_ids": torch.zeros(1, 1, dtype=torch.long),
    }


def test_sgf_record_runs_history_then_target_and_builds_full_replay() -> None:
    calls = []

    def fake_run(local_batch, frame_count, *, spec, scheduler_factory, config, **kwargs):
        calls.append(local_batch)
        call_id = len(calls)
        target = slice(int(spec.history_latent_frames) + 1, frame_count)
        target_frames = frame_count - target.start

        video_scheduler = scheduler_factory(shift=1.0, sigma_min=0.0, extra_one_step=True)
        video_scheduler.set_timesteps(config.num_inference_steps)
        video_sample = torch.full((1, 1, target_frames, 1, 1, 1), 10.0 * call_id)
        for timestep in video_scheduler.timesteps:
            video_sample = video_scheduler.step(torch.ones_like(video_sample), timestep, video_sample)

        action_scheduler = scheduler_factory(shift=1.0, sigma_min=0.0, extra_one_step=True)
        action_scheduler.set_timesteps(config.action_num_inference_steps)
        action_sample = torch.full((1, 1, target_frames, 1, 1), 20.0 * call_id)
        for timestep in action_scheduler.timesteps:
            action_sample = action_scheduler.step(torch.ones_like(action_sample), timestep, action_sample)

        pred_latents = local_batch["latents"].clone()
        pred_actions = local_batch["actions"].clone()
        pred_geometry = local_batch["geometry_rgb"].clone()
        pred_latents[:, :, target] += 100.0 * call_id
        pred_actions[:, :, target] += 10.0 * call_id
        pred_geometry[:, target] += 1000.0 * call_id
        return SimpleNamespace(
            pred_latents=pred_latents,
            pred_actions=pred_actions,
            action_geometry_rgb=pred_geometry,
        )

    batch = _sgf_batch()
    original = {key: value.clone() for key, value in batch.items() if torch.is_tensor(value)}
    context = record_sgf_trajectory(
        batch,
        transformer=object(),
        config=SimpleNamespace(num_inference_steps=2, action_num_inference_steps=3),
        spec=_SGFSpec(),
        device=torch.device("cpu"),
        empty_text_emb=batch["empty_text_emb"],
        decode_latents_to_rgb_views=lambda value: value,
        video_record_step=1,
        action_record_step=2,
        run_inference=fake_run,
    )

    assert len(calls) == 2
    assert calls[0]["latents"].shape[2] == 4
    torch.testing.assert_close(calls[1]["latents"][:, :, 1:4], batch["latents"][:, :, 1:4] + 100)
    torch.testing.assert_close(calls[1]["actions"][:, :, 1:4], batch["actions"][:, :, 1:4] + 10)
    torch.testing.assert_close(calls[1]["latents"][:, :, 4], batch["latents"][:, :, 4])

    expected_frames = torch.tensor([[False, True, True, True, False, True, True, True]])
    assert torch.equal(context.masks.video, expected_frames)
    assert torch.equal(context.masks.action.any(dim=(1, 3, 4)), expected_frames)
    torch.testing.assert_close(
        context.batch["latents"][:, :, 1:4],
        batch["latents"][:, :, 1:4] + 100,
    )
    torch.testing.assert_close(
        context.batch["latents"][:, :, 5:8],
        batch["latents"][:, :, 5:8] + 200,
    )
    torch.testing.assert_close(
        context.batch["actions"][:, :, 1:4],
        batch["actions"][:, :, 1:4] + 10,
    )
    torch.testing.assert_close(
        context.batch["actions"][:, :, 5:8],
        batch["actions"][:, :, 5:8] + 20,
    )
    torch.testing.assert_close(
        context.batch["geometry_rgb"][:, 1:4],
        batch["geometry_rgb"][:, 1:4] + 1000,
    )
    torch.testing.assert_close(
        context.batch["geometry_rgb"][:, 5:8],
        batch["geometry_rgb"][:, 5:8] + 2000,
    )
    assert torch.all(context.timesteps.video[:, 1:4] == 500)
    assert torch.all(context.timesteps.video[:, 5:8] == 500)
    assert torch.allclose(
        context.timesteps.action[:, 1:4],
        torch.full((1, 3), 1000 / 3),
    )
    assert torch.allclose(
        context.timesteps.action[:, 5:8],
        torch.full((1, 3), 1000 / 3),
    )
    assert torch.allclose(context.noisy.video[:, :, 1:4], torch.full((1, 1, 3, 1, 1, 1), 9.5))
    assert torch.allclose(context.noisy.video[:, :, 5:8], torch.full((1, 1, 3, 1, 1, 1), 19.5))
    assert torch.equal(context.noisy.video[:, :, 0], context.generated.video[:, :, 0])
    assert torch.equal(context.noisy.video[:, :, 4], context.generated.video[:, :, 4])
    for key, value in original.items():
        torch.testing.assert_close(batch[key], value)


def test_rollout_feeds_generated_target_back_as_next_history(monkeypatch) -> None:
    calls = []

    def fake_run(local_batch, frame_count, **kwargs):
        calls.append((local_batch, kwargs["config"]))
        call_id = len(calls)
        pred_latents = local_batch["latents"].clone()
        pred_actions = local_batch["actions"].clone()
        pred_geometry = local_batch["geometry_rgb"].clone()
        pred_latents[:, :, 2:] += 100.0 * call_id
        pred_actions[:, :, 2:] += 10.0 * call_id
        pred_geometry[:, 2:] += 1000.0 * call_id
        return SimpleNamespace(
            pred_latents=pred_latents,
            pred_actions=pred_actions,
            action_geometry_rgb=pred_geometry,
        )

    monkeypatch.setattr("distillation.rollout.run_mot_inference", fake_run)
    config = SimpleNamespace(num_inference_steps=25, action_num_inference_steps=50)
    spec = SimpleNamespace(
        latent_frames_per_action_chunk_per_view=2,
        history_latent_frames=2,
        target_latent_frames=2,
        total_latent_frames=4,
    )

    batch = _batch()
    original_latents = batch["latents"].clone()
    result = autoregressive_rollout(
        batch,
        transformer=object(),
        config=config,
        spec=spec,
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1),
        decode_latents_to_rgb_views=lambda value: value,
        video_num_steps=3,
        action_num_steps=4,
        chunk_pairs=2,
    )

    assert len(calls) == 2
    torch.testing.assert_close(calls[1][0]["latents"][:, :, :2], result.pred_latents[:, :, 2:4])
    torch.testing.assert_close(calls[1][0]["actions"][:, :, :2], result.pred_actions[:, :, 2:4])
    torch.testing.assert_close(calls[1][0]["geometry_rgb"][:, :2], result.pred_geometry_rgb[:, 2:4])
    assert calls[0][1].num_inference_steps == 3
    assert calls[0][1].action_num_inference_steps == 4
    assert result.pred_latents.shape[2] == 6
    assert result.target_latents.shape[2] == 6
    torch.testing.assert_close(batch["latents"], original_latents)


def test_rollout_artifact_manifest_contains_video_actions_and_tensors(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "distillation.rollout._save_video",
        lambda _result, _decode, path, _fps: path.write_bytes(b"video"),
    )
    monkeypatch.setattr(
        "distillation.rollout._save_action_plot",
        lambda _result, _batch, path: path.write_bytes(b"plot"),
    )
    batch = _batch()
    result = SimpleNamespace(
        pred_latents=batch["latents"],
        target_latents=batch["latents"],
        pred_actions=batch["actions"],
        target_actions=batch["actions"],
        pred_geometry_rgb=batch["geometry_rgb"],
        target_geometry_rgb=batch["geometry_rgb"],
        action_valid_mask=batch["action_valid_mask"],
        chunk_pairs=2,
        chunk_frames=2,
    )

    artifacts = save_rollout_artifacts(
        result,
        batch,
        decode_latents_to_rgb_views=lambda value: value,
        output_dir=tmp_path,
    )

    assert all(tmp_path.joinpath(name).is_file() for name in (
        "rollout_target_vs_generated.mp4",
        "rollout_actions.png",
        "rollout.pt",
        "metadata.json",
    ))
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["chunk_pairs"] == 2
    assert metadata["clean_target_anchor_per_chunk"] is True
    assert artifacts["video"].endswith("rollout_target_vs_generated.mp4")
