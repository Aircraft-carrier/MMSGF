"""Simplified incremental rollout pipeline tests."""
import pytest
import torch

from distillation.pipeline import KVCache, SelfGradientForcingTrainingPipeline
from wan_va.utils.scheduler import FlowMatchScheduler


def _scheduler() -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return scheduler


class _FakeGenerator:
    """Minimal wrapper implementing the pipeline's video/action contract."""

    def __init__(self) -> None:
        self.video_scheduler = _scheduler()
        self.action_scheduler = _scheduler()
        self.predict_video_calls = 0
        self.predict_action_calls = 0
        self.commits = []

    def generate_video(self, noisy, timestep, frame_ids, stream_ids, *, cache, text_emb):
        del timestep, stream_ids, cache, text_emb
        self.predict_video_calls += 1
        return noisy * 0.5

    def generate_action(self, noisy, timestep, frame_ids, *, cache, text_emb):
        del timestep, cache, text_emb
        self.predict_action_calls += 1
        return noisy * 0.5

    def commit_video(self, latents, frame_ids, stream_ids, *, cache, text_emb):
        del stream_ids, cache, text_emb
        self.commits.append(("video", tuple(frame_ids), tuple(latents.shape)))

    def commit_action(self, actions, frame_ids, *, cache, text_emb):
        del cache, text_emb
        self.commits.append(("action", tuple(frame_ids), tuple(actions.shape)))


def test_kv_cache_transactions_keep_only_committed_kv() -> None:
    cache = KVCache()
    key = torch.randn(1, 2, 1, 1)
    cache.append(0, key, key.clone())

    transaction_id = cache.new_transaction_id()
    cache.append(0, key + 1, key + 1, transaction_id=transaction_id)
    key_all, _ = cache.materialize(0, transaction_id=transaction_id)
    assert key_all.shape == (1, 4, 1, 1)

    cache.discard(transaction_id)
    key_all, _ = cache.materialize(0)
    assert key_all.shape == (1, 2, 1, 1)
    assert cache.committed_token_count(0) == 2

    commit_id = cache.new_transaction_id()
    cache.append(0, key + 2, key + 2, transaction_id=commit_id)
    cache.commit(commit_id)
    assert cache.committed_token_count(0) == 4
    cache.assert_no_transactions()

    with pytest.raises(ValueError):
        cache.materialize(1)


def test_pipeline_rollout_commits_blocks_and_records_exit() -> None:
    generator = _FakeGenerator()
    pipeline = SelfGradientForcingTrainingPipeline(
        denoising_step_list={
            "video": [1000, 500],
            "action": [1000, 500, 250],
        },
        generator=generator,
        num_frame_per_block=2,
        per_rank_exit_step=True,
    )
    batch = {
        "latents": torch.zeros(1, 1, 4, 1, 1, 1),
        "actions": torch.zeros(1, 1, 4, 1, 1),
        "action_valid_mask": torch.ones(1, 1, 4, 1, 1, dtype=torch.bool),
        "stream_ids": torch.zeros(1, 1, dtype=torch.long),
        "text_emb": torch.zeros(1, 1, 1),
    }
    result = pipeline.generate(
        batch,
        rollout_frames=2,
        history_frames=1,
        device=torch.device("cpu"),
    )

    assert tuple(result.pred_frame_ids) == (2, 3)
    assert tuple(result.video_hat.shape) == (1, 1, 2, 1, 1, 1)
    assert tuple(result.action_hat.shape) == (1, 1, 2, 1, 1)
    assert tuple(result.video_noisy_at_t.shape) == (1, 1, 2, 1, 1, 1)
    assert tuple(result.action_noisy_at_t.shape) == (1, 1, 2, 1, 1)
    assert isinstance(result.video_exit_timestep, float)
    assert isinstance(result.action_exit_timestep, float)
    # Video denoises through 2 steps, action through 3, then clean commits.
    assert generator.predict_video_calls == 2
    assert generator.predict_action_calls == 3
    assert generator.commits == [
        ("video", (0,), (1, 1, 1, 1, 1, 1)),
        ("action", (0,), (1, 1, 1, 1, 1)),
        ("video", (1,), (1, 1, 1, 1, 1, 1)),
        ("action", (1,), (1, 1, 1, 1, 1)),
        ("video", (2, 3), (1, 1, 2, 1, 1, 1)),
        ("action", (2, 3), (1, 1, 2, 1, 1)),
    ]

    video_exit_id = result.video_exit_id
    action_exit_id = result.action_exit_id
    assert video_exit_id in (0, 1)
    assert action_exit_id in (0, 1, 2)
    expected_video_from = 1000.0 if video_exit_id == 0 else 500.0
    expected_video_to = 500.0 if video_exit_id == 0 else 0.0
    expected_action_from = (
        1000.0 if action_exit_id == 0 else 500.0 if action_exit_id == 1 else 250.0
    )
    expected_action_to = (
        500.0 if action_exit_id == 0 else 250.0 if action_exit_id == 1 else 0.0
    )
    assert result.video_denoised_timestep_from == expected_video_from
    assert result.video_denoised_timestep_to == expected_video_to
    assert result.action_denoised_timestep_from == expected_action_from
    assert result.action_denoised_timestep_to == expected_action_to
