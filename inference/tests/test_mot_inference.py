from types import SimpleNamespace

import torch

from inference.mot_inference import (
    run_geometry_inference,
    run_mot_inference,
    run_video_inference,
)


class _FakeScheduler:
    instances = []

    def __init__(self, **_kwargs):
        self.timesteps = torch.tensor([2.0, 1.0])
        self.predictions = []
        self.__class__.instances.append(self)

    def set_timesteps(self, _steps):
        return None

    def step(self, prediction, _timestep, sample):
        self.predictions.append(prediction.detach().clone())
        return sample


class _FakeTransformer:
    def __init__(self):
        self.events = []
        self.conditions = []
        self.action_inputs = []

    def __call__(self, input_dict, *, mode):
        if mode == "precompute_geometry":
            rgb = input_dict["rgb"].detach().clone()
            slot_valid = input_dict["slot_valid_mask"].detach().clone()
            frames = rgb.shape[1]
            condition = {
                "final_geometry": {"condition_id": len(self.conditions)},
                "layer_registers": {},
                "depth": torch.full((1, frames * 4, 1, 1, 1), len(self.conditions) + 1.0),
                "depth_conf": torch.full((1, frames * 4, 1, 1), len(self.conditions) + 21.0),
                "points": torch.full((1, frames * 4, 1, 1, 1, 3), len(self.conditions) + 11.0),
            }
            self.conditions.append((rgb, slot_valid, condition))
            self.events.append((mode, None, condition))
            return condition

        condition = input_dict["geometry_dict"]["precomputed_condition"]
        text_kind = "cond" if bool(input_dict["latent_dict"]["text_emb"].any()) else "uncond"
        self.events.append((mode, text_kind, condition))
        if mode == "inference_video":
            value = 3.0 if text_kind == "cond" else 1.0
            return {"latent_pred": torch.full_like(input_dict["latent_dict"]["noisy_latents"], value)}
        if mode == "inference_action":
            self.action_inputs.append(input_dict["latent_dict"]["noisy_latents"].detach().clone())
            return {"action_pred": torch.zeros_like(input_dict["action_dict"]["noisy_latents"])}
        raise AssertionError(f"unexpected mode: {mode}")


def test_inference_is_gt_history_then_pred_video_geometry_and_action(monkeypatch) -> None:
    spec = SimpleNamespace(
        total_latent_frames=8,
        history_latent_frames=4,
        target_latent_frames=4,
        latent_frames_per_action_chunk_per_view=4,
        attention_window_size=4,
        action_per_frame=1,
    )
    config = SimpleNamespace(
        snr_shift=5.0,
        action_snr_shift=1.0,
        num_inference_steps=25,
        action_num_inference_steps=50,
        guidance_scale=5.0,
        action_guidance_scale=1.0,
        vae_temporal_factor=4,
    )
    transformer = _FakeTransformer()
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.full_like(value, 4.0))
    decode_calls = []

    def decode_latents_to_rgb_views(latents):
        decode_calls.append(latents.detach().clone())
        return torch.full(
            (
                latents.shape[0],
                1 + 4 * (latents.shape[2] - 1),
                latents.shape[3],
                3,
                1,
                1,
            ),
            9.0,
            dtype=latents.dtype,
            device=latents.device,
        )

    latents = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1, 1)
    geometry_rgb = torch.full((1, 8, 4, 1, 3, 1, 1), 7.0)
    geometry_rgb[:, :4] = 1.0
    geometry_rgb[:, 4] = 2.0
    batch = {
        "latents": latents,
        "actions": torch.zeros(1, 1, 8, 1, 1),
        "action_loss_mask": torch.ones(1, 1, 8, 1, 1, dtype=torch.bool),
        "action_valid_mask": torch.ones(1, 1, 8, 1, 1, dtype=torch.bool),
        "text_emb": torch.ones(1, 1, 1),
        "geometry_rgb": geometry_rgb,
        "geometry_group_valid_mask": torch.ones(1, 8, 4, dtype=torch.bool),
        "stream_ids": torch.zeros(1, 1, dtype=torch.long),
        "video_latent_valid_mask": torch.ones(1, 8, dtype=torch.bool),
    }
    _FakeScheduler.instances.clear()

    result = run_mot_inference(
        batch,
        frame_count=8,
        transformer=transformer,
        config=config,
        spec=spec,
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1),
        decode_latents_to_rgb_views=decode_latents_to_rgb_views,
        scheduler_factory=_FakeScheduler,
    )

    modes = [(mode, text_kind) for mode, text_kind, _condition in transformer.events]
    assert modes == [
        ("precompute_geometry", None),
        ("inference_video", "cond"),
        ("inference_video", "uncond"),
        ("inference_video", "cond"),
        ("inference_video", "uncond"),
        ("precompute_geometry", None),
        ("inference_action", "cond"),
        ("inference_action", "cond"),
    ]

    video_condition_rgb, video_slot_valid, video_condition = transformer.conditions[0]
    action_condition_rgb, action_slot_valid, action_condition = transformer.conditions[1]
    assert torch.all(video_condition_rgb[:, :5] != 0)
    assert torch.all(video_condition_rgb[:, 5:] == 0)
    assert torch.all(video_slot_valid[:, :5])
    assert not bool(video_slot_valid[:, 5:].any())
    assert torch.all(action_condition_rgb[:, :4] == 1)
    assert torch.all(action_condition_rgb[:, 4] == 2)
    assert torch.all(action_condition_rgb[:, 5:] == 9)
    assert bool(action_slot_valid.all())
    assert all(condition is video_condition for _, _, condition in transformer.events[1:5])
    assert all(condition is action_condition for _, _, condition in transformer.events[6:])

    assert len(decode_calls) == 1
    assert torch.all(decode_calls[0][:, :, 0] == latents[:, :, 4:5].to(torch.bfloat16))
    assert torch.all(decode_calls[0][:, :, 1:] == 4)
    torch.testing.assert_close(result.pred_latents[:, :, :5], latents[:, :, :5].to(torch.bfloat16))
    assert torch.all(result.pred_latents[:, :, 5:] == 4)
    assert all(torch.all(value[:, :, 5:] == 4) for value in transformer.action_inputs)

    assert result.pred_points is action_condition["points"]
    assert torch.all(result.pred_points == 12)
    assert torch.all(result.pred_depth == 2)
    assert torch.all(result.pred_depth_conf == 22)
    torch.testing.assert_close(result.action_geometry_rgb, action_condition_rgb)

    video_scheduler = _FakeScheduler.instances[0]
    assert len(video_scheduler.predictions) == 2
    assert all(torch.all(prediction == 11) for prediction in video_scheduler.predictions)
    assert result.pred_latents.dtype == torch.bfloat16
    assert video_condition_rgb.dtype == torch.bfloat16


def test_video_inference_stops_before_geometry_heads_and_action(monkeypatch) -> None:
    spec = SimpleNamespace(
        total_latent_frames=8,
        history_latent_frames=4,
        target_latent_frames=4,
        latent_frames_per_action_chunk_per_view=4,
        attention_window_size=4,
    )
    config = SimpleNamespace(
        snr_shift=5.0,
        num_inference_steps=2,
        guidance_scale=5.0,
        vae_temporal_factor=4,
    )
    transformer = _FakeTransformer()
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.full_like(value, 4.0))
    batch = {
        "latents": torch.zeros(1, 1, 8, 1, 1, 1),
        "actions": torch.zeros(1, 1, 8, 1, 1),
        "action_loss_mask": torch.ones(1, 1, 8, 1, 1, dtype=torch.bool),
        "action_valid_mask": torch.ones(1, 1, 8, 1, 1, dtype=torch.bool),
        "text_emb": torch.ones(1, 1, 1),
        "geometry_rgb": torch.ones(1, 8, 4, 1, 3, 1, 1),
        "geometry_group_valid_mask": torch.ones(1, 8, 4, dtype=torch.bool),
        "stream_ids": torch.zeros(1, 1, dtype=torch.long),
        "video_latent_valid_mask": torch.ones(1, 8, dtype=torch.bool),
    }

    result = run_video_inference(
        batch,
        frame_count=8,
        transformer=transformer,
        config=config,
        spec=spec,
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1),
        decode_latents_to_rgb_views=lambda latents: torch.ones(
            latents.shape[0],
            13,
            latents.shape[3],
            3,
            1,
            1,
            dtype=latents.dtype,
        ),
        scheduler_factory=_FakeScheduler,
    )

    assert [(mode, text_kind) for mode, text_kind, _ in transformer.events] == [
        ("precompute_geometry", None),
        ("inference_video", "cond"),
        ("inference_video", "uncond"),
        ("inference_video", "cond"),
        ("inference_video", "uncond"),
    ]
    assert len(transformer.conditions) == 1
    assert result.pred_latents.shape == batch["latents"].shape
    assert result.action_geometry_rgb.shape == batch["geometry_rgb"].shape


def test_geometry_inference_uses_gt_rgb_without_video_or_action() -> None:
    spec = SimpleNamespace(
        total_latent_frames=8,
        latent_frames_per_action_chunk_per_view=4,
        attention_window_size=4,
    )
    transformer = _FakeTransformer()
    geometry_rgb = torch.arange(8, dtype=torch.float32).reshape(
        1,
        8,
        1,
        1,
        1,
        1,
        1,
    ).expand(-1, -1, 4, -1, 3, -1, -1)
    batch = {
        "geometry_rgb": geometry_rgb,
        "geometry_group_valid_mask": torch.ones(1, 8, 4, dtype=torch.bool),
        "stream_ids": torch.zeros(1, 1, dtype=torch.long),
    }

    result = run_geometry_inference(
        batch,
        frame_count=8,
        transformer=transformer,
        spec=spec,
        device=torch.device("cpu"),
    )

    assert [(mode, text_kind) for mode, text_kind, _ in transformer.events] == [
        ("precompute_geometry", None)
    ]
    torch.testing.assert_close(result.geometry_rgb, geometry_rgb.to(torch.bfloat16))
    assert torch.all(result.pred_depth == 1)
    assert torch.all(result.pred_depth_conf == 21)
    assert torch.all(result.pred_points == 11)
