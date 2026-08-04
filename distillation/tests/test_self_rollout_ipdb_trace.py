"""Small CPU-only self_rollout trace test.

Run normally with::

    PYTHONPATH=. pytest -q distillation/tests/test_self_rollout_ipdb_trace.py -s

Enter ipdb before calling the real rollout engine with::

    SELF_ROLLOUT_IPDB=1 PYTHONPATH=. pytest -q \
        distillation/tests/test_self_rollout_ipdb_trace.py -s

To step through the engine itself, put ``import ipdb; ipdb.set_trace()`` at
the desired line in ``distillation/self_rollout/engine.py``. Useful locations
are the beginning of the frame loop, ``sample_video``, ``sample_action``, and
the three ``commit_*_phase`` helpers.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from distillation.self_rollout.engine import self_rollout
from distillation.self_rollout.geometry_cache import EncodedGeometryFrame


class _TraceScheduler:
    """A two-step scheduler that exposes the real engine control flow."""

    def __init__(self) -> None:
        self.timesteps = torch.tensor([1.0, 0.5])

    def step(self, model_output, timestep, sample):
        del timestep, sample
        return model_output


class _TraceMOTAdapter:
    """Deterministic stand-in for the real transformer-backed MOT adapter."""

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple[int, ...], str]] = []

    def commit_video(self, latents, *, frame_ids, source, **kwargs):
        del latents, kwargs
        self.events.append(("video", tuple(frame_ids), source.name))

    def commit_action(self, actions, *, frame_ids, source, **kwargs):
        del actions, kwargs
        self.events.append(("action", tuple(frame_ids), source.name))

    def predict_video(self, sample, *, frame_id, **kwargs):
        del kwargs
        return torch.full_like(sample, float(frame_id))

    def predict_action(self, sample, *, frame_id, **kwargs):
        del kwargs
        return torch.full_like(sample, float(frame_id + 10))


class _TraceGeometryAdapter:
    """Small geometry adapter that satisfies the engine cache contract."""

    def __init__(self) -> None:
        self.events: list[tuple[int, str]] = []

    def encode_history_and_commit(
        self,
        rgb,
        *,
        frame_ids,
        slot_valid_mask,
        state,
        source,
        **kwargs,
    ):
        return [
            self.encode_and_commit(
                rgb[:, index : index + 1],
                frame_id=frame_id,
                slot_valid_mask=slot_valid_mask[:, index : index + 1],
                state=state,
                source=source,
                **kwargs,
            )
            for index, frame_id in enumerate(frame_ids)
        ]

    def encode_and_commit(self, rgb, *, frame_id, state, source, **kwargs):
        del kwargs
        self.events.append((int(frame_id), source.name))
        encoded = EncodedGeometryFrame(
            frame_id=int(frame_id),
            rgb=rgb,
            final_tokens=torch.empty(0),
            patch_hw=(1, 1),
            image_hw=(1, 1),
            patch_token_start=1,
            cached_outputs=[],
            layer_registers={},
        )
        state.geometry_cache.frames[int(frame_id)] = encoded
        return encoded


def _batch(frames: int = 5) -> dict[str, torch.Tensor]:
    frame_values = torch.arange(frames, dtype=torch.float32)
    return {
        # [B,C,F,V,H,W], V=2
        "latents": frame_values.reshape(1, 1, frames, 1, 1, 1).expand(
            -1, -1, -1, 2, -1, -1
        ).clone(),
        # [B,C,F,N,D]
        "actions": frame_values.reshape(1, 1, frames, 1, 1),
        # [B,F,S,V,3,H,W], S == vae_temporal_factor
        "geometry_rgb": frame_values.reshape(1, frames, 1, 1, 1, 1, 1)
        .expand(-1, -1, 4, 2, 3, -1, -1)
        .clone(),
        "video_latent_valid_mask": torch.ones(1, frames, dtype=torch.bool),
        "geometry_group_valid_mask": torch.ones(1, frames, 4, dtype=torch.bool),
        "action_valid_mask": torch.ones(1, 1, frames, 1, 1, dtype=torch.bool),
        "stream_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "text_emb": torch.ones(1, 1, 1),
    }


def _decode_latents_to_rgb(latents: torch.Tensor) -> torch.Tensor:
    """Return the VAE-like 1 + 4 * (L - 1) RGB frame layout."""

    batch, _channels, frames, views, _height, _width = latents.shape
    values = latents[:, 0, :, 0, 0, 0]
    first = values[:, 0].reshape(batch, 1, 1, 1, 1, 1)
    decoded = [first.expand(batch, 1, views, 3, 1, 1)]
    for frame_id in range(1, frames):
        value = values[:, frame_id].reshape(batch, 1, 1, 1, 1, 1)
        decoded.append(value.expand(batch, 4, views, 3, 1, 1))
    return torch.cat(decoded, dim=1)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        distill=SimpleNamespace(
            generation_shape={
                "profile_name": "segmented_history_strict_geometry_v1",
                "order_mode": "segmented",
                "history_frames": 2,
                "chunk_size": 2,
                "window_size": 16,
            }
        ),
        vae_temporal_factor=4,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        seed=0,
    )


def _spec() -> SimpleNamespace:
    return SimpleNamespace(
        history_latent_frames=2,
        latent_frames_per_action_chunk_per_view=3,
    )


def _run_trace():
    mot = _TraceMOTAdapter()
    geometry = _TraceGeometryAdapter()

    if os.getenv("SELF_ROLLOUT_IPDB"):
        import ipdb

        ipdb.set_trace()

    result = self_rollout(
        _batch(),
        transformer=object(),
        config=_config(),
        spec=_spec(),
        device=torch.device("cpu"),
        empty_text_emb=torch.zeros(1, 1, 1),
        decode_latents_to_rgb_views=_decode_latents_to_rgb,
        video_num_steps=2,
        action_num_steps=2,
        rollout_frames=2,
        mot_adapter=mot,
        geometry_adapter=geometry,
        schedulers=SimpleNamespace(
            video=_TraceScheduler(),
            action=_TraceScheduler(),
        ),
    )
    return result, mot, geometry


def test_self_rollout_cpu_trace_without_model_or_data() -> None:
    """Exercise the real engine with deterministic CPU-only collaborators."""

    result, mot, geometry = _run_trace()

    assert [(kind, frame_ids, source) for kind, frame_ids, source in mot.events] == [
        ("video", (0, 1), "HISTORY"),
        ("action", (0, 1), "HISTORY"),
        ("video", (2,), "ANCHOR"),
        ("action", (2,), "ANCHOR"),
        ("video", (3,), "PREDICTED"),
        ("action", (3,), "PREDICTED"),
        ("video", (4,), "PREDICTED"),
        ("action", (4,), "PREDICTED"),
    ]
    assert geometry.events == [
        (0, "HISTORY"),
        (1, "HISTORY"),
        (2, "ANCHOR"),
        (3, "PREDICTED"),
        (4, "PREDICTED"),
    ]

    torch.testing.assert_close(
        result.pred_latents[:, :, 3],
        torch.full((1, 1, 2, 1, 1), 3.0),
    )
    torch.testing.assert_close(
        result.pred_actions[:, :, 3],
        torch.full((1, 1, 1, 1), 13.0),
    )
    torch.testing.assert_close(
        result.pred_latents[:, :, 4],
        torch.full((1, 1, 2, 1, 1), 4.0),
    )
    assert result.pred_geometry_rgb.shape == (1, 5, 4, 2, 3, 1, 1)
    assert result.diagnostics["sources"][3] == {
        "video": "predicted",
        "geometry": "predicted",
        "action": "predicted",
    }
    assert result.diagnostics["replacements"] == []


def main() -> None:
    """Run the same trace directly and print the important rollout state."""

    result, mot, geometry = _run_trace()
    print("self_rollout CPU trace: PASS")
    print("MOT events:")
    for event in mot.events:
        print(f"  {event}")
    print(f"Geometry events: {geometry.events}")
    print(f"pred_latents shape: {tuple(result.pred_latents.shape)}")
    print(f"pred_actions shape: {tuple(result.pred_actions.shape)}")
    print(f"pred_geometry_rgb shape: {tuple(result.pred_geometry_rgb.shape)}")
    print(f"frame 3 latent value: {result.pred_latents[:, :, 3].flatten().tolist()}")
    print(f"frame 3 action value: {result.pred_actions[:, :, 3].flatten().tolist()}")
    print(f"diagnostics: {result.diagnostics}")

# PYTHONPATH=. python distillation/tests/test_self_rollout_ipdb_trace.py
if __name__ == "__main__":
    main()
