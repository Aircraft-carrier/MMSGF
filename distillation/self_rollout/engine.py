"""End-to-end latent -> geometry -> action incremental self rollout."""
from __future__ import annotations

from typing import Any, Callable

import torch

from .geometry_cache import GeometryIncrementalAdapter, GeometryRolloutCache
from .mot_adapter import MOTIncrementalAdapter
from .provider import GroundTruthProvider, GroundTruthStep
from .recorder import SelfRolloutRecorder
from .result import RolloutResult
from .scheduler import RolloutSchedulers, build_rollout_schedulers
from .state import CacheSource, RolloutPhase, RolloutState


def decoded_rgb_to_geometry_groups(
    decoded_rgb: torch.Tensor,
    *,
    latent_frames: int,
    vae_temporal_factor: int,
) -> torch.Tensor:
    """Distillation-owned copy of the MOT latent/RGB temporal grouping contract."""

    latent_frames = int(latent_frames)
    vae_temporal_factor = int(vae_temporal_factor)
    expected = 1 + vae_temporal_factor * (latent_frames - 1)
    if decoded_rgb.ndim != 6 or decoded_rgb.shape[1] != expected:
        raise ValueError(
            f"decoded RGB must be [B,{expected},V,C,H,W], got {tuple(decoded_rgb.shape)}"
        )
    first = decoded_rgb[:, 0:1].expand(
        -1,
        vae_temporal_factor,
        -1,
        -1,
        -1,
        -1,
    )
    groups = [first]
    for latent_index in range(1, latent_frames):
        start = 1 + vae_temporal_factor * (latent_index - 1)
        groups.append(decoded_rgb[:, start : start + vae_temporal_factor])
    return torch.stack(groups, dim=1)


def _frame_video_valid(batch: dict[str, Any], frame_id: int, device: torch.device):
    mask = batch.get("video_latent_valid_mask")
    if mask is None:
        return None
    return mask[:, frame_id : frame_id + 1].to(device=device, dtype=torch.bool)


def _frame_action_valid(batch: dict[str, Any], frame_id: int, device: torch.device):
    mask = batch.get("action_valid_mask")
    if mask is None:
        return None
    return mask[:, :, frame_id : frame_id + 1].to(device=device, dtype=torch.bool)


def _frame_geometry_valid(batch: dict[str, Any], frame_id: int, device: torch.device):
    mask = batch.get("geometry_group_valid_mask")
    if mask is None:
        rgb = batch["geometry_rgb"]
        return torch.ones(
            (rgb.shape[0], 1, rgb.shape[2]),
            dtype=torch.bool,
            device=device,
        )
    return mask[:, frame_id : frame_id + 1].to(device=device, dtype=torch.bool)


def _source_name(source: CacheSource | None) -> str | None:
    return None if source is None else source.name.lower()


def _config_value(container: Any, name: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(name, default)
    return getattr(container, name, default)


def _validate_rollout_inputs(
    batch: dict[str, Any],
    *,
    config: Any,
    spec: Any,
    rollout_frames: int,
) -> None:
    required = {
        "latents",
        "actions",
        "geometry_rgb",
        "action_valid_mask",
        "stream_ids",
        "text_emb",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"self_rollout batch is missing {missing}")
    for name in required:
        if not torch.is_tensor(batch[name]):
            raise TypeError(f"self_rollout batch[{name!r}] must be a tensor")

    latents = batch["latents"]
    actions = batch["actions"]
    geometry = batch["geometry_rgb"]
    action_valid = batch["action_valid_mask"]
    stream_ids = batch["stream_ids"]
    if latents.ndim != 6:
        raise ValueError(f"latents must be [B,C,F,V,H,W], got {tuple(latents.shape)}")
    if actions.ndim != 5:
        raise ValueError(f"actions must be [B,C,F,N,D], got {tuple(actions.shape)}")
    if geometry.ndim != 7:
        raise ValueError(
            "geometry_rgb must be [B,F,S,V,3,H,W], "
            f"got {tuple(geometry.shape)}"
        )
    batch_size, _channels, frames, views = latents.shape[:4]
    if actions.shape[0] != batch_size or actions.shape[2] != frames:
        raise ValueError("latent/action batch and frame axes must match")
    if geometry.shape[0] != batch_size or geometry.shape[1] != frames:
        raise ValueError("latent/geometry batch and frame axes must match")
    if geometry.shape[3] != views or geometry.shape[4] != 3:
        raise ValueError("geometry views must match latents and geometry RGB needs 3 channels")
    if tuple(stream_ids.shape) != (batch_size, views):
        raise ValueError(
            f"stream_ids must be [{batch_size},{views}], got {tuple(stream_ids.shape)}"
        )
    if batch["text_emb"].shape[0] not in {1, batch_size}:
        raise ValueError("text_emb batch must be 1 or match latent batch size")
    expected_action_valid = (
        batch_size,
        actions.shape[1],
        frames,
        actions.shape[3],
        1,
    )
    if tuple(action_valid.shape) != expected_action_valid:
        raise ValueError(
            f"action_valid_mask must be {expected_action_valid}, "
            f"got {tuple(action_valid.shape)}"
        )
    if "video_latent_valid_mask" in batch and tuple(
        batch["video_latent_valid_mask"].shape
    ) != (batch_size, frames):
        raise ValueError(
            f"video_latent_valid_mask must be [{batch_size},{frames}]"
        )
    if "geometry_group_valid_mask" in batch and tuple(
        batch["geometry_group_valid_mask"].shape
    ) != (batch_size, frames, geometry.shape[2]):
        raise ValueError(
            "geometry_group_valid_mask must match geometry [B,F,S] axes"
        )

    temporal_factor = int(config.vae_temporal_factor)
    if geometry.shape[2] != temporal_factor:
        raise ValueError(
            f"geometry group slots must equal vae_temporal_factor={temporal_factor}, "
            f"got {geometry.shape[2]}"
        )
    configured_height = _config_value(config, "height")
    configured_width = _config_value(config, "width")
    if configured_height is not None and geometry.shape[-2] != int(configured_height):
        raise ValueError("geometry height must match config.height")
    if configured_width is not None and geometry.shape[-1] != int(configured_width):
        raise ValueError("geometry width must match config.width")

    history_frames = int(spec.history_latent_frames)
    anchor = history_frames
    end_frame = anchor + int(rollout_frames)
    if anchor >= frames:
        raise ValueError("batch does not contain the target anchor frame")
    if end_frame >= frames:
        raise ValueError(
            f"rollout needs frames through {end_frame}, but batch has {frames}"
        )

    distill = config.distill
    if (
        _config_value(distill, "rollout_chunk_pairs") is not None
        and _config_value(distill, "rollout_horizon_frames") is not None
    ):
        raise ValueError(
            "rollout_chunk_pairs is a legacy fixed-window setting; do not set it "
            "together with rollout_horizon_frames"
        )


@torch.no_grad()
def self_rollout(
    batch: dict[str, Any],
    *,
    transformer,
    config: Any,
    spec: Any,
    device: torch.device,
    empty_text_emb: torch.Tensor,
    decode_latents_to_rgb_views: Callable[[torch.Tensor], torch.Tensor],
    video_num_steps: int,
    action_num_steps: int,
    rollout_frames: int,
    ground_truth_provider: GroundTruthProvider | None = None,
    replacement_policy: str = "require_ground_truth",
    generator: torch.Generator | None = None,
    mot_adapter: MOTIncrementalAdapter | None = None,
    geometry_adapter: GeometryIncrementalAdapter | None = None,
    schedulers: RolloutSchedulers | None = None,
    recorder: SelfRolloutRecorder | None = None,
) -> RolloutResult:
    """Run a distillation-owned incremental rollout without fixed-window inference."""

    rollout_frames = int(rollout_frames)
    if rollout_frames <= 0:
        raise ValueError(f"rollout_frames must be positive, got {rollout_frames}")
    if replacement_policy not in {"require_ground_truth", "recompute_predicted"}:
        raise ValueError(f"unsupported replacement_policy={replacement_policy!r}")
    _validate_rollout_inputs(
        batch,
        config=config,
        spec=spec,
        rollout_frames=rollout_frames,
    )

    history_frames = int(spec.history_latent_frames)
    anchor = history_frames
    end_frame = anchor + rollout_frames

    shape = config.distill.generation_shape
    profile_name = str(
        shape.get("profile_name", "segmented_history_strict_geometry_v1")
    )
    if profile_name != "segmented_history_strict_geometry_v1":
        raise ValueError(f"unsupported self_rollout profile {profile_name!r}")
    configured_history = int(shape.get("history_frames", history_frames))
    if configured_history != history_frames:
        raise ValueError(
            "generation_shape.history_frames must match MOT spec: "
            f"{configured_history} != {history_frames}"
        )
    chunk_size = int(shape["chunk_size"])
    window_size = int(shape["window_size"])
    if str(shape.get("order_mode", "segmented")) != "segmented":
        raise ValueError("self_rollout requires generation_shape.order_mode='segmented'")

    geometry_cache = GeometryRolloutCache()
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(getattr(config, "seed", 0)))
    state = RolloutState(geometry_cache=geometry_cache, generator=generator)
    mot_adapter = mot_adapter or MOTIncrementalAdapter(
        transformer,
        history_frames=history_frames,
        chunk_size=chunk_size,
        window_size=window_size,
    )
    geometry_adapter = geometry_adapter or GeometryIncrementalAdapter(
        transformer,
        history_frames=history_frames,
        chunk_size=chunk_size,
        window_size=window_size,
    )
    schedulers = schedulers or build_rollout_schedulers(
        config,
        video_num_steps=video_num_steps,
        action_num_steps=action_num_steps,
    )
    if recorder is not None:
        recorder.validate(
            video_num_steps=video_num_steps,
            action_num_steps=action_num_steps,
        )

    def assert_video_phase(
        frame_id: int,
        source: CacheSource,
        version_id: int,
    ) -> None:
        barrier = getattr(mot_adapter, "assert_video_commit", None)
        if barrier is not None:
            barrier(
                state,
                frame_id=frame_id,
                source=source,
                version_id=version_id,
            )
        state.assert_cache_versions()

    def assert_geometry_phase(
        frame_id: int,
        source: CacheSource,
        version_id: int,
    ) -> None:
        barrier = getattr(geometry_adapter, "assert_geometry_commit", None)
        if barrier is not None:
            barrier(
                state,
                frame_id=frame_id,
                source=source,
                version_id=version_id,
            )
        state.assert_cache_versions()

    def assert_action_phase(
        frame_id: int,
        source: CacheSource,
        version_id: int,
    ) -> None:
        barrier = getattr(mot_adapter, "assert_action_commit", None)
        if barrier is not None:
            barrier(
                state,
                frame_id=frame_id,
                source=source,
                version_id=version_id,
            )
        state.assert_cache_versions()

    latents = batch["latents"].to(device=device)
    actions = batch["actions"].to(device=device)
    geometry_rgb = batch["geometry_rgb"].to(device=device)
    stream_ids = batch["stream_ids"].to(device=device)
    text_emb = batch["text_emb"].to(device=device)
    empty_text_emb = batch.get("empty_text_emb", empty_text_emb).to(device=device)
    working_batch = dict(batch)
    working_batch["video_latent_valid_mask"] = batch.get(
        "video_latent_valid_mask",
        torch.ones(latents.shape[0], latents.shape[2], dtype=torch.bool),
    ).to(device=device, dtype=torch.bool).clone()
    working_batch["geometry_group_valid_mask"] = batch.get(
        "geometry_group_valid_mask",
        torch.ones(
            geometry_rgb.shape[0],
            geometry_rgb.shape[1],
            geometry_rgb.shape[2],
            dtype=torch.bool,
        ),
    ).to(device=device, dtype=torch.bool).clone()
    working_batch["action_valid_mask"] = batch["action_valid_mask"].to(
        device=device,
        dtype=torch.bool,
    ).clone()
    target_latents = latents.clone()
    target_actions = actions.clone()
    target_geometry_rgb = geometry_rgb.clone()

    history_ids = list(range(history_frames))
    mot_adapter.commit_video(
        latents[:, :, :history_frames],
        frame_ids=history_ids,
        stream_ids=stream_ids,
        text_emb=text_emb,
        state=state,
        source=CacheSource.HISTORY,
        version_id=1,
        valid_frames=(
            working_batch["video_latent_valid_mask"][:, :history_frames]
        ),
    )
    for frame_id in history_ids:
        frame = state.frame(frame_id)
        frame.video_latent = latents[:, :, frame_id : frame_id + 1]
        frame.video_source = CacheSource.HISTORY
        frame.video_version = 1
    for frame_id in history_ids:
        assert_video_phase(frame_id, CacheSource.HISTORY, 1)
    for frame_id in history_ids:
        encoded = geometry_adapter.encode_and_commit(
            geometry_rgb[:, frame_id : frame_id + 1],
            frame_id=frame_id,
            slot_valid_mask=_frame_geometry_valid(working_batch, frame_id, device),
            state=state,
            source=CacheSource.HISTORY,
            version_id=1,
        )
        frame = state.frame(frame_id)
        frame.geometry_rgb = geometry_rgb[:, frame_id : frame_id + 1]
        frame.geometry_state = encoded
        frame.geometry_source = CacheSource.HISTORY
        frame.geometry_version = 1
        assert_geometry_phase(frame_id, CacheSource.HISTORY, 1)
    mot_adapter.commit_action(
        actions[:, :, :history_frames],
        frame_ids=history_ids,
        text_emb=text_emb,
        state=state,
        source=CacheSource.HISTORY,
        version_id=1,
        valid_mask=(
            working_batch["action_valid_mask"][:, :, :history_frames]
        ),
    )
    for frame_id in history_ids:
        frame = state.frame(frame_id)
        frame.action = actions[:, :, frame_id : frame_id + 1]
        frame.action_source = CacheSource.HISTORY
        frame.action_version = 1
    for frame_id in history_ids:
        assert_action_phase(frame_id, CacheSource.HISTORY, 1)

    mot_adapter.commit_video(
        latents[:, :, anchor : anchor + 1],
        frame_ids=[anchor],
        stream_ids=stream_ids,
        text_emb=text_emb,
        state=state,
        source=CacheSource.ANCHOR,
        version_id=1,
        valid_frames=_frame_video_valid(working_batch, anchor, device),
    )
    anchor_state = state.frame(anchor)
    anchor_state.video_latent = latents[:, :, anchor : anchor + 1]
    anchor_state.video_source = CacheSource.ANCHOR
    anchor_state.video_version = 1
    assert_video_phase(anchor, CacheSource.ANCHOR, 1)
    anchor_geometry = geometry_adapter.encode_and_commit(
        geometry_rgb[:, anchor : anchor + 1],
        frame_id=anchor,
        slot_valid_mask=_frame_geometry_valid(working_batch, anchor, device),
        state=state,
        source=CacheSource.ANCHOR,
        version_id=1,
    )
    anchor_state.geometry_rgb = geometry_rgb[:, anchor : anchor + 1]
    anchor_state.geometry_state = anchor_geometry
    anchor_state.geometry_source = CacheSource.ANCHOR
    anchor_state.geometry_version = 1
    assert_geometry_phase(anchor, CacheSource.ANCHOR, 1)
    mot_adapter.commit_action(
        actions[:, :, anchor : anchor + 1],
        frame_ids=[anchor],
        text_emb=text_emb,
        state=state,
        source=CacheSource.ANCHOR,
        version_id=1,
        valid_mask=_frame_action_valid(working_batch, anchor, device),
    )
    anchor_state.action = actions[:, :, anchor : anchor + 1]
    anchor_state.action_source = CacheSource.ANCHOR
    anchor_state.action_version = 1
    assert_action_phase(anchor, CacheSource.ANCHOR, 1)

    pending_ground_truth: dict[int, GroundTruthStep] = {}

    def predicted_geometry_for(frame_id: int, current_video: torch.Tensor) -> torch.Tensor:
        sequence = [state.frame(index).video_latent for index in range(anchor, frame_id)]
        if any(value is None for value in sequence):
            raise RuntimeError("target latent sequence is incomplete")
        target_latents = torch.cat([*sequence, current_video], dim=2)
        decoded = decode_latents_to_rgb_views(target_latents)
        groups = decoded_rgb_to_geometry_groups(
            decoded,
            latent_frames=target_latents.shape[2],
            vae_temporal_factor=int(config.vae_temporal_factor),
        )
        return groups[:, -1:].to(device=device, dtype=geometry_rgb.dtype)

    def sample_video(frame_id: int) -> torch.Tensor:
        reference = latents[:, :, frame_id : frame_id + 1]
        sample = torch.randn(
            reference.shape,
            device=device,
            dtype=reference.dtype,
            generator=state.generator,
        )
        guidance_scale = float(getattr(config, "guidance_scale", 1.0))
        for step_index, timestep in enumerate(schedulers.video.timesteps.to(device)):
            if recorder is not None:
                recorder.observe(
                    "video",
                    frame_id=frame_id,
                    step_index=step_index,
                    timestep=timestep,
                    sample=sample,
                )
            conditional = mot_adapter.predict_video(
                sample,
                timestep=timestep,
                frame_id=frame_id,
                stream_ids=stream_ids,
                text_emb=text_emb,
                state=state,
                valid_frames=_frame_video_valid(working_batch, frame_id, device),
            )
            if guidance_scale == 1.0:
                prediction = conditional
            else:
                unconditional = mot_adapter.predict_video(
                    sample,
                    timestep=timestep,
                    frame_id=frame_id,
                    stream_ids=stream_ids,
                    text_emb=empty_text_emb,
                    state=state,
                    valid_frames=_frame_video_valid(working_batch, frame_id, device),
                )
                prediction = unconditional + guidance_scale * (
                    conditional - unconditional
                )
            sample = schedulers.video.step(prediction, timestep, sample)
        return sample

    def sample_action(frame_id: int) -> torch.Tensor:
        if float(getattr(config, "action_guidance_scale", 1.0)) != 1.0:
            raise ValueError("self_rollout currently requires action_guidance_scale=1")
        reference = actions[:, :, frame_id : frame_id + 1]
        sample = torch.randn(
            reference.shape,
            device=device,
            dtype=reference.dtype,
            generator=state.generator,
        )
        valid = _frame_action_valid(working_batch, frame_id, device)
        if valid is not None:
            sample = sample * valid.to(dtype=sample.dtype)
        for step_index, timestep in enumerate(schedulers.action.timesteps.to(device)):
            if recorder is not None:
                recorder.observe(
                    "action",
                    frame_id=frame_id,
                    step_index=step_index,
                    timestep=timestep,
                    sample=sample,
                )
            prediction = mot_adapter.predict_action(
                sample,
                timestep=timestep,
                frame_id=frame_id,
                text_emb=text_emb,
                state=state,
                valid_mask=valid,
            )
            sample = schedulers.action.step(prediction, timestep, sample)
            if valid is not None:
                sample = sample * valid.to(dtype=sample.dtype)
        return sample

    def commit_frame(
        frame_id: int,
        *,
        video: torch.Tensor,
        geometry: torch.Tensor,
        action: torch.Tensor,
        video_source: CacheSource,
        geometry_source: CacheSource,
        action_source: CacheSource,
        video_version: int,
        geometry_version: int,
        action_version: int,
    ) -> None:
        mot_adapter.commit_video(
            video,
            frame_ids=[frame_id],
            stream_ids=stream_ids,
            text_emb=text_emb,
            state=state,
            source=video_source,
            version_id=video_version,
            valid_frames=_frame_video_valid(working_batch, frame_id, device),
        )
        frame = state.frame(frame_id)
        frame.video_latent = video
        frame.video_source = video_source
        frame.video_version = video_version
        assert_video_phase(frame_id, video_source, video_version)
        state.save_phase_checkpoint(frame_id, RolloutPhase.GEOMETRY)
        encoded = geometry_adapter.encode_and_commit(
            geometry,
            frame_id=frame_id,
            slot_valid_mask=_frame_geometry_valid(working_batch, frame_id, device),
            state=state,
            source=geometry_source,
            version_id=geometry_version,
        )
        frame.geometry_rgb = geometry
        frame.geometry_state = encoded
        frame.geometry_source = geometry_source
        frame.geometry_version = geometry_version
        assert_geometry_phase(frame_id, geometry_source, geometry_version)
        state.save_phase_checkpoint(frame_id, RolloutPhase.ACTION)
        mot_adapter.commit_action(
            action,
            frame_ids=[frame_id],
            text_emb=text_emb,
            state=state,
            source=action_source,
            version_id=action_version,
            valid_mask=_frame_action_valid(working_batch, frame_id, device),
        )
        frame.action = action
        frame.action_source = action_source
        frame.action_version = action_version
        assert_action_phase(frame_id, action_source, action_version)

    def replace_current(
        frame_id: int,
        gt: GroundTruthStep,
        *,
        prediction_generator_state: torch.Tensor,
        action_generator_state: torch.Tensor,
    ) -> None:
        if int(gt.frame_id) != frame_id:
            raise ValueError("replace_current frame mismatch")
        gt_geometry_rgb = gt.geometry_rgb
        if gt_geometry_rgb is None and gt.geometry_state is not None:
            gt_geometry_rgb = getattr(gt.geometry_state, "rgb", None)
            if gt_geometry_rgb is None:
                raise ValueError(
                    "ground_truth geometry_state must expose its source rgb so "
                    "strict-history K/V can be re-encoded"
                )

        if replacement_policy == "require_ground_truth":
            if gt.video_latent is not None and gt_geometry_rgb is None:
                raise ValueError(
                    "video replacement invalidates geometry; require_ground_truth "
                    "needs gt.geometry_rgb"
                )
            if (
                gt.video_latent is not None or gt_geometry_rgb is not None
            ) and gt.action is None:
                raise ValueError(
                    "video/geometry replacement invalidates action; "
                    "require_ground_truth needs gt.action"
                )

        def checked(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            if tuple(value.shape) != tuple(reference.shape):
                raise ValueError(
                    f"ground-truth {name} must be {tuple(reference.shape)}, "
                    f"got {tuple(value.shape)}"
                )
            return value.to(device=device, dtype=reference.dtype)

        video_reference = latents[:, :, frame_id : frame_id + 1]
        geometry_reference = geometry_rgb[:, frame_id : frame_id + 1]
        action_reference = actions[:, :, frame_id : frame_id + 1]
        old_frame = state.frame(frame_id)
        video_version = old_frame.video_version + int(gt.video_latent is not None)
        geometry_version = old_frame.geometry_version + int(
            gt.video_latent is not None or gt_geometry_rgb is not None
        )
        action_version = old_frame.action_version + int(
            gt.video_latent is not None
            or gt_geometry_rgb is not None
            or gt.action is not None
        )

        if gt.video_valid is not None:
            valid_reference = working_batch["video_latent_valid_mask"][
                :, frame_id : frame_id + 1
            ]
            working_batch["video_latent_valid_mask"][:, frame_id : frame_id + 1] = checked(
                "video_valid",
                gt.video_valid,
                valid_reference,
            ).to(dtype=torch.bool)
        if gt.action_valid is not None:
            valid_reference = working_batch["action_valid_mask"][
                :, :, frame_id : frame_id + 1
            ]
            working_batch["action_valid_mask"][:, :, frame_id : frame_id + 1] = checked(
                "action_valid",
                gt.action_valid,
                valid_reference,
            ).to(dtype=torch.bool)

        if gt.video_latent is not None:
            state.delete_predicted(frame_id, RolloutPhase.LATENT)
            state.generator.set_state(action_generator_state)
            video = checked("video_latent", gt.video_latent, video_reference)
            target_latents[:, :, frame_id : frame_id + 1] = video
            video_source = CacheSource.GROUND_TRUTH
            geometry = (
                checked("geometry_rgb", gt_geometry_rgb, geometry_reference)
                if gt_geometry_rgb is not None
                else predicted_geometry_for(frame_id, video)
            )
            geometry_source = (
                CacheSource.GROUND_TRUTH
                if gt_geometry_rgb is not None
                else CacheSource.PREDICTED
            )
            if gt_geometry_rgb is not None:
                target_geometry_rgb[:, frame_id : frame_id + 1] = geometry
            if gt.action is not None:
                action = checked("action", gt.action, action_reference)
                target_actions[:, :, frame_id : frame_id + 1] = action
                action_source = CacheSource.GROUND_TRUTH
            elif replacement_policy == "recompute_predicted":
                # V/G must be committed before action is recomputed.
                mot_adapter.commit_video(
                    video,
                    frame_ids=[frame_id],
                    stream_ids=stream_ids,
                    text_emb=text_emb,
                    state=state,
                    source=video_source,
                    version_id=video_version,
                    valid_frames=_frame_video_valid(working_batch, frame_id, device),
                )
                frame = state.frame(frame_id)
                frame.video_latent = video
                frame.video_source = video_source
                frame.video_version = video_version
                assert_video_phase(frame_id, video_source, video_version)
                state.save_phase_checkpoint(frame_id, RolloutPhase.GEOMETRY)
                encoded = geometry_adapter.encode_and_commit(
                    geometry,
                    frame_id=frame_id,
                    slot_valid_mask=_frame_geometry_valid(working_batch, frame_id, device),
                    state=state,
                    source=geometry_source,
                    version_id=geometry_version,
                )
                frame.geometry_rgb = geometry
                frame.geometry_state = encoded
                frame.geometry_source = geometry_source
                frame.geometry_version = geometry_version
                assert_geometry_phase(frame_id, geometry_source, geometry_version)
                state.save_phase_checkpoint(frame_id, RolloutPhase.ACTION)
                action = sample_action(frame_id)
                action_source = CacheSource.PREDICTED
                mot_adapter.commit_action(
                    action,
                    frame_ids=[frame_id],
                    text_emb=text_emb,
                    state=state,
                    source=action_source,
                    version_id=action_version,
                    valid_mask=_frame_action_valid(working_batch, frame_id, device),
                )
                frame.action = action
                frame.action_source = action_source
                frame.action_version = action_version
                assert_action_phase(frame_id, action_source, action_version)
                action = None
            else:
                raise ValueError(
                    "video replacement invalidates action; provide gt.action or "
                    "use replacement_policy='recompute_predicted'"
                )
            if action is not None:
                commit_frame(
                    frame_id,
                    video=video,
                    geometry=geometry,
                    action=action,
                    video_source=video_source,
                    geometry_source=geometry_source,
                    action_source=action_source,
                    video_version=video_version,
                    geometry_version=geometry_version,
                    action_version=action_version,
                )
        elif gt_geometry_rgb is not None:
            state.delete_predicted(frame_id, RolloutPhase.GEOMETRY)
            frame = state.frame(frame_id)
            geometry = checked("geometry_rgb", gt_geometry_rgb, geometry_reference)
            target_geometry_rgb[:, frame_id : frame_id + 1] = geometry
            encoded = geometry_adapter.encode_and_commit(
                geometry,
                frame_id=frame_id,
                slot_valid_mask=_frame_geometry_valid(working_batch, frame_id, device),
                state=state,
                source=CacheSource.GROUND_TRUTH,
                version_id=geometry_version,
            )
            frame.geometry_rgb = geometry
            frame.geometry_state = encoded
            frame.geometry_source = CacheSource.GROUND_TRUTH
            frame.geometry_version = geometry_version
            assert_geometry_phase(
                frame_id,
                CacheSource.GROUND_TRUTH,
                geometry_version,
            )
            state.save_phase_checkpoint(frame_id, RolloutPhase.ACTION)
            if gt.action is not None:
                action = checked("action", gt.action, action_reference)
                target_actions[:, :, frame_id : frame_id + 1] = action
                action_source = CacheSource.GROUND_TRUTH
            elif replacement_policy == "recompute_predicted":
                action = sample_action(frame_id)
                action_source = CacheSource.PREDICTED
            else:
                raise ValueError(
                    "geometry replacement invalidates action; provide gt.action or "
                    "use replacement_policy='recompute_predicted'"
                )
            mot_adapter.commit_action(
                action,
                frame_ids=[frame_id],
                text_emb=text_emb,
                state=state,
                source=action_source,
                version_id=action_version,
                valid_mask=_frame_action_valid(working_batch, frame_id, device),
            )
            frame.action = action
            frame.action_source = action_source
            frame.action_version = action_version
            assert_action_phase(frame_id, action_source, action_version)
        elif gt.action is not None:
            state.delete_predicted(frame_id, RolloutPhase.ACTION)
            action = checked("action", gt.action, action_reference)
            target_actions[:, :, frame_id : frame_id + 1] = action
            mot_adapter.commit_action(
                action,
                frame_ids=[frame_id],
                text_emb=text_emb,
                state=state,
                source=CacheSource.GROUND_TRUTH,
                version_id=action_version,
                valid_mask=_frame_action_valid(working_batch, frame_id, device),
            )
            frame = state.frame(frame_id)
            frame.action = action
            frame.action_source = CacheSource.GROUND_TRUTH
            frame.action_version = action_version
            assert_action_phase(
                frame_id,
                CacheSource.GROUND_TRUTH,
                action_version,
            )
        state.generator.set_state(prediction_generator_state)
        state.predictions.replacements.append(
            {
                "frame_id": frame_id,
                "components": [
                    name
                    for name, value in (
                        ("video", gt.video_latent),
                        ("geometry", gt_geometry_rgb),
                        ("action", gt.action),
                    )
                    if value is not None
                ],
                "policy": replacement_policy,
                "caused_by_action_frame_id": gt.caused_by_action_frame_id,
            }
        )

    for frame_id in range(anchor + 1, end_frame + 1):
        state.save_checkpoint_before(frame_id)
        try:
            predicted_video = sample_video(frame_id)
            state.predictions.video[frame_id] = predicted_video.detach().clone()
            mot_adapter.commit_video(
                predicted_video,
                frame_ids=[frame_id],
                stream_ids=stream_ids,
                text_emb=text_emb,
                state=state,
                source=CacheSource.PREDICTED,
                version_id=1,
                valid_frames=_frame_video_valid(working_batch, frame_id, device),
            )
            frame = state.frame(frame_id)
            frame.video_latent = predicted_video
            frame.video_source = CacheSource.PREDICTED
            frame.video_version = 1
            assert_video_phase(frame_id, CacheSource.PREDICTED, 1)
            state.save_phase_checkpoint(frame_id, RolloutPhase.GEOMETRY)

            predicted_geometry = predicted_geometry_for(frame_id, predicted_video)
            state.predictions.geometry[frame_id] = predicted_geometry.detach().clone()
            encoded = geometry_adapter.encode_and_commit(
                predicted_geometry,
                frame_id=frame_id,
                slot_valid_mask=_frame_geometry_valid(working_batch, frame_id, device),
                state=state,
                source=CacheSource.PREDICTED,
                version_id=1,
            )
            frame.geometry_rgb = predicted_geometry
            frame.geometry_state = encoded
            frame.geometry_source = CacheSource.PREDICTED
            frame.geometry_version = 1
            assert_geometry_phase(frame_id, CacheSource.PREDICTED, 1)
            state.save_phase_checkpoint(frame_id, RolloutPhase.ACTION)

            predicted_action = sample_action(frame_id)
            state.predictions.action[frame_id] = predicted_action.detach().clone()
            mot_adapter.commit_action(
                predicted_action,
                frame_ids=[frame_id],
                text_emb=text_emb,
                state=state,
                source=CacheSource.PREDICTED,
                version_id=1,
                valid_mask=_frame_action_valid(working_batch, frame_id, device),
            )
            frame.action = predicted_action
            frame.action_source = CacheSource.PREDICTED
            frame.action_version = 1
            assert_action_phase(frame_id, CacheSource.PREDICTED, 1)
            prediction_generator_state = state.generator.get_state().clone()
            action_snapshot = state.phase_checkpoints[
                (frame_id, RolloutPhase.ACTION)
            ]
            if action_snapshot.generator_state is None:
                raise RuntimeError("ACTION checkpoint is missing generator state")
            action_generator_state = action_snapshot.generator_state.clone()

            gt = pending_ground_truth.pop(frame_id, None)
            if gt is None and ground_truth_provider is not None:
                gt = ground_truth_provider.maybe_get(
                    frame_id=frame_id,
                    predicted_action=predicted_action,
                    state=state.public_view(),
                )
            if gt is not None:
                if gt.frame_id == frame_id:
                    replace_current(
                        frame_id,
                        gt,
                        prediction_generator_state=prediction_generator_state,
                        action_generator_state=action_generator_state,
                    )
                elif gt.frame_id > frame_id:
                    pending_ground_truth[int(gt.frame_id)] = gt
                else:
                    raise ValueError(
                        "provider returned a past frame; online replay of already-consumed "
                        "future frames is not supported in the first implementation"
                    )
        except Exception:
            state.restore_before(frame_id)
            raise

    horizon = end_frame + 1
    if recorder is not None:
        recorder.require_frames(list(range(anchor + 1, end_frame + 1)))
    pred_latents = latents[:, :, :horizon].clone()
    pred_actions = actions[:, :, :horizon].clone()
    pred_geometry = geometry_rgb[:, :horizon].clone()
    for frame_id, value in state.predictions.video.items():
        pred_latents[:, :, frame_id : frame_id + 1] = value
    for frame_id, value in state.predictions.action.items():
        pred_actions[:, :, frame_id : frame_id + 1] = value
    for frame_id, value in state.predictions.geometry.items():
        pred_geometry[:, frame_id : frame_id + 1] = value

    sources = {
        frame_id: {
            "video": _source_name(frame.video_source),
            "geometry": _source_name(frame.geometry_source),
            "action": _source_name(frame.action_source),
        }
        for frame_id, frame in state.semantic_frames.items()
    }
    return RolloutResult(
        pred_latents=pred_latents,
        target_latents=target_latents[:, :, :horizon].clone(),
        pred_actions=pred_actions,
        target_actions=target_actions[:, :, :horizon].clone(),
        pred_geometry_rgb=pred_geometry,
        target_geometry_rgb=target_geometry_rgb[:, :horizon].clone(),
        action_valid_mask=working_batch["action_valid_mask"][:, :, :horizon].clone(),
        chunk_pairs=1,
        chunk_frames=int(spec.latent_frames_per_action_chunk_per_view),
        diagnostics={
            "sources": sources,
            "versions": {
                frame_id: {
                    "video": frame.video_version,
                    "geometry": frame.geometry_version,
                    "action": frame.action_version,
                }
                for frame_id, frame in state.semantic_frames.items()
            },
            "replacements": state.predictions.replacements,
            "profile": "segmented_history_strict_geometry_v1",
            "profile_version": 1,
            "rollout_frames": rollout_frames,
            "pending_ground_truth_frames": sorted(pending_ground_truth),
            "mot_cache_tokens": {
                layer_id: state.mot_cache.committed_token_count(layer_id)
                for layer_id in state.mot_cache.layer_ids
            },
            "geometry_cache_tokens": {
                layer_id: geometry_cache.relation_cache.committed_token_count(layer_id)
                for layer_id in geometry_cache.relation_cache.layer_ids
            },
        },
    )
