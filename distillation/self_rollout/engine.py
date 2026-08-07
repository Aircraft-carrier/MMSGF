"""End-to-end Video -> Action incremental self rollout."""
from __future__ import annotations

from typing import Any

import torch

from distillation.model.autoregressive_types import AutoregressiveModelRequest

from .provider import GroundTruthProvider, GroundTruthStep
from .recorder import SelfRolloutRecorder
from .result import RolloutResult
from .scheduler import RolloutSchedulers, build_rollout_schedulers
from .state import CacheSource, RolloutPhase, RolloutState


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


def _source_name(source: CacheSource | None) -> str | None:
    return None if source is None else source.name.lower()


def _validate_rollout_inputs(
    batch: dict[str, Any],
    *,
    spec: Any,
    rollout_frames: int,
) -> None:
    required = {"latents", "actions", "action_valid_mask", "stream_ids", "text_emb"}
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"self_rollout batch is missing {missing}")
    for name in required:
        if not torch.is_tensor(batch[name]):
            raise TypeError(f"self_rollout batch[{name!r}] must be a tensor")

    latents = batch["latents"]
    actions = batch["actions"]
    action_valid = batch["action_valid_mask"]
    stream_ids = batch["stream_ids"]
    if latents.ndim != 6:
        raise ValueError(f"latents must be [B,C,F,V,H,W], got {tuple(latents.shape)}")
    if actions.ndim != 5:
        raise ValueError(f"actions must be [B,C,F,N,D], got {tuple(actions.shape)}")
    batch_size, _channels, frames, views = latents.shape[:4]
    if actions.shape[0] != batch_size or actions.shape[2] != frames:
        raise ValueError("latent/action batch and frame axes must match")
    if tuple(stream_ids.shape) != (batch_size, views):
        raise ValueError(
            f"stream_ids must be [{batch_size},{views}], got {tuple(stream_ids.shape)}"
        )
    if batch["text_emb"].shape[0] not in {1, batch_size}:
        raise ValueError("text_emb batch must be 1 or match latent batch size")
    expected_action_valid = (batch_size, actions.shape[1], frames, actions.shape[3], 1)
    if tuple(action_valid.shape) != expected_action_valid:
        raise ValueError(
            f"action_valid_mask must be {expected_action_valid}, got {tuple(action_valid.shape)}"
        )
    if "video_latent_valid_mask" in batch and tuple(
        batch["video_latent_valid_mask"].shape
    ) != (batch_size, frames):
        raise ValueError(f"video_latent_valid_mask must be [{batch_size},{frames}]")

    anchor = int(spec.history_latent_frames)
    end_frame = anchor + int(rollout_frames)
    if anchor >= frames:
        raise ValueError("batch does not contain the target anchor frame")
    if end_frame >= frames:
        raise ValueError(f"rollout needs frames through {end_frame}, but batch has {frames}")


@torch.no_grad()
def self_rollout(
    batch: dict[str, Any],
    *,
    transformer,
    config: Any,
    spec: Any,
    device: torch.device,
    empty_text_emb: torch.Tensor,
    video_num_steps: int,
    action_num_steps: int,
    rollout_frames: int,
    ground_truth_provider: GroundTruthProvider | None = None,
    replacement_policy: str = "require_ground_truth",
    generator: torch.Generator | None = None,
    schedulers: RolloutSchedulers | None = None,
    recorder: SelfRolloutRecorder | None = None,
) -> RolloutResult:
    """Run a distillation-owned incremental Video+Action rollout."""

    rollout_frames = int(rollout_frames)
    if rollout_frames <= 0:
        raise ValueError(f"rollout_frames must be positive, got {rollout_frames}")
    if replacement_policy not in {"require_ground_truth", "recompute_predicted"}:
        raise ValueError(f"unsupported replacement_policy={replacement_policy!r}")
    _validate_rollout_inputs(batch, spec=spec, rollout_frames=rollout_frames)

    history_frames = int(spec.history_latent_frames)
    anchor = history_frames
    end_frame = anchor + rollout_frames
    shape = config.distill.generation_shape
    profile_name = str(shape.get("profile_name", "segmented_history_va_v1"))
    if profile_name != "segmented_history_va_v1":
        raise ValueError(f"unsupported self_rollout profile {profile_name!r}")
    if int(shape.get("history_frames", history_frames)) != history_frames:
        raise ValueError("generation_shape.history_frames must match MOT spec")
    if str(shape.get("order_mode", "segmented")) != "segmented":
        raise ValueError("self_rollout requires generation_shape.order_mode='segmented'")

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(getattr(config, "seed", 0)))
    state = RolloutState(generator=generator)
    native_model = getattr(transformer, "module", transformer)
    if not hasattr(native_model, "forward_autoregressive"):
        raise TypeError(
            "self_rollout requires AutoregressiveVAMOTTransformer3DModel"
        )

    def invoke_model(operation: str, **payload):
        return transformer(
            AutoregressiveModelRequest(operation=operation, payload=payload),
            mode="self_rollout",
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

    latents = batch["latents"].to(device=device)
    actions = batch["actions"].to(device=device)
    stream_ids = batch["stream_ids"].to(device=device)
    text_emb = batch["text_emb"].to(device=device)
    empty_text_emb = batch.get("empty_text_emb", empty_text_emb).to(device=device)
    working_batch = dict(batch)
    working_batch["video_latent_valid_mask"] = batch.get(
        "video_latent_valid_mask",
        torch.ones(latents.shape[0], latents.shape[2], dtype=torch.bool),
    ).to(device=device, dtype=torch.bool).clone()
    working_batch["action_valid_mask"] = batch["action_valid_mask"].to(
        device=device, dtype=torch.bool
    ).clone()
    target_latents = latents.clone()
    target_actions = actions.clone()

    def assert_phase(name: str, frame_id: int, source: CacheSource, version: int):
        getattr(native_model, name)(
            state,
            frame_id=frame_id,
            source=source,
            version_id=version,
        )
        state.assert_cache_versions()

    def commit_video(
        frame_id: int,
        video: torch.Tensor,
        *,
        source: CacheSource,
        version: int,
        save_action_checkpoint: bool = True,
    ) -> None:
        invoke_model(
            "commit_video",
            latents=video,
            frame_ids=[frame_id],
            stream_ids=stream_ids,
            text_emb=text_emb,
            state=state,
            source=source,
            version_id=version,
            valid_frames=_frame_video_valid(working_batch, frame_id, device),
        )
        frame = state.frame(frame_id)
        frame.video_latent = video
        frame.video_source = source
        frame.video_version = version
        assert_phase("assert_video_commit", frame_id, source, version)
        if save_action_checkpoint:
            state.save_phase_checkpoint(frame_id, RolloutPhase.ACTION)

    def commit_action(
        frame_id: int,
        action: torch.Tensor,
        *,
        source: CacheSource,
        version: int,
    ) -> None:
        invoke_model(
            "commit_action",
            actions=action,
            frame_ids=[frame_id],
            text_emb=text_emb,
            state=state,
            source=source,
            version_id=version,
            valid_mask=_frame_action_valid(working_batch, frame_id, device),
        )
        frame = state.frame(frame_id)
        frame.action = action
        frame.action_source = source
        frame.action_version = version
        assert_phase("assert_action_commit", frame_id, source, version)

    history_ids = list(range(history_frames))
    invoke_model(
        "commit_video",
        latents=latents[:, :, :history_frames],
        frame_ids=history_ids,
        stream_ids=stream_ids,
        text_emb=text_emb,
        state=state,
        source=CacheSource.HISTORY,
        version_id=1,
        valid_frames=working_batch["video_latent_valid_mask"][:, :history_frames],
    )
    invoke_model(
        "commit_action",
        actions=actions[:, :, :history_frames],
        frame_ids=history_ids,
        text_emb=text_emb,
        state=state,
        source=CacheSource.HISTORY,
        version_id=1,
        valid_mask=working_batch["action_valid_mask"][:, :, :history_frames],
    )
    for frame_id in history_ids:
        frame = state.frame(frame_id)
        frame.video_latent = latents[:, :, frame_id : frame_id + 1]
        frame.action = actions[:, :, frame_id : frame_id + 1]
        frame.video_source = frame.action_source = CacheSource.HISTORY
        frame.video_version = frame.action_version = 1
        assert_phase("assert_video_commit", frame_id, CacheSource.HISTORY, 1)
        assert_phase("assert_action_commit", frame_id, CacheSource.HISTORY, 1)

    commit_video(
        anchor,
        latents[:, :, anchor : anchor + 1],
        source=CacheSource.ANCHOR,
        version=1,
        save_action_checkpoint=False,
    )
    commit_action(
        anchor,
        actions[:, :, anchor : anchor + 1],
        source=CacheSource.ANCHOR,
        version=1,
    )

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
            conditional = invoke_model(
                "predict_video",
                sample=sample,
                timestep=timestep,
                frame_id=frame_id,
                stream_ids=stream_ids,
                text_emb=text_emb,
                state=state,
                valid_frames=_frame_video_valid(working_batch, frame_id, device),
            ).prediction
            prediction = conditional
            if guidance_scale != 1.0:
                unconditional = invoke_model(
                    "predict_video",
                    sample=sample,
                    timestep=timestep,
                    frame_id=frame_id,
                    stream_ids=stream_ids,
                    text_emb=empty_text_emb,
                    state=state,
                    valid_frames=_frame_video_valid(working_batch, frame_id, device),
                ).prediction
                prediction = unconditional + guidance_scale * (conditional - unconditional)
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
            prediction = invoke_model(
                "predict_action",
                sample=sample,
                timestep=timestep,
                frame_id=frame_id,
                text_emb=text_emb,
                state=state,
                valid_mask=valid,
            ).prediction
            sample = schedulers.action.step(prediction, timestep, sample)
            if valid is not None:
                sample = sample * valid.to(dtype=sample.dtype)
        return sample

    def checked(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"ground-truth {name} must be {tuple(reference.shape)}, got {tuple(value.shape)}"
            )
        return value.to(device=device, dtype=reference.dtype)

    def replace_current(
        frame_id: int,
        gt: GroundTruthStep,
        *,
        prediction_generator_state: torch.Tensor,
        action_generator_state: torch.Tensor,
    ) -> None:
        if int(gt.frame_id) != frame_id:
            raise ValueError("replace_current frame mismatch")
        if replacement_policy == "require_ground_truth" and (
            gt.video_latent is not None and gt.action is None
        ):
            raise ValueError("video replacement requires gt.action")

        if gt.video_valid is not None:
            reference = working_batch["video_latent_valid_mask"][:, frame_id : frame_id + 1]
            working_batch["video_latent_valid_mask"][:, frame_id : frame_id + 1] = checked(
                "video_valid", gt.video_valid, reference
            ).to(dtype=torch.bool)
        if gt.action_valid is not None:
            reference = working_batch["action_valid_mask"][:, :, frame_id : frame_id + 1]
            working_batch["action_valid_mask"][:, :, frame_id : frame_id + 1] = checked(
                "action_valid", gt.action_valid, reference
            ).to(dtype=torch.bool)

        frame = state.frame(frame_id)
        video_version = frame.video_version + int(gt.video_latent is not None)
        action_version = frame.action_version + int(
            gt.video_latent is not None or gt.action is not None
        )
        if gt.video_latent is not None:
            state.delete_predicted(frame_id, RolloutPhase.LATENT)
            state.generator.set_state(action_generator_state)
            video = checked(
                "video_latent", gt.video_latent, latents[:, :, frame_id : frame_id + 1]
            )
            target_latents[:, :, frame_id : frame_id + 1] = video
            commit_video(
                frame_id,
                video,
                source=CacheSource.GROUND_TRUTH,
                version=video_version,
            )
            if gt.action is None:
                action = sample_action(frame_id)
                action_source = CacheSource.PREDICTED
            else:
                action = checked(
                    "action", gt.action, actions[:, :, frame_id : frame_id + 1]
                )
                target_actions[:, :, frame_id : frame_id + 1] = action
                action_source = CacheSource.GROUND_TRUTH
            commit_action(frame_id, action, source=action_source, version=action_version)
        elif gt.action is not None:
            state.delete_predicted(frame_id, RolloutPhase.ACTION)
            action = checked("action", gt.action, actions[:, :, frame_id : frame_id + 1])
            target_actions[:, :, frame_id : frame_id + 1] = action
            commit_action(
                frame_id,
                action,
                source=CacheSource.GROUND_TRUTH,
                version=action_version,
            )
        state.generator.set_state(prediction_generator_state)
        state.predictions.replacements.append(
            {
                "frame_id": frame_id,
                "components": [
                    name
                    for name, value in (("video", gt.video_latent), ("action", gt.action))
                    if value is not None
                ],
                "policy": replacement_policy,
                "caused_by_action_frame_id": gt.caused_by_action_frame_id,
            }
        )

    pending_ground_truth: dict[int, GroundTruthStep] = {}
    for frame_id in range(anchor + 1, end_frame + 1):
        state.save_checkpoint_before(frame_id)
        try:
            predicted_video = sample_video(frame_id)
            state.predictions.video[frame_id] = predicted_video.detach().clone()
            commit_video(
                frame_id,
                predicted_video,
                source=CacheSource.PREDICTED,
                version=1,
            )
            predicted_action = sample_action(frame_id)
            state.predictions.action[frame_id] = predicted_action.detach().clone()
            commit_action(
                frame_id,
                predicted_action,
                source=CacheSource.PREDICTED,
                version=1,
            )
            prediction_generator_state = state.generator.get_state().clone()
            action_snapshot = state.phase_checkpoints[(frame_id, RolloutPhase.ACTION)]
            if action_snapshot.generator_state is None:
                raise RuntimeError("ACTION checkpoint is missing generator state")

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
                        action_generator_state=action_snapshot.generator_state.clone(),
                    )
                elif gt.frame_id > frame_id:
                    pending_ground_truth[int(gt.frame_id)] = gt
                else:
                    raise ValueError("provider returned a past frame")
        except Exception:
            state.restore_before(frame_id)
            raise

    horizon = end_frame + 1
    if recorder is not None:
        recorder.require_frames(list(range(anchor + 1, end_frame + 1)))
    pred_latents = latents[:, :, :horizon].clone()
    pred_actions = actions[:, :, :horizon].clone()
    for frame_id, value in state.predictions.video.items():
        pred_latents[:, :, frame_id : frame_id + 1] = value
    for frame_id, value in state.predictions.action.items():
        pred_actions[:, :, frame_id : frame_id + 1] = value

    sources = {
        frame_id: {
            "video": _source_name(frame.video_source),
            "action": _source_name(frame.action_source),
        }
        for frame_id, frame in state.semantic_frames.items()
    }
    return RolloutResult(
        pred_latents=pred_latents,
        target_latents=target_latents[:, :, :horizon].clone(),
        pred_actions=pred_actions,
        target_actions=target_actions[:, :, :horizon].clone(),
        action_valid_mask=working_batch["action_valid_mask"][:, :, :horizon].clone(),
        chunk_pairs=1,
        chunk_frames=int(spec.latent_frames_per_action_chunk_per_view),
        diagnostics={
            "sources": sources,
            "versions": {
                frame_id: {
                    "video": frame.video_version,
                    "action": frame.action_version,
                }
                for frame_id, frame in state.semantic_frames.items()
            },
            "replacements": state.predictions.replacements,
            "profile": "segmented_history_va_v1",
            "profile_version": 2,
            "rollout_frames": rollout_frames,
            "pending_ground_truth_frames": sorted(pending_ground_truth),
            "mot_cache_tokens": {
                layer_id: state.mot_cache.committed_token_count(layer_id)
                for layer_id in state.mot_cache.layer_ids
            },
        },
    )
