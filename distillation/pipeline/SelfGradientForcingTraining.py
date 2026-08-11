"""Incremental Self Gradient Forcing training rollout pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence

import torch
import torch.distributed as dist

from distillation.schema import DenoisyInterval, VAPair

from .base_pipeline import BasePipeline

if TYPE_CHECKING:
    from distillation.model.wan_wrapper import WanDiffusionWrapper


class DenoisingStepList(Protocol):
    """Linear denoising progress d, one strictly descending list per modality."""

    video: Sequence[int]
    action: Sequence[int]


@dataclass(frozen=True, slots=True)
class RecordedDenoiseState:
    sample: torch.Tensor
    timestep: torch.Tensor
    exit_id: int
    denoisy_from: float
    denoisy_to: float


class SelfRolloutRecorder:
    """Record the exit-step noisy input and its interval per modality/frame."""

    def __init__(self, *, video: DenoisyInterval, action: DenoisyInterval) -> None:
        self.intervals = {"video": video, "action": action}
        self.records: dict[tuple[str, int], RecordedDenoiseState] = {}

    def observe(
        self,
        modality: Literal["video", "action"],
        *,
        frame_id: int,
        step_index: int,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> None:
        interval = self.intervals[modality]
        if step_index != interval.exit_id:
            return
        self.records[(modality, int(frame_id))] = RecordedDenoiseState(
            sample=sample.detach().clone(),
            timestep=timestep.detach().clone(),
            exit_id=interval.exit_id,
            denoisy_from=interval.denoisy_from,
            denoisy_to=interval.denoisy_to,
        )

    def require_frames(self, frame_ids: list[int]) -> None:
        expected = set(frame_ids)
        for modality in self.intervals:
            recorded = {frame for m, frame in self.records if m == modality}
            missing = sorted(expected - recorded)
            if missing:
                raise RuntimeError(
                    f"self rollout did not record {modality} frames {missing}"
                )


@dataclass(frozen=True)
class RolloutResult:
    """Predicted rollout content and organized exit information.

    ``predicted_clean`` covers only ``pred_frame_ids``; no GT tensors are kept.
    ``noisy_at_t`` contains the exit-step noisy inputs, and the denoised timestep
    interval/exits summarize the recorder state.
    """

    pred_frame_ids: tuple[int, ...]
    predicted_clean: VAPair
    noisy_at_t: VAPair
    video_exit_timestep: float
    action_exit_timestep: float
    video_exit_id: int
    action_exit_id: int
    video_denoised_timestep_from: float
    video_denoised_timestep_to: float
    action_denoised_timestep_from: float
    action_denoised_timestep_to: float


class SelfGradientForcingTrainingPipeline(BasePipeline):
    """No-grad incremental Video+Action rollout mirroring the reference SGF pipeline.

    The constructor follows
    ``Self_Gradient_Forcing/pipeline/self_gradient_forcing_training.py``:

    - ``denoising_step_list``: container with independent video/action linear
      denoising progress d lists (``.video`` / ``.action`` or dict keys).
    - ``scheduler``: FlowMatchScheduler used to warp/re-noise the video stream;
      the action stream uses ``generator.action_scheduler``.
    - ``generator``: wrapper exposing ``generate_video/generate_action`` (x0) and
      ``commit_video/commit_action``.
    - ``num_frame_per_block``: frames denoised together in one diffusion call.
    - ``per_rank_exit_step``: sample the exit id locally per rank (True) or
      broadcast a rank-0 sample (False). Video and action sample their own id.
    """

    def __init__(
        self,
        denoising_step_list: DenoisingStepList | dict[str, Sequence[int]],
        generator: WanDiffusionWrapper,
        *,
        num_frame_per_block: int = 1,
        per_rank_exit_step: bool = True,
    ) -> None:
        super().__init__(generator, num_frame_per_block=num_frame_per_block)
        if isinstance(denoising_step_list, dict):
            video_steps = denoising_step_list["video"]
            action_steps = denoising_step_list["action"]
        else:
            video_steps = denoising_step_list.video
            action_steps = denoising_step_list.action
        self.video_denoising_step_list = self._normalize_steps(video_steps)
        self.action_denoising_step_list = self._normalize_steps(action_steps)
        self.scheduler = generator.video_scheduler
        self.action_scheduler = generator.action_scheduler
        self.per_rank_exit_step = bool(per_rank_exit_step)

    @staticmethod
    def _normalize_steps(values) -> tuple[int, ...]:
        steps = tuple(int(value) for value in values)
        if steps and steps[-1] == 0:
            steps = steps[:-1]
        return steps

    def _sample_exit_id(self, num_steps: int, device: torch.device) -> int:
        if self.per_rank_exit_step:
            return int(torch.randint(num_steps, (), device=device).item())
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                index = torch.randint(num_steps, (1,), device=device)
            else:
                index = torch.empty(1, dtype=torch.long, device=device)
            dist.broadcast(index, src=0)
            return int(index.item())
        return int(torch.randint(num_steps, (1,), device=device).item())

    @staticmethod
    def _interval(steps: tuple[int, ...], exit_id: int) -> DenoisyInterval:
        return DenoisyInterval(
            exit_id=exit_id,
            denoisy_from=float(steps[exit_id]),
            denoisy_to=(
                float(steps[exit_id + 1])
                if exit_id + 1 < len(steps)
                else 0.0
            ),
        )

    def _make_recorder(
        self,
        video_exit_id: int,
        action_exit_id: int,
    ) -> SelfRolloutRecorder:
        return SelfRolloutRecorder(
            video=self._interval(self.video_denoising_step_list, video_exit_id),
            action=self._interval(self.action_denoising_step_list, action_exit_id),
        )

    @staticmethod
    def _record_block(
        records: dict[int, torch.Tensor],
        block: torch.Tensor,
        frame_ids: list[int],
    ) -> None:
        for offset, frame_id in enumerate(frame_ids):
            records[int(frame_id)] = block[:, :, offset : offset + 1].detach().clone()

    def _sample_video(
        self,
        frame_ids: list[int],
        steps: torch.Tensor,
        exit_id: int,
        recorder: SelfRolloutRecorder,
        *,
        latents: torch.Tensor,
        stream_ids: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        start = int(frame_ids[0])
        sample = torch.randn(
            (latents.shape[0], latents.shape[1], len(frame_ids), *latents.shape[3:]),
            device=latents.device,
            dtype=latents.dtype,
        )
        for step_index, timestep in enumerate(steps):
            if step_index == exit_id:
                for offset, frame_id in enumerate(frame_ids):
                    recorder.observe(
                        "video",
                        frame_id=frame_id,
                        step_index=step_index,
                        timestep=timestep,
                        sample=sample[:, :, offset : offset + 1],
                    )
            x0 = self.generator.generate_video(
                sample,
                timestep,
                frame_ids,
                stream_ids,
                cache=self.cache,
                text_emb=text_emb,
            )
            if step_index + 1 == len(steps):
                sample = x0
            else:
                sample = self.scheduler.add_noise(
                    x0,
                    torch.randn_like(x0),
                    steps[step_index + 1],
                )
        return sample

    def _sample_action(
        self,
        frame_ids: list[int],
        steps: torch.Tensor,
        exit_id: int,
        recorder: SelfRolloutRecorder,
        *,
        actions: torch.Tensor,
        valid_mask: torch.Tensor | None,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        start = int(frame_ids[0])
        sample = torch.randn(
            (actions.shape[0], actions.shape[1], len(frame_ids), *actions.shape[3:]),
            device=actions.device,
            dtype=actions.dtype,
        )
        if valid_mask is not None:
            sample = sample * valid_mask[:, :, start : start + len(frame_ids)].to(
                dtype=sample.dtype
            )
        for step_index, timestep in enumerate(steps):
            if step_index == exit_id:
                for offset, frame_id in enumerate(frame_ids):
                    recorder.observe(
                        "action",
                        frame_id=frame_id,
                        step_index=step_index,
                        timestep=timestep,
                        sample=sample[:, :, offset : offset + 1],
                    )
            x0 = self.generator.generate_action(
                sample,
                timestep,
                frame_ids,
                cache=self.cache,
                text_emb=text_emb,
            )
            if step_index + 1 == len(steps):
                sample = x0
            else:
                sample = self.generator.action_scheduler.add_noise(
                    x0,
                    torch.randn_like(x0),
                    steps[step_index + 1],
                )
            if valid_mask is not None:
                sample = sample * valid_mask[:, :, start : start + len(frame_ids)].to(
                    dtype=sample.dtype
                )
        return sample

    @torch.no_grad()
    def generate(
        self,
        batch: dict[str, Any],
        *,
        rollout_frames: int,
        history_frames: int,
        device: torch.device,
        recorder: SelfRolloutRecorder | None = None,
    ) -> RolloutResult:
        """Run one no-grad incremental rollout and optionally record exit states."""
        # dmd.py owns the shared warp function but also constructs this pipeline.
        # Import at execution time to avoid a module-initialization cycle.
        from distillation.model.dmd import warp_denoisy_progress

        rollout_frames = int(rollout_frames)
        history_frames = int(history_frames)

        latents = batch["latents"].to(device=device)
        actions = batch["actions"].to(device=device)
        stream_ids = batch["stream_ids"].to(device=device)
        text_emb = batch["text_emb"].to(device=device)
        action_valid = batch.get("action_valid_mask")
        if action_valid is not None:
            action_valid = action_valid.to(device=device, dtype=torch.bool)

        video_steps = warp_denoisy_progress(
            torch.tensor(self.video_denoising_step_list, dtype=torch.float32),
            self.scheduler,
        ).to(device=device)
        action_steps = warp_denoisy_progress(
            torch.tensor(self.action_denoising_step_list, dtype=torch.float32),
            self.action_scheduler,
        ).to(device=device)
        video_exit_id = self._sample_exit_id(len(video_steps), device)
        action_exit_id = self._sample_exit_id(len(action_steps), device)
        if recorder is None:
            recorder = self._make_recorder(video_exit_id, action_exit_id)

        self.reset_cache()
        self.build_history_cache(
            batch,
            history_frames=history_frames,
            device=device,
        )

        anchor = history_frames
        generated_start = anchor + 1
        generated_end = generated_start + rollout_frames
        predicted_video: dict[int, torch.Tensor] = {}
        predicted_action: dict[int, torch.Tensor] = {}
        for start in range(
            generated_start,
            generated_end,
            self.num_frame_per_block,
        ):
            frame_ids = list(
                range(start, min(start + self.num_frame_per_block, generated_end))
            )
            video = self._sample_video(
                frame_ids,
                video_steps,
                video_exit_id,
                recorder,
                latents=latents,
                stream_ids=stream_ids,
                text_emb=text_emb,
            )
            self.generator.commit_video(
                video,
                frame_ids=frame_ids,
                stream_ids=stream_ids,
                cache=self.cache,
                text_emb=text_emb,
            )
            self._record_block(predicted_video, video, frame_ids)
            action = self._sample_action(
                frame_ids,
                action_steps,
                action_exit_id,
                recorder,
                actions=actions,
                valid_mask=action_valid,
                text_emb=text_emb,
            )
            self.generator.commit_action(
                action,
                frame_ids=frame_ids,
                cache=self.cache,
                text_emb=text_emb,
            )
            self._record_block(predicted_action, action, frame_ids)

        generated_ids = list(range(generated_start, generated_end))
        recorder.require_frames(generated_ids)
        self.cache.assert_no_transactions()
        return self._build_result(
            predicted_video=predicted_video,
            predicted_action=predicted_action,
            generated_ids=generated_ids,
            recorder=recorder,
        )

    def _build_result(
        self,
        *,
        predicted_video: dict[int, torch.Tensor],
        predicted_action: dict[int, torch.Tensor],
        generated_ids: list[int],
        recorder: SelfRolloutRecorder,
    ) -> RolloutResult:
        predicted_clean = VAPair(
            video=torch.cat([predicted_video[f] for f in generated_ids], dim=2),
            action=torch.cat([predicted_action[f] for f in generated_ids], dim=2),
        )
        noisy_at_t = VAPair(
            video=torch.cat(
                [recorder.records[("video", f)].sample for f in generated_ids],
                dim=2,
            ),
            action=torch.cat(
                [recorder.records[("action", f)].sample for f in generated_ids],
                dim=2,
            ),
        )
        first = generated_ids[0]
        video_record = recorder.records[("video", first)]
        action_record = recorder.records[("action", first)]
        video_exit_timestep = float(video_record.timestep)
        action_exit_timestep = float(action_record.timestep)
        return RolloutResult(
            pred_frame_ids=tuple(generated_ids),
            predicted_clean=predicted_clean,
            noisy_at_t=noisy_at_t,
            video_exit_timestep=video_exit_timestep,
            action_exit_timestep=action_exit_timestep,
            video_exit_id=video_record.exit_id,
            action_exit_id=action_record.exit_id,
            video_denoised_timestep_from=video_record.denoisy_from,
            video_denoised_timestep_to=video_record.denoisy_to,
            action_denoised_timestep_from=action_record.denoisy_from,
            action_denoised_timestep_to=action_record.denoisy_to,
        )
