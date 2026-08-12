"""Periodic rollout decoding and visualization shared by distillation trainers."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re
from typing import Any, Callable

import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw, ImageFont

from wan_va.dataset.mot_dataset import (
    quaternion_xyzw_to_matrix,
    relative_20d_to_absolute_actions,
)
from wan_va.mot_spec import mot_spec_from_config


def should_visualize_rollout(step: int, interval: int) -> bool:
    """Return whether a completed optimizer step triggers visualization."""
    return int(interval) > 0 and int(step) > 0 and int(step) % int(interval) == 0


def rollout_output_dir(save_root: str | Path, step: int) -> Path:
    """Return the required directory for one optimizer-step rollout."""
    return Path(save_root) / "rollouts" / f"step_{int(step):08d}"


def rollout_sample_dir(save_root: str | Path, step: int, sample_index: int) -> Path:
    """Return one training-batch sample directory inside a step rollout."""
    return rollout_output_dir(save_root, step) / f"sample_{int(sample_index):03d}"


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _batch_item(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        return value[index].detach().cpu()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _view_names(batch: dict[str, Any], sample_index: int) -> list[str]:
    names = batch["view_names"]
    # default_collate transposes a per-sample list of view names into one tuple
    # per view. Keep support for an already uncollated list as well.
    if names and isinstance(names[0], (list, tuple)):
        return [str(per_view[sample_index]) for per_view in names]
    return [str(name) for name in names]


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return name or "view"


def compose_comparison_frame(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    view_name: str,
    step: int,
    is_prediction: bool,
) -> np.ndarray:
    """Place prediction/condition left of GT without resizing either image."""
    prediction = np.asarray(prediction, dtype=np.uint8)
    ground_truth = np.asarray(ground_truth, dtype=np.uint8)
    if prediction.shape != ground_truth.shape or prediction.ndim != 3:
        raise ValueError("prediction and ground_truth must be matching HWC frames")
    left = Image.fromarray(prediction, mode="RGB")
    right = Image.fromarray(ground_truth, mode="RGB")
    font = ImageFont.load_default()
    banner_height = max(28, prediction.shape[0] // 12)
    left_draw = ImageDraw.Draw(left)
    right_draw = ImageDraw.Draw(right)
    left_draw.rectangle((0, 0, left.width, banner_height), fill=(180, 70, 20))
    right_draw.rectangle((0, 0, right.width, banner_height), fill=(30, 90, 170))
    left_label = "Prediction" if is_prediction else "Condition (not prediction)"
    left_draw.text(
        (8, 7),
        f"{left_label} | {view_name} | step {int(step)}",
        fill="white",
        font=font,
    )
    right_draw.text(
        (8, 7),
        f"Ground Truth | {view_name} | step {int(step)}",
        fill="white",
        font=font,
    )
    return np.concatenate([np.asarray(left), np.asarray(right)], axis=1)


def absolute_action_trajectories(
    batch: dict[str, Any],
    predicted_action: torch.Tensor,
    pred_frame_ids: tuple[int, ...],
    *,
    sample_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Convert full GT and condition+prediction actions to absolute 16D EEF.

    Both streams are first denormalized with the sample's q01/q99 values. The
    production dataset conversion then applies each token's absolute reference
    state, so the returned positions and rotations use the same world-frame
    convention as data loading. ``prediction_start`` counts valid action tokens
    before the generated latent-frame range.
    """
    gt = batch["actions"][sample_index].detach().cpu().clone()
    pred = gt.clone()
    ids = torch.tensor(pred_frame_ids, dtype=torch.long)
    pred[:, ids] = predicted_action[sample_index].detach().cpu()
    valid = batch["action_valid_mask"][sample_index, :, :, :, 0].detach().cpu()
    valid_tokens = valid.all(dim=0)
    references = (
        batch["action_reference_states"][sample_index, :, :, :, 0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
    )
    q01 = batch["action_q01"][sample_index].detach().cpu().view(1, 1, -1)
    q99 = batch["action_q99"][sample_index].detach().cpu().view(1, 1, -1)

    def denormalize(values: torch.Tensor) -> np.ndarray:
        values = values[:, :, :, 0].permute(1, 2, 0)
        relative = (values + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01
        return relative[valid_tokens].numpy()

    reference_values = references[valid_tokens].numpy()
    gt_absolute = relative_20d_to_absolute_actions(
        reference_values,
        denormalize(gt),
    )
    pred_absolute = relative_20d_to_absolute_actions(
        reference_values,
        denormalize(pred),
    )
    packed_indices = torch.arange(valid_tokens.numel()).reshape(valid_tokens.shape)
    valid_times = np.arange(int(valid_tokens.sum().item()), dtype=np.float32)
    generated_mask = torch.zeros_like(valid_tokens)
    generated_mask[ids] = True
    generated_valid = valid_tokens & generated_mask
    prediction_start = int((valid_tokens & ~generated_mask).sum().item())
    if generated_valid.any():
        first_generated_flat = int(packed_indices[generated_valid].min().item())
        prediction_start = int(
            (packed_indices[valid_tokens] < first_generated_flat).sum().item()
        )
    return gt_absolute, pred_absolute, valid_times, prediction_start


def write_action_plot(
    path: str | Path,
    gt_absolute: np.ndarray,
    pred_absolute: np.ndarray,
    token_times: np.ndarray,
    *,
    fps: float,
    prediction_start: int,
    step: int,
) -> None:
    """Save an interactive Plotly trajectory as a self-contained HTML file."""
    import plotly.graph_objects as go

    seconds = token_times / float(fps)
    figure = go.Figure()
    all_positions = np.concatenate(
        (
            gt_absolute[:, 0:3],
            gt_absolute[:, 8:11],
            pred_absolute[:, 0:3],
            pred_absolute[:, 8:11],
        ),
        axis=0,
    )
    orientation_scale = max(float(np.ptp(all_positions, axis=0).max()) * 0.03, 1e-3)
    axis_specs = (
        ("X", "#f43f5e"),
        ("Y", "#10b981"),
        ("Z", "#2563eb"),
    )
    trajectory_specs = (
        (
            "GT",
            gt_absolute,
            [
                [0.0, "#cffafe"],
                [0.25, "#22d3ee"],
                [0.5, "#3b82f6"],
                [0.75, "#4338ca"],
                [1.0, "#172554"],
            ],
            0,
        ),
        (
            "Prediction",
            pred_absolute,
            [
                [0.0, "#fef3c7"],
                [0.25, "#fbbf24"],
                [0.5, "#f97316"],
                [0.75, "#e11d48"],
                [1.0, "#881337"],
            ],
            max(int(prediction_start) - 1, 0),
        ),
    )
    orientation_stride = 4
    for series_index, (
        name,
        actions,
        colorscale,
        series_start,
    ) in enumerate(trajectory_specs):
        series_times = seconds[series_start:]
        for arm, offset, line_width in (
            ("Left", 0, 9),
            ("Right", 8, 7),
        ):
            position = actions[series_start:, offset : offset + 3]
            colorbar = {
                "title": {"text": f"{name} time (s)", "side": "top"},
                "orientation": "h",
                "x": 0.22 if series_index == 0 else 0.62,
                "xanchor": "center",
                "y": -0.14,
                "yanchor": "top",
                "len": 0.3,
                "thickness": 13,
                "outlinewidth": 0,
                "bgcolor": "rgba(0,0,0,0)",
                "tickfont": {"size": 11},
            }
            figure.add_trace(
                go.Scatter3d(
                    x=position[:, 0],
                    y=position[:, 1],
                    z=position[:, 2],
                    mode="lines",
                    name=f"{name} {arm} trajectory",
                    legendgroup=f"{name} {arm}",
                    line={
                        "width": line_width,
                        "color": series_times,
                        "colorscale": colorscale,
                        "cmin": float(series_times.min()),
                        "cmax": float(series_times.max()),
                        "dash": "solid",
                        "showscale": arm == "Left",
                        "colorbar": colorbar,
                    },
                    customdata=series_times[:, None],
                    hovertemplate=(
                        f"{name} {arm}<br>"
                        "t=%{customdata[0]:.2f} s<br>"
                        "x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}"
                        "<extra></extra>"
                    ),
                    showlegend=True,
                )
            )
            if series_index == 1 and len(position):
                boundary = actions[prediction_start, offset : offset + 3]
                figure.add_trace(
                    go.Scatter3d(
                        x=[boundary[0]],
                        y=[boundary[1]],
                        z=[boundary[2]],
                        mode="markers+text",
                        marker={
                            "size": 9,
                            "color": "#111827",
                            "symbol": "diamond",
                            "line": {"color": "#ffffff", "width": 1.5},
                        },
                        text=["Prediction starts" if arm == "Left" else ""],
                        textposition="top center",
                        textfont={"color": "#111827", "size": 12},
                        name=f"{arm} prediction starts",
                        showlegend=False,
                    )
                )

            rotations = quaternion_xyzw_to_matrix(
                actions[series_start:, offset + 3 : offset + 7]
            )
            orientation_ids = np.arange(0, len(position), orientation_stride)
            orientation_positions = position[orientation_ids]
            orientation_rotations = rotations[orientation_ids]
            for axis_index, (axis_name, axis_color) in enumerate(axis_specs):
                axis_end = (
                    orientation_positions
                    + orientation_rotations[:, :, axis_index] * orientation_scale
                )
                coordinates = [[], [], []]
                for origin, endpoint in zip(orientation_positions, axis_end):
                    for dimension in range(3):
                        coordinates[dimension].extend(
                            (float(origin[dimension]), float(endpoint[dimension]), None)
                        )
                figure.add_trace(
                    go.Scatter3d(
                        x=coordinates[0],
                        y=coordinates[1],
                        z=coordinates[2],
                        mode="lines",
                        line={
                            "color": axis_color,
                            "width": 2.2,
                            "dash": "solid",
                        },
                        opacity=0.7,
                        name=(
                            f"{name} orientation axes"
                            if arm == "Left" and axis_index == 0
                            else f"{name} orientation {axis_name}"
                        ),
                        legendgroup=f"{name} orientation axes",
                        showlegend=arm == "Left" and axis_index == 0,
                        hoverinfo="skip",
                    )
                )
    figure.update_layout(
        title={
            "text": (
                "Absolute End-Effector Trajectories"
                f"<br><sup>Ground Truth vs. Prediction · Optimizer Step {int(step)}</sup>"
            ),
            "x": 0.41,
            "xanchor": "center",
            "font": {"size": 24, "family": "Arial, sans-serif", "color": "#0f172a"},
        },
        height=900,
        width=1500,
        margin={"l": 70, "r": 250, "t": 105, "b": 180},
        paper_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "size": 13, "color": "#334155"},
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": "#cbd5e1",
            "font": {"family": "Arial, sans-serif", "size": 12, "color": "#0f172a"},
        },
        legend={
            "x": 0.85,
            "xanchor": "left",
            "y": 0.95,
            "yanchor": "top",
            "bgcolor": "rgba(255, 255, 255, 0.92)",
            "bordercolor": "#e2e8f0",
            "borderwidth": 1,
            "font": {"size": 12, "color": "#334155"},
        },
        scene={
            "xaxis": {
                "title": "X",
                "backgroundcolor": "#f8fafc",
                "gridcolor": "#e2e8f0",
                "zerolinecolor": "#94a3b8",
                "linecolor": "#cbd5e1",
                "showbackground": True,
            },
            "yaxis": {
                "title": "Y",
                "backgroundcolor": "#f8fafc",
                "gridcolor": "#e2e8f0",
                "zerolinecolor": "#94a3b8",
                "linecolor": "#cbd5e1",
                "showbackground": True,
            },
            "zaxis": {
                "title": "Z",
                "backgroundcolor": "#f8fafc",
                "gridcolor": "#e2e8f0",
                "zerolinecolor": "#94a3b8",
                "linecolor": "#cbd5e1",
                "showbackground": True,
            },
            "aspectmode": "data",
            "domain": {"x": [0.0, 0.82], "y": [0.13, 1.0]},
            "camera": {
                "eye": {"x": 1.5, "y": 1.5, "z": 1.05},
                "up": {"x": 0, "y": 0, "z": 1},
            },
        },
    )
    html_path = Path(path).with_suffix(".html")
    figure.write_html(html_path, include_plotlyjs=True, full_html=True)


def _decode_video_latents(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    batch, channels, frames, views, height, width = latents.shape
    packed = latents.permute(0, 3, 1, 2, 4, 5).reshape(
        batch * views, channels, frames, height, width
    )
    dtype = next(vae.parameters()).dtype
    device = next(vae.parameters()).device
    packed = packed.to(device=device, dtype=dtype)
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    decoded = vae.decode(packed * std + mean, return_dict=False)[0]
    decoded = (decoded.float() * 0.5 + 0.5).clamp(0, 1)
    return decoded.permute(0, 2, 1, 3, 4).reshape(
        batch, views, decoded.shape[2], decoded.shape[1], decoded.shape[3], decoded.shape[4]
    ).permute(0, 2, 1, 3, 4, 5).contiguous()


@contextmanager
def _fixed_rollout_state(model: torch.nn.Module, device: torch.device, seed: int):
    devices = [device.index] if device.type == "cuda" else []
    was_training = model.training
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        model.eval()
        try:
            # No autograd graph is built for visualization rollout.
            # The caller's `with _fixed_rollout_state(...):` body runs here.
            with torch.no_grad():
                yield
        finally:
            # Stage 3/3: Always restore the original mode, even if the
            # rollout, VAE decode, or MP4 write raises an exception.
            model.train(was_training)


class RolloutVisualizer:
    """Own the fixed sample and shared rollout/decode/write workflow."""

    def __init__(
        self,
        *,
        config: Any,
        device: torch.device,
        rollout: Callable[[dict[str, Any]], Any],
        rollout_model: Callable[[], torch.nn.Module],
        get_vae: Callable[[], torch.nn.Module],
    ) -> None:
        self.config = config
        self.device = device
        self.rollout = rollout
        self.rollout_model = rollout_model
        self.get_vae = get_vae
        self.interval = int(
            getattr(config.distill, "rollout_visualization_interval", 0)
        )
        self.rank = int(getattr(config, "rank", 0))

    def maybe_run(self, step: int, batch: dict[str, Any]) -> Path | None:
        if not should_visualize_rollout(step, self.interval):
            return None
        model = self.rollout_model()
        seed = int(getattr(self.config, "train_seed", 42))
        with _fixed_rollout_state(model, self.device, seed):
            result = self.rollout(batch)
        output = None
        if self.rank == 0:
            output = rollout_output_dir(self.config.save_root, step)
            output.mkdir(parents=True, exist_ok=True)
            self._write(output, batch, result, int(step))
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return output

    @torch.no_grad()
    def _write(self, output: Path, batch: dict[str, Any], result: Any, step: int) -> None:
        # Stage 1/3: read the MOT data layout.
        #
        # This spec contains:
        # - latent_frames_per_action_chunk_per_view, typically L=4
        # - vae_input_frame_count, typically Q=13
        # - vae_temporal_factor, typically K=4
        spec = mot_spec_from_config(self.config)

        # The rollout predicts only future latent frames.
        #
        # Example:
        #   history latent frames = [0, 1, 2, 3]
        #   anchor latent frame   = [4]
        #   prediction frames     = [5, 6, 7]
        #
        # pred_ids = (5, 6, 7)
        pred_ids = tuple(int(value) for value in result.pred_frame_ids)

        # The VAE is a causal temporal decoder.
        # To decode predicted frame 5 correctly, provide the preceding clean latent frame 4
        # so the decoder has temporal context before the prediction interval.
        previous_id = pred_ids[0] - 1

        # batch["latents"]:
        #   [B, Cv, F, V, Hl, Wl]
        #
        # predecessor:
        #   [B, Cv, 1, V, Hl, Wl]
        #
        # predicted_clean.video:
        #   [B, Cv, G, V, Hl, Wl]
        #
        # concat:
        #   [B, Cv, 1 + G, V, Hl, Wl]
        decode_latents = torch.cat(
            (
                batch["latents"][:, :, previous_id : previous_id + 1].to(self.device),
                result.predicted_clean.video,
            ),
            dim=2,
        )

        # _decode_video_latents does two things:
        #
        # 1. [B,Cv,F,V,Hl,Wl]
        #      -> [B*V,Cv,F,Hl,Wl]
        #    Send each camera view to the VAE as an independent video.
        #
        # 2. De-normalize the latent and run vae.decode.
        #
        # Returns:
        #   [B, R, V, 3, H, W]
        #
        # If G=3 and K=4, four latent frames (1+G) are input,
        # and the decoder typically outputs R = 1 + K*G = 13 RGB frames.
        decoded = _decode_video_latents(self.get_vae(), decode_latents)

        # decoded[:, 0] corresponds to the preceding clean latent previous_id;
        # it only provides context to the causal VAE and is not a prediction result.
        #
        # After removing the first frame:
        #   [B, 1 + K*G, V, 3, H, W]
        #       -> [B, K*G, V, 3, H, W]
        #
        # Example G=3, K=4:
        #   13 decoded frames -> 12 prediction display frames
        # Each generated latent after its clean predecessor represents exactly
        # vae_temporal_factor original-rate frames.
        predicted_rgb = decoded[:, 1:]

        # Stage 2/3: align predicted RGB with the complete GT timeline.
        #
        # The batch keeps the original RGB frames loaded by the dataset:
        #
        # vae_rgb_history:
        #   [B, Thistory, V, 3, H, W]
        #
        # vae_rgb_target:
        #   [B, Ttarget, V, 3, H, W]
        #
        # Concatenation produces the complete GT video timeline:
        #   [B, Ttotal, V, 3, H, W]
        #
        # Common current case:
        #   Thistory=13
        #   Ttarget=13
        #   Ttotal=26
        gt_rgb = torch.cat(
            (batch["vae_rgb_history"], batch["vae_rgb_target"]), dim=1
        ).float()

        # Number of sampled RGB frames corresponding to each action/video chunk.
        # Currently chunk_raw_frames is usually 13.
        chunk_raw_frames = spec.vae_input_frame_count

        # Map the first predicted latent frame id to the complete RGB timeline.
        #
        # Formula breakdown:
        #
        # 1. pred_ids[0] // L
        #    Determine which chunk contains the predicted frame.
        #
        # 2. chunk_index * Q
        #    Get that chunk's start position on the complete RGB timeline.
        #
        # 3. pred_ids[0] % L
        #    Get the latent frame's position within the chunk.
        #
        # 4. The first latent corresponds to one frame;
        #    each subsequent latent corresponds to K RGB frames, so:
        #
        #       1 + K * (latent_offset - 1)
        #
        # Example:
        #   L=4, Q=13, K=4, pred_ids[0]=5
        #
        #   chunk_index  = 5 // 4 = 1
        #   latent_offset = 5 % 4 = 1
        #
        #   raw_prediction_start
        #     = 1*13 + 1 + 4*(1-1)
        #     = 14
        raw_prediction_start = (
            (pred_ids[0] // spec.latent_frames_per_action_chunk_per_view) * chunk_raw_frames
            + 1
            + spec.vae_temporal_factor
            * (pred_ids[0] % spec.latent_frames_per_action_chunk_per_view - 1)
        )

        # The prediction interval is left-closed and right-open:
        # [raw_prediction_start, raw_prediction_end)
        #
        # Example:
        #   start=14
        #   predicted_rgb.shape[1]=12
        #   end=26
        #
        # Thus, in the complete 26-frame display sequence:
        #   [0,14)  = condition/history
        #   [14,26) = prediction
        raw_prediction_end = raw_prediction_start + predicted_rgb.shape[1]

        # Stage 3/3: write step-level shared metadata.
        #
        # These values are shared by the whole batch, so store them at the step root.
        common_metadata = {
            # The completed optimizer step.
            "optimizer_step": step,

            # Latent frame ids predicted by the rollout.
            # Example: [5, 6, 7]
            "predicted_latent_frame_ids": list(pred_ids),

            # Corresponding RGB frame range on the complete MP4 timeline.
            # Example: [14, 26]
            "predicted_raw_frame_range": [raw_prediction_start, raw_prediction_end],

            # Recursively convert EasyDict, Path, torch.dtype, and similar values to JSON values.
            "config": _json_value(dict(self.config)),
        }

        # Output:
        # <save_root>/rollouts/step_00000500/metadata.json
        (output / "metadata.json").write_text(
            json.dumps(common_metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # Stage 4: build a complete comparison video for every batch sample.
        #
        # gt_rgb.shape[0] is the batch size B.
        # Each batch sample is written to a separate sample_000, sample_001, etc. directory.
        for sample_index in range(gt_rgb.shape[0]):
            sample_dir = rollout_sample_dir(
                self.config.save_root,
                step,
                sample_index,
            )
            sample_dir.mkdir(exist_ok=True)

            # display initially contains the complete GT video:
            #   [Ttotal, V, 3, H, W]
            #
            # Therefore the condition/history interval still shows the real input.
            display = gt_rgb[sample_index].clone()

            # Replace only the future interval actually covered by the rollout with predicted RGB.
            #
            # Before replacement:
            #   display[0:14]  = GT condition/history
            #   display[14:26] = GT future
            #
            # After replacement:
            #   display[0:14]  = GT condition/history
            #   display[14:26] = predicted future
            display[raw_prediction_start:raw_prediction_end] = predicted_rgb[
                sample_index
            ].to(display.device)

            # source_fps is the original action/video data fps, for example 50.
            source_fps = float(_batch_item(batch["fps"], sample_index))

            # The dataset samples one RGB frame every video_downsample_ratio frames.
            #
            # Example:
            #   source_fps=50
            #   video_downsample_ratio=4
            #   video_fps=12.5
            #
            # Writing the MP4 at 12.5 fps keeps playback time aligned with the original data.
            video_fps = source_fps / float(self.config.video_downsample_ratio)

            # default_collate transposes view names into:
            # [
            #   ("head_sample0", "head_sample1"),
            #   ("left_sample0", "left_sample1"),
            #   ...
            # ]
            #
            # _view_names restores the names for the current sample as:
            # ["head", "left_wrist", "right_wrist"]
            view_names = _view_names(batch, sample_index)

            # Generate one MP4 per camera view.
            for view_index, view_name in enumerate(view_names):
                path = sample_dir / f"view_{view_index:02d}_{_safe_name(view_name)}.mp4"
                with imageio.get_writer(
                    path,
                    fps=video_fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    # Do not automatically scale H/W to a macroblock size; preserve the original resolution.
                    macro_block_size=None,
                ) as writer:
                    # Iterate over the complete timeline, not only the prediction range.
                    for frame_index in range(gt_rgb.shape[1]):
                        # Left content:
                        # - condition interval comes from GT
                        # - prediction interval comes from the rollout
                        #
                        # [3,H,W] -> [H,W,3]
                        # float [0,1] -> uint8 [0,255]
                        pred_frame = (
                            display[frame_index, view_index]
                            .permute(1, 2, 0)
                            .clamp(0, 1)
                            .mul(255)
                            .round()
                            .byte()
                            .cpu()
                            .numpy()
                        )

                        # The right side is always the real GT.
                        gt_frame = (
                            gt_rgb[sample_index, frame_index, view_index]
                            .permute(1, 2, 0)
                            .clamp(0, 1)
                            .mul(255)
                            .round()
                            .byte()
                            .cpu()
                            .numpy()
                        )

                        # Output frame:
                        #
                        # left  [H,W,3]
                        # right [H,W,3]
                        # concat -> [H,2W,3]
                        #
                        # Left label for the condition interval:
                        #   "Condition (not prediction)"
                        #
                        # Left label for the prediction interval:
                        #   "Prediction"
                        #
                        # The right side is always labeled:
                        #   "Ground Truth"
                        writer.append_data(
                            compose_comparison_frame(
                                pred_frame,
                                gt_frame,
                                view_name=view_name,
                                step=step,
                                is_prediction=(
                                    raw_prediction_start
                                    <= frame_index
                                    < raw_prediction_end
                                ),
                            )
                        )

            # Stage 5: convert relative actions to absolute trajectories and write Plotly.
            #
            # result.predicted_clean.action:
            #   [B, Ca=20, G, N, 1]
            #
            # absolute_action_trajectories performs the following internally:
            #
            # 1. Copy the complete GT action:
            #      pred = gt.clone()
            #
            # 2. Write predicted actions only into the latent frames listed by pred_ids:
            #      pred[:, pred_ids] = predicted_action
            #
            #    Therefore:
            #      the prefix of pred = condition GT
            #      the generated segment of pred = model prediction
            #
            # 3. Use action_valid_mask to remove invalid/padding tokens.
            #
            # 4. De-normalize with q01/q99:
            #
            #      relative =
            #        (normalized + 1) / 2
            #        * (q99 - q01 + 1e-6)
            #        + q01
            #
            # 5. Call the production conversion function from the dataset:
            #
            #      relative_20d_to_absolute_actions(
            #          action_reference_states,
            #          relative_actions,
            #      )
            #
            # Returns:
            #   gt_actions   [Tvalid, 16]
            #   pred_actions [Tvalid, 16]
            #   token_times  [Tvalid]
            #   prediction_start int
            gt_actions, pred_actions, token_times, prediction_start = (
                absolute_action_trajectories(
                    batch,
                    result.predicted_clean.action,
                    pred_ids,
                    sample_index=sample_index,
                )
            )

            # Write an interactive Plotly HTML with GT and Prediction overlaid
            # in one 3D scene. Blue/red gradients indicate time, while sparse
            # local XYZ axes represent rotation.
            #
            # Action time:
            #   seconds = token_times / source_fps
            #
            # A boundary marker is placed at prediction_start.
            write_action_plot(
                sample_dir / "actions.html",
                gt_actions,
                pred_actions,
                token_times,
                fps=source_fps,
                prediction_start=prediction_start,
                step=step,
            )

            # Stage 6: write metadata for the individual sample.
            sample_metadata = {
                **common_metadata,

                # Index of the current sample in the training batch.
                "sample_index": sample_index,

                # Original text condition, not the text embedding.
                "text": str(_batch_item(batch["text"], sample_index)),

                # Frame ids actually used to load RGB.
                # Out-of-range frames have already been padded to the episode boundary.
                "frame_ids": _batch_item(
                    batch["frame_ids"], sample_index
                ).tolist(),

                # Frame ids requested before padding.
                # Comparing these with frame_ids reveals where boundary padding occurred.
                "requested_frame_ids": _batch_item(
                    batch["requested_frame_ids"], sample_index
                ).tolist(),

                # All camera view names for the current sample.
                "view_names": view_names,

                # Original action/video data frame rate.
                "source_fps": source_fps,

                # Actual playback frame rate of the sampled MP4.
                "video_fps": video_fps,
            }

            # Output:
            # <save_root>/
            #   rollouts/
            #     step_00000500/
            #       metadata.json
            #       sample_000/
            #         metadata.json
            #         actions.html
            #         view_00_observation.images.cam_high.mp4
            #         view_01_observation.images.cam_left_wrist.mp4
            #         view_02_observation.images.cam_right_wrist.mp4
            (sample_dir / "metadata.json").write_text(
                json.dumps(sample_metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
