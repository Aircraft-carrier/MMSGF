#!/usr/bin/env python
"""Fixed-window MOT inference on prepared LeRobot rows."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.utils.data import default_collate

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wan_va.dataset import MotTrainData
from wan_va.checkpoint_retention import EVAL_IN_PROGRESS_MARKER
from wan_va.configs.mot_eval_cfg import make_mot_eval_cfg
from wan_va.dataset.mot_dataset import (
    quaternion_xyzw_to_matrix,
    relative_20d_to_absolute_actions,
    rotation_matrix_to_rot6d_cols,
)
from wan_va.dataset.ply_io import write_ply
from wan_va.mot_spec import mot_spec_from_config
from wan_va.modules.fa4_attention import validate_fa4_training_environment
from wan_va.modules.model_3dva_mot import ThreeDVAMOTTransformer3DModel
from wan_va.modules.utils import WanVAEStreamingWrapper, load_vae
from wan_va.modules.vggto_loss import normalize_depth_targets, normalize_point_targets
from wan_va.utils import init_logger, logger

from inference.mot_inference import (
    MOT_INFERENCE_DTYPE,
    run_geometry_inference,
    run_mot_inference,
    run_video_inference,
)
from inference.convert_dcp_to_safetensors import convert_dcp_to_safetensors

DEFAULT_WAN22_MODEL_ROOT = (
    "/workspace/cache/huggingface_cache/hub/"
    "models--robbyant--lingbot-va-base/snapshots/"
    "68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c"
)
MOT_DATASET_ROOT = _REPO_ROOT / "data" / "data" / "umi_mot_full_data_train_0712_final"
POINTCLOUD_MANIFEST = "mot_final_training_pointcloud_manifest.jsonl"
NON_POINTCLOUD_MANIFEST = "mot_final_training_non_pointcloud_manifest.jsonl"
EVAL_SOURCE_DATASETS = ("lumos_lerobot", "genrobot_lerobot")
DCP_DIR_NAME = "distributed_state"
SAFETENSORS_NAME = "diffusion_pytorch_model.safetensors"

MOT_ACTION_20D_LABELS = [
    "r0_dx",
    "r0_dy",
    "r0_dz",
    "r0_r6_0",
    "r0_r6_1",
    "r0_r6_2",
    "r0_r6_3",
    "r0_r6_4",
    "r0_r6_5",
    "r0_grip",
    "r1_dx",
    "r1_dy",
    "r1_dz",
    "r1_r6_0",
    "r1_r6_1",
    "r1_r6_2",
    "r1_r6_3",
    "r1_r6_4",
    "r1_r6_5",
    "r1_grip",
]
MOT_ABSOLUTE_ACTION_20D_LABELS = [
    "r0_x",
    "r0_y",
    "r0_z",
    "r0_r6_0",
    "r0_r6_1",
    "r0_r6_2",
    "r0_r6_3",
    "r0_r6_4",
    "r0_r6_5",
    "r0_grip",
    "r1_x",
    "r1_y",
    "r1_z",
    "r1_r6_0",
    "r1_r6_1",
    "r1_r6_2",
    "r1_r6_3",
    "r1_r6_4",
    "r1_r6_5",
    "r1_grip",
]


def _denormalize_actions(
    actions: torch.Tensor,
    norm_stat: dict[str, Any],
    *,
    q01=None,
    q99=None,
    action_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert normalized actions to chronological valid action vectors in real units."""

    if actions.ndim != 5 or actions.shape[0] != 1:
        raise ValueError(f"Expected action tensor [1,C,F,N,1], got {tuple(actions.shape)}")
    if q01 is None or q99 is None:
        q01 = norm_stat["q01"]
        q99 = norm_stat["q99"]
    q01 = torch.as_tensor(q01, dtype=actions.dtype, device=actions.device).view(1, -1, 1, 1, 1)
    q99 = torch.as_tensor(q99, dtype=actions.dtype, device=actions.device).view(1, -1, 1, 1, 1)
    denorm = (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    denorm = denorm[0, :, :, :, 0].permute(1, 2, 0).reshape(-1, actions.shape[1])
    if action_valid_mask is not None:
        mask = action_valid_mask.to(device=actions.device, dtype=torch.bool)
        if mask.shape != actions.shape:
            raise ValueError(f"action_valid_mask shape {tuple(mask.shape)} does not match actions {tuple(actions.shape)}")
        token_valid = mask[0, :, :, :, 0].all(dim=0).reshape(-1)
        denorm = denorm[token_valid]
    return denorm.detach().cpu()


def _absolute_16d_to_plot_20d(actions: torch.Tensor) -> torch.Tensor:
    """Convert absolute `[N,16]` xyzw-quat actions to `[N,20]` xyz+rot6d+grip."""

    if actions.ndim != 2 or actions.shape[-1] != 16:
        raise ValueError(f"Expected absolute actions [N,16], got {tuple(actions.shape)}")
    actions_np = actions.detach().cpu().numpy()
    out = np.empty((actions_np.shape[0], 20), dtype=np.float32)
    for arm_idx in range(2):
        raw_off = arm_idx * 8
        plot_off = arm_idx * 10
        out[:, plot_off : plot_off + 3] = actions_np[:, raw_off : raw_off + 3]
        rot = quaternion_xyzw_to_matrix(actions_np[:, raw_off + 3 : raw_off + 7])
        out[:, plot_off + 3 : plot_off + 9] = rotation_matrix_to_rot6d_cols(rot)
        out[:, plot_off + 9] = actions_np[:, raw_off + 7]
    return torch.from_numpy(out)


def _save_action_20d_plot(
    *,
    pred: np.ndarray,
    gt: np.ndarray,
    labels: list[str],
    history_steps: int,
    break_at_history: bool,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if pred.shape != gt.shape or gt.ndim != 2 or gt.shape[0] == 0:
        raise ValueError(f"Expected matching non-empty action arrays [T,C], got pred={pred.shape}, gt={gt.shape}")
    fig, axes = plt.subplots(4, 5, figsize=(20, 10), sharex=True)
    x = np.arange(gt.shape[0])
    for idx, ax in enumerate(axes.flat):
        if idx >= gt.shape[1]:
            ax.axis("off")
            continue
        if history_steps > 0:
            ax.axvspan(
                -0.5,
                history_steps - 0.5,
                color="#dbeafe",
                alpha=0.55,
                linewidth=0,
                label="history" if idx == 0 else None,
            )
        if history_steps < gt.shape[0]:
            ax.axvspan(
                history_steps - 0.5,
                gt.shape[0] - 0.5,
                color="#fef3c7",
                alpha=0.35,
                linewidth=0,
                label="generated" if idx == 0 else None,
            )
            ax.axvline(history_steps - 0.5, color="#b45309", linestyle="--", linewidth=0.8)
        segments = (slice(0, gt.shape[0]),)
        if break_at_history and 0 < history_steps < gt.shape[0]:
            segments = (slice(0, history_steps), slice(history_steps, gt.shape[0]))
        for segment_idx, segment in enumerate(segments):
            ax.plot(
                x[segment],
                gt[segment, idx],
                color="#1f77b4",
                label="gt" if segment_idx == 0 else None,
                linewidth=1.2,
            )
            ax.plot(
                x[segment],
                pred[segment, idx],
                color="#ff7f0e",
                label="pred" if segment_idx == 0 else None,
                linewidth=1.0,
            )
        if gt.shape[0] > 1:
            ax.set_xlim(0, gt.shape[0] - 1)
        ax.set_title(labels[idx] if idx < len(labels) else f"ch{idx}", fontsize=8)
        ax.grid(True, linewidth=0.3)
    axes.flat[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _views_to_wide_frames(rgb_views: torch.Tensor) -> np.ndarray:
    if rgb_views.ndim == 6:
        rgb_views = rgb_views[0]
    if rgb_views.ndim != 5:
        raise ValueError(f"Expected rgb views [T,V,C,H,W], got {tuple(rgb_views.shape)}")
    frames = rgb_views.detach().cpu().float().permute(0, 3, 1, 4, 2).contiguous()
    frames = frames.reshape(frames.shape[0], frames.shape[1], frames.shape[2] * frames.shape[3], frames.shape[4])
    return np.clip(frames.numpy() * 255.0, 0, 255).astype(np.uint8)


def _interleave_gt_pred_view_frames(
    pred_frames: np.ndarray,
    gt_rgb: torch.Tensor,
    *,
    history_count: int = 0,
) -> np.ndarray:
    gt_views = gt_rgb.detach().cpu().float()[0].permute(0, 1, 3, 4, 2).numpy()
    gt_views = np.clip(gt_views * 255.0, 0, 255).astype(np.uint8)
    pred_frames = np.asarray(pred_frames)
    if pred_frames.dtype != np.uint8:
        pred_frames = np.clip(pred_frames * 255.0, 0, 255).astype(np.uint8)

    frame_count = min(int(pred_frames.shape[0]), int(gt_views.shape[0]))
    views = int(gt_views.shape[1])
    if views <= 0:
        raise ValueError("gt_rgb must contain at least one view")
    pred_view_width = int(pred_frames.shape[2]) // views
    if pred_view_width <= 0:
        raise ValueError(f"pred frame width {pred_frames.shape[2]} is too small for {views} views")

    frames = []
    for frame_id in range(frame_count):
        columns = []
        for view_id in range(views):
            phase = "HISTORY" if frame_id < int(history_count) else "TARGET"
            columns.append(
                _annotate_panel(gt_views[frame_id, view_id], f"GT | VIEW {view_id} | {phase}", (37, 99, 235))
            )
            if frame_id < int(history_count):
                pred_view = gt_views[frame_id, view_id]
            else:
                pred_view = pred_frames[frame_id, :, view_id * pred_view_width : (view_id + 1) * pred_view_width]
            columns.append(_annotate_panel(pred_view, f"PRED | VIEW {view_id} | {phase}", (217, 119, 6)))
        frames.append(np.concatenate(columns, axis=1))
    return np.asarray(frames)


def _annotate_panel(frame: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    banner_h = max(24, min(42, height // 10))
    draw.rectangle((0, 0, width, banner_h), fill=color)
    draw.rectangle((0, 0, max(6, width // 120), height), fill=color)
    draw.text((10, max(4, banner_h // 4)), label, fill=(255, 255, 255))
    return np.asarray(image)


def _colorize_depth(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap_name: str,
) -> np.ndarray:
    from matplotlib import colormaps

    scale = max(float(vmax) - float(vmin), 1e-6)
    normalized = np.clip((values - float(vmin)) / scale, 0.0, 1.0)
    rgb = np.asarray(colormaps[cmap_name](normalized, bytes=True)[..., :3], dtype=np.uint8)
    rgb[~valid] = 0
    return rgb


def _depth_visualization_frames(
    *,
    rgb: np.ndarray,
    pred_depth: np.ndarray,
    pred_conf: np.ndarray,
    gt_depth: np.ndarray | None = None,
    gt_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    pred_depth = np.asarray(pred_depth, dtype=np.float32)
    pred_conf = np.asarray(pred_conf, dtype=np.float32)
    if pred_depth.ndim != 4 or pred_conf.shape != pred_depth.shape:
        raise ValueError(
            "Depth visualization expects matching pred depth/conf [T,V,H,W], "
            f"got pred={pred_depth.shape}, conf={pred_conf.shape}"
        )
    expected_rgb_shape = (*pred_depth.shape[:2], 3, *pred_depth.shape[2:])
    if rgb.shape != expected_rgb_shape:
        raise ValueError(f"Depth visualization expects RGB {expected_rgb_shape}, got {rgb.shape}")
    if (gt_depth is None) != (gt_valid_mask is None):
        raise ValueError("GT depth and valid mask must either both be provided or both be omitted")

    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8).transpose(0, 1, 3, 4, 2)
    pred_valid = np.isfinite(pred_depth)
    if not pred_valid.any():
        raise ValueError("Depth visualization has no finite predicted depth pixels")

    has_gt = gt_depth is not None
    comparison_valid = None
    if has_gt:
        gt_depth = np.asarray(gt_depth, dtype=np.float32)
        gt_valid = np.asarray(gt_valid_mask, dtype=bool).copy()
        if gt_depth.shape != pred_depth.shape or gt_valid.shape != pred_depth.shape:
            raise ValueError(
                "Depth visualization expects matching GT depth/valid [T,V,H,W], "
                f"got gt={gt_depth.shape}, valid={gt_valid.shape}, pred={pred_depth.shape}"
            )
        gt_valid &= np.isfinite(gt_depth)
        comparison_valid = gt_valid & pred_valid
        if not comparison_valid.any():
            raise ValueError("Depth visualization has no valid GT/pred comparison pixels")
        depth_values = np.concatenate((gt_depth[gt_valid], pred_depth[comparison_valid]))
    else:
        gt_valid = None
        depth_values = pred_depth[pred_valid]

    depth_min, depth_max = np.quantile(depth_values, (0.01, 0.99))
    if depth_max <= depth_min:
        depth_max = depth_min + 1e-6

    conf_valid = np.isfinite(pred_conf) & (pred_conf > 0)
    if not conf_valid.any():
        raise ValueError("Depth visualization has no finite positive confidence pixels")
    log_conf = np.log(pred_conf.clip(min=1e-6))
    conf_min, conf_max = np.quantile(log_conf[conf_valid], (0.01, 0.99))
    if conf_max <= conf_min:
        conf_max = conf_min + 1e-6

    if has_gt:
        diff = np.abs(pred_depth - gt_depth)
        diff_max = max(float(np.quantile(diff[comparison_valid], 0.99)), 1e-6)

    frames = []
    for frame_idx in range(pred_depth.shape[0]):
        view_rows = []
        for view_idx in range(pred_depth.shape[1]):
            panels = [
                _annotate_panel(rgb[frame_idx, view_idx], f"RGB | VIEW {view_idx}", (31, 41, 55))
            ]
            if has_gt:
                gt_panel = _colorize_depth(
                    gt_depth[frame_idx, view_idx],
                    gt_valid[frame_idx, view_idx],
                    vmin=float(depth_min),
                    vmax=float(depth_max),
                    cmap_name="viridis",
                )
                panels.append(_annotate_panel(gt_panel, f"GT DEPTH | VIEW {view_idx}", (37, 99, 235)))
            pred_panel = _colorize_depth(
                pred_depth[frame_idx, view_idx],
                pred_valid[frame_idx, view_idx],
                vmin=float(depth_min),
                vmax=float(depth_max),
                cmap_name="viridis",
            )
            panels.append(_annotate_panel(pred_panel, f"PRED DEPTH | VIEW {view_idx}", (217, 119, 6)))
            if has_gt:
                diff_panel = _colorize_depth(
                    diff[frame_idx, view_idx],
                    comparison_valid[frame_idx, view_idx],
                    vmin=0.0,
                    vmax=diff_max,
                    cmap_name="magma",
                )
                panels.append(_annotate_panel(diff_panel, f"DEPTH DIFF | VIEW {view_idx}", (185, 28, 28)))
            conf_panel = _colorize_depth(
                log_conf[frame_idx, view_idx],
                conf_valid[frame_idx, view_idx],
                vmin=float(conf_min),
                vmax=float(conf_max),
                cmap_name="plasma",
            )
            panels.append(_annotate_panel(conf_panel, f"PRED CONF | VIEW {view_idx}", (126, 34, 206)))
            view_rows.append(np.concatenate(panels, axis=1))
        frames.append(np.concatenate(view_rows, axis=0))
    return np.asarray(frames)


def _reservoir_sample_manifest(
    path: Path,
    *,
    count: int,
    has_pointcloud: bool,
    source_dataset: str,
    action_chunk_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row.get("has_pointcloud")) != has_pointcloud:
                raise ValueError(
                    f"{path}:{line_number} has_pointcloud={row.get('has_pointcloud')!r}, expected {has_pointcloud}"
                )
            if row.get("source_dataset") != source_dataset:
                continue
            if _full_window_start_bounds(row, action_chunk_size=action_chunk_size) is None:
                continue
            seen += 1
            if len(selected) < count:
                selected.append(row)
                continue
            replacement = rng.randrange(seen)
            if replacement < count:
                selected[replacement] = row
    if len(selected) != count:
        raise RuntimeError(f"Need {count} {source_dataset} rows from {path}, found {seen}")
    return selected


def _select_inference_rows(
    *,
    dataset_root: Path,
    action_chunk_size: int = 48,
    rng: random.Random,
) -> list[dict[str, Any]]:
    meta_dir = dataset_root / "meta"
    rows = []
    for source_dataset in EVAL_SOURCE_DATASETS:
        for manifest_name, has_pointcloud in (
            (POINTCLOUD_MANIFEST, True),
            (NON_POINTCLOUD_MANIFEST, False),
        ):
            rows.extend(
                _reservoir_sample_manifest(
                    meta_dir / manifest_name,
                    count=1,
                    has_pointcloud=has_pointcloud,
                    source_dataset=source_dataset,
                    action_chunk_size=action_chunk_size,
                    rng=rng,
                )
            )
    return rows


def _full_window_start_bounds(
    row: dict[str, Any],
    *,
    action_chunk_size: int,
) -> tuple[int, int] | None:
    valid_start, valid_end = (int(value) for value in row["valid_start_range"])
    segment = row["segment"]
    start = max(valid_start, int(segment["start_frame"]) + int(action_chunk_size) + 1)
    end = min(valid_end, int(segment["end_frame"]) - int(action_chunk_size) - 1)
    return None if end < start else (start, end)


def _sample_full_window_start(
    row: dict[str, Any],
    *,
    action_chunk_size: int,
    rng: random.Random,
) -> int:
    bounds = _full_window_start_bounds(row, action_chunk_size=action_chunk_size)
    if bounds is None:
        segment = row["segment"]
        raise ValueError(
            "Selected episode has no full inference window without padding: "
            f"episode={row.get('episode_index')} valid_start_range={row['valid_start_range']} "
            f"segment=[{segment['start_frame']},{segment['end_frame']})"
        )
    return rng.randint(*bounds)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_model_paths(checkpoint_root: str | Path, wan22_model_root: str | Path, output_dir: Path) -> None:
    checkpoint_path = Path(checkpoint_root).expanduser()
    if checkpoint_path.name == "transformer":
        checkpoint_path = checkpoint_path.parent
    paths = [str(checkpoint_path), str(Path(wan22_model_root).expanduser())]

    seen: set[str] = set()
    with (output_dir / "model.jsonl").open("w", encoding="utf-8") as f:
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            f.write(json.dumps(path, ensure_ascii=False) + "\n")


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


class MOTInferenceSession:
    """Model/VAE resources for standalone inference; no training state."""

    def __init__(self, *, config, transformer, device: torch.device):
        self.config = config
        self.transformer = transformer
        self.device = device
        self.dtype = MOT_INFERENCE_DTYPE
        self._vae = None
        self._empty_text_emb = None

    def move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return _move_to_device(batch, self.device)

    def _get_vae(self):
        if self._vae is None:
            self._vae = load_vae(
                os.path.join(self.config.wan22_pretrained_model_name_or_path, "vae"),
                torch_dtype=self.dtype,
                torch_device=self.device,
            ).eval()
            self._vae.requires_grad_(False)
        return self._vae

    def _get_empty_text_emb(self) -> torch.Tensor:
        if self._empty_text_emb is None:
            path = Path(self.config.empty_emb_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            self._empty_text_emb = torch.load(path, map_location="cpu", weights_only=False)
        return self._empty_text_emb

    @torch.no_grad()
    def _encode_one_view_latent(self, frames: torch.Tensor) -> torch.Tensor:
        spec = mot_spec_from_config(self.config)
        if frames.shape[0] != spec.vae_input_frame_count:
            raise ValueError(f"MOT VAE input must have {spec.vae_input_frame_count} frames, got {frames.shape[0]}")
        vae = self._get_vae()
        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype
        wrapper = WanVAEStreamingWrapper(vae)
        video = frames.permute(1, 0, 2, 3)[None].to(device=vae_device, dtype=vae_dtype) * 2.0 - 1.0
        chunks = [wrapper.encode_chunk(video[:, :, :1])]
        for latent_idx in range(1, spec.latent_frames_per_action_chunk_per_view):
            start = 1 + spec.vae_temporal_factor * (latent_idx - 1)
            chunks.append(wrapper.encode_chunk(video[:, :, start : start + spec.vae_temporal_factor]))
        mu, _logvar = torch.chunk(torch.cat(chunks, dim=2), 2, dim=1)
        latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
        return ((mu - latents_mean) / latents_std)[0].to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _encode_vae_rgb(self, vae_rgb: torch.Tensor) -> torch.Tensor:
        if vae_rgb.ndim != 6:
            raise ValueError(f"MOT vae_rgb must be [B,T,V,C,H,W], got {tuple(vae_rgb.shape)}")
        per_batch = []
        for batch_idx in range(vae_rgb.shape[0]):
            per_view = [
                self._encode_one_view_latent(vae_rgb[batch_idx, :, view_idx])
                for view_idx in range(vae_rgb.shape[2])
            ]
            per_batch.append(torch.stack(per_view, dim=2))
        return torch.stack(per_batch, dim=0)

    @torch.no_grad()
    def materialize_batch_latents(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "latents" in batch:
            return batch
        if "vae_rgb_history" not in batch or "vae_rgb_target" not in batch:
            raise KeyError("MOT batch must contain either latents or vae_rgb_history/vae_rgb_target")
        out = dict(batch)
        out["latents"] = torch.cat(
            [self._encode_vae_rgb(batch["vae_rgb_history"]), self._encode_vae_rgb(batch["vae_rgb_target"])],
            dim=2,
        )
        return out

    @torch.no_grad()
    def decode_latents_to_rgb_views(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self._get_vae()
        batch_size, channels, latent_frames, views, height, width = latents.shape
        per_view_latents = latents.permute(0, 3, 1, 2, 4, 5).reshape(
            batch_size * views,
            channels,
            latent_frames,
            height,
            width,
        ).to(device=self.device, dtype=vae.dtype)
        latents_mean = torch.tensor(vae.config.latents_mean, device=self.device, dtype=per_view_latents.dtype).view(
            1, -1, 1, 1, 1
        )
        latents_std = torch.tensor(vae.config.latents_std, device=self.device, dtype=per_view_latents.dtype).view(
            1, -1, 1, 1, 1
        )
        decoded = vae.decode(per_view_latents * latents_std + latents_mean, return_dict=False)[0]
        decoded = (decoded.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        frames = decoded.permute(0, 2, 1, 3, 4).reshape(
            batch_size,
            views,
            decoded.shape[2],
            decoded.shape[1],
            decoded.shape[3],
            decoded.shape[4],
        ).permute(0, 2, 1, 3, 4, 5).contiguous()
        return frames.to(device=latents.device, dtype=latents.dtype)

    def run_video(self, batch: dict[str, Any], frame_count: int):
        return run_video_inference(
            batch,
            frame_count,
            transformer=self.transformer,
            config=self.config,
            spec=mot_spec_from_config(self.config),
            device=self.device,
            empty_text_emb=self._get_empty_text_emb(),
            decode_latents_to_rgb_views=self.decode_latents_to_rgb_views,
        )

    def run_geometry(self, batch: dict[str, Any], frame_count: int):
        return run_geometry_inference(
            batch,
            frame_count,
            transformer=self.transformer,
            spec=mot_spec_from_config(self.config),
            device=self.device,
        )

    def run_full(self, batch: dict[str, Any], frame_count: int):
        return run_mot_inference(
            batch,
            frame_count,
            transformer=self.transformer,
            config=self.config,
            spec=mot_spec_from_config(self.config),
            device=self.device,
            empty_text_emb=self._get_empty_text_emb(),
            decode_latents_to_rgb_views=self.decode_latents_to_rgb_views,
        )

    def run(self, batch: dict[str, Any], frame_count: int, *, mode: str):
        if mode == "video":
            return self.run_video(batch, frame_count)
        if mode == "geometry":
            return self.run_geometry(batch, frame_count)
        if mode == "full":
            return self.run_full(batch, frame_count)
        raise ValueError(f"unsupported inference mode={mode!r}")

    def save_action_plot(self, pred_actions, gt_actions, sample_dir: Path, *, batch: dict[str, Any]) -> None:
        required_metadata = ("action_q01", "action_q99", "action_reference_states", "action_valid_mask")
        missing_metadata = [key for key in required_metadata if key not in batch]
        if missing_metadata:
            raise KeyError(f"MOT inference batch missing per-sample action metadata: {missing_metadata}")

        frame_count = int(pred_actions.shape[2])
        action_valid_mask = batch["action_valid_mask"][:1, :, :frame_count]
        token_valid = action_valid_mask[0, :, :, :, 0].all(dim=0)
        spec = mot_spec_from_config(self.config)
        valid_steps = int(token_valid.sum().item())
        history_steps = int(token_valid[: spec.history_latent_frames].sum().item())
        expected_steps = 2 * spec.action_chunk_size
        if valid_steps != expected_steps or history_steps != spec.action_chunk_size:
            raise ValueError(
                "MOT action plots require one complete 48-step history and 48-step target window: "
                f"valid_steps={valid_steps}, history_steps={history_steps}"
            )
        denorm_kwargs = {
            "q01": batch["action_q01"][:1],
            "q99": batch["action_q99"][:1],
            "action_valid_mask": action_valid_mask,
        }
        pred_relative = _denormalize_actions(pred_actions[:1, :, :frame_count], self.config.norm_stat, **denorm_kwargs)
        gt_relative = _denormalize_actions(gt_actions[:1, :, :frame_count], self.config.norm_stat, **denorm_kwargs)
        _save_action_20d_plot(
            pred=pred_relative.numpy(),
            gt=gt_relative.numpy(),
            labels=MOT_ACTION_20D_LABELS,
            history_steps=history_steps,
            break_at_history=True,
            path=sample_dir / "action_plot_denorm_20d_relative.png",
        )

        refs = batch["action_reference_states"][:1, :, :frame_count, :, 0]
        refs = refs[0].permute(1, 2, 0).reshape(-1, refs.shape[1]).detach().cpu().numpy()
        refs = refs[token_valid.reshape(-1).detach().cpu().numpy()]
        pred_absolute = torch.from_numpy(relative_20d_to_absolute_actions(refs, pred_relative.numpy()))
        gt_absolute = torch.from_numpy(relative_20d_to_absolute_actions(refs, gt_relative.numpy()))
        _save_action_20d_plot(
            pred=_absolute_16d_to_plot_20d(pred_absolute).numpy(),
            gt=_absolute_16d_to_plot_20d(gt_absolute).numpy(),
            labels=MOT_ABSOLUTE_ACTION_20D_LABELS,
            history_steps=history_steps,
            break_at_history=False,
            path=sample_dir / "action_plot_denorm_20d_absolute.png",
        )

    @torch.no_grad()
    def save_video(self, pred_latents, gt_rgb, sample_dir: Path) -> None:
        spec = mot_spec_from_config(self.config)
        if int(pred_latents.shape[2]) != spec.total_latent_frames:
            raise ValueError(
                f"MOT video export requires {spec.total_latent_frames} latent frames, got {pred_latents.shape[2]}"
            )
        expected_rgb_frames = 2 * spec.vae_input_frame_count
        if int(gt_rgb.shape[1]) != expected_rgb_frames:
            raise ValueError(f"MOT GT video export requires {expected_rgb_frames} frames, got {gt_rgb.shape[1]}")

        history_views = self.decode_latents_to_rgb_views(
            pred_latents[:1, :, : spec.history_latent_frames],
        )
        target_views = self.decode_latents_to_rgb_views(
            pred_latents[:1, :, spec.history_latent_frames :],
        )
        if history_views.shape[1] != spec.vae_input_frame_count or target_views.shape[1] != spec.vae_input_frame_count:
            raise ValueError(
                "Each MOT history/target VAE decode must produce "
                f"{spec.vae_input_frame_count} frames, got {history_views.shape[1]} and {target_views.shape[1]}"
            )
        pred_frames = _views_to_wide_frames(torch.cat([history_views, target_views], dim=1)[0])
        history_count = spec.vae_input_frame_count
        frames = _interleave_gt_pred_view_frames(
            pred_frames,
            gt_rgb,
            history_count=history_count,
        )
        if len(frames):
            imageio.mimsave(
                sample_dir / "video_gt_pred_side_by_side.mp4",
                frames,
                fps=int(self.config.inference_video_fps),
            )

    @torch.no_grad()
    def save_depth_video(
        self,
        pred_depth: torch.Tensor,
        pred_depth_conf: torch.Tensor,
        geometry_rgb: torch.Tensor,
        batch: dict[str, Any],
        sample_dir: Path,
    ) -> str:
        pred_depth = pred_depth[:1].detach().cpu().float()
        if pred_depth.ndim != 6 or pred_depth.shape[-1] != 1:
            raise ValueError(f"Expected pred depth [B,T,V,H,W,1], got {tuple(pred_depth.shape)}")
        pred_depth = pred_depth[..., 0]

        pred_depth_conf = pred_depth_conf[:1].detach().cpu().float()
        if tuple(pred_depth_conf.shape) != tuple(pred_depth.shape):
            raise ValueError(
                "Pred depth/conf mismatch: "
                f"depth={tuple(pred_depth.shape)}, conf={tuple(pred_depth_conf.shape)}"
            )

        geometry_rgb = geometry_rgb[:1].detach().cpu().float()
        expected_rgb_shape = (
            pred_depth.shape[0],
            pred_depth.shape[1],
            4,
            pred_depth.shape[2],
            3,
            pred_depth.shape[3],
            pred_depth.shape[4],
        )
        if tuple(geometry_rgb.shape) != expected_rgb_shape:
            raise ValueError(f"Expected geometry RGB {expected_rgb_shape}, got {tuple(geometry_rgb.shape)}")
        representative_rgb = geometry_rgb[:, :, 0]

        has_pointcloud = bool(batch["has_pointcloud"].reshape(-1)[0].item())
        visualization_kwargs = {}
        if has_pointcloud:
            gt_points = batch["geometry_pts3d"][:1].detach().cpu().float()
            gt_point_valid = batch["geometry_point_valid_mask"][:1].detach().cpu().bool()
            gt_depth, gt_depth_valid, _ = normalize_depth_targets(gt_points, gt_point_valid)
            gt_depth = gt_depth[:, :, 0]
            gt_depth_valid = gt_depth_valid[:, :, 0]
            if tuple(gt_depth.shape) != tuple(pred_depth.shape):
                raise ValueError(
                    "Pred/GT representative depth mismatch: "
                    f"pred={tuple(pred_depth.shape)}, gt={tuple(gt_depth.shape)}"
                )
            visualization_kwargs = {
                "gt_depth": gt_depth[0].numpy(),
                "gt_valid_mask": gt_depth_valid[0].numpy(),
            }

        frames = _depth_visualization_frames(
            rgb=representative_rgb[0].numpy(),
            pred_depth=pred_depth[0].numpy(),
            pred_conf=pred_depth_conf[0].numpy(),
            **visualization_kwargs,
        )
        filename = (
            "depth_rgb_gt_pred_diff_conf.mp4"
            if has_pointcloud
            else "depth_rgb_pred_conf.mp4"
        )
        imageio.mimsave(sample_dir / filename, frames, fps=int(self.config.inference_video_fps))
        return filename


def _save_pointcloud_plys(
    *,
    points: torch.Tensor,
    point_valid: torch.Tensor,
    rgb: torch.Tensor,
    group_valid: torch.Tensor,
    sample_dir: Path,
    directory_name: str,
) -> list[dict[str, Any]]:
    points = points.detach().cpu().float().numpy()
    point_valid = point_valid.detach().cpu().bool().numpy()
    rgb = rgb.detach().cpu().float()
    group_valid = group_valid.detach().cpu().bool().numpy()
    if points.ndim != 6 or point_valid.shape != points.shape[:-1]:
        raise ValueError(f"Unexpected pointcloud shape: points={points.shape}, valid={point_valid.shape}")
    if rgb.shape[:3] != points.shape[:3] or rgb.shape[3] != 3:
        raise ValueError(f"Unexpected pointcloud RGB shape: points={points.shape}, rgb={tuple(rgb.shape)}")
    if group_valid.shape != points.shape[:2]:
        raise ValueError(f"Unexpected pointcloud group mask: points={points.shape}, groups={group_valid.shape}")

    colors = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 1, 2, 4, 5, 3).numpy()
    output_dir = sample_dir / directory_name
    records = []
    for group_idx, slot_idx in np.argwhere(group_valid):
        for view_idx in range(points.shape[2]):
            filename = f"group_{group_idx:02d}_slot_{slot_idx}_view_{view_idx}.ply"
            path = output_dir / filename
            vertex_count = write_ply(
                path,
                points[group_idx, slot_idx, view_idx],
                colors[group_idx, slot_idx, view_idx],
                point_valid[group_idx, slot_idx, view_idx],
            )
            records.append(
                {
                    "path": str(path.relative_to(sample_dir)),
                    "geometry_group": int(group_idx),
                    "geometry_slot": int(slot_idx),
                    "view": int(view_idx),
                    "num_vertices": int(vertex_count),
                }
            )
    return records


def _save_gt_pointcloud_plys(sample: dict[str, Any], sample_dir: Path) -> list[dict[str, Any]]:
    if not bool(torch.as_tensor(sample["has_pointcloud"]).item()):
        return []
    return _save_pointcloud_plys(
        points=sample["geometry_pts3d"],
        point_valid=sample["geometry_point_valid_mask"],
        rgb=sample["geometry_rgb"],
        group_valid=sample["geometry_group_valid_mask"],
        sample_dir=sample_dir,
        directory_name="gt_pointcloud",
    )


def _save_pred_pointcloud_plys(
    pred_points: torch.Tensor,
    pred_geometry_rgb: torch.Tensor,
    batch: dict[str, Any],
    sample_dir: Path,
) -> list[dict[str, Any]]:
    group_valid = batch["geometry_group_valid_mask"][:1].detach().cpu().bool()
    if group_valid.ndim != 3:
        raise ValueError(f"Expected geometry group mask [B,G,S], got {tuple(group_valid.shape)}")
    bsz, groups, group_size = group_valid.shape
    pred_points = pred_points[:1].detach().cpu().float()
    if pred_points.ndim != 6 or pred_points.shape[-1] != 3:
        raise ValueError(f"Expected pred points [B,G*S,V,H,W,3], got {tuple(pred_points.shape)}")
    _, frames, views, height, width, channels = pred_points.shape
    expected_shape = (bsz, groups * group_size, views, height, width, channels)
    if tuple(pred_points.shape) != expected_shape:
        raise ValueError(f"Expected pred points {expected_shape}, got {tuple(pred_points.shape)}")
    pred_points = pred_points.reshape(bsz, groups, group_size, views, height, width, channels)

    pred_geometry_rgb = pred_geometry_rgb[:1].detach().cpu().float()
    expected_rgb_shape = (bsz, groups, group_size, views, 3, height, width)
    if tuple(pred_geometry_rgb.shape) != expected_rgb_shape:
        raise ValueError(f"Expected pred geometry RGB {expected_rgb_shape}, got {tuple(pred_geometry_rgb.shape)}")

    if bool(batch["has_pointcloud"].reshape(-1)[0].item()):
        gt_points = batch["geometry_pts3d"][:1].detach().cpu().float()
        gt_point_valid = batch["geometry_point_valid_mask"][:1].detach().cpu().bool()
        if tuple(gt_points.shape) != tuple(pred_points.shape):
            raise ValueError(f"Expected GT points {tuple(pred_points.shape)}, got {tuple(gt_points.shape)}")
        _, point_valid, point_scale = normalize_point_targets(gt_points, gt_point_valid)
        pred_points = pred_points * point_scale[..., None]
    else:
        point_valid = torch.isfinite(pred_points).all(dim=-1)

    return _save_pointcloud_plys(
        points=pred_points[0],
        point_valid=point_valid[0],
        rgb=pred_geometry_rgb[0],
        group_valid=group_valid[0],
        sample_dir=sample_dir,
        directory_name="pred_pointcloud",
    )


def _safe_sample_name(row: dict[str, Any], idx: int) -> str:
    task = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in row.get("task_name", "task"))
    return f"{idx:02d}_{row.get('source_dataset', 'src')}_{task}_ep{int(row.get('episode_index', 0)):06d}"


def _resolve_transformer_path(checkpoint_root: str | Path) -> Path:
    transformer_path = Path(checkpoint_root)
    if transformer_path.name != "transformer":
        transformer_path = transformer_path / "transformer"
    if not transformer_path.is_dir():
        raise FileNotFoundError(f"Missing transformer checkpoint directory: {transformer_path}")
    return transformer_path


def _make_config(eval_cfg):
    dataset_root = Path(eval_cfg.dataset_root)
    mot_config = json.loads((dataset_root / "meta" / "mot_config.json").read_text(encoding="utf-8"))

    config = SimpleNamespace(
        action_dim=int(mot_config["action_dim"]),
        norm_stat=mot_config["norm_stat"],
        norm_stats_by_task=mot_config["norm_stats_by_task"],
        action_chunk_size=int(mot_config["action_chunk_size"]),
        video_downsample_ratio=int(mot_config["video_downsample_ratio"]),
        vae_temporal_factor=int(mot_config.get("vae_temporal_factor", 4)),
        empty_emb_path=str(dataset_root / "empty_emb.pt"),
        text_emb_cache_path=str(dataset_root / "text_emb_cache.pt"),
        action_cache_manifest_path=str(dataset_root / "cache" / "actions" / "action_cache_manifest.jsonl"),
        wan22_pretrained_model_name_or_path=str(Path(eval_cfg.wan22_model_root)),
        device=str(eval_cfg.device),
        masked_attn_backend=str(eval_cfg.masked_attn_backend),
        snr_shift=5.0,
        action_snr_shift=1.0,
        guidance_scale=float(eval_cfg.guidance_scale),
        action_guidance_scale=float(eval_cfg.action_guidance_scale),
        num_inference_steps=int(eval_cfg.num_inference_steps),
        action_num_inference_steps=int(eval_cfg.action_num_inference_steps),
        inference_video_fps=int(eval_cfg.inference_video_fps),
    )
    if config.num_inference_steps <= 0 or config.action_num_inference_steps <= 0:
        raise ValueError("MOT inference step counts must be positive")
    if config.inference_video_fps <= 0:
        raise ValueError("inference_video_fps must be positive")
    spec = mot_spec_from_config(config)
    if (
        spec.action_chunk_size != 48
        or spec.video_downsample_ratio != 4
        or spec.vae_temporal_factor != 4
        or spec.history_latent_frames != 4
        or spec.target_latent_frames != 4
        or spec.vae_input_frame_count != 13
    ):
        raise ValueError(f"Dataset does not match the current MOT inference protocol: {spec}")
    return config


def _load_transformer_checkpoint(transformer_path: Path):
    weight_path = transformer_path / SAFETENSORS_NAME
    if weight_path.is_file():
        return ThreeDVAMOTTransformer3DModel.from_pretrained(
            str(transformer_path),
            torch_dtype=MOT_INFERENCE_DTYPE,
        )

    checkpoint_root = transformer_path.parent
    dcp_dir = checkpoint_root / DCP_DIR_NAME
    required = (checkpoint_root / "_SUCCESS", dcp_dir / ".metadata")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint has neither {weight_path} nor a complete DCP model state; missing: {', '.join(missing)}"
        )

    transformer = ThreeDVAMOTTransformer3DModel.from_config(str(transformer_path))
    transformer.to(dtype=MOT_INFERENCE_DTYPE)
    transformer.vggto.keep_heads_fp32_()
    model_state = transformer.state_dict()
    dcp.load({"model": model_state}, checkpoint_id=dcp_dir, no_dist=True)
    transformer.load_state_dict(model_state, strict=True)
    return transformer


def _validate_transformer_dtypes(transformer) -> None:
    mismatches = []
    fp32_prefixes = ("vggto.dense_head.", "vggto.point_head.")
    for name, parameter in transformer.named_parameters():
        if not parameter.is_floating_point():
            continue
        expected = torch.float32 if name.startswith(fp32_prefixes) else MOT_INFERENCE_DTYPE
        if parameter.dtype != expected:
            mismatches.append(f"{name}: expected {expected}, got {parameter.dtype}")
            if len(mismatches) == 10:
                break
    if mismatches:
        raise TypeError("MOT inference model dtype mismatch: " + "; ".join(mismatches))


def _set_masked_attention_backend(transformer, backend: str) -> None:
    transformer.masked_attn_backend = backend
    transformer.vggto.masked_attn_backend = backend
    for block in getattr(transformer, "mot_blocks", ()):
        block.masked_attn_backend = backend


def _build_runner(checkpoint_root: str | Path, config):
    device = torch.device(config.device)
    backend = str(config.masked_attn_backend)
    if backend == "auto":
        backend = "fa4" if device.type == "cuda" else "dense"
    if backend not in {"fa4", "dense"}:
        raise ValueError(f"MOT evaluation backend must be 'auto', 'fa4', or 'dense', got {backend!r}")
    if backend == "fa4":
        validate_fa4_training_environment(device)

    transformer_path = _resolve_transformer_path(checkpoint_root)
    transformer = _load_transformer_checkpoint(transformer_path)
    _set_masked_attention_backend(transformer, backend)
    config.masked_attn_backend = backend
    transformer.to(device=device)
    transformer.vggto.keep_heads_fp32_()
    _validate_transformer_dtypes(transformer)
    transformer.eval()
    transformer.requires_grad_(False)

    return MOTInferenceSession(
        config=config,
        transformer=transformer,
        device=device,
    )


def _build_dataset(config, manifest_path: Path) -> MotTrainData:
    spec = mot_spec_from_config(config)
    return MotTrainData(
        manifest_path=manifest_path,
        action_sequence_length=spec.action_sequence_length,
        action_dim=config.action_dim,
        norm_stats_by_task=config.norm_stats_by_task,
        action_chunk_size=spec.action_chunk_size,
        video_downsample_ratio=spec.video_downsample_ratio,
        text_emb_cache_path=config.text_emb_cache_path,
        action_cache_manifest_path=config.action_cache_manifest_path,
        random_start=False,
    )


@torch.no_grad()
def _run_one_sample(
    runner: MOTInferenceSession,
    dataset: MotTrainData,
    sample_idx: int,
    output_dir: Path,
    *,
    mode: str,
) -> None:
    row = dataset.rows[sample_idx]
    sample_dir = output_dir / _safe_sample_name(row, sample_idx)
    sample_dir.mkdir(parents=True, exist_ok=True)
    start_frame = int(row["inference_start_frame"])
    sample = dataset.get_window(sample_idx, start_frame)
    gt_pointcloud_plys = (
        _save_gt_pointcloud_plys(sample, sample_dir)
        if mode in {"geometry", "full"}
        else []
    )
    batch = runner.move_batch_to_device(default_collate([sample]))
    if mode in {"video", "full"}:
        batch = runner.materialize_batch_latents(batch)
    spec = mot_spec_from_config(runner.config)
    frame_count = spec.total_latent_frames
    result = runner.run(batch, frame_count, mode=mode)
    pred_pointcloud_plys = []
    depth_vis = None

    torch.save(
        batch["geometry_rgb"][:, :frame_count, 0].detach().cpu(),
        sample_dir / "gt_rgb_context.pt",
    )
    if mode in {"video", "full"}:
        torch.save(result.pred_latents.detach().cpu(), sample_dir / "pred_latents.pt")
        torch.save(result.pred_rgb.detach().cpu(), sample_dir / "pred_rgb_context.pt")
        torch.save(
            batch["latents"][:, :, :frame_count].detach().cpu(),
            sample_dir / "gt_latents.pt",
        )
        runner.save_video(
            result.pred_latents,
            torch.cat(
                [batch["vae_rgb_history"], batch["vae_rgb_target"]],
                dim=1,
            ),
            sample_dir,
        )

    if mode in {"geometry", "full"}:
        geometry_rgb = (
            result.geometry_rgb
            if mode == "geometry"
            else result.action_geometry_rgb
        )
        torch.save(result.pred_depth.detach().cpu(), sample_dir / "pred_depth.pt")
        torch.save(result.pred_points.detach().cpu(), sample_dir / "pred_points.pt")
        pred_pointcloud_plys = _save_pred_pointcloud_plys(
            result.pred_points,
            geometry_rgb,
            batch,
            sample_dir,
        )
        depth_vis = runner.save_depth_video(
            result.pred_depth,
            result.pred_depth_conf,
            geometry_rgb,
            batch,
            sample_dir,
        )

    if mode == "full":
        gt_actions = batch["actions"]
        torch.save(
            result.pred_actions.detach().cpu(),
            sample_dir / "pred_actions_norm.pt",
        )
        torch.save(gt_actions.detach().cpu(), sample_dir / "gt_actions_norm.pt")
        if "action_loss_mask" in batch:
            torch.save(
                batch["action_loss_mask"].detach().cpu(),
                sample_dir / "action_loss_mask.pt",
            )
        runner.save_action_plot(
            result.pred_actions,
            gt_actions,
            sample_dir,
            batch=batch,
        )
    metadata = {
        **row,
        "inference_mode": mode,
        "gt_pointcloud_plys": gt_pointcloud_plys,
        "pred_pointcloud_plys": pred_pointcloud_plys,
        "depth_vis": depth_vis,
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved MOT inference artifacts to {sample_dir}")


def _eval_cfg_namespace(eval_cfg) -> SimpleNamespace:
    if isinstance(eval_cfg, dict):
        return SimpleNamespace(**eval_cfg)
    if hasattr(eval_cfg, "items"):
        return SimpleNamespace(**dict(eval_cfg.items()))
    return SimpleNamespace(**vars(eval_cfg))


def _run_evaluation_artifacts(
    checkpoint_root: str | Path,
    eval_cfg,
    output_dir: str | Path,
) -> None:
    eval_cfg = _eval_cfg_namespace(eval_cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_model_paths(checkpoint_root, eval_cfg.wan22_model_root, output_dir)

    dataset_root = Path(eval_cfg.dataset_root)
    config = _make_config(eval_cfg)
    rng = random.Random(int(eval_cfg.seed))
    selected_rows = _select_inference_rows(
        dataset_root=dataset_root,
        action_chunk_size=config.action_chunk_size,
        rng=rng,
    )
    for row in selected_rows:
        row["inference_group"] = "pointcloud" if row["has_pointcloud"] else "non_pointcloud"
        row["inference_start_frame"] = _sample_full_window_start(
            row,
            action_chunk_size=config.action_chunk_size,
            rng=rng,
        )
    selected_manifest = output_dir / "selected_samples.jsonl"
    _write_jsonl(selected_manifest, selected_rows)

    torch.manual_seed(int(eval_cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(eval_cfg.seed))
    runner = _build_runner(checkpoint_root, config)
    dataset = _build_dataset(config, selected_manifest)
    group_dir = output_dir / "dataset"
    for idx in range(len(dataset)):
        _run_one_sample(runner, dataset, idx, group_dir, mode=str(eval_cfg.mode))
    logger.info(f"Completed MOT inference: {output_dir}")


def run_checkpoint_evaluation(
    checkpoint_root: str | Path,
    eval_cfg,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    checkpoint_root = Path(checkpoint_root)
    eval_cfg = _eval_cfg_namespace(eval_cfg)
    if eval_cfg.mode not in {"video", "geometry", "full"}:
        raise ValueError(f"unsupported inference mode={eval_cfg.mode!r}")

    transformer_path = _resolve_transformer_path(checkpoint_root)
    weight_path = transformer_path / SAFETENSORS_NAME
    if not weight_path.is_file():
        convert_dcp_to_safetensors(checkpoint_root)
        gc.collect()

    if output_dir is None:
        configured_root = getattr(eval_cfg, "output_root", None)
        output_root = Path(configured_root) if configured_root else _REPO_ROOT / "inference_logs"
        output_dir = output_root / checkpoint_root.name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    success_marker = output_dir / "_SUCCESS"
    success_marker.unlink(missing_ok=True)
    _run_evaluation_artifacts(checkpoint_root, eval_cfg, output_dir)
    success_marker.touch()
    return output_dir


def _default_eval_cfg():
    dataset_root = Path(os.getenv("MOT_DATASET_ROOT", str(MOT_DATASET_ROOT)))
    wan22_model_root = os.getenv(
        "WAN22_MODEL_ROOT",
        os.getenv("WAN22_PRETRAINED_MODEL_PATH", DEFAULT_WAN22_MODEL_ROOT),
    )
    return make_mot_eval_cfg(
        dataset_root=dataset_root,
        wan22_model_root=wan22_model_root,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run current fixed-window MOT checkpoint evaluation.")
    parser.add_argument("--checkpoint-root", required=True, help="Checkpoint root containing transformer/")
    parser.add_argument(
        "--mode",
        choices=("video", "geometry", "full"),
        default=None,
        help="Override eval_cfg.mode.",
    )
    parser.add_argument(
        "--eval-cfg-json",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help="Override eval_cfg.device for manual CPU/GPU evaluation.")
    return parser.parse_args(argv)


def main():
    init_logger()
    args = parse_args()
    eval_cfg = (
        _eval_cfg_namespace(json.loads(args.eval_cfg_json))
        if args.eval_cfg_json is not None
        else _default_eval_cfg()
    )
    if args.mode is not None:
        eval_cfg.mode = args.mode
    if args.device is not None:
        eval_cfg.device = args.device

    checkpoint_root = Path(args.checkpoint_root)
    marker = checkpoint_root / EVAL_IN_PROGRESS_MARKER
    marker.touch(exist_ok=True)
    try:
        run_checkpoint_evaluation(
            checkpoint_root,
            eval_cfg,
            output_dir=args.output_dir,
        )
    finally:
        marker.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
