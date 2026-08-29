#!/usr/bin/env python
"""Fixed-window Video+Action MOT inference on prepared LeRobot rows."""
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

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.utils.data import default_collate

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.convert_dcp_to_safetensors import convert_dcp_to_safetensors
from inference.mot_inference import (
    MOT_INFERENCE_DTYPE,
    run_mot_inference,
    run_video_inference,
)
from wan_va.configs.mot_eval_cfg import make_mot_eval_cfg
from wan_va.dataset import MotTrainData
from wan_va.modules.fa4_attention import validate_fa4_training_environment
from wan_va.modules.model_va_mot import VAMOTTransformer3DModel
from wan_va.modules.utils import WanVAEStreamingWrapper, load_vae
from wan_va.mot_spec import mot_spec_from_config
from wan_va.utils import init_logger, logger

DEFAULT_WAN22_MODEL_ROOT = (
    "/workspace/cache/huggingface_cache/hub/"
    "models--robbyant--lingbot-va-base/snapshots/"
    "68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c"
)
MOT_DATASET_ROOT = _REPO_ROOT / "data" / "data" / "umi_mot_full_data_train_0712_final"
TRAINING_MANIFEST = "mot_final_training_manifest.jsonl"
EVAL_SOURCE_DATASETS = ("lumos_lerobot", "genrobot_lerobot")
DCP_DIR_NAME = "distributed_state"
SAFETENSORS_NAME = "diffusion_pytorch_model.safetensors"


def _full_window_start_bounds(
    row: dict[str, Any], *, action_chunk_size: int
) -> tuple[int, int] | None:
    valid_start, valid_end = (int(value) for value in row["valid_start_range"])
    segment = row["segment"]
    start = max(valid_start, int(segment["start_frame"]) + int(action_chunk_size) + 1)
    end = min(valid_end, int(segment["end_frame"]) - int(action_chunk_size) - 1)
    return None if end < start else (start, end)


def _sample_full_window_start(
    row: dict[str, Any], *, action_chunk_size: int, rng: random.Random
) -> int:
    bounds = _full_window_start_bounds(row, action_chunk_size=action_chunk_size)
    if bounds is None:
        raise ValueError(
            "Selected episode has no complete inference window: "
            f"episode={row.get('episode_index')} range={row['valid_start_range']}"
        )
    return rng.randint(*bounds)


def _reservoir_sample_manifest(
    path: Path,
    *,
    count: int,
    source_dataset: str,
    action_chunk_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_dataset") != source_dataset:
                continue
            if _full_window_start_bounds(row, action_chunk_size=action_chunk_size) is None:
                continue
            seen += 1
            if len(selected) < count:
                selected.append(row)
            else:
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
    source_datasets: tuple[str, ...] = EVAL_SOURCE_DATASETS,
) -> list[dict[str, Any]]:
    manifest = dataset_root / "meta" / TRAINING_MANIFEST
    rows: list[dict[str, Any]] = []
    for source_dataset in source_datasets:
        rows.extend(
            _reservoir_sample_manifest(
                manifest,
                count=1,
                source_dataset=source_dataset,
                action_chunk_size=action_chunk_size,
                rng=rng,
            )
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _views_to_wide_frames(rgb_views: torch.Tensor) -> np.ndarray:
    if rgb_views.ndim == 6:
        rgb_views = rgb_views[0]
    frames = rgb_views.detach().cpu().float().permute(0, 3, 1, 4, 2).contiguous()
    frames = frames.reshape(
        frames.shape[0], frames.shape[1], frames.shape[2] * frames.shape[3], frames.shape[4]
    )
    return np.clip(frames.numpy() * 255.0, 0, 255).astype(np.uint8)


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
            raise ValueError(
                f"MOT VAE input must have {spec.vae_input_frame_count} frames, got {frames.shape[0]}"
            )
        vae = self._get_vae()
        wrapper = WanVAEStreamingWrapper(vae)
        video = frames.permute(1, 0, 2, 3)[None].to(device=self.device, dtype=vae.dtype) * 2 - 1
        chunks = [wrapper.encode_chunk(video[:, :, :1])]
        for index in range(1, spec.latent_frames_per_action_chunk_per_view):
            start = 1 + spec.vae_temporal_factor * (index - 1)
            chunks.append(wrapper.encode_chunk(video[:, :, start : start + spec.vae_temporal_factor]))
        mu, _logvar = torch.chunk(torch.cat(chunks, dim=2), 2, dim=1)
        mean = torch.tensor(vae.config.latents_mean, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
        return ((mu - mean) / std)[0].to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _encode_vae_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        per_batch = []
        for batch_index in range(rgb.shape[0]):
            per_batch.append(
                torch.stack(
                    [
                        self._encode_one_view_latent(rgb[batch_index, :, view_index])
                        for view_index in range(rgb.shape[2])
                    ],
                    dim=2,
                )
            )
        return torch.stack(per_batch)

    @torch.no_grad()
    def materialize_batch_latents(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "latents" in batch:
            return batch
        out = dict(batch)
        out["latents"] = torch.cat(
            [self._encode_vae_rgb(batch["vae_rgb_history"]), self._encode_vae_rgb(batch["vae_rgb_target"])],
            dim=2,
        )
        return out

    @torch.no_grad()
    def decode_latents_to_rgb_views(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self._get_vae()
        batch, channels, frames, views, height, width = latents.shape
        packed = latents.permute(0, 3, 1, 2, 4, 5).reshape(
            batch * views, channels, frames, height, width
        ).to(device=self.device, dtype=vae.dtype)
        mean = torch.tensor(vae.config.latents_mean, device=self.device, dtype=packed.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=self.device, dtype=packed.dtype).view(1, -1, 1, 1, 1)
        decoded = vae.decode(packed * std + mean, return_dict=False)[0]
        decoded = (decoded.float() * 0.5 + 0.5).clamp(0, 1)
        return decoded.permute(0, 2, 1, 3, 4).reshape(
            batch, views, decoded.shape[2], decoded.shape[1], decoded.shape[3], decoded.shape[4]
        ).permute(0, 2, 1, 3, 4, 5).contiguous()

    def run(self, batch: dict[str, Any], frame_count: int, *, mode: str):
        common = dict(
            transformer=self.transformer,
            config=self.config,
            spec=mot_spec_from_config(self.config),
            device=self.device,
            empty_text_emb=self._get_empty_text_emb(),
            decode_latents_to_rgb_views=self.decode_latents_to_rgb_views,
        )
        if mode == "video":
            return run_video_inference(batch, frame_count, **common)
        if mode == "full":
            return run_mot_inference(batch, frame_count, **common)
        raise ValueError(f"unsupported inference mode={mode!r}")

    @torch.no_grad()
    def save_video(self, pred_latents: torch.Tensor, sample_dir: Path) -> None:
        frames = _views_to_wide_frames(self.decode_latents_to_rgb_views(pred_latents[:1]))
        if len(frames):
            import imageio.v2 as imageio

            imageio.mimsave(
                sample_dir / "video_pred.mp4",
                frames,
                fps=int(self.config.inference_video_fps),
            )


def _resolve_transformer_path(checkpoint_root: str | Path) -> Path:
    path = Path(checkpoint_root)
    if path.name != "transformer":
        path = path / "transformer"
    if not path.is_dir():
        raise FileNotFoundError(f"Missing transformer checkpoint directory: {path}")
    return path


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
    mot_spec_from_config(config)
    return config


def _load_transformer_checkpoint(transformer_path: Path):
    weight_path = transformer_path / SAFETENSORS_NAME
    if weight_path.is_file():
        return VAMOTTransformer3DModel.from_pretrained(
            str(transformer_path), torch_dtype=MOT_INFERENCE_DTYPE
        )
    checkpoint_root = transformer_path.parent
    dcp_dir = checkpoint_root / DCP_DIR_NAME
    transformer = VAMOTTransformer3DModel.from_config(str(transformer_path))
    transformer.to(dtype=MOT_INFERENCE_DTYPE)
    model_state = transformer.state_dict()
    dcp.load({"model": model_state}, checkpoint_id=dcp_dir, no_dist=True)
    transformer.load_state_dict(model_state, strict=True)
    return transformer


def _validate_transformer_dtypes(transformer) -> None:
    mismatches = [
        f"{name}: {parameter.dtype}"
        for name, parameter in transformer.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != MOT_INFERENCE_DTYPE
    ]
    if mismatches:
        raise TypeError("MOT inference model dtype mismatch: " + "; ".join(mismatches[:10]))


def _set_masked_attention_backend(transformer, backend: str) -> None:
    transformer.masked_attn_backend = backend
    for block in getattr(transformer, "mot_blocks", ()):
        block.masked_attn_backend = backend


def _build_runner(checkpoint_root: str | Path, config):
    device = torch.device(config.device)
    backend = str(config.masked_attn_backend)
    if backend == "auto":
        backend = "fa4" if device.type == "cuda" else "dense"
    if backend == "fa4":
        validate_fa4_training_environment(device)
    transformer = _load_transformer_checkpoint(_resolve_transformer_path(checkpoint_root))
    _set_masked_attention_backend(transformer, backend)
    config.masked_attn_backend = backend
    transformer.to(device=device).eval().requires_grad_(False)
    _validate_transformer_dtypes(transformer)
    return MOTInferenceSession(config=config, transformer=transformer, device=device)


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
def _run_one_sample(runner, dataset, sample_idx: int, output_dir: Path, *, mode: str) -> None:
    row = dataset.rows[sample_idx]
    sample_dir = output_dir / f"{sample_idx:02d}_{row.get('source_dataset', 'src')}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample = dataset.get_window(sample_idx, int(row["inference_start_frame"]))
    batch = runner.materialize_batch_latents(
        runner.move_batch_to_device(default_collate([sample]))
    )
    frame_count = mot_spec_from_config(runner.config).total_latent_frames
    result = runner.run(batch, frame_count, mode=mode)
    torch.save(result.pred_latents.detach().cpu(), sample_dir / "pred_latents.pt")
    torch.save(result.pred_rgb.detach().cpu(), sample_dir / "pred_rgb.pt")
    runner.save_video(result.pred_latents, sample_dir)
    if mode == "full":
        torch.save(result.pred_actions.detach().cpu(), sample_dir / "pred_actions.pt")
    (sample_dir / "metadata.json").write_text(
        json.dumps({**row, "inference_mode": mode}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _eval_cfg_namespace(eval_cfg) -> SimpleNamespace:
    if isinstance(eval_cfg, dict):
        return SimpleNamespace(**eval_cfg)
    if hasattr(eval_cfg, "items"):
        return SimpleNamespace(**dict(eval_cfg.items()))
    return SimpleNamespace(**vars(eval_cfg))


def _run_evaluation_artifacts(checkpoint_root, eval_cfg, output_dir) -> None:
    eval_cfg = _eval_cfg_namespace(eval_cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _make_config(eval_cfg)
    rng = random.Random(int(eval_cfg.seed))
    rows = _select_inference_rows(
        dataset_root=Path(eval_cfg.dataset_root),
        action_chunk_size=config.action_chunk_size,
        rng=rng,
        source_datasets=tuple(
            getattr(eval_cfg, "source_datasets", EVAL_SOURCE_DATASETS)
        ),
    )
    for row in rows:
        row["inference_start_frame"] = _sample_full_window_start(
            row, action_chunk_size=config.action_chunk_size, rng=rng
        )
    manifest = output_dir / "selected_samples.jsonl"
    _write_jsonl(manifest, rows)
    runner = _build_runner(checkpoint_root, config)
    dataset = _build_dataset(config, manifest)
    for index in range(len(dataset)):
        _run_one_sample(runner, dataset, index, output_dir / "dataset", mode=eval_cfg.mode)


def run_checkpoint_evaluation(checkpoint_root, eval_cfg, *, output_dir=None) -> Path:
    checkpoint_root = Path(checkpoint_root)
    eval_cfg = _eval_cfg_namespace(eval_cfg)
    if eval_cfg.mode not in {"video", "full"}:
        raise ValueError(f"unsupported inference mode={eval_cfg.mode!r}")
    transformer_path = _resolve_transformer_path(checkpoint_root)
    if not (transformer_path / SAFETENSORS_NAME).is_file():
        convert_dcp_to_safetensors(checkpoint_root)
        gc.collect()
    if output_dir is None:
        root = Path(getattr(eval_cfg, "output_root", None) or _REPO_ROOT / "inference_logs")
        output_dir = root / checkpoint_root.name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_evaluation_artifacts(checkpoint_root, eval_cfg, output_dir)
    (output_dir / "_SUCCESS").touch()
    return output_dir


def _default_eval_cfg():
    return make_mot_eval_cfg(
        dataset_root=Path(os.getenv("MOT_DATASET_ROOT", str(MOT_DATASET_ROOT))),
        wan22_model_root=os.getenv("WAN22_MODEL_ROOT", DEFAULT_WAN22_MODEL_ROOT),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run fixed-window Video+Action evaluation")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--mode", choices=("video", "full"), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--source-dataset",
        dest="source_datasets",
        action="append",
        default=None,
        help="Manifest source_dataset to sample; repeat to select multiple sources",
    )
    return parser.parse_args(argv)


def main():
    init_logger()
    args = parse_args()
    config = _default_eval_cfg()
    if args.mode is not None:
        config.mode = args.mode
    if args.device is not None:
        config.device = args.device
    if args.source_datasets is not None:
        config.source_datasets = tuple(args.source_datasets)
    output = run_checkpoint_evaluation(
        args.checkpoint_root, config, output_dir=args.output_dir
    )
    logger.info(f"Completed MOT inference: {output}")


if __name__ == "__main__":
    main()
