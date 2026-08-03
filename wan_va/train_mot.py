# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
3DVA_MOT post-training entrypoint.

The current MOT trainer uses the fixed-window protocol: each dataset item
contains one clean history chunk plus one target chunk, video latents are encoded
online from raw RGB with the frozen Wan VAE, and diffusion timesteps are sampled
per `[batch, target_chunk]`.

* the model is `ThreeDVAMOTTransformer3DModel`, initialized from LingBot-VA and
  VGGTO checkpoints;
* the dataset is `MotTrainData`, returning VAE RGB windows plus
  representative RGB/depth labels for the VGGTO geometry branch;
* the loss combines LingBot video/action flow-matching losses with a VGGT-style
  depth loss, with video/action loss masked off for clean history frames.
"""

import argparse
from bisect import bisect_left
import ctypes
from functools import partial
import gc
import os
import random
import socket
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import json

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    get_model_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader, Sampler
from safetensors.torch import save_file
from tqdm import tqdm

_WAN_VA_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _WAN_VA_DIR.parent
for _path in (str(_REPO_ROOT), str(_WAN_VA_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from configs import VA_CONFIGS
from dataset import (
    MotBalancedMixDataset,
    MotGeometryLeRobotData,
    MotPureLeRobotData,
    validate_mot_batch_for_forward,
    validate_mot_geometry_batch,
)
from dataset.mot_dataset import (
    MOT_DEFAULT_ACTION_CACHE_SIZE,
    MOT_DEFAULT_POINT_STORE_CACHE_SIZE,
    MOT_DEFAULT_VIDEO_DECODER_CACHE_SIZE,
    MOT_SUPPORTED_VIEW_COUNTS,
    decode_mot_runtime_cache_stats,
)
from distributed.util import _configure_model, init_distributed
from modules.fa4_attention import validate_fa4_training_environment
from modules.model_3dva_mot import ThreeDVAMOTTransformer3DModel
from modules.vggto_loss import (
    VGGT_MONITOR_ERROR_HISTOGRAM_BINS,
    compute_vggto_depth_loss,
    compute_vggto_point_loss,
    geometry_error_quantiles_from_histogram,
    geometry_pearson_from_stats,
)
from modules.utils import WanVAEStreamingWrapper, load_vae
from wan_va.checkpoint_eval import CheckpointEvaluationQueue
from wan_va.checkpoint_retention import prune_successful_checkpoints
from wan_va.mot_spec import (
    MOTWindowSpec,
    mot_spec_from_config as _mot_spec_from_config,
)
from utils import FlowMatchScheduler, init_logger, logger, sample_timestep_id, warmup_constant_lambda


MOT_DCP_DIR_NAME = "distributed_state"
MOT_DCP_FORMAT_VERSION = 2
MOT_CHECKPOINT_METADATA_NAME = "checkpoint_metadata.json"
MOT_TRANSFORMER_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
MOT_VGGTO_TOPOLOGY = "synchronized_multiview_v1"
MOT_OPTIMIZATION_COMPOSITIONS = ("g", "v", "a", "vg", "va", "ag", "vag")
MOT_PERFORMANCE_TIMING_KEYS = (
    "data_fetch",
    "data_barrier",
    "input_transfer",
    "vae_encode",
    "input_prepare",
    "model_forward",
    "loss",
    "loss_finite_sync",
    "backward",
    "grad_clip",
    "grad_finite_sync",
    "optimizer_step",
    "metrics_logging",
    "checkpoint",
    "post_step_barrier",
    "gc_collect",
    "post_gc_barrier",
    "iteration",
)

MOT_GEOMETRY_LOG_KEYS = (
    "depth_loss_raw",
    "point_loss_raw",
    "loss_conf_depth",
    "loss_reg_depth",
    "loss_grad_depth",
    "depth_conf_mean",
    "depth_conf_max",
    "loss_conf_point",
    "loss_reg_point",
    "loss_grad_point",
    "point_conf_mean",
    "point_conf_max",
    "depth_error_p50",
    "depth_error_p90",
    "point_error_p50",
    "point_error_p90",
    "point_normal_angle_mean",
    "depth_conf_error_correlation",
    "point_conf_error_correlation",
)
MOT_GEOMETRY_GLOBAL_DETAIL_KEYS = (
    "depth_error_p50",
    "depth_error_p90",
    "point_error_p50",
    "point_error_p90",
    "point_normal_angle_mean",
    "depth_conf_error_correlation",
    "point_conf_error_correlation",
)
MOT_GEOMETRY_MAX_LOG_KEYS = (
    "depth_conf_max",
    "point_conf_max",
)

_PROC_STATUS_MEMORY_FIELDS = {
    "VmRSS": "rss_bytes",
    "VmLck": "locked_bytes",
    "VmPin": "pinned_bytes",
    "RssAnon": "rss_anon_bytes",
    "RssFile": "rss_file_bytes",
    "RssShmem": "rss_shmem_bytes",
    "VmData": "vm_data_bytes",
    "VmSwap": "vm_swap_bytes",
}
_HOST_MEMORY_FIELDS = {
    "MemTotal": "total_bytes",
    "MemAvailable": "available_bytes",
    "Cached": "cached_bytes",
    "AnonPages": "anon_bytes",
    "Mapped": "mapped_bytes",
    "Shmem": "shmem_bytes",
    "Unevictable": "unevictable_bytes",
    "Mlocked": "mlocked_bytes",
    "Slab": "slab_bytes",
    "SReclaimable": "slab_reclaimable_bytes",
    "SUnreclaim": "slab_unreclaimable_bytes",
    "KernelStack": "kernel_stack_bytes",
    "PageTables": "pagetables_bytes",
}
_SMAPS_ROLLUP_FIELDS = {
    "Pss": "pss_bytes",
    "Pss_Anon": "pss_anon_bytes",
    "Pss_File": "pss_file_bytes",
    "Pss_Shmem": "pss_shmem_bytes",
    "Private_Dirty": "private_dirty_bytes",
    "Shared_Dirty": "shared_dirty_bytes",
    "Anonymous": "anonymous_bytes",
    "Locked": "locked_bytes",
}
_CGROUP_MEMORY_STAT_FIELDS = (
    "anon",
    "file",
    "shmem",
    "file_mapped",
    "inactive_file",
    "active_file",
    "pagetables",
    "slab",
)


class _Mallinfo2(ctypes.Structure):
    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]


try:
    _MALLINFO2_FUNC = ctypes.CDLL(None).mallinfo2
    _MALLINFO2_FUNC.argtypes = []
    _MALLINFO2_FUNC.restype = _Mallinfo2
except (AttributeError, OSError):
    _MALLINFO2_FUNC = None


def _seed_mot_training(seed: int) -> None:
    """Seed model initialization and per-rank training RNGs reproducibly."""

    seed = int(seed)
    if seed < 0:
        raise ValueError(f"MOT training seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_proc_key_values(path: Path, fields: dict[str, str]) -> dict[str, int]:
    result = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                key, separator, value = line.partition(":")
                output_key = fields.get(key)
                if not separator or output_key is None:
                    continue
                parts = value.split()
                if parts:
                    result[output_key] = int(parts[0]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
        return result
    return result


def _read_process_memory(pid: int, *, include_expensive: bool = False) -> dict[str, Any]:
    pid = int(pid)
    proc_root = Path(f"/proc/{pid}")
    result: dict[str, Any] = {"pid": pid}
    result.update(_read_proc_key_values(proc_root / "status", _PROC_STATUS_MEMORY_FIELDS))
    try:
        with (proc_root / "status").open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Threads:"):
                    result["threads"] = int(line.split()[1])
                    break
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError, IndexError):
        pass
    if not include_expensive:
        return result

    result.update(_read_proc_key_values(proc_root / "smaps_rollup", _SMAPS_ROLLUP_FIELDS))
    try:
        with os.scandir(proc_root / "fd") as entries:
            result["fd_count"] = sum(1 for _ in entries)
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        pass
    try:
        with (proc_root / "maps").open("r", encoding="utf-8") as f:
            result["vma_count"] = sum(1 for _ in f)
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        pass
    return result


def _read_host_memory() -> dict[str, int]:
    return _read_proc_key_values(Path("/proc/meminfo"), _HOST_MEMORY_FIELDS)


def _read_allocator_stats() -> dict[str, Any]:
    result: dict[str, Any] = {"python_allocated_blocks": int(sys.getallocatedblocks())}
    if _MALLINFO2_FUNC is None:
        return result
    info = _MALLINFO2_FUNC()
    result.update({name: int(getattr(info, name)) for name, _ in _Mallinfo2._fields_})
    return result


def _read_gc_stats() -> dict[str, Any]:
    return {
        "enabled": gc.isenabled(),
        "count": list(gc.get_count()),
        "threshold": list(gc.get_threshold()),
        "generations": [dict(stats) for stats in gc.get_stats()],
    }


def _cgroup_v2_root() -> Path | None:
    try:
        with Path("/proc/self/cgroup").open("r", encoding="utf-8") as f:
            for line in f:
                hierarchy, controllers, relative = line.rstrip("\n").split(":", 2)
                if hierarchy == "0" and not controllers:
                    return Path("/sys/fs/cgroup") / relative.lstrip("/")
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def _read_int_file(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_space_separated_ints(path: Path) -> dict[str, int]:
    result = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split()
                result[key] = int(value)
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return result
    return result


def _read_memory_pressure(path: Path) -> dict[str, dict[str, float | int]]:
    result = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                values: dict[str, float | int] = {}
                for item in parts[1:]:
                    key, value = item.split("=", 1)
                    values[key] = int(value) if key == "total" else float(value)
                result[parts[0]] = values
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        return result
    return result


def _read_cgroup_memory() -> dict[str, Any] | None:
    root = _cgroup_v2_root()
    if root is None:
        return None
    memory_stat = _read_space_separated_ints(root / "memory.stat")
    return {
        "path": str(root),
        "current_bytes": _read_int_file(root / "memory.current"),
        "max_bytes": _read_int_file(root / "memory.max"),
        "stat": {key: memory_stat[key] for key in _CGROUP_MEMORY_STAT_FIELDS if key in memory_stat},
        "events": _read_space_separated_ints(root / "memory.events"),
        "pressure": _read_memory_pressure(root / "memory.pressure"),
    }


def _process_rss_mib(pid: int) -> float:
    """Read one process's resident memory without adding a psutil dependency."""

    return float(_read_process_memory(pid).get("rss_bytes", 0)) / (1024.0**2)


def _wandb_init_settings(config):
    save_root = Path(config.save_root)
    wandb_dir = save_root / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    mode = os.getenv("WANDB_MODE", getattr(config, "wandb_mode", "offline"))
    os.environ["WANDB_MODE"] = mode
    os.environ["WANDB_DIR"] = str(save_root)
    return {
        "entity": os.environ.get("WANDB_TEAM_NAME"),
        "project": os.getenv("WANDB_PROJECT", "va_robotwin"),
        "config": config,
        "mode": mode,
        "name": getattr(config, "wandb_name", "umi_3dwam_train"),
        # W&B creates its local `wandb/` directory under this root, so run files
        # land in `<save_root>/wandb`.
        "dir": str(save_root),
    }, wandb_dir, mode


def resolve_optimization_composition(value: Any) -> str:
    composition = str(value).strip().lower()
    if composition not in MOT_OPTIMIZATION_COMPOSITIONS:
        raise ValueError(
            "optimization_composition must be one of "
            f"{MOT_OPTIMIZATION_COMPOSITIONS}, got {value!r}"
        )
    return composition


def optimization_branches(composition: str) -> frozenset[str]:
    return frozenset(resolve_optimization_composition(composition))


def _mot_execution_route(composition: str) -> str:
    return "geometry" if resolve_optimization_composition(composition) == "g" else "joint"


class _MOTViewBatchSampler(Sampler[list[int]]):
    """Build deterministic view-homogeneous local batches for one rank."""

    def __init__(
        self,
        dataset,
        *,
        num_replicas: int,
        rank: int,
        num_microsteps: int,
        max_views_per_gpu: int,
        pointcloud_sample_period: int | None = None,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}")
        self.num_microsteps = int(num_microsteps)
        if self.num_microsteps <= 0:
            raise ValueError(f"num_microsteps must be positive, got {self.num_microsteps}")
        self.max_views_per_gpu = int(max_views_per_gpu)
        if self.max_views_per_gpu < 3:
            raise ValueError(
                "max_views_per_gpu must be at least 3, "
                f"got {self.max_views_per_gpu}"
            )
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.start_step = 0

        if isinstance(dataset, MotBalancedMixDataset):
            self._mixed = True
            self._source_buckets = {
                0: dataset.pointcloud_dataset.view_buckets,
                1: dataset.pure_dataset.view_buckets,
            }
            pointcloud_sample_period = int(pointcloud_sample_period or 0)
            if pointcloud_sample_period <= 0:
                raise ValueError(
                    "pointcloud_sample_period must be positive for mixed MOT training"
                )
            self.pointcloud_sample_period = pointcloud_sample_period
        else:
            self._mixed = False
            self._source_buckets = {0: dataset.view_buckets}
            self.pointcloud_sample_period = None

        domains = [set(buckets) for buckets in self._source_buckets.values()]
        if not domains or not domains[0]:
            raise ValueError("MOT dataset must expose at least one nonempty view bucket")
        if any(domain != domains[0] for domain in domains[1:]):
            raise ValueError(
                "pointcloud and non-pointcloud sources must expose the same "
                "nonempty native-view domain"
            )
        unsupported = domains[0].difference(MOT_SUPPORTED_VIEW_COUNTS)
        if unsupported:
            raise ValueError(f"unsupported native view counts: {sorted(unsupported)}")
        self.available_view_counts = tuple(sorted(domains[0]))
        for view_count in self.available_view_counts:
            global_batch_size = (
                self.num_replicas * (self.max_views_per_gpu // view_count)
            )
            if self._mixed:
                period = int(self.pointcloud_sample_period)
                required_by_source = (
                    (global_batch_size + period - 1) // period,
                    global_batch_size - global_batch_size // period,
                )
            else:
                required_by_source = (global_batch_size,)
            for source, required in enumerate(required_by_source):
                available = len(self._source_buckets[source][view_count])
                if available < required:
                    raise ValueError(
                        "native-view bucket is too small for a duplicate-free "
                        "global microbatch: "
                        f"source={source}, V={view_count}, available={available}, "
                        f"required={required}"
                    )

    def __len__(self) -> int:
        return self.num_microsteps - self.start_step

    def set_start_step(self, start_step: int) -> None:
        start_step = int(start_step)
        if start_step < 0 or start_step > self.num_microsteps:
            raise ValueError(
                f"start_step must be in [0, {self.num_microsteps}], got {start_step}"
            )
        self.start_step = start_step

    @staticmethod
    def _mix_u64(value: int) -> int:
        value &= (1 << 64) - 1
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
        value &= (1 << 64) - 1
        value = (value ^ (value >> 27)) * 0x94D049BB133111EB
        value &= (1 << 64) - 1
        return value ^ (value >> 31)

    def view_count_for_step(self, step: int) -> int:
        if len(self.available_view_counts) == 1:
            return self.available_view_counts[0]
        mixed = self._mix_u64(self.seed + int(step) * 0x9E3779B97F4A7C15)
        return self.available_view_counts[mixed % len(self.available_view_counts)]

    def local_batch_size_for_view(self, view_count: int) -> int:
        view_count = int(view_count)
        if view_count not in self.available_view_counts:
            raise ValueError(
                f"view_count={view_count} is not available; "
                f"available={self.available_view_counts}"
            )
        return self.max_views_per_gpu // view_count

    def __iter__(self):
        source_positions = {
            (source, view_count): 0
            for source in self._source_buckets
            for view_count in self.available_view_counts
        }
        order_cache: dict[tuple[int, int], list[int]] = {}
        global_slot_start = 0

        def source_index(source: int, view_count: int, position: int) -> int:
            bucket = self._source_buckets[source][view_count]
            offset = int(position) % len(bucket)
            if not self.shuffle:
                return int(bucket[offset])
            cache_key = (source, view_count)
            order = order_cache.get(cache_key)
            if order is None:
                generator = torch.Generator()
                order_seed = self._mix_u64(
                    self.seed
                    + source * 10_000_019
                    + view_count * 1_000_003
                )
                generator.manual_seed(order_seed & ((1 << 63) - 1))
                order = torch.randperm(len(bucket), generator=generator).tolist()
                order_cache[cache_key] = order
            return int(bucket[order[offset]])

        for step in range(self.num_microsteps):
            view_count = self.view_count_for_step(step)
            local_batch_size = self.local_batch_size_for_view(view_count)
            global_batch_size = self.num_replicas * local_batch_size

            if self._mixed:
                period = int(self.pointcloud_sample_period)
                pointcloud_count = (
                    (global_slot_start + global_batch_size) // period
                    - global_slot_start // period
                )
                pointcloud_slots = tuple(
                    sorted(
                        ((offset * global_batch_size) // pointcloud_count + step)
                        % global_batch_size
                        for offset in range(pointcloud_count)
                    )
                ) if pointcloud_count else ()
                pointcloud_slot_set = set(pointcloud_slots)
                source_counts = (pointcloud_count, global_batch_size - pointcloud_count)
            else:
                pointcloud_slots = tuple(range(global_batch_size))
                pointcloud_slot_set = set(pointcloud_slots)
                source_counts = (global_batch_size,)

            if step >= self.start_step:
                local_indices = []
                rank_slot_start = self.rank * local_batch_size
                for local_offset in range(local_batch_size):
                    global_offset = rank_slot_start + local_offset
                    pointcloud_before = bisect_left(pointcloud_slots, global_offset)
                    source = 0 if global_offset in pointcloud_slot_set else 1
                    source_offset = (
                        pointcloud_before
                        if source == 0
                        else global_offset - pointcloud_before
                    )
                    position = source_positions[(source, view_count)] + source_offset
                    row_index = source_index(source, view_count, position)
                    if self._mixed:
                        row_index = self.dataset.encode_source_index(source, row_index)
                    local_indices.append(row_index)
                yield local_indices

            for source, count in enumerate(source_counts):
                source_positions[(source, view_count)] += count
            global_slot_start += global_batch_size


def _dist_max_float(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _dist_mean_float(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def _distributed_geometry_active_rank_scale(
    local_active_rank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale active rank-local G scalars around distributed gradient averaging."""

    global_active_ranks = local_active_rank.detach().to(dtype=torch.float32).clone()
    world_size = 1
    if dist.is_initialized():
        dist.all_reduce(global_active_ranks, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    scale = torch.where(
        global_active_ranks > 0,
        global_active_ranks.new_tensor(float(world_size)) / global_active_ranks.clamp_min(1.0),
        torch.zeros_like(global_active_ranks),
    )
    return scale, global_active_ranks


def apply_ac_mot(
    model: ThreeDVAMOTTransformer3DModel,
    *,
    execution_route: str = "joint",
) -> None:
    """Apply non-nested activation checkpointing for the selected route."""

    if execution_route == "joint":
        for layer_id, block in enumerate(model.mot_blocks):
            model.mot_blocks[layer_id] = ptd_checkpoint_wrapper(
                block,
                preserve_rng_state=False,
            )
        return
    if execution_route != "geometry":
        raise ValueError(f"unsupported execution_route={execution_route!r}")
    for block in model.mot_blocks:
        if block.geometry is not None:
            block.geometry = ptd_checkpoint_wrapper(
                block.geometry,
                preserve_rng_state=False,
            )


def _validate_vggto_checkpoint_determinism(model: ThreeDVAMOTTransformer3DModel) -> None:
    """Guard the preserve_rng_state=False contract used by VGGTO checkpointing."""

    violations = []
    block_lists = {
        "frame_blocks": model.vggto.frame_blocks,
        "cross_view_blocks": model.vggto.cross_view_blocks,
        "inter_frame_blocks": model.vggto.inter_frame_blocks,
    }
    for list_name, blocks in block_lists.items():
        for layer_id, block in enumerate(blocks):
            if not any(True for _ in block.parameters()):
                continue
            sample_drop_ratio = float(getattr(block, "sample_drop_ratio", 0.0))
            if sample_drop_ratio != 0.0:
                violations.append(f"{list_name}.{layer_id}.sample_drop_ratio={sample_drop_ratio}")
            for module_name, module in block.named_modules():
                if isinstance(module, torch.nn.Dropout) and float(module.p) != 0.0:
                    suffix = f".{module_name}" if module_name else ""
                    violations.append(f"{list_name}.{layer_id}{suffix}.p={float(module.p)}")
    if violations:
        raise ValueError(
            "VGGTO activation checkpointing uses preserve_rng_state=False, but stochastic "
            "modules are enabled: " + ", ".join(violations)
        )


def apply_ac_vggto(model: ThreeDVAMOTTransformer3DModel) -> None:
    """Checkpoint every parameterized VGGTO Transformer block independently."""

    _validate_vggto_checkpoint_determinism(model)
    for layer_id, block in enumerate(model.vggto.frame_blocks):
        model.vggto.frame_blocks[layer_id] = ptd_checkpoint_wrapper(
            block,
            preserve_rng_state=False,
        )
    for blocks in (model.vggto.cross_view_blocks, model.vggto.inter_frame_blocks):
        for layer_id, block in enumerate(blocks):
            if any(True for _ in block.parameters()):
                blocks[layer_id] = ptd_checkpoint_wrapper(
                    block,
                    preserve_rng_state=False,
                )


def shard_mot_model(
    model,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    *,
    execution_route: str = "joint",
):
    """FSDP-shard the MOT-specific module layout.

    The original `shard_model` assumes `model.blocks`. Here we shard the MOT
    blocks and the VGGTO tower, then shard the root module. This
    preserves the official FSDP training style while respecting the VA-G model
    structure.
    """

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}
    head_mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    head_fsdp_config = {"mp_policy": head_mp_policy, "reshard_after_forward": True}

    if execution_route not in {"joint", "geometry"}:
        raise ValueError(f"unsupported execution_route={execution_route!r}")

    for block in model.mot_blocks:
        if execution_route == "joint":
            fully_shard(block, **fsdp_config)
            continue
        fully_shard(block.video_block, **fsdp_config)
        fully_shard(block.action_block, **fsdp_config)
        if block.geometry is not None:
            fully_shard(block.geometry, **fsdp_config)
    for block in model.vggto.frame_blocks:
        fully_shard(block, **fsdp_config)
    for block in model.vggto.cross_view_blocks:
        if any(True for _ in block.parameters()):
            fully_shard(block, **fsdp_config)
    for block in model.vggto.inter_frame_blocks:
        if any(True for _ in block.parameters()):
            fully_shard(block, **fsdp_config)
    fully_shard(model.vggto.patch_embed, **fsdp_config)
    fully_shard(model.vggto.dense_head, **head_fsdp_config)
    fully_shard(model.vggto.point_head, **head_fsdp_config)
    fully_shard(model, **fsdp_config)
    return model


def _move_to_device(value: Any, device: torch.device, non_blocking: bool = False):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device, non_blocking=non_blocking) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device, non_blocking=non_blocking) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device, non_blocking=non_blocking) for item in value)
    return value


def _resolve_mot_dataset_paths(dataset_root: str | Path, mot_config: dict[str, Any]) -> dict[str, str]:
    """Resolve MOT dataset paths from the selected root, ignoring stale absolute metadata.

    Some early prepared datasets were copied/renamed after generation, leaving
    `mot_config.json` with absolute paths to the old root.  Training should use
    the explicit dataset root selected by config/env and only fall back to the
    embedded paths when the local files are absent.
    """

    root = Path(dataset_root).resolve()
    manifest = root / "meta" / "mot_final_training_pointcloud_manifest.jsonl"
    empty_emb = root / "empty_emb.pt"
    return {
        "dataset_path": str(root),
        "mot_manifest_path": str(manifest if manifest.is_file() else Path(mot_config["mot_manifest_path"])),
        "empty_emb_path": str(empty_emb if empty_emb.is_file() else Path(mot_config["empty_emb_path"])),
    }


def _apply_cfg_text_dropout(text_emb: torch.Tensor, empty_emb: torch.Tensor, cfg_prob: float, *, training: bool) -> torch.Tensor:
    cfg_prob = float(cfg_prob)
    if not training or cfg_prob <= 0.0:
        return text_emb
    if text_emb.ndim < 2:
        raise ValueError(f"text_emb must include batch and sequence dims, got {tuple(text_emb.shape)}")
    empty = empty_emb.to(device=text_emb.device, dtype=text_emb.dtype)
    while empty.ndim < text_emb.ndim:
        empty = empty.unsqueeze(0)
    if empty.shape[0] == 1:
        empty = empty.expand(text_emb.shape[0], *empty.shape[1:])
    if empty.shape != text_emb.shape:
        raise ValueError(f"empty text embedding shape {tuple(empty.shape)} does not match text_emb {tuple(text_emb.shape)}")
    if cfg_prob >= 1.0:
        return empty.clone()
    mask = torch.rand((text_emb.shape[0],), device=text_emb.device) < cfg_prob
    if not bool(mask.any()):
        return text_emb
    out = text_emb.clone()
    out[mask] = empty[mask]
    return out


def _scheduler_sigmas_for_timesteps(train_scheduler, timesteps: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    scheduler_timesteps = train_scheduler.timesteps.to(device=device, dtype=timesteps.dtype)
    timestep_ids = torch.argmin((scheduler_timesteps[:, None, None] - timesteps[None]).abs(), dim=0)
    return train_scheduler.sigmas.to(device=device, dtype=dtype)[timestep_ids]


def _sample_mot_chunk_timesteps(
    *,
    batch_size: int,
    spec: MOTWindowSpec,
    train_scheduler,
    device: torch.device,
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
) -> torch.Tensor:
    timestep_ids = sample_timestep_id(
        batch_size=int(batch_size) * spec.target_chunks,
        min_timestep_bd=min_timestep_bd,
        max_timestep_bd=max_timestep_bd,
        num_train_timesteps=train_scheduler.num_train_timesteps,
    )
    target_timesteps = train_scheduler.timesteps[timestep_ids].to(device=device).reshape(int(batch_size), spec.target_chunks)
    target_timesteps = target_timesteps.repeat_interleave(spec.latent_frames_per_action_chunk_per_view, dim=1)
    history = torch.zeros((int(batch_size), spec.history_latent_frames), device=device, dtype=target_timesteps.dtype)
    return torch.cat([history, target_timesteps], dim=1)


def build_mot_train_dataset(config):
    spec = _mot_spec_from_config(config)
    data_profile = _mot_execution_route(config.optimization_composition)
    configured_action_sequence_length = int(getattr(config, "action_sequence_length", spec.action_sequence_length))
    if configured_action_sequence_length != spec.action_sequence_length:
        raise ValueError(
            "config.action_sequence_length must equal action_chunk_size+1: "
            f"got {configured_action_sequence_length}, expected {spec.action_sequence_length}"
        )

    def make_dataset(manifest_path):
        is_pointcloud_manifest = Path(manifest_path).resolve() == Path(config.mot_manifest_path).resolve()
        dataset_cls = MotGeometryLeRobotData if is_pointcloud_manifest else MotPureLeRobotData
        return dataset_cls(
            manifest_path=manifest_path,
            action_sequence_length=spec.action_sequence_length,
            action_dim=config.action_dim,
            norm_stats_by_task=getattr(config, "norm_stats_by_task", None),
            action_chunk_size=spec.action_chunk_size,
            video_downsample_ratio=spec.video_downsample_ratio,
            text_emb_cache_path=getattr(config, "text_emb_cache_path", getattr(config, "empty_emb_path", None)),
            empty_emb_path=getattr(config, "empty_emb_path", None),
            action_cache_manifest_path=getattr(config, "action_cache_manifest_path", None),
            video_decoder_cache_size=int(
                getattr(config, "video_decoder_cache_size", MOT_DEFAULT_VIDEO_DECODER_CACHE_SIZE)
            ),
            point_store_cache_size=int(
                getattr(config, "point_store_cache_size", MOT_DEFAULT_POINT_STORE_CACHE_SIZE)
            ),
            action_cache_size=int(
                getattr(config, "action_cache_size", MOT_DEFAULT_ACTION_CACHE_SIZE)
            ),
            random_start=True,
            data_profile=data_profile,
        )

    pointcloud_dataset = make_dataset(config.mot_manifest_path)
    if data_profile == "geometry":
        return pointcloud_dataset
    non_pointcloud_manifest_path = getattr(config, "non_pointcloud_manifest_path", None)
    if non_pointcloud_manifest_path:
        pure_manifest = Path(non_pointcloud_manifest_path)
        if not pure_manifest.is_file():
            raise FileNotFoundError(pure_manifest)
        with pure_manifest.open("r", encoding="utf-8") as f:
            has_pure_rows = any(line.strip() for line in f)
        if not has_pure_rows:
            return pointcloud_dataset
        return MotBalancedMixDataset(
            pointcloud_dataset=pointcloud_dataset,
            pure_dataset=make_dataset(non_pointcloud_manifest_path),
        )
    return pointcloud_dataset


def _build_mot_train_sampler(train_dataset, config):
    grad_accum = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
    num_microsteps = max(1, int(config.num_steps)) * grad_accum
    pointcloud_sample_period = (
        getattr(config, "pointcloud_sample_period", None)
        if isinstance(train_dataset, MotBalancedMixDataset)
        else None
    )
    return _MOTViewBatchSampler(
        train_dataset,
        num_replicas=int(getattr(config, "world_size", 1)),
        rank=int(getattr(config, "rank", 0)),
        num_microsteps=num_microsteps,
        max_views_per_gpu=int(config.max_views_per_gpu),
        pointcloud_sample_period=pointcloud_sample_period,
        seed=int(getattr(config, "sampler_seed", 42)),
        shuffle=True,
    )


def _mot_dataloader_kwargs(config, train_sampler):
    kwargs = {
        "batch_sampler": train_sampler,
        "num_workers": config.load_worker,
        "pin_memory": bool(getattr(config, "dataloader_pin_memory", getattr(config, "pin_memory", True))),
    }
    if int(config.load_worker) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(getattr(config, "dataloader_prefetch_factor", 2))
    return kwargs


def build_mot_param_groups(model, base_lr: float, vggto_lr_multiplier: float, weight_decay: float):
    """Create optimizer groups for the reviewed all-trainable strategy.

    Discussed training decision: all parameters remain trainable, but native VGGTO
    pretrained modules use a smaller LR to reduce early damage to the geometry
    prior.  The MoT G joint streams live under `mot_blocks.*.geometry`; although
    initialized from VGGTO inter-frame layers, they are the new VA-G fusion parameters and
    stay on the main LR.
    """

    main_params = []
    vggto_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("vggto."):
            vggto_params.append(param)
        else:
            main_params.append(param)

    groups = []
    if main_params:
        groups.append(
            {
                "params": main_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "lingbot_and_mot",
            }
        )
    if vggto_params:
        groups.append(
            {
                "params": vggto_params,
                "lr": base_lr * vggto_lr_multiplier,
                "weight_decay": weight_decay,
                "name": "vggto_pretrained",
            }
        )
    return groups


def _mot_parameter_branch(name: str) -> str:
    if name.startswith("vggto.") or ".geometry." in name:
        return "g"
    if name.startswith("action_") or ".action_block." in name:
        return "a"
    return "v"


def apply_mot_parameter_ownership(
    model,
    optimization_composition: str,
) -> dict[str, dict[str, int | bool]]:
    """Set trainability from the exclusive V/A/G parameter-owner mapping."""

    selected = optimization_branches(optimization_composition)
    summary = {
        branch: {
            "selected": branch in selected,
            "parameter_tensors": 0,
            "parameters": 0,
        }
        for branch in "vag"
    }
    for name, parameter in model.named_parameters():
        branch = _mot_parameter_branch(name)
        parameter.requires_grad_(branch in selected)
        summary[branch]["parameter_tensors"] += 1
        summary[branch]["parameters"] += parameter.numel()

    missing = [
        branch
        for branch, values in summary.items()
        if values["parameter_tensors"] == 0
    ]
    if missing:
        raise RuntimeError(f"MOT parameter branch classification found no parameters for: {missing}")
    return summary


def _validate_positive_finite_weight(config, name: str) -> float:
    value = float(getattr(config, name))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"active {name} must be finite and positive, got {value}")
    return value


def validate_mot_training_config(config) -> str:
    composition = resolve_optimization_composition(
        getattr(config, "optimization_composition", None)
    )
    config.optimization_composition = composition
    branches = optimization_branches(composition)

    raw_max_views_per_gpu = getattr(config, "max_views_per_gpu", 0)
    if isinstance(raw_max_views_per_gpu, bool) or not isinstance(
        raw_max_views_per_gpu,
        (int, np.integer),
    ):
        raise ValueError(
            "max_views_per_gpu must be an integer at least 3, "
            f"got {raw_max_views_per_gpu!r}"
        )
    max_views_per_gpu = int(raw_max_views_per_gpu)
    if max_views_per_gpu < 3:
        raise ValueError(
            "max_views_per_gpu must be an integer at least 3, "
            f"got {max_views_per_gpu}"
        )
    config.max_views_per_gpu = max_views_per_gpu

    if getattr(config, "resume_from", None) and getattr(config, "initialize_from", None):
        raise ValueError("resume_from and initialize_from are mutually exclusive")

    if "v" in branches:
        _validate_positive_finite_weight(config, "video_loss_weight")
    if "a" in branches:
        _validate_positive_finite_weight(config, "action_loss_weight")
    if "g" in branches:
        _validate_positive_finite_weight(config, "geometry_loss_weight")
        component_weights = [
            float(getattr(config, "depth_loss_weight")),
            float(getattr(config, "point_loss_weight")),
        ]
        if any(not np.isfinite(value) or value < 0 for value in component_weights):
            raise ValueError(
                "active geometry component weights must be finite and non-negative, "
                f"got depth={component_weights[0]}, point={component_weights[1]}"
            )
        if not any(value > 0 for value in component_weights):
            raise ValueError(
                "active G objective requires a positive depth_loss_weight or "
                "point_loss_weight"
            )
    return composition


def _configure_adamw_foreach(optimizer: torch.optim.AdamW) -> None:
    """Apply the tested non-fused AdamW mode to fresh or resumed state."""

    execution_options = {
        "fused": False,
        "foreach": True,
        "capturable": False,
    }
    optimizer.defaults.update(execution_options)
    for group in optimizer.param_groups:
        group.update(execution_options)
    for state in optimizer.state.values():
        step = state.get("step")
        if torch.is_tensor(step) and step.device.type != "cpu":
            state["step"] = step.detach().cpu()


def _frame_weighted_mse(pred, target, timestep_weight, mask=None, video_latent_loss_mask=None):
    """LingBot-compatible frame-wise MSE over tensors shaped `[B,C,F,...]`."""

    batch_size, _, frames = pred.shape[:3]
    loss = F.mse_loss(pred.float(), target.float().detach(), reduction="none")
    weight_shape = [batch_size, 1, frames] + [1] * (pred.ndim - 3)
    loss = loss * timestep_weight.reshape(weight_shape)

    mask_bool = None
    if mask is not None:
        mask_bool = mask.bool().to(device=loss.device)
        loss = torch.where(mask_bool.expand_as(loss), loss, torch.zeros_like(loss))

    # Move frame to axis 1, flatten B*F, and normalize each frame independently.
    order = [0, 2] + list(range(3, pred.ndim)) + [1]
    loss = loss.permute(order).flatten(0, 1).flatten(1)
    if mask_bool is None:
        denom = torch.ones_like(loss).sum(dim=1)
    else:
        mask_flat = mask_bool.expand_as(pred).permute(order).flatten(0, 1).flatten(1)
        denom = mask_flat.sum(dim=1)
    per_frame = loss.sum(dim=1) / denom.clamp_min(1e-6)

    if video_latent_loss_mask is None:
        return per_frame.mean()

    video_latent_loss_mask = video_latent_loss_mask.to(
        device=per_frame.device,
        dtype=torch.bool,
    ).reshape(batch_size, frames)
    per_frame = per_frame.reshape(batch_size, frames)
    per_frame = torch.where(video_latent_loss_mask, per_frame, torch.zeros_like(per_frame))
    supervised_frames = video_latent_loss_mask.sum(dim=1)
    per_sample = per_frame.sum(dim=1) / supervised_frames.to(dtype=per_frame.dtype).clamp_min(1)
    return per_sample.mean()


def _action_weighted_mse(pred, target, timestep_weight, action_loss_mask):
    """LingBot-VA per-frame action MSE with an outer valid-frame mean."""

    loss = F.mse_loss(pred.float(), target.float().detach(), reduction="none")
    weight = timestep_weight.reshape(timestep_weight.shape[0], 1, timestep_weight.shape[1], 1, 1)
    mask_bool = action_loss_mask.bool().to(device=loss.device)
    loss = torch.where(mask_bool.expand_as(loss), loss * weight, torch.zeros_like(loss))
    batch_size, _, frames = loss.shape[:3]
    loss_by_frame = loss.permute(0, 2, 1, 3, 4).reshape(batch_size, frames, -1)
    mask_by_frame = mask_bool.expand_as(loss).permute(0, 2, 1, 3, 4).reshape(
        batch_size,
        frames,
        -1,
    )
    valid_elements = mask_by_frame.sum(dim=-1)
    per_frame = loss_by_frame.sum(dim=-1) / valid_elements.to(loss.dtype).clamp_min(1)
    valid_frames = valid_elements > 0
    return torch.where(valid_frames, per_frame, torch.zeros_like(per_frame)).sum() / valid_frames.sum().to(
        loss.dtype
    ).clamp_min(1)


def _scalar_for_log(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = float(value.detach().cpu().item())
    else:
        value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return value


def _detach_log_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    if isinstance(value, DTensor):
        return value.to_local()
    return value


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def _cross_view_named_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    named_parameters = []
    seen = set()
    for layer_id, block in enumerate(model.vggto.cross_view_blocks):
        for name, parameter in block.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            named_parameters.append((f"cross_view_blocks.{layer_id}.{name}", parameter))
    return named_parameters


def _distributed_parameter_norm_stats(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    square_sum = torch.zeros((), dtype=torch.float64, device=device)
    params_with_grad = 0
    all_finite = True
    for _, parameter in named_parameters:
        if parameter.grad is None:
            continue
        grad = _local_tensor(parameter.grad.detach())
        params_with_grad += 1
        all_finite = all_finite and bool(torch.isfinite(grad).all())
        square_sum += grad.float().square().sum().to(dtype=torch.float64)

    count_and_finite = torch.tensor(
        [params_with_grad, int(all_finite)],
        dtype=torch.int64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_and_finite, op=dist.ReduceOp.MIN)
    return {
        "grad_norm": float(square_sum.sqrt().item()),
        "params_with_grad": int(count_and_finite[0].item()),
        "parameter_tensors": len(named_parameters),
        "all_finite": bool(count_and_finite[1].item()),
    }


def _snapshot_local_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> list[tuple[str, torch.nn.Parameter, torch.Tensor]]:
    return [
        (name, parameter, _local_tensor(parameter.detach()).clone())
        for name, parameter in named_parameters
    ]


def _distributed_parameter_update_stats(
    snapshots: list[tuple[str, torch.nn.Parameter, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    update_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    parameter_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    all_finite = True
    for _, parameter, before in snapshots:
        after = _local_tensor(parameter.detach())
        delta = after.float() - before.float()
        all_finite = all_finite and bool(torch.isfinite(delta).all())
        update_square_sum += delta.square().sum().to(dtype=torch.float64)
        parameter_square_sum += before.float().square().sum().to(dtype=torch.float64)

    finite_tensor = torch.tensor(int(all_finite), dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(update_square_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(parameter_square_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
    update_norm = float(update_square_sum.sqrt().item())
    parameter_norm = float(parameter_square_sum.sqrt().item())
    return {
        "update_norm": update_norm,
        "relative_update_norm": update_norm / max(parameter_norm, torch.finfo(torch.float64).tiny),
        "update_all_finite": bool(finite_tensor.item()),
    }


def _first_batch_meta_value(value):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        item = value.flatten()[0].detach().cpu().item()
        return int(item) if isinstance(item, (int, np.integer)) or float(item).is_integer() else item
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        item = value.reshape(-1)[0].item()
        return int(item) if isinstance(item, (int, np.integer)) else item
    if isinstance(value, (list, tuple)):
        return None if not value else _first_batch_meta_value(value[0])
    return value


def _batch_meta_for_nan_log(batch):
    meta = batch.get("meta") if isinstance(batch, dict) else None
    if not isinstance(meta, dict):
        return {}
    keys = (
        "sample_index",
        "task_uid",
        "source_lerobot_task_dir",
        "data_file",
        "episode_index",
        "start_frame",
        "source_dataset",
        "has_pointcloud",
        "pointcloud_session_dir",
    )
    return {key: _first_batch_meta_value(meta[key]) for key in keys if key in meta}


def _make_nan_log_record(
    *,
    step: int,
    trigger_stage: str,
    losses: dict[str, Any],
    total_norm,
    loss_finite: bool,
    grad_finite: bool | None,
    global_has_nan: bool,
    rank_has_nan: bool,
    rank: int,
    nan_locations: list[dict[str, Any]],
    batch: dict[str, Any] | None = None,
    rank_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    record = {
        "step": int(step),
        "trigger_stage": str(trigger_stage),
        "global_has_nan": bool(global_has_nan),
        "rank_has_nan": bool(rank_has_nan),
        "rank": int(rank),
        "loss_finite": bool(loss_finite),
        "grad_finite": None if grad_finite is None else bool(grad_finite),
        "total_norm": _scalar_for_log(total_norm),
        "nan_locations": nan_locations,
        "batch_meta": _batch_meta_for_nan_log(batch or {}),
    }
    for key in (
        "loss",
        "latent_loss",
        "action_loss",
        "video_loss_weight",
        "action_loss_weight",
        "depth_loss",
        "depth_loss_raw",
        "geometry_loss_weight",
        "geometry_active_rank_scale",
        "geometry_global_active_ranks",
        "depth_loss_weight",
        "point_loss",
        "point_loss_raw",
        "point_loss_weight",
    ):
        if key in losses:
            record[key] = _scalar_for_log(losses[key])
    if rank_records is not None:
        record["rank_records"] = rank_records
    return record


class MOTTrainer:
    def __init__(self, config):
        self.config = config
        self.optimization_composition = validate_mot_training_config(config)
        self.optimization_branches = optimization_branches(
            self.optimization_composition
        )
        self.execution_route = _mot_execution_route(self.optimization_composition)
        gc_interval = int(getattr(config, "gc_interval", 0) or 0)
        if gc_interval < 0:
            raise ValueError(f"gc_interval must be non-negative, got {gc_interval}")
        Path(config.save_root).mkdir(parents=True, exist_ok=True)
        if config.enable_wandb and config.rank == 0:
            import wandb

            wandb_kwargs, wandb_dir, wandb_mode = _wandb_init_settings(config)
            if wandb_mode != "offline" and os.environ.get("WANDB_BASE_URL") and os.environ.get("WANDB_API_KEY"):
                wandb.login(host=os.environ["WANDB_BASE_URL"], key=os.environ["WANDB_API_KEY"])
            self.wandb = wandb
            self.wandb.init(**wandb_kwargs)
            logger.info(f"WandB logging enabled: mode={wandb_mode}, dir={wandb_dir}")

        self.step = 0
        self.optimizer_step = 0
        self.skipped_step_total = 0
        self.nonfinite_loss_total = 0
        self.nonfinite_grad_total = 0
        self.dataset_skip_total = 0
        self._last_checkpoint_step = None
        self.gc_interval = gc_interval
        self.max_checkpoints = int(getattr(config, "max_checkpoints", 3))
        if self.max_checkpoints < 1:
            raise ValueError(f"max_checkpoints must be positive, got {self.max_checkpoints}")
        if bool(getattr(config, "save_full_state", False)) and int(
            getattr(config, "gradient_accumulation_steps", 1)
        ) != 1:
            raise ValueError(
                "DCP full-state checkpoints currently require gradient_accumulation_steps=1 "
                "because partial accumulated gradients are not checkpointed"
            )
        if bool(getattr(config, "eval_with_cpu", False)) and not bool(
            getattr(config, "save_full_state", False)
        ):
            raise ValueError("eval_with_cpu requires DCP checkpoints with save_full_state=True")
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.checkpoint_process_group = None
        needs_checkpoint_collectives = bool(getattr(config, "save_full_state", False)) or bool(
            getattr(config, "resume_from", None)
        )
        if needs_checkpoint_collectives and dist.is_initialized():
            if not dist.is_gloo_available():
                raise RuntimeError("DCP full-state checkpoints require the Gloo backend")
            # Keep checkpoint collectives off the CUDA/NCCL group used by training.
            self.checkpoint_process_group = dist.new_group(backend="gloo")
        self.empty_text_emb = None
        self.performance_jsonl_enabled = bool(getattr(config, "performance_jsonl_enabled", False))
        self.performance_jsonl_interval = max(1, int(getattr(config, "performance_jsonl_interval", 1)))
        self.performance_jsonl_max_steps = max(1, int(getattr(config, "performance_jsonl_max_steps", 500)))
        self.memory_jsonl_enabled = bool(getattr(config, "memory_jsonl_enabled", False))
        self.memory_jsonl_interval = int(getattr(config, "memory_jsonl_interval", 100))
        self.memory_smaps_interval = int(getattr(config, "memory_smaps_interval", 500))
        if self.memory_jsonl_interval <= 0:
            raise ValueError(f"memory_jsonl_interval must be positive, got {self.memory_jsonl_interval}")
        if self.memory_smaps_interval <= 0:
            raise ValueError(f"memory_smaps_interval must be positive, got {self.memory_smaps_interval}")
        self._worker_cache_stats: dict[int, dict[str, dict[str, int]]] = {}
        if self.performance_jsonl_enabled and config.rank == 0:
            logger.info(
                "Per-step performance diagnostics enabled: "
                f"path={Path(config.save_root) / 'performance.jsonl'}, "
                f"interval={self.performance_jsonl_interval}, "
                f"max_steps={self.performance_jsonl_max_steps}"
            )
        if self.memory_jsonl_enabled and config.rank == 0:
            logger.info(
                "Memory diagnostics enabled: "
                f"path={Path(config.save_root) / 'memory.jsonl'}, "
                f"interval={self.memory_jsonl_interval}, "
                f"smaps_interval={self.memory_smaps_interval}"
            )
        if getattr(config, "masked_attn_backend", "fa4") == "fa4":
            # FA4 plan point: fail before data/model setup if the requested custom-mask
            # backend is not importable or this rank is not running on H100/SM90.
            validate_fa4_training_environment(self.device)

        # Setup dataloaders before CUDA/FSDP init to avoid fork deadlocks in dataset workers.
        logger.info("Setting up MOT dataset...")
        train_dataset = build_mot_train_dataset(config)
        train_sampler = _build_mot_train_sampler(train_dataset, config)
        if config.rank == 0:
            spec = _mot_spec_from_config(config)
            logger.info(
                "MOT fixed-window training: "
                f"optimization_composition={self.optimization_composition} "
                f"execution_route={self.execution_route} "
                f"max_views_per_gpu={config.max_views_per_gpu} "
                f"available_views={train_sampler.available_view_counts} "
                f"local_batch_sizes={{{', '.join(f'{v}: {train_sampler.local_batch_size_for_view(v)}' for v in train_sampler.available_view_counts)}}} "
                f"history_chunks={spec.history_chunks} "
                f"target_chunks={spec.target_chunks} latent_frames_per_action_chunk_per_view={spec.latent_frames_per_action_chunk_per_view} "
                f"video_downsample_ratio={spec.video_downsample_ratio} action_sequence_length={spec.action_sequence_length} "
                f"target_actions_per_sample={spec.target_actions_per_sample}"
            )
        if config.rank == 0:
            if isinstance(train_dataset, MotBalancedMixDataset):
                pointcloud_period = train_sampler.pointcloud_sample_period
                logger.info(
                    "MOT source sampling: "
                    f"pointcloud:non_pointcloud=1:{pointcloud_period - 1} "
                    f"pointcloud_sample_period={pointcloud_period} "
                    f"pointcloud_episodes={len(train_dataset.pointcloud_dataset)} "
                    f"non_pointcloud_episodes={len(train_dataset.pure_dataset)}"
                )
            logger.info(
                "MOT dynamic batch sampler: "
                f"microsteps_per_rank={len(train_sampler)} "
                f"dataset_len={len(train_dataset)}"
            )
        self.train_loader = DataLoader(train_dataset, **_mot_dataloader_kwargs(config, train_sampler))

        logger.info("Loading 3DVA_MOT transformer...")
        # Dataset/W&B setup must not perturb the controlled model initialization.
        _seed_mot_training(int(getattr(config, "train_seed", 42)))
        self.transformer = self._load_transformer()
        self.transformer.requires_grad_(True)
        branch_summary = apply_mot_parameter_ownership(
            self.transformer,
            self.optimization_composition,
        )
        if not any(parameter.requires_grad for parameter in self.transformer.parameters()):
            raise ValueError("MOT training requires at least one unfrozen V/A/G branch")
        if config.rank == 0:
            logger.info(f"MOT parameter ownership: {branch_summary}")

        logger.info("Setting up MOT activation checkpointing ...")
        apply_ac_mot(self.transformer, execution_route=self.execution_route)
        apply_ac_vggto(self.transformer)

        logger.info("Setting up MOT FSDP...")
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=partial(
                shard_mot_model,
                execution_route=self.execution_route,
            ),
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        if not dist.is_initialized():
            self.transformer.vggto.keep_heads_fp32_()
        self.transformer.train()

        optimizer_groups = build_mot_param_groups(
            self.transformer,
            base_lr=config.learning_rate,
            vggto_lr_multiplier=getattr(config, "vggto_lr_multiplier", 0.1),
            weight_decay=config.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
        )
        _configure_adamw_foreach(self.optimizer)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps),
        )
        if getattr(self.config, "resume_from", None):
            self._load_training_state(self.config.resume_from)
        self.performance_jsonl_start_step = self.step

        self.train_scheduler_latent = None
        self.train_scheduler_action = None
        if self.execution_route == "joint":
            self.train_scheduler_latent = FlowMatchScheduler(
                shift=self.config.snr_shift,
                sigma_min=0.0,
                extra_one_step=True,
            )
            self.train_scheduler_latent.set_timesteps(1000, training=True)
            self.train_scheduler_action = FlowMatchScheduler(
                shift=self.config.action_snr_shift,
                sigma_min=0.0,
                extra_one_step=True,
            )
            self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        eval_cfg = getattr(config, "eval_cfg", None)
        if bool(getattr(config, "eval_with_cpu", False)) and eval_cfg is None:
            raise ValueError("eval_with_cpu requires config.eval_cfg")
        eval_output_root = (
            getattr(eval_cfg, "output_root", None)
            if eval_cfg is not None
            else None
        )
        self.checkpoint_evaluation_queue = CheckpointEvaluationQueue(
            enabled=(config.rank == 0 and bool(getattr(config, "eval_with_cpu", False))),
            eval_cfg=(eval_cfg or {}),
            output_root=(eval_output_root or Path(config.save_root) / "evaluations"),
            queue_dir=Path(config.save_root) / "evaluation_queue",
        )
        if config.rank == 0 and bool(getattr(config, "eval_with_cpu", False)):
            logger.info(
                "External CPU checkpoint evaluation queue enabled: "
                f"queue_dir={Path(config.save_root) / 'evaluation_queue'}, "
                f"output_root={eval_output_root or Path(config.save_root) / 'evaluations'}"
            )
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.train_loader_iter = None
        self._training_gc_started = False
        self._training_gc_was_enabled = None
        self.train_vae = None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _load_transformer(self):
        resume_from = getattr(self.config, "resume_from", None)
        initialize_from = getattr(self.config, "initialize_from", None)
        if resume_from:
            checkpoint_path = Path(resume_from)
            self._validate_dcp_checkpoint_layout(checkpoint_path)
            transformer_path = checkpoint_path / "transformer"
            if self.config.rank == 0:
                logger.info(f"Preparing MOT transformer for full-state resume: {transformer_path}")
            model_config = ThreeDVAMOTTransformer3DModel.load_config(transformer_path)
            saved_backend = model_config.get("masked_attn_backend", "fa4")
            current_backend = getattr(self.config, "masked_attn_backend", "fa4")
            if saved_backend != current_backend:
                raise ValueError(
                    "Full-state resume requires the same masked attention backend: "
                    f"checkpoint={saved_backend}, current={current_backend}"
                )
            with torch.device("meta"):
                model = ThreeDVAMOTTransformer3DModel.from_config(model_config)
            model.to_empty(device=self.device)
            return model

        if initialize_from:
            checkpoint_path = Path(initialize_from)
            self._validate_transformer_checkpoint_layout(checkpoint_path)
            transformer_path = checkpoint_path / "transformer"
            if self.config.rank == 0:
                logger.info(f"Initializing MOT stage from complete transformer: {transformer_path}")
            return ThreeDVAMOTTransformer3DModel.from_pretrained(
                transformer_path,
                torch_dtype=torch.float32,
            )

        config_overrides = {
            "max_num_views": max(3, len(self.config.obs_cam_keys)),
            "action_dim": int(self.config.action_dim),
            "action_hidden_dim": 768,
            "action_ffn_dim": 3072,
            "attn_mode": "torch",
            "num_layers": 30,
            "vggto_depth": 30,
            "vggto_register_attention_indices": tuple(range(0, 30, 2)),
            "vggto_cached_layer_indices": (5, 15, 21, 29),
            "vggto_pretrained_register_attention_indices": (2, 8, 12, 18, 26),
            "vggto_inserted_layer_indices": (4, 6, 10, 14, 22, 24),
            "vggto_converted_layer_indices": (0, 16, 20, 28),
            # Explicit masked-attention backend for VA-G joint attention. If set
            # to "fa4", the model fails fast unless FlashAttention-4 is usable.
            "masked_attn_backend": getattr(self.config, "masked_attn_backend", "fa4"),
            "init_noise_seed": int(getattr(self.config, "init_noise_seed", 42)),
            "vggto_cross_view_init_scale": float(
                getattr(self.config, "vggto_cross_view_init_scale", 1.0)
            ),
        }
        init_model_from_lingbot = bool(
            getattr(self.config, "init_model_from_lingbot", True)
        )
        video_transformer_path = (
            self.config.lingbot_transformer_path
            if init_model_from_lingbot
            else self.config.wan22_transformer_path
        )
        source_paths = {
            "video_transformer_path": video_transformer_path,
            "vggto_checkpoint_path": getattr(self.config, "vggto_checkpoint_path", None),
            "vggt_checkpoint_path": getattr(self.config, "vggt_checkpoint_path", None),
        }
        missing_sources = [name for name, path in source_paths.items() if not path]
        if missing_sources:
            raise ValueError(
                "Constructing a new MOT transformer requires source checkpoints: "
                + ", ".join(missing_sources)
            )
        model, report = ThreeDVAMOTTransformer3DModel.from_lingbot_and_vggto(
            video_transformer_path,
            source_paths["vggto_checkpoint_path"],
            source_paths["vggt_checkpoint_path"],
            init_model_from_lingbot=init_model_from_lingbot,
            config_overrides=config_overrides,
        )
        if self.config.rank == 0:
            logger.info(
                f"Initialized MOT transformer from {report.video_source}/VGGTO: "
                f"video_keys={report.loaded_video_keys}, "
                f"vggto_keys={report.loaded_vggto_required_keys}, "
                f"vggt_point_keys={report.loaded_vggt_point_keys}, "
                f"ignored_vggto_keys={len(report.ignored_vggto_keys)}, "
                f"action_copy={len(report.action_init.copied)}, "
                f"action_interpolate={len(report.action_init.interpolated)}, "
                f"action_scale={len(report.action_init.scaled)}, "
                f"action_shared={len(report.action_init.shared)}, "
                f"action_random={report.action_init.random}, "
                "cross_view_layer_scale_factor="
                f"{report.vggto_24_to_30.cross_view_layer_scale_factor}, "
                f"notes={report.notes}"
            )
            logger.info(
                "Action initialization coverage: "
                f"copied={report.action_init.copied}, "
                f"interpolated={report.action_init.interpolated}, "
                f"scaled={report.action_init.scaled}, "
                f"shared={report.action_init.shared}, "
                f"random={report.action_init.random}, "
                f"random_stats={report.action_init.random_stats}, "
                f"seed={report.action_init.seed}"
            )
        return model

    def _set_sampler_resume_offset(self) -> None:
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if not hasattr(sampler, "set_start_step"):
            return
        completed_step = int(self.step)
        sampler.set_start_step(completed_step)
        if self.config.rank == 0:
            logger.info(
                "MOT sampler resumed from completed microstep "
                f"{completed_step}"
            )

    def _get_next_batch(self):
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        self._start_training_gc_control()

        try:
            batch = next(self.train_loader_iter)
        except StopIteration as exc:
            raise RuntimeError(
                "MOT view-aware batch sampler exhausted before training completed"
            ) from exc
        return batch

    def _start_training_gc_control(self) -> None:
        if getattr(self, "_training_gc_started", False):
            return
        self._training_gc_started = True
        self._training_gc_was_enabled = gc.isenabled()
        gc.disable()
        logger.info(
            "Training-rank GC control started: "
            f"rank={int(getattr(self.config, 'rank', 0))}, "
            f"automatic_gc_enabled={gc.isenabled()}, "
            f"generation_2_interval={self.gc_interval} (0 means disabled)"
        )

    def _restore_training_gc_control(self) -> None:
        if not getattr(self, "_training_gc_started", False):
            return
        if bool(self._training_gc_was_enabled):
            gc.enable()
        self._training_gc_started = False
        self._training_gc_was_enabled = None

    def _synchronize_and_collect_garbage(self, *, step_completed: bool) -> tuple[float, float, float]:
        post_step_barrier_time = 0.0
        if dist.is_initialized():
            barrier_start = time.perf_counter()
            dist.barrier()
            post_step_barrier_time = time.perf_counter() - barrier_start

        gc_interval = int(getattr(self, "gc_interval", getattr(self.config, "gc_interval", 0)) or 0)
        gc_due = (
            step_completed
            and gc_interval > 0
            and self.step > 0
            and self.step % gc_interval == 0
        )
        if not gc_due:
            return post_step_barrier_time, 0.0, 0.0

        self._write_memory_snapshot(phase="before_gc", include_smaps=True)
        gc_start = time.perf_counter()
        collected = gc.collect(2)
        gc_collect_time = time.perf_counter() - gc_start
        self._write_memory_snapshot(
            phase="after_gc",
            include_smaps=True,
            event={
                "generation": 2,
                "collected": int(collected),
                "duration_seconds": gc_collect_time,
            },
        )

        post_gc_barrier_time = 0.0
        if dist.is_initialized():
            barrier_start = time.perf_counter()
            dist.barrier()
            post_gc_barrier_time = time.perf_counter() - barrier_start
        if int(getattr(self.config, "rank", 0)) == 0:
            logger.info(
                f"Scheduled generation-2 GC at step {self.step}: "
                f"collected={collected}, duration={gc_collect_time:.3f}s"
            )
        return post_step_barrier_time, gc_collect_time, post_gc_barrier_time

    def _should_report_current_step(self) -> bool:
        log_interval = max(1, int(getattr(self.config, "log_interval", 1)))
        next_step = self.step + 1
        return self.step == 0 or next_step % log_interval == 0 or next_step >= int(self.config.num_steps)

    def _should_write_performance_jsonl(self) -> bool:
        if not bool(getattr(self, "performance_jsonl_enabled", False)):
            return False
        next_step = self.step + 1
        profile_step = next_step - int(getattr(self, "performance_jsonl_start_step", 0))
        if profile_step <= 0 or profile_step > int(getattr(self, "performance_jsonl_max_steps", 500)):
            return False
        interval = max(1, int(getattr(self, "performance_jsonl_interval", 1)))
        return profile_step == 1 or profile_step % interval == 0

    def _start_phase_timer(self):
        if self.device.type != "cuda":
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _stop_phase_timer(self, start):
        if isinstance(start, float):
            return time.perf_counter() - start
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        return start, end

    @staticmethod
    def _phase_elapsed_seconds(value) -> float:
        if value is None:
            return 0.0
        if isinstance(value, tuple):
            start, end = value
            return float(start.elapsed_time(end)) / 1000.0
        return float(value)

    def _write_performance_jsonl(
        self,
        timing_records: list[dict[str, Any]],
        *,
        training_step: int,
        skipped_step: bool,
    ) -> None:
        local_times = {key: 0.0 for key in MOT_PERFORMANCE_TIMING_KEYS}
        local_pointcloud_samples = 0
        local_dataset_skip_count = 0
        local_samples = 0
        native_view_microsteps = {view_count: 0 for view_count in MOT_SUPPORTED_VIEW_COUNTS}
        for timing_record in timing_records:
            local_pointcloud_samples += int(timing_record.get("pointcloud_samples", 0))
            local_dataset_skip_count += int(timing_record.get("dataset_skip_count", 0))
            local_samples += int(timing_record.get("local_samples", 1))
            native_views = int(timing_record.get("native_views", 0))
            if native_views in native_view_microsteps:
                native_view_microsteps[native_views] += 1
            for key in (
                "data_fetch",
                "data_barrier",
                "metrics_logging",
                "checkpoint",
                "post_step_barrier",
                "gc_collect",
                "post_gc_barrier",
                "iteration",
            ):
                local_times[key] += float(timing_record.get(key, 0.0))
            phases = timing_record.get("phases", {})
            for key in (
                "input_transfer",
                "vae_encode",
                "input_prepare",
                "model_forward",
                "loss",
                "loss_finite_sync",
                "backward",
                "grad_clip",
                "grad_finite_sync",
                "optimizer_step",
            ):
                local_times[key] += self._phase_elapsed_seconds(phases.get(key))

        local_times["data_total"] = local_times["data_fetch"] + local_times["data_barrier"]
        local_times["forward_total"] = sum(
            local_times[key]
            for key in (
                "input_transfer",
                "vae_encode",
                "input_prepare",
                "model_forward",
                "loss",
                "loss_finite_sync",
            )
        )
        local_times["optimizer_total"] = sum(
            local_times[key]
            for key in ("grad_clip", "grad_finite_sync", "optimizer_step")
        )
        accounted = sum(
            local_times[key]
            for key in MOT_PERFORMANCE_TIMING_KEYS
            if key != "iteration"
        )
        local_times["unaccounted"] = max(0.0, local_times["iteration"] - accounted)

        timing_keys = (
            *MOT_PERFORMANCE_TIMING_KEYS,
            "data_total",
            "forward_total",
            "optimizer_total",
            "unaccounted",
        )
        local_values = torch.tensor(
            [
                *(local_times[key] for key in timing_keys),
                local_pointcloud_samples,
                local_dataset_skip_count,
                local_samples,
            ],
            device=self.device,
            dtype=torch.float64,
        )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if dist.is_initialized():
            gathered = torch.empty(world_size * local_values.numel(), device=self.device, dtype=torch.float64)
            dist.all_gather_into_tensor(gathered, local_values)
            rank_values = gathered.reshape(world_size, local_values.numel())
        else:
            rank_values = local_values.unsqueeze(0)
        rank_values = rank_values.cpu()
        pointcloud_samples_by_rank = rank_values[:, len(timing_keys)].to(dtype=torch.int64)
        dataset_skip_count_by_rank = rank_values[:, len(timing_keys) + 1].to(dtype=torch.int64)
        samples_by_rank = rank_values[:, len(timing_keys) + 2].to(dtype=torch.int64)

        local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", 1)))
        timings = {}
        for column, key in enumerate(timing_keys):
            values = rank_values[:, column]
            maximum = float(values.max().item())
            average = float(values.mean().item())
            slowest_rank = int(values.argmax().item())
            top_values, top_ranks = torch.topk(values, k=min(3, world_size))
            timings[key] = {
                "min_seconds": float(values.min().item()),
                "avg_seconds": average,
                "max_seconds": maximum,
                "max_over_avg": maximum / average if average > 0 else 0.0,
                "slowest_rank": slowest_rank,
                "slowest_node": slowest_rank // local_world_size,
                "slowest_local_rank": slowest_rank % local_world_size,
                "slowest_rank_pointcloud_samples": int(pointcloud_samples_by_rank[slowest_rank].item()),
                "slowest_rank_dataset_skip_count": int(dataset_skip_count_by_rank[slowest_rank].item()),
                "top_slowest": [
                    {"rank": int(rank), "seconds": float(value)}
                    for value, rank in zip(top_values.tolist(), top_ranks.tolist())
                ],
            }

        global_samples = int(samples_by_rank.sum().item())
        iteration_max = timings["iteration"]["max_seconds"]
        record = {
            "timestamp_unix": time.time(),
            "training_step": int(training_step),
            "optimizer_step": int(self.optimizer_step),
            "skipped_step": bool(skipped_step),
            "world_size": world_size,
            "local_world_size": local_world_size,
            "micro_steps": len(timing_records),
            "max_views_per_gpu": int(getattr(self.config, "max_views_per_gpu", 0)),
            "native_view_microsteps": native_view_microsteps,
            "local_samples": int(samples_by_rank[int(self.config.rank)].item()),
            "global_samples": global_samples,
            "global_samples_per_second": global_samples / iteration_max if iteration_max > 0 else 0.0,
            "pointcloud_samples": int(pointcloud_samples_by_rank.sum().item()),
            "pointcloud_ranks": torch.nonzero(pointcloud_samples_by_rank, as_tuple=False).flatten().tolist(),
            "dataset_skip_count": int(dataset_skip_count_by_rank.sum().item()),
            "timings": timings,
        }
        if int(self.config.rank) != 0:
            return
        path = Path(self.config.save_root) / "performance.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _aggregate_log_records(self, records: list[dict[str, Any]]) -> dict[str, float]:
        """Reduce one logging window with conditional geometry denominators."""

        zero = torch.zeros((), device=self.device, dtype=torch.float64)
        reduction_parts = []
        reduction_layout = []
        branches = getattr(self, "optimization_branches", frozenset("vag"))
        sample_log_keys = ["total_loss_raw", "total_norm"]
        if "v" in branches:
            sample_log_keys.extend(("weighted_video_loss_raw", "latent_loss_raw"))
        if "a" in branches:
            sample_log_keys.extend(("weighted_action_loss_raw", "action_loss_raw"))
        if "g" in branches:
            sample_log_keys.extend(("weighted_depth_loss_raw", "weighted_point_loss_raw"))
            sample_log_keys.extend(MOT_GEOMETRY_LOG_KEYS)

        for key in sample_log_keys:
            value_sum = zero.clone()
            if key in MOT_GEOMETRY_GLOBAL_DETAIL_KEYS + MOT_GEOMETRY_MAX_LOG_KEYS:
                continue
            value_count = zero.clone()
            geometry_key = key in MOT_GEOMETRY_LOG_KEYS
            for record in records:
                value = record.get(key)
                if not torch.is_tensor(value) or value.numel() != 1:
                    continue
                value = value.detach().to(device=self.device, dtype=torch.float64)
                valid = torch.isfinite(value)
                if geometry_key:
                    geometry_valid = record.get("geometry_metric_valid")
                    valid = valid & torch.as_tensor(geometry_valid, device=self.device, dtype=torch.bool)
                value_sum = value_sum + torch.where(valid, value, zero)
                value_count = value_count + valid.to(dtype=torch.float64)
            reduction_parts.extend((value_sum, value_count))
            reduction_layout.append((key, len(reduction_parts) - 2))

        data_count_keys = [
            "data_pointcloud_samples",
            "data_pure_samples",
            "data_dataset_skip_count",
            "grad_clip_event",
            "grad_clip_count",
        ]
        if "v" in branches:
            data_count_keys.extend(
                (
                    "data_video_supervised_num",
                    "data_video_supervised_den",
                    "data_video_valid_num",
                    "data_video_valid_den",
                )
            )
        if "a" in branches:
            data_count_keys.extend(
                (
                    "data_action_supervised_num",
                    "data_action_supervised_den",
                    "data_action_valid_num",
                    "data_action_valid_den",
                )
            )
        if "g" in branches:
            data_count_keys.extend(
                (
                    "data_geometry_valid_slot_num",
                    "data_geometry_valid_slot_den",
                    "data_geometry_valid_pixel_num",
                    "data_geometry_valid_pixel_den",
                )
            )
        data_start = len(reduction_parts)
        for key in data_count_keys:
            values = [
                record[key].detach().to(device=self.device, dtype=torch.float64)
                for record in records
                if torch.is_tensor(record.get(key))
            ]
            reduction_parts.append(torch.stack(values).sum() if values else zero.clone())

        reduced = torch.stack(reduction_parts)
        if dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

        result = {}
        for key, offset in reduction_layout:
            value_sum = reduced[offset]
            value_count = reduced[offset + 1]
            result[key] = (
                float((value_sum / value_count).item())
                if value_count.item() > 0
                else float("nan")
            )
        for index, key in enumerate(data_count_keys):
            result[key] = float(reduced[data_start + index].item())

        max_keys = []
        if "v" in branches:
            max_keys.append("latent_loss_raw")
        if "a" in branches:
            max_keys.append("action_loss_raw")
        if "g" in branches:
            max_keys.extend(("depth_loss_raw", "point_loss_raw", *MOT_GEOMETRY_MAX_LOG_KEYS))

        def reduce_maxima() -> None:
            local_max = []
            for key in max_keys:
                values = []
                geometry_key = key in {"depth_loss_raw", "point_loss_raw"} or key in MOT_GEOMETRY_MAX_LOG_KEYS
                for record in records:
                    value = record.get(key)
                    if not torch.is_tensor(value) or value.numel() != 1 or not bool(torch.isfinite(value).item()):
                        continue
                    if geometry_key and not bool(record["geometry_metric_valid"].item()):
                        continue
                    values.append(value.detach().to(device=self.device, dtype=torch.float64))
                local_max.append(torch.stack(values).max() if values else zero.new_full((), float("-inf")))
            if not local_max:
                return
            maxima = torch.stack(local_max)
            if dist.is_initialized():
                dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
            for key, value in zip(max_keys, maxima):
                output_key = key if key in MOT_GEOMETRY_MAX_LOG_KEYS else f"max_{key}"
                result[output_key] = float(value.item()) if torch.isfinite(value) else float("nan")

        if "g" not in branches:
            reduce_maxima()
            return result

        detail_specs = (
            ("depth_error_histogram", VGGT_MONITOR_ERROR_HISTOGRAM_BINS),
            ("point_error_histogram", VGGT_MONITOR_ERROR_HISTOGRAM_BINS),
            ("depth_conf_error_correlation_stats", 6),
            ("point_conf_error_correlation_stats", 6),
            ("point_normal_angle_stats", 2),
        )
        detail_parts = []
        detail_layout = {}
        detail_offset = 0
        for key, size in detail_specs:
            values = [
                record[key].detach().to(device=self.device, dtype=torch.float64).reshape(size)
                for record in records
                if torch.is_tensor(record.get(key))
                and record[key].numel() == size
                and bool(record["geometry_metric_valid"].item())
            ]
            detail_parts.append(torch.stack(values).sum(dim=0) if values else zero.new_zeros(size))
            detail_layout[key] = (detail_offset, size)
            detail_offset += size
        detail_reduced = torch.cat(detail_parts)
        if dist.is_initialized():
            dist.all_reduce(detail_reduced, op=dist.ReduceOp.SUM)

        def detail_value(key: str) -> torch.Tensor:
            offset, size = detail_layout[key]
            return detail_reduced[offset : offset + size]

        depth_quantiles = geometry_error_quantiles_from_histogram(detail_value("depth_error_histogram"))
        point_quantiles = geometry_error_quantiles_from_histogram(detail_value("point_error_histogram"))
        depth_correlation = geometry_pearson_from_stats(detail_value("depth_conf_error_correlation_stats"))
        point_correlation = geometry_pearson_from_stats(detail_value("point_conf_error_correlation_stats"))
        normal_sum, normal_count = detail_value("point_normal_angle_stats")
        normal_mean = torch.where(
            normal_count > 0,
            normal_sum / normal_count.clamp_min(1),
            normal_sum.new_full((), float("nan")),
        )
        result.update(
            {
                "depth_error_p50": float(depth_quantiles[0].item()),
                "depth_error_p90": float(depth_quantiles[1].item()),
                "point_error_p50": float(point_quantiles[0].item()),
                "point_error_p90": float(point_quantiles[1].item()),
                "depth_conf_error_correlation": float(depth_correlation.item()),
                "point_conf_error_correlation": float(point_correlation.item()),
                "point_normal_angle_mean": float(normal_mean.item()),
            }
        )
        reduce_maxima()
        return result

    def _consume_runtime_cache_stats(self, batch: dict[str, Any]) -> None:
        raw_stats = batch.pop("_runtime_cache_stats", None)
        payloads = raw_stats if isinstance(raw_stats, (list, tuple)) else (raw_stats,)
        has_pointcloud = batch.get("has_pointcloud")
        pointcloud_flags = has_pointcloud.reshape(-1) if torch.is_tensor(has_pointcloud) else None
        for sample_idx, payload in enumerate(payloads):
            try:
                stats = decode_mot_runtime_cache_stats(payload)
            except ValueError:
                continue
            source = "unknown"
            if pointcloud_flags is not None and sample_idx < pointcloud_flags.numel():
                source = "pointcloud" if bool(pointcloud_flags[sample_idx].item()) else "non_pointcloud"
            worker_pid = stats.get("worker_pid")
            if worker_pid is not None:
                self._worker_cache_stats.setdefault(int(worker_pid), {})[source] = stats

    def _memory_snapshot_due(self, step: int) -> bool:
        if not bool(getattr(self, "memory_jsonl_enabled", False)):
            return False
        interval = max(1, int(getattr(self, "memory_jsonl_interval", 100)))
        return int(step) == 1 or int(step) % interval == 0

    def _memory_snapshot_needs_smaps(self, step: int) -> bool:
        interval = max(1, int(getattr(self, "memory_smaps_interval", 500)))
        return int(step) <= 1 or int(step) % interval == 0

    def _worker_memory_records(self, *, include_expensive: bool) -> list[dict[str, Any]]:
        cache_by_pid = getattr(self, "_worker_cache_stats", {})
        records = []
        for worker in getattr(getattr(self, "train_loader_iter", None), "_workers", ()):
            pid = int(worker.pid)
            record = _read_process_memory(pid, include_expensive=include_expensive)
            cache_stats = cache_by_pid.get(pid)
            if cache_stats is not None:
                record["caches"] = {source: dict(stats) for source, stats in cache_stats.items()}
            records.append(record)
        return records

    def _local_memory_snapshot(
        self,
        *,
        phase: str,
        include_smaps: bool,
        event: dict[str, Any] | None,
    ) -> dict[str, Any]:
        rank = int(getattr(self.config, "rank", 0))
        local_rank = int(getattr(self.config, "local_rank", 0))
        record = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": socket.gethostname(),
            "phase": str(phase),
            "process": _read_process_memory(os.getpid(), include_expensive=include_smaps),
            "workers": self._worker_memory_records(include_expensive=include_smaps),
            "gc": _read_gc_stats(),
            "allocator": _read_allocator_stats(),
        }
        if local_rank == 0:
            record["cgroup"] = _read_cgroup_memory()
            record["host_memory"] = _read_host_memory()
        if event is not None:
            record["event"] = event
        return record

    def _write_memory_snapshot(
        self,
        *,
        phase: str,
        include_smaps: bool = False,
        event: dict[str, Any] | None = None,
    ) -> None:
        if not bool(getattr(self, "memory_jsonl_enabled", False)):
            return
        local_record = self._local_memory_snapshot(
            phase=phase,
            include_smaps=include_smaps,
            event=event,
        )
        if dist.is_initialized():
            rank_records = [None] * dist.get_world_size()
            dist.all_gather_object(rank_records, local_record)
        else:
            rank_records = [local_record]
        if int(getattr(self.config, "rank", 0)) != 0:
            return
        record = {
            "timestamp_unix": time.time(),
            "training_step": int(self.step),
            "phase": str(phase),
            "ranks": rank_records,
            "nodes": [
                {
                    "rank": rank_record["rank"],
                    "hostname": rank_record["hostname"],
                    "cgroup": rank_record["cgroup"],
                    "host_memory": rank_record["host_memory"],
                }
                for rank_record in rank_records
                if int(rank_record.get("local_rank", -1)) == 0
            ],
        }
        path = Path(self.config.save_root) / "memory.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _performance_metrics(self, timing_records: list[dict[str, Any]]) -> dict[str, float]:
        """Collect low-frequency timing and memory values, taking the slowest rank."""

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        local_times = {key: 0.0 for key in ("data_load", "forward", "backward", "optimizer")}
        for record in timing_records:
            local_times["data_load"] += float(record.get("data_load", 0.0))
            phases = record.get("phases", {})
            for key in ("forward", "backward", "optimizer"):
                local_times[key] += self._phase_elapsed_seconds(phases.get(key))
        local_times["step"] = sum(local_times.values())

        worker_rss = 0.0
        for worker in getattr(self.train_loader_iter, "_workers", ()):
            worker_rss += _process_rss_mib(worker.pid)
        if self.device.type == "cuda":
            gpu_allocated = torch.cuda.memory_allocated(self.device) / (1024.0**2)
            gpu_reserved = torch.cuda.memory_reserved(self.device) / (1024.0**2)
            gpu_peak = torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
        else:
            gpu_allocated = gpu_reserved = gpu_peak = 0.0

        values = torch.tensor(
            [
                local_times["data_load"],
                local_times["forward"],
                local_times["backward"],
                local_times["optimizer"],
                local_times["step"],
                gpu_allocated,
                gpu_reserved,
                gpu_peak,
                _process_rss_mib(os.getpid()),
                worker_rss,
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.MAX)

        local_timed_samples = sum(int(record.get("local_samples", 1)) for record in timing_records)
        timed_samples_tensor = torch.tensor(
            local_timed_samples,
            device=self.device,
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(timed_samples_tensor, op=dist.ReduceOp.SUM)
        else:
            timed_samples_tensor *= int(getattr(self.config, "world_size", 1))
        timed_samples = float(timed_samples_tensor.item())
        step_time = float(values[4].item())
        result = {
            "perf/data_load_time_max": float(values[0].item()),
            "perf/forward_time_max": float(values[1].item()),
            "perf/backward_time_max": float(values[2].item()),
            "perf/optimizer_time_max": float(values[3].item()),
            "perf/step_time_max": step_time,
            "perf/samples_per_second": timed_samples / step_time if step_time > 0 else float("nan"),
            "memory/gpu_allocated_max": float(values[5].item()),
            "memory/gpu_reserved_max": float(values[6].item()),
            "memory/gpu_peak_allocated_max": float(values[7].item()),
            "memory/host_rss": float(values[8].item()),
            "memory/dataloader_workers_rss": float(values[9].item()),
        }
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        return result

    def _get_train_vae(self):
        if self.train_vae is None:
            self.train_vae = load_vae(
                os.path.join(self.config.wan22_pretrained_model_name_or_path, "vae"),
                torch_dtype=self.dtype,
                torch_device=self.device,
            ).eval()
            self.train_vae.requires_grad_(False)
        return self.train_vae

    def _get_empty_text_emb(self):
        if self.empty_text_emb is None:
            empty_path = Path(getattr(self.config, "empty_emb_path"))
            if not empty_path.is_file():
                raise FileNotFoundError(empty_path)
            self.empty_text_emb = torch.load(empty_path, map_location="cpu", weights_only=False)
        return self.empty_text_emb

    def _train_text_emb(
        self,
        text_emb: torch.Tensor,
        empty_text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg_prob = float(getattr(self.config, "cfg_prob", 0.0))
        if cfg_prob <= 0.0:
            return text_emb
        return _apply_cfg_text_dropout(
            text_emb,
            self._get_empty_text_emb() if empty_text_emb is None else empty_text_emb,
            cfg_prob,
            training=True,
        )

    @torch.no_grad()
    def _encode_one_view_latent(self, frames: torch.Tensor) -> torch.Tensor:
        spec = _mot_spec_from_config(self.config)
        if frames.shape[0] != spec.vae_input_frame_count:
            raise ValueError(f"MOT VAE input must have {spec.vae_input_frame_count} frames, got {frames.shape[0]}")
        vae = self._get_train_vae()
        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype
        wrapper = WanVAEStreamingWrapper(vae)
        video = frames.permute(1, 0, 2, 3)[None].to(device=vae_device, dtype=vae_dtype) * 2.0 - 1.0
        chunks = [wrapper.encode_chunk(video[:, :, :1])]
        for latent_idx in range(1, spec.latent_frames_per_action_chunk_per_view):
            start = 1 + spec.vae_temporal_factor * (latent_idx - 1)
            end = start + spec.vae_temporal_factor
            chunks.append(wrapper.encode_chunk(video[:, :, start:end]))
        enc = torch.cat(chunks, dim=2)
        mu, _logvar = torch.chunk(enc, 2, dim=1)
        latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
        latent = (mu - latents_mean) / latents_std
        return latent[0].to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _encode_vae_rgb(self, vae_rgb: torch.Tensor) -> torch.Tensor:
        if vae_rgb.ndim != 6:
            raise ValueError(f"MOT vae_rgb must be [B,T,V,C,H,W], got {tuple(vae_rgb.shape)}")
        batch_size, _, views = vae_rgb.shape[:3]
        per_batch = []
        for batch_idx in range(batch_size):
            per_view = []
            for view_idx in range(views):
                per_view.append(self._encode_one_view_latent(vae_rgb[batch_idx, :, view_idx]))
            per_batch.append(torch.stack(per_view, dim=2))
        return torch.stack(per_batch, dim=0)

    @torch.no_grad()
    def _materialize_batch_latents(self, batch_dict):
        if "latents" in batch_dict:
            return batch_dict
        if "vae_rgb_history" not in batch_dict or "vae_rgb_target" not in batch_dict:
            raise KeyError("MOT batch must contain either latents or vae_rgb_history/vae_rgb_target")
        out = dict(batch_dict)
        history = self._encode_vae_rgb(batch_dict["vae_rgb_history"])
        target = self._encode_vae_rgb(batch_dict["vae_rgb_target"])
        out["latents"] = torch.cat([history, target], dim=2)
        return out

    @torch.no_grad()
    def _add_video_noise(
        self,
        latent,
        train_scheduler,
        spec: MOTWindowSpec,
        video_latent_loss_mask,
        noisy_cond_prob=0.0,
    ):
        if latent.ndim not in {5, 6}:
            raise ValueError(f"MOT latent/action tensor must be [B,C,F,...], got {tuple(latent.shape)}")
        batch_size, _, frames = latent.shape[:3]
        if frames != spec.total_latent_frames:
            raise ValueError(f"MOT video expects F={spec.total_latent_frames} latent frames, got {frames}")
        video_latent_loss_mask = video_latent_loss_mask.to(device=latent.device, dtype=torch.bool)
        if tuple(video_latent_loss_mask.shape) != (batch_size, frames):
            raise ValueError(
                f"video_latent_loss_mask must be [B,{frames}], got {tuple(video_latent_loss_mask.shape)}"
            )

        # NOTE: 兼容 chunk > 1 的情况，但是现在默认只有一个 chunk （1 个 chunk = 1 history + 1 tgt）
        timesteps = _sample_mot_chunk_timesteps(
            batch_size=batch_size,
            spec=spec,
            train_scheduler=train_scheduler,
            device=self.device,
        )
        # non-valid 处 t 归零
        timesteps = torch.where(video_latent_loss_mask, timesteps, torch.zeros_like(timesteps))
        noise = torch.zeros_like(latent).normal_()
        sigma = _scheduler_sigmas_for_timesteps(train_scheduler, timesteps, dtype=latent.dtype, device=latent.device)
        sigma_shape = [batch_size, 1, frames] + [1] * (latent.ndim - 3)

        # 加噪
        noisy_latents = (1.0 - sigma.reshape(sigma_shape)) * latent + sigma.reshape(sigma_shape) * noise
        # velocity gt
        targets = train_scheduler.training_target(latent, noise, timesteps)

        keep_shape = [batch_size, 1, frames] + [1] * (latent.ndim - 3)
        keep = video_latent_loss_mask.reshape(keep_shape)

        # non-valid 处不加噪且 gt velocity为 0
        noisy_latents = torch.where(keep, noisy_latents, latent)
        targets = torch.where(keep, targets, torch.zeros_like(targets))

        # batch 级别选择是否给 clean condition 加噪
        if torch.rand(1).item() < noisy_cond_prob:
            # moderate noise in the snr_shift=5 config
            cond_timesteps = _sample_mot_chunk_timesteps(
                batch_size=batch_size,
                spec=spec,
                train_scheduler=train_scheduler,
                device=self.device,
                min_timestep_bd=0.77,
                max_timestep_bd=0.95,
            )
            cond_noise = torch.zeros_like(latent).normal_()
            cond_sigma = _scheduler_sigmas_for_timesteps(train_scheduler, cond_timesteps, dtype=latent.dtype, device=latent.device)
            # history 不加噪 （理论上 t=0，但是现在设置了 extra_one_step=True，严格来说是一步微小噪声），而 target 根据 cond_timesteps 加噪
            latent = (1.0 - cond_sigma.reshape(sigma_shape)) * latent + cond_sigma.reshape(sigma_shape) * cond_noise
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        return {
            "timesteps": timesteps,
            "noisy_latents": noisy_latents,
            "targets": targets,
            "latent": latent,
            "cond_timesteps": cond_timesteps,
        }

    @torch.no_grad()
    def _add_action_noise(self, actions, train_scheduler, spec: MOTWindowSpec, action_loss_mask):
        expected_shape = (spec.total_latent_frames, spec.action_per_frame, 1)
        if actions.ndim != 5 or actions.shape[2:] != expected_shape:
            raise ValueError(
                "MOT actions must be "
                f"[B,C,{spec.total_latent_frames},{spec.action_per_frame},1], got {tuple(actions.shape)}"
            )
        batch_size = actions.shape[0]
        action_loss_mask = action_loss_mask.to(device=actions.device, dtype=torch.bool)
        if action_loss_mask.shape != actions.shape:
            raise ValueError(
                f"action_loss_mask must be {tuple(actions.shape)}, got {tuple(action_loss_mask.shape)}"
            )
        timesteps = _sample_mot_chunk_timesteps(
            batch_size=batch_size,
            spec=spec,
            train_scheduler=train_scheduler,
            device=self.device,
        )

        # [B,C,F,N,1] -> [B,F]
        action_frame_loss_mask = action_loss_mask.any(dim=(1, 3, 4))
        # non-valid 部分 t=0
        timesteps = torch.where(action_frame_loss_mask, timesteps, torch.zeros_like(timesteps))
        noise = torch.zeros_like(actions).normal_()
        sigma = _scheduler_sigmas_for_timesteps(train_scheduler, timesteps, dtype=actions.dtype, device=actions.device)
        sigma_shape = [batch_size, 1, spec.total_latent_frames, 1, 1]
        noisy_latents = (1.0 - sigma.reshape(sigma_shape)) * actions + sigma.reshape(sigma_shape) * noise
        targets = train_scheduler.training_target(actions, noise, timesteps)

        # non-vaild 部分为 clean action
        noisy_latents = torch.where(action_loss_mask, noisy_latents, actions)
        targets = torch.where(action_loss_mask, targets, torch.zeros_like(targets))
        return {
            "timesteps": timesteps,
            "noisy_latents": noisy_latents,
            "targets": targets,
            "latent": actions,
            "cond_timesteps": torch.zeros_like(timesteps),
        }


    @torch.no_grad()
    def _prepare_geometry_fields(self, batch_dict):
        return {
            "rgb": batch_dict["geometry_rgb"].to(
                dtype=getattr(self, "dtype", batch_dict["geometry_rgb"].dtype)
            ),
            "pts3d": batch_dict["geometry_pts3d"],
            "valid_mask": batch_dict["geometry_point_valid_mask"],
            "slot_valid_mask": batch_dict["geometry_group_valid_mask"],
        }

    @torch.no_grad()
    def _prepare_joint_input_dict(self, batch_dict, *, add_noise=True):
        spec = _mot_spec_from_config(self.config)
        video_latent_loss_mask = batch_dict["video_latent_loss_mask"].to(
            device=batch_dict["latents"].device,
            dtype=torch.bool,
        )
        video_latent_valid_mask = batch_dict["video_latent_valid_mask"].to(
            device=batch_dict["latents"].device,
            dtype=torch.bool,
        )
        if video_latent_valid_mask.shape != video_latent_loss_mask.shape:
            raise ValueError(
                f"video_latent_valid_mask must be {tuple(video_latent_loss_mask.shape)}, got {tuple(video_latent_valid_mask.shape)}"
            )
        action_loss_mask = batch_dict["action_loss_mask"].to(
            device=batch_dict["actions"].device,
            dtype=torch.bool,
        )

        action_valid_mask = batch_dict["action_valid_mask"].to(
            device=batch_dict["actions"].device,
            dtype=torch.bool,
        )
        if add_noise:
            latent_dict = self._add_video_noise(
                latent=batch_dict["latents"],
                train_scheduler=self.train_scheduler_latent,
                spec=spec,
                noisy_cond_prob=float(getattr(self.config, "video_noisy_cond_prob", 0.5)),
                video_latent_loss_mask=video_latent_loss_mask,
            )
            action_dict = self._add_action_noise(
                actions=batch_dict["actions"],
                train_scheduler=self.train_scheduler_action,
                spec=spec,
                action_loss_mask=action_loss_mask,
            )
        else:
            latent_dict = {"latent": batch_dict["latents"]}
            action_dict = {"latent": batch_dict["actions"]}
        text_emb = self._train_text_emb(
            batch_dict["text_emb"],
            batch_dict.get("empty_text_emb"),
        )

        latent_dict["text_emb"] = text_emb
        latent_dict["video_latent_loss_mask"] = video_latent_loss_mask
        latent_dict["video_latent_valid_mask"] = video_latent_valid_mask
        action_dict["text_emb"] = text_emb
        action_dict["action_loss_mask"] = action_loss_mask
        action_dict["action_valid_mask"] = action_valid_mask

        geometry_dict = self._prepare_geometry_fields(batch_dict)
        geometry_dict["stream_ids"] = batch_dict["stream_ids"]

        input_dict = {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "geometry_dict": geometry_dict,
            "monitoring_dict": {
                "has_pointcloud": batch_dict.get("has_pointcloud"),
                "dataset_skip_count": batch_dict.get("dataset_skip_count"),
            },
            "chunk_size": spec.latent_frames_per_action_chunk_per_view,
            "window_size": spec.attention_window_size,
        }
        validate_mot_batch_for_forward(batch_dict, action_sequence_length=spec.action_sequence_length)
        return input_dict

    def _prepare_geometry_input_dict(self, batch_dict):
        spec = _mot_spec_from_config(self.config)
        validate_mot_geometry_batch(
            batch_dict,
            action_sequence_length=spec.action_sequence_length,
        )
        selected = {
            key: batch_dict[key]
            for key in (
                "geometry_rgb",
                "geometry_pts3d",
                "geometry_point_valid_mask",
                "geometry_group_valid_mask",
                "has_pointcloud",
            )
        }
        if "dataset_skip_count" in batch_dict:
            selected["dataset_skip_count"] = batch_dict["dataset_skip_count"]
        selected = _move_to_device(
            selected,
            self.device,
            non_blocking=(self.device.type == "cuda"),
        )
        geometry_dict = self._prepare_geometry_fields(selected)
        model_input = {
            "rgb": geometry_dict["rgb"],
            "slot_valid_mask": geometry_dict["slot_valid_mask"],
            "chunk_size": spec.latent_frames_per_action_chunk_per_view,
            "window_size": spec.attention_window_size,
        }
        return {
            "geometry_model_input": model_input,
            "geometry_dict": {
                key: geometry_dict[key]
                for key in ("pts3d", "valid_mask", "slot_valid_mask")
            },
            "monitoring_dict": {
                "has_pointcloud": selected["has_pointcloud"],
                "dataset_skip_count": selected.get("dataset_skip_count"),
            },
        }

    def convert_input_format(self, input_dict):
        return _move_to_device(input_dict, self.device, non_blocking=(self.device.type == "cuda"))

    def _compute_video_objective(self, input_dict, pred):
        latent_targets = input_dict["latent_dict"]["targets"]
        batch_size, frames = input_dict["latent_dict"]["timesteps"].shape
        latent_training_weight = self.train_scheduler_latent.training_weight(
            input_dict["latent_dict"]["timesteps"].flatten()
        ).reshape(batch_size, frames)
        video_latent_loss_mask = input_dict["latent_dict"].get("video_latent_loss_mask")
        latent_loss = _frame_weighted_mse(
            pred["latent_pred"],
            latent_targets,
            latent_training_weight,
            video_latent_loss_mask=video_latent_loss_mask,
        )
        video_weight = float(self.config.video_loss_weight)
        weighted = video_weight * latent_loss
        valid_mask = input_dict["latent_dict"]["video_latent_valid_mask"]
        return weighted, {
            "latent_loss_raw": latent_loss.detach(),
            "weighted_video_loss_raw": weighted.detach(),
            "video_loss_weight": latent_loss.new_tensor(video_weight),
            "data_video_supervised_num": video_latent_loss_mask.sum().to(dtype=torch.float32),
            "data_video_supervised_den": latent_loss.new_tensor(video_latent_loss_mask.numel()),
            "data_video_valid_num": valid_mask.sum().to(dtype=torch.float32),
            "data_video_valid_den": latent_loss.new_tensor(valid_mask.numel()),
        }

    def _compute_action_objective(self, input_dict, pred):
        action_targets = input_dict["action_dict"]["targets"]
        action_timesteps = input_dict["action_dict"]["timesteps"]
        action_training_weight = self.train_scheduler_action.training_weight(
            action_timesteps.flatten()
        ).reshape(action_timesteps.shape)
        action_loss = _action_weighted_mse(
            pred["action_pred"],
            action_targets,
            action_training_weight,
            input_dict["action_dict"]["action_loss_mask"],
        )
        action_weight = float(self.config.action_loss_weight)
        weighted = action_weight * action_loss
        action_loss_mask = input_dict["action_dict"]["action_loss_mask"]
        action_valid_mask = input_dict["action_dict"]["action_valid_mask"]
        return weighted, {
            "action_loss_raw": action_loss.detach(),
            "weighted_action_loss_raw": weighted.detach(),
            "action_loss_weight": action_loss.new_tensor(action_weight),
            "data_action_supervised_num": action_loss_mask.sum().to(dtype=torch.float32),
            "data_action_supervised_den": action_loss.new_tensor(action_loss_mask.numel()),
            "data_action_valid_num": action_valid_mask.sum().to(dtype=torch.float32),
            "data_action_valid_den": action_loss.new_tensor(action_valid_mask.numel()),
        }

    def _compute_geometry_objective(self, input_dict, pred):
        detailed_metrics = bool(input_dict.get("detailed_metrics", False))
        geometry = input_dict["geometry_dict"]
        geometry_valid_mask = geometry["valid_mask"].bool()
        geometry_slot_mask = geometry["slot_valid_mask"].bool()
        batch_size = geometry_valid_mask.shape[0]
        depth_weight = float(self.config.depth_loss_weight)
        point_weight = float(self.config.point_loss_weight)
        component_losses = []
        component_valid_counts = []
        metrics = {}

        if depth_weight > 0:
            depth_loss, depth_metrics = compute_vggto_depth_loss(
                pred["depth"],
                pred["depth_conf"],
                geometry["pts3d"],
                geometry_valid_mask,
                gradient_loss_fn=getattr(self.config, "gradient_loss_fn", "grad"),
                valid_range=float(getattr(self.config, "valid_range", 0.98)),
                gamma=float(getattr(self.config, "gamma", 1.0)),
                alpha=float(getattr(self.config, "alpha", 0.2)),
                detailed_metrics=detailed_metrics,
            )
            component_losses.append(("depth", depth_weight, depth_loss))
            component_valid_counts.append(depth_metrics["depth_valid_pixels"])
            metrics.update(depth_metrics)
        if point_weight > 0:
            point_loss, point_metrics = compute_vggto_point_loss(
                pred["points"],
                pred["points_conf"],
                geometry["pts3d"],
                geometry_valid_mask,
                gradient_loss_fn=getattr(self.config, "point_gradient_loss_fn", "normal"),
                valid_range=float(getattr(self.config, "valid_range", 0.98)),
                gamma=float(getattr(self.config, "gamma", 1.0)),
                alpha=float(getattr(self.config, "alpha", 0.2)),
                detailed_metrics=detailed_metrics,
            )
            component_losses.append(("point", point_weight, point_loss))
            component_valid_counts.append(point_metrics["point_valid_pixels"])
            metrics.update(point_metrics)

        reference = component_losses[0][2]
        monitoring = input_dict.get("monitoring_dict", {})
        has_pointcloud = monitoring.get("has_pointcloud")
        if has_pointcloud is None:
            has_pointcloud = geometry_valid_mask.reshape(batch_size, -1).any(dim=1)
        else:
            has_pointcloud = has_pointcloud.to(device=reference.device, dtype=torch.bool).reshape(batch_size)
        pointcloud_samples = has_pointcloud.sum()
        geometry_rank_active = (
            (pointcloud_samples > 0)
            & torch.stack(
                [count.to(device=reference.device) for count in component_valid_counts]
            ).max().ge(100)
        ).to(dtype=torch.float32)
        geometry_rank_scale, global_active_ranks = _distributed_geometry_active_rank_scale(
            geometry_rank_active
        )
        geometry_rank_scale = geometry_rank_scale.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        geometry_weight = float(self.config.geometry_loss_weight)
        weighted_total = reference.new_zeros(())
        out = {
            "geometry_loss_weight": reference.new_tensor(geometry_weight),
            "geometry_active_rank_scale": geometry_rank_scale.detach(),
            "geometry_global_active_ranks": global_active_ranks.to(device=reference.device),
            "geometry_metric_valid": geometry_rank_active,
        }
        for name, component_weight, component_loss in component_losses:
            weighted = (
                geometry_weight
                * component_weight
                * geometry_rank_scale
                * component_loss
            )
            weighted_total = weighted_total + weighted
            out[f"{name}_loss_raw"] = component_loss.detach()
            out[f"weighted_{name}_loss_raw"] = weighted.detach()
            out[f"{name}_loss_weight"] = component_loss.new_tensor(component_weight)
        out.update(
            {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in metrics.items()
            }
        )
        pointcloud_slot_mask = has_pointcloud[:, None, None]
        out.update(
            data_geometry_valid_slot_num=(geometry_slot_mask & pointcloud_slot_mask).sum().to(dtype=torch.float32),
            data_geometry_valid_slot_den=pointcloud_samples.to(dtype=torch.float32) * geometry_slot_mask[0].numel(),
            data_geometry_valid_pixel_num=geometry_valid_mask.sum().to(dtype=torch.float32),
            data_geometry_valid_pixel_den=pointcloud_samples.to(dtype=torch.float32) * geometry_valid_mask[0].numel(),
        )
        return weighted_total, out

    def compute_loss(self, input_dict, pred):
        branches = getattr(self, "optimization_branches", None)
        if branches is None:
            branches = optimization_branches(self.config.optimization_composition)
        contributions = []
        out = {}
        if "v" in branches:
            weighted, metrics = self._compute_video_objective(input_dict, pred)
            contributions.append(weighted)
            out.update(metrics)
        if "a" in branches:
            weighted, metrics = self._compute_action_objective(input_dict, pred)
            contributions.append(weighted)
            out.update(metrics)
        if "g" in branches:
            weighted, metrics = self._compute_geometry_objective(input_dict, pred)
            contributions.append(weighted)
            out.update(metrics)

        total_loss = contributions[0]
        for contribution in contributions[1:]:
            total_loss = total_loss + contribution
        inv_accum = 1.0 / float(self.gradient_accumulation_steps)
        out["loss"] = total_loss * inv_accum
        out["total_loss_raw"] = total_loss.detach()
        for name in ("latent", "action", "depth", "point"):
            raw = out.get(f"{name}_loss_raw")
            if raw is not None:
                out[f"{name}_loss"] = raw * inv_accum

        geometry_valid_mask = input_dict["geometry_dict"]["valid_mask"].bool()
        batch_size = geometry_valid_mask.shape[0]
        reference = total_loss
        monitoring = input_dict.get("monitoring_dict", {})
        has_pointcloud = monitoring.get("has_pointcloud")
        if has_pointcloud is None:
            has_pointcloud = geometry_valid_mask.reshape(batch_size, -1).any(dim=1)
        else:
            has_pointcloud = has_pointcloud.to(device=reference.device, dtype=torch.bool).reshape(batch_size)
        pointcloud_samples = has_pointcloud.sum()
        dataset_skip_count = monitoring.get("dataset_skip_count")
        if dataset_skip_count is None:
            dataset_skip_count = torch.zeros(batch_size, device=reference.device, dtype=torch.int64)
        else:
            dataset_skip_count = dataset_skip_count.to(device=reference.device, dtype=torch.int64).reshape(batch_size)
        out.update(
            {
                "data_pointcloud_samples": pointcloud_samples.to(dtype=torch.float32),
                "data_pure_samples": (batch_size - pointcloud_samples).to(dtype=torch.float32),
                "data_dataset_skip_count": dataset_skip_count.sum().to(dtype=torch.float32),
                "data_local_samples": reference.new_tensor(batch_size, dtype=torch.float32),
                "data_native_views": reference.new_tensor(
                    geometry_valid_mask.shape[3] if geometry_valid_mask.ndim >= 6 else 0,
                    dtype=torch.float32,
                ),
            }
        )
        return out

    def _distributed_any(self, value: bool) -> bool:
        if not dist.is_initialized():
            return bool(value)
        flag = torch.tensor(1 if value else 0, device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _losses_all_finite(self, losses: dict[str, Any]) -> bool:
        for key in ("loss", "latent_loss", "action_loss", "depth_loss", "point_loss"):
            value = losses.get(key)
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                return False
        return True

    def _nonfinite_grad_locations(self, max_items: int = 20) -> list[dict[str, Any]]:
        locations = []
        for name, param in self.transformer.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            finite = torch.isfinite(grad)
            if bool(finite.all()):
                continue
            locations.append(
                {
                    "kind": "grad",
                    "name": name,
                    "num_nonfinite": int((~finite).sum().detach().cpu().item()),
                }
            )
            if len(locations) >= int(max_items):
                break
        return locations

    def _write_nan_log(
        self,
        *,
        trigger_stage: str,
        losses: dict[str, Any],
        total_norm,
        loss_finite: bool,
        grad_finite: bool | None,
        global_has_nan: bool,
        rank_has_nan: bool,
        nan_locations: list[dict[str, Any]],
        batch: dict[str, Any],
    ) -> None:
        local_record = {
            "rank": int(self.config.rank),
            "loss_finite": bool(loss_finite),
            "grad_finite": None if grad_finite is None else bool(grad_finite),
            "rank_has_nan": bool(rank_has_nan),
            "nan_locations": nan_locations,
        }
        if dist.is_initialized():
            rank_records = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(rank_records, local_record)
        else:
            rank_records = [local_record]
        if int(self.config.rank) != 0:
            return
        all_locations = []
        for item in rank_records:
            if not item:
                continue
            for location in item.get("nan_locations", []):
                location = dict(location)
                location["rank"] = int(item["rank"])
                all_locations.append(location)
        record = _make_nan_log_record(
            step=self.step,
            trigger_stage=trigger_stage,
            losses=losses,
            total_norm=total_norm,
            loss_finite=loss_finite,
            grad_finite=grad_finite,
            global_has_nan=global_has_nan,
            rank_has_nan=rank_has_nan,
            rank=int(self.config.rank),
            nan_locations=all_locations[:20],
            batch=batch,
            rank_records=rank_records,
        )
        path = Path(self.config.save_root) / "nan.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _skip_optimizer_step_for_nan(
        self,
        *,
        trigger_stage: str,
        losses: dict[str, Any],
        total_norm,
        loss_finite: bool,
        grad_finite: bool | None,
        global_has_nan: bool,
        rank_has_nan: bool,
        nan_locations: list[dict[str, Any]],
        batch: dict[str, Any],
    ) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._write_nan_log(
            trigger_stage=trigger_stage,
            losses=losses,
            total_norm=total_norm,
            loss_finite=loss_finite,
            grad_finite=grad_finite,
            global_has_nan=global_has_nan,
            rank_has_nan=rank_has_nan,
            nan_locations=nan_locations,
            batch=batch,
        )

    def _train_step(
        self,
        batch,
        batch_idx,
        *,
        collect_detailed_metrics: bool = False,
        measure_performance: bool = False,
    ):
        phase_timings = {}
        forward_timer = self._start_phase_timer() if measure_performance else None
        batch_prepare_timer = self._start_phase_timer() if measure_performance else None
        execution_route = getattr(self, "execution_route", "joint")
        if execution_route == "geometry":
            input_transfer_timer = self._start_phase_timer() if measure_performance else None
            input_dict = self._prepare_geometry_input_dict(batch)
            if measure_performance:
                phase_timings["input_transfer"] = self._stop_phase_timer(input_transfer_timer)
                phase_timings["vae_encode"] = 0.0
                phase_timings["input_prepare"] = 0.0
            model_input = input_dict["geometry_model_input"]
            model_mode = "train_geometry"
        else:
            input_transfer_timer = self._start_phase_timer() if measure_performance else None
            batch = self.convert_input_format(batch)
            if measure_performance:
                phase_timings["input_transfer"] = self._stop_phase_timer(input_transfer_timer)
            vae_encode_timer = self._start_phase_timer() if measure_performance else None
            batch = self._materialize_batch_latents(batch)
            if measure_performance:
                phase_timings["vae_encode"] = self._stop_phase_timer(vae_encode_timer)
            input_prepare_timer = self._start_phase_timer() if measure_performance else None
            input_dict = self._prepare_joint_input_dict(batch)
            if measure_performance:
                phase_timings["input_prepare"] = self._stop_phase_timer(input_prepare_timer)
            model_input = input_dict
            model_mode = "train"
        if measure_performance:
            phase_timings["batch_prepare"] = self._stop_phase_timer(batch_prepare_timer)
        input_dict["detailed_metrics"] = bool(collect_detailed_metrics)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if hasattr(self.transformer, "set_requires_gradient_sync"):
            self.transformer.set_requires_gradient_sync(should_sync)

        model_forward_timer = self._start_phase_timer() if measure_performance else None
        output = self.transformer(model_input, mode=model_mode)
        if measure_performance:
            phase_timings["model_forward"] = self._stop_phase_timer(model_forward_timer)
        loss_timer = self._start_phase_timer() if measure_performance else None
        losses = self.compute_loss(input_dict, output)
        local_loss_finite = self._losses_all_finite(losses)
        if measure_performance:
            phase_timings["loss"] = self._stop_phase_timer(loss_timer)
        loss_sync_timer = self._start_phase_timer() if measure_performance else None
        global_loss_bad = self._distributed_any(not local_loss_finite)
        if measure_performance:
            phase_timings["loss_finite_sync"] = self._stop_phase_timer(loss_sync_timer)
            phase_timings["forward"] = self._stop_phase_timer(forward_timer)
        losses["nonfinite_loss_event"] = bool(global_loss_bad)
        losses["nonfinite_grad_event"] = False
        losses["optimizer_step_event"] = False
        losses["grad_clip_event"] = torch.zeros((), device=self.device)
        losses["grad_clip_count"] = torch.zeros((), device=self.device)
        if global_loss_bad:
            self._skip_optimizer_step_for_nan(
                trigger_stage="loss",
                losses=losses,
                total_norm=None,
                loss_finite=local_loss_finite,
                grad_finite=None,
                global_has_nan=True,
                rank_has_nan=not local_loss_finite,
                nan_locations=[],
                batch=batch,
            )
            losses["total_norm"] = torch.tensor(float("nan"), device=self.device)
            losses["should_log"] = True
            losses["skipped_step"] = True
            losses["_phase_timings"] = phase_timings
            return {key: value.detach() if torch.is_tensor(value) else value for key, value in losses.items()}

        backward_timer = self._start_phase_timer() if measure_performance else None
        losses["loss"].backward()
        if measure_performance:
            phase_timings["backward"] = self._stop_phase_timer(backward_timer)

        grad_summary = {}
        cross_view_parameters = []
        if should_sync and collect_detailed_metrics:
            cross_view_parameters = _cross_view_named_parameters(self.transformer)
            if not cross_view_parameters:
                raise RuntimeError("no trainable VGGTO cross-view parameters found for monitoring")
            grad_summary["vggto_cross_view"] = _distributed_parameter_norm_stats(
                cross_view_parameters,
                device=self.device,
            )

        if should_sync:
            optimizer_timer = self._start_phase_timer() if measure_performance else None
            grad_clip_timer = self._start_phase_timer() if measure_performance else None
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            if measure_performance:
                phase_timings["grad_clip"] = self._stop_phase_timer(grad_clip_timer)
            local_grad_finite = bool(torch.isfinite(total_norm).all())
            nan_locations = []
            if not local_grad_finite:
                nan_locations = self._nonfinite_grad_locations()
            grad_sync_timer = self._start_phase_timer() if measure_performance else None
            global_grad_bad = self._distributed_any(not local_grad_finite)
            if measure_performance:
                phase_timings["grad_finite_sync"] = self._stop_phase_timer(grad_sync_timer)
            if global_grad_bad:
                losses["nonfinite_grad_event"] = True
                self._skip_optimizer_step_for_nan(
                    trigger_stage="grad",
                    losses=losses,
                    total_norm=total_norm,
                    loss_finite=local_loss_finite,
                    grad_finite=local_grad_finite,
                    global_has_nan=True,
                    rank_has_nan=not local_grad_finite,
                    nan_locations=nan_locations,
                    batch=batch,
                )
                losses["skipped_step"] = True
            else:
                optimizer_step_timer = self._start_phase_timer() if measure_performance else None
                cross_view_snapshots = (
                    _snapshot_local_parameters(cross_view_parameters)
                    if cross_view_parameters
                    else []
                )
                self.optimizer.step()
                if cross_view_snapshots:
                    grad_summary["vggto_cross_view"].update(
                        _distributed_parameter_update_stats(
                            cross_view_snapshots,
                            device=self.device,
                        )
                    )
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                if measure_performance:
                    phase_timings["optimizer_step"] = self._stop_phase_timer(optimizer_step_timer)
                losses["skipped_step"] = False
                losses["optimizer_step_event"] = True
                log_total_norm = _detach_log_tensor(total_norm)
                losses["grad_clip_event"] = (log_total_norm > 2.0).to(dtype=torch.float32)
                losses["grad_clip_count"] = torch.ones((), device=self.device)

            if measure_performance:
                phase_timings["optimizer"] = self._stop_phase_timer(optimizer_timer)

            losses["total_norm"] = _detach_log_tensor(total_norm)
            losses["should_log"] = True
        else:
            losses["should_log"] = False
            losses["skipped_step"] = False
        losses["grad_summary"] = grad_summary
        losses["_phase_timings"] = phase_timings
        return {key: value.detach() if torch.is_tensor(value) else value for key, value in losses.items()}

    @staticmethod
    def _dcp_options() -> StateDictOptions:
        return StateDictOptions(full_state_dict=False, cpu_offload=True, strict=True)

    def _dcp_state_dict(self) -> dict[str, Any]:
        model_state, optimizer_state = get_state_dict(
            self.transformer,
            self.optimizer,
            options=self._dcp_options(),
        )
        return {"model": model_state, "optimizer": optimizer_state}

    def _capture_rng_state(self) -> dict[str, Any]:
        numpy_state = np.random.get_state()
        cuda_state = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state(self.device).cpu()
        return {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_state[0],
                "state": numpy_state[1].tolist(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cpu": torch.get_rng_state().cpu().tolist(),
            "torch_cuda": cuda_state.tolist() if cuda_state is not None else None,
        }

    def _collect_rng_states(self) -> list[dict[str, Any]]:
        local_state = self._capture_rng_state()
        if not dist.is_initialized():
            return [local_state]
        states = [None] * dist.get_world_size()
        group = getattr(self, "checkpoint_process_group", None)
        if group is None:
            dist.all_gather_object(states, local_state)
        else:
            dist.all_gather_object(states, local_state, group=group)
        return states

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None:
            if self.device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("Checkpoint contains CUDA RNG state but the resume rank has no CUDA device")
            torch.cuda.set_rng_state(
                torch.tensor(cuda_state, dtype=torch.uint8),
                self.device,
            )

    def _trainer_state(
        self,
        rng_states_by_rank: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if rng_states_by_rank is None:
            rng_states_by_rank = [self._capture_rng_state()]
        return {
            "format_version": MOT_DCP_FORMAT_VERSION,
            "step": int(self.step),
            "optimizer_step": int(self.optimizer_step),
            "skipped_step_total": int(self.skipped_step_total),
            "nonfinite_loss_total": int(self.nonfinite_loss_total),
            "nonfinite_grad_total": int(self.nonfinite_grad_total),
            "dataset_skip_total": int(self.dataset_skip_total),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "rng_states_by_rank": rng_states_by_rank,
            "metadata": {
                "checkpoint_type": "dcp_sharded",
                "vggto_attention_topology": MOT_VGGTO_TOPOLOGY,
                "torch_version": str(torch.__version__),
                "world_size": int(getattr(self.config, "world_size", 1)),
                "optimization_composition": self.optimization_composition,
                "max_views_per_gpu": int(self.config.max_views_per_gpu),
                "masked_attn_backend": str(getattr(self.config, "masked_attn_backend", "fa4")),
                "sampler_seed": int(getattr(self.config, "sampler_seed", 42)),
            },
        }

    @staticmethod
    def _validate_transformer_checkpoint_layout(checkpoint_path: Path) -> dict[str, Any]:
        required_files = (
            checkpoint_path / "_SUCCESS",
            checkpoint_path / MOT_CHECKPOINT_METADATA_NAME,
            checkpoint_path / "transformer" / "config.json",
            checkpoint_path / "transformer" / MOT_TRANSFORMER_WEIGHTS_NAME,
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Stage initialization requires a completed new-schema checkpoint; missing: "
                + ", ".join(missing)
            )
        with (checkpoint_path / MOT_CHECKPOINT_METADATA_NAME).open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if int(metadata.get("format_version", -1)) != MOT_DCP_FORMAT_VERSION:
            raise ValueError(
                "Unsupported MOT checkpoint format_version: "
                f"{metadata.get('format_version')!r}"
            )
        if metadata.get("checkpoint_type") != "mot_training":
            raise ValueError("MOT checkpoint metadata has an invalid checkpoint_type")
        if metadata.get("vggto_attention_topology") != MOT_VGGTO_TOPOLOGY:
            raise ValueError(
                "MOT checkpoint does not use the synchronized multi-view VGGTO topology"
            )
        return metadata

    @classmethod
    def _validate_dcp_checkpoint_layout(cls, checkpoint_path: Path) -> None:
        cls._validate_transformer_checkpoint_layout(checkpoint_path)
        required_files = (
            checkpoint_path / "training_state.pt",
            checkpoint_path / MOT_DCP_DIR_NAME / ".metadata",
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Full-state resume requires a completed DCP checkpoint; missing: "
                + ", ".join(missing)
            )

    def _validate_trainer_state(self, training_state: dict[str, Any]) -> None:
        if int(training_state.get("format_version", -1)) != MOT_DCP_FORMAT_VERSION:
            raise ValueError(
                "Unsupported MOT DCP checkpoint format_version: "
                f"{training_state.get('format_version')!r}"
            )
        required_state = {"step", "lr_scheduler_state_dict", "rng_states_by_rank"}
        missing_state = required_state.difference(training_state)
        if missing_state:
            raise ValueError(
                "MOT DCP trainer state is missing required fields: "
                + ", ".join(sorted(missing_state))
            )

        metadata = training_state.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("checkpoint_type") != "dcp_sharded":
            raise ValueError("MOT DCP trainer state has invalid checkpoint metadata")
        if metadata.get("vggto_attention_topology") != MOT_VGGTO_TOPOLOGY:
            raise ValueError(
                "MOT DCP checkpoint does not use the synchronized multi-view VGGTO topology: "
                f"checkpoint={metadata.get('vggto_attention_topology')!r}, "
                f"current={MOT_VGGTO_TOPOLOGY!r}"
            )
        saved_torch_version = metadata.get("torch_version")
        if saved_torch_version != str(torch.__version__) and self.config.rank == 0:
            logger.warning(
                "Loading MOT DCP checkpoint across PyTorch versions: "
                f"checkpoint={saved_torch_version}, current={torch.__version__}. "
                "Compatibility will be validated by the DCP load."
            )

        compatible_fields = {
            "world_size": int(getattr(self.config, "world_size", 1)),
            "optimization_composition": self.optimization_composition,
            "max_views_per_gpu": int(self.config.max_views_per_gpu),
            "masked_attn_backend": str(getattr(self.config, "masked_attn_backend", "fa4")),
            "sampler_seed": int(getattr(self.config, "sampler_seed", 42)),
        }
        mismatches = {
            key: (metadata.get(key), current)
            for key, current in compatible_fields.items()
            if metadata.get(key) != current
        }
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={saved}, current={current}"
                for key, (saved, current) in mismatches.items()
            )
            raise ValueError(f"Incompatible MOT full-state resume configuration; {details}")

        rng_states = training_state["rng_states_by_rank"]
        expected_world_size = int(getattr(self.config, "world_size", 1))
        if not isinstance(rng_states, list) or len(rng_states) != expected_world_size:
            raise ValueError(
                "MOT DCP trainer state must contain one RNG state per rank: "
                f"checkpoint={len(rng_states) if isinstance(rng_states, list) else None}, "
                f"current={expected_world_size}"
            )

    def _collect_checkpoint_error(self, local_error: str | None) -> str | None:
        if not dist.is_initialized():
            return local_error
        if local_error is not None:
            logger.error(
                f"MOT checkpoint local failure on rank {int(self.config.rank)}: {local_error}"
            )
        errors = [None] * dist.get_world_size()
        group = getattr(self, "checkpoint_process_group", None)
        if group is None:
            dist.all_gather_object(errors, local_error)
        else:
            dist.all_gather_object(errors, local_error, group=group)
        return next((error for error in errors if error is not None), None)

    def _create_checkpoint_temp_dir(self, checkpoint_dir: Path) -> Path:
        payload = [None, None]
        if self.config.rank == 0:
            try:
                if checkpoint_dir.exists():
                    raise FileExistsError(f"Refusing to overwrite existing checkpoint: {checkpoint_dir}")
                temp_dir = self.save_dir / f".{checkpoint_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
                temp_dir.mkdir(parents=False, exist_ok=False)
                payload[0] = str(temp_dir)
            except Exception as exc:
                payload[1] = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            group = getattr(self, "checkpoint_process_group", None)
            if group is None:
                dist.broadcast_object_list(payload, src=0)
            else:
                dist.broadcast_object_list(payload, src=0, group=group)
        if payload[1] is not None:
            raise RuntimeError(payload[1])
        if payload[0] is None:
            raise RuntimeError("Rank 0 did not provide a checkpoint temporary directory")
        return Path(payload[0])

    def _write_transformer_config(self, checkpoint_dir: Path) -> Path:
        transformer_dir = checkpoint_dir / "transformer"
        transformer_dir.mkdir(parents=True, exist_ok=False)
        config_dict = dict(self.transformer.config)
        config_dict.pop("_name_or_path", None)
        config_dict["_class_name"] = "ThreeDVAMOTTransformer3DModel"
        with (transformer_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        return transformer_dir

    def _write_transformer_export(self, checkpoint_dir: Path, state_dict: dict[str, torch.Tensor]) -> None:
        transformer_dir = self._write_transformer_config(checkpoint_dir)
        save_file(state_dict, transformer_dir / MOT_TRANSFORMER_WEIGHTS_NAME)

    def _write_checkpoint_metadata(self, checkpoint_dir: Path, *, has_full_state: bool) -> None:
        metadata = {
            "format_version": MOT_DCP_FORMAT_VERSION,
            "checkpoint_type": "mot_training",
            "vggto_attention_topology": MOT_VGGTO_TOPOLOGY,
            "optimization_composition": self.optimization_composition,
            "has_full_state": bool(has_full_state),
        }
        with (checkpoint_dir / MOT_CHECKPOINT_METADATA_NAME).open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    def _load_training_state(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        self._validate_dcp_checkpoint_layout(checkpoint_path)
        training_state_path = checkpoint_path / "training_state.pt"

        payload = [None, None]
        if self.config.rank == 0:
            try:
                logger.info(f"Loading MOT DCP trainer state from {training_state_path}")
                payload[0] = torch.load(training_state_path, map_location="cpu", weights_only=True)
            except Exception as exc:
                payload[1] = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            group = getattr(self, "checkpoint_process_group", None)
            if group is None:
                dist.broadcast_object_list(payload, src=0)
            else:
                dist.broadcast_object_list(payload, src=0, group=group)
        if payload[1] is not None:
            raise RuntimeError(f"Failed to load MOT DCP trainer state: {payload[1]}")
        training_state = payload[0]
        if not isinstance(training_state, dict):
            raise ValueError("MOT DCP trainer state must be a dictionary")
        self._validate_trainer_state(training_state)

        dcp_state = self._dcp_state_dict()
        dcp.load(
            dcp_state,
            checkpoint_id=checkpoint_path / MOT_DCP_DIR_NAME,
            process_group=getattr(self, "checkpoint_process_group", None),
        )
        set_state_dict(
            self.transformer,
            self.optimizer,
            model_state_dict=dcp_state["model"],
            optim_state_dict=dcp_state["optimizer"],
            options=self._dcp_options(),
        )
        _configure_adamw_foreach(self.optimizer)
        del dcp_state

        self.lr_scheduler.load_state_dict(training_state["lr_scheduler_state_dict"])
        self.step = int(training_state["step"])
        self.optimizer_step = int(training_state.get("optimizer_step", self.step))
        self.skipped_step_total = int(training_state.get("skipped_step_total", 0))
        self.nonfinite_loss_total = int(training_state.get("nonfinite_loss_total", 0))
        self.nonfinite_grad_total = int(training_state.get("nonfinite_grad_total", 0))
        self.dataset_skip_total = int(training_state.get("dataset_skip_total", 0))
        self._set_sampler_resume_offset()
        rank = int(getattr(self.config, "rank", 0))
        self._restore_rng_state(training_state["rng_states_by_rank"][rank])
        self._last_checkpoint_step = self.step

        if dist.is_initialized():
            group = getattr(self, "checkpoint_process_group", None)
            if group is None:
                dist.barrier()
            else:
                dist.barrier(group=group)
        if self.config.rank == 0:
            logger.info(f"MOT DCP full training state loaded; resuming from step {self.step}")

    def save_checkpoint(self) -> Path:
        checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
        save_full_state = bool(getattr(self.config, "save_full_state", False))
        if save_full_state and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        temp_dir = self._create_checkpoint_temp_dir(checkpoint_dir)

        local_exception = None
        local_error = None
        try:
            if save_full_state:
                if self.config.rank == 0:
                    logger.info(f"Saving sharded MOT DCP state at step {self.step}")
                dcp_state = self._dcp_state_dict()
                dcp.save(
                    dcp_state,
                    checkpoint_id=temp_dir / MOT_DCP_DIR_NAME,
                    process_group=getattr(self, "checkpoint_process_group", None),
                )
                del dcp_state
                state_dict = get_model_state_dict(
                    self.transformer,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
                rng_states_by_rank = self._collect_rng_states()
                if self.config.rank == 0:
                    logger.info(f"Saving complete MOT transformer to {checkpoint_dir / 'transformer'}")
                    self._write_transformer_export(temp_dir, state_dict)
                    torch.save(
                        self._trainer_state(rng_states_by_rank),
                        temp_dir / "training_state.pt",
                    )
                    self._write_checkpoint_metadata(temp_dir, has_full_state=True)
                del state_dict
            else:
                state_dict = get_model_state_dict(
                    self.transformer,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
                if self.config.rank == 0:
                    logger.info(f"Saving MOT transformer export to {checkpoint_dir / 'transformer'}")
                    self._write_transformer_export(temp_dir, state_dict)
                    self._write_checkpoint_metadata(temp_dir, has_full_state=False)
                del state_dict
        except (Exception, dcp.CheckpointException) as exc:
            local_exception = exc
            local_error = f"{type(exc).__name__}: {exc}"

        checkpoint_error = self._collect_checkpoint_error(local_error)
        if checkpoint_error is not None:
            if self.config.rank == 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
            error = RuntimeError(f"Failed to save MOT checkpoint at step {self.step}: {checkpoint_error}")
            if local_exception is not None:
                raise error from local_exception
            raise error

        publish_exception = None
        publish_error = None
        if self.config.rank == 0:
            try:
                (temp_dir / "_SUCCESS").write_text("", encoding="utf-8")
                temp_dir.replace(checkpoint_dir)
            except Exception as exc:
                publish_exception = exc
                publish_error = f"{type(exc).__name__}: {exc}"
        publish_error = self._collect_checkpoint_error(publish_error)
        if publish_error is not None:
            if self.config.rank == 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
            error = RuntimeError(f"Failed to publish MOT checkpoint at step {self.step}: {publish_error}")
            if publish_exception is not None:
                raise error from publish_exception
            raise error

        retention_exception = None
        retention_error = None
        if self.config.rank == 0:
            try:
                removed = prune_successful_checkpoints(
                    self.save_dir,
                    max_checkpoints=self.max_checkpoints,
                )
                if removed:
                    logger.info("Removed old MOT checkpoints: " + ", ".join(path.name for path in removed))
            except Exception as exc:
                retention_exception = exc
                retention_error = f"{type(exc).__name__}: {exc}"
        retention_error = self._collect_checkpoint_error(retention_error)
        if retention_error is not None:
            error = RuntimeError(
                f"MOT checkpoint step {self.step} was saved, but retention failed: {retention_error}"
            )
            if retention_exception is not None:
                raise error from retention_exception
            raise error

        self._last_checkpoint_step = int(self.step)
        if self.config.rank == 0:
            logger.info(f"MOT checkpoint saved successfully at step {self.step}")
        return checkpoint_dir

    def _enqueue_checkpoint_evaluation(self, checkpoint_dir: Path) -> None:
        if self.config.rank != 0 or not bool(getattr(self.config, "eval_with_cpu", False)):
            return
        try:
            enqueued = self.checkpoint_evaluation_queue.enqueue(checkpoint_dir)
        except Exception as exc:
            logger.warning(
                f"Failed to enqueue CPU evaluation for {checkpoint_dir.name}: "
                f"{type(exc).__name__}: {exc}"
            )
            return
        if enqueued:
            logger.info(f"Queued external CPU evaluation for {checkpoint_dir.name}")
        else:
            logger.info(f"CPU evaluation already queued or completed for {checkpoint_dir.name}")

    def train(self):
        logger.info(f"Starting MOT training for {self.config.num_steps} steps...")
        self.transformer.train()
        self._write_memory_snapshot(phase="train_start", include_smaps=True)

        progress_timing_reset_step = self.step + 20
        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="MOT Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
        )

        self.optimizer.zero_grad()
        log_records = []
        timing_records = []
        performance_timing_records = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            should_report = self._should_report_current_step()
            should_write_performance = self._should_write_performance_jsonl()
            measure_timing = should_report or should_write_performance
            iteration_start = time.perf_counter() if should_write_performance else None
            data_fetch_start = time.perf_counter() if measure_timing else None
            batch = self._get_next_batch()
            self._consume_runtime_cache_stats(batch)
            if should_write_performance:
                has_pointcloud = batch.get("has_pointcloud")
                skip_count = batch.get("dataset_skip_count")
                pointcloud_samples = (
                    int(torch.as_tensor(has_pointcloud).sum().item())
                    if has_pointcloud is not None
                    else 0
                )
                dataset_skip_count = (
                    int(torch.as_tensor(skip_count).sum().item())
                    if skip_count is not None
                    else 0
                )
            else:
                pointcloud_samples = 0
                dataset_skip_count = 0
            data_fetch_time = (
                time.perf_counter() - data_fetch_start
                if data_fetch_start is not None
                else 0.0
            )

            # All ranks must finish data loading before any rank enters FSDP forward.
            data_barrier_start = time.perf_counter() if measure_timing else None
            if dist.is_initialized():
                dist.barrier()
            data_barrier_time = (
                time.perf_counter() - data_barrier_start
                if data_barrier_start is not None
                else 0.0
            )
            data_load_time = data_fetch_time + data_barrier_time

            losses = self._train_step(
                batch,
                step_in_accumulation,
                collect_detailed_metrics=should_report,
                measure_performance=measure_timing,
            )
            metrics_logging_start = time.perf_counter() if should_write_performance else None
            log_records.append(losses)
            if should_report:
                timing_records.append(
                    {
                        "data_load": data_load_time,
                        "local_samples": int(losses["data_local_samples"].item()),
                        "native_views": int(losses["data_native_views"].item()),
                        "phases": losses.get("_phase_timings", {}),
                    }
                )

            if bool(losses.get("optimizer_step_event", False)):
                self.optimizer_step += 1
            if bool(losses.get("nonfinite_loss_event", False)):
                self.nonfinite_loss_total += 1
            if bool(losses.get("nonfinite_grad_event", False)):
                self.nonfinite_grad_total += 1

            skipped_step = bool(losses.get("skipped_step", False))
            if skipped_step:
                self.skipped_step_total += 1
                step_in_accumulation = 0
            else:
                step_in_accumulation += 1

            checkpoint_due = False
            if losses["should_log"]:
                if should_report:
                    summary = self._aggregate_log_records(log_records)
                    performance = self._performance_metrics(timing_records)
                    self.dataset_skip_total += int(round(summary["data_dataset_skip_count"]))

                    def ratio(numerator: str, denominator: str) -> float:
                        den = summary[denominator]
                        return summary[numerator] / den if den > 0 else float("nan")

                    data_metrics = {
                        "data/pointcloud_sample_ratio": (
                            summary["data_pointcloud_samples"]
                            / (summary["data_pointcloud_samples"] + summary["data_pure_samples"])
                            if summary["data_pointcloud_samples"] + summary["data_pure_samples"] > 0
                            else float("nan")
                        ),
                        "data/dataset_skip_total": self.dataset_skip_total,
                        "data/source_pointcloud_count": summary["data_pointcloud_samples"],
                        "data/source_pure_count": summary["data_pure_samples"],
                        "data/max_views_per_gpu": int(getattr(self.config, "max_views_per_gpu", 0)),
                        "data/native_views": int(losses["data_native_views"].item()),
                        "data/local_batch_size": int(losses["data_local_samples"].item()),
                        "data/global_sample_count": int(losses["data_local_samples"].item())
                        * int(getattr(self.config, "world_size", 1)),
                    }
                    if "v" in self.optimization_branches:
                        data_metrics.update(
                            {
                                "data/video_supervised_frame_ratio": ratio(
                                    "data_video_supervised_num", "data_video_supervised_den"
                                ),
                                "data/video_valid_frame_ratio": ratio(
                                    "data_video_valid_num", "data_video_valid_den"
                                ),
                            }
                        )
                    if "a" in self.optimization_branches:
                        data_metrics.update(
                            {
                                "data/action_supervised_token_ratio": ratio(
                                    "data_action_supervised_num", "data_action_supervised_den"
                                ),
                                "data/action_valid_token_ratio": ratio(
                                    "data_action_valid_num", "data_action_valid_den"
                                ),
                            }
                        )
                    geometry_metrics = {}
                    if "g" in self.optimization_branches:
                        data_metrics.update(
                            {
                                "data/geometry_valid_slot_ratio": ratio(
                                    "data_geometry_valid_slot_num", "data_geometry_valid_slot_den"
                                ),
                                "data/geometry_valid_pixel_ratio": ratio(
                                    "data_geometry_valid_pixel_num", "data_geometry_valid_pixel_den"
                                ),
                                "data/geometry_active_ranks": float(
                                    losses["geometry_global_active_ranks"].item()
                                ),
                            }
                        )
                        correlations = [
                            summary[key]
                            for key in (
                                "depth_conf_error_correlation",
                                "point_conf_error_correlation",
                            )
                            if key in summary and np.isfinite(summary[key])
                        ]
                        geometry_metrics = {
                            "geometry/depth_error_p50": summary.get("depth_error_p50", float("nan")),
                            "geometry/depth_error_p90": summary.get("depth_error_p90", float("nan")),
                            "geometry/point_error_p50": summary.get("point_error_p50", float("nan")),
                            "geometry/point_error_p90": summary.get("point_error_p90", float("nan")),
                            "geometry/normal_angle_mean": summary.get("point_normal_angle_mean", float("nan")),
                            "geometry/depth_conf_error_correlation": summary.get(
                                "depth_conf_error_correlation", float("nan")
                            ),
                            "geometry/point_conf_error_correlation": summary.get(
                                "point_conf_error_correlation", float("nan")
                            ),
                            "geometry/conf_error_correlation": (
                                float(sum(correlations) / len(correlations))
                                if correlations
                                else float("nan")
                            ),
                        }
                    grad_clip_count = summary["grad_clip_count"]
                    learning_rates = self.lr_scheduler.get_last_lr()
                    optim_metrics = {
                        "optim/optimizer_step": self.optimizer_step,
                        "optim/skipped_step_total": self.skipped_step_total,
                        "optim/nonfinite_loss_total": self.nonfinite_loss_total,
                        "optim/nonfinite_grad_total": self.nonfinite_grad_total,
                        "optim/grad_norm_pre_clip": summary["total_norm"],
                        "optim/grad_clip_fraction": (
                            summary["grad_clip_event"] / grad_clip_count
                            if grad_clip_count > 0
                            else float("nan")
                        ),
                        "optim/lr_main": learning_rates[0],
                        "optim/lr_vggto": learning_rates[1] if len(learning_rates) > 1 else learning_rates[0],
                    }
                    train_metrics = {
                        "train/total_loss": summary["total_loss_raw"],
                        "train/optimization_composition": self.optimization_composition,
                        "train/execution_route": self.execution_route,
                        "train/data_profile": self.execution_route,
                    }
                    if "v" in self.optimization_branches:
                        train_metrics["train/weighted_video_loss"] = summary["weighted_video_loss_raw"]
                    if "a" in self.optimization_branches:
                        train_metrics["train/weighted_action_loss"] = summary["weighted_action_loss_raw"]
                    if "g" in self.optimization_branches:
                        train_metrics.update(
                            {
                                "train/weighted_depth_loss": summary["weighted_depth_loss_raw"],
                                "train/weighted_point_loss": summary["weighted_point_loss_raw"],
                            }
                        )
                    log_records = []
                    timing_records = []
                step_in_accumulation = 0

                if self.config.rank == 0:
                    total_norm = losses["total_norm"]
                    grad_summary = losses.get("grad_summary", {})
                    progress_bar.n += 1
                    if progress_bar.n == progress_timing_reset_step:
                        progress_bar.initial = progress_bar.n
                        progress_bar.last_print_n = progress_bar.n
                        progress_bar.start_t = progress_bar.last_print_t = progress_bar._time()
                    if should_report:
                        postfix = {
                            "V": int(losses["data_native_views"].item()),
                            "B": int(losses["data_local_samples"].item()),
                            "step": self.step,
                            "grad_norm": f"{total_norm.item():.2f}",
                            "skipped": int(skipped_step),
                            "lr": f"{learning_rates[0]:.2e}",
                        }
                        if "v" in self.optimization_branches:
                            postfix.update(
                                latent_loss=f"{summary['latent_loss_raw']:.4f}",
                                video_w=f"{losses['video_loss_weight'].item():.3f}",
                            )
                        if "a" in self.optimization_branches:
                            postfix.update(
                                action_loss=f"{summary['action_loss_raw']:.4f}",
                                action_w=f"{losses['action_loss_weight'].item():.3f}",
                            )
                        if "g" in self.optimization_branches:
                            postfix.update(
                                depth_loss=f"{summary['depth_loss_raw']:.4f}",
                                point_loss=f"{summary['point_loss_raw']:.4f}",
                                geo_w=f"{losses['geometry_loss_weight'].item():.3f}",
                                depth_w=f"{losses['depth_loss_weight'].item():.3f}",
                                point_w=f"{losses['point_loss_weight'].item():.3f}",
                            )
                        progress_bar.set_postfix(postfix)
                    if grad_summary:
                        logger.info("Gradient flow summary: " + json.dumps(grad_summary, sort_keys=True))
                    if should_report and self.config.enable_wandb:
                        grad_metrics = {}
                        for group_name, values in grad_summary.items():
                            grad_metrics[f"grad_flow/{group_name}_norm"] = values["grad_norm"]
                            grad_metrics[f"grad_flow/{group_name}_params_with_grad"] = values["params_with_grad"]
                            grad_metrics[f"grad_flow/{group_name}_all_finite"] = float(values["all_finite"])
                            if "update_norm" in values:
                                grad_metrics[f"grad_flow/{group_name}_update_norm"] = values["update_norm"]
                                grad_metrics[f"grad_flow/{group_name}_relative_update_norm"] = values[
                                    "relative_update_norm"
                                ]
                                grad_metrics[f"grad_flow/{group_name}_update_all_finite"] = float(
                                    values["update_all_finite"]
                                )
                        loss_metrics = {}
                        if "v" in self.optimization_branches:
                            loss_metrics.update(
                                {
                                    "loss_metrics/global_avg_video_loss": summary["latent_loss_raw"],
                                    "loss_metrics/global_max_video_loss": summary["max_latent_loss_raw"],
                                    "loss_metrics/video_loss_weight": float(losses["video_loss_weight"].item()),
                                }
                            )
                        if "a" in self.optimization_branches:
                            loss_metrics.update(
                                {
                                    "loss_metrics/global_avg_action_loss": summary["action_loss_raw"],
                                    "loss_metrics/global_max_action_loss": summary["max_action_loss_raw"],
                                    "loss_metrics/action_loss_weight": float(losses["action_loss_weight"].item()),
                                }
                            )
                        if "g" in self.optimization_branches:
                            loss_metrics.update(
                                {
                                    "loss_metrics/global_avg_depth_loss": summary["depth_loss_raw"],
                                    "loss_metrics/global_avg_point_loss": summary["point_loss_raw"],
                                    "loss_metrics/global_max_depth_loss": summary["max_depth_loss_raw"],
                                    "loss_metrics/global_max_point_loss": summary["max_point_loss_raw"],
                                    "loss_metrics/geometry_loss_weight": float(
                                        losses["geometry_loss_weight"].item()
                                    ),
                                    "loss_metrics/geometry_active_rank_scale": float(
                                        losses["geometry_active_rank_scale"].item()
                                    ),
                                    "loss_metrics/depth_loss_weight": float(losses["depth_loss_weight"].item()),
                                    "loss_metrics/point_loss_weight": float(losses["point_loss_weight"].item()),
                                    **{
                                        f"loss_metrics/{key}": summary[key]
                                        for key in (
                                            "loss_conf_depth",
                                            "loss_reg_depth",
                                            "loss_grad_depth",
                                            "depth_conf_mean",
                                            "depth_conf_max",
                                            "loss_conf_point",
                                            "loss_reg_point",
                                            "loss_grad_point",
                                            "point_conf_mean",
                                            "point_conf_max",
                                        )
                                        if key in summary
                                    },
                                }
                            )
                        self.wandb.log(
                            {
                                **loss_metrics,
                                "grad_norm": total_norm.item(),
                                "lr": learning_rates[0],
                                **train_metrics,
                                **optim_metrics,
                                **geometry_metrics,
                                **performance,
                                **data_metrics,
                                **grad_metrics,
                            },
                            step=self.step,
                        )

                self.step += 1
                checkpoint_due = self.step % self.config.save_interval == 0

            metrics_logging_time = (
                time.perf_counter() - metrics_logging_start
                if metrics_logging_start is not None
                else 0.0
            )
            checkpoint_time = 0.0
            step_completed = bool(losses["should_log"])
            if step_completed and checkpoint_due:
                self._write_memory_snapshot(phase="before_checkpoint", include_smaps=True)
            elif step_completed and self._memory_snapshot_due(self.step):
                self._write_memory_snapshot(
                    phase="periodic",
                    include_smaps=self._memory_snapshot_needs_smaps(self.step),
                )
            if checkpoint_due:
                if self.config.rank == 0:
                    logger.info(f"Starting save MOT model at step {self.step}")
                checkpoint_start = time.perf_counter() if should_write_performance else None
                checkpoint_dir = self.save_checkpoint()
                self._write_memory_snapshot(phase="after_checkpoint", include_smaps=True)
                if checkpoint_start is not None:
                    checkpoint_time = time.perf_counter() - checkpoint_start
                self._enqueue_checkpoint_evaluation(checkpoint_dir)

            (
                post_step_barrier_time,
                gc_collect_time,
                post_gc_barrier_time,
            ) = self._synchronize_and_collect_garbage(
                step_completed=step_completed,
            )

            if should_write_performance:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                performance_timing_records.append(
                    {
                        "data_fetch": data_fetch_time,
                        "data_barrier": data_barrier_time,
                        "pointcloud_samples": pointcloud_samples,
                        "dataset_skip_count": dataset_skip_count,
                        "local_samples": int(losses["data_local_samples"].item()),
                        "native_views": int(losses["data_native_views"].item()),
                        "phases": losses.get("_phase_timings", {}),
                        "metrics_logging": metrics_logging_time,
                        "checkpoint": checkpoint_time,
                        "post_step_barrier": post_step_barrier_time,
                        "gc_collect": gc_collect_time,
                        "post_gc_barrier": post_gc_barrier_time,
                        "iteration": time.perf_counter() - iteration_start,
                    }
                )
                if losses["should_log"]:
                    self._write_performance_jsonl(
                        performance_timing_records,
                        training_step=self.step,
                        skipped_step=skipped_step,
                    )
                    performance_timing_records = []

        if getattr(self, "_last_checkpoint_step", None) != int(self.step):
            if self.config.rank == 0:
                logger.info(f"Saving final MOT model at step {self.step}")
            self._write_memory_snapshot(phase="before_final_checkpoint", include_smaps=True)
            checkpoint_dir = self.save_checkpoint()
            self._write_memory_snapshot(phase="after_final_checkpoint", include_smaps=True)
            self._enqueue_checkpoint_evaluation(checkpoint_dir)

        progress_bar.close()
        self._restore_training_gc_control()
        logger.info("MOT training completed!")


def run(args):
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    train_seed = int(getattr(config, "train_seed", 42))
    _seed_mot_training(train_seed)

    if rank == 0:
        logger.info(f"Using MOT config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"MOT model-initialization seed: {train_seed}")

    trainer = MOTTrainer(config)
    runtime_seed = train_seed + rank
    _seed_mot_training(runtime_seed)
    if rank == 0:
        logger.info(f"MOT runtime seed: base={train_seed}, rank0={runtime_seed}")
    try:
        trainer.train()
    finally:
        restore_gc = getattr(trainer, "_restore_training_gc_control", None)
        if restore_gc is not None:
            restore_gc()


def main():
    parser = argparse.ArgumentParser(description="Train 3DVA_MOT for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default="umi_3dwam_train",
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
