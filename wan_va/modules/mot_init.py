"""
Initialization and checkpoint-loading helpers for 3DVA_MOT.

This path initializes LingBot from the original transformer checkpoint and
initializes the geometry path from a VGGT-Omega-derived VGGTO checkpoint. Camera and text
alignment heads are intentionally ignored; the VGGTO geometry path loads the
patch embed, frame/inter-frame blocks, shared registers, and dense depth head.
The relative local-point head is loaded separately from an original VGGT
checkpoint because the VGGT-Omega checkpoint does not contain one.
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file


@dataclass(frozen=True)
class VGGTO24To30LayerPlan:
    """Explicit, auditable topology for warm-starting VGGTO-30 from VGGTO-24."""

    source_depth: int
    target_depth: int
    target_to_source: tuple[Optional[int], ...]
    frame_source_by_target: tuple[int, ...]
    mot_targets: tuple[int, ...]
    mot_source_by_target: tuple[int, ...]
    native_targets: tuple[int, ...]
    inserted_targets: tuple[int, ...]
    converted_mot_targets: tuple[int, ...]
    original_register_targets: tuple[int, ...]
    source_register_layers: tuple[int, ...]
    source_cache_layers: tuple[int, ...]
    target_cache_layers: tuple[int, ...]

    def source_for_frame(self, target_layer: int) -> int:
        return self.frame_source_by_target[target_layer]

    def source_for_mot(self, target_layer: int) -> int:
        try:
            index = self.mot_targets.index(target_layer)
        except ValueError as exc:
            raise ValueError(f"target layer {target_layer} is not a MOT layer") from exc
        return self.mot_source_by_target[index]

    def source_for_native(self, target_layer: int) -> int:
        if target_layer not in self.native_targets:
            raise ValueError(f"target layer {target_layer} is not a native inter-frame layer")
        source_layer = self.target_to_source[target_layer]
        if source_layer is None:
            raise AssertionError(f"native target layer {target_layer} has no source layer")
        return source_layer

    @property
    def cache_source_by_target(self) -> Dict[int, int]:
        return dict(zip(self.target_cache_layers, self.source_cache_layers))

    @property
    def covered_source_inter_frame_layers(self) -> tuple[int, ...]:
        sources = {
            *(self.source_for_native(layer) for layer in self.native_targets),
            *self.mot_source_by_target,
        }
        return tuple(sorted(sources))


VGGTO_24_TO_30_TARGET_TO_SOURCE = (
    0,
    1,
    2,
    3,
    None,
    4,
    None,
    5,
    6,
    7,
    None,
    8,
    9,
    10,
    None,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    None,
    18,
    None,
    19,
    20,
    21,
    22,
    23,
)
VGGTO_30_MOT_TARGETS = tuple(range(0, 30, 2))
VGGTO_30_NATIVE_TARGETS = tuple(range(1, 30, 2))
VGGTO_30_INSERTED_TARGETS = (4, 6, 10, 14, 22, 24)
VGGTO_30_CONVERTED_MOT_TARGETS = (0, 16, 20, 28)
VGGTO_30_ORIGINAL_REGISTER_TARGETS = (2, 8, 12, 18, 26)

# An inserted frame block copies the lower-side nearest mapped frame block. Its
# two residual LayerScales are then zeroed, so the inserted block starts as an
# exact identity without modifying the checkpoint tensor it copied.
VGGTO_30_FRAME_SOURCE_BY_TARGET = (
    0,
    1,
    2,
    3,
    3,
    4,
    4,
    5,
    6,
    7,
    7,
    8,
    9,
    10,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    17,
    18,
    18,
    19,
    20,
    21,
    22,
    23,
)

# Existing and converted MOT blocks use their mapped source. Inserted MOT
# blocks reuse the nearest original register block discussed in the 30/15 plan.
VGGTO_30_MOT_SOURCE_BY_TARGET = (
    0,
    2,
    2,
    6,
    6,
    9,
    9,
    9,
    12,
    14,
    16,
    20,
    20,
    20,
    22,
)

VGGTO_24_TO_30_PLAN = VGGTO24To30LayerPlan(
    source_depth=24,
    target_depth=30,
    target_to_source=VGGTO_24_TO_30_TARGET_TO_SOURCE,
    frame_source_by_target=VGGTO_30_FRAME_SOURCE_BY_TARGET,
    mot_targets=VGGTO_30_MOT_TARGETS,
    mot_source_by_target=VGGTO_30_MOT_SOURCE_BY_TARGET,
    native_targets=VGGTO_30_NATIVE_TARGETS,
    inserted_targets=VGGTO_30_INSERTED_TARGETS,
    converted_mot_targets=VGGTO_30_CONVERTED_MOT_TARGETS,
    original_register_targets=VGGTO_30_ORIGINAL_REGISTER_TARGETS,
    source_register_layers=(2, 6, 9, 14, 20),
    source_cache_layers=(4, 11, 17, 23),
    target_cache_layers=(5, 15, 21, 29),
)


@dataclass
class VGGTO24To30LoadReport:
    plan: VGGTO24To30LayerPlan = VGGTO_24_TO_30_PLAN
    loaded_patch_embed_keys: int = 0
    loaded_frame_block_keys: int = 0
    loaded_cross_view_keys: int = 0
    loaded_native_inter_frame_keys: int = 0
    loaded_dense_head_keys: int = 0
    deferred_mot_inter_frame_keys: int = 0
    cross_view_layer_scale_factor: float = 1.0
    zeroed_frame_layer_scales: Dict[int, List[str]] = field(default_factory=dict)

    @property
    def loaded_model_keys(self) -> int:
        return (
            self.loaded_patch_embed_keys
            + self.loaded_frame_block_keys
            + self.loaded_cross_view_keys
            + self.loaded_native_inter_frame_keys
            + self.loaded_dense_head_keys
        )


@dataclass
class InitReport:
    video_source: str
    video_path: str
    vggto_path: str
    vggt_point_path: str
    loaded_video_keys: int = 0
    loaded_vggto_required_keys: int = 0
    loaded_vggt_point_keys: int = 0
    ignored_vggto_keys: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    vggto_24_to_30: Optional[VGGTO24To30LoadReport] = None
    action_init: Optional["ActionInitReport"] = None


@dataclass
class ActionInitReport:
    seed: int
    copied: List[str] = field(default_factory=list)
    interpolated: List[str] = field(default_factory=list)
    scaled: Dict[str, float] = field(default_factory=dict)
    shared: List[str] = field(default_factory=list)
    random: List[str] = field(default_factory=list)
    random_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def initialized_parameter_count(self) -> int:
        return len(self.copied) + len(self.interpolated) + len(self.scaled) + len(self.random)


def check_diffusers_transformer_complete(path: str, *, source_name: str) -> None:
    """Fail fast if a Diffusers transformer has missing shard files.

    The user explicitly wanted real-model tests to prove the final checkpoint is
    complete.  We therefore do not skip incomplete downloads.
    """

    transformer = Path(path)
    if not transformer.is_dir():
        raise FileNotFoundError(f"{source_name} transformer path does not exist: {path}")
    index_path = transformer / "diffusion_pytorch_model.safetensors.index.json"
    if not (transformer / "config.json").is_file():
        raise FileNotFoundError(f"Missing {source_name} transformer config.json in {path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {source_name} safetensors index in {path}")
    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    missing = [name for name in shard_names if not (transformer / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{source_name} checkpoint is incomplete. Missing shards: "
            + ", ".join(missing[:8])
        )


ACTION_RANDOM_PARAMETER_NAMES = (
    "action_embedder.weight",
    "action_proj_out.weight",
    "action_proj_out.bias",
)


@torch.no_grad()
def initialize_wan_patch_embedding_(
    target_linear: torch.nn.Linear,
    source_conv: torch.nn.Conv3d,
) -> None:
    """Copy a Diffusers Wan Conv3d patch embed into the equivalent Linear."""

    if source_conv.bias is None:
        raise ValueError("Wan patch embedding must have a bias")
    target_linear.load_state_dict(
        {
            "weight": source_conv.weight.detach().flatten(1),
            "bias": source_conv.bias.detach(),
        },
        strict=True,
    )


def resize_tensor_sequential_1d(
    source: torch.Tensor,
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    """Resize every mismatched axis with FP32 1D linear interpolation."""

    if source.ndim != len(target_shape):
        raise ValueError(
            f"source rank {source.ndim} does not match target rank {len(target_shape)}"
        )
    if tuple(source.shape) == tuple(target_shape):
        return source.detach().to(device="cpu", dtype=torch.float32).clone()

    out = source.detach().to(device="cpu", dtype=torch.float32)
    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        permutation = [axis for axis in range(out.ndim) if axis != dim] + [dim]
        inverse = [0] * out.ndim
        for index, axis in enumerate(permutation):
            inverse[axis] = index
        permuted = out.permute(*permutation).contiguous()
        prefix = permuted.shape[:-1]
        permuted = F.interpolate(
            permuted.reshape(-1, 1, permuted.shape[-1]),
            size=int(new_size),
            mode="linear",
            align_corners=True,
        ).reshape(*prefix, int(new_size))
        out = permuted.permute(*inverse).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise AssertionError(
            f"interpolation produced {tuple(out.shape)}, expected {tuple(target_shape)}"
        )
    return out


@torch.no_grad()
def initialize_action_io_(
    action_encoder: torch.nn.Linear,
    action_head: torch.nn.Linear,
    *,
    seed: int,
) -> None:
    """Initialize only the task-specific action I/O with a CPU-local RNG."""

    if action_encoder.bias is not None:
        raise ValueError("action encoder must be bias-free")
    if action_encoder.weight.is_meta or action_head.weight.is_meta:
        return

    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def uniform_parameter_(parameter: torch.Tensor, bound: float) -> None:
        value = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
        value.uniform_(-bound, bound, generator=generator)
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    uniform_parameter_(action_encoder.weight, 1.0 / math.sqrt(action_encoder.in_features))
    uniform_parameter_(action_head.weight, 1.0 / math.sqrt(action_head.in_features))
    if action_head.bias is None:
        raise ValueError("action output head must have a bias")
    uniform_parameter_(action_head.bias, 1.0 / math.sqrt(action_head.in_features))


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    value = tensor.detach().to(device="cpu", dtype=torch.float32)
    return {
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def _parameter_module(module: torch.nn.Module, parameter_name: str) -> torch.nn.Module:
    module_path, _, _ = parameter_name.rpartition(".")
    return dict(module.named_modules())[module_path]


@torch.no_grad()
def initialize_action_expert_from_video_(
    target_model: torch.nn.Module,
    source_video_model: torch.nn.Module,
) -> ActionInitReport:
    """Strictly initialize the 768D action expert from the LingBot video path."""

    target_blocks = list(target_model.mot_blocks)
    source_blocks = list(source_video_model.blocks)
    if len(target_blocks) != 30 or len(source_blocks) != 30:
        raise ValueError(
            "action initialization requires exactly 30 paired layers: "
            f"target={len(target_blocks)}, source={len(source_blocks)}"
        )

    report = ActionInitReport(seed=int(target_model.init_noise_seed))
    classified: Dict[str, str] = {}

    def classify(name: str, category: str) -> None:
        previous = classified.get(name)
        if previous is not None:
            raise ValueError(f"action parameter {name} classified twice: {previous}, {category}")
        classified[name] = category

    def copy_parameter(
        *,
        target_name: str,
        target_parameter: torch.Tensor,
        source_parameter: torch.Tensor,
        is_linear_weight: bool,
    ) -> None:
        target_shape = tuple(target_parameter.shape)
        source_shape = tuple(source_parameter.shape)
        resized = resize_tensor_sequential_1d(source_parameter, target_shape)
        scale = None
        if is_linear_weight and source_shape[1] != target_shape[1]:
            scale = math.sqrt(float(source_shape[1]) / float(target_shape[1]))
            resized.mul_(scale)
        target_parameter.copy_(
            resized.to(device=target_parameter.device, dtype=target_parameter.dtype)
        )
        if scale is not None:
            classify(target_name, "scaled")
            report.scaled[target_name] = scale
        elif source_shape != target_shape:
            classify(target_name, "interpolated")
            report.interpolated.append(target_name)
        else:
            classify(target_name, "copied")
            report.copied.append(target_name)

    for layer_id, (target_composite, source_block) in enumerate(zip(target_blocks, source_blocks)):
        target_block = target_composite.action_block
        target_parameters = dict(target_block.named_parameters())
        source_parameters = dict(source_block.named_parameters())
        missing = sorted(set(target_parameters) - set(source_parameters))
        if missing:
            raise KeyError(
                f"video block {layer_id} has no source for action parameters: {missing}"
            )
        for local_name, target_parameter in target_parameters.items():
            source_parameter = source_parameters[local_name]
            target_module = _parameter_module(target_block, local_name)
            source_module = _parameter_module(source_block, local_name)
            leaf_name = local_name.rsplit(".", 1)[-1]
            is_linear_weight = (
                isinstance(target_module, torch.nn.Linear)
                and isinstance(source_module, torch.nn.Linear)
                and leaf_name == "weight"
            )
            copy_parameter(
                target_name=f"mot_blocks.{layer_id}.action_block.{local_name}",
                target_parameter=target_parameter,
                source_parameter=source_parameter,
                is_linear_weight=is_linear_weight,
            )

        forbidden_cross_parameters = {
            name
            for name, _ in target_block.attn2.named_parameters()
            if name.startswith(("to_k.", "to_v.", "norm_k."))
        }
        if forbidden_cross_parameters:
            raise ValueError(
                f"action block {layer_id} owns shared text K/V parameters: "
                f"{sorted(forbidden_cross_parameters)}"
            )
        shared_local_names = sorted(
            name
            for name, _ in target_composite.video_block.attn2.named_parameters()
            if name.startswith(("to_k.", "to_v.", "norm_k."))
        )
        expected_shared = {
            "to_k.weight",
            "to_k.bias",
            "to_v.weight",
            "to_v.bias",
            "norm_k.weight",
        }
        if set(shared_local_names) != expected_shared:
            raise ValueError(
                f"video block {layer_id} shared text parameter mismatch: "
                f"expected={sorted(expected_shared)}, got={shared_local_names}"
            )
        report.shared.extend(
            f"mot_blocks.{layer_id}.video_block.attn2.{name}"
            for name in shared_local_names
        )

    target_condition = target_model.action_condition_embedder
    source_condition = source_video_model.condition_embedder
    target_parameters = dict(target_condition.named_parameters())
    source_parameters = dict(source_condition.named_parameters())
    missing = sorted(set(target_parameters) - set(source_parameters))
    if missing:
        raise KeyError(f"video conditioning has no source for action parameters: {missing}")
    for local_name, target_parameter in target_parameters.items():
        target_module = _parameter_module(target_condition, local_name)
        source_module = _parameter_module(source_condition, local_name)
        copy_parameter(
            target_name=f"action_condition_embedder.{local_name}",
            target_parameter=target_parameter,
            source_parameter=source_parameters[local_name],
            is_linear_weight=(
                isinstance(target_module, torch.nn.Linear)
                and isinstance(source_module, torch.nn.Linear)
                and local_name.rsplit(".", 1)[-1] == "weight"
            ),
        )

    copy_parameter(
        target_name="action_scale_shift_table",
        target_parameter=target_model.action_scale_shift_table,
        source_parameter=source_video_model.scale_shift_table,
        is_linear_weight=False,
    )

    shared_text_context = sorted(
        f"condition_embedder.text_embedder.{name}"
        for name, _ in target_model.condition_embedder.text_embedder.named_parameters()
    )
    expected_text_context = {
        "condition_embedder.text_embedder.linear_1.weight",
        "condition_embedder.text_embedder.linear_1.bias",
        "condition_embedder.text_embedder.linear_2.weight",
        "condition_embedder.text_embedder.linear_2.bias",
    }
    if set(shared_text_context) != expected_text_context:
        raise ValueError(
            "shared text context parameter mismatch: "
            f"expected={sorted(expected_text_context)}, got={shared_text_context}"
        )
    report.shared.extend(shared_text_context)

    random_parameters = {
        name: parameter
        for name, parameter in target_model.named_parameters()
        if name in ACTION_RANDOM_PARAMETER_NAMES
    }
    if set(random_parameters) != set(ACTION_RANDOM_PARAMETER_NAMES):
        raise ValueError(
            "random action parameter whitelist mismatch: "
            f"expected={list(ACTION_RANDOM_PARAMETER_NAMES)}, got={sorted(random_parameters)}"
        )
    for name in ACTION_RANDOM_PARAMETER_NAMES:
        classify(name, "random")
        report.random.append(name)
        report.random_stats[name] = _tensor_stats(random_parameters[name])

    action_owned = {
        name
        for name, _ in target_model.named_parameters()
        if name.startswith("action_") or ".action_block." in name
    }
    uncovered = sorted(action_owned - set(classified))
    unexpected = sorted(set(classified) - action_owned)
    if uncovered or unexpected:
        raise ValueError(
            "action initialization coverage mismatch: "
            f"uncovered={uncovered}, unexpected={unexpected}"
        )

    nonfinite = [
        name
        for name, parameter in target_model.named_parameters()
        if name in action_owned and not bool(torch.isfinite(parameter).all())
    ]
    if nonfinite:
        raise ValueError(f"non-finite initialized action parameters: {nonfinite}")

    report.copied.sort()
    report.interpolated.sort()
    report.shared.sort()
    report.random.sort()
    return report


def _resolve_checkpoint_file(path: str, *, label: str) -> Path:
    ckpt = Path(path)
    if ckpt.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin", "model.pt", "model.pth"):
            candidate = ckpt / name
            if candidate.is_file():
                ckpt = candidate
                break
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing {label} checkpoint file: {ckpt}")
    return ckpt


def load_vggto_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ckpt = _resolve_checkpoint_file(path, label="VGGTO")
    if str(ckpt).endswith(".safetensors"):
        return load_file(str(ckpt), device="cpu")
    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("state_dict", "model"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    return raw


def _strip_model_prefixes(key: str) -> str:
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def load_vggt_point_head_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load only `point_head.*` tensors from an original VGGT checkpoint."""

    ckpt = _resolve_checkpoint_file(path, label="VGGT")
    point_state = {}
    if ckpt.suffix == ".safetensors":
        with safe_open(str(ckpt), framework="pt", device="cpu") as checkpoint:
            for raw_key in checkpoint.keys():
                key = _strip_model_prefixes(raw_key)
                if key.startswith("point_head."):
                    point_state[key[len("point_head.") :]] = checkpoint.get_tensor(raw_key)
    else:
        state = normalize_vggto_keys(load_vggto_state_dict(str(ckpt)))
        point_state = {
            key[len("point_head.") :]: value
            for key, value in state.items()
            if key.startswith("point_head.")
        }
    if not point_state:
        raise KeyError(f"VGGT checkpoint contains no point_head.* weights: {ckpt}")
    return point_state


def head_aware_hidden_map(target_dim: int = 3072, source_heads: int = 16, source_head_dim: int = 64, target_heads: int = 24, target_head_dim: int = 128) -> torch.Tensor:
    """
    按照 head 去初始化新增的 channel。
    VGGTO: 1024 = 16 heads * 64 dim
    MoT/LingBot carrier: 3072 = 24 heads * 128 dim
    二者建立一个 channel 的映射。
    """

    out = []
    for channel in range(target_dim):
        # 遍历每个 channel 计算当前 channel 属于哪个 head 中的哪一 dimension
        target_head = channel // target_head_dim
        target_d = channel % target_head_dim
        # 24 个 head 压回原来的 16 个 head
        source_head = math.floor(target_head * source_heads / target_heads)
        # 把每个 head 内部的 128 维映射到 VGGTO head 内部的 64 维
        source_d = math.floor(target_d * source_head_dim / target_head_dim)
        # 合成 source 里的真实 channel index，得到 out[new_channel] = source_index
        out.append(source_head * source_head_dim + source_d)
    return torch.tensor(out, dtype=torch.long)


def flat_repeat_map(target_dim: int, source_dim: int) -> torch.Tensor:
    return torch.floor(torch.arange(target_dim, dtype=torch.float64) * source_dim / target_dim).long()


def _counts(mapping: torch.Tensor, source_dim: int) -> torch.Tensor:
    counts = torch.bincount(mapping.cpu(), minlength=source_dim).float()
    counts[counts == 0] = 1.0
    return counts


def copy_rescale_linear_(
    target: torch.nn.Linear,
    source_weight: torch.Tensor,
    source_bias: Optional[torch.Tensor],
    out_map: torch.Tensor,
    in_map: torch.Tensor,
    *,
    noise_std: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> None:
    # Design note: copy-rescale follows Net2Net/bert2BERT-style widening.
    # Dividing by input-copy count preserves the source linear function when
    # multiple target input channels point back to the same source channel.
    counts = _counts(in_map, source_weight.shape[1]).to(source_weight.device)
    weight = source_weight[out_map][:, in_map] / counts[in_map].to(source_weight.device)[None, :]
    if noise_std > 0:
        noise = torch.randn(weight.shape, generator=generator, dtype=weight.dtype) * noise_std
        weight = weight + noise
    target.weight.data.copy_(weight.to(target.weight.dtype))
    if target.bias is not None and source_bias is not None:
        target.bias.data.copy_(source_bias[out_map].to(target.bias.dtype))


def copy_norm_(target: torch.nn.Module, source_weight: torch.Tensor, source_bias: Optional[torch.Tensor], mapping: torch.Tensor) -> None:
    if hasattr(target, "weight") and target.weight is not None:
        target.weight.data.copy_(source_weight[mapping].to(target.weight.dtype))
    if hasattr(target, "bias") and target.bias is not None and source_bias is not None:
        target.bias.data.copy_(source_bias[mapping].to(target.bias.dtype))


def init_up_down_(up: torch.nn.Linear, down: torch.nn.Linear, mapping: torch.Tensor) -> None:
    """Initialize exact head-aware Up/Down adapters.

    Up copies a VGGTO source channel into every mapped MoT channel.  Down averages
    the mapped MoT channels back to the VGGTO source channel.  We intentionally do
    not add tiny noise here so `Down(Up(x))` starts as an identity.
    """

    up.weight.data.zero_()
    up.bias.data.zero_()
    down.weight.data.zero_()
    down.bias.data.zero_()
    counts = _counts(mapping, down.out_features).to(down.weight.device)
    for target_channel, source_channel in enumerate(mapping.tolist()):
        # Design note: head-aware Up/Down preserves Down(Up(x)) == x at initialization.
        up.weight.data[target_channel, source_channel] = 1.0
        down.weight.data[source_channel, target_channel] = 1.0 / counts[source_channel]


def split_qkv(weight: torch.Tensor, bias: Optional[torch.Tensor], dim: int = 1024):
    weights = weight.reshape(3, dim, weight.shape[1])
    biases = bias.reshape(3, dim) if bias is not None else [None, None, None]
    return weights, biases


def init_geometry_from_vggto_inter_frame_(
    target_block,
    vggto_state: Mapping[str, torch.Tensor],
    source_prefix: str,
    *,
    hidden_map: torch.Tensor,
    generator: torch.Generator,
    noise_scale: float = 1e-5,
) -> None:
    # Design note: the VGGTO path keeps residual state in 1024 dimensions.
    # Only Q/K/V output channels and O input channels are widened to the 3072
    # MoT carrier, matching the FastWAM-style design we reviewed.
    qkv_w = vggto_state[f"{source_prefix}.attn.qkv.weight"]
    qkv_b = vggto_state[f"{source_prefix}.attn.qkv.bias"]
    qkv_bias_mask = vggto_state[f"{source_prefix}.attn.qkv.bias_mask"]
    qkv_b = qkv_b * qkv_bias_mask.to(device=qkv_b.device, dtype=qkv_b.dtype)
    qkv_weights, qkv_biases = split_qkv(qkv_w, qkv_b, dim=target_block.g_dim)

    source_hidden = target_block.g_dim
    identity_map = torch.arange(source_hidden, dtype=torch.long)

    target_qkv_w = []
    target_qkv_b = []
    for idx in range(3):
        tmp = torch.nn.Linear(target_block.g_dim, target_block.attn_dim)
        copy_rescale_linear_(
            tmp,
            qkv_weights[idx],
            qkv_biases[idx],
            hidden_map,
            identity_map,
            noise_std=noise_scale * float(qkv_weights[idx].std()),
            generator=generator,
        )
        target_qkv_w.append(tmp.weight.data)
        target_qkv_b.append(tmp.bias.data)
    target_block.qkv.weight.data.copy_(torch.cat(target_qkv_w, dim=0).to(target_block.qkv.weight.dtype))
    target_block.qkv.bias.data.copy_(torch.cat(target_qkv_b, dim=0).to(target_block.qkv.bias.dtype))

    copy_rescale_linear_(
        target_block.proj,
        vggto_state[f"{source_prefix}.attn.proj.weight"],
        vggto_state[f"{source_prefix}.attn.proj.bias"],
        identity_map,
        hidden_map,
        noise_std=noise_scale * float(vggto_state[f"{source_prefix}.attn.proj.weight"].std()),
        generator=generator,
    )

    target_block.mlp.fc1.weight.data.copy_(vggto_state[f"{source_prefix}.mlp.fc1.weight"].to(target_block.mlp.fc1.weight.dtype))
    target_block.mlp.fc1.bias.data.copy_(vggto_state[f"{source_prefix}.mlp.fc1.bias"].to(target_block.mlp.fc1.bias.dtype))
    target_block.mlp.fc2.weight.data.copy_(vggto_state[f"{source_prefix}.mlp.fc2.weight"].to(target_block.mlp.fc2.weight.dtype))
    target_block.mlp.fc2.bias.data.copy_(vggto_state[f"{source_prefix}.mlp.fc2.bias"].to(target_block.mlp.fc2.bias.dtype))

    copy_norm_(target_block.norm1, vggto_state[f"{source_prefix}.norm1.weight"], vggto_state[f"{source_prefix}.norm1.bias"], identity_map)
    copy_norm_(target_block.norm2, vggto_state[f"{source_prefix}.norm2.weight"], vggto_state[f"{source_prefix}.norm2.bias"], identity_map)

    source_head_dim = qkv_w.shape[0] // 3 // 16
    head_dim_map = flat_repeat_map(target_block.head_dim, source_head_dim)
    copy_norm_(target_block.q_norm, vggto_state[f"{source_prefix}.attn.q_norm.weight"], vggto_state[f"{source_prefix}.attn.q_norm.bias"], head_dim_map)
    copy_norm_(target_block.k_norm, vggto_state[f"{source_prefix}.attn.k_norm.weight"], vggto_state[f"{source_prefix}.attn.k_norm.bias"], head_dim_map)
    target_block.ls1.data.copy_(vggto_state[f"{source_prefix}.ls1.gamma"].to(target_block.ls1.dtype))
    target_block.ls2.data.copy_(vggto_state[f"{source_prefix}.ls2.gamma"].to(target_block.ls2.dtype))


def normalize_vggto_keys(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state.items():
        key = _strip_model_prefixes(key)
        normalized[key] = value
    return normalized


def ignored_vggto_keys(state: Mapping[str, torch.Tensor]) -> List[str]:
    return [
        key
        for key in state
        if key.startswith(("camera_head.", "text_alignment_head.", "aggregator.camera_token"))
    ]


def assert_required_vggto_prefixes(vggto_state: Mapping[str, torch.Tensor], *, depth: int = 24) -> None:
    missing = []
    for prefix in ("aggregator.patch_embed", "aggregator.frame_blocks", "aggregator.inter_frame_blocks", "dense_head"):
        if not any(k.startswith(prefix + ".") for k in vggto_state):
            missing.append(prefix + ".*")
    if "aggregator.register_token" not in vggto_state:
        missing.append("aggregator.register_token")
    for layer in range(depth):
        for stem in (f"aggregator.frame_blocks.{layer}", f"aggregator.inter_frame_blocks.{layer}"):
            if not any(k.startswith(stem + ".") for k in vggto_state):
                missing.append(stem + ".*")
    if missing:
        raise KeyError("VGGTO checkpoint missing required keys: " + ", ".join(missing[:20]))


def _vggto_layer_ids(vggto_state: Mapping[str, torch.Tensor], prefix: str) -> tuple[int, ...]:
    marker = prefix + "."
    layer_ids = set()
    for key in vggto_state:
        if not key.startswith(marker):
            continue
        layer_text = key[len(marker) :].split(".", 1)[0]
        if layer_text.isdigit():
            layer_ids.add(int(layer_text))
    return tuple(sorted(layer_ids))


def assert_vggto_source_depth(vggto_state: Mapping[str, torch.Tensor], *, expected_depth: int = 24) -> None:
    """Require exactly ``expected_depth`` frame and inter-frame source layers."""

    expected = tuple(range(expected_depth))
    layer_ids = {
        "frame_blocks": _vggto_layer_ids(vggto_state, "aggregator.frame_blocks"),
        "inter_frame_blocks": _vggto_layer_ids(vggto_state, "aggregator.inter_frame_blocks"),
    }
    mismatched = {
        name: ids
        for name, ids in layer_ids.items()
        if ids != expected
    }
    if mismatched:
        details = "; ".join(f"{name}={list(ids)}" for name, ids in mismatched.items())
        raise ValueError(
            f"VGGTO source must contain exactly {expected_depth} layers indexed "
            f"0..{expected_depth - 1}; {details}"
        )
    assert_required_vggto_prefixes(vggto_state, depth=expected_depth)


def validate_vggto_24_to_30_plan(
    plan: VGGTO24To30LayerPlan = VGGTO_24_TO_30_PLAN,
) -> None:
    """Fail fast if a mapping edit breaks the intended 30/15 partition."""

    if len(plan.target_to_source) != plan.target_depth:
        raise ValueError("target_to_source length must equal target_depth")
    if len(plan.frame_source_by_target) != plan.target_depth:
        raise ValueError("frame_source_by_target length must equal target_depth")
    if len(plan.mot_targets) != 15 or len(plan.native_targets) != 15:
        raise ValueError("VGGTO-30 must contain exactly 15 MOT and 15 native layers")
    if set(plan.mot_targets) & set(plan.native_targets):
        raise ValueError("MOT and native target layers must be disjoint")
    if set(plan.mot_targets) | set(plan.native_targets) != set(range(plan.target_depth)):
        raise ValueError("MOT and native target layers must cover all target layers")
    if len(plan.mot_source_by_target) != len(plan.mot_targets):
        raise ValueError("each MOT target must have one source layer")
    if any(source is not None for source in (plan.target_to_source[i] for i in plan.inserted_targets)):
        raise ValueError("inserted targets must be marked None in target_to_source")
    mapped_register_sources = tuple(plan.target_to_source[i] for i in plan.original_register_targets)
    if mapped_register_sources != plan.source_register_layers:
        raise ValueError(
            "original register targets must preserve source register order: "
            f"got {mapped_register_sources}, expected {plan.source_register_layers}"
        )
    if plan.covered_source_inter_frame_layers != tuple(range(plan.source_depth)):
        raise ValueError(
            "native and MOT source mappings must cover every source inter-frame layer exactly by role"
        )
    if plan.cache_source_by_target != dict(zip(plan.target_cache_layers, plan.source_cache_layers)):
        raise ValueError("cache layer mapping is inconsistent")


def vggto_module_state_for_prefix(
    vggto_state: Mapping[str, torch.Tensor],
    source_prefix: str,
    *,
    zero_residual_layer_scales: bool = False,
) -> tuple[Dict[str, torch.Tensor], tuple[str, ...]]:
    """Extract one module state, optionally making its residual path identity."""

    marker = source_prefix + "."
    module_state = {
        key[len(marker) :]: value
        for key, value in vggto_state.items()
        if key.startswith(marker)
    }
    if not module_state:
        raise KeyError(f"VGGTO checkpoint contains no keys under {source_prefix}.*")

    zeroed = ()
    if zero_residual_layer_scales:
        expected = ("ls1.gamma", "ls2.gamma")
        missing = [key for key in expected if key not in module_state]
        if missing:
            raise KeyError(
                f"cannot make {source_prefix} an identity; missing LayerScale keys: "
                + ", ".join(missing)
            )
        module_state = dict(module_state)
        for key in expected:
            module_state[key] = torch.zeros_like(module_state[key])
        zeroed = expected
    return module_state, zeroed


def load_vggto_module_from_prefix_(
    target_module: torch.nn.Module,
    vggto_state: Mapping[str, torch.Tensor],
    source_prefix: str,
    *,
    zero_residual_layer_scales: bool = False,
) -> tuple[int, tuple[str, ...]]:
    """Strictly copy one source-prefix module into an isomorphic target module."""

    module_state, zeroed = vggto_module_state_for_prefix(
        vggto_state,
        source_prefix,
        zero_residual_layer_scales=zero_residual_layer_scales,
    )
    target_module.load_state_dict(module_state, strict=True)
    return len(module_state), zeroed


def load_vggto_24_into_30_(
    target_vggto: torch.nn.Module,
    vggto_state: Mapping[str, torch.Tensor],
    *,
    plan: VGGTO24To30LayerPlan = VGGTO_24_TO_30_PLAN,
    cross_view_layer_scale_factor: float = 1.0,
) -> VGGTO24To30LoadReport:
    """Load the isomorphic VGGTO portions for the explicit 24-to-30 plan.

    Each native source inter-frame block is copied into both synchronized
    cross-view and same-view inter-frame target blocks. Register-token
    averaging, point-head loading, and widened MOT geometry
    initialization remain separate because they are not isomorphic copies.
    ``report.plan.source_for_mot(target_layer)`` gives the source inter-frame
    layer for each of those 15 geometry initializations.
    Only the cross-view copy's residual LayerScale parameters are multiplied by
    ``cross_view_layer_scale_factor``; the same-view copy remains unchanged.
    """

    cross_view_layer_scale_factor = float(cross_view_layer_scale_factor)
    if (
        not math.isfinite(cross_view_layer_scale_factor)
        or cross_view_layer_scale_factor < 0
    ):
        raise ValueError(
            "cross_view_layer_scale_factor must be finite and non-negative"
        )

    vggto_state = normalize_vggto_keys(vggto_state)
    validate_vggto_24_to_30_plan(plan)
    assert_vggto_source_depth(vggto_state, expected_depth=plan.source_depth)

    target_depth = int(getattr(target_vggto, "depth", len(target_vggto.frame_blocks)))
    if target_depth != plan.target_depth:
        raise ValueError(f"target VGGTO depth must be {plan.target_depth}, got {target_depth}")
    if len(target_vggto.frame_blocks) != plan.target_depth:
        raise ValueError(f"target must contain {plan.target_depth} frame blocks")
    if len(target_vggto.inter_frame_blocks) != plan.target_depth:
        raise ValueError(f"target must contain {plan.target_depth} inter-frame slots")
    if len(target_vggto.cross_view_blocks) != plan.target_depth:
        raise ValueError(f"target must contain {plan.target_depth} cross-view slots")

    target_mot_layers = set(getattr(target_vggto, "register_attention_indices", ()))
    if target_mot_layers != set(plan.mot_targets):
        raise ValueError(
            f"target MOT layers must be {list(plan.mot_targets)}, got {sorted(target_mot_layers)}"
        )
    target_cache_layers = tuple(getattr(target_vggto, "cached_layer_indices", ()))
    if target_cache_layers != plan.target_cache_layers:
        raise ValueError(
            f"target cache layers must be {list(plan.target_cache_layers)}, "
            f"got {list(target_cache_layers)}"
        )
    parameterized_mot_slots = {
        relation: [
            layer
            for layer in plan.mot_targets
            if any(True for _ in blocks[layer].parameters())
        ]
        for relation, blocks in (
            ("cross_view", target_vggto.cross_view_blocks),
            ("inter_frame", target_vggto.inter_frame_blocks),
        )
    }
    parameterized_mot_slots = {
        relation: layers for relation, layers in parameterized_mot_slots.items() if layers
    }
    if parameterized_mot_slots:
        raise ValueError(
            "target MOT relation slots must be parameterless placeholders: "
            + "; ".join(
                f"{relation}={','.join(map(str, layers))}"
                for relation, layers in parameterized_mot_slots.items()
            )
        )

    report = VGGTO24To30LoadReport(
        plan=plan,
        cross_view_layer_scale_factor=cross_view_layer_scale_factor,
    )
    report.loaded_patch_embed_keys, _ = load_vggto_module_from_prefix_(
        target_vggto.patch_embed,
        vggto_state,
        "aggregator.patch_embed",
    )

    for target_layer in range(plan.target_depth):
        source_layer = plan.source_for_frame(target_layer)
        loaded, zeroed = load_vggto_module_from_prefix_(
            target_vggto.frame_blocks[target_layer],
            vggto_state,
            f"aggregator.frame_blocks.{source_layer}",
            zero_residual_layer_scales=target_layer in plan.inserted_targets,
        )
        report.loaded_frame_block_keys += loaded
        if zeroed:
            report.zeroed_frame_layer_scales[target_layer] = list(zeroed)

    for target_layer in plan.native_targets:
        source_layer = plan.source_for_native(target_layer)
        loaded, _ = load_vggto_module_from_prefix_(
            target_vggto.cross_view_blocks[target_layer],
            vggto_state,
            f"aggregator.inter_frame_blocks.{source_layer}",
        )
        with torch.no_grad():
            cross_view_block = target_vggto.cross_view_blocks[target_layer]
            cross_view_block.ls1.gamma.mul_(cross_view_layer_scale_factor)
            cross_view_block.ls2.gamma.mul_(cross_view_layer_scale_factor)
        report.loaded_cross_view_keys += loaded
        loaded, _ = load_vggto_module_from_prefix_(
            target_vggto.inter_frame_blocks[target_layer],
            vggto_state,
            f"aggregator.inter_frame_blocks.{source_layer}",
        )
        report.loaded_native_inter_frame_keys += loaded

    report.loaded_dense_head_keys, _ = load_vggto_module_from_prefix_(
        target_vggto.dense_head,
        vggto_state,
        "dense_head",
    )
    report.deferred_mot_inter_frame_keys = sum(
        len(vggto_module_state_for_prefix(
            vggto_state,
            f"aggregator.inter_frame_blocks.{source_layer}",
        )[0])
        for source_layer in sorted(set(plan.mot_source_by_target))
    )
    return report
