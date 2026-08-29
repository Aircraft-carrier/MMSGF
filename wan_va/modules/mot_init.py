"""Initialization helpers for the Video+Action MOT model."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class InitReport:
    video_source: str
    video_path: str
    loaded_video_keys: int = 0
    notes: List[str] = field(default_factory=list)
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
    target_blocks = list(target_model.mot_blocks)
    source_blocks = list(source_video_model.blocks)
    if len(target_blocks) != len(source_blocks):
        raise ValueError(
            "action initialization requires paired layers: "
            f"target={len(target_blocks)}, source={len(source_blocks)}"
        )
    report = ActionInitReport(seed=int(target_model.init_noise_seed))
    classified: Dict[str, str] = {}

    def classify(name: str, category: str) -> None:
        if name in classified:
            raise ValueError(f"action parameter {name} classified twice")
        classified[name] = category

    def copy_parameter(target_name, target_parameter, source_parameter, is_linear_weight):
        target_shape = tuple(target_parameter.shape)
        source_shape = tuple(source_parameter.shape)
        resized = resize_tensor_sequential_1d(source_parameter, target_shape)
        scale = None
        if is_linear_weight and source_shape[1] != target_shape[1]:
            scale = math.sqrt(float(source_shape[1]) / float(target_shape[1]))
            resized.mul_(scale)
        target_parameter.copy_(resized.to(target_parameter.device, target_parameter.dtype))
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
        source_parameters = dict(source_block.named_parameters())
        for local_name, target_parameter in target_block.named_parameters():
            if local_name not in source_parameters:
                raise KeyError(f"video block {layer_id} has no source for {local_name}")
            target_module = _parameter_module(target_block, local_name)
            source_module = _parameter_module(source_block, local_name)
            copy_parameter(
                f"mot_blocks.{layer_id}.action_block.{local_name}",
                target_parameter,
                source_parameters[local_name],
                isinstance(target_module, torch.nn.Linear)
                and isinstance(source_module, torch.nn.Linear)
                and local_name.rsplit(".", 1)[-1] == "weight",
            )
        report.shared.extend(
            f"mot_blocks.{layer_id}.video_block.attn2.{name}"
            for name, _ in target_composite.video_block.attn2.named_parameters()
            if name.startswith(("to_k.", "to_v.", "norm_k."))
        )

    target_condition = target_model.action_condition_embedder
    source_condition = source_video_model.condition_embedder
    source_parameters = dict(source_condition.named_parameters())
    for local_name, target_parameter in target_condition.named_parameters():
        target_module = _parameter_module(target_condition, local_name)
        source_module = _parameter_module(source_condition, local_name)
        copy_parameter(
            f"action_condition_embedder.{local_name}",
            target_parameter,
            source_parameters[local_name],
            isinstance(target_module, torch.nn.Linear)
            and isinstance(source_module, torch.nn.Linear)
            and local_name.rsplit(".", 1)[-1] == "weight",
        )
    copy_parameter(
        "action_scale_shift_table",
        target_model.action_scale_shift_table,
        source_video_model.scale_shift_table,
        False,
    )
    report.shared.extend(
        f"condition_embedder.text_embedder.{name}"
        for name, _ in target_model.condition_embedder.text_embedder.named_parameters()
    )
    random_parameters = dict(target_model.named_parameters())
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
    if uncovered:
        raise ValueError(f"action initialization coverage mismatch: {uncovered}")
    report.copied.sort()
    report.interpolated.sort()
    report.shared.sort()
    report.random.sort()
    return report
