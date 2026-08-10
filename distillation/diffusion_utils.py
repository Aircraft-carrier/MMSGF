"""Flow-matching tensor utilities for MOT distillation."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from distillation.schema import DenoisyInterval

if TYPE_CHECKING:
    from wan_va.utils.scheduler import FlowMatchScheduler


def sigmas_for_timesteps(
    scheduler: "FlowMatchScheduler",
    timesteps: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Look up scheduler sigmas for an arbitrary timestep tensor.

    原生 FlowMatchScheduler 的部分接口只正确处理一维 timestep；蒸馏需要
    ``[B,F]``，例如 ``[[0,0,700,320]]``。这里把 scheduler 表扩成
    ``[S,1,1]``，逐样本逐帧找最近离散点，返回同形 ``[B,F]`` sigma。
    数值 timestep=0 可能映射到一个很小但非零的 sigma，所以 condition 是否
    保持 clean 必须由调用方 mask 显式保证。
    """
    scheduler_timesteps = scheduler.timesteps.to(timesteps.device).reshape(
        -1, *([1] * timesteps.ndim)
    )
    indices = (scheduler_timesteps - timesteps.unsqueeze(0)).abs().argmin(dim=0)
    return scheduler.sigmas.to(timesteps.device)[indices].to(dtype=dtype)


def broadcast_frame_values(values: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    """Broadcast ``[B,F]`` values over a MOT sample's channel/spatial axes.

    video: ``[B,F] -> [B,1,F,1,1,1]``；
    action: ``[B,F] -> [B,1,F,1,1]``。frame 轴固定是 sample 的 dim=2。
    """
    return values.reshape(
        values.shape[0],
        1,
        values.shape[1],
        *([1] * (sample.ndim - 3)),
    )


def sample_consistency_timesteps(
    scheduler: "FlowMatchScheduler",
    num_steps: int,
    shape: tuple[int, int],
    device: torch.device,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a non-degenerate pair of scheduler states for consistency training.

    纸面示例（scheduler 从高噪声排到低噪声）：

    ``schedule = [1000, 750, 500, 250, 0]``，``num_steps = 2`` 时，
    ``stride = 2``。允许采样的当前索引只能是 ``0, 1, 2``，对应的下一
    索引是 ``2, 3, 4``。因此监督位置一定满足 ``next_sigma < sigma``，不会
    出现旧实现中尾部截断导致的 ``t_next == t`` 退化 pair。

    mask=False 的 condition/padding frame 仍返回数值 0；调用方随后还会用
    ``torch.where`` 显式保留 clean value，所以不依赖 scheduler 是否真的含
    有 sigma=0 的离散点。
    """
    schedule = scheduler.timesteps.to(device)
    if len(schedule) < 2:
        raise ValueError("Consistency training requires at least two scheduler states")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    stride = min(max(1, len(schedule) // num_steps), len(schedule) - 1)
    current_state_count = len(schedule) - stride
    timestep_ids = torch.randint(current_state_count, shape, device=device)
    next_ids = timestep_ids + stride
    timesteps = schedule[timestep_ids]
    next_timesteps = schedule[next_ids]
    return (
        torch.where(mask, timesteps, 0),
        torch.where(mask, next_timesteps, 0),
    )


def add_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: "FlowMatchScheduler",
) -> torch.Tensor:
    """Apply ``x_t=(1-sigma)*x0+sigma*noise`` independently per frame.

    单值手算：x0=2、noise=6、sigma=.75 时 x_t=5。该函数只做公式；是否
    对某个位置加噪由上层 ``add_noise_to_va`` 的 V/A mask 决定。
    """
    sigma = sigmas_for_timesteps(scheduler, timesteps, dtype=clean.dtype)
    sigma = broadcast_frame_values(sigma, clean)
    return (1.0 - sigma) * clean + sigma * noise


def renoise_x0(
    x0: torch.Tensor,
    next_timesteps: torch.Tensor,
    scheduler: "FlowMatchScheduler",
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add fresh Gaussian noise to an x0 prediction at the next timestep."""
    noise = torch.randn(
        x0.shape,
        device=x0.device,
        dtype=x0.dtype,
        generator=generator,
    )
    return add_noise(x0, noise, next_timesteps, scheduler), noise


def sample_interval_timesteps(
    interval: DenoisyInterval,
    shape: tuple[int, int],
    device: torch.device,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Sample integer model timesteps within one exit-derived DMD interval."""
    if len(shape) != 2:
        raise ValueError(f"timestep shape must be [B,F], got {shape}")
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(
            f"timestep mask must have shape {shape}, got {tuple(mask.shape)}"
        )
    low = int(math.ceil(float(interval.denoisy_to)))
    high = int(math.floor(float(interval.denoisy_from))) + 1
    if high <= low:
        raise ValueError(
            "denoisy interval contains no integer timestep: "
            f"[{interval.denoisy_to}, {interval.denoisy_from}]"
        )
    timesteps = torch.randint(low, high, shape, device=device)
    return torch.where(mask.to(device=device, dtype=torch.bool), timesteps, 0)


def flow_to_x0(
    flow: torch.Tensor,
    noisy_sample: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: "FlowMatchScheduler",
) -> torch.Tensor:
    """Convert predicted ``flow=noise-x0`` by ``x0_hat=x_t-sigma*flow``.

    若上例模型预测 flow=4，则 5-.75*4=2，准确恢复 clean。condition 位置仍
    应由上层用原 clean 覆盖，不能仅依赖 timestep 数值 0。
    """
    sigma = sigmas_for_timesteps(scheduler, timesteps, dtype=noisy_sample.dtype)
    return noisy_sample - flow * broadcast_frame_values(sigma, noisy_sample)


def consistency_prediction(
    flow: torch.Tensor,
    noisy_sample: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: "FlowMatchScheduler",
    *,
    sigma_data: float,
) -> torch.Tensor:
    """Apply Flash-WAM's variance-preserving video consistency boundary scaling."""
    sigma = sigmas_for_timesteps(scheduler, timesteps, dtype=noisy_sample.dtype)
    sigma = broadcast_frame_values(sigma, noisy_sample)
    sigma_data = noisy_sample.new_tensor(float(sigma_data))
    denominator = sigma.square() + sigma_data.square()
    c_skip = sigma_data.square() / denominator
    c_out = sigma * sigma_data / denominator.sqrt()
    x0 = noisy_sample - sigma * flow
    return c_skip * noisy_sample + c_out * x0


def flow_step(
    flow: torch.Tensor,
    sample: torch.Tensor,
    timesteps: torch.Tensor,
    next_timesteps: torch.Tensor,
    scheduler: "FlowMatchScheduler",
) -> torch.Tensor:
    """Move one flow line by ``x_next=x_t+(sigma_next-sigma_t)*flow``."""
    sigma = sigmas_for_timesteps(scheduler, timesteps, dtype=sample.dtype)
    next_sigma = sigmas_for_timesteps(scheduler, next_timesteps, dtype=sample.dtype)
    delta = broadcast_frame_values(next_sigma - sigma, sample)
    return sample + flow * delta
