"""Flow-matching tensor utilities for MOT distillation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from distillation.model.wan_wrapper import (
    broadcast_frame_values,
    sigmas_for_timesteps,
)

if TYPE_CHECKING:
    from wan_va.utils.scheduler import FlowMatchScheduler


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
