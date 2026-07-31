"""Minimal FSDP2 optimizer memory control used during leak diagnosis."""

from __future__ import annotations

import gc
import json
import os
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
try:
    from torch.distributed.tensor.debug import (
        _get_fast_path_sharding_prop_cache_stats,
        _get_python_sharding_prop_cache_info,
    )
except ImportError:
    _get_fast_path_sharding_prop_cache_stats = None
    _get_python_sharding_prop_cache_info = None


def _rss_bytes() -> int:
    with open("/proc/self/statm", encoding="ascii") as f:
        resident_pages = int(f.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(1234)
    many_params = os.environ.get("MOT_MINIMAL_MANY_PARAMS", "0") == "1"
    if many_params:
        param_count = int(os.environ.get("MOT_MINIMAL_PARAM_COUNT", "1200"))
        model = nn.ParameterList(
            nn.Parameter(torch.randn(1024, device=device)) for _ in range(param_count)
        )
    else:
        blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 512))
            for _ in range(4)
        ).to(device)
        model = nn.Sequential(*blocks)
        for block in blocks:
            fully_shard(block)
    fully_shard(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer_step = optimizer.step
    if os.environ.get("MOT_MINIMAL_COMPILE_OPTIMIZER", "0") == "1":
        optimizer_step = torch.compile(optimizer_step)
    gc.disable()
    num_steps = int(os.environ.get("MOT_MINIMAL_STEPS", "1000"))
    for step in range(1, num_steps + 1):
        if many_params:
            for parameter in model.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.ones_like(parameter)
        else:
            x = torch.randn(8, 512, device=device)
            model(x).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_step()
        if not many_params:
            optimizer.zero_grad(set_to_none=True)
        gc.collect(0)
        if step == 1 or step % 100 == 0:
            dist.barrier()
            if rank == 0:
                fast_hits, fast_misses = (-1, -1)
                python_hits = python_misses = python_size = -1
                if _get_fast_path_sharding_prop_cache_stats is not None:
                    fast_hits, fast_misses = _get_fast_path_sharding_prop_cache_stats()
                    python_cache = _get_python_sharding_prop_cache_info()
                    python_hits = python_cache.hits
                    python_misses = python_cache.misses
                    python_size = python_cache.currsize
                print(
                    json.dumps(
                        {
                            "fast_hits": fast_hits,
                            "fast_misses": fast_misses,
                            "python_cache_hits": python_hits,
                            "python_cache_misses": python_misses,
                            "python_cache_size": python_size,
                            "python_blocks": sys.getallocatedblocks(),
                            "rank": rank,
                            "rss_bytes": _rss_bytes(),
                            "step": step,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
