"""Run bounded MOT training while recording CPython garbage collections."""

from __future__ import annotations

import atexit
import copy
import gc
import json
import os
import socket
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any


def _argument_value(name: str, default: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def _configure_diagnostic_config(config, *, num_steps: int) -> None:
    """Enable bounded diagnostics without changing the production step horizon."""

    fresh_start = os.environ.get("MOT_DIAG_FRESH_START", "0")
    if fresh_start not in {"0", "1"}:
        raise ValueError(f"MOT_DIAG_FRESH_START must be 0 or 1, got {fresh_start!r}")
    if fresh_start == "1":
        config.resume_from = None
    action_chunk_size = os.environ.get("MOT_DIAG_ACTION_CHUNK_SIZE")
    if action_chunk_size is not None:
        config.action_chunk_size = int(action_chunk_size)
        config.action_sequence_length = int(action_chunk_size) + 1
    config.enable_wandb = False
    config.gc_interval = 0
    config.performance_jsonl_enabled = True
    config.performance_jsonl_interval = int(os.environ.get("MOT_DIAG_PERFORMANCE_INTERVAL", "1"))
    config.performance_jsonl_max_steps = num_steps
    config.memory_jsonl_enabled = True
    config.memory_jsonl_interval = int(os.environ.get("MOT_DIAG_MEMORY_INTERVAL", "10"))
    config.memory_smaps_interval = int(os.environ.get("MOT_DIAG_SMAPS_INTERVAL", "25"))
    config.log_interval = int(os.environ.get("MOT_DIAG_LOG_INTERVAL", "10"))
    config.save_interval = int(config.num_steps) + 1


class _GCRecorder:
    def __init__(self, output_dir: Path, *, step_getter, generations: set[int] | None = None) -> None:
        self.pid = os.getpid()
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.step_getter = step_getter
        self.generations = generations or {2}
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"gc_rank_{self.rank:03d}.jsonl"
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o660)
        self.starts: dict[int, tuple[int, float]] = {}
        self._write(
            {
                "event": "process_start",
                "gc_enabled": gc.isenabled(),
                "gc_threshold": list(gc.get_threshold()),
                "hostname": socket.gethostname(),
                "local_rank": self.local_rank,
                "pid": self.pid,
                "rank": self.rank,
                "timestamp_unix": time.time(),
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        if self.fd < 0:
            return
        os.write(self.fd, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))

    def callback(self, phase: str, info: dict[str, Any]) -> None:
        # DataLoader workers inherit callbacks under fork; keep one file writer per rank.
        if os.getpid() != self.pid:
            return
        generation = int(info.get("generation", -1))
        if generation not in self.generations:
            return
        if phase == "start":
            self.starts[generation] = (time.perf_counter_ns(), time.time())
            return
        if phase != "stop":
            return
        start_ns, start_unix = self.starts.pop(
            generation,
            (time.perf_counter_ns(), time.time()),
        )
        stop_ns = time.perf_counter_ns()
        self._write(
            {
                "collected": int(info.get("collected", 0)),
                "duration_seconds": (stop_ns - start_ns) / 1_000_000_000.0,
                "event": "gc",
                "generation": generation,
                "local_rank": self.local_rank,
                "pid": self.pid,
                "rank": self.rank,
                "start_timestamp_unix": start_unix,
                "timestamp_unix": time.time(),
                "training_step": int(self.step_getter()),
                "uncollectable": int(info.get("uncollectable", 0)),
            }
        )

    def close(self) -> None:
        if self.fd < 0:
            return
        if os.getpid() != self.pid:
            os.close(self.fd)
            self.fd = -1
            return
        try:
            gc.callbacks.remove(self.callback)
        except ValueError:
            pass
        self._write(
            {
                "event": "process_stop",
                "gc_enabled": gc.isenabled(),
                "pid": self.pid,
                "rank": self.rank,
                "timestamp_unix": time.time(),
            }
        )
        os.close(self.fd)
        self.fd = -1


class _FSDPTreeFlattenCounter:
    def __init__(self) -> None:
        self.pid = os.getpid()
        self.counts = {"param_group": 0, "state": 0}
        self.previous_counts = dict(self.counts)
        self.patches = []

    def install(self) -> None:
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group, _fsdp_state

        for name, module in (("param_group", _fsdp_param_group), ("state", _fsdp_state)):
            original = module.tree_flatten

            def counted_tree_flatten(*args, _name=name, _original=original, **kwargs):
                if os.getpid() == self.pid:
                    self.counts[_name] += 1
                return _original(*args, **kwargs)

            module.tree_flatten = counted_tree_flatten
            self.patches.append((module, original, counted_tree_flatten))

    def step_counts(self) -> dict[str, int]:
        deltas = {
            name: count - self.previous_counts[name]
            for name, count in self.counts.items()
        }
        self.previous_counts = dict(self.counts)
        return deltas

    def close(self) -> None:
        for module, original, replacement in reversed(self.patches):
            if module.tree_flatten is replacement:
                module.tree_flatten = original
        self.patches.clear()


def main() -> None:
    from wan_va import train_mot

    config_name = _argument_value("--config-name", "umi_3dwam_train")
    num_steps = int(os.environ.get("MOT_DIAG_NUM_STEPS", "150"))
    gc_mode = os.environ.get("MOT_DIAG_GC_MODE", "observe").strip().lower()
    count_tree_flatten = os.environ.get("MOT_DIAG_COUNT_FSDP_TREE_FLATTEN", "0")
    trace_python_allocations = os.environ.get("MOT_DIAG_TRACEMALLOC", "0")
    tracemalloc_frames = int(os.environ.get("MOT_DIAG_TRACEMALLOC_FRAMES", "5"))
    tracemalloc_start_step = int(os.environ.get("MOT_DIAG_TRACEMALLOC_START_STEP", "0"))
    clear_dtensor_fast_cache = os.environ.get("MOT_DIAG_CLEAR_DTENSOR_FAST_CACHE", "0")
    record_dtensor_cache = os.environ.get("MOT_DIAG_DTENSOR_CACHE_STATS", "0")
    optimizer_mode = os.environ.get("MOT_DIAG_OPTIMIZER_MODE", "production")
    if num_steps <= 0:
        raise ValueError(f"MOT_DIAG_NUM_STEPS must be positive, got {num_steps}")
    if gc_mode not in {
        "observe",
        "disabled",
        "step_gen0",
        "step_gen1",
        "step_gen2_freeze",
    }:
        raise ValueError(
            "MOT_DIAG_GC_MODE must be observe, disabled, step_gen0, step_gen1, "
            "or step_gen2_freeze, "
            f"got {gc_mode!r}"
        )
    if count_tree_flatten not in {"0", "1"}:
        raise ValueError(
            "MOT_DIAG_COUNT_FSDP_TREE_FLATTEN must be 0 or 1, "
            f"got {count_tree_flatten!r}"
        )
    if trace_python_allocations not in {"0", "1"}:
        raise ValueError(
            f"MOT_DIAG_TRACEMALLOC must be 0 or 1, got {trace_python_allocations!r}"
        )
    if tracemalloc_frames <= 0:
        raise ValueError(
            f"MOT_DIAG_TRACEMALLOC_FRAMES must be positive, got {tracemalloc_frames}"
        )
    if not 0 <= tracemalloc_start_step <= 20:
        raise ValueError(
            "MOT_DIAG_TRACEMALLOC_START_STEP must be between 0 and 20, "
            f"got {tracemalloc_start_step}"
        )
    if clear_dtensor_fast_cache not in {"0", "1"}:
        raise ValueError(
            "MOT_DIAG_CLEAR_DTENSOR_FAST_CACHE must be 0 or 1, "
            f"got {clear_dtensor_fast_cache!r}"
        )
    if record_dtensor_cache not in {"0", "1"}:
        raise ValueError(
            "MOT_DIAG_DTENSOR_CACHE_STATS must be 0 or 1, "
            f"got {record_dtensor_cache!r}"
        )
    if optimizer_mode not in {"production", "foreach", "foreach_capturable", "unfused"}:
        raise ValueError(
            "MOT_DIAG_OPTIMIZER_MODE must be production, foreach, "
            "foreach_capturable, or unfused, "
            f"got {optimizer_mode!r}"
        )

    config = copy.deepcopy(train_mot.VA_CONFIGS[config_name])
    _configure_diagnostic_config(config, num_steps=num_steps)
    train_mot.VA_CONFIGS._cache[config_name] = config

    class DiagnosticTrainer(train_mot.MOTTrainer):
        def __init__(self, trainer_config) -> None:
            super().__init__(trainer_config)
            # Keep the production horizon through sampler, scheduler, and checkpoint
            # loading, then bound only this disposable process's training loop.
            self.config.num_steps = self.step + num_steps
            self._gc_recorder = None
            self._gc_was_enabled = gc.isenabled()
            self._tree_flatten_counter = None
            self._tracemalloc_baseline = None
            self._tracemalloc_started = False
            if count_tree_flatten == "1":
                self._tree_flatten_counter = _FSDPTreeFlattenCounter()
                self._tree_flatten_counter.install()
            if clear_dtensor_fast_cache == "1":
                from torch.distributed.tensor.debug import _clear_sharding_prop_cache

                _clear_sharding_prop_cache()
            if optimizer_mode in {"foreach", "foreach_capturable", "unfused"}:
                self.optimizer.defaults["fused"] = False
                self.optimizer.defaults["foreach"] = optimizer_mode != "unfused"
                self.optimizer.defaults["capturable"] = optimizer_mode == "foreach_capturable"
                for group in self.optimizer.param_groups:
                    group["fused"] = False
                    group["foreach"] = optimizer_mode != "unfused"
                    group["capturable"] = optimizer_mode == "foreach_capturable"

        def _start_gc_diagnostic(self) -> None:
            if self._gc_recorder is not None:
                return
            if gc_mode in {"disabled", "step_gen0", "step_gen1", "step_gen2_freeze"}:
                gc.disable()
            if gc_mode == "step_gen2_freeze":
                gc.freeze()
            diagnostic_generation = {
                "step_gen0": 0,
                "step_gen1": 1,
                "step_gen2_freeze": 2,
            }.get(gc_mode, 2)
            self._gc_recorder = _GCRecorder(
                Path(self.config.save_root) / "gc_diagnostics",
                step_getter=lambda: self.step if gc_mode.startswith("step_gen") else self.step + 1,
                generations={diagnostic_generation} if gc_mode.startswith("step_gen") else {2},
            )
            gc.callbacks.append(self._gc_recorder.callback)
            atexit.register(self._gc_recorder.close)
            if trace_python_allocations == "1" and tracemalloc_start_step == 0:
                tracemalloc.start(tracemalloc_frames)
                self._tracemalloc_started = True

        def _stop_gc_diagnostic(self) -> None:
            if self._tree_flatten_counter is not None:
                self._tree_flatten_counter.close()
                self._tree_flatten_counter = None
            if self._gc_recorder is not None:
                self._gc_recorder.close()
                self._gc_recorder = None
            if self._gc_was_enabled:
                gc.enable()
            if self._tracemalloc_started:
                tracemalloc.stop()
                self._tracemalloc_started = False

        def _record_tracemalloc(self) -> None:
            if (
                trace_python_allocations == "1"
                and not self._tracemalloc_started
                and self.step >= tracemalloc_start_step
            ):
                tracemalloc.start(tracemalloc_frames)
                self._tracemalloc_started = True
            if not self._tracemalloc_started or self._gc_recorder is None:
                return
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            self._gc_recorder._write(
                {
                    "current_bytes": current_bytes,
                    "event": "tracemalloc",
                    "peak_bytes": peak_bytes,
                    "pid": os.getpid(),
                    "rank": int(self.config.rank),
                    "timestamp_unix": time.time(),
                    "training_step": int(self.step),
                }
            )
            if self.step == 20:
                self._tracemalloc_baseline = tracemalloc.take_snapshot()
            elif self.step == self.config.num_steps and self._tracemalloc_baseline is not None:
                snapshot = tracemalloc.take_snapshot()
                stats = snapshot.compare_to(self._tracemalloc_baseline, "traceback")
                records = []
                for stat in stats:
                    if stat.size_diff <= 0 and stat.count_diff <= 0:
                        continue
                    records.append(
                        {
                            "count": stat.count,
                            "count_diff": stat.count_diff,
                            "size_bytes": stat.size,
                            "size_diff_bytes": stat.size_diff,
                            "traceback": list(stat.traceback.format()),
                        }
                    )
                    if len(records) == 200:
                        break
                path = (
                    Path(self.config.save_root)
                    / "gc_diagnostics"
                    / f"tracemalloc_rank_{int(self.config.rank):03d}.json"
                )
                path.write_text(
                    json.dumps(
                        {
                            "baseline_step": 20,
                            "final_step": int(self.step),
                            "stats": records,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        def _record_dtensor_cache(self) -> None:
            if (
                clear_dtensor_fast_cache != "1"
                and record_dtensor_cache != "1"
            ) or self._gc_recorder is None:
                return
            from torch.distributed.tensor.debug import (
                _clear_fast_path_sharding_prop_cache,
                _get_fast_path_sharding_prop_cache_stats,
                _get_python_sharding_prop_cache_info,
            )

            info = _get_python_sharding_prop_cache_info()
            fast_hits, fast_misses = _get_fast_path_sharding_prop_cache_stats()
            from torch.distributed.tensor import DTensor

            tensor_meta_info = (
                DTensor._op_dispatcher.sharding_propagator._propagate_tensor_meta_cached.cache_info()
            )
            clear_start = time.perf_counter()
            if clear_dtensor_fast_cache == "1":
                _clear_fast_path_sharding_prop_cache()
            clear_duration = time.perf_counter() - clear_start
            from wan_va.modules import fa4_attention

            self._gc_recorder._write(
                {
                    "cleared": clear_dtensor_fast_cache == "1",
                    "clear_duration_seconds": clear_duration,
                    "currsize": info.currsize,
                    "event": "dtensor_sharding_cache",
                    "fa4_chunk_cache_size": len(fa4_attention._CHUNK_BLOCK_CACHE),
                    "fa4_mot_cache_size": len(fa4_attention._MOT_BLOCK_CACHE),
                    "fast_hits": fast_hits,
                    "fast_misses": fast_misses,
                    "hits": info.hits,
                    "maxsize": info.maxsize,
                    "misses": info.misses,
                    "tensor_meta_currsize": tensor_meta_info.currsize,
                    "tensor_meta_hits": tensor_meta_info.hits,
                    "tensor_meta_misses": tensor_meta_info.misses,
                    "pid": os.getpid(),
                    "rank": int(self.config.rank),
                    "timestamp_unix": time.time(),
                    "training_step": int(self.step),
                }
            )

        def _start_training_gc_control(self) -> None:
            # The diagnostic modes own GC state so observe can retain automatic GC.
            return

        def _get_next_batch(self):
            if self.train_loader_iter is None:
                # Fork persistent workers while they still have the default GC state.
                self.train_loader_iter = iter(self.train_loader)
                self._start_gc_diagnostic()
            return super()._get_next_batch()

        def _synchronize_and_collect_garbage(
            self,
            *,
            step_completed: bool,
        ) -> tuple[float, float, float]:
            (
                post_step_barrier_time,
                gc_collect_time,
                post_gc_barrier_time,
            ) = super()._synchronize_and_collect_garbage(step_completed=step_completed)
            if not step_completed:
                return post_step_barrier_time, gc_collect_time, post_gc_barrier_time

            if self._tree_flatten_counter is not None and self._gc_recorder is not None:
                counts = self._tree_flatten_counter.step_counts()
                self._gc_recorder._write(
                    {
                        "event": "fsdp_tree_flatten",
                        "param_group_calls": counts["param_group"],
                        "pid": os.getpid(),
                        "rank": int(self.config.rank),
                        "state_calls": counts["state"],
                        "timestamp_unix": time.time(),
                        "training_step": int(self.step),
                    }
                )
            self._record_tracemalloc()
            self._record_dtensor_cache()

            if not gc_mode.startswith("step_gen"):
                return post_step_barrier_time, gc_collect_time, post_gc_barrier_time

            gc_start = time.perf_counter()
            generation = {
                "step_gen0": 0,
                "step_gen1": 1,
                "step_gen2_freeze": 2,
            }[gc_mode]
            gc.collect(generation)
            gc_collect_time = time.perf_counter() - gc_start
            if train_mot.dist.is_initialized():
                barrier_start = time.perf_counter()
                train_mot.dist.barrier()
                post_gc_barrier_time = time.perf_counter() - barrier_start
            return post_step_barrier_time, gc_collect_time, post_gc_barrier_time

        def save_checkpoint(self) -> Path:
            self._last_checkpoint_step = int(self.step)
            return Path(self.config.save_root) / "checkpoints" / f"checkpoint_step_{self.step}"

        def train(self):
            try:
                return super().train()
            finally:
                self._stop_gc_diagnostic()

    train_mot.MOTTrainer = DiagnosticTrainer
    train_mot.init_logger()
    train_mot.main()


if __name__ == "__main__":
    main()
