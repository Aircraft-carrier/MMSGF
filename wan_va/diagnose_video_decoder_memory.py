"""Standalone torchcodec decoder memory diagnostic."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path

import torch
from torchcodec.decoders import VideoDecoder

from wan_va.dataset.mot_dataset import _BoundedLRUCache
from wan_va.train_mot import _read_allocator_stats, _read_process_memory


def _video_paths(manifest: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for view in row["views"]:
                path = Path(view["video_path"])
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
            if len(paths) >= count:
                return paths
    raise RuntimeError(f"manifest contains only {len(paths)} unique videos")


def _snapshot(*, iteration: int, cache: _BoundedLRUCache, phase: str) -> None:
    process = _read_process_memory(os.getpid(), include_expensive=True)
    allocator = _read_allocator_stats()
    print(
        json.dumps(
            {
                "allocator": allocator,
                "cache": cache.stats(),
                "iteration": iteration,
                "phase": phase,
                "process": process,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reuse", "rotate"), required=True)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--cache-size", type=int, default=2)
    parser.add_argument("--video-count", type=int, default=32)
    parser.add_argument("--sample-interval", type=int, default=20)
    args = parser.parse_args()

    torch.set_num_threads(1)
    manifest = Path(
        "data/data/umi_mot_full_data_train_0712_final/meta/"
        "mot_final_training_non_pointcloud_manifest.jsonl"
    )
    paths = _video_paths(manifest, args.video_count)
    active_paths = paths[:2] if args.mode == "reuse" else paths
    cache = _BoundedLRUCache(args.cache_size, name="video_decoder_cache_size")

    _snapshot(iteration=0, cache=cache, phase="start")
    for iteration in range(1, args.iterations + 1):
        path = active_paths[(iteration - 1) % len(active_paths)]
        decoder = cache.get(path)
        if decoder is None:
            decoder = VideoDecoder(
                path,
                dimension_order="NCHW",
                device="cpu",
                seek_mode="approximate",
            )
            cache.put(path, decoder)
        frame_count = len(decoder)
        start = (iteration * 37) % max(1, frame_count - 49)
        frames = decoder.get_frames_at([start + offset * 4 for offset in range(13)])
        del frames, decoder
        if iteration % args.sample_interval == 0:
            _snapshot(iteration=iteration, cache=cache, phase="decode")

    cache.clear()
    gc.collect()
    _snapshot(iteration=args.iterations, cache=cache, phase="after_clear_gc")
    ctypes.CDLL(None).malloc_trim(0)
    _snapshot(iteration=args.iterations, cache=cache, phase="after_malloc_trim")


if __name__ == "__main__":
    main()
