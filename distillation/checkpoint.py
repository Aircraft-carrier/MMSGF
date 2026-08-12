"""Distillation checkpoint discovery."""
from __future__ import annotations

from pathlib import Path


def find_latest_successful_checkpoint(checkpoint_root: Path) -> Path | None:
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        if path.is_dir() and (path / "_SUCCESS").exists():
            name = path.name
            if name.startswith("checkpoint_step_"):
                try:
                    step = int(name[len("checkpoint_step_") :])
                    candidates.append((step, path))
                except ValueError:
                    continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]
