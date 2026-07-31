import re
import shutil
from pathlib import Path


_CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint_step_(\d+)$")
EVAL_IN_PROGRESS_MARKER = "_EVAL_IN_PROGRESS"


def prune_successful_checkpoints(
    save_dir: str | Path,
    *,
    max_checkpoints: int,
) -> list[Path]:
    """Remove old completed checkpoints and return their paths."""

    if isinstance(max_checkpoints, bool) or not isinstance(max_checkpoints, int):
        raise TypeError("max_checkpoints must be an integer")
    if max_checkpoints < 1:
        raise ValueError(f"max_checkpoints must be positive, got {max_checkpoints}")

    completed = []
    for path in Path(save_dir).iterdir():
        match = _CHECKPOINT_DIR_PATTERN.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            continue
        if not (path / "_SUCCESS").is_file():
            continue
        completed.append((int(match.group(1)), path.name, path))

    completed.sort(key=lambda item: (item[0], item[1]))
    unprotected = [
        item
        for item in completed
        if not (item[2] / EVAL_IN_PROGRESS_MARKER).is_file()
    ]
    removed = [path for _, _, path in unprotected[:-max_checkpoints]]
    for path in removed:
        shutil.rmtree(path)
    return removed
