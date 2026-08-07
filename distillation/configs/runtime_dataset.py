"""Runtime dataset compatibility for distillation configs."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        if not dst.exists() and not dst.is_symlink():
            shutil.copy2(src, dst)


def _existing_path(value: str | None, *, roots: tuple[Path, ...] = ()) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    candidates = [path]
    for root in roots:
        if path.is_absolute():
            # Prepared metadata can contain absolute paths from the machine
            # that built the dataset.  Distillation-owned assets keep the
            # same conventional basename under the selected dataset root.
            candidates.extend(
                (
                    root / path.name,
                    root / "cache" / "actions" / path.name,
                    root / "meta" / path.name,
                )
            )
        else:
            candidates.append(root / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _first_manifest_norm_stats_key(source_root: Path) -> str | None:
    for name in ("mot_final_training_manifest.jsonl",):
        path = source_root / "meta" / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("norm_stats_key")
            if key is not None:
                return str(key)
    return None


def prepare_distillation_dataset_config() -> None:
    """Select a distillation-owned MOT_DATASET_ROOT before wan_va config import.

    The current distillation dataset is a symlink to a read-only prepared root
    whose metadata may reference stale absolute paths and may omit the legacy
    top-level ``norm_stat`` field expected by ``wan_va`` training config.
    Create a small shadow root containing only patched metadata and symlinks to
    the original manifests; sample rows still point to the real parquet/video
    payloads.
    """

    repo_root = Path(__file__).resolve().parents[2]
    requested_root = Path(
        os.environ.get(
            "MOT_DATASET_ROOT",
            str(repo_root / "data" / "umi_distill_train"),
        )
    ).resolve()
    source_root = requested_root
    mot_config_path = source_root / "meta" / "mot_config.json"
    if not mot_config_path.is_file():
        return

    try:
        mot_config = json.loads(mot_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        source_root = (repo_root / "data" / "umi_distill_train").resolve()
        mot_config_path = source_root / "meta" / "mot_config.json"
        mot_config = json.loads(mot_config_path.read_text(encoding="utf-8"))

    if mot_config.get("_distillation_shadow_config") is True:
        os.environ["MOT_DATASET_ROOT"] = str(requested_root)
        return

    shadow_root = repo_root / "train_logs" / "distill_runtime_dataset" / source_root.name
    shadow_meta = shadow_root / "meta"
    shadow_meta.mkdir(parents=True, exist_ok=True)

    for name in ("mot_final_training_manifest.jsonl",):
        src = source_root / "meta" / name
        if src.is_file():
            _link_or_copy(src, shadow_meta / name)

    if "norm_stat" not in mot_config:
        stats_by_task = mot_config.get("norm_stats_by_task")
        if not stats_by_task:
            raise ValueError(
                "Distillation MOT dataset is missing both norm_stat and "
                f"norm_stats_by_task: {mot_config_path}"
            )
        default_norm_key = (
            mot_config.get("norm_stats_key")
            or _first_manifest_norm_stats_key(source_root)
            or sorted(stats_by_task)[0]
        )
        if str(default_norm_key) not in stats_by_task:
            raise ValueError(
                "Distillation MOT dataset norm_stats_key is not present in "
                f"norm_stats_by_task: {default_norm_key!r}"
            )
        mot_config["norm_stat"] = stats_by_task[str(default_norm_key)]
        mot_config["norm_stats_key"] = str(default_norm_key)

    mot_config["mot_manifest_path"] = str(
        shadow_meta / "mot_final_training_manifest.jsonl"
    )

    text_cache_root = repo_root / "train_logs" / "distill_assets" / "text_cache"
    empty_emb = (
        _existing_path(os.environ.get("MOT_EMPTY_EMB_PATH"), roots=(source_root, repo_root))
        or _existing_path(str(text_cache_root / "empty_emb.pt"))
        or _existing_path(mot_config.get("empty_emb_path"), roots=(source_root, repo_root))
    )
    text_cache = (
        _existing_path(os.environ.get("MOT_TEXT_EMB_CACHE_PATH"), roots=(source_root, repo_root))
        or _existing_path(str(text_cache_root / "text_emb_cache.pt"))
        or _existing_path(mot_config.get("text_emb_cache_path"), roots=(source_root, repo_root))
    )
    if empty_emb is not None:
        mot_config["empty_emb_path"] = empty_emb
    if text_cache is not None:
        mot_config["text_emb_cache_path"] = text_cache

    action_cache = _existing_path(
        os.environ.get("MOT_ACTION_CACHE_MANIFEST")
        or mot_config.get("action_cache_manifest_path"),
        roots=(source_root, repo_root),
    )
    mot_config["action_cache_manifest_path"] = action_cache or ""
    mot_config["_distillation_shadow_config"] = True

    target_config = shadow_meta / "mot_config.json"
    temp_config = shadow_meta / f".mot_config.{os.getpid()}.tmp"
    temp_config.write_text(
        json.dumps(mot_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_config.replace(target_config)
    os.environ["MOT_DATASET_ROOT"] = str(shadow_root)


def apply_distillation_runtime_overrides(config: Any) -> Any:
    load_worker = os.environ.get("MOT_LOAD_WORKER")
    if load_worker is not None:
        config.load_worker = int(load_worker)
    return config
