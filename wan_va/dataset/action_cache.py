#!/usr/bin/env python3
"""Build mmap-friendly action arrays for MOT raw-window training."""

from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq


ACTION_CACHE_FORMAT = "mot_action_state_cache_v2"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _cache_index_key(path: Path) -> str:
    return str(path.resolve()) if path.exists() else str(path)


def _resolve_payload_path(manifest_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def load_action_cache_index(path: str | Path | None) -> dict[str, dict[str, Path | None]]:
    if path is None:
        return {}
    manifest_path = Path(path)
    index: dict[str, dict[str, Path | None]] = {}
    for row in _read_jsonl(manifest_path):
        data_file = Path(row["data_file"])
        payload = {
            "actions_npy": _resolve_payload_path(manifest_path, row["actions_npy"]),
            "states_npy": _resolve_payload_path(manifest_path, row["states_npy"]),
            "index_npy": _resolve_payload_path(manifest_path, row["index_npy"]) if row.get("index_npy") else None,
        }
        index[str(data_file)] = payload
        if data_file.exists():
            index[str(data_file.resolve())] = payload
    return index


def _resolve_dataset_path(dataset_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return dataset_root / path


def _collect_data_files(dataset_root: Path, mot_config: dict[str, Any]) -> list[Path]:
    data_files: set[Path] = set()
    manifest_values = [
        mot_config.get("mot_manifest_path"),
    ]
    for manifest_value in manifest_values:
        manifest_path = _resolve_dataset_path(dataset_root, manifest_value)
        if manifest_path is None or not manifest_path.is_file():
            continue
        for row in _read_jsonl(manifest_path):
            data_file = row.get("data_file") or (row.get("output") or {}).get("data_file")
            if data_file:
                data_files.add(Path(data_file))
    return sorted(data_files, key=str)


def _collect_data_files_from_manifests(manifest_paths: Iterable[Path]) -> list[Path]:
    data_files: set[Path] = set()
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            continue
        for row in _read_jsonl(manifest_path):
            data_file = row.get("data_file") or (row.get("output") or {}).get("data_file")
            if data_file:
                data_files.add(Path(data_file))
    return sorted(data_files, key=str)


def _cache_dir_for(output_root: Path, data_file: Path) -> Path:
    digest = hashlib.sha256(str(data_file).encode("utf-8")).hexdigest()
    return output_root / "files" / digest[:2] / digest


def _build_one(data_file: Path, output_root: Path, overwrite: bool) -> dict[str, Any]:
    cache_dir = _cache_dir_for(output_root, data_file)
    actions_npy = cache_dir / "actions.npy"
    states_npy = cache_dir / "states.npy"
    index_npy = cache_dir / "index.npy"
    meta_path = cache_dir / "meta.json"
    if actions_npy.is_file() and states_npy.is_file() and meta_path.is_file() and not overwrite:
        meta = _read_json(meta_path)
        if not bool(meta.get("has_index")) or not index_npy.is_file():
            raise ValueError(f"LeRobot action cache requires an index column: {data_file}")
        return {
            "data_file": str(data_file),
            "actions_npy": str(actions_npy),
            "states_npy": str(states_npy),
            "index_npy": str(index_npy),
            "num_rows": int(meta["num_rows"]),
            "action_dim": int(meta["action_dim"]),
            "state_dim": int(meta["state_dim"]),
        }

    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    cache_dir.mkdir(parents=True, exist_ok=True)
    schema_names = set(pq.read_schema(data_file).names)
    if "index" not in schema_names:
        raise ValueError(f"LeRobot action cache requires an index column: {data_file}")
    table = pq.read_table(data_file, columns=["action", "observation.state", "index"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
    if actions.shape[0] != states.shape[0] or actions.shape[0] != indices.shape[0]:
        raise ValueError(
            f"LeRobot action cache length mismatch in {data_file}: "
            f"actions={actions.shape[0]} states={states.shape[0]} index={indices.shape[0]}"
        )
    np.save(actions_npy, actions)
    np.save(states_npy, states)
    np.save(index_npy, indices)

    meta = {
        "format": ACTION_CACHE_FORMAT,
        "data_file": str(data_file),
        "actions_npy": str(actions_npy),
        "states_npy": str(states_npy),
        "index_npy": str(index_npy),
        "has_index": True,
        "num_rows": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]) if actions.ndim == 2 else 0,
        "state_dim": int(states.shape[1]) if states.ndim == 2 else 0,
    }
    _write_json(meta_path, meta)
    return {
        "data_file": str(data_file),
        "actions_npy": str(actions_npy),
        "states_npy": str(states_npy),
        "index_npy": str(index_npy),
        "num_rows": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]) if actions.ndim == 2 else 0,
        "state_dim": int(states.shape[1]) if states.ndim == 2 else 0,
    }


def build_mot_action_cache(
    *,
    dataset_root: Path,
    output_root: Path,
    manifest_paths: list[Path] | None = None,
    max_files: int | None = None,
    num_workers: int = 1,
    overwrite: bool = False,
    update_mot_config: bool = False,
) -> Path:
    mot_config_path = dataset_root / "meta" / "mot_config.json"
    mot_config = None
    if manifest_paths is None:
        mot_config = _read_json(mot_config_path)
        data_files = _collect_data_files(dataset_root, mot_config)
    else:
        data_files = _collect_data_files_from_manifests(manifest_paths)
    if max_files is not None:
        data_files = data_files[: int(max_files)]
    if not data_files:
        raise ValueError(f"No action parquet files found from {mot_config_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    if int(num_workers) > 1:
        with futures.ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            rows = list(executor.map(_build_one, data_files, [output_root] * len(data_files), [overwrite] * len(data_files)))
    else:
        rows = [_build_one(data_file, output_root, overwrite) for data_file in data_files]

    manifest_path = output_root / "action_cache_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    _write_json(
        output_root / "cache_info.json",
        {
            "format": ACTION_CACHE_FORMAT,
            "dataset_root": str(dataset_root),
            "manifest_path": str(manifest_path),
            "num_files": len(rows),
            "total_rows": int(sum(int(row["num_rows"]) for row in rows)),
        },
    )
    if update_mot_config:
        if mot_config is None:
            mot_config = _read_json(mot_config_path)
        mot_config["action_cache_manifest_path"] = str(manifest_path)
        _write_json(mot_config_path, mot_config)
    return manifest_path
