"""Build action/state mmap cache as an explicit dataset preparation step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .action_cache import build_mot_action_cache


def build_action_cache_for_dataset(
    *,
    dataset_root: str | Path,
    output_root: str | Path | None = None,
    manifest_paths: list[str | Path] | None = None,
    max_files: int | None = None,
    num_workers: int = 16,
    overwrite: bool = False,
    update_mot_config: bool = False,
) -> Path:
    dataset_root = Path(dataset_root).resolve()
    action_output_root = Path(output_root).resolve() if output_root is not None else dataset_root / "cache" / "actions"
    resolved_manifest_paths = None if manifest_paths is None else [Path(path).resolve() for path in manifest_paths]
    return build_mot_action_cache(
        dataset_root=dataset_root,
        output_root=action_output_root,
        manifest_paths=resolved_manifest_paths,
        max_files=max_files,
        num_workers=int(num_workers),
        overwrite=overwrite,
        update_mot_config=update_mot_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mmap-friendly MOT action/state cache.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, action="append", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--update-mot-config", action="store_true")
    args = parser.parse_args()

    manifest_path = build_action_cache_for_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        manifest_paths=args.manifest_path,
        max_files=args.max_files,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        update_mot_config=args.update_mot_config,
    )
    print(json.dumps({"action_cache_manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
