#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ID="richengZ/umi_data_with_pointcloud"
DATE_ROOT=""
DEST_DIR="/team_data/umi_data/lumos_lerobot"
REVISION="master"
MAX_WORKERS="2"
STAGING_ROOT="/team_data/umi_data/.modelscope_pointcloud_staging"
TASK_FILTER=""
DRY_RUN=0
FORCE_DOWNLOAD=0
OVERWRITE_POINTCLOUD=0
OVERWRITE_RAW=0
KEEP_STAGE=0
LOG_ROOT="/data_fcy/code/lingbot-va/modelscope_pointcloud_logs"
PYTHON_BIN="${PYTHON_BIN:-/team_data/fcy/basics/miniconda3/bin/python}"
CONDA_BIN="$(dirname "$PYTHON_BIN")"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-$CONDA_BIN/modelscope}"
ZSTD_BIN="${ZSTD_BIN:-$CONDA_BIN/zstd}"


usage() {
  cat <<'EOF'
Usage: ./download_modelscope_pointcloud.sh [options]

Sync compressed pointcloud tasks from ModelScope:
  - download task/raw_lerobot_idx.jsonl to the local task directory
  - download task/pointcloud/pointcloud*.tar.zst[.part-*] directly under the local task directory
  - after one task's parts are complete, extract in the local task directory and delete the parts

Options:
  --repo ID                 ModelScope dataset id. Default: richengZ/umi_data_with_pointcloud
  --date YYYYMMDD           Remote date directory. Default: all remote date directories, sorted
  --dest DIR                Local dataset root. Default: /team_data/umi_data/lumos_lerobot
  --revision REV            Dataset revision/branch. Default: master
  --task DATE/task_xxx      Process one task. You may also pass only task_xxx.
  --max-workers N           ModelScope download concurrency. Default: 2
  --staging-root DIR        Temporary locks/markers root. Data downloads go to --dest.
  --force-download          Pass --force to ModelScope and re-download local files.
  --overwrite-pointcloud    Replace existing local task/pointcloud.
  --overwrite-raw           Replace existing raw_lerobot_idx.jsonl.
  --keep-stage              Keep local archive parts after successful extraction.
  --log-root DIR            Persistent log root. Default: ./modelscope_pointcloud_logs
  --dry-run                 Print configuration only; no network and no writes.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_ID="$2"; shift 2 ;;
    --date) DATE_ROOT="$2"; shift 2 ;;
    --dest) DEST_DIR="$2"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    --task) TASK_FILTER="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --staging-root) STAGING_ROOT="$2"; shift 2 ;;
    --force-download) FORCE_DOWNLOAD=1; shift ;;
    --overwrite-pointcloud) OVERWRITE_POINTCLOUD=1; shift ;;
    --overwrite-raw) OVERWRITE_RAW=1; shift ;;
    --keep-stage) KEEP_STAGE=1; shift ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$DEST_DIR" || "$DEST_DIR" == "/" ]]; then
  echo "Refusing unsafe destination: '$DEST_DIR'" >&2
  exit 2
fi

if [[ "$DRY_RUN" == "1" ]]; then
  cat <<EOF
Repository:        $REPO_ID
Revision:          $REVISION
Remote date roots: $(if [[ -n "$DATE_ROOT" ]]; then echo "$DATE_ROOT"; else echo "<all remote date directories, sorted>"; fi)
Target root:       $DEST_DIR
Staging root:      $STAGING_ROOT
Log root:          $LOG_ROOT
HTTP proxy:        ${http_proxy:-<unset>}
HTTPS proxy:       ${https_proxy:-<unset>}
Task filter:       ${TASK_FILTER:-<all>}
Max workers:       $MAX_WORKERS
Downloader:        one modelscope download command per task
Download dir:      $DEST_DIR
Raw idx:           download task/raw_lerobot_idx.jsonl
Pointcloud mode:   local task/pointcloud/pointcloud*.tar.zst[.part-*] then in-place extract/delete
Remote complete:   require data/meta/pointcloud/videos/raw_lerobot_idx.jsonl before extract
Existing raw:      $(if [[ "$OVERWRITE_RAW" == "1" ]]; then echo overwrite; else echo keep; fi)
Existing pc:       $(if [[ "$OVERWRITE_POINTCLOUD" == "1" ]]; then echo overwrite; else echo skip non-empty pointcloud dirs; fi)
EOF
  exit 0
fi

for cmd in "$PYTHON_BIN" "$MODELSCOPE_BIN" tar "$ZSTD_BIN" flock; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Cannot find required command: $cmd" >&2
    exit 127
  fi
done

export REPO_ID DATE_ROOT DEST_DIR REVISION MAX_WORKERS STAGING_ROOT TASK_FILTER LOG_ROOT
export MODELSCOPE_BIN ZSTD_BIN
export FORCE_DOWNLOAD OVERWRITE_POINTCLOUD OVERWRITE_RAW KEEP_STAGE

"$PYTHON_BIN" - <<'PY'
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

from modelscope.hub.api import HubApi

repo_id = os.environ["REPO_ID"]
date_root_filter = os.environ["DATE_ROOT"].strip("/")
date_root = date_root_filter
dest_root = Path(os.environ["DEST_DIR"]).resolve()
revision = os.environ["REVISION"]
max_workers = os.environ["MAX_WORKERS"]
staging_root = Path(os.environ["STAGING_ROOT"]).resolve()
log_root_base = Path(os.environ["LOG_ROOT"]).resolve()
task_filter = os.environ.get("TASK_FILTER", "").strip("/")
force_download = os.environ["FORCE_DOWNLOAD"] == "1"
overwrite_pointcloud = os.environ["OVERWRITE_POINTCLOUD"] == "1"
overwrite_raw = os.environ["OVERWRITE_RAW"] == "1"
keep_stage = os.environ["KEEP_STAGE"] == "1"
modelscope_bin = os.environ["MODELSCOPE_BIN"]
zstd_bin = os.environ["ZSTD_BIN"]
token = os.environ.get("MODELSCOPE_API_TOKEN") or None

date_selection = date_root_filter or "<all>"
run_key = hashlib.sha1(f"{repo_id}@{revision}:{date_selection}:{dest_root}".encode()).hexdigest()[:16]
run_root = staging_root / "pointcloud_0331_sync" / run_key
locks_root = run_root / "locks"
markers_root = run_root / "markers"
logs_root = log_root_base / run_key
status_root = logs_root / "status"
events_log = logs_root / "events.tsv"
for directory in (locks_root, markers_root, logs_root, status_root):
    directory.mkdir(parents=True, exist_ok=True)
dest_root.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(message, flush=True)


def task_status_path(task_rel: str) -> Path:
    return status_root / (safe_key(task_rel) + ".status")


def append_event(task_rel: str, event: str, detail: str = "") -> None:
    safe_detail = detail.replace("\t", " ").replace("\n", " ")
    with open(events_log, "a", encoding="utf-8") as handle:
        handle.write(f"{task_rel}\t{event}\t{safe_detail}\n")


def write_task_status(task_rel: str, status: str, detail: str = "") -> None:
    task_status_path(task_rel).write_text(f"{status}\n{detail}\n", encoding="utf-8")
    append_event(task_rel, status, detail)


def read_status_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not status_root.exists():
        return counts
    for path in status_root.glob("*.status"):
        try:
            status = path.read_text(encoding="utf-8").splitlines()[0].strip()
        except Exception:
            status = "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def print_persistent_summary() -> None:
    counts = read_status_counts()
    wanted = ["downloaded", "extracted", "remote_incomplete", "download_failed", "extract_failed", "skipped_existing", "locked", "no_parts"]
    summary = " ".join(f"{key}={counts.get(key, 0)}" for key in wanted)
    log(f"[LOG] events={events_log}")
    log(f"[LOG] status_dir={status_root}")
    log(f"[PERSISTENT_SUMMARY] {summary}")


def die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(code)


def safe_rel_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value}")
    return str(path)


def safe_key(value: str) -> str:
    digest = hashlib.sha1(value.encode()).hexdigest()[:16]
    tail = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[-96:]
    return f"{digest}_{tail}"


def local_dir_has_files(path: Path) -> bool:
    if not path.exists():
        return False
    for _, _, files in os.walk(path):
        if files:
            return True
    return False


POINTCLOUD_ARCHIVE_RE = re.compile(r"^pointcloud(?:_[^.]*)?\.tar\.zst(?:\.part-\d+)?$")


def is_pointcloud_archive_file(path: Path | PurePosixPath) -> bool:
    return POINTCLOUD_ARCHIVE_RE.match(path.name) is not None


def local_pointcloud_exists(task_rel: str) -> bool:
    target = dest_root / task_rel / "pointcloud"
    if not target.exists():
        return False
    for _, dirs, files in os.walk(target):
        dirs[:] = [name for name in dirs if not name.startswith(".pointcloud_extract_tmp.")]
        if any(not is_pointcloud_archive_file(PurePosixPath(name)) for name in files):
            return True
    return False


REQUIRED_COMPLETE_CHILDREN = (
    "data",
    "meta",
    "pointcloud",
    "videos",
    "raw_lerobot_idx.jsonl",
)


def remote_task_completeness(task_rel: str, task_items: list[dict]) -> tuple[bool, list[str]]:
    task_path = PurePosixPath(task_rel)
    present: set[str] = set()
    for item in task_items:
        item_path = PurePosixPath(item.get("Path", ""))
        if item_path.parent == task_path:
            present.add(item_path.name)
    missing = [name for name in REQUIRED_COMPLETE_CHILDREN if name not in present]
    return not missing, missing


def missing_text(missing: list[str]) -> str:
    return ",".join(missing) if missing else "-"


def api_files(api: HubApi, root_path: str, recursive: bool = False, page_size: int = 500) -> list[dict]:
    files: list[dict] = []
    page_number = 1
    while True:
        page = api.get_dataset_files(
            repo_id=repo_id,
            revision=revision,
            root_path=root_path,
            recursive=recursive,
            page_number=page_number,
            page_size=page_size,
        )
        files.extend(page)
        if len(page) < page_size:
            break
        page_number += 1
    return files


def load_task_remote_state(api: HubApi, task_rel: str) -> dict:
    task_items = api_files(api, task_rel, recursive=False)
    remote_complete, remote_missing = remote_task_completeness(task_rel, task_items)
    raw_meta = next((item for item in task_items if item.get("Path") == f"{task_rel}/raw_lerobot_idx.jsonl"), None)

    has_pointcloud_dir = any(item.get("Path") == f"{task_rel}/pointcloud" for item in task_items)
    pc_items = api_files(api, f"{task_rel}/pointcloud", recursive=False) if has_pointcloud_dir else []
    parts = [
        item for item in pc_items
        if item.get("Type") == "blob" and is_pointcloud_archive_file(PurePosixPath(item.get("Path", "")))
    ]
    parts.sort(key=lambda item: item["Path"])
    return {
        "task_rel": task_rel,
        "raw_meta": raw_meta,
        "parts": parts,
        "remote_complete": remote_complete,
        "remote_missing": remote_missing,
    }


def task_lock(task_rel: str):
    lock_path = locks_root / (safe_key(task_rel) + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def marker_path(task_rel: str) -> Path:
    return markers_root / (safe_key(task_rel) + ".done")


def write_marker(task_rel: str, raw_status: str, pc_status: str, part_count: int) -> None:
    marker_path(task_rel).write_text(
        "\n".join([
            f"repo={repo_id}",
            f"revision={revision}",
            f"date={date_root}",
            f"dest={dest_root}",
            f"task={task_rel}",
            f"raw={raw_status}",
            f"pointcloud={pc_status}",
            f"parts={part_count}",
            f"pid={os.getpid()}",
        ]) + "\n"
    )


def local_repo_file(remote_path: str) -> Path:
    return dest_root / safe_rel_path(remote_path)


def local_file_matches_meta(meta: dict) -> bool:
    path = local_repo_file(meta["Path"])
    expected_size = int(meta.get("Size") or 0)
    if not path.exists():
        return False
    return expected_size <= 0 or path.stat().st_size == expected_size


def assert_single_task_download(task_rel: str, files: list[str]) -> None:
    prefix = task_rel.rstrip("/") + "/"
    mismatched = [path for path in files if not safe_rel_path(path).startswith(prefix)]
    if mismatched:
        raise RuntimeError(
            f"refusing to run one modelscope download across multiple tasks for {task_rel}: "
            + ", ".join(mismatched[:5])
        )


def run_modelscope_download(local_dir: Path, task_rel: str, files: list[str]) -> None:
    if not files:
        return
    assert_single_task_download(task_rel, files)
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        modelscope_bin, "download",
        "--repo-type", "dataset",
        "--revision", revision,
        "--local_dir", str(local_dir),
        "--max-workers", str(max_workers),
    ]
    if force_download:
        cmd.append("--force")
    cmd.extend([repo_id, *files])
    log(f"[MODELSCOPE_TASK] {task_rel}: one modelscope download command")
    log("[MODELSCOPE] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def verify_local_files(metas: list[dict]) -> None:
    for meta in metas:
        path = local_repo_file(meta["Path"])
        expected_size = int(meta.get("Size") or 0)
        if not path.exists():
            raise RuntimeError(f"missing downloaded file: {path}")
        if expected_size > 0 and path.stat().st_size != expected_size:
            raise RuntimeError(f"size mismatch: {path} got {path.stat().st_size}, expected {expected_size}")


def finalize_raw(raw_meta: dict, task_rel: str, needed_raw: bool) -> str:
    if raw_meta is None:
        return "missing_remote"
    target = dest_root / task_rel / "raw_lerobot_idx.jsonl"
    if target.exists():
        return "downloaded" if needed_raw else "exists"
    raise RuntimeError(f"raw file was not downloaded: {target}")


def extract_parts(part_metas: list[dict], task_rel: str) -> Path:
    part_paths = [local_repo_file(meta["Path"]) for meta in sorted(part_metas, key=lambda item: item["Path"])]
    if not part_paths:
        raise RuntimeError(f"no pointcloud parts for {task_rel}")
    missing = [str(path) for path in part_paths if not path.exists()]
    if missing:
        raise RuntimeError("missing pointcloud parts before extract: " + ", ".join(missing[:5]))

    task_dir = dest_root / task_rel
    task_extract_dir = task_dir / f".pointcloud_extract_tmp.{os.getpid()}"
    if task_extract_dir.exists():
        shutil.rmtree(task_extract_dir)
    task_extract_dir.mkdir(parents=True)

    log(f"[EXTRACT] {task_rel}: extracting {len(part_paths)} local parts")
    zstd_proc = subprocess.Popen([zstd_bin, "-d", "-q"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    tar_proc = subprocess.Popen(["tar", "-xf", "-", "-C", str(task_extract_dir)], stdin=zstd_proc.stdout)
    assert zstd_proc.stdin is not None
    assert zstd_proc.stdout is not None
    zstd_proc.stdout.close()
    try:
        for part in part_paths:
            with open(part, "rb") as handle:
                shutil.copyfileobj(handle, zstd_proc.stdin, length=1024 * 1024)
    finally:
        zstd_proc.stdin.close()
    zstd_rc = zstd_proc.wait()
    tar_rc = tar_proc.wait()
    if zstd_rc != 0 or tar_rc != 0:
        raise RuntimeError(f"extract failed for {task_rel}: zstd={zstd_rc}, tar={tar_rc}")
    return task_extract_dir


def cleanup_local_parts(part_metas: list[dict]) -> int:
    removed = 0
    for meta in part_metas:
        path = local_repo_file(meta["Path"])
        if path.exists() and is_pointcloud_archive_file(path):
            path.unlink()
            removed += 1
    return removed


def find_extracted_pointcloud(task_extract_dir: Path, task_rel: str) -> Path:
    expected = task_extract_dir / task_rel / "pointcloud"
    if expected.exists():
        return expected
    direct = task_extract_dir / "pointcloud"
    if direct.exists():
        return direct
    children = list(task_extract_dir.iterdir())
    if any(child.is_dir() and child.name.startswith("multi_sessions_") for child in children):
        return task_extract_dir
    candidates = [path for path in task_extract_dir.rglob("pointcloud") if path.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"cannot locate extracted pointcloud directory for {task_rel}")


def install_pointcloud(task_extract_dir: Path, task_rel: str) -> str:
    target = dest_root / task_rel / "pointcloud"
    if local_pointcloud_exists(task_rel) and not overwrite_pointcloud:
        return "exists"

    source = find_extracted_pointcloud(task_extract_dir, task_rel)
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        destination = target / child.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(str(child), str(destination))
    return "restored"


def process_task(task: dict) -> str:
    task_rel = safe_rel_path(task["task_rel"])
    lock_fd = task_lock(task_rel)
    if lock_fd is None:
        log(f"[LOCKED] {task_rel}: another process is handling it")
        write_task_status(task_rel, "locked", "another process is handling it")
        return "locked"

    try:
        raw_meta = task.get("raw_meta")
        parts = task["parts"]
        remote_complete = bool(task.get("remote_complete"))
        remote_missing = list(task.get("remote_missing") or [])
        target_raw = dest_root / task_rel / "raw_lerobot_idx.jsonl"
        target_pc = dest_root / task_rel / "pointcloud"
        marker = marker_path(task_rel)
        local_pc_exists = local_pointcloud_exists(task_rel)

        if marker.exists() and not local_pc_exists and not overwrite_pointcloud:
            log(f"[STALE_MARKER] {task_rel}: marker exists but local pointcloud is missing/empty")

        if marker.exists() and target_raw.exists() and local_pc_exists and not overwrite_raw and not overwrite_pointcloud and not force_download:
            log(f"[SKIP] {task_rel}: complete marker and local pointcloud exist")
            write_task_status(task_rel, "extracted", "complete marker and local pointcloud exist")
            return "skip_marker"

        need_raw = raw_meta is not None and (overwrite_raw or not target_raw.exists())
        need_pc = bool(parts) and (overwrite_pointcloud or not local_pc_exists)
        if not need_pc and not local_pc_exists and not parts:
            latest_task = load_task_remote_state(api, task_rel)
            if latest_task["parts"]:
                raw_meta = latest_task.get("raw_meta") or raw_meta
                parts = latest_task["parts"]
                remote_complete = bool(latest_task["remote_complete"])
                remote_missing = list(latest_task["remote_missing"])
                need_raw = raw_meta is not None and (overwrite_raw or not target_raw.exists())
                need_pc = overwrite_pointcloud or not local_pc_exists
                log(f"[REMOTE_REFRESH] {task_rel}: found {len(parts)} pointcloud parts after refresh")
        if not need_raw and not need_pc:
            if local_pc_exists:
                write_marker(task_rel, "exists", "exists", len(parts))
                write_task_status(task_rel, "skipped_existing", "raw and pointcloud already exist")
                log(f"[SKIP] {task_rel}: raw and pointcloud already exist")
                return "skip_existing"
            latest_task = load_task_remote_state(api, task_rel)
            if not latest_task["remote_complete"]:
                missing = missing_text(latest_task["remote_missing"])
                write_task_status(task_rel, "remote_incomplete", f"missing={missing} parts={len(parts)}")
                log(f"[WAIT_REMOTE_COMPLETE] {task_rel}: missing={missing}; skip extract")
                return "remote_incomplete"
            write_task_status(task_rel, "no_parts", "remote complete but no pointcloud archive parts")
            log(f"[NO_PARTS] {task_rel}: remote complete but no pointcloud archive parts")
            return "no_parts"

        if need_pc and overwrite_pointcloud and target_pc.exists():
            shutil.rmtree(target_pc)

        files_to_download: list[str] = []
        metas_to_verify: list[dict] = []
        if need_raw:
            metas_to_verify.append(raw_meta)
            if overwrite_raw or force_download or not local_file_matches_meta(raw_meta):
                files_to_download.append(raw_meta["Path"])
        if need_pc:
            metas_to_verify.extend(parts)
            for meta in parts:
                if force_download or not local_file_matches_meta(meta):
                    files_to_download.append(meta["Path"])

        if files_to_download:
            log(f"[TASK] {task_rel}: downloading {len(files_to_download)} files to local task path with ModelScope progress")
        else:
            log(f"[TASK] {task_rel}: all required archive files already exist locally; extracting")
        try:
            run_modelscope_download(dest_root, task_rel, files_to_download)
            verify_local_files(metas_to_verify)
        except Exception as exc:
            write_task_status(task_rel, "download_failed", str(exc))
            raise
        write_task_status(task_rel, "downloaded", f"files={len(files_to_download)} parts={len(parts)}")

        raw_status = finalize_raw(raw_meta, task_rel, need_raw) if raw_meta is not None else "missing_remote"
        if need_pc:
            latest_task = load_task_remote_state(api, task_rel)
            if not latest_task["remote_complete"]:
                missing = missing_text(latest_task["remote_missing"])
                write_task_status(task_rel, "remote_incomplete", f"missing={missing} downloaded_parts={len(parts)}")
                log(f"[WAIT_REMOTE_COMPLETE] {task_rel}: missing={missing}; downloaded parts kept locally; skip extract")
                return "remote_incomplete"

            latest_raw_meta = latest_task.get("raw_meta")
            latest_parts = latest_task["parts"]
            final_files_to_download: list[str] = []
            final_metas_to_verify: list[dict] = []
            if latest_raw_meta is not None:
                final_metas_to_verify.append(latest_raw_meta)
                if overwrite_raw or force_download or not local_file_matches_meta(latest_raw_meta):
                    final_files_to_download.append(latest_raw_meta["Path"])
            final_metas_to_verify.extend(latest_parts)
            for meta in latest_parts:
                if force_download or not local_file_matches_meta(meta):
                    final_files_to_download.append(meta["Path"])

            if final_files_to_download:
                log(f"[TASK] {task_rel}: remote complete; downloading {len(final_files_to_download)} final/missing files")
                run_modelscope_download(dest_root, task_rel, final_files_to_download)
            verify_local_files(final_metas_to_verify)

            if latest_raw_meta is not None:
                raw_status = finalize_raw(latest_raw_meta, task_rel, need_raw or raw_meta is None)
            parts = latest_parts
            if not parts:
                pc_status = "no_remote_parts"
            else:
                extracted_dir = None
                try:
                    extracted_dir = extract_parts(parts, task_rel)
                    pc_status = install_pointcloud(extracted_dir, task_rel)
                except Exception as exc:
                    write_task_status(task_rel, "extract_failed", str(exc))
                    raise
                finally:
                    if extracted_dir is not None:
                        shutil.rmtree(extracted_dir, ignore_errors=True)
                if not keep_stage:
                    removed_parts = cleanup_local_parts(parts)
                    log(f"[CLEANUP] {task_rel}: removed {removed_parts} local archive parts")
        else:
            if not local_pointcloud_exists(task_rel):
                latest_task = load_task_remote_state(api, task_rel)
                if not latest_task["remote_complete"]:
                    missing = missing_text(latest_task["remote_missing"])
                    write_task_status(task_rel, "remote_incomplete", f"missing={missing} parts={len(parts)}")
                    log(f"[WAIT_REMOTE_COMPLETE] {task_rel}: missing={missing}; skip extract")
                    return "remote_incomplete"
            pc_status = "exists" if local_pointcloud_exists(task_rel) else "no_remote_parts"
            if pc_status == "exists" and not keep_stage:
                removed_parts = cleanup_local_parts(parts)
                if removed_parts:
                    log(f"[CLEANUP] {task_rel}: removed {removed_parts} stale local archive parts")

        write_marker(task_rel, raw_status, pc_status, len(parts))
        if pc_status == "restored":
            write_task_status(task_rel, "extracted", f"raw={raw_status} parts={len(parts)}")
        elif pc_status == "exists":
            write_task_status(task_rel, "skipped_existing", f"raw={raw_status} parts={len(parts)}")
        else:
            write_task_status(task_rel, "no_parts", f"raw={raw_status} parts={len(parts)}")
        log(f"[DONE] {task_rel}: raw={raw_status} pointcloud={pc_status}")
        return "done"
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


log(f"[INFO] repo={repo_id} revision={revision} date_selection={date_selection}")
log(f"[INFO] dest={dest_root}")
log(f"[INFO] state={run_root}")
log(f"[INFO] download_local_dir={dest_root}")
log(f"[INFO] logs={logs_root}")

try:
    api = HubApi(token=token)
    if date_root_filter:
        date_roots = [date_root_filter]
    else:
        root_items = api_files(api, "", recursive=False)
        date_roots = sorted(item["Path"] for item in root_items if item.get("Type") == "tree")
except Exception as exc:
    die("[ERROR] Failed to read ModelScope dataset files. Check proxy, network, and `modelscope login`.\n" + f"Detail: {exc}")

counts: dict[str, int] = {}
matched_any_task = False
for current_date_root in date_roots:
    date_root = current_date_root
    try:
        date_items = api_files(api, date_root, recursive=False)
    except Exception as exc:
        die(f"[ERROR] Failed to read ModelScope dataset date root {date_root}.\nDetail: {exc}")

    task_paths = [item["Path"] for item in date_items if item.get("Type") == "tree"]
    if task_filter:
        task_paths = [
            path for path in task_paths
            if path == task_filter or path.endswith("/" + task_filter) or PurePosixPath(path).name == task_filter
        ]
    if not task_paths:
        if date_root_filter:
            die(f"[ERROR] No tasks found under remote {date_root} matching filter {task_filter or '<all>'}")
        log(f"[INFO] no tasks under {date_root} matching filter {task_filter or '<all>'}; skip")
        continue

    matched_any_task = True
    tasks: list[dict] = []
    for task_rel in sorted(task_paths):
        tasks.append(load_task_remote_state(api, task_rel))

    log(f"[INFO] found {len(tasks)} tasks under {date_root}")
    for task in tasks:
        raw_state = "yes" if task["raw_meta"] is not None else "no"
        complete_state = "yes" if task["remote_complete"] else "no"
        missing = missing_text(task["remote_missing"])
        log(f"[PLAN] {task['task_rel']}: raw={raw_state} pointcloud_parts={len(task['parts'])} remote_complete={complete_state} missing={missing}")

    for task in tasks:
        try:
            status = process_task(task)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            write_task_status(task["task_rel"], "download_failed", f"modelscope exited with code {exc.returncode}")
            log(f"[FAILED] {task['task_rel']}: modelscope exited with code {exc.returncode}")
        except Exception as exc:
            status = "failed"
            if not task_status_path(task["task_rel"]).exists():
                write_task_status(task["task_rel"], "download_failed", str(exc))
            log(f"[FAILED] {task['task_rel']}: {exc}")
        counts[status] = counts.get(status, 0) + 1

if not matched_any_task:
    die(f"[ERROR] No tasks found under remote date roots matching filter {task_filter or '<all>'}")

log("[SUMMARY] " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
print_persistent_summary()
if counts.get("failed", 0):
    raise SystemExit(1)
PY
