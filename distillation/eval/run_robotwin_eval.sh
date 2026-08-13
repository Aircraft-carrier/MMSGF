#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash distillation/eval/run_robotwin_eval.sh \
    <checkpoint_root> <dataset_root> <model_root> <run_output_dir> \
    [task|all|tasks.json] [clean|random] [gpu_id] [base_port]

Defaults:
  policy repo:     /zsh/code/MMSGF
  RoboTwin repo:   /zsh/benckmark/RoboTwin
  policy Python:   /zsh/miniconda3/envs/linbotva/bin/python
  RoboTwin Python: /zsh/miniconda3/envs/robotwin2/bin/python

Useful environment variables:
  PARA_NUM_PER_GPU=1 EVAL_NUM_EPISODES=100 TASK_MAX_RETRIES=2
  CLEAN_TASK_CONFIG=eval_mv_clean_both RANDOM_TASK_CONFIG=demo_randomized
  TASK_WALL_TIMEOUT_SECONDS=0 MP4_STALL_TIMEOUT_SECONDS=600
  RESUME_LATEST_RUN_DATE=true RESUME_INCOMPLETE_TASK_PROGRESS=true
  RUN_DATE=20260813_120000 INSTRUCTION_TYPE=unseen
  SERVER_EXTRA_ARGS="--dtype bfloat16 --prediction-chunks 1"
EOF
}

if [[ $# -lt 4 ]]; then
  usage
  exit 1
fi

CHECKPOINT_ROOT=$1
DATASET_ROOT=$2
MODEL_ROOT=$3
RUN_OUTPUT_DIR=$4
TASK_SELECTOR=${5:-all}
ENVIRONMENT=${6:-clean}
GPU_ID=${7:-0}
BASE_PORT=${8:-8000}

POLICY_REPO=${POLICY_REPO:-/zsh/code/MMSGF}
ROBOTWIN_REPO=${ROBOTWIN_REPO:-/zsh/benckmark/RoboTwin}
POLICY_PYTHON=${POLICY_PYTHON:-/zsh/miniconda3/envs/linbotva/bin/python}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/zsh/miniconda3/envs/robotwin2/bin/python}
POLICY_SERVER_ENTRYPOINT=${POLICY_SERVER_ENTRYPOINT:-distillation.eval.server}
ROBOTWIN_CLIENT_ENTRYPOINT=${ROBOTWIN_CLIENT_ENTRYPOINT:-${POLICY_REPO}/distillation/eval/robotwin_client.py}
SERVER_BIND_HOST=${SERVER_BIND_HOST:-127.0.0.1}
CLIENT_HOST=${CLIENT_HOST:-127.0.0.1}
PARA_NUM_PER_GPU=${PARA_NUM_PER_GPU:-1}
EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES:-100}
TASK_MAX_RETRIES=${TASK_MAX_RETRIES:-2}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-1800}
TASK_WALL_TIMEOUT_SECONDS=${TASK_WALL_TIMEOUT_SECONDS:-0}
MP4_STALL_TIMEOUT_SECONDS=${MP4_STALL_TIMEOUT_SECONDS:-600}
WATCHDOG_INTERVAL_SECONDS=${WATCHDOG_INTERVAL_SECONDS:-30}
PROCESS_CLEANUP_GRACE_SECONDS=${PROCESS_CLEANUP_GRACE_SECONDS:-5}
RESUME_LATEST_RUN_DATE=${RESUME_LATEST_RUN_DATE:-true}
RESUME_INCOMPLETE_TASK_PROGRESS=${RESUME_INCOMPLETE_TASK_PROGRESS:-true}
CONTINUE_ON_TASK_FAILURE=${CONTINUE_ON_TASK_FAILURE:-true}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-unseen}
SEED=${SEED:-0}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-}
CLIENT_EXTRA_ARGS=${CLIENT_EXTRA_ARGS:-}

case "${ENVIRONMENT}" in
  clean)
    TASK_CONFIG=${TASK_CONFIG:-${CLEAN_TASK_CONFIG:-eval_mv_clean_both}}
    ;;
  random)
    TASK_CONFIG=${TASK_CONFIG:-${RANDOM_TASK_CONFIG:-demo_randomized}}
    ;;
  *)
    echo "Environment must be clean or random, got: ${ENVIRONMENT}" >&2
    exit 1
    ;;
esac

RESULT_FILE_NAME="_result_${ENVIRONMENT}.txt"
PROGRESS_FILE_NAME="_progress_${ENVIRONMENT}.json"
FAILED_FILE_NAME="_timeout_or_failed_${ENVIRONMENT}.txt"

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

absolute_path() {
  "${ROBOTWIN_PYTHON}" - "$1" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

validate_inputs() {
  local task_config_path
  task_config_path="${ROBOTWIN_REPO}/task_config/${TASK_CONFIG}"
  if [[ "${task_config_path}" != *.yml && "${task_config_path}" != *.yaml ]]; then
    task_config_path="${task_config_path}.yml"
  fi
  [[ -d "${POLICY_REPO}" ]] || { echo "Policy repo not found: ${POLICY_REPO}" >&2; exit 1; }
  [[ -d "${ROBOTWIN_REPO}" ]] || { echo "RoboTwin repo not found: ${ROBOTWIN_REPO}" >&2; exit 1; }
  [[ -x "${POLICY_PYTHON}" ]] || { echo "Policy Python not found: ${POLICY_PYTHON}" >&2; exit 1; }
  [[ -x "${ROBOTWIN_PYTHON}" ]] || { echo "RoboTwin Python not found: ${ROBOTWIN_PYTHON}" >&2; exit 1; }
  [[ -f "${CHECKPOINT_ROOT}/checkpoint_metadata.json" ]] || { echo "Checkpoint metadata not found under ${CHECKPOINT_ROOT}" >&2; exit 1; }
  [[ -f "${DATASET_ROOT}/meta/mot_config.json" ]] || { echo "Dataset mot_config.json not found under ${DATASET_ROOT}" >&2; exit 1; }
  [[ -d "${MODEL_ROOT}" ]] || { echo "Model root not found: ${MODEL_ROOT}" >&2; exit 1; }
  [[ -f "${ROBOTWIN_CLIENT_ENTRYPOINT}" ]] || { echo "RoboTwin client not found: ${ROBOTWIN_CLIENT_ENTRYPOINT}" >&2; exit 1; }
  [[ -f "${task_config_path}" ]] || { echo "Task config not found: ${task_config_path}" >&2; exit 1; }
  [[ "${PARA_NUM_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo "PARA_NUM_PER_GPU must be positive" >&2; exit 1; }
  [[ "${EVAL_NUM_EPISODES}" =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_NUM_EPISODES must be positive" >&2; exit 1; }
}

export_runtime_env() {
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
  export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-8}"
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
  export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
  export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
}

load_tasks() {
  local selector=$1
  if [[ "${selector}" == "all" ]]; then
    "${ROBOTWIN_PYTHON}" - "${ROBOTWIN_REPO}/task_config/_eval_step_limit.yml" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    for task in yaml.safe_load(handle):
        print(task)
PY
  elif [[ -f "${selector}" && "${selector}" == *.json ]]; then
    "${ROBOTWIN_PYTHON}" - "${selector}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    items = json.load(handle)
if not isinstance(items, list):
    raise ValueError("task JSON must contain a list")
for item in items:
    print(item if isinstance(item, str) else item["task"])
PY
  else
    printf '%s\n' "${selector}"
  fi
}

latest_run_date() {
  find "${RUN_OUTPUT_DIR}/${ENVIRONMENT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
    | grep -E '^[0-9]{8}_[0-9]{6}$' | sort | tail -n 1
}

update_summary() {
  "${ROBOTWIN_PYTHON}" - "${RUN_CONFIG_DIR}" "${ENVIRONMENT}" "${TASK_CONFIG}" \
    "${RUN_DATE}" "${CHECKPOINT_ROOT}" "${EVAL_NUM_EPISODES}" "${ALL_TASKS[@]}" <<'PY'
import fcntl, json, os, re, sys
from datetime import datetime
from pathlib import Path

run_dir = Path(sys.argv[1])
environment, task_config, run_date, checkpoint = sys.argv[2:6]
episodes = int(sys.argv[6])
expected_tasks = sys.argv[7:]
number = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

def result_rate(path):
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        matches = number.findall(line)
        if matches:
            return float(matches[-1])
    raise ValueError(f"no numeric result in {path}")

run_dir.mkdir(parents=True, exist_ok=True)
with (run_dir / ".summary.lock").open("w") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    tasks, rates = {}, []
    counts = {"completed": 0, "failed_or_timeout": 0, "running_or_incomplete": 0, "pending": 0}
    for task in expected_tasks:
        task_dir = run_dir / task
        result = task_dir / f"_result_{environment}.txt"
        progress = task_dir / f"_progress_{environment}.json"
        failed = task_dir / f"_timeout_or_failed_{environment}.txt"
        status, rate = "pending", None
        if result.is_file():
            status, rate = "completed", result_rate(result)
            rates.append(rate)
        elif failed.is_file():
            status = "failed_or_timeout"
        elif task_dir.exists():
            status = "running_or_incomplete"
        counts[status] += 1
        tasks[task] = {
            "status": status,
            "success_rate": rate,
            "result_file": str(result),
            "progress_file": str(progress),
            "timeout_or_failed_file": str(failed),
        }
    summary = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir), "run_date": run_date, "environment": environment,
        "task_config": task_config, "checkpoint": checkpoint,
        "eval_num_episodes": episodes, "expected_tasks": len(expected_tasks),
        "completed_tasks": counts["completed"],
        "failed_or_timeout_tasks": counts["failed_or_timeout"],
        "running_or_incomplete_tasks": counts["running_or_incomplete"],
        "pending_tasks": counts["pending"],
        "average_success_rate": None if not rates else sum(rates) / len(rates),
        "tasks": tasks,
    }
    temporary = run_dir / f".summary.json.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, run_dir / "summary.json")
PY
}

skip_timed_out_seed() {
  local progress=$1 task=$2
  [[ -f "${progress}" ]] || return 0
  "${ROBOTWIN_PYTHON}" - "${progress}" "${task}" <<'PY'
import json, os, sys
from datetime import datetime
from pathlib import Path

path, task = Path(sys.argv[1]), sys.argv[2]
progress = json.loads(path.read_text(encoding="utf-8"))
if progress.get("task_name") != task:
    raise ValueError(f"progress task mismatch in {path}")
seed = int(progress["next_seed"])
progress.setdefault("skipped_timeout_seeds", []).append(seed)
progress["next_seed"] = seed + 1
progress["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

wait_for_server() {
  local port=$1
  "${POLICY_PYTHON}" - "${CLIENT_HOST}" "${port}" "${SERVER_READY_TIMEOUT}" <<'PY'
import http.client, sys, time
host, port, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline, error = time.time() + timeout, None
while time.time() < deadline:
    try:
        connection = http.client.HTTPConnection(host, port, timeout=2)
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        response.read()
        connection.close()
        if response.status == 200:
            raise SystemExit(0)
    except Exception as exc:
        error = exc
    time.sleep(2)
raise SystemExit(f"server {host}:{port} was not ready before timeout: {error}")
PY
}

kill_tree() {
  local pid=$1 signal=${2:-TERM} child
  [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1 || return 0
  for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
    kill_tree "${child}" "${signal}"
  done
  kill "-${signal}" "${pid}" >/dev/null 2>&1 || true
}

clean_pid() {
  local pid=$1 label=$2
  [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1 || return 0
  echo "Stopping ${label} pid=${pid}"
  kill_tree "${pid}" TERM
  local deadline=$(( $(date +%s) + PROCESS_CLEANUP_GRACE_SECONDS ))
  while ps -p "${pid}" >/dev/null 2>&1 && (( $(date +%s) < deadline )); do sleep 1; done
  if ps -p "${pid}" >/dev/null 2>&1; then kill_tree "${pid}" KILL; fi
  wait "${pid}" >/dev/null 2>&1 || true
}

start_server() {
  local shard=$1 port=$2
  local -a extra=()
  if [[ -n "${SERVER_EXTRA_ARGS}" ]]; then read -r -a extra <<< "${SERVER_EXTRA_ARGS}"; fi
  (
    cd "${POLICY_REPO}"
    export_runtime_env
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export PYTHONPATH="${POLICY_REPO}:${PYTHONPATH:-}"
    exec "${POLICY_PYTHON}" -u -m "${POLICY_SERVER_ENTRYPOINT}" \
      --checkpoint-root "${CHECKPOINT_ROOT}" --dataset-root "${DATASET_ROOT}" \
      --model-root "${MODEL_ROOT}" --host "${SERVER_BIND_HOST}" --port "${port}" \
      --device cuda:0 "${extra[@]}"
  ) >"${RUN_CONFIG_DIR}/server_shard_${shard}.log" 2>&1 &
  STARTED_SERVER_PID=$!
  echo "Started server shard=${shard} pid=${STARTED_SERVER_PID} port=${port}"
}

run_client() {
  local port=$1 task=$2 task_dir=$3 progress=$4
  local -a extra=()
  if [[ -n "${CLIENT_EXTRA_ARGS}" ]]; then read -r -a extra <<< "${CLIENT_EXTRA_ARGS}"; fi
  (
    cd "${ROBOTWIN_REPO}"
    export_runtime_env
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export PYTHONPATH="${ROBOTWIN_REPO}:${POLICY_REPO}:${PYTHONPATH:-}"
    exec "${ROBOTWIN_PYTHON}" -u "${ROBOTWIN_CLIENT_ENTRYPOINT}" \
      --robotwin-repo "${ROBOTWIN_REPO}" --host "${CLIENT_HOST}" --port "${port}" \
      --task-name "${task}" --task-config "${TASK_CONFIG}" \
      --eval-num-episodes "${EVAL_NUM_EPISODES}" --eval-output-dir "${task_dir}" \
      --result-file-name "${RESULT_FILE_NAME}" \
      --resume-from-progress "${RESUME_INCOMPLETE_TASK_PROGRESS}" \
      --resume-progress-path "${progress}" --seed "${SEED}" \
      --instruction-type "${INSTRUCTION_TYPE}" "${extra[@]}"
  ) >"${task_dir}/client.log" 2>&1 &
  CLIENT_PID=$!
}

wait_with_watchdog() {
  local pid=$1 task=$2 task_dir=$3
  local started now latest_mtime process_state
  started=$(date +%s)
  while true; do
    process_state=$(ps -o stat= -p "${pid}" 2>/dev/null || true)
    case "${process_state}" in
      ""|Z*) break ;;
    esac
    now=$(date +%s)
    if (( TASK_WALL_TIMEOUT_SECONDS > 0 && now - started > TASK_WALL_TIMEOUT_SECONDS )); then
      echo "Task wall timeout: ${task}" >&2
      clean_pid "${pid}" "client:${task}"
      return 124
    fi
    if (( MP4_STALL_TIMEOUT_SECONDS > 0 )); then
      latest_mtime=$(find "${task_dir}" -maxdepth 1 -type f -name 'episode*.mp4' -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1 | cut -d. -f1)
      latest_mtime=${latest_mtime:-${started}}
      if (( now - latest_mtime > MP4_STALL_TIMEOUT_SECONDS )); then
        echo "Task MP4 stall timeout: ${task}" >&2
        clean_pid "${pid}" "client:${task}"
        return 124
      fi
    fi
    sleep "${WATCHDOG_INTERVAL_SECONDS}"
  done
  wait "${pid}"
}

validate_inputs
POLICY_REPO=$(absolute_path "${POLICY_REPO}")
ROBOTWIN_REPO=$(absolute_path "${ROBOTWIN_REPO}")
CHECKPOINT_ROOT=$(absolute_path "${CHECKPOINT_ROOT}")
DATASET_ROOT=$(absolute_path "${DATASET_ROOT}")
MODEL_ROOT=$(absolute_path "${MODEL_ROOT}")
ROBOTWIN_CLIENT_ENTRYPOINT=$(absolute_path "${ROBOTWIN_CLIENT_ENTRYPOINT}")
mkdir -p "${RUN_OUTPUT_DIR}"
RUN_OUTPUT_DIR=$(absolute_path "${RUN_OUTPUT_DIR}")

if [[ -z "${RUN_DATE:-}" ]]; then
  if is_true "${RESUME_LATEST_RUN_DATE}"; then RUN_DATE=$(latest_run_date || true); fi
  RUN_DATE=${RUN_DATE:-$(date '+%Y%m%d_%H%M%S')}
fi
RUN_CONFIG_DIR="${RUN_OUTPUT_DIR}/${ENVIRONMENT}/${RUN_DATE}"
mkdir -p "${RUN_CONFIG_DIR}"

mapfile -t ALL_TASKS < <(load_tasks "${TASK_SELECTOR}")
[[ ${#ALL_TASKS[@]} -gt 0 ]] || { echo "No task selected" >&2; exit 1; }
TASKS=()
for task in "${ALL_TASKS[@]}"; do
  [[ -f "${RUN_CONFIG_DIR}/${task}/${RESULT_FILE_NAME}" ]] || TASKS+=("${task}")
done
update_summary
if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "All ${#ALL_TASKS[@]} tasks already completed: ${RUN_CONFIG_DIR}/summary.json"
  exit 0
fi

SERVER_PIDS=()
WORKER_PIDS=()
PORTS=()
CLEANUP_RUNNING=0
cleanup() {
  [[ ${CLEANUP_RUNNING} -eq 0 ]] || return 0
  CLEANUP_RUNNING=1
  trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]:-}"; do clean_pid "${pid}" task-worker; done
  for pid in "${SERVER_PIDS[@]:-}"; do clean_pid "${pid}" policy-server; done
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

for (( shard=0; shard<PARA_NUM_PER_GPU; shard++ )); do
  PORTS[shard]=$(( BASE_PORT + shard ))
  start_server "${shard}" "${PORTS[shard]}"
  SERVER_PIDS[shard]=${STARTED_SERVER_PID}
done
for (( shard=0; shard<PARA_NUM_PER_GPU; shard++ )); do
  wait_for_server "${PORTS[shard]}"
done

run_shard() {
  local shard=$1 total=$2
  local size=$(( (${#TASKS[@]} + total - 1) / total ))
  local begin=$(( shard * size )) end=$(( begin + size ))
  (( begin < ${#TASKS[@]} )) || return 0
  (( end <= ${#TASKS[@]} )) || end=${#TASKS[@]}
  local task task_dir result progress failed attempt status
  for (( index=begin; index<end; index++ )); do
    task=${TASKS[index]}
    task_dir="${RUN_CONFIG_DIR}/${task}"
    result="${task_dir}/${RESULT_FILE_NAME}"
    progress="${task_dir}/${PROGRESS_FILE_NAME}"
    failed="${task_dir}/${FAILED_FILE_NAME}"
    mkdir -p "${task_dir}"
    rm -f "${failed}"
    update_summary
    status=1
    for (( attempt=0; attempt<=TASK_MAX_RETRIES; attempt++ )); do
      echo "$(date '+%F %T') START shard=${shard} task=${task} attempt=$((attempt + 1))"
      run_client "${PORTS[shard]}" "${task}" "${task_dir}" "${progress}"
      set +e
      wait_with_watchdog "${CLIENT_PID}" "${task}" "${task_dir}"
      status=$?
      set -e
      [[ ${status} -eq 0 && -f "${result}" ]] && break
      if [[ ${status} -eq 124 ]]; then skip_timed_out_seed "${progress}" "${task}"; fi
      echo "$(date '+%F %T') RETRY task=${task} status=${status}" >&2
    done
    if [[ ! -f "${result}" ]]; then
      printf 'Timestamp: %s\nTask: %s\nLast Status: %s\nAttempts: %s\n' \
        "$(date '+%F %T')" "${task}" "${status}" "$((TASK_MAX_RETRIES + 1))" >"${failed}"
      update_summary
      is_true "${CONTINUE_ON_TASK_FAILURE}" && continue
      return 1
    fi
    rm -f "${failed}"
    echo "$(date '+%F %T') DONE task=${task} result=${result}"
    update_summary
  done
}

for (( shard=0; shard<PARA_NUM_PER_GPU; shard++ )); do
  run_shard "${shard}" "${PARA_NUM_PER_GPU}" &
  WORKER_PIDS[shard]=$!
done

failed=0
for shard in "${!WORKER_PIDS[@]}"; do
  wait "${WORKER_PIDS[shard]}" || failed=1
done
update_summary
echo "Evaluation summary: ${RUN_CONFIG_DIR}/summary.json"
exit "${failed}"
