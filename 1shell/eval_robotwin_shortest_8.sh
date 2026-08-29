#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT_ROOT="${REPO_ROOT}/train_logs/robotwin_mot/0816_robotwin50_4gpu_run5/checkpoints/checkpoint_step_18000"
DATASET_ROOT="${REPO_ROOT}/data/robotwin_clean_50"
MODEL_ROOT="/zsh/cache/hf_cache/hub/models--robbyant--lingbot-va-base/snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c"
TASKS_JSON="${REPO_ROOT}/inference/eval/robotwin_shortest_8_tasks.json"
RUN_OUTPUT_DIR="${REPO_ROOT}/inference_logs/robotwin_eval_step18000_shortest8"
ROBOTWIN_PYTHON="/zsh/miniconda3/envs/robotwin2/bin/python"

export EVAL_NUM_EPISODES=10
export PARA_NUM_PER_GPU=2
export TASK_MAX_RETRIES=2
export CONTINUE_ON_TASK_FAILURE=true
export RESUME_LATEST_RUN_DATE=false
export RESUME_INCOMPLETE_TASK_PROGRESS=true
export RUN_DATE="${RUN_DATE:-$(date '+%Y%m%d_%H%M%S')}"
export TASK_WALL_TIMEOUT_SECONDS=7200
export MP4_STALL_TIMEOUT_SECONDS=600
export SERVER_EXTRA_ARGS="--video-num-steps 25 --action-num-steps 50 --execution-action-count 16"

LAUNCH_LOG_DIR="${RUN_OUTPUT_DIR}/launcher_logs/${RUN_DATE}"
SHARD_DIR="${LAUNCH_LOG_DIR}/task_shards"
mkdir -p "${SHARD_DIR}"

"${ROBOTWIN_PYTHON}" - "${TASKS_JSON}" "${SHARD_DIR}" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1], encoding="utf-8") as handle:
    tasks = json.load(handle)
if len(tasks) != 8:
    raise ValueError(f"expected 8 tasks, got {len(tasks)}")

output_dir = Path(sys.argv[2])
for gpu_id in range(4):
    with (output_dir / f"gpu{gpu_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(tasks[gpu_id * 2 : gpu_id * 2 + 2], handle, indent=2)
        handle.write("\n")
PY

PIDS=()

cleanup() {
  trap - INT TERM
  if (( ${#PIDS[@]} > 0 )); then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit 130
}
trap cleanup INT TERM

for gpu_id in 0 1 2 3; do
  base_port=$((8000 + gpu_id * 10))
  gpu_output_dir="${RUN_OUTPUT_DIR}/gpu${gpu_id}"
  task_shard="${SHARD_DIR}/gpu${gpu_id}.json"
  launcher_log="${LAUNCH_LOG_DIR}/gpu${gpu_id}.log"

  echo "GPU ${gpu_id}: tasks=${task_shard}, ports=${base_port}-$((base_port + 1)), log=${launcher_log}"
  bash "${REPO_ROOT}/inference/eval/run_robotwin_eval.sh" \
    "${CHECKPOINT_ROOT}" \
    "${DATASET_ROOT}" \
    "${MODEL_ROOT}" \
    "${gpu_output_dir}" \
    "${task_shard}" \
    clean "${gpu_id}" "${base_port}" \
    >"${launcher_log}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "Evaluation logs: ${LAUNCH_LOG_DIR}"
exit "${status}"
