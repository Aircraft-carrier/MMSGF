#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="/zsh/miniconda3/envs/linbotva/bin/python"
ROBOTWIN_PYTHON_BIN="/zsh/miniconda3/envs/robotwin2/bin/python"
CHECKPOINT_ROOT="${REPO_ROOT}/train_logs/robotwin_mot/0816_robotwin50_4gpu_run5/checkpoints/checkpoint_step_18000"
DATASET_ROOT="${REPO_ROOT}/data/robotwin_clean_50"
MODEL_ROOT="/zsh/cache/hf_cache/hub/models--robbyant--lingbot-va-base/snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c"

TASK_NAME="${TASK_NAME:-click_bell}"
GPU_ID="${GPU_ID:-0}"
BASE_PORT="${BASE_PORT:-8000}"
RUN_OUTPUT_DIR="${REPO_ROOT}/inference_logs/robotwin_eval_step18000_single_task/${TASK_NAME}"

export POLICY_PYTHON="${PYTHON_BIN}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON_BIN}"
export EVAL_NUM_EPISODES=1
export PARA_NUM_PER_GPU=1
export TASK_MAX_RETRIES=0
export CONTINUE_ON_TASK_FAILURE=false
export RESUME_LATEST_RUN_DATE=false
export RESUME_INCOMPLETE_TASK_PROGRESS=true
export RUN_DATE="${RUN_DATE:-$(date '+%Y%m%d_%H%M%S')}"
export TASK_WALL_TIMEOUT_SECONDS=7200
export MP4_STALL_TIMEOUT_SECONDS=600
export SERVER_EXTRA_ARGS="--video-num-steps 25 --action-num-steps 50 --execution-action-count 16"
export CLIENT_EXTRA_ARGS="--save-predicted-video true"

exec bash "${REPO_ROOT}/inference/eval/run_robotwin_eval.sh" \
  "${CHECKPOINT_ROOT}" \
  "${DATASET_ROOT}" \
  "${MODEL_ROOT}" \
  "${RUN_OUTPUT_DIR}" \
  "${TASK_NAME}" \
  clean "${GPU_ID}" "${BASE_PORT}"
