#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MOT_DATASET_ROOT="${MOT_DATASET_ROOT:-/workspace/code/lingbot-va/data/0715_3k_subset_final_train}"
export MOT_OPTIMIZATION_COMPOSITION="g"
export MOT_MAX_VIEWS_PER_GPU="${MOT_MAX_VIEWS_PER_GPU:-4}"
export MOT_EVAL_WITH_CPU="${MOT_EVAL_WITH_CPU:-1}"
export MOT_EVAL_MODE="geometry"

export MOT_NUM_STEPS="${MOT_NUM_STEPS:-1000000}"
export MOT_SAVE_INTERVAL="${MOT_SAVE_INTERVAL:-2000}"
export MOT_LOG_INTERVAL="${MOT_LOG_INTERVAL:-100}"

export NGPU=4
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/train_logs/umi_subset3k_newmot_geometry_only/${RUN_STAMP}}"

exec bash "${SCRIPT_DIR}/train_mot_mixed_4gpu.sh" "$@"
