#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGE1_CHECKPOINT="${REPO_ROOT}/train_logs/umi_subset3k_newmot_geometry_only/0723_001712/checkpoints/checkpoint_step_60000"

for required_file in \
    "${STAGE1_CHECKPOINT}/_SUCCESS" \
    "${STAGE1_CHECKPOINT}/checkpoint_metadata.json" \
    "${STAGE1_CHECKPOINT}/transformer/config.json" \
    "${STAGE1_CHECKPOINT}/transformer/diffusion_pytorch_model.safetensors"
do
    if [ ! -f "${required_file}" ]; then
        echo "Missing stage-1 checkpoint file: ${required_file}" >&2
        exit 1
    fi
done

unset MOT_RESUME_FROM
export MOT_INITIALIZE_FROM="${STAGE1_CHECKPOINT}"

export MOT_DATASET_ROOT="${MOT_DATASET_ROOT:-/workspace/code/lingbot-va/data/0715_3k_subset_final_train}"
export MOT_OPTIMIZATION_COMPOSITION="v"
export MOT_MAX_VIEWS_PER_GPU="${MOT_MAX_VIEWS_PER_GPU:-8}"
export MOT_EVAL_WITH_CPU="${MOT_EVAL_WITH_CPU:-1}"
export MOT_EVAL_MODE="video"

export MOT_NUM_STEPS="${MOT_NUM_STEPS:-1000000}"
export MOT_SAVE_INTERVAL="${MOT_SAVE_INTERVAL:-2000}"
export MOT_LOG_INTERVAL="${MOT_LOG_INTERVAL:-100}"

export NGPU=4
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/train_logs/umi_subset3k_newmot_stage2_video_only/${RUN_STAMP}}"

exec bash "${SCRIPT_DIR}/train_mot_mixed_4gpu.sh" "$@"
