#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${MOT_DATASET_ROOT:?Set MOT_DATASET_ROOT to the prepared RoboTwin dataset}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG_NAME="robotwin_mot_train"
NGPU=4
MASTER_PORT="${MASTER_PORT:-29561}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/train_logs/robotwin_mot/${RUN_STAMP}}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-${SAVE_ROOT}/torchrun_logs}"

for required_file in \
    "${MOT_DATASET_ROOT}/meta/mot_config.json" \
    "${MOT_DATASET_ROOT}/meta/mot_final_training_manifest.jsonl" \
    "${MOT_DATASET_ROOT}/empty_emb.pt" \
    "${MOT_DATASET_ROOT}/text_emb_cache.pt"; do
    if [ ! -f "${required_file}" ]; then
        echo "Missing prepared RoboTwin asset: ${required_file}" >&2
        exit 1
    fi
done

mkdir -p "${TORCHRUN_LOG_DIR}"

"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    --log-dir "${TORCHRUN_LOG_DIR}" \
    --tee 3 \
    --local-ranks-filter="${LOG_RANK:-0}" \
    -m wan_va.train_mot \
    --config-name "${CONFIG_NAME}" \
    --save-root "${SAVE_ROOT}" \
    "$@"
