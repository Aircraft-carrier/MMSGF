#!/usr/bin/env bash

set -euo pipefail
set -x

umask 007

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
CPU_EVAL_NUM_THREADS="${CPU_EVAL_NUM_THREADS:-32}"
export cv2_NUM_THREADS="${cv2_NUM_THREADS:-0}"

export MOT_DATASET_ROOT="${MOT_DATASET_ROOT:-/workspace/code/lingbot-va/data/0715_3k_subset_final_train}"
export MOT_VIDEO_DOWNSAMPLE_RATIO="${MOT_VIDEO_DOWNSAMPLE_RATIO:-4}"
export MOT_EVAL_WITH_CPU="${MOT_EVAL_WITH_CPU:-1}"
export MOT_EVAL_MODE="${MOT_EVAL_MODE:-video}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WAN22_PRETRAINED_MODEL_PATH="${WAN22_PRETRAINED_MODEL_PATH:-/workspace/cache/huggingface_cache/hub/models--robbyant--lingbot-va-base/snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c}"
export WAN22_DIFFUSERS_MODEL_PATH="${WAN22_DIFFUSERS_MODEL_PATH:-/workspace/model/wan2_2_diffusers}"
export INIT_MODEL_FROM_LINGBOT="${INIT_MODEL_FROM_LINGBOT:-1}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-umi_va_mot_wam}"

export MOT_MAX_VIEWS_PER_GPU="${MOT_MAX_VIEWS_PER_GPU:-8}"

CONFIG_NAME="${CONFIG_NAME:-umi_3dwam_train}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/train_logs/umi_subset3k_newmot_video_only/${RUN_STAMP}}"

NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29561}"
LOG_RANK="${LOG_RANK:-0}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/basics/miniconda3/envs/mywam/bin/python}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-${SAVE_ROOT}/torchrun_logs}"
SHOW_ALL_RANK_LOGS="${SHOW_ALL_RANK_LOGS:-0}"

if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python"
fi

if [ ! -f "${MOT_DATASET_ROOT}/meta/mot_config.json" ]; then
    echo "Missing MOT dataset config: ${MOT_DATASET_ROOT}/meta/mot_config.json" >&2
    exit 1
fi

train_args=(
    --config-name "${CONFIG_NAME}"
    --save-root "${SAVE_ROOT}"
)

mkdir -p "${TORCHRUN_LOG_DIR}"

CPU_EVAL_WORKER_PID=""
CPU_EVAL_QUEUE_DIR="${SAVE_ROOT}/evaluation_queue"
CPU_EVAL_STOP_FILE="${CPU_EVAL_QUEUE_DIR}/_STOP"
CPU_EVAL_READY_FILE="${CPU_EVAL_QUEUE_DIR}/_WORKER_READY"
CPU_EVAL_WORKER_LOG="${SAVE_ROOT}/cpu_eval_worker.log"

stop_cpu_eval_worker() {
    if [ -n "${CPU_EVAL_WORKER_PID}" ]; then
        mkdir -p "${CPU_EVAL_QUEUE_DIR}"
        touch "${CPU_EVAL_STOP_FILE}"
    fi
}

case "${MOT_EVAL_WITH_CPU,,}" in
    1|true|yes|on)
        mkdir -p "${CPU_EVAL_QUEUE_DIR}"
        rm -f "${CPU_EVAL_STOP_FILE}" "${CPU_EVAL_READY_FILE}"
        setsid nice -n 10 ionice -c 3 env \
            CUDA_VISIBLE_DEVICES="" \
            OMP_NUM_THREADS="${CPU_EVAL_NUM_THREADS}" \
            MKL_NUM_THREADS="${CPU_EVAL_NUM_THREADS}" \
            OPENBLAS_NUM_THREADS="${CPU_EVAL_NUM_THREADS}" \
            NUMEXPR_NUM_THREADS="${CPU_EVAL_NUM_THREADS}" \
            "${PYTHON_BIN}" -m wan_va.checkpoint_eval \
            --queue-dir "${CPU_EVAL_QUEUE_DIR}" \
            --stop-file "${CPU_EVAL_STOP_FILE}" \
            >> "${CPU_EVAL_WORKER_LOG}" 2>&1 < /dev/null &
        CPU_EVAL_WORKER_PID=$!
        trap stop_cpu_eval_worker EXIT
        for _ in {1..100}; do
            if [ -f "${CPU_EVAL_READY_FILE}" ]; then
                break
            fi
            if ! kill -0 "${CPU_EVAL_WORKER_PID}" 2>/dev/null; then
                echo "CPU evaluation worker exited during startup; see ${CPU_EVAL_WORKER_LOG}" >&2
                exit 1
            fi
            sleep 0.1
        done
        if [ ! -f "${CPU_EVAL_READY_FILE}" ]; then
            echo "CPU evaluation worker did not become ready; see ${CPU_EVAL_WORKER_LOG}" >&2
            exit 1
        fi
        ;;
    0|false|no|off)
        ;;
    *)
        echo "MOT_EVAL_WITH_CPU must be a boolean flag, got ${MOT_EVAL_WITH_CPU}" >&2
        exit 1
        ;;
esac

torchrun_args=(
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    --log-dir "${TORCHRUN_LOG_DIR}" \
    --tee 3
)

if [ "${SHOW_ALL_RANK_LOGS}" != "1" ]; then
    torchrun_args+=(--local-ranks-filter="${LOG_RANK}")
fi

"${PYTHON_BIN}" -m torch.distributed.run \
    "${torchrun_args[@]}" \
    -m wan_va.train_mot "${train_args[@]}" "$@"
