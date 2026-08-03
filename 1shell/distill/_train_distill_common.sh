#!/usr/bin/env bash

set -euo pipefail
set -x

umask 007

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export cv2_NUM_THREADS="${cv2_NUM_THREADS:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/basics/miniconda3/envs/mywam/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python"
fi

NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29561}"
LOG_RANK="${LOG_RANK:-0}"
SHOW_ALL_RANK_LOGS="${SHOW_ALL_RANK_LOGS:-0}"

METHOD="${DISTILL_METHOD:?DISTILL_METHOD must be set}"
SAVE_ROOT="${DISTILL_SAVE_ROOT:?DISTILL_SAVE_ROOT must be set}"

mkdir -p "${SAVE_ROOT}"
TORCHRUN_LOG_DIR="${SAVE_ROOT}/torchrun_logs"
mkdir -p "${TORCHRUN_LOG_DIR}"

train_args=(
    --method "${METHOD}"
    --save-root "${SAVE_ROOT}"
)

if [ -n "${DISTILL_STUDENT_INIT:-}" ]; then
    train_args+=(--student-init "${DISTILL_STUDENT_INIT}")
fi
if [ -n "${DISTILL_TEACHER_CHECKPOINT:-}" ]; then
    train_args+=(--teacher-checkpoint "${DISTILL_TEACHER_CHECKPOINT}")
fi
if [ -n "${DISTILL_REAL_SCORE_CHECKPOINT:-}" ]; then
    train_args+=(--real-score-checkpoint "${DISTILL_REAL_SCORE_CHECKPOINT}")
fi
if [ -n "${DISTILL_FAKE_SCORE_INIT:-}" ]; then
    train_args+=(--fake-score-init "${DISTILL_FAKE_SCORE_INIT}")
fi
if [ -n "${DISTILL_RESUME_FROM:-}" ]; then
    train_args+=(--resume-from "${DISTILL_RESUME_FROM}")
fi
if [ -n "${DISTILL_ROLLOUT_INTERVAL:-}" ]; then
    train_args+=(--rollout-interval "${DISTILL_ROLLOUT_INTERVAL}")
fi
if [ -n "${DISTILL_ROLLOUT_VIDEO_NUM_STEPS:-}" ]; then
    train_args+=(--rollout-video-num-steps "${DISTILL_ROLLOUT_VIDEO_NUM_STEPS}")
fi
if [ -n "${DISTILL_ROLLOUT_ACTION_NUM_STEPS:-}" ]; then
    train_args+=(--rollout-action-num-steps "${DISTILL_ROLLOUT_ACTION_NUM_STEPS}")
fi
if [ -n "${DISTILL_ROLLOUT_CHUNK_PAIRS:-}" ]; then
    train_args+=(--rollout-chunk-pairs "${DISTILL_ROLLOUT_CHUNK_PAIRS}")
fi
if [ -n "${DISTILL_CFG_MIN:-}" ]; then
    train_args+=(--cfg-min "${DISTILL_CFG_MIN}")
fi
if [ -n "${DISTILL_CFG_MAX:-}" ]; then
    train_args+=(--cfg-max "${DISTILL_CFG_MAX}")
fi
if [ -n "${DISTILL_SIGMA_DATA:-}" ]; then
    train_args+=(--sigma-data "${DISTILL_SIGMA_DATA}")
fi
torchrun_args=(
    --nproc_per_node="${NGPU}"
    --master_port="${MASTER_PORT}"
    --log-dir "${TORCHRUN_LOG_DIR}"
    --tee 3
)

if [ "${SHOW_ALL_RANK_LOGS}" != "1" ]; then
    torchrun_args+=(--local-ranks-filter="${LOG_RANK}")
fi

"${PYTHON_BIN}" -m torch.distributed.run \
    "${torchrun_args[@]}" \
    -m distillation.train "${train_args[@]}" "$@"
