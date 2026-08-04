#!/usr/bin/env bash

set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/1shell/distill/_distill_paths.sh"

PYTHON_BIN="${PYTHON_BIN:-/workspace/basics/miniconda3/envs/mywam/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python"
fi

PIPELINE_ROOT="${PIPELINE_ROOT:-${REPO_ROOT}/train_logs/distill_pipeline/$(date +%m%d_%H%M%S)}"
STUDENT_INIT="${STUDENT_INIT:-${DISTILL_DEFAULT_STUDENT_INIT}}"
NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29561}"
RESUME_ARGS=()
if [ -n "${PIPELINE_RESUME_METHOD:-}" ]; then
    RESUME_ARGS+=(--resume-method "${PIPELINE_RESUME_METHOD}")
fi
if [ -n "${PIPELINE_RESUME_FROM:-}" ]; then
    RESUME_ARGS+=(--resume-from "${PIPELINE_RESUME_FROM}")
fi

require_distill_dataset
require_distill_checkpoint "pipeline student-init" "${STUDENT_INIT}"

exec "${PYTHON_BIN}" -m distillation.workflow \
    --pipeline-root "${PIPELINE_ROOT}" \
    --student-init "${STUDENT_INIT}" \
    --ngpu "${NGPU}" \
    --master-port "${MASTER_PORT}" \
    "${RESUME_ARGS[@]}"
