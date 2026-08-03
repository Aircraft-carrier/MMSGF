#!/usr/bin/env bash

set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/basics/miniconda3/envs/mywam/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python"
fi

PIPELINE_ROOT="${PIPELINE_ROOT:-${REPO_ROOT}/train_logs/distill_pipeline/$(date +%m%d_%H%M%S)}"
STUDENT_INIT="${STUDENT_INIT:?STUDENT_INIT must be set to a transformer checkpoint}"
NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29561}"
RESUME_ARGS=()
if [ -n "${PIPELINE_RESUME_METHOD:-}" ]; then
    RESUME_ARGS+=(--resume-method "${PIPELINE_RESUME_METHOD}")
fi
if [ -n "${PIPELINE_RESUME_FROM:-}" ]; then
    RESUME_ARGS+=(--resume-from "${PIPELINE_RESUME_FROM}")
fi

exec "${PYTHON_BIN}" -m distillation.workflow \
    --pipeline-root "${PIPELINE_ROOT}" \
    --student-init "${STUDENT_INIT}" \
    --ngpu "${NGPU}" \
    --master-port "${MASTER_PORT}" \
    "${RESUME_ARGS[@]}"
