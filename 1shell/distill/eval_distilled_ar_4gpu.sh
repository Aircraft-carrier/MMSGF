#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/1shell/distill/_distill_paths.sh"

PYTHON_BIN="${PYTHON_BIN:-/workspace/basics/miniconda3/envs/mywam/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python"
fi

CHECKPOINT="${CHECKPOINT:-${DISTILL_DEFAULT_STUDENT_INIT}}"
DATASET_ROOT="${MOT_DATASET_ROOT:-${DISTILL_DEFAULT_DATASET_ROOT}}"

require_distill_dataset
require_distill_checkpoint "eval" "${CHECKPOINT}"

exec "${PYTHON_BIN}" -m inference.mot_chunk_infer \
    --checkpoint "${CHECKPOINT}" \
    --dataset-root "${DATASET_ROOT}" \
    "$@"
