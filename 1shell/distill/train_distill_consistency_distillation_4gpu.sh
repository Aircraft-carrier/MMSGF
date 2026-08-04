#!/usr/bin/env bash

set -euo pipefail

export DISTILL_METHOD="consistency_distillation"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_distill_paths.sh"

if [ -z "${DISTILL_STUDENT_INIT:-}" ] && [ -n "${DISTILL_AR_CHECKPOINT:-}" ]; then
    export DISTILL_STUDENT_INIT="${DISTILL_AR_CHECKPOINT}"
fi
if [ -z "${DISTILL_TEACHER_CHECKPOINT:-}" ] && [ -n "${DISTILL_AR_CHECKPOINT:-}" ]; then
    export DISTILL_TEACHER_CHECKPOINT="${DISTILL_AR_CHECKPOINT}"
fi
if [ -z "${DISTILL_STUDENT_INIT:-}" ] || [ -z "${DISTILL_TEACHER_CHECKPOINT:-}" ]; then
    echo "Set DISTILL_AR_CHECKPOINT to a completed autoregressive distillation checkpoint, or set DISTILL_STUDENT_INIT and DISTILL_TEACHER_CHECKPOINT explicitly." >&2
    exit 1
fi
exec "${SCRIPT_DIR}/_train_distill_common.sh" "$@"
