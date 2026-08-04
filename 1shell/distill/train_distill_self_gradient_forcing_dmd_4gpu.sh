#!/usr/bin/env bash

set -euo pipefail

export DISTILL_METHOD="self_gradient_forcing_dmd"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_distill_paths.sh"

if [ -z "${DISTILL_STUDENT_INIT:-}" ] && [ -n "${DISTILL_CONSISTENCY_CHECKPOINT:-}" ]; then
    export DISTILL_STUDENT_INIT="${DISTILL_CONSISTENCY_CHECKPOINT}"
fi
if [ -z "${DISTILL_REAL_SCORE_CHECKPOINT:-}" ] && [ -n "${DISTILL_AR_CHECKPOINT:-}" ]; then
    export DISTILL_REAL_SCORE_CHECKPOINT="${DISTILL_AR_CHECKPOINT}"
fi
if [ -z "${DISTILL_FAKE_SCORE_INIT:-}" ] && [ -n "${DISTILL_AR_CHECKPOINT:-}" ]; then
    export DISTILL_FAKE_SCORE_INIT="${DISTILL_AR_CHECKPOINT}"
fi
if [ -z "${DISTILL_STUDENT_INIT:-}" ] || [ -z "${DISTILL_REAL_SCORE_CHECKPOINT:-}" ] || [ -z "${DISTILL_FAKE_SCORE_INIT:-}" ]; then
    echo "Set DISTILL_CONSISTENCY_CHECKPOINT and DISTILL_AR_CHECKPOINT, or set DISTILL_STUDENT_INIT, DISTILL_REAL_SCORE_CHECKPOINT, and DISTILL_FAKE_SCORE_INIT explicitly." >&2
    exit 1
fi
exec "${SCRIPT_DIR}/_train_distill_common.sh" "$@"
