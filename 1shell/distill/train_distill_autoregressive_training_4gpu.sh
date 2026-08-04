#!/usr/bin/env bash

set -euo pipefail

export DISTILL_METHOD="autoregressive_training"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_distill_paths.sh"

export DISTILL_STUDENT_INIT="${DISTILL_STUDENT_INIT:-${DISTILL_DEFAULT_STUDENT_INIT}}"
exec "${SCRIPT_DIR}/_train_distill_common.sh" "$@"
