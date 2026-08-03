#!/usr/bin/env bash

set -euo pipefail

export DISTILL_METHOD="consistency_distillation"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_train_distill_common.sh" "$@"
