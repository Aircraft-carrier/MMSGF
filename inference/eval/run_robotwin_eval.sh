#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
POLICY_REPO=${POLICY_REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
export POLICY_REPO
export POLICY_SERVER_ENTRYPOINT=inference.eval.server
export ROBOTWIN_CLIENT_ENTRYPOINT="${POLICY_REPO}/distillation/eval/robotwin_client.py"

exec bash "${POLICY_REPO}/distillation/eval/run_robotwin_eval.sh" "$@"
