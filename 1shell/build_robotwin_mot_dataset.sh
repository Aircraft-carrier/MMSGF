#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash 1shell/build_robotwin_mot_dataset.sh [task ...]

Examples:
  bash 1shell/build_robotwin_mot_dataset.sh
  bash 1shell/build_robotwin_mot_dataset.sh adjust_bottle
  bash 1shell/build_robotwin_mot_dataset.sh adjust_bottle lift_pot

With no task arguments, all tasks under ROBOTWIN_SOURCE_ROOT are built. When
tasks are provided, the output dataset is rebuilt using only those tasks.

Environment overrides:
  ROBOTWIN_SOURCE_ROOT, MOT_DATASET_ROOT, TEXT_MODEL_PATH, PREPARE_DEVICE,
  PYTHON_BIN
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ROBOTWIN_SOURCE_ROOT="${ROBOTWIN_SOURCE_ROOT:-${REPO_ROOT}/playground/Dataset/clean_robotwin}"
MOT_DATASET_ROOT="${MOT_DATASET_ROOT:-${REPO_ROOT}/data/robotwin_clean_50}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-/zsh/cache/hf_cache/hub/models--robbyant--lingbot-va-base/snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c}"
PREPARE_DEVICE="${PREPARE_DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/zsh/miniconda3/envs/linbotva/bin/python}"

if [ ! -d "${ROBOTWIN_SOURCE_ROOT}" ]; then
    echo "Missing RoboTwin source dataset: ${ROBOTWIN_SOURCE_ROOT}" >&2
    exit 1
fi
if [ ! -d "${TEXT_MODEL_PATH}/tokenizer" ] || \
   [ ! -d "${TEXT_MODEL_PATH}/text_encoder" ]; then
    echo "Missing tokenizer/text encoder under TEXT_MODEL_PATH: ${TEXT_MODEL_PATH}" >&2
    exit 1
fi

tasks=("$@")
build_args=(
    --source-root "${ROBOTWIN_SOURCE_ROOT}"
    --output-root "${MOT_DATASET_ROOT}"
    --model-root "${TEXT_MODEL_PATH}"
    --device "${PREPARE_DEVICE}"
)
for task in "${tasks[@]}"; do
    build_args+=(--task "${task}")
done

"${PYTHON_BIN}" -m wan_va.dataset.build_robotwin_eef_training_dataset \
    "${build_args[@]}"
