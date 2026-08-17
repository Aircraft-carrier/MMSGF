#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/train_logs/robotwin_mot/0816_robotwin50_4gpu_run5}"
RESUME_STEP="${RESUME_STEP:-18000}"
RESUME_FROM="${RESUME_FROM:-${SAVE_ROOT}/checkpoints/checkpoint_step_${RESUME_STEP}}"

required_checkpoint_files=(
    "${RESUME_FROM}/_SUCCESS"
    "${RESUME_FROM}/checkpoint_metadata.json"
    "${RESUME_FROM}/training_state.pt"
    "${RESUME_FROM}/distributed_state/.metadata"
    "${RESUME_FROM}/transformer/config.json"
    "${RESUME_FROM}/transformer/diffusion_pytorch_model.safetensors"
)
for path in "${required_checkpoint_files[@]}"; do
    if [ ! -f "${path}" ]; then
        echo "Missing resume checkpoint file: ${path}" >&2
        exit 1
    fi
done

export SAVE_ROOT

echo "Restarting RoboTwin MOT training"
echo "Save root:   ${SAVE_ROOT}"
echo "Resume from: ${RESUME_FROM}"

exec "${REPO_ROOT}/1shell/train_robotwin_mot_4gpu.sh" \
    --resume-from "${RESUME_FROM}" \
    "$@"
