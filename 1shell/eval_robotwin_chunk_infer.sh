#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/zsh/miniconda3/envs/linbotva/bin/python"
CHECKPOINT_ROOT="${REPO_ROOT}/train_logs/robotwin_mot/0816_robotwin50_4gpu_run5/checkpoints/checkpoint_step_18000"
MOT_DATASET_ROOT="${MOT_DATASET_ROOT:-${REPO_ROOT}/data/robotwin_clean_50}"
WAN22_MODEL_ROOT="${WAN22_MODEL_ROOT:-${REPO_ROOT}/playground/Pretrained_models/Wan2.2-TI2V-5B}"
SOURCE_DATASET="${SOURCE_DATASET:-robotwin_eef_clean_50}"
MODE="${MODE:-full}"
GPU_ID="${GPU_ID:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/inference_logs/$(basename "$(dirname "$(dirname "${CHECKPOINT_ROOT}")")")_$(basename "${CHECKPOINT_ROOT}")_${MODE}}"

required_paths=(
    "${PYTHON_BIN}"
    "${CHECKPOINT_ROOT}/_SUCCESS"
    "${CHECKPOINT_ROOT}/transformer/config.json"
    "${CHECKPOINT_ROOT}/transformer/diffusion_pytorch_model.safetensors"
    "${MOT_DATASET_ROOT}/meta/mot_config.json"
    "${MOT_DATASET_ROOT}/meta/mot_final_training_manifest.jsonl"
    "${MOT_DATASET_ROOT}/empty_emb.pt"
    "${MOT_DATASET_ROOT}/text_emb_cache.pt"
    "${WAN22_MODEL_ROOT}/vae/config.json"
    "${WAN22_MODEL_ROOT}/vae/diffusion_pytorch_model.safetensors"
)
for path in "${required_paths[@]}"; do
    if [ ! -e "${path}" ]; then
        echo "Missing required path: ${path}" >&2
        exit 1
    fi
done

export MOT_DATASET_ROOT WAN22_MODEL_ROOT
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "Checkpoint: ${CHECKPOINT_ROOT}"
echo "Dataset:    ${MOT_DATASET_ROOT} (${SOURCE_DATASET})"
echo "VAE root:   ${WAN22_MODEL_ROOT}"
echo "Output:     ${OUTPUT_DIR}"
echo "GPU:        ${GPU_ID}"

exec "${PYTHON_BIN}" -m inference.mot_chunk_infer \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --mode "${MODE}" \
    --device cuda:0 \
    --source-dataset "${SOURCE_DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
