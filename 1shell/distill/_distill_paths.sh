#!/usr/bin/env bash

set -euo pipefail

if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

UNI3DWAM_ROOT="${UNI3DWAM_ROOT:-/fcy/code/uni3dwam}"
FCY_CACHE_ROOT="${FCY_CACHE_ROOT:-/fcy/cache}"
FCY_HF_CACHE_ROOT="${FCY_HF_CACHE_ROOT:-${FCY_CACHE_ROOT}/huggingface_cache}"
ZSH_CACHE_ROOT="${ZSH_CACHE_ROOT:-/zsh/cache}"
ZSH_HF_CACHE_ROOT="${ZSH_HF_CACHE_ROOT:-${ZSH_CACHE_ROOT}/hf_cache}"
DISTILL_DEFAULT_HF_CACHE_ROOT="${DISTILL_DEFAULT_HF_CACHE_ROOT:-${ZSH_HF_CACHE_ROOT}}"
if [ ! -d "${DISTILL_DEFAULT_HF_CACHE_ROOT}" ] && [ -d "${FCY_HF_CACHE_ROOT}" ]; then
    DISTILL_DEFAULT_HF_CACHE_ROOT="${FCY_HF_CACHE_ROOT}"
fi

DISTILL_DEFAULT_DATASET_ROOT="${DISTILL_DEFAULT_DATASET_ROOT:-${REPO_ROOT}/data/umi_distill_train}"
DISTILL_DEFAULT_STUDENT_INIT="${DISTILL_DEFAULT_STUDENT_INIT:-${REPO_ROOT}/models/uni3dwam_video_only_step62000}"
DISTILL_DEFAULT_STAGE1_TEACHER_CHECKPOINT="${DISTILL_DEFAULT_STAGE1_TEACHER_CHECKPOINT:-${DISTILL_DEFAULT_STUDENT_INIT}}"
DISTILL_DEFAULT_GEOMETRY_INIT="${DISTILL_DEFAULT_GEOMETRY_INIT:-${REPO_ROOT}/models/uni3dwam_geometry_only_step60000}"
DISTILL_DEFAULT_VIDEO_FROM_WAN_INIT="${DISTILL_DEFAULT_VIDEO_FROM_WAN_INIT:-${REPO_ROOT}/models/uni3dwam_video_from_wan_step20000}"
DISTILL_DEFAULT_WAN22_PRETRAINED_MODEL_PATH="${DISTILL_DEFAULT_WAN22_PRETRAINED_MODEL_PATH:-${DISTILL_DEFAULT_HF_CACHE_ROOT}/hub/models--robbyant--lingbot-va-base/snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c}"

export MOT_DATASET_ROOT="${MOT_DATASET_ROOT:-${DISTILL_DEFAULT_DATASET_ROOT}}"
export MOT_POINTCLOUD_SAMPLE_PERIOD="${MOT_POINTCLOUD_SAMPLE_PERIOD:-8}"
export MOT_VIDEO_DOWNSAMPLE_RATIO="${MOT_VIDEO_DOWNSAMPLE_RATIO:-4}"
export MOT_EVAL_WITH_CPU="${MOT_EVAL_WITH_CPU:-0}"
export MOT_EVAL_MODE="${MOT_EVAL_MODE:-video}"
export MOT_OPTIMIZATION_COMPOSITION="${MOT_OPTIMIZATION_COMPOSITION:-v}"
export MOT_MAX_VIEWS_PER_GPU="${MOT_MAX_VIEWS_PER_GPU:-8}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-umi_distillation}"

export HF_HOME="${DISTILL_HF_HOME:-${DISTILL_DEFAULT_HF_CACHE_ROOT}}"
export HF_DATASETS_CACHE="${DISTILL_HF_DATASETS_CACHE:-${DISTILL_DEFAULT_HF_CACHE_ROOT}/datasets}"
export HUGGINGFACE_HUB_CACHE="${DISTILL_HUGGINGFACE_HUB_CACHE:-${DISTILL_DEFAULT_HF_CACHE_ROOT}/hub}"
export TRANSFORMERS_CACHE="${DISTILL_TRANSFORMERS_CACHE:-${DISTILL_DEFAULT_HF_CACHE_ROOT}/hub}"
export WAN22_PRETRAINED_MODEL_PATH="${WAN22_PRETRAINED_MODEL_PATH:-${DISTILL_DEFAULT_WAN22_PRETRAINED_MODEL_PATH}}"

require_distill_path() {
    local label="$1"
    local path="$2"
    if [ ! -e "${path}" ]; then
        echo "Missing ${label}: ${path}" >&2
        return 1
    fi
}

require_distill_checkpoint() {
    local label="$1"
    local checkpoint="$2"
    require_distill_path "${label} checkpoint dir" "${checkpoint}"
    require_distill_path "${label} _SUCCESS" "${checkpoint}/_SUCCESS"
    require_distill_path "${label} transformer config" "${checkpoint}/transformer/config.json"
    require_distill_path "${label} transformer weights" "${checkpoint}/transformer/diffusion_pytorch_model.safetensors"
}

require_distill_dataset() {
    require_distill_path "MOT dataset config" "${MOT_DATASET_ROOT}/meta/mot_config.json"
    require_distill_path "MOT pointcloud manifest" "${MOT_DATASET_ROOT}/meta/mot_final_training_pointcloud_manifest.jsonl"
    require_distill_path "MOT non-pointcloud manifest" "${MOT_DATASET_ROOT}/meta/mot_final_training_non_pointcloud_manifest.jsonl"
}
