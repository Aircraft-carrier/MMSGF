#!/usr/bin/env bash

set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/basics/miniconda3/envs/lingbotVA/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python"
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
if [ -z "${CHECKPOINT_ROOT}" ]; then
    echo "Set CHECKPOINT_ROOT to a checkpoint directory containing transformer/" >&2
    exit 1
fi

export MOT_DATASET_ROOT="${MOT_DATASET_ROOT:-${REPO_ROOT}/data/umi_mot_mixed_weighted_full_stride4}"
export WAN22_MODEL_ROOT="${WAN22_MODEL_ROOT:-/workspace/cache/huggingface_cache/hub/models--robbyant--lingbot-va-base/snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"${PYTHON_BIN}" inference/mot_chunk_infer.py \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --mode "${MODE:-full}" \
    --device "${DEVICE:-cuda:0}" \
    "$@"
