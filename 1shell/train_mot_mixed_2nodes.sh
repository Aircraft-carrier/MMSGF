#!/usr/bin/env bash

set -euo pipefail

umask 007

source /workspace/basics/miniconda3/etc/profile.d/conda.sh
conda activate lingbotVA
which python
which torchrun

REPO_ROOT="/workspace/code/lingbot-va"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export cv2_NUM_THREADS=0

export MOT_DATASET_ROOT="${MOT_DATASET_ROOT:-${REPO_ROOT}/data/data/umi_mot_full_data_train_0712_final}"
export MOT_VIDEO_DOWNSAMPLE_RATIO="${MOT_VIDEO_DOWNSAMPLE_RATIO:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WAN22_PRETRAINED_MODEL_PATH="${WAN22_PRETRAINED_MODEL_PATH:-/workspace/cache/huggingface_cache/hub/models--robbyant--lingbot-va-base/snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c}"
export VGGTO_CHECKPOINT_PATH="${VGGTO_CHECKPOINT_PATH:-/workspace/model/vggt-omega/vggt_omega_1b_512.pt}"
export VGGT_CHECKPOINT_PATH="${VGGT_CHECKPOINT_PATH:-/workspace/model/vggt/model.safetensors}"

CONFIG_NAME="${CONFIG_NAME:-umi_3dwam_train}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/basics/miniconda3/envs/lingbotVA/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/train_logs/umi_mot_full_pretrain_0712_2nodes_not_ring}"

# export NCCL_ALGO="${NCCL_ALGO:-RING}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-12345}"
export NODE_RANK="${NODE_RANK:-${RANK:-0}}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NNODES="${NNODES:-${WORLD_SIZE:-2}}"
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
export NCCL_IB_DISABLE=0
export NCCL_NET=IB

TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-${SAVE_ROOT}/torchrun_logs/node_${NODE_RANK}}"


train_args=(
    --config-name "${CONFIG_NAME}"
    --save-root "${SAVE_ROOT}"
)

mkdir -p "${TORCHRUN_LOG_DIR}"

torchrun_args=(
    --nproc_per_node="${GPUS_PER_NODE}"
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    --log-dir "${TORCHRUN_LOG_DIR}"
    --tee 3
)

"${PYTHON_BIN}" -m torch.distributed.run \
    "${torchrun_args[@]}" \
    -m wan_va.train_mot "${train_args[@]}" "$@"
