#!/usr/bin/env bash
# Launch tile-encoder training on the pod.
# Usage:   bash cloud/03_run_train.sh <encoder> <pretext> <dataset> <trainer>
# Example: bash cloud/03_run_train.sh resnet50 full tcga-brca long
#          bash cloud/03_run_train.sh prov-gigapath full tcga-brca long   # needs .env with HF_TOKEN
set -euo pipefail
cd "$(dirname "$0")/.."

ENCODER="${1:-resnet50}"
PRETEXT="${2:-full}"
DATASET="${3:-tcga-brca}"
TRAINER="${4:-long}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace/micromamba}"
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

# Single cloud node: let NCCL use P2P / NVLink. The DAIC NCCL_*_DISABLE flags only
# slow a single pod down, so make sure they are NOT set.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
unset NCCL_P2P_DISABLE NCCL_IB_DISABLE 2>/dev/null || true

# Keep the GPUs fed: num_workers is set to 8 in configs/dataset/base-tcga-brca.yaml.
# Override per-pod to match its vCPU count (e.g. 12-16) if the GPU still stalls:
#   sed -i 's/^\(\s*num_workers:\).*/\1 16/' configs/dataset/base-tcga-brca.yaml

echo ">> encoder=$ENCODER pretext=$PRETEXT dataset=$DATASET trainer=$TRAINER"
micromamba run -n augur python scripts/model_training/train_tile_encoder.py \
  --encoder "$ENCODER" --pretext "$PRETEXT" --dataset "$DATASET" --trainer "$TRAINER"
