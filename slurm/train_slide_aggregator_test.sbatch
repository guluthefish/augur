#!/bin/sh
#SBATCH --job-name=agg-test
#SBATCH --gres=gpu:2
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/output/aggregator-test-%j.out
#SBATCH --error=slurm/output/aggregator-test-%j.err

set -eu

REPO="/tudelft.net/staff-umbrella/IMGEN"
IMG="$REPO/containers/augur.sif"
OUTDIR="$REPO/slurm/output"

CONFIG_DIR="$REPO/configs"
TRAINING_CFG="training-aggregator-test.yaml"
TILE_PRETEXT="full"
DATA_CFG="feature_dataset-TCGA-BRCA-$TILE_PRETEXT.yaml"
AGGREGATOR_NAME="dual-clam-mb-gated"
ENCODER_ARCHITECTURE="resnet50"

export PYTHONPATH="/work${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# DAIC: use node-local /tmp for temporary data and clean it up.
tmpdir="/tmp/${USER}/${SLURM_JOBID}"
mkdir -p "$tmpdir"
clean_up() { rm -rf "$tmpdir"; }
trap clean_up EXIT

echo "training_config: $TRAINING_CFG"
echo "data_config:     $DATA_CFG"
echo "aggregator:      $AGGREGATOR_NAME / $ENCODER_ARCHITECTURE / $TILE_PRETEXT"

# Run on compute node via srun; execute inside container.
srun apptainer exec \
    --nv \
    -B "$REPO:/work" \
    -B "$OUTDIR:/out" \
    -B "$tmpdir:/tmp" \
    -B "$tmpdir:/dev/shm" \
    --env TORCH_NCCL_BLOCKING_WAIT=1 \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    --pwd /work \
    "$IMG" \
    /opt/conda/envs/augur/bin/python scripts/model_training/train_slide_aggregator.py \
    --config-dir "$CONFIG_DIR" \
    --training-config "$TRAINING_CFG" \
    --data-config "$DATA_CFG" \
    --aggregator-name "$AGGREGATOR_NAME" \
    --encoder-architecture "$ENCODER_ARCHITECTURE" \
    --tile-pretext "$TILE_PRETEXT"
