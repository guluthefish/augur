#!/bin/sh
#SBATCH --job-name=eval-none
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/output/evaluation-resnet50-none-%j.out
#SBATCH --error=slurm/output/evaluation-resnet50-none-%j.err

set -eu

REPO="/tudelft.net/staff-umbrella/IMGEN"
IMG="$REPO/containers/augur.sif"
OUTDIR="$REPO/slurm/output"
MODEL_NAME="resnet50-none"
MODEL_CONFIG="$REPO/configs/model-${MODEL_NAME}.yaml"
EMBEDDING_SOURCE="decoder"
DATA_CONFIG="$REPO/configs/tile_dataset-TCGA-BRCA-none.yaml"


# DAIC: use node-local /tmp for temporary data and clean it up
tmpdir="/tmp/${USER}/${SLURM_JOBID}"
mkdir -p "$tmpdir"
clean_up() { rm -rf "$tmpdir"; }
trap clean_up EXIT

# Run the evaluation script inside the container
srun apptainer exec \
    --nv \
    -B "$REPO:/work" \
    -B "$OUTDIR:/out" \
    -B "$tmpdir:/tmp" \
    -B "$tmpdir:/dev/shm" \
    --pwd /work \
    "$IMG" \
    /opt/conda/envs/augur/bin/python \
    scripts/evaluation/tile_feature_space.py \
    --model-config "$MODEL_CONFIG" \
    --dataset-config "$DATA_CONFIG" \
    --embedding-source "$EMBEDDING_SOURCE" \
    --class-threshold 0.5 \
    --dominance-threshold 0.5 \
    --portion-per-sample 1.0
