#!/bin/sh
#SBATCH --job-name=jigmag
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/output/visualization-%j.out
#SBATCH --error=slurm/output/visualization-%j.err

set -eu

REPO="/tudelft.net/staff-umbrella/IMGEN"
IMG="$REPO/containers/augur.sif"
OUTDIR="$REPO/slurm/output"
MODEL_NAME="resnet50-jigmag"
MODEL_CONFIG="$REPO/configs/model-${MODEL_NAME}.yaml"
EMBEDDING_SOURCE="pre-logits"
DATA_CONFIG="$REPO/configs/tile_dataset-TCGA-BRCA-jigmag.yaml"
TOP_CLASSES=5
LABEL_SOURCE="model-prediction"


# DAIC: use node-local /tmp for temporary data and clean it up
tmpdir="/tmp/${USER}/${SLURM_JOBID}"
mkdir -p "$tmpdir"
clean_up() { rm -rf "$tmpdir"; }
trap clean_up EXIT

# Run the visualization script inside the container
srun apptainer exec \
    --nv \
    -B "$REPO:/work" \
    -B "$OUTDIR:/out" \
    -B "$tmpdir:/tmp" \
    -B "$tmpdir:/dev/shm" \
    --pwd /work \
    "$IMG" \
    /opt/conda/envs/augur/bin/python \
    scripts/visualization/tile_embeddings.py \
    --model-config "$MODEL_CONFIG" \
    --dataset-config "$DATA_CONFIG" \
    --output-basename "$MODEL_NAME" \
    --embedding-source "$EMBEDDING_SOURCE" \
    --label-source "$LABEL_SOURCE" \
    --class-threshold 0.5 \
    --top-classes "$TOP_CLASSES" \
    --portion-per-sample 1.0
