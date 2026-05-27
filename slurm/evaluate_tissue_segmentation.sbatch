#!/bin/sh
#SBATCH --job-name=eval-tissue-seg
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/output/tissue-seg-eval-%j.out
#SBATCH --error=slurm/output/tissue-seg-eval-%j.err

set -eu

REPO="/tudelft.net/staff-umbrella/IMGEN"
IMG="$REPO/containers/augur.sif"
OUTDIR="$REPO/slurm/output"

PRETEXTS="full none hematoxylin jigmag magnification"
MODEL_CONFIGS=""
for pretext in $PRETEXTS; do
    MODEL_CONFIGS="$MODEL_CONFIGS $REPO/configs/model-resnet50-${pretext}.yaml"
done

DATA_CONFIG="$REPO/configs/tile_dataset-TCGA-BRCA-full.yaml"


# DAIC: use node-local /tmp for temporary data and clean it up
tmpdir="/tmp/${USER}/${SLURM_JOBID}"
mkdir -p "$tmpdir"
clean_up() { rm -rf "$tmpdir"; }
trap clean_up EXIT

# Run the tissue-segmentation evaluation inside the container.
# All five resnet50 pretexts are scored on the same shared test split.
srun apptainer exec \
    --nv \
    -B "$REPO:/work" \
    -B "$OUTDIR:/out" \
    -B "$tmpdir:/tmp" \
    -B "$tmpdir:/dev/shm" \
    --pwd /work \
    "$IMG" \
    /opt/conda/envs/augur/bin/python \
    scripts/evaluation/tissue_segmentation.py \
    --model-configs $MODEL_CONFIGS \
    --dataset-config "$DATA_CONFIG" \
    --portion-per-sample 1.0
