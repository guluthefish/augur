#!/bin/sh
#SBATCH --job-name=pre-jigmag
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=slurm/output/resnet50-jigmag-precompute-features-%j.out
#SBATCH --error=slurm/output/resnet50-jigmag-precompute-features-%j.err

# One-shot precompute of frozen-encoder tile features for every slide.
#
# Runs as a single GPU / single process (no DDP) and walks every labelled
# slide once, writing one .pt feature file per slide to a shared output
# directory. The script is resume-friendly: --skip-existing means a requeued
# or re-submitted job picks up where the previous run stopped.
#
# precompute_tile_features.py composes its tile-model and dataset configs
# from the same partial-merge helpers train_tile_encoder.py uses, so editing
# ENCODER / PRETEXT / DATASET below picks the matching:
#   configs/tile-model/base-<ENCODER>.yaml
#   configs/tile-model/pretext-<PRETEXT>-<ENCODER>.yaml  (omit for `none`)
#   configs/dataset/base-<DATASET>.yaml                  (slide flavor)
# The output directory is `<root_dir>/features/<encoder>-<pretext_tag>` —
# matching the layout `flavor-feature.yaml` derives — so the aggregator's
# feature flavor picks this cache up automatically. Edit the SBATCH
# --job-name / --output / --error above to match when changing ENCODER or
# PRETEXT.

set -eu

REPO="/tudelft.net/staff-umbrella/IMGEN"
IMG="$REPO/containers/augur.sif"
OUTDIR="$REPO/slurm/output"

CONFIG_DIR="$REPO/configs"
ENCODER="resnet50"
PRETEXT="jigmag"
DATASET="tcga-brca"

# Encoding throughput knobs. float16 halves GPU memory at zero quality cost for a
# frozen encoder; A40s have 48 GB so CHUNK_SIZE can be generous.
PRECISION="float16"
CHUNK_SIZE="128"

# Where the per-slide .pt files land. Mirrors the default
# `<root_dir>/features/<encoder>-<pretext>` the python script auto-derives;
# passed explicitly here for shell-side visibility and so the up-front
# `mkdir -p` works on the shared FS before the container starts.
FEATURES_DIR="$REPO/data/TCGA-BRCA/features/${ENCODER}-${PRETEXT}"
FEATURES_LOG_DIR="$REPO/logs/precompute_tile_features"
mkdir -p "$FEATURES_DIR" "$FEATURES_LOG_DIR"

export PYTHONPATH="/work${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# DAIC: use node-local /tmp for HF cache + scratch and clean it up on exit.
tmpdir="/tmp/${USER}/${SLURM_JOBID}"
mkdir -p "$tmpdir"
clean_up() { rm -rf "$tmpdir"; }
trap clean_up EXIT

# Stage the HF Hub cache to node-local storage so timm doesn't hit NFS on
# every encoder construction.
hf_cache_local="$tmpdir/huggingface"
if [ -d "$HOME/.cache/huggingface" ]; then
    echo "Staging HF Hub cache to $hf_cache_local ..."
    cp -r "$HOME/.cache/huggingface" "$hf_cache_local"
else
    mkdir -p "$hf_cache_local"
fi

# For HF-Hub-backed encoders, pre-cache the weights so the actual run can't
# race or fail on a cold cache. Encoder → timm model id map mirrors
# configs/tile-model/base-<ENCODER>.yaml.
case "$ENCODER" in
  prov-gigapath) HF_MODEL_ID="hf_hub:prov-gigapath/prov-gigapath" ;;
  *)             HF_MODEL_ID="" ;;
esac
if [ -n "$HF_MODEL_ID" ]; then
    echo "Pre-caching $HF_MODEL_ID into the node-local HF cache ..."
    apptainer exec \
        -B "$REPO:/work" \
        -B "$tmpdir:/tmp" \
        --env HF_HOME=/tmp/huggingface \
        --pwd /work \
        "$IMG" \
        /opt/conda/envs/augur/bin/python -c "
from dotenv import load_dotenv
load_dotenv()
import timm
timm.create_model(
    '$HF_MODEL_ID',
    pretrained=True,
    num_classes=0,
    global_pool='',
)
print('Pre-cached $HF_MODEL_ID.')
"
fi

echo "encoder:       $ENCODER"
echo "pretext:       $PRETEXT"
echo "dataset:       $DATASET"
echo "features_dir:  $FEATURES_DIR"
echo "precision:     $PRECISION"
echo "chunk_size:    $CHUNK_SIZE"

# Single-task / single-GPU run — encoding is embarrassingly parallel across
# slides but the script processes them sequentially. No DDP, so --ntasks=1.
srun --ntasks=1 apptainer exec \
    --nv \
    -B "$REPO:/work" \
    -B "$OUTDIR:/out" \
    -B "$tmpdir:/tmp" \
    --env HF_HOME=/tmp/huggingface \
    --pwd /work \
    "$IMG" \
    /opt/conda/envs/augur/bin/python scripts/model_training/precompute_tile_features.py \
        --config-dir "$CONFIG_DIR" \
        --encoder "$ENCODER" \
        --pretext "$PRETEXT" \
        --dataset "$DATASET" \
        --output-dir "$FEATURES_DIR" \
        --log-dir "$FEATURES_LOG_DIR" \
        --device cuda \
        --precision "$PRECISION" \
        --chunk-size "$CHUNK_SIZE" \
        --skip-existing
