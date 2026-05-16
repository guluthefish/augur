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
# Edit the four variables below (MODEL_CFG, DATA_CFG, ENCODER_ARCHITECTURE,
# TILE_PRETEXT) to switch encoders/cohorts. The output directory is derived
# from ENCODER_ARCHITECTURE + TILE_PRETEXT so different encoder variants land
# in disjoint caches.

set -eu

REPO="/tudelft.net/staff-umbrella/IMGEN"
IMG="$REPO/containers/augur.sif"
OUTDIR="$REPO/slurm/output"

CONFIG_DIR="$REPO/configs"
MODEL_CFG="model-resnet50-jigmag.yaml"
DATA_CFG="slide_dataset-TCGA-BRCA.yaml"
ENCODER_ARCHITECTURE="resnet50"
TILE_PRETEXT="jigmag"

# Encoding throughput knobs. float16 halves GPU memory at zero quality cost for a
# frozen encoder; A40s have 48 GB so CHUNK_SIZE can be generous.
PRECISION="float16"
CHUNK_SIZE="128"

# Where the per-slide .pt files land. Lives on the shared FS so the cache
# survives the job; downstream training reads from here.
FEATURES_DIR="$REPO/data/TCGA-BRCA/features/${ENCODER_ARCHITECTURE}-${TILE_PRETEXT}"
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
# race or fail on a cold cache.
case "$ENCODER_ARCHITECTURE" in
  *gigapath*)
    echo "Pre-caching $ENCODER_ARCHITECTURE weights into the node-local HF cache ..."
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
    'hf_hub:prov-gigapath/prov-gigapath',
    pretrained=True,
    num_classes=0,
    global_pool='',
)
print('Pre-cached prov-gigapath.')
"
    ;;
esac

echo "model_config:  $MODEL_CFG"
echo "data_config:   $DATA_CFG"
echo "encoder:       $ENCODER_ARCHITECTURE / $TILE_PRETEXT"
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
    -B "$tmpdir:/dev/shm" \
    --env HF_HOME=/tmp/huggingface \
    --pwd /work \
    "$IMG" \
    /opt/conda/envs/augur/bin/python scripts/model_training/precompute_tile_features.py \
    --config-dir "$CONFIG_DIR" \
    --model-config "$MODEL_CFG" \
    --data-config "$DATA_CFG" \
    --output-dir "$FEATURES_DIR" \
    --log-dir "$FEATURES_LOG_DIR" \
    --device cuda \
    --precision "$PRECISION" \
    --chunk-size "$CHUNK_SIZE" \
    --skip-existing
