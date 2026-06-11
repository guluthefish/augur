#!/bin/sh
#SBATCH --job-name=tile-timeout
#SBATCH --gres=gpu:2
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=slurm/output/resnet50-timeout-%j.out
#SBATCH --error=slurm/output/resnet50-timeout-%j.err
#SBATCH --signal=TERM@180

# Timeout-path SMOKE TEST for the node-crash fix. Uses TRAINER=smoketest-timeout
# (configs/trainer/base-smoketest-timeout.yaml): trains continuously with a tiny
# `max_time` (5 min) so the run actually reaches the time budget and we can
# observe the exact path that used to crash nodes — the job self-stops at a batch
# boundary, checkpoints, and Apptainer exits cleanly *before* any SLURM SIGKILL.
# It writes to a throwaway dir and never resumes, so it cannot touch the real
# checkpoints/logs. Run it on an EXCLUSIVE node: `sbatch --exclusive ...`.

set -eu

ulimit -n 65536

REPO="/tudelft.net/staff-umbrella/IMGEN"
IMG="$REPO/containers/augur.sif"
OUTDIR="$REPO/slurm/output"

# Config tokens (same partial-merge scheme as train_tile_encoder.sbatch):
#   ENCODER  → configs/tile-model/base-<ENCODER>.yaml
#   PRETEXT  → configs/tile-model/pretext-<PRETEXT>-<ENCODER>.yaml
#   DATASET  → configs/dataset/base-<DATASET>.yaml (+ flavor-tile.yaml)
#   TRAINER  → configs/trainer/base-<TRAINER>.yaml
ENCODER="resnet50"
PRETEXT="full"
DATASET="tcga-brca"
TRAINER="smoketest-timeout"

export PYTHONPATH="/work${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# NCCL transport overrides: GPU peer-to-peer and InfiniBand auto-detection
# hang during the first BROADCAST on some DAIC gpu* nodes. Force TCP sockets,
# which complete reliably (slightly slower but consistent).
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# DAIC: use node-local /tmp for temporary data and clean it up
tmpdir="/tmp/${USER}/${SLURM_JOBID}"
mkdir -p "$tmpdir"
clean_up() { rm -rf "$tmpdir"; }
trap clean_up EXIT

# Stage the HF Hub cache to node-local storage.
hf_cache_local="$tmpdir/huggingface"
if [ -d "$HOME/.cache/huggingface" ]; then
    echo "Staging HF Hub cache to $hf_cache_local ..."
    cp -r "$HOME/.cache/huggingface" "$hf_cache_local"
else
    mkdir -p "$hf_cache_local"
fi

echo "encoder: $ENCODER"
echo "pretext: $PRETEXT"
echo "dataset: $DATASET"
echo "trainer: $TRAINER"

# Run on compute node via srun; execute inside container
srun apptainer exec \
    --nv \
    -B "$REPO:/work" \
    -B "$OUTDIR:/out" \
    -B "$tmpdir:/tmp" \
    --env HF_HOME=/tmp/huggingface \
    --env NCCL_P2P_DISABLE=1 \
    --env NCCL_IB_DISABLE=1 \
    --env NCCL_DEBUG=WARN \
    --pwd /work \
    "$IMG" \
    /opt/conda/envs/augur/bin/python scripts/model_training/train_tile_encoder.py \
        --encoder "$ENCODER" \
        --pretext "$PRETEXT" \
        --dataset "$DATASET" \
        --trainer "$TRAINER"
