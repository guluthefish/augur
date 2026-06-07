#!/bin/sh
#SBATCH --job-name=aggr-test
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm/output/test-aggregator-%j.out
#SBATCH --error=slurm/output/test-aggregator-%j.err

set -eu

REPO="/tudelft.net/staff-umbrella/IMGEN"
IMG="$REPO/containers/augur.sif"
OUTDIR="$REPO/slurm/output"

CONFIG_DIR="$REPO/configs"

# --- Aggregator architecture --------------------------------------------------
BASE="mil"           # clam | mil
VARIANT="attention"  # sb | mb (clam) ; mean | max | attention (mil)
ADD_ON=""            # "" | gated   (only with attention-based variants)
SUBTASKS=""          # "" or e.g. "sbs" or "sbs dbs"  (only with BASE=clam)

# --- Encoder used to extract features (must match the precompute job) --------
ENCODER="resnet50"
PRETEXT="full"

# --- Dataset + trainer recipe ------------------------------------------------
DATASET="tcga-brca"
TRAINER="test"    # default (production GPU/DDP) | test (fast-dev)

# --- Cross-validation --------------------------------------------------------
# N_FOLDS must match params.n_folds in configs/dataset/base-<DATASET>.yaml;
# both values are passed through so the slurm submission stays self-describing.
# FOLD_IDX defaults to 0 for single-shot runs and is overridden by
# SLURM_ARRAY_TASK_ID when this script is submitted as an array.
N_FOLDS=5
FOLD_IDX="${SLURM_ARRAY_TASK_ID:-0}"

# --- Feature cache -----------------------------------------------------------
# 1 → load precomputed features (TCGAFeatureDataset); 0 → encode on the fly.
PRECOMPUTED=1
# When PRECOMPUTED=1 and FEATURES_DIR is empty, the python script derives
# `<root_dir>/features/<ENCODER>-<PRETEXT>` automatically. Set explicitly
# below to point at the archived cache instead.
FEATURES_DIR="$REPO/data/TCGA-BRCA/features/archive-${ENCODER}-${PRETEXT}"

export PYTHONPATH="/work${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# DAIC: use node-local /tmp for temporary data and clean it up.
tmpdir="/tmp/${USER}/${SLURM_JOBID}"
mkdir -p "$tmpdir"
clean_up() { rm -rf "$tmpdir"; }
trap clean_up EXIT

echo "base:       $BASE"
echo "variant:    $VARIANT"
echo "add_on:     ${ADD_ON:-<none>}"
echo "subtasks:   ${SUBTASKS:-<none>}"
echo "encoder:    $ENCODER"
echo "pretext:    $PRETEXT"
echo "dataset:    $DATASET"
echo "trainer:    $TRAINER"
echo "precomputed: $PRECOMPUTED"
echo "n_folds:    $N_FOLDS"
echo "fold_idx:   $FOLD_IDX"
if [ "$PRECOMPUTED" = "1" ] && [ -n "$FEATURES_DIR" ]; then
    echo "features_dir: $FEATURES_DIR"
fi

# Build the optional CLI flags. `${VAR:+--flag "$VAR"}` expands to nothing
# when VAR is empty, so omitted axes don't introduce stray empty arguments
# that argparse would choke on.
#   - SUBTASKS is nargs='*': pass the tokens unquoted so word-splitting
#     yields one argv entry per token.
#   - PRECOMPUTED is a boolean flag (no value).
PRECOMPUTED_FLAG=""
if [ "$PRECOMPUTED" = "1" ]; then
    PRECOMPUTED_FLAG="--precomputed"
fi

# shellcheck disable=SC2086
srun apptainer exec \
    --nv \
    -B "$REPO:/work" \
    -B "$OUTDIR:/out" \
    -B "$tmpdir:/tmp" \
    --env TORCH_NCCL_BLOCKING_WAIT=1 \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    --env NCCL_DEBUG=INFO \
    --pwd /work \
    "$IMG" \
    /opt/conda/envs/augur/bin/torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=2 \
        scripts/model_training/train_slide_aggregator.py \
            --config-dir "$CONFIG_DIR" \
            --base "$BASE" \
            --variant "$VARIANT" \
            ${ADD_ON:+--add-on "$ADD_ON"} \
            ${SUBTASKS:+--subtask $SUBTASKS} \
            --encoder "$ENCODER" \
            --pretext "$PRETEXT" \
            --dataset "$DATASET" \
            --trainer "$TRAINER" \
            --n-folds "$N_FOLDS" \
            --fold-idx "$FOLD_IDX" \
            $PRECOMPUTED_FLAG \
            ${FEATURES_DIR:+--features-dir "$FEATURES_DIR"}