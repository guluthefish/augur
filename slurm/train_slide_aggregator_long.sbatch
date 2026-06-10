#!/bin/sh
#SBATCH --job-name=lg-attn-full
#SBATCH --array=0-4
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/output/mil-attention-resnet50-full-fold%a-%A.out
#SBATCH --error=slurm/output/mil-attention-resnet50-full-fold%a-%A.err
#SBATCH --signal=TERM@180

# train_slide_aggregator.py composes its configs from the partial-merge
# helpers in augur/utils/config.py. The tokens below feed:
#   BASE     → configs/aggregator/base-<BASE>.yaml
#   VARIANT  → configs/aggregator/variant-<VARIANT>.yaml
#   ADD_ON   → configs/aggregator/add-on-<ADD_ON>.yaml          (optional)
#   SUBTASKS → configs/aggregator/subtask-<TOKEN>.yaml ...      (optional;
#              only valid with BASE=clam; pass multiple space-separated
#              tokens, e.g. "sbs dbs")
#   ENCODER  → configs/aggregator/encoder-<ENCODER>.yaml
#   PRETEXT  → configs/aggregator/pretext-<PRETEXT>.yaml
#   DATASET  → configs/dataset/base-<DATASET>.yaml
#              (+ flavor-feature.yaml when PRECOMPUTED=1)
#   TRAINER  → configs/trainer/base-<TRAINER>.yaml
# Long-running variant: --qos=long, --time=7-00:00:00, TRAINER=long (which sets
# max_time ~3h below the 7-day limit so Lightning self-stops and checkpoints
# cleanly before the SLURM kill). This script is submitted as a 5-element slurm
# array (one task per fold). Each task reads its SLURM_ARRAY_TASK_ID and uses it
# as --fold-idx; N_FOLDS must match params.n_folds in
# configs/dataset/base-<DATASET>.yaml. Submit with
# `sbatch slurm/train_slide_aggregator_long.sbatch` to launch all 5 folds, or
# narrow the array (e.g. `sbatch --array=2 ...`) for a single fold.
# The run name (and therefore checkpoint / log filename) is
#   <BASE>[-<SUBTASKS>...]-<VARIANT>[-<ADD_ON>]-<ENCODER>-<PRETEXT>-fold<IDX>
# Edit the SBATCH --job-name / --output / --error above to match when
# changing any of these tokens.

set -eu

ulimit -n 65536

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
TRAINER="long"    # default (production GPU/DDP) | test (fast-dev) | long (7-day)

# --- Cross-validation --------------------------------------------------------
# N_FOLDS must match params.n_folds in configs/dataset/base-<DATASET>.yaml;
# both values are passed through so the slurm submission stays self-describing.
# FOLD_IDX is driven by SLURM_ARRAY_TASK_ID (one fold per array task); when
# this script is sourced outside an array it falls back to fold 0.
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
    /opt/conda/envs/augur/bin/python \
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
