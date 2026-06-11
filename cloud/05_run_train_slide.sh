#!/usr/bin/env bash
# Train the slide-level aggregator (CLAM/MIL) on the precomputed tile features, for
# each trained encoder and across all CV folds. Wraps
# scripts/model_training/train_slide_aggregator.py.
#
#   reads features from : data/TCGA-BRCA/features/<encoder>-<pretext>/   (from 04_precompute_features.sh)
#   writes checkpoints to: checkpoints/<run_name>.pth + outputs/model_training/
#   run name            : <base>[-<subtask>]-<variant>[-<add-on>]-<encoder>-<pretext>-fold<idx>
#
# Run after 04. The aggregator operates on cached feature bags, so it's small and fast.
# Defaults reproduce the clam-full-mb-gated runs.
#
# Usage:
#   bash cloud/05_run_train_slide.sh                         # sweep: default pretexts x all folds
#   bash cloud/05_run_train_slide.sh full                    # only the resnet50-full features
#   FOLD=0 bash cloud/05_run_train_slide.sh full             # one fold of one pretext
#   SUBTASK="" bash cloud/05_run_train_slide.sh              # plain CLAM (no signature subtasks)
#   BASE=mil VARIANT=attention ADD_ON=gated SUBTASK="" bash cloud/05_run_train_slide.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace/micromamba}"
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

# --- aggregator architecture (defaults reproduce clam-full-mb-gated) ---------
BASE="${BASE:-clam}"          # clam | mil
VARIANT="${VARIANT:-mb}"      # clam: sb|mb ; mil: mean|max|attention
ADD_ON="${ADD_ON:-gated}"     # "" | gated   (attention-based variants only)
SUBTASK="${SUBTASK:-full}"    # "" | "sbs dbs id cnv" | full  (clam only; needs signature labels)
ENCODER="${ENCODER:-resnet50}"
DATASET="${DATASET:-tcga-brca}"
TRAINER="${TRAINER:-default}" # default | long | test
N_FOLDS="${N_FOLDS:-5}"       # must match params.n_folds in configs/dataset/base-<dataset>.yaml
ROOT="${ROOT:-data/TCGA-BRCA}"

# Pretext variants whose feature sets to train on (override via arguments).
PRETEXTS=("$@")
if [ "${#PRETEXTS[@]}" -eq 0 ]; then
  PRETEXTS=(full jigmag magnification hematoxylin)
fi

# Folds: a single FOLD if set, else 0 .. N_FOLDS-1.
if [ -n "${FOLD:-}" ]; then
  FOLDS=("$FOLD")
else
  mapfile -t FOLDS < <(seq 0 $((N_FOLDS - 1)))
fi

for P in "${PRETEXTS[@]}"; do
  FEAT="$ROOT/features/${ENCODER}-${P}"
  if [ ! -d "$FEAT" ]; then
    echo "!! Features not found: $FEAT/"
    echo "   Precompute first:  bash cloud/04_precompute_features.sh $P   (skipping ${ENCODER}-${P})"
    continue
  fi
  for F in "${FOLDS[@]}"; do
    echo
    echo "==================================================================="
    echo ">> aggregator ${BASE}${SUBTASK:+-${SUBTASK// /-}}-${VARIANT}${ADD_ON:+-${ADD_ON}}-${ENCODER}-${P}  | fold ${F} of ${N_FOLDS}"
    echo "==================================================================="
    # shellcheck disable=SC2086  # SUBTASK is nargs='*': intentional word-splitting
    micromamba run -n augur python scripts/model_training/train_slide_aggregator.py \
      --config-dir configs \
      --base "$BASE" \
      --variant "$VARIANT" \
      ${ADD_ON:+--add-on "$ADD_ON"} \
      ${SUBTASK:+--subtask $SUBTASK} \
      --encoder "$ENCODER" \
      --pretext "$P" \
      --dataset "$DATASET" \
      --trainer "$TRAINER" \
      --n-folds "$N_FOLDS" \
      --fold-idx "$F" \
      --precomputed
  done
done

echo
echo ">> Done. Aggregator checkpoints in checkpoints/  (and outputs/model_training/)."
