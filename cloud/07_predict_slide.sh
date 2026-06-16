#!/usr/bin/env bash
# Dump slide-level predictions from the trained aggregator (CLAM/MIL), for each
# trained encoder and across all CV folds. Wraps scripts/evaluation/slide_results.py.
#
#   reads checkpoints from: checkpoints/<run_name>.pth          (saved by 05_run_train_slide.sh)
#   reads features from   : data/TCGA-BRCA/features/<encoder>-<pretext>/   (from 04_precompute_features.sh)
#   writes predictions to : outputs/evaluation/<run_name>/{subtyping.txt, <subtask>.txt}
#   run name              : <base>[-<subtask>]-<variant>[-<add-on>]-<encoder>-<pretext>-fold<idx>
#
# Run after 05. Predicts on each fold's held-out TEST split (the partition the model
# never saw in training). subtyping.txt holds per-slide class probabilities + the
# predicted/true subtype; each <subtask>.txt holds the predicted COSMIC signature
# exposures. Defaults match the clam-full-mb-gated runs trained by 05.
#
# Usage:
#   bash cloud/07_predict_slide.sh                          # sweep: default pretexts x all folds
#   bash cloud/07_predict_slide.sh full                     # only the resnet50-full features
#   FOLD=0 bash cloud/07_predict_slide.sh full              # one fold of one pretext
#   SUBTASK="" bash cloud/07_predict_slide.sh               # plain CLAM (no signature subtasks)
#   SPLIT=val bash cloud/07_predict_slide.sh full           # predict on the val split instead
#   BASE=mil VARIANT=attention ADD_ON=gated SUBTASK="" bash cloud/07_predict_slide.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace/micromamba}"
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

# --- aggregator architecture (defaults reproduce clam-full-mb-gated, as in 05) ---
BASE="${BASE:-clam}"          # clam | mil
VARIANT="${VARIANT:-mb}"      # clam: sb|mb ; mil: mean|max|attention
ADD_ON="${ADD_ON:-gated}"     # "" | gated   (attention-based variants only)
SUBTASK="${SUBTASK:-full}"    # "" | "sbs dbs id cnv" | full  (clam only; needs signature labels)
ENCODER="${ENCODER:-resnet50}"
DATASET="${DATASET:-tcga-brca}"
N_FOLDS="${N_FOLDS:-5}"       # must match params.n_folds in configs/dataset/base-<dataset>.yaml
ROOT="${ROOT:-data/TCGA-BRCA}"
SPLIT="${SPLIT:-test}"        # train | val | test  (test = the held-out fold partition)
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}" # 1 = per-slide; >1 pads bags and leaks padding into attention

# Pretext variants whose feature sets to evaluate on (override via arguments).
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
  # Canonical run name (matches the checkpoint stem composed by the aggregator
  # config: subtasks collapse to `full` when the whole set is selected).
  RUN_NAME="${BASE}${SUBTASK:+-${SUBTASK// /-}}-${VARIANT}${ADD_ON:+-${ADD_ON}}-${ENCODER}-${P}"
  for F in "${FOLDS[@]}"; do
    CKPT="checkpoints/${RUN_NAME}-fold${F}.pth"
    echo
    echo "==================================================================="
    echo ">> predict ${RUN_NAME}  | fold ${F} of ${N_FOLDS} | split ${SPLIT}"
    echo "==================================================================="
    if [ ! -f "$CKPT" ]; then
      echo "!! Aggregator checkpoint not found: $CKPT"
      echo "   Train it first:  FOLD=$F bash cloud/05_run_train_slide.sh $P   (skipping this fold)"
      continue
    fi
    # shellcheck disable=SC2086  # SUBTASK is nargs='*': intentional word-splitting
    micromamba run -n augur python scripts/evaluation/slide_results.py \
      --config-dir configs \
      --base "$BASE" \
      --variant "$VARIANT" \
      ${ADD_ON:+--add-on "$ADD_ON"} \
      ${SUBTASK:+--subtask $SUBTASK} \
      --encoder "$ENCODER" \
      --pretext "$P" \
      --dataset "$DATASET" \
      --n-folds "$N_FOLDS" \
      --fold-idx "$F" \
      --split "$SPLIT" \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --precomputed
  done
done

echo
echo ">> Done. Slide predictions in outputs/evaluation/<run_name>/  (bundle with 06_save_results.sh)."
