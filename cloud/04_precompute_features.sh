#!/usr/bin/env bash
# Precompute frozen-encoder tile features for the slide-level aggregator, once per
# trained ResNet50 encoder. Wraps scripts/model_training/precompute_tile_features.py.
#
#   reads weights from : checkpoints/<encoder>-<pretext>.pth   (saved by 03_run_train_tile.sh)
#   writes features to : data/TCGA-BRCA/features/<encoder>-<pretext>/<slide_id>.pt
#                        (auto-derived to match the aggregator's feature flavor)
#
# Run after tile-encoder training, in the same pod session. The encoder forward is
# frozen (fast), but it reads every tissue tile of every slide, so the slides must
# still be staged. --skip-existing is on by default, so re-runs resume.
#
# Usage:
#   bash cloud/04_precompute_features.sh                       # default sweep below
#   bash cloud/04_precompute_features.sh full jigmag           # only these pretexts
#   CHUNK_SIZE=256 PRECISION=bfloat16 bash cloud/04_precompute_features.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace/micromamba}"
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

ENCODER="${ENCODER:-resnet50}"
DATASET="${DATASET:-tcga-brca}"
PRECISION="${PRECISION:-float16}"   # frozen forward -> fp16 is faster & lighter
CHUNK_SIZE="${CHUNK_SIZE:-128}"     # tiles per GPU forward pass
DEVICE="${DEVICE:-cuda}"

# Pretext variants to precompute (one trained encoder each). Override via arguments.
PRETEXTS=("$@")
if [ "${#PRETEXTS[@]}" -eq 0 ]; then
  PRETEXTS=(full jigmag magnification hematoxylin)
fi

for P in "${PRETEXTS[@]}"; do
  CKPT="checkpoints/${ENCODER}-${P}.pth"
  echo
  echo "==================================================================="
  echo ">> precompute features: ${ENCODER}-${P}"
  echo "==================================================================="
  if [ ! -f "$CKPT" ]; then
    echo "!! Trained encoder not found: $CKPT"
    echo "   Train it first:  bash cloud/03_run_train_tile.sh $ENCODER $P $DATASET long"
    echo "   (training saves the encoder weights to $CKPT). Skipping this variant."
    continue
  fi
  micromamba run -n augur python scripts/model_training/precompute_tile_features.py \
    --encoder "$ENCODER" \
    --pretext "$P" \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --precision "$PRECISION" \
    --chunk-size "$CHUNK_SIZE"
done

echo
echo ">> Done. Per-slide features under data/TCGA-BRCA/features/${ENCODER}-<pretext>/"
