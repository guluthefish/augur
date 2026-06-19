#!/usr/bin/env bash
# Render DualCLAM-MB attention heatmaps for every BCSS tissue-labeled slide.
# Wraps scripts/evaluation/attention_heat_map.py, one invocation per slide.
#
#   reads checkpoint from : checkpoints/clam[-<subtask>]-mb[-gated]-<encoder>-<pretext>.pth
#   reads features from   : data/TCGA-BRCA/features/<encoder>-<pretext>/
#   reads tissue labels   : data/TCGA-BRCA/metadata/bcss_{roi_bounds,slide_metadata}.csv
#   writes figures to     : outputs/attention_<PRETEXT>_<SUBTASK>/<slide_name>/
#
# The set of slides is exactly the BCSS-annotated cohort (rows of
# bcss_roi_bounds.csv). Each slide's ROI is taken from that file, so the chosen
# ROI lands on the annotated region and the tissue-label overlay (figure 6)
# renders. The aggregator is always multi-branch (DualCLAM-MB), as required by
# the Python script.
#
# Args (positional):
#   $1 = SUBTASK   one of: sbs | dbs | id | cnv | full   (COSMIC signature head)
#   $2 = PRETEXT   one of: full | hematoxylin | jigmag | magnification | none
#
# Env overrides:
#   TASK=subtyping     attention branch to visualize (subtyping | <x>_regression | cnv ...)
#   ENCODER=resnet50   encoder token
#   DATASET=tcga-brca  dataset base token (selects configs/dataset/base-<DATASET>.yaml)
#   ROOT=data/TCGA-BRCA          dataset root (must match the dataset config root_dir)
#   ADD_ON=gated       "" to drop the gated add-on
#   DEVICE=cuda        auto | cpu | cuda | cuda:N
#   TOP_K=5            tiles per extreme-attention row
#
# Usage:
#   bash cloud/08_attention_heat_map.sh cnv full              # cnv head, full-pretext encoder
#   TASK=cnv_regression bash cloud/08_attention_heat_map.sh cnv full
#   bash cloud/08_attention_heat_map.sh full magnification    # all four heads, magnification encoder
set -euo pipefail
cd "$(dirname "$0")/.."

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace/micromamba}"
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

# --- positional args ---
SUBTASK="${1:-}"
PRETEXT="${2:-}"
if [ -z "$SUBTASK" ] || [ -z "$PRETEXT" ]; then
  echo "Usage: bash cloud/08_attention_heat_map.sh <SUBTASK> <PRETEXT>" >&2
  echo "  SUBTASK: sbs | dbs | id | cnv | full" >&2
  echo "  PRETEXT: full | hematoxylin | jigmag | magnification | none" >&2
  exit 2
fi

# --- config (env overridable) ---
TASK="${TASK:-subtyping}"
ENCODER="${ENCODER:-resnet50}"
DATASET="${DATASET:-tcga-brca}"
ROOT="${ROOT:-data/TCGA-BRCA}"
ADD_ON="${ADD_ON:-gated}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-5}"

# Path-safe tag for the output folder (SUBTASK may be multi-token, e.g. "sbs dbs").
SUBTASK_TAG="${SUBTASK// /-}"
OUT_ROOT="outputs/attention_${PRETEXT}_${SUBTASK_TAG}"

ROI_BOUNDS="$ROOT/metadata/bcss_roi_bounds.csv"
SLIDE_META="$ROOT/metadata/bcss_slide_metadata.csv"
FEAT_DIR="$ROOT/features/${ENCODER}-${PRETEXT}"

for f in "$ROI_BOUNDS" "$SLIDE_META"; do
  if [ ! -f "$f" ]; then
    echo "!! Tissue metadata not found: $f" >&2
    echo "   Stage the BCSS metadata first (see cloud/02_stage_data.sh)." >&2
    exit 1
  fi
done

# Canonical subtask token for the checkpoint stem (mirrors
# augur.utils.config.load_aggregator_config: the full set collapses to "full").
EXPANDED="$(for t in $SUBTASK; do
  if [ "$t" = "full" ]; then printf 'sbs\ndbs\nid\ncnv\n'; else printf '%s\n' "$t"; fi
done | sort -u)"
if [ "$(printf '%s\n' "$EXPANDED" | grep -c .)" -eq 4 ]; then
  CKPT_SUBTASK="full"
else
  CKPT_SUBTASK="$(printf '%s' "$EXPANDED" | paste -sd- -)"
fi
CKPT="checkpoints/clam${CKPT_SUBTASK:+-${CKPT_SUBTASK}}-mb${ADD_ON:+-${ADD_ON}}-${ENCODER}-${PRETEXT}-fold0.pth"

echo "=== attention heatmaps :: subtask=${SUBTASK} pretext=${PRETEXT} task=${TASK} ==="
echo "    checkpoint : $CKPT"
echo "    features   : $FEAT_DIR"
echo "    output     : $OUT_ROOT/<slide_name>/"

if [ ! -d "$FEAT_DIR" ]; then
  echo "!! WARNING: feature cache not found: $FEAT_DIR (slides will be encoded live, slowly)." >&2
fi

# Join ROI bounds with slide metadata on slide_name. Each tissue slide has one
# ROI row. Emits tab-separated: slide_name xmin ymin xmax ymax submitter filename
JOINED="$(awk -F, '
  NR==FNR { if (FNR > 1) { submitter[$5]=$1; fname[$5]=$2 } next }
  FNR > 1 {
    sn=$1
    if (sn in submitter)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", sn, $2, $3, $4, $5, submitter[sn], fname[sn]
  }
' "$SLIDE_META" "$ROI_BOUNDS")"

TOTAL="$(printf '%s\n' "$JOINED" | grep -c . || true)"
echo "    slides     : $TOTAL BCSS-annotated"
echo

OK=0; FAIL=0; SKIP=0
while IFS=$'\t' read -r SLIDE_NAME XMIN YMIN XMAX YMAX SUBMITTER FILENAME; do
  [ -z "$SLIDE_NAME" ] && continue

  ROI_W=$((XMAX - XMIN))
  ROI_H=$((YMAX - YMIN))

  # Locate the .svs under ordered_data (the file_id subdir is a content hash).
  SLIDE_PATH="$(find "$ROOT/ordered_data/$SUBMITTER/images" -name "$FILENAME" -print -quit 2>/dev/null || true)"
  if [ -z "$SLIDE_PATH" ]; then
    SLIDE_PATH="$(find "$ROOT/ordered_data" -name "$FILENAME" -print -quit 2>/dev/null || true)"
  fi
  if [ -z "$SLIDE_PATH" ]; then
    echo "-- skip ${SLIDE_NAME}: slide file not found ($FILENAME)"
    SKIP=$((SKIP + 1))
    continue
  fi

  # Require precomputed features. compute_slide_attention() looks up
  # <features-dir>/<slide_id>.pt, where slide_id is the file_id directory name;
  # without it the Python script falls back to live tile encoding (slow, and
  # needs an encoder config this batch does not supply). Skip rather than crash.
  SLIDE_ID="$(basename "$(dirname "$SLIDE_PATH")")"
  if [ ! -f "$FEAT_DIR/$SLIDE_ID.pt" ]; then
    echo "-- skip ${SLIDE_NAME}: no cached features ($FEAT_DIR/$SLIDE_ID.pt)"
    SKIP=$((SKIP + 1))
    continue
  fi

  OUT_DIR="${OUT_ROOT}/${SLIDE_NAME}"
  echo ">> ${SLIDE_NAME}  roi=(${XMIN},${YMIN},${ROI_W},${ROI_H}) -> ${OUT_DIR}"

  # shellcheck disable=SC2086  # SUBTASK is nargs='*': intentional word-splitting
  if micromamba run -n augur python scripts/evaluation/attention_heat_map.py \
      --task "$TASK" \
      --subtask $SUBTASK \
      ${ADD_ON:+--add-on "$ADD_ON"} \
      --encoder "$ENCODER" \
      --pretext "$PRETEXT" \
      --dataset "$DATASET" \
      --features-dir "$FEAT_DIR" \
      --slide-path "$SLIDE_PATH" \
      --roi "$XMIN" "$YMIN" "$ROI_W" "$ROI_H" \
      --output-dir "$OUT_DIR" \
      --top-k "$TOP_K" \
      --device "$DEVICE" \
      --tissue-label; then
    OK=$((OK + 1))
  else
    echo "!! failed: ${SLIDE_NAME}"
    FAIL=$((FAIL + 1))
  fi
done < <(printf '%s\n' "$JOINED")

echo
echo ">> Done. ok=${OK} failed=${FAIL} skipped=${SKIP} of ${TOTAL}. Figures in ${OUT_ROOT}/<slide_name>/."
