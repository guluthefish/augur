#!/usr/bin/env bash
# Stage ALL TCGA-BRCA data onto the pod from the public sources, by running the repo's
# scripts/data_handling/ pipeline in order:
#   1. download_tcga       -> whole-slide images from GDC (+ manifest atlas)
#   2. download_pam50      -> PAM50 subtype labels
#   3. download_bcss       -> BCSS tissue-segmentation masks
#   4. extract_signatures  -> COSMIC SBS/DBS/ID/CN signature labels (SigProfiler)
# All four read the same --config. Everything lands on the pod's container disk under
# data/TCGA-BRCA/ (ephemeral) -- run once per pod session, then do your whole sweep
# before terminating. (Comment out a line below to skip a step.)
set -euo pipefail
cd "$(dirname "$0")/.."

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace/micromamba}"
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

CONFIG="data/config/TCGA-BRCA.yaml"
ROOT="data/TCGA-BRCA"
MANIFEST_NAME="gdc_manifest.2026-01-12.144408.txt"   # must match manifest_raw: in $CONFIG
RAW_DIR="$ROOT/manifests/raw"
MAX_FILES="${MAX_FILES:-}"                            # optional: subset the GDC slide manifest to first N files

# extract_signatures needs a writable volume for SigProfiler's reference genome, which
# it installs on first use (multi-GB, slow the first time).
export SIGPROFILERMATRIXGENERATOR_VOLUME="$PWD/sigprofiler_volume"
mkdir -p "$SIGPROFILERMATRIXGENERATOR_VOLUME"

# --- put the raw GDC manifest where download_tcga.py expects it ---------------------
# A tracked copy ships in cloud/ so a fresh clone is self-contained.
mkdir -p "$RAW_DIR"
if [ ! -f "cloud/$MANIFEST_NAME" ]; then
  echo "!! Manifest not found at cloud/$MANIFEST_NAME"
  echo "   Commit it with the repo (it's ~4 MB), or scp it to the pod, then re-run."
  exit 1
fi
if [ -n "$MAX_FILES" ]; then
  echo ">> Subsetting GDC manifest to first $MAX_FILES files"
  head -n "$((MAX_FILES + 1))" "cloud/$MANIFEST_NAME" > "$RAW_DIR/$MANIFEST_NAME"   # +1 keeps the header
else
  cp "cloud/$MANIFEST_NAME" "$RAW_DIR/$MANIFEST_NAME"
fi

run_step () {   # run_step "<label>" <script.py>
  echo
  echo "==================================================================="
  echo ">> $1"
  echo "==================================================================="
  micromamba run -n data_helper python "scripts/data_handling/$2" --config "$CONFIG"
}

run_step "1/4  download_tcga      (GDC whole-slide images)"     download_tcga.py
run_step "2/4  download_pam50     (PAM50 subtype labels)"       download_pam50.py
run_step "3/4  download_bcss      (BCSS tissue masks)"          download_bcss.py
run_step "4/4  extract_signatures (SigProfiler SBS/DBS/ID/CN)"  extract_signatures.py

echo
echo ">> All data staged under $ROOT/  (slides: ordered_data/, labels: labels/)"
