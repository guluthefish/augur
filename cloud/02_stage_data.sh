#!/usr/bin/env bash
# Stage TCGA-BRCA onto the pod by downloading straight from GDC (open-access, no token),
# via the repo's own scripts/data_handling/download_tcga.py in the data_helper env.
# Everything lands on the pod's container disk under data/TCGA-BRCA/ (ephemeral) -- run
# this once per pod session, then do your whole sweep before terminating.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="data/config/TCGA-BRCA.yaml"
ROOT="data/TCGA-BRCA"
MANIFEST_NAME="gdc_manifest.2026-01-12.144408.txt"   # must match manifest_raw: in $CONFIG
RAW_DIR="$ROOT/manifests/raw"
MAX_FILES="${MAX_FILES:-}"                            # optional: download only the first N files (subset)

# Put the raw GDC manifest where download_tcga.py expects it. A tracked copy ships in
# cloud/ so a fresh clone is self-contained.
mkdir -p "$RAW_DIR"
if [ ! -f "cloud/$MANIFEST_NAME" ]; then
  echo "!! Manifest not found at cloud/$MANIFEST_NAME"
  echo "   Commit it with the repo (it's ~4 MB), or scp it to the pod, then re-run."
  exit 1
fi
if [ -n "$MAX_FILES" ]; then
  echo ">> Subsetting manifest to first $MAX_FILES files"
  head -n "$((MAX_FILES + 1))" "cloud/$MANIFEST_NAME" > "$RAW_DIR/$MANIFEST_NAME"   # +1 keeps the header
else
  cp "cloud/$MANIFEST_NAME" "$RAW_DIR/$MANIFEST_NAME"
fi

eval "$(micromamba shell hook -s bash)"
echo ">> Downloading TCGA-BRCA from GDC (open-access; a few hours for the full set)..."
micromamba run -n data_helper python scripts/data_handling/download_tcga.py --config "$CONFIG"
echo ">> Slides staged under $ROOT/ordered_data/"
