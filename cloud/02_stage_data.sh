#!/usr/bin/env bash
# Download TCGA-BRCA whole-slide images from GDC straight onto this pod.
# TCGA-BRCA slides are open-access, so no GDC token is required.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="data/config/TCGA-BRCA.yaml"
ROOT="data/TCGA-BRCA"
# Update if your raw manifest filename differs (see `manifest_raw:` in $CONFIG).
MANIFEST="$ROOT/manifests/raw/gdc_manifest.2026-01-12.144408.txt"

mkdir -p "$ROOT/manifests/raw"
if [ ! -f "$MANIFEST" ]; then
  echo "!! Raw manifest missing: $MANIFEST"
  echo "   Copy it from staff-umbrella, e.g.:"
  echo "     scp <user>@<daic>:/tudelft.net/staff-umbrella/IMGEN/$MANIFEST $MANIFEST"
  echo "   then re-run this script."
  exit 1
fi

eval "$(micromamba shell hook -s bash)"

# Real entrypoint. NOTE: slurm/download_data.sbatch references a stale module
# path (augur.scripts.data_handling.main_script) that no longer exists.
echo ">> Downloading + reordering TCGA-BRCA (this can take a few hours)..."
micromamba run -n data_helper python scripts/data_handling/download_tcga.py --config "$CONFIG"

echo ">> Done. Slides are under $ROOT/ordered_data/"
