#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm/output/data-%j.out
#SBATCH --error=slurm/output/data-%j.err

set -eu

REPO="/tudelft.net/staff-umbrella/IMGEN"
IMG="$REPO/containers/data_helper.sif"
OUTDIR="$REPO/slurm/output"
CONFIG="data/config/TCGA-BRCA.yaml"
SIGPROFILER_REF_PATH="$REPO/sigprofiler_volume"

# DAIC: use node-local /tmp for temporary data and clean it up
tmpdir="/tmp/${USER}/${SLURM_JOBID}"
mkdir -p "$tmpdir"
clean_up() { rm -rf "$tmpdir"; }
trap clean_up EXIT

# Run on compute node via srun; execute inside container
srun apptainer exec \
    -B "$REPO:/work" \
    -B "$OUTDIR:/out" \
    -B "$tmpdir:/tmp" \
    --pwd /work \
    "$IMG" \
    env PYTHONPATH=/work SIGPROFILERMATRIXGENERATOR_VOLUME="$SIGPROFILER_REF_PATH" \
    python -m augur.scripts.data_handling.main_script --config "$CONFIG"
