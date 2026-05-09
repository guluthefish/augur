#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/output/container-image-%j.out
#SBATCH --error=slurm/output/container-image-%j.err

set -eu

REPO="/tudelft.net/staff-umbrella/IMGEN"
DEF="$REPO/containers/vexdr.def"
IMG="$REPO/containers/vexdr.sif"

# DAIC: use node-local /tmp for temporary data and clean it up
tmpdir="/tmp/${USER}/${SLURM_JOBID}"
mkdir -p "$tmpdir"
clean_up() { rm -rf "$tmpdir"; }
trap clean_up EXIT

srun apptainer build \
    "$IMG" "$DEF"
