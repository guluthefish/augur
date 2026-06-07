#!/bin/sh
#SBATCH --job-name=tile-test
#SBATCH --gres=gpu:2
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=slurm/output/test-tile-%j.out
#SBATCH --error=slurm/output/test-tile-%j.err

set -eu

REPO="/tudelft.net/staff-umbrella/IMGEN"
IMG="$REPO/containers/augur.sif"
OUTDIR="$REPO/slurm/output"

# train_tile_encoder.py now composes its configs from the partial-merge
# helpers in augur/utils/config.py. The four tokens below feed:
#   ENCODER  → configs/tile-model/base-<ENCODER>.yaml
#   PRETEXT  → configs/tile-model/pretext-<PRETEXT>-<ENCODER>.yaml
#              (also propagated into the dataset's `tasks` list)
#   DATASET  → configs/dataset/base-<DATASET>.yaml (+ flavor-tile.yaml)
#   TRAINER  → configs/trainer/base-<TRAINER>.yaml
# Edit the SBATCH --job-name / --output / --error above to match when
# changing ENCODER or PRETEXT.
ENCODER="resnet50"
PRETEXT="magnification"
DATASET="tcga-brca"
TRAINER="test"

export PYTHONPATH="/work${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# NCCL transport overrides: GPU peer-to-peer and InfiniBand auto-detection
# hang during the first BROADCAST on some DAIC gpu* nodes. Force TCP sockets,
# which complete reliably (slightly slower but consistent).
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# DAIC: use node-local /tmp for temporary data and clean it up
tmpdir="/tmp/${USER}/${SLURM_JOBID}"
mkdir -p "$tmpdir"
clean_up() { rm -rf "$tmpdir"; }
trap clean_up EXIT

# Stage the HF Hub cache to node-local storage.
hf_cache_local="$tmpdir/huggingface"
if [ -d "$HOME/.cache/huggingface" ]; then
    echo "Staging HF Hub cache to $hf_cache_local ..."
    cp -r "$HOME/.cache/huggingface" "$hf_cache_local"
else
    mkdir -p "$hf_cache_local"
fi

# Pre-cache any HF Hub-backed encoder weights in a single process before srun
# so both ranks see a fully populated cache and do not race on timm/HF filelocks.
# Encoder → timm model id map mirrors configs/tile-model/base-<ENCODER>.yaml.
case "$ENCODER" in
  prov-gigapath) HF_MODEL_ID="hf_hub:prov-gigapath/prov-gigapath" ;;
  *)             HF_MODEL_ID="" ;;
esac
if [ -n "$HF_MODEL_ID" ]; then
    echo "Pre-caching $HF_MODEL_ID into the node-local HF cache ..."
    apptainer exec \
        -B "$REPO:/work" \
        -B "$tmpdir:/tmp" \
        --env HF_HOME=/tmp/huggingface \
        --pwd /work \
        "$IMG" \
        /opt/conda/envs/augur/bin/python -c "
from dotenv import load_dotenv
load_dotenv()
import timm
timm.create_model(
    '$HF_MODEL_ID',
    pretrained=True,
    num_classes=0,
    global_pool='',
)
print('Pre-cached $HF_MODEL_ID.')
"
fi

echo "encoder: $ENCODER"
echo "pretext: $PRETEXT"
echo "dataset: $DATASET"
echo "trainer: $TRAINER"

# Run on compute node via srun; execute inside container
srun apptainer exec \
    --nv \
    -B "$REPO:/work" \
    -B "$OUTDIR:/out" \
    -B "$tmpdir:/tmp" \
    -B "$tmpdir:/dev/shm" \
    --env HF_HOME=/tmp/huggingface \
    --env NCCL_P2P_DISABLE=1 \
    --env NCCL_IB_DISABLE=1 \
    --env NCCL_DEBUG=WARN \
    --pwd /work \
    "$IMG" \
    /opt/conda/envs/augur/bin/python scripts/model_training/train_tile_encoder.py \
        --encoder "$ENCODER" \
        --pretext "$PRETEXT" \
        --dataset "$DATASET" \
        --trainer "$TRAINER"
