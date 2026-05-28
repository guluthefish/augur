#!/bin/sh
#SBATCH --job-name=long-hematoxylin
#SBATCH --gres=gpu:2
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=slurm/output/resnet50-hematoxylin-%j.out
#SBATCH --error=slurm/output/resnet50-hematoxylin-%j.err

set -eu

REPO="/tudelft.net/staff-umbrella/IMGEN"
IMG="$REPO/containers/augur.sif"
OUTDIR="$REPO/slurm/output"

MODEL_NAME="resnet50"
TASK_NAME="hematoxylin"

MODEL_CFG="$REPO/configs/model-$MODEL_NAME-$TASK_NAME.yaml"
DATA_CFG="$REPO/configs/tile_dataset-TCGA-BRCA-$TASK_NAME.yaml"
TRAIN_CFG="$REPO/configs/training-$MODEL_NAME-$TASK_NAME.yaml"
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
case "$MODEL_NAME" in
  *gigapath*)
    echo "Pre-caching $MODEL_NAME weights into the node-local HF cache ..."
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
    'hf_hub:resnet50/resnet50',
    pretrained=True,
    num_classes=0,
    global_pool='',
)
print('Pre-cached resnet50.')
"
    ;;
esac

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
    --model-config "$MODEL_CFG" --dataset-config "$DATA_CFG" --training-config "$TRAIN_CFG"
