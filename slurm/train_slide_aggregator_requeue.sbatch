#!/bin/sh
#SBATCH --job-name=mbg-re
#SBATCH --gres=gpu:2
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/output/clam-mb-gated-resnet50-full-%j.out
#SBATCH --error=slurm/output/clam-mb-gated-resnet50-full-%j.err

# Note: do not 'set -e' here — we want to survive a non-zero srun exit so we
# can run the post-step requeue check below.
set -u

REPO="/tudelft.net/staff-umbrella/IMGEN"
IMG="$REPO/containers/vexdr.sif"
OUTDIR="$REPO/slurm/output"

CONFIG_DIR="$REPO/configs"
TRAINING_CFG="training-aggregator.yaml"
DATA_CFG="slide_dataset-TCGA-BRCA.yaml"
AGGREGATOR_NAME="clam-mb-gated"
ENCODER_ARCHITECTURE="resnet50"
TILE_PRETEXT="full"
export PYTHONPATH="/work${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

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
case "$ENCODER_ARCHITECTURE" in
  *gigapath*)
    echo "Pre-caching $ENCODER_ARCHITECTURE weights into the node-local HF cache ..."
    apptainer exec \
        -B "$REPO:/work" \
        -B "$tmpdir:/tmp" \
        --env HF_HOME=/tmp/huggingface \
        --pwd /work \
        "$IMG" \
        /opt/conda/envs/vexdr/bin/python -c "
from dotenv import load_dotenv
load_dotenv()
import timm
timm.create_model(
    'hf_hub:prov-gigapath/prov-gigapath',
    pretrained=True,
    num_classes=0,
    global_pool='',
)
print('Pre-cached prov-gigapath.')
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
    --env TORCH_NCCL_BLOCKING_WAIT=1 \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    --pwd /work \
    "$IMG" \
    /opt/conda/envs/vexdr/bin/python scripts/model_training/train_slide_aggregator.py \
    --config-dir "$CONFIG_DIR" \
    --training-config "$TRAINING_CFG" \
    --data-config "$DATA_CFG" \
    --aggregator-name "$AGGREGATOR_NAME" \
    --encoder-architecture "$ENCODER_ARCHITECTURE" \
    --tile-pretext "$TILE_PRETEXT"
SRUN_EXIT=$?

# Lightning's auto-requeue handler runs `scontrol requeue` from inside the
# container, where scontrol is not on the PATH (exit code 127). It still
# saves last.ckpt successfully. Detect that case from the .err log and
# requeue from the host, where scontrol IS available.
ERR_FILE="$OUTDIR/$AGGREGATOR_NAME-$ENCODER_ARCHITECTURE-$TILE_PRETEXT-${SLURM_JOB_ID}.err"
if grep -q "Handling auto-requeue signal" "$ERR_FILE" 2>/dev/null; then
    echo "Lightning requested requeue (scontrol failed inside container). Requeuing from host..."
    scontrol requeue "$SLURM_JOB_ID"
fi

exit "$SRUN_EXIT"
