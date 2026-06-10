#!/usr/bin/env bash
# Create the augur (training) + data_helper (gdc-client) micromamba envs on a
# fresh RunPod / cloud GPU pod. Mirrors containers/augur.def, no apptainer needed.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. micromamba (skip if already present)
if ! command -v micromamba >/dev/null 2>&1; then
  echo ">> Installing micromamba..."
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace/micromamba}"
  mkdir -p "$MAMBA_ROOT_PREFIX"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "$MAMBA_ROOT_PREFIX" bin/micromamba
  export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"
fi
eval "$(micromamba shell hook -s bash)"
export CONDA_OVERRIDE_CUDA=12.4   # mirrors the container build

# 2. Training env
echo ">> Creating training env (augur)..."
micromamba create -y -n augur -f envs/augur.yaml
micromamba run -n augur pip install --no-deps --no-build-isolation -e .

# 3. Data env (includes gdc-client 2.3)
echo ">> Creating data env (data_helper)..."
micromamba create -y -n data_helper -f envs/data_helper.yaml
micromamba run -n data_helper pip install --no-deps --no-build-isolation -e .

# 4. Sanity check
echo ">> Torch sees the GPUs:"
micromamba run -n augur python -c \
  "import torch; print('cuda', torch.cuda.is_available(), 'n_gpu', torch.cuda.device_count())"
echo ">> Setup complete."
