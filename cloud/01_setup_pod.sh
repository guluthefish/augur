#!/usr/bin/env bash
# Bootstrap a fresh RunPod / cloud GPU pod for Augur: system tools + the augur
# (training) and data_helper (gdc-client) micromamba envs. Mirrors containers/augur.def.
set -euo pipefail
cd "$(dirname "$0")/.."

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace/micromamba}"
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

# 0. Handy system tools (the base image is minimal)
echo ">> Installing tmux / screen ..."
apt-get update -qq && apt-get install -y -qq tmux screen >/dev/null 2>&1 || true

# 1. micromamba (skip if already present)
if ! command -v micromamba >/dev/null 2>&1; then
  echo ">> Installing micromamba..."
  mkdir -p "$MAMBA_ROOT_PREFIX"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "$MAMBA_ROOT_PREFIX" bin/micromamba
fi

# Make micromamba available in future interactive shells (new tmux panes, fresh SSH)
if ! grep -q 'micromamba/bin' "$HOME/.bashrc" 2>/dev/null; then
  echo "export MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX" >> "$HOME/.bashrc"
  echo 'export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"'  >> "$HOME/.bashrc"
fi

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
echo ">> Setup complete. Open a new shell (or 'source ~/.bashrc') and micromamba is on PATH."
