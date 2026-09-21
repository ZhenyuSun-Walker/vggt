#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TDHOMES_CONDA_BIN="${TDHOMES_CONDA_BIN:-conda}"
TDHOMES_ENV_NAME="${TDHOMES_ENV_NAME:-vggt}"
TDHOMES_SOURCE_ENV="${TDHOMES_SOURCE_ENV:-VGGT}"

if "$TDHOMES_CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$TDHOMES_ENV_NAME"; then
  echo "Conda environment already exists: $TDHOMES_ENV_NAME"
elif "$TDHOMES_CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$TDHOMES_SOURCE_ENV"; then
  "$TDHOMES_CONDA_BIN" create -y -n "$TDHOMES_ENV_NAME" --clone "$TDHOMES_SOURCE_ENV"
else
  "$TDHOMES_CONDA_BIN" create -y -n "$TDHOMES_ENV_NAME" python=3.10 pip
  "$TDHOMES_CONDA_BIN" run -n "$TDHOMES_ENV_NAME" python -m pip install \
    torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
fi

"$TDHOMES_CONDA_BIN" run -n "$TDHOMES_ENV_NAME" python -m pip install \
  --no-deps --no-build-isolation -e "$REPO_ROOT"
"$TDHOMES_CONDA_BIN" run -n "$TDHOMES_ENV_NAME" python -c \
  'import torch, torchvision; from vggt.models.vggt import VGGT; print("torch", torch.__version__); print("torchvision", torchvision.__version__); print("cuda_available", torch.cuda.is_available()); print("VGGT", VGGT.__module__)'
