#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=VGGT
SOURCE_ENV=Pi3
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment ${ENV_NAME} already exists; leaving it unchanged." >&2
  exit 0
fi
conda create -y -n "${ENV_NAME}" --clone "${SOURCE_ENV}"
conda run -n "${ENV_NAME}" python -m pip install -i https://pypi.org/simple/ -r requirements-train.txt

echo "Environment ${ENV_NAME} is ready. Run: bash scripts/train_td_fusion_8a100.sh"
