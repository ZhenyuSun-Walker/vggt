#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "VGGT" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate VGGT
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [[ "${NPROC_PER_NODE}" != "8" ]]; then
  echo "This launcher is configured for exactly 8 local A100 processes (NPROC_PER_NODE=8)." >&2
  exit 2
fi
if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
  GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  GPU_COUNT="$(printf '%s\n' "${GPU_NAMES}" | sed '/^$/d' | wc -l)"
  if [[ "${GPU_COUNT}" -lt 8 ]] || ! printf '%s\n' "${GPU_NAMES}" | grep -qi 'A100'; then
    echo "Expected at least 8 NVIDIA A100 GPUs; found:" >&2
    printf '%s\n' "${GPU_NAMES}" >&2
    exit 2
  fi
fi

export PYTHONPATH="${ROOT_DIR}/training:${ROOT_DIR}:${PYTHONPATH:-}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec torchrun --standalone --nproc_per_node=8 training/launch.py --config-name td_fusion_8a100 "$@"
