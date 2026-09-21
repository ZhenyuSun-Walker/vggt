#!/usr/bin/env bash
set -euo pipefail

TDHOMES_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TDHOMES_REPO_ROOT"

TDHOMES_PYTHON="${TDHOMES_PYTHON:-/home/ma-user/anaconda3/envs/vggt/bin/python}"
TDHOMES_MODEL="${TDHOMES_MODEL:-ckpts/VGGT-1B/model.safetensors}"
TDHOMES_OUTPUT="${TDHOMES_OUTPUT:-evals/mv-recon_VGGT}"
TDHOMES_GPUS="${TDHOMES_GPUS:-0}"
TDHOMES_EXTRA_ARGS="${TDHOMES_EXTRA_ARGS:-}"
TDHOMES_OMP_NUM_THREADS="${TDHOMES_OMP_NUM_THREADS:-}"
if [[ -n "$TDHOMES_OMP_NUM_THREADS" ]]; then
  export OMP_NUM_THREADS="$TDHOMES_OMP_NUM_THREADS"
else
  unset OMP_NUM_THREADS
fi
read -r -a TDHOMES_GPU_IDS <<< "$TDHOMES_GPUS"
read -r -a TDHOMES_EXTRA_ARG_LIST <<< "$TDHOMES_EXTRA_ARGS"
TDHOMES_NUM_SHARDS="${#TDHOMES_GPU_IDS[@]}"
TDHOMES_LOG_DIR="${TDHOMES_LOG_DIR:-$TDHOMES_OUTPUT/logs}"
mkdir -p "$TDHOMES_LOG_DIR"

TDHOMES_PIDS=()
for TDHOMES_SHARD_INDEX in "${!TDHOMES_GPU_IDS[@]}"; do
  TDHOMES_GPU_ID="${TDHOMES_GPU_IDS[$TDHOMES_SHARD_INDEX]}"
  TDHOMES_LOG="$TDHOMES_LOG_DIR/shard-${TDHOMES_SHARD_INDEX}.log"
  echo "Starting mv_recon shard $TDHOMES_SHARD_INDEX/$TDHOMES_NUM_SHARDS on GPU $TDHOMES_GPU_ID"
  CUDA_VISIBLE_DEVICES="$TDHOMES_GPU_ID" "$TDHOMES_PYTHON" evaluation/eval_mv_recon_td_homes.py \
    --model "$TDHOMES_MODEL" \
    --output-dir "$TDHOMES_OUTPUT" \
    --num-shards "$TDHOMES_NUM_SHARDS" \
    --shard-index "$TDHOMES_SHARD_INDEX" \
    --skip-aggregate \
    "${TDHOMES_EXTRA_ARG_LIST[@]}" \
    >"$TDHOMES_LOG" 2>&1 &
  TDHOMES_PIDS+=("$!")
done

TDHOMES_STATUS=0
for TDHOMES_PID in "${TDHOMES_PIDS[@]}"; do
  wait "$TDHOMES_PID" || TDHOMES_STATUS=1
done
if [[ "$TDHOMES_STATUS" -ne 0 ]]; then
  echo "At least one mv_recon shard failed; inspect $TDHOMES_LOG_DIR" >&2
  exit "$TDHOMES_STATUS"
fi

"$TDHOMES_PYTHON" evaluation/eval_mv_recon_td_homes.py \
  --model "$TDHOMES_MODEL" \
  --output-dir "$TDHOMES_OUTPUT" \
  --num-shards "$TDHOMES_NUM_SHARDS" \
  --aggregate-only \
  "${TDHOMES_EXTRA_ARG_LIST[@]}"

echo "Complete mv_recon results: $TDHOMES_OUTPUT"
