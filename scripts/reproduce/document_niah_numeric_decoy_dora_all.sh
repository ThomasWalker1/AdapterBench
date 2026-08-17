#!/bin/bash
# Exact five-seed D2A DoRA confirmation and compact aggregation.
# Sequential on one device by default; pass a GPU CSV to run the seeds in parallel.
#
# Usage: scripts/reproduce/document_niah_numeric_decoy_dora_all.sh [DEVICE_OR_GPU_CSV] [ROOT]
set -euo pipefail
cd "$(dirname "$0")/../.."

TARGET="${1:-cuda:0}"
ROOT="${2:-results/repro/d2a_numeric_decoy/dora}"
SEEDS=(2901 2902 2903 2905 2906)

if [[ "$TARGET" == *,* ]]; then
  IFS=',' read -ra GPUS <<< "$TARGET"
  .venv/bin/adapterbench preflight --setting d2a --devices "$(printf 'cuda:%s,' "${GPUS[@]}" | sed 's/,$//')" --output "$ROOT"
  pids=()
  for index in "${!SEEDS[@]}"; do
    gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
    SKIP_PREFLIGHT=1 bash scripts/reproduce/document_niah_numeric_decoy_dora.sh \
      "cuda:${gpu}" "${SEEDS[$index]}" "$ROOT" &
    pids+=("$!")
    if (( ${#pids[@]} == ${#GPUS[@]} )); then wait "${pids[@]}"; pids=(); fi
  done
  if (( ${#pids[@]} )); then wait "${pids[@]}"; fi
else
  for SEED in "${SEEDS[@]}"; do
    SKIP_PREFLIGHT=1 bash scripts/reproduce/document_niah_numeric_decoy_dora.sh "$TARGET" "$SEED" "$ROOT"
  done
fi
.venv/bin/python scripts/d2a_niah_aggregate.py --root "$ROOT" --min-seeds 5 --train-length 512
