#!/usr/bin/env bash
set -euo pipefail

# Four independent, resumable selective-erasure scale probes.  CUDA remapping
# makes each child use its assigned physical GPU as cuda:0.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pids=()
for spec in "0 0.5" "1 1.0" "2 2.0" "3 4.0"; do
  read -r gpu scale <<<"$spec"
  output="results/i2p_selective_erase_scale_${scale}"
  mkdir -p "$output"
  (
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" \
      .venv/bin/python scripts/i2p_selective_erasure_probe.py \
        --device cuda:0 \
        --steps 300 \
        --scene-batch-size 2 \
        --train-scenes 128 \
        --eval-scenes 32 \
        --eval-seeds 2 \
        --eval-batch-size 4 \
        --scale "$scale" \
        --log-every 10 \
        --checkpoint-every 25 \
        --output "$output"
  ) >"$output/run.log" 2>&1 &
  pids+=("$!")
  echo "gpu=$gpu scale=$scale pid=${pids[-1]} output=$output"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
