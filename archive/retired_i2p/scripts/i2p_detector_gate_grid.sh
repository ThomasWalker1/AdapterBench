#!/usr/bin/env bash
set -euo pipefail

# Absolute-erasure follow-up: target CLIP penalty is twice the retain reward.
# Each row varies LoRA scale and pixel-fidelity strength, then runs the
# independent Grounding DINO gate on the finished checkpoint.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pids=()
for spec in \
  "0 1.0 0.25 e2_s1_f025" \
  "1 2.0 0.25 e2_s2_f025" \
  "2 2.0 1.00 e2_s2_f100" \
  "3 4.0 1.00 e2_s4_f100"
do
  read -r gpu scale fidelity tag <<<"$spec"
  output="results/i2p_selective_detector_${tag}"
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
        --erase-weight 2 \
        --retain-weight 1 \
        --fidelity-weight "$fidelity" \
        --detector-eval \
        --save-grid \
        --log-every 10 \
        --checkpoint-every 25 \
        --output "$output"
  ) >"$output/run.log" 2>&1 &
  pids+=("$!")
  echo "gpu=$gpu scale=$scale fidelity=$fidelity pid=${pids[-1]} output=$output"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
