#!/usr/bin/env bash
set -euo pipefail

# Directly train the generated/static LoRAs against the UnHype-style frozen
# denoising target, while keeping Grounding DINO evaluation-only.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pids=()
for spec in \
  "0 1.0 0.5 s1_ng05" \
  "1 2.0 0.5 s2_ng05" \
  "2 1.0 1.0 s1_ng10" \
  "3 2.0 1.0 s2_ng10"
do
  read -r gpu scale guidance tag <<<"$spec"
  output="results/i2p_selective_unhype_${tag}"
  mkdir -p "$output"
  (
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" \
      .venv/bin/python scripts/i2p_selective_erasure_probe.py \
        --device cuda:0 \
        --steps 1000 \
        --scene-batch-size 2 \
        --train-scenes 128 \
        --eval-scenes 32 \
        --eval-seeds 2 \
        --eval-batch-size 4 \
        --scale "$scale" \
        --train-objective diffusion \
        --negative-guidance "$guidance" \
        --detector-eval \
        --save-grid \
        --log-every 25 \
        --checkpoint-every 100 \
        --output "$output"
  ) >"$output/run.log" 2>&1 &
  pids+=("$!")
  echo "gpu=$gpu scale=$scale guidance=$guidance pid=${pids[-1]} output=$output"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
