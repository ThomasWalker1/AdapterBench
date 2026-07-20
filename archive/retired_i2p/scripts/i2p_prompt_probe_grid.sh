#!/bin/bash
# Mechanism-free I2P go/no-go scale probe. One independent paired
# prompt-hypernetwork/static-reference run per GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1

PIDS=()
for SPEC in "0:0.5" "1:1" "2:2" "3:4"; do
  GPU="${SPEC%%:*}"
  SCALE="${SPEC##*:}"
  OUT="results/i2p_prompt_probe_sgd/scale${SCALE}_s777"
  mkdir -p "$OUT"
  echo "[$(date)] launch GPU=$GPU scale=$SCALE -> $OUT"
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/i2p_prompt_conditioning_probe.py \
    --device cuda:0 --steps 300 --batch-size 2 \
    --train-prompts 256 --eval-prompts 64 --eval-batch-size 4 --eval-seeds 1 \
    --rank 4 --scale "$SCALE" --latent-dim 256 --head-dim 256 \
    --optimizer sgd --lr 1e-3 --momentum 0.9 --grad-clip 1 \
    --checkpoint-every 25 --log-every 10 --seed 777 --output "$OUT" \
    > "$OUT/run.log" 2>&1 &
  PIDS+=("$!")
done
wait "${PIDS[@]}"
echo "[$(date)] all I2P prompt-conditioning scale probes finished"
