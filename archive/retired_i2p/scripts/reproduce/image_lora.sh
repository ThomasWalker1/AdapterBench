#!/bin/bash
# Reproduce the LoRA baseline entry in leaderboards/image_reward_tilting.md.
# Operating point: scale 4, lambda 0.25, rank 16, 3000 steps, over 3 seeds.
# Matched IR-gain +0.160±0.015; reward-swap (red) control -3.34; matched-control ~+3.50.
# Usage: scripts/reproduce/image_lora.sh [DEVICE]   (default cuda:0)
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE="${1:-cuda:0}"
OUT=results/repro/image_lora
for SEED in 777 778 779; do
  # matched adapter (ImageReward) — also scores the reward-swap (red) control internally
  .venv/bin/adapterbench i2p-hypernoise --reward imagereward \
    --scale 4 --reg-weight 0.25 --steps 3000 --eval-every 500 \
    --rank 16 --batch-size 2 --n-seeds 2 --seed "$SEED" --device "$DEVICE" \
    --output "$OUT/imagereward_s$SEED"
  # reward-swap control adapter (redness) at the same operating point
  .venv/bin/adapterbench i2p-hypernoise --reward red \
    --scale 4 --reg-weight 0.25 --steps 3000 --eval-every 500 \
    --rank 16 --batch-size 2 --n-seeds 2 --seed "$SEED" --device "$DEVICE" \
    --output "$OUT/red_s$SEED"
done
.venv/bin/python scripts/i2p_hypernoise_aggregate.py --out "$OUT"
