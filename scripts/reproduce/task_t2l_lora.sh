#!/bin/bash
# Reproduce the LoRA baseline entry in leaderboards/task_conditioned_t2l.md.
# Paper-matched recipe: 128 descriptions, batch 8, lr 2.5e-5, warmup 0.1, eval-limit 80,
# adversarial control, over 3 seeds. Matched-adversarial ~+0.033 across all 4 families.
# Default 150k steps is the confirmed controlled positive; pass STEPS=1000000 for paper scale.
# Restart-safe (--checkpoint-every 10000): re-run the identical command to resume.
# Usage: scripts/reproduce/task_t2l_lora.sh [DEVICE] [STEPS]   (defaults cuda:0, 150000)
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE="${1:-cuda:0}"
STEPS="${2:-150000}"
for SEED in 777 778 779; do
  .venv/bin/adapterbench t2p-sft-pilot --all-decontam-tasks --adapters lora \
    --max-descriptions 128 --batch-size 8 --grad-accum-steps 1 \
    --learning-rate 2.5e-5 --warmup-frac 0.1 \
    --eval-tasks arc_easy,arc_challenge,hellaswag,boolq --eval-limit 80 \
    --adversarial-control --seeds "$SEED" --steps "$STEPS" \
    --checkpoint-every 10000 --device "$DEVICE" \
    --output "results/repro/task_t2l_lora/s$SEED"
  .venv/bin/python scripts/t2p_rigor_aggregate.py \
    --results "results/repro/task_t2l_lora/s$SEED/results.jsonl"
done
