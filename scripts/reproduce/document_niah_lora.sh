#!/bin/bash
# Reproduce the LoRA baseline entry in leaderboards/document_niah_d2l.md.
# Unified D2L NIAH: one run per seed trains at 256 tokens on a REALISTIC haystack (needle among
# real Wikipedia prose, BoolQ passages) and evaluates at 256 (in-distribution headline) AND a
# length sweep to 8192 (the difficulty knob). scale ~45 (default), lr 4e-5, 12000 steps, 512 docs.
# Headline (in-distribution, 256): matched-control = +0.887 +/- 0.143 over 5 seeds, ctxswap 0.000.
# Length-gen crossover(0.5) = 4096 tokens = 16x the 256 training length.
# Retrieval is a stochastic phase transition (onset ~3.7k-5.4k steps) -- multi-seed is essential.
# Restart-safe: re-run the identical command to resume. Usage: [DEVICE]  (default cuda:0)
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE="${1:-cuda:0}"
.venv/bin/adapterbench preflight --setting d2l --devices "$DEVICE" --output results/repro/document_niah_realistic
for SEED in 777 778 779 780 781; do
  .venv/bin/adapterbench d2p-niah --adapters lora --needle-style realistic \
    --context-lengths 256 --eval-context-lengths 256,512,1024,2048,4096,8192 \
    --num-train-documents 512 --steps 12000 --eval-every 1000 \
    --learning-rate 4e-5 --n-latents 208 --num-blocks 8 --eval-limit 32 \
    --seed "$SEED" --device "$DEVICE" \
    --output "results/repro/document_niah_realistic/s$SEED"
done

.venv/bin/python scripts/d2p_niah_aggregate.py \
  --root results/repro/document_niah_realistic --min-seeds 5 --train-length 256
