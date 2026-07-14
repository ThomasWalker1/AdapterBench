#!/bin/bash
# Reproduce the LoRA baseline entry in leaderboards/document_niah_d2l.md.
# Base NIAH retrieval: scale ~45 (default), lr 4e-5, 6000 steps, context 384, 512 docs.
# Matched retrieval acc 1.0, context-swap control 0.0 -> matched-control 1.0.
# Runs 3 seeds (the current archived baseline is single-seed; this fills the multi-seed gap).
# Retrieval is a stochastic phase transition (onset ~1.6k-4.5k steps) -- multi-seed is essential.
# Restart-safe: re-run the identical command to resume. Usage: [DEVICE]  (default cuda:0)
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE="${1:-cuda:0}"
for SEED in 777 778 779; do
  .venv/bin/adapterbench d2p-niah --adapters lora --needle-style generic \
    --context-lengths 384 --num-train-documents 512 --steps 6000 --eval-every 500 \
    --learning-rate 4e-5 --n-latents 208 --num-blocks 8 --eval-limit 32 \
    --seed "$SEED" --device "$DEVICE" \
    --output "results/repro/document_niah_lora/s$SEED"
done

# Length-generalization variant (difficulty knob): train short, eval a length sweep.
# for SEED in 777 778 779; do
#   .venv/bin/adapterbench d2p-niah --adapters lora --seed "$SEED" --needle-style generic \
#     --context-lengths 128,256 --eval-context-lengths 256,512,1024,2048,4096,8192 \
#     --num-train-documents 512 --steps 6000 --learning-rate 4e-5 --n-latents 208 --num-blocks 8 \
#     --device "$DEVICE" --output "results/repro/document_lengthgen_lora/s$SEED"
# done
