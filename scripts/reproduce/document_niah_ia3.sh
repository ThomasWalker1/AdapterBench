#!/bin/bash
# Reproduce the canonical D2L (IA)^3 row in leaderboards/document_niah_d2l.md.
# Selected by the IA3-only autoresearch protocol: scale 64, lr 2e-5, warmup 0.03,
# 36k steps, realistic haystack, 256-token training length, and three confirmation seeds.
# Usage: scripts/reproduce/document_niah_ia3.sh [DEVICE]
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE="${1:-cuda:0}"
SCALE="64"
ROOT="results/repro/document_niah_ia3/scale${SCALE}"
.venv/bin/adapterbench preflight --setting d2l --devices "$DEVICE" --output "$ROOT"
for SEED in 786 787 788; do
  .venv/bin/adapterbench d2p-niah --adapters ia3 --ia3-scaling "$SCALE" --needle-style realistic \
    --context-lengths 256 --eval-context-lengths 256,512,1024,2048,4096,8192 \
    --num-train-documents 512 --steps 36000 --eval-every 1000 --learning-rate 2e-5 --warmup-frac 0.03 \
    --n-latents 208 --num-blocks 8 --eval-limit 32 --seed "$SEED" --device "$DEVICE" \
    --output "$ROOT/s$SEED"
done
.venv/bin/python scripts/d2p_niah_aggregate.py --root "$ROOT" --min-seeds 3 --train-length 256
