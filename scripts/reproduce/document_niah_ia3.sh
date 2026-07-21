#!/bin/bash
# Reproduce the D2L (IA)^3 experiment at one selected multiplier scale (three seeds).
# Usage: scripts/reproduce/document_niah_ia3.sh [DEVICE] [IA3_SCALE]
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE="${1:-cuda:0}"; SCALE="${2:-1.0}"
ROOT="results/repro/document_niah_ia3/scale${SCALE}"
.venv/bin/adapterbench preflight --setting d2l --devices "$DEVICE" --output "$ROOT"
for SEED in 777 778 779; do
  .venv/bin/adapterbench d2p-niah --adapters ia3 --ia3-scaling "$SCALE" --needle-style realistic \
    --context-lengths 256 --eval-context-lengths 256,512,1024,2048,4096,8192 \
    --num-train-documents 512 --steps 12000 --eval-every 1000 --learning-rate 4e-5 \
    --n-latents 208 --num-blocks 8 --eval-limit 32 --seed "$SEED" --device "$DEVICE" \
    --output "$ROOT/s$SEED"
done
.venv/bin/python scripts/d2p_niah_aggregate.py --root "$ROOT" --min-seeds 3 --train-length 256
