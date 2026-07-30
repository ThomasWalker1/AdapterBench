#!/bin/bash
# Exact one-seed D2L LoKr reproduction.  The selected scale/LR are frozen from
# the logged scout; this script never performs selection.
# Usage: scripts/reproduce/document_niah_numeric_decoy_lokr.sh [DEVICE] [SEED] [ROOT]
set -euo pipefail
cd "$(dirname "$0")/../.."

DEVICE="${1:-cuda:0}"
SEED="${2:-3003}"
ROOT="${3:-results/repro/d2l_numeric_decoy/lokr}"
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  .venv/bin/adapterbench preflight --setting d2l --devices "$DEVICE" --output "$ROOT"
fi
.venv/bin/adapterbench d2p-niah --adapters lokr --lokr-scaling 16 \
  --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
  --context-lengths 512 --eval-context-lengths 512,1024,2048,4096,8192,16384,32768 \
  --num-train-documents 512 --steps 36000 --eval-every 8000 --learning-rate 2e-5 --warmup-steps 1080 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 --eval-seed 2904 \
  --seed "$SEED" --device "$DEVICE" --output "$ROOT/s$SEED"
