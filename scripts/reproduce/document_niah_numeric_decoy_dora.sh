#!/bin/bash
# Exact one-seed D2A DoRA reproduction. The selected scale/LR are frozen from the logged
# autoresearch search; this script never performs selection.
#
# Selected point: --codec-scaling 64, --learning-rate 2e-5, 32000 steps, warmup 960.
# Scale 64 came from a x2 refinement bracketing a window an exploratory screen located:
# DoRA's scale and LR axes INTERACT, so its declared x4 ladder (0.0625-16) swept at the
# substrate-default LR 4e-5 was retrieval-dead at every point, and scale 64 itself is dead
# at 4e-5 but reaches the gate ceiling at 2e-5. Scales 32/64/128 all reach gate 1.000 and
# are a tied set; 64 is the hard-length AUC argmax. One x2 step either side of LR 2e-5
# destroys retrieval (4e-5 dead, 1e-5 collapses to 0.083), so the band is narrow in LR.
#
# Usage: scripts/reproduce/document_niah_numeric_decoy_dora.sh [DEVICE] [SEED] [ROOT]
set -euo pipefail
cd "$(dirname "$0")/../.."

DEVICE="${1:-cuda:0}"
SEED="${2:-2901}"
ROOT="${3:-results/repro/d2a_numeric_decoy/dora}"
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  .venv/bin/adapterbench preflight --setting d2a --devices "$DEVICE" --output "$ROOT"
fi
.venv/bin/adapterbench d2a-niah --adapters dora --codec-scaling 64 \
  --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
  --context-lengths 512 --eval-context-lengths 512,1024,2048,4096,8192,16384,32768 \
  --num-train-documents 512 --steps 32000 --eval-every 8000 --learning-rate 2e-5 --warmup-steps 960 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 --eval-seed 2904 \
  --seed "$SEED" --device "$DEVICE" --output "$ROOT/s$SEED"
