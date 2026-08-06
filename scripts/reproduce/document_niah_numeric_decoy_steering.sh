#!/bin/bash
# Exact one-seed D2A steering reproduction.  The selected scale/LR are frozen from
# the logged autoresearch search; this script never performs selection.
#
# Selected point: --codec-scaling 32, --learning-rate 2e-5, 32000 steps, warmup 960.
# Scale 32 came from a x2 refinement of steering's own x4 identity-centred ladder:
# scale 64 retrieves as well but converges on only 1 of 3 selection seeds, and
# 128/256 diverge outright, so the stable band is narrow and 32 sits inside it.
#
# Usage: scripts/reproduce/document_niah_numeric_decoy_steering.sh [DEVICE] [SEED] [ROOT]
set -euo pipefail
cd "$(dirname "$0")/../.."

DEVICE="${1:-cuda:0}"
SEED="${2:-2901}"
ROOT="${3:-results/repro/d2a_numeric_decoy/steering}"
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  .venv/bin/adapterbench preflight --setting d2a --devices "$DEVICE" --output "$ROOT"
fi
.venv/bin/adapterbench d2a-niah --adapters steering --codec-scaling 32 \
  --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
  --context-lengths 512 --eval-context-lengths 512,1024,2048,4096,8192,16384,32768 \
  --num-train-documents 512 --steps 32000 --eval-every 8000 --learning-rate 2e-5 --warmup-steps 960 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 --eval-seed 2904 \
  --seed "$SEED" --device "$DEVICE" --output "$ROOT/s$SEED"
