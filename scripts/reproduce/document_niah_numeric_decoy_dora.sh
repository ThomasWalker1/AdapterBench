#!/bin/bash
# Exact one-seed D2A DoRA reproduction. The selected scale/LR are frozen from the logged
# autoresearch search; this script never performs selection.
#
# Selected point: --codec-scaling 64, --learning-rate 2e-5, 32000 steps, warmup 960.
# DoRA's scale and LR axes interact, so the point was located by a joint scale x LR sweep and
# then re-derived on protocol: scales 32/64/128 at lr 2e-5 all reach the 512-token gate
# ceiling (hard-length AUC 0.983/1.000/0.983, a tied set of three, 64 the AUC argmax), and the
# LR axis closes interior at 2e-5. The band is narrow in LR - one x2 step either side of 2e-5
# loses retrieval - which is why the learning rate is pinned here rather than left to a default.
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
