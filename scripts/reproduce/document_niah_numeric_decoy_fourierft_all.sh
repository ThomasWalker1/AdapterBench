#!/bin/bash
# Exact five-seed D2A FourierFT reproduction and aggregation.
# Usage: scripts/reproduce/document_niah_numeric_decoy_fourierft_all.sh [DEVICE] [ROOT]
set -euo pipefail
cd "$(dirname "$0")/../.."

DEVICE="${1:-cuda:0}"
ROOT="${2:-results/repro/d2a_numeric_decoy/fourierft}"
for SEED in 5301 5302 5303 5304 5305; do
  SKIP_PREFLIGHT=1 bash scripts/reproduce/document_niah_numeric_decoy_fourierft.sh "$DEVICE" "$SEED" "$ROOT"
done
.venv/bin/python scripts/d2a_niah_aggregate.py --root "$ROOT" --min-seeds 5 --train-length 512
