#!/bin/bash
# Exact three-seed D2A LoKr reproduction and compact aggregation.
# Usage: scripts/reproduce/document_niah_numeric_decoy_lokr_all.sh [DEVICE] [ROOT]
set -euo pipefail
cd "$(dirname "$0")/../.."

DEVICE="${1:-cuda:0}"
ROOT="${2:-results/repro/d2a_numeric_decoy/lokr}"
for SEED in 3003 3004 3005; do
  SKIP_PREFLIGHT=1 bash scripts/reproduce/document_niah_numeric_decoy_lokr.sh "$DEVICE" "$SEED" "$ROOT"
done
.venv/bin/python scripts/d2a_niah_aggregate.py --root "$ROOT" --min-seeds 3 --train-length 512
