#!/bin/bash
# Exact three-seed T2A FourierFT reproduction and aggregation.
set -euo pipefail
cd "$(dirname "$0")/../.."
HYPER_GPUS="${1:-0,1,2,3}"; STATIC_GPUS="${2:-4,5,6,7}"
ROOT="${ROOT:-results/repro/t2a_fourierft_scale16_lr1e-4_confirm}"
export ROOT
for SEED in 5111 5112 5113; do
  bash scripts/reproduce/task_t2a_fourierft.sh "$SEED" "$HYPER_GPUS" "$STATIC_GPUS"
done
.venv/bin/python scripts/t2a_release_aggregate.py --root "$ROOT/hyper" --seeds 5111,5112,5113
