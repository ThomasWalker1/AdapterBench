#!/bin/bash
# Full three-seed reproduction for the measured T2L LoKr result.
set -euo pipefail
cd "$(dirname "$0")/../.."
HYPER_GPUS="${1:-0,1,2,3}"; STATIC_GPUS="${2:-4,5,6,7}"
ROOT="${ROOT:-results/repro/t2l_lokr_scale16_lr1e-4_confirm}"
export ROOT
for SEED in 2811 2812 2813; do
  bash scripts/reproduce/task_t2l_lokr.sh "$SEED" "$HYPER_GPUS" "$STATIC_GPUS"
done
.venv/bin/python scripts/t2l_release_aggregate.py --root "$ROOT/hyper" --seeds 2811,2812,2813
