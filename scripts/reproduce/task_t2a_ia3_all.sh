#!/bin/bash
# Full three-seed T2A (IA)^3 reproduction at the selected free hyperparameters.
set -euo pipefail
cd "$(dirname "$0")/../.."
HYPER_GPUS="${1:-0,1,2,3}"
STATIC_GPUS="${2:-4,5,6,7}"
ROOT="${ROOT:-results/repro/t2a_ia3_scale16_lr4e-4}"
export ROOT
for SEED in 1901 1902 1903; do
  bash scripts/reproduce/task_t2a_ia3.sh "$SEED" "$HYPER_GPUS" "$STATIC_GPUS"
done
.venv/bin/python scripts/t2a_release_aggregate.py --root "$ROOT/hyper" --seeds 1901,1902,1903
