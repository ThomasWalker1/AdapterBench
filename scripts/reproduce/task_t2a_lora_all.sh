#!/bin/bash
# Full three-seed T2A reference reproduction; each seed is independently restart-safe.
set -euo pipefail
cd "$(dirname "$0")/../.."
HYPER_GPUS="${1:-0,1,2,3}"
STATIC_GPUS="${2:-4,5,6,7}"
ROOT="${ROOT:-results/repro/t2a_lora_scale22.627417_lr1e-4}"
export ROOT
for SEED in 1801 1802 1803; do
  bash scripts/reproduce/task_t2a_lora.sh "$SEED" "$HYPER_GPUS" "$STATIC_GPUS"
done
.venv/bin/python scripts/t2a_release_aggregate.py --root "$ROOT/hyper" --seeds 1801,1802,1803
