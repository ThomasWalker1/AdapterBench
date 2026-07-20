#!/bin/bash
# Full three-seed T2L reference reproduction; each seed is independently restart-safe.
set -euo pipefail
cd "$(dirname "$0")/../.."
HYPER_GPUS="${1:-0,1,2,3}"
STATIC_GPUS="${2:-4,5,6,7}"
for SEED in 777 2 3; do
  bash scripts/reproduce/task_t2l_lora.sh "$SEED" "$HYPER_GPUS" "$STATIC_GPUS"
done
.venv/bin/python scripts/t2l_release_aggregate.py
