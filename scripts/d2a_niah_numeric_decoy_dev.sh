#!/bin/bash
# Development-only scale selection for a single queried needle plus numeric decoys.
# Usage: scripts/d2a_niah_numeric_decoy_dev.sh ADAPTER GPU_CSV SCALES_CSV [SEED] [ROOT]
set -euo pipefail
cd "$(dirname "$0")/.."

export NEEDLE_STYLE=realistic_numeric_decoys
export NUMERIC_DECOY_COUNT=4
export TRAIN_CONTEXT_LENGTHS=512
export GATE_LENGTH=512
export EVAL_LENGTHS=1024,2048,4096,8192,16384,32768
export EVAL_LIMIT=12
export EVAL_SEED=1802
export RUNG1_STEPS=8000
export RUNG2_STEPS=16000
export FINAL_STEPS=32000
export WARMUP_STEPS=960

exec bash scripts/d2a_niah_multifidelity.sh "$@"
