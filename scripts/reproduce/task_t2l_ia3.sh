#!/bin/bash
# Reproduce one selected T2L (IA)^3 seed.
#
# Selected free HPs: scale=16, lr=4e-4, warmup=0.1, and 8,000 steps.
# Usage: scripts/reproduce/task_t2l_ia3.sh [SEED] [GPUS_HYPER] [GPUS_STATIC]
#   e.g. scripts/reproduce/task_t2l_ia3.sh 1901 0,1,2,3 4,5,6,7
set -euo pipefail
cd "$(dirname "$0")/../.."
SEED="${1:-1901}"; GH="${2:-0,1,2,3}"; GS="${3:-4,5,6,7}"
ROOT="${ROOT:-results/repro/t2l_ia3_scale16_lr4e-4}"
STEPS=8000 LR=4e-4 LIMIT=40 PGB=16 CE_LIMIT=64 ACC_LIMIT=48 \
  bash scripts/t2l_codec_trial.sh ia3 16 "$SEED" "$GH" "$GS" "$ROOT"
