#!/bin/bash
# Reproduce one selected T2L LoKr confirmation seed.
set -euo pipefail
cd "$(dirname "$0")/../.."
SEED="${1:-2704}"; GH="${2:-0,1,2,3}"; GS="${3:-4,5,6,7}"
ROOT="${ROOT:-results/repro/t2l_lokr_scale16_lr2e-4}"
STEPS=6000 LR=2e-4 LIMIT=40 PGB=16 CE_LIMIT=64 ACC_LIMIT=48 \
  bash scripts/t2l_codec_trial.sh lokr 16 "$SEED" "$GH" "$GS" "$ROOT"
