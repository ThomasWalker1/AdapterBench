#!/bin/bash
# Reproduce one selected T2A FourierFT confirmation seed.
# Selected by the three-selection-seed stability gate: scale 16, LR 1e-4.
set -euo pipefail
cd "$(dirname "$0")/../.."
SEED="${1:-5111}"; GH="${2:-0,1,2,3}"; GS="${3:-4,5,6,7}"
ROOT="${ROOT:-results/repro/t2a_fourierft_scale16_lr1e-4_confirm}"
STEPS=6000 LR=1e-4 LIMIT=40 PGB=16 CE_LIMIT=64 ACC_LIMIT=48 \
  bash scripts/t2a_codec_trial.sh fourierft 16 "$SEED" "$GH" "$GS" "$ROOT"
