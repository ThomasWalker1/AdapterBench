#!/bin/bash
# Reproduce one selected T2L LoKr confirmation seed.
# Selected point (AUTORESEARCH §3b stability gate): scale 16, LR 1e-4. The earlier
# LR 2e-4 point was unstable (2/4 seeds diverged); LR 1e-4 converges 3/3.
set -euo pipefail
cd "$(dirname "$0")/../.."
SEED="${1:-2811}"; GH="${2:-0,1,2,3}"; GS="${3:-4,5,6,7}"
ROOT="${ROOT:-results/repro/t2l_lokr_scale16_lr1e-4_confirm}"
STEPS=6000 LR=1e-4 LIMIT=40 PGB=16 CE_LIMIT=64 ACC_LIMIT=48 \
  bash scripts/t2l_codec_trial.sh lokr 16 "$SEED" "$GH" "$GS" "$ROOT"
