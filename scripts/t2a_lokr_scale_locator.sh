#!/bin/bash
# Detached T2A LoKr scale locator.  The ladder is centered on LoKr's own
# identity-scale convention (scale=1), not on another codec's selected scale.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-2702}"
GH="${HYPER_GPUS:-0,1,2,3}"
GS="${STATIC_GPUS:-4,5,6,7}"
ROOT="${ROOT:-results/autoresearch/t2a/lokr/scale_locator}"
SCALES="${SCALES:-0.0625,0.25,1,4,16}"
IFS=',' read -ra LADDER <<< "$SCALES"

for SCALE in "${LADDER[@]}"; do
  # 6k is a LoKr-owned common scout budget, declared before any scale result;
  # it is deliberately not copied from another codec's selected step count.
  # 1e-4 is the setting's declared DDP default and was only accepted here after
  # the finite LoKr smoke; it is not imported from another codec's selection.
  STEPS=6000 PGB=16 LR=1e-4 CE_LIMIT=24 ACC_LIMIT=0 \
    bash scripts/t2a_lokr_record_trial.sh scale_locator "$SCALE" "$SEED" "$GH" "$GS" \
      "$ROOT/scale${SCALE}" 1e-4 6000 24 0
done
