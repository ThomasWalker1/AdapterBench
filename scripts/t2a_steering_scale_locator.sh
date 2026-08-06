#!/bin/bash
# Detached T2A steering scale locator.  The ladder is centred on steering's own
# identity-scale convention, not on any other codec's selected scale: the update is
# h -> h + scale * v with v emitted by a zero-initialized head, so every scale is
# exactly the frozen model at initialization and the manifest default 1.0 is the
# natural centre.  Points run sequentially because each one uses the released
# effective batch (4 GPUs x PGB) for both the conditioned hypernetwork and its
# equal-shape static reference, consuming all eight GPUs.
#
# Usage: SEED=4702 bash scripts/t2a_steering_scale_locator.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-4702}"
GH="${HYPER_GPUS:-0,1,2,3}"
GS="${STATIC_GPUS:-4,5,6,7}"
ROOT="${ROOT:-results/autoresearch/t2a/steering/scale_locator}"
SCALES="${SCALES:-0.0625,0.25,1,4,16}"
# 8000 steps is steering's declared common scout budget for this setting, fixed
# before any scale result; 1e-4 is the T2A DDP command's setting default.
STEPS="${STEPS:-8000}"; LR="${LR:-1e-4}"; CE_LIMIT="${CE_LIMIT:-24}"; ACC_LIMIT="${ACC_LIMIT:-0}"
IFS=',' read -ra LADDER <<< "$SCALES"

for SCALE in "${LADDER[@]}"; do
  STEPS="$STEPS" PGB="${PGB:-16}" LR="$LR" CE_LIMIT="$CE_LIMIT" ACC_LIMIT="$ACC_LIMIT" \
    bash scripts/t2a_steering_record_trial.sh scale_locator "$SCALE" "$SEED" "$GH" "$GS" \
      "$ROOT/scale${SCALE}" "$LR" "$STEPS" "$CE_LIMIT" "$ACC_LIMIT"
done
