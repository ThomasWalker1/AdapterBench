#!/bin/bash
# T2A DoRA scale locator for one role.
#
# The ladder is centred on DoRA's OWN identity-scale convention, never on another codec's
# selected scale: the update is W' = m ⊙ (W0 + scale*B@A)/||W0 + scale*B@A||_row with the
# generated magnitude delta also carried at `scale`, so scale 1.0 (the manifest default)
# adds a unit-weight directional update and a magnitude delta commensurate with the frozen
# row norms, and the zero-initialised head is exactly the frozen model at every scale.
# Declared ladder: x4 spacing over 0.0625 - 16, safety limits 1/256 and 256.
#
# Rungs run sequentially because each uses the released effective batch (4 GPUs x PGB 16).
# Run the two roles concurrently on disjoint GPU sets to fill the machine.
#
# Usage: ROLE=hyper GPUS=0,1,2,3 SEED=6702 bash scripts/t2a_dora_scale_locator.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ROLE="${ROLE:?ROLE required (hyper|static)}"
GPUS="${GPUS:-0,1,2,3}"
SEED="${SEED:-6702}"
SCALES="${SCALES:-0.0625,0.25,1,4,16}"
# 8000 steps is DoRA's declared common scout budget for this setting, fixed before any
# scale result (it is the T2A budget lora/ia3/steering also used); 1e-4 is the T2A DDP
# command's setting default learning rate.
STEPS="${STEPS:-8000}"; LR="${LR:-1e-4}"
ROOT="${ROOT:-results/autoresearch/t2a/dora/scale_locator}"

IFS=',' read -ra LADDER <<< "$SCALES"
for SCALE in "${LADDER[@]}"; do
  echo "=== dora scale locator: role=$ROLE scale=$SCALE lr=$LR seed=$SEED gpus=$GPUS ($(date)) ==="
  bash scripts/t2a_dora_trial.sh scale_locator "$ROLE" "$SCALE" "$LR" "$SEED" "$GPUS" \
    "$ROOT/scale${SCALE}_lr${LR}_${ROLE}" "$STEPS"
done
echo "=== dora scale locator done: role=$ROLE ($(date)) ==="
