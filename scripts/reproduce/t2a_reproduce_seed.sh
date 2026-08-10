#!/bin/bash
# Train one T2A confirmation seed for a codec at the selected operating point.
#
# Trains the hypernetwork and independently selected static* control on disjoint GPU sets.
# Scoring the report split is a separate one-shot step — run it only after every
# confirmation seed for the codec (or all codecs) has finished training.
#
# Usage: scripts/reproduce/t2a_reproduce_seed.sh CODEC [SEED] [GPUS_HYPER] [GPUS_STATIC]
#   e.g. scripts/reproduce/t2a_reproduce_seed.sh lora 1741 0,1,2,3 4,5,6,7
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck source=scripts/reproduce/t2a_codec_config.sh
source scripts/reproduce/t2a_codec_config.sh

CODEC="${1:?codec required}"; shift
t2a_load_codec_config "$CODEC"

SEED="${1:-${CONFIRMATION_SEEDS[0]}}"; shift
GH="${1:-0,1,2,3}"; GS="${2:-4,5,6,7}"
t2a_assert_confirmation_seed "$SEED"

export HF_HUB_OFFLINE=1
PGB="${PGB:-16}"; LIMIT="${LIMIT:-40}"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  .venv/bin/adapterbench preflight --setting t2a --devices "$GH,$GS" \
    --output "$BASE_ROOT/$HYPER_TAG/hyper/s$SEED"
fi

echo "=== T2A confirmation train codec=$CODEC seed=$SEED ($(date)) ==="
echo "===   hyper: scale=$HYPER_SCALE lr=$HYPER_LR   static*: scale=$STATIC_SCALE lr=$STATIC_LR steps=$STEPS ==="

STEPS="$STEPS" LR="$HYPER_LR" ROLES=hyper SKIP_EVAL=1 SKIP_PREFLIGHT=1 PGB="$PGB" LIMIT="$LIMIT" \
  bash scripts/t2a_codec_trial.sh "$CODEC" "$HYPER_SCALE" "$SEED" "$GH" "$GH" \
  "$BASE_ROOT/$HYPER_TAG"

STEPS="$STEPS" LR="$STATIC_LR" ROLES=static SKIP_EVAL=1 SKIP_PREFLIGHT=1 PGB="$PGB" LIMIT="$LIMIT" \
  bash scripts/t2a_codec_trial.sh "$CODEC" "$STATIC_SCALE" "$SEED" "$GS" "$GS" \
  "$BASE_ROOT/$STATIC_TAG"

echo "=== DONE codec=$CODEC seed=$SEED ($(date)) ==="
echo "After all confirmation seeds are trained, score once with:"
echo "  bash scripts/reproduce/t2a_score_codec.sh $CODEC"
echo "  .venv/bin/python scripts/t2a_confirmation_result.py"
