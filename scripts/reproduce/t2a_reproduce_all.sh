#!/bin/bash
# Full three-seed T2A confirmation reproduction for one codec.
#
# Trains every confirmation seed, verifies the set, scores the report split once,
# and prints the headline aggregate via t2a_confirmation_result.py.
#
# Usage: scripts/reproduce/t2a_reproduce_all.sh CODEC [GPUS_HYPER] [GPUS_STATIC]
#   e.g. scripts/reproduce/t2a_reproduce_all.sh lora 0,1,2,3 4,5,6,7
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck source=scripts/reproduce/t2a_codec_config.sh
source scripts/reproduce/t2a_codec_config.sh

CODEC="${1:?codec required}"; shift
t2a_load_codec_config "$CODEC"
GH="${1:-0,1,2,3}"; GS="${2:-4,5,6,7}"

for i in "${!CONFIRMATION_SEEDS[@]}"; do
  SEED="${CONFIRMATION_SEEDS[$i]}"
  if [[ "$i" -gt 0 ]]; then
    SKIP_PREFLIGHT=1 bash scripts/reproduce/t2a_reproduce_seed.sh "$CODEC" "$SEED" "$GH" "$GS"
  else
    bash scripts/reproduce/t2a_reproduce_seed.sh "$CODEC" "$SEED" "$GH" "$GS"
  fi
done

bash scripts/reproduce/t2a_score_codec.sh "$CODEC"
.venv/bin/python scripts/t2a_confirmation_result.py
