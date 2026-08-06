#!/bin/bash
# Held-out confirmation for the four-decoy single-needle D2A setting.
# The scale and learning rate are locked from the dev locator; do not tune against
# this split. By default runs all three confirmation seeds sequentially on one GPU.
# Set SEEDS=2901 (etc.) to reproduce or schedule an individual seed.
#
# Usage: scripts/reproduce/document_niah_numeric_decoy.sh ADAPTER [DEVICE] [ROOT]
set -euo pipefail
cd "$(dirname "$0")/../.."

ADAPTER="${1:?adapter required (lora or ia3)}"
DEVICE="${2:-cuda:0}"
ROOT="${3:-results/repro/d2a_numeric_decoy/${ADAPTER}}"
SEEDS="${SEEDS:-2901,2902,2903,2905,2906}"
case "$ADAPTER" in
  lora) SCALE=100; LR=4e-5; SCALE_FLAG=--lora-scaling ;;
  ia3) SCALE=64; LR=2e-5; SCALE_FLAG=--ia3-scaling ;;
  *) echo "unsupported adapter: $ADAPTER (expected lora or ia3)" >&2; exit 2 ;;
esac
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  .venv/bin/adapterbench preflight --setting d2a --devices "$DEVICE" --output "$ROOT"
fi
for SEED in "${SEED_ARRAY[@]}"; do
  .venv/bin/adapterbench d2a-niah --adapters "$ADAPTER" "$SCALE_FLAG" "$SCALE" \
    --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
    --context-lengths 512 --eval-context-lengths 512,1024,2048,4096,8192,16384,32768 \
    --num-train-documents 512 --steps 32000 --eval-every 8000 --learning-rate "$LR" --warmup-steps 960 \
    --n-latents 208 --num-blocks 8 --eval-limit 32 --eval-seed 2904 \
    --seed "$SEED" --device "$DEVICE" --output "$ROOT/s$SEED"
done
if [[ "$SEEDS" == "2901,2902,2903,2905,2906" ]]; then
  .venv/bin/python scripts/d2a_niah_aggregate.py --root "$ROOT" --min-seeds 5 --train-length 512
fi
