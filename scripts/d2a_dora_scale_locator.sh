#!/bin/bash
# D2A DoRA multi-fidelity scale locator at the LOCKED numeric-decoy setting.
#
# Thin, recorded wrapper around scripts/d2a_niah_multifidelity.sh: it fixes the shipped
# substrate (realistic-prose haystack, 4 numeric decoys, 512-token training length,
# 208 latents, 8 cross-attention blocks) and sweeps only DoRA's output scale, promoting
# candidates by controlled hard-length AUC across the standard 8k/16k/32k rungs.
#
# The ladder is centred on DoRA's own identity convention (scale 1.0 = unit-weight
# directional update plus a magnitude delta commensurate with the frozen row norms; the
# zero-initialised head is the frozen model at any scale), x4 spacing, safety limits
# 1/256 and 256. It deliberately does NOT start from LoRA's D2A parity scale (~45) or any
# other codec's selected scale.
#
# Selection uses the dev eval seed 1802 at 12 examples/bin; the held-out eval seed 2904
# and the 32-example instrument belong to confirmation only.
#
# Usage: bash scripts/d2a_dora_scale_locator.sh [GPU_CSV] [SEED] [SCALES_CSV]
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${1:-4,5,6,7}"
SEED="${2:-902}"
SCALES="${3:-0.0625,0.25,1,4,16}"
ROOT="${ROOT:-results/autoresearch/d2a/dora/scale_locator}"
LEDGER="${LEDGER:-results/autoresearch/d2a/dora/state.jsonl}"
mkdir -p "$ROOT" "$(dirname "$LEDGER")"

NEEDLE_STYLE=realistic_numeric_decoys \
NUMERIC_DECOY_COUNT=4 \
TRAIN_CONTEXT_LENGTHS=512 \
GATE_LENGTH=512 \
EVAL_LENGTHS=1024,2048,4096,8192,16384,32768 \
EVAL_LIMIT=12 \
EVAL_SEED=1802 \
RUNG1_STEPS=8000 \
RUNG2_STEPS=16000 \
FINAL_STEPS=32000 \
WARMUP_STEPS=960 \
LEARNING_RATE="${LEARNING_RATE:-4e-5}" \
  bash scripts/d2a_niah_multifidelity.sh dora "$GPUS" "$SCALES" "$SEED" "$ROOT"
