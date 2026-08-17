#!/bin/bash
# D2A DoRA upward scale-boundary probe (AUTORESEARCH.md §2 two-extension cap).
#
# The declared ladder (0.0625-16) is retrieval-dead at every point through 32k, so no
# interior optimum exists to close the axis on. This probes the two allowed steps above
# the ladder's top at its own x4 spacing - 64 and 256 (the declared safety limit) - at the
# substrate-default LR, one axis at a time. Same locked numeric-decoy substrate, same scout
# seed and dev eval instrument as the ladder, so the points are directly comparable.
#
# Each point runs the full 32k budget in one job (no promotion: there is nothing to promote
# against), one scale per GPU, and appends a ledger record.
#
# Usage: bash scripts/d2a_dora_boundary_probe.sh [GPU_CSV] [SEED] [SCALES_CSV]
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${1:-4,5}"; SEED="${2:-902}"; SCALES="${3:-64,256}"
ROOT="${ROOT:-results/autoresearch/d2a/dora/scale_boundary}"
LEDGER="${LEDGER:-results/autoresearch/d2a/dora/state.jsonl}"
LR="${LEARNING_RATE:-4e-5}"; STEPS="${STEPS:-32000}"; WARMUP="${WARMUP_STEPS:-960}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
IFS=',' read -ra LADDER <<< "$SCALES"
mkdir -p "$ROOT"

pids=()
for index in "${!LADDER[@]}"; do
  SCALE="${LADDER[$index]}"
  GPU="${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}"
  OUT="$ROOT/scale${SCALE}/s${SEED}"
  mkdir -p "$OUT"
  ( .venv/bin/adapterbench d2a-niah --adapters dora --codec-scaling "$SCALE" \
      --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
      --context-lengths 512 --eval-context-lengths 512,1024,2048,4096,8192,16384,32768 \
      --num-train-documents 512 --steps "$STEPS" --eval-every 8000 --learning-rate "$LR" \
      --warmup-steps "$WARMUP" --n-latents 208 --num-blocks 8 --eval-limit 12 --eval-seed 1802 \
      --seed "$SEED" --device "cuda:${GPU}" --output "$OUT" > "$OUT/probe_${STEPS}.log" 2>&1 ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || echo "probe pid $pid exited nonzero (recorded, not retried)" >&2; done

.venv/bin/python scripts/d2a_dora_record_probe.py --root "$ROOT" --scales "$SCALES" \
  --seed "$SEED" --learning-rate "$LR" --steps "$STEPS" --ledger "$LEDGER"
