#!/bin/bash
# D2A DoRA protocol refinement around the located operating window (AUTORESEARCH.md §2/§3).
#
# The scale x LR screen (scripts/d2a_dora_explore.sh) located the window at scale 64, lr 2e-5,
# where the 512-token gate is reached by 8k steps. This script comes back on protocol at the
# full declared instrument to bracket that window: a x2 scale refinement either side of 64,
# plus the lower LR neighbour.
# Nothing outside the free axes moves, and the eval instrument is the declared one
# (all seven lengths, 12 examples/bin, dev eval seed 1802, scout seed).
#
# Each point is one full 32k run, one per GPU, recorded to the append-only ledger.
#
# Usage: bash scripts/d2a_dora_refine.sh [GPU_CSV] [SEED] ["SCALE:LR ..."]
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${1:-4,5,6,7}"; SEED="${2:-902}"
GRID="${3:-32:2e-5 64:2e-5 128:2e-5 64:1e-5}"
STEPS="${STEPS:-32000}"; WARMUP="${WARMUP_STEPS:-960}"
EVAL_LENGTHS="${EVAL_LENGTHS:-512,1024,2048,4096,8192,16384,32768}"
EVAL_LIMIT="${EVAL_LIMIT:-12}"; EVAL_SEED="${EVAL_SEED:-1802}"
ROOT="${ROOT:-results/autoresearch/d2a/dora/refinement}"
LEDGER="${LEDGER:-results/autoresearch/d2a/dora/state.jsonl}"
PHASE="${PHASE:-scale_refinement}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
read -ra POINTS <<< "$GRID"
mkdir -p "$ROOT"

pids=()
for index in "${!POINTS[@]}"; do
  point="${POINTS[$index]}"; SCALE="${point%%:*}"; LR="${point##*:}"
  GPU="${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}"
  OUT="$ROOT/scale${SCALE}_lr${LR}/s${SEED}"
  mkdir -p "$OUT"
  echo "=== refine scale=$SCALE lr=$LR steps=$STEPS gpu=$GPU ($(date)) ==="
  ( .venv/bin/adapterbench d2a-niah --adapters dora --codec-scaling "$SCALE" \
      --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
      --context-lengths 512 --eval-context-lengths "$EVAL_LENGTHS" \
      --num-train-documents 512 --steps "$STEPS" --eval-every 8000 --learning-rate "$LR" \
      --warmup-steps "$WARMUP" --n-latents 208 --num-blocks 8 --eval-limit "$EVAL_LIMIT" \
      --eval-seed "$EVAL_SEED" --seed "$SEED" --device "cuda:${GPU}" --output "$OUT" \
      > "$OUT/refine_${STEPS}.log" 2>&1 || echo "point scale=$SCALE lr=$LR exited nonzero" >&2
    .venv/bin/python scripts/d2a_dora_record_probe.py --root "$ROOT" --scales "$SCALE" \
      --seed "$SEED" --learning-rate "$LR" --steps "$STEPS" --ledger "$LEDGER" \
      --phase "$PHASE" --log-glob "refine_*.log" --scale-dir "scale${SCALE}_lr${LR}" ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
echo "=== refinement complete ($(date)) ==="
