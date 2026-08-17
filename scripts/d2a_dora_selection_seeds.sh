#!/bin/bash
# D2A DoRA §3b stability gate: the promoted operating point on three selection seeds.
#
# Operating point (from results/autoresearch/d2a/dora/state.jsonl's
# selection_promotion_declaration): scale 64, LR 2e-5, 32000 steps, warmup 960.
# Seeds 6201/6202/6203 are disjoint from the scout (902) and from the five confirmation
# seeds. Dev eval instrument only (seed 1802, 12 examples/bin) — the held-out eval seed
# 2904 and 32-example instrument belong to confirmation.
#
# A candidate is eligible only if it converges on at least 2 of 3 seeds: finite loss,
# clears the frozen helpfulness floor at the shortest in-distribution length, and shows no
# training divergence. The divergence rate is recorded as a first-class selection metric.
#
# Usage: bash scripts/d2a_dora_selection_seeds.sh [GPU_CSV] [SCALE] [LR]
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${1:-4,5,6}"; SCALE="${2:-64}"; LR="${3:-2e-5}"
SEEDS=(6201 6202 6203)
STEPS="${STEPS:-32000}"; WARMUP="${WARMUP_STEPS:-960}"
ROOT="${ROOT:-results/autoresearch/d2a/dora/selection}"
LEDGER="${LEDGER:-results/autoresearch/d2a/dora/state.jsonl}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
mkdir -p "$ROOT"

pids=()
for index in "${!SEEDS[@]}"; do
  SEED="${SEEDS[$index]}"
  GPU="${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}"
  OUT="$ROOT/scale${SCALE}_lr${LR}/s${SEED}"
  mkdir -p "$OUT"
  echo "=== selection seed=$SEED scale=$SCALE lr=$LR gpu=$GPU ($(date)) ==="
  ( .venv/bin/adapterbench d2a-niah --adapters dora --codec-scaling "$SCALE" \
      --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
      --context-lengths 512 --eval-context-lengths 512,1024,2048,4096,8192,16384,32768 \
      --num-train-documents 512 --steps "$STEPS" --eval-every 8000 --learning-rate "$LR" \
      --warmup-steps "$WARMUP" --n-latents 208 --num-blocks 8 --eval-limit 12 \
      --eval-seed 1802 --seed "$SEED" --device "cuda:${GPU}" --output "$OUT" \
      > "$OUT/selection_${STEPS}.log" 2>&1 || echo "seed $SEED exited nonzero" >&2
    .venv/bin/python scripts/d2a_dora_record_probe.py --root "$ROOT" --scales "$SCALE" \
      --seed "$SEED" --learning-rate "$LR" --steps "$STEPS" --ledger "$LEDGER" \
      --phase selection_seed --log-glob "selection_*.log" --scale-dir "scale${SCALE}_lr${LR}" ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
echo "=== selection seeds complete ($(date)) ==="
