#!/bin/bash
# Joint scale x learning-rate sweep for D2A DoRA — a wide, short, low-resolution screen whose
# only job is to locate the operating window before the protocol refines it.
#
# Why joint rather than one axis at a time: DoRA's two free axes interact. Scale 64 does not
# retrieve at the substrate-default lr 4e-5 and reaches the 512-token gate ceiling at 2e-5, so
# sweeping scale at a single learning rate cannot see the window. Nothing this screen produces
# may be quoted as a benchmark result or used to select an operating point: a window found here
# is re-derived from scratch under AUTORESEARCH.md (declared ladder, 3 selection seeds,
# stability gate, 5 fresh confirmation seeds) before it can become a row.
#
# What it holds fixed, so a window found here transfers:
#   * substrate  — 512-token training length, 512 documents, realistic_numeric_decoys with 4
#                  decoys, 208 latents, 8 cross-attention blocks, Qwen3-0.6B, exit layer;
#   * shape      — DoRA rank 8 at down_proj, magnitude vector included (its shape identity);
#   * instrument — scout seed 902 and the DEV eval seed 1802. The confirmation seeds
#                  (2901-2906) and the held-out eval seed 2904 are never touched here.
#
# What it varies: scale and learning rate (free axes), at a short 8k budget, scored at the gate
# length plus 1024 only — the 16k/32k eval bins dominate wall-clock and the gate is the
# screening signal. That makes the AUC incomparable with the protocol rungs BY DESIGN; only the
# gate column is comparable.
#
# Usage: bash scripts/d2a_dora_explore.sh "GPU_CSV" "SCALE:LR SCALE:LR ..." [STEPS]
#   e.g. bash scripts/d2a_dora_explore.sh 4,7 "4:2e-5 16:2e-5 1:2e-5 64:2e-5"
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${1:?GPU CSV required}"; GRID="${2:?grid required as SCALE:LR pairs}"; STEPS="${3:-8000}"
SEED="${SEED:-902}"; WARMUP="${WARMUP_STEPS:-960}"; EVAL_LENGTHS="${EVAL_LENGTHS:-512,1024}"
EVAL_LIMIT="${EVAL_LIMIT:-12}"; EVAL_SEED="${EVAL_SEED:-1802}"
ROOT="${ROOT:-results/autoresearch/d2a/dora/exploration}"
LEDGER="${LEDGER:-results/autoresearch/d2a/dora/state.jsonl}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
read -ra POINTS <<< "$GRID"
mkdir -p "$ROOT"

run_point() {
  local scale="$1" lr="$2" gpu="$3"
  local out="$ROOT/scale${scale}_lr${lr}/s${SEED}"
  mkdir -p "$out"
  .venv/bin/adapterbench d2a-niah --adapters dora --codec-scaling "$scale" \
    --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
    --context-lengths 512 --eval-context-lengths "$EVAL_LENGTHS" \
    --num-train-documents 512 --steps "$STEPS" --eval-every 4000 --learning-rate "$lr" \
    --warmup-steps "$WARMUP" --n-latents 208 --num-blocks 8 --eval-limit "$EVAL_LIMIT" \
    --eval-seed "$EVAL_SEED" --seed "$SEED" --device "cuda:${gpu}" --output "$out" \
    > "$out/screen_${STEPS}.log" 2>&1 || echo "point scale=$scale lr=$lr exited nonzero" >&2
  .venv/bin/python scripts/d2a_dora_record_screen.py --run "$out" --scale "$scale" \
    --learning-rate "$lr" --steps "$STEPS" --warmup-steps "$WARMUP" --seed "$SEED" \
    --eval-lengths "$EVAL_LENGTHS" --eval-limit "$EVAL_LIMIT" --ledger "$LEDGER"
}

pids=()
index=0
for point in "${POINTS[@]}"; do
  SCALE="${point%%:*}"; LR="${point##*:}"
  GPU="${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}"
  echo "=== screen scale=$SCALE lr=$LR steps=$STEPS gpu=$GPU ($(date)) ==="
  run_point "$SCALE" "$LR" "$GPU" &
  pids+=("$!")
  index=$((index + 1))
  if (( ${#pids[@]} == ${#GPU_ARRAY[@]} )); then
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    pids=()
  fi
done
if (( ${#pids[@]} )); then for pid in "${pids[@]}"; do wait "$pid" || true; done; fi
echo "=== screen complete ($(date)) ==="
