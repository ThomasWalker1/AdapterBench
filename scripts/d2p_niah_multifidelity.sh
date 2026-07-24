#!/bin/bash
# Restart-safe D2L multi-fidelity locator. It promotes codec-scale candidates by
# controlled log-length AUC rather than loss, reusing one checkpoint per candidate.
#
# Usage: scripts/d2p_niah_multifidelity.sh ADAPTER GPU_CSV SCALES_CSV [SEED] [ROOT]
# Example: scripts/d2p_niah_multifidelity.sh lora 0,1,2,3 1,3,10,30 902
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:?adapter required}"; GPUS="${2:?GPU csv required}"; SCALES_CSV="${3:?scale csv required}"
SEED="${4:-902}"; ROOT="${5:-results/autoresearch/d2l/$ADAPTER/multifidelity}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
IFS=',' read -ra CANDIDATES <<< "$SCALES_CSV"
case "$ADAPTER" in
  lora) SCALE_FLAG="--lora-scaling"; LR="4e-5" ;;
  ia3) SCALE_FLAG="--ia3-scaling"; LR="2e-5" ;;
  *) echo "unsupported adapter: $ADAPTER (expected lora or ia3)" >&2; exit 2 ;;
esac
LR="${LEARNING_RATE:-$LR}"
NEEDLE_STYLE="${NEEDLE_STYLE:-realistic}"
NUMERIC_DECOY_COUNT="${NUMERIC_DECOY_COUNT:-0}"
TRAIN_CONTEXT_LENGTHS="${TRAIN_CONTEXT_LENGTHS:-256}"
GATE_LENGTH="${GATE_LENGTH:-256}"
EVAL_LENGTHS="${EVAL_LENGTHS:-512,1024,2048,4096,8192}"
EVAL_LIMIT="${EVAL_LIMIT:-32}"
EVAL_SEED="${EVAL_SEED:-778}"
RUNG1_STEPS="${RUNG1_STEPS:-8000}"
RUNG2_STEPS="${RUNG2_STEPS:-16000}"
FINAL_STEPS="${FINAL_STEPS:-36000}"
WARMUP_STEPS="${WARMUP_STEPS:-$((FINAL_STEPS * 3 / 100))}"

run_rung() {
  local steps="$1"; local index=0; local pids=()
  for scale in "${CANDIDATES[@]}"; do
    local gpu="${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}"
    local output="$ROOT/scale${scale}/s${SEED}"
    mkdir -p "$output"
    (.venv/bin/adapterbench d2p-niah --adapters "$ADAPTER" "$SCALE_FLAG" "$scale" \
      --needle-style "$NEEDLE_STYLE" --numeric-decoy-count "$NUMERIC_DECOY_COUNT" \
      --context-lengths "$TRAIN_CONTEXT_LENGTHS" --eval-context-lengths "$GATE_LENGTH,$EVAL_LENGTHS" \
      --num-train-documents 512 --steps "$steps" --eval-every "$RUNG1_STEPS" --learning-rate "$LR" \
      --warmup-steps "$WARMUP_STEPS" --n-latents 208 --num-blocks 8 --eval-limit "$EVAL_LIMIT" --eval-seed "$EVAL_SEED" \
      --seed "$SEED" --device "cuda:${gpu}" --output "$output" > "$output/rung_${steps}.log" 2>&1) &
    pids+=("$!"); index=$((index + 1))
    if (( ${#pids[@]} == ${#GPU_ARRAY[@]} )); then wait "${pids[@]}"; pids=(); fi
  done
  # Under `set -e`, a bare false arithmetic test would terminate the launcher when
  # the final batch filled every GPU and `pids` was already drained above.
  if (( ${#pids[@]} )); then
    wait "${pids[@]}"
  fi
}

run_rung "$RUNG1_STEPS"
mapfile -t CANDIDATES < <(.venv/bin/python scripts/d2p_niah_select.py --root "$ROOT" --adapter "$ADAPTER" --hard-lengths "$EVAL_LENGTHS" --gate-length "$GATE_LENGTH" --steps "$RUNG1_STEPS" --top 3 --format scales --retain-all-if-no-gate | tr ' ' '\n')
run_rung "$RUNG2_STEPS"
mapfile -t CANDIDATES < <(.venv/bin/python scripts/d2p_niah_select.py --root "$ROOT" --adapter "$ADAPTER" --hard-lengths "$EVAL_LENGTHS" --gate-length "$GATE_LENGTH" --steps "$RUNG2_STEPS" --top 1 --format scales --retain-all-if-no-gate | tr ' ' '\n')
run_rung "$FINAL_STEPS"
.venv/bin/python scripts/d2p_niah_select.py --root "$ROOT" --adapter "$ADAPTER" --hard-lengths "$EVAL_LENGTHS" --gate-length "$GATE_LENGTH" --steps "$FINAL_STEPS" --top 1
