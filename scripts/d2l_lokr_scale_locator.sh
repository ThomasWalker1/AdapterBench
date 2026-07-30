#!/bin/bash
# LoKr's D2L scale locator: a five-point ladder centered on LoKr's own identity
# scale (ΔW = scale * L ⊗ R), using the standard checkpointed D2L rungs.
#
# Usage: scripts/d2l_lokr_scale_locator.sh GPU_CSV [SEED] [ROOT]
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${1:?GPU csv required}"; SEED="${2:-3002}"; ROOT="${3:-results/autoresearch/d2l/lokr/scale_locator}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
SCALES=(0.0625 0.25 1 4 16)
declare -a PIDS=()
for index in "${!SCALES[@]}"; do
  scale="${SCALES[$index]}"; gpu="${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}"
  bash scripts/d2l_lokr_record_trial.sh scale_locator "$scale" "$SEED" "$gpu" 8000 "$ROOT/scale${scale}/s${SEED}" &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
.venv/bin/python scripts/d2p_niah_select.py --root "$ROOT" --adapter lokr \
  --hard-lengths 1024,2048,4096,8192,16384,32768 --gate-length 512 --steps 8000 --top 3 --retain-all-if-no-gate
