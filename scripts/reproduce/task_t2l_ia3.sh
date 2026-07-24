#!/bin/bash
# Reproduce one T2L (IA)^3 seed at a selected codec-specific multiplier scale.
#
# Usage: scripts/reproduce/task_t2l_ia3.sh [SEED] [GPUS_HYPER] [GPUS_STATIC] [IA3_SCALE]
#   e.g. scripts/reproduce/task_t2l_ia3.sh 777 0,1,2,3 4,5,6,7 1.0
set -euo pipefail
cd "$(dirname "$0")/../.."
SEED="${1:-777}"; GH="${2:-0,1,2,3}"; GS="${3:-4,5,6,7}"; SCALE="${4:-1.0}"
NO_COMPILE="${NO_COMPILE:-1}"  # this runtime has no compatible Triton; semantics are unchanged
export HF_HUB_OFFLINE=1
ROOT="results/repro/t2l_ia3/gemma2b_stripdef_scale${SCALE}"
HY="$ROOT/hyper"; ST="$ROOT/static"
mkdir -p "$HY" "$ST"
.venv/bin/adapterbench preflight --setting t2l --devices "$GH,$GS" --output "$HY/s${SEED}"

echo "=== T2L (IA)^3, seed $SEED, scale $SCALE ($(date)) ==="
OUT="$HY/s${SEED}" SEED=$SEED ADAPTER=ia3 IA3_SCALE="$SCALE" NO_COMPILE="$NO_COMPILE" STRIPDEF=1 STATIC=0 STEPS=20000 LR=1e-4 SNAP=5000 LIMIT=40 ELIMIT=20 \
  scripts/t2l_base_diag.sh google/gemma-2-2b-it "ia3_hyper_scale${SCALE}" "$GH" 16 > "$HY/s${SEED}_train.log" 2>&1 &
HPID=$!
OUT="$ST/s${SEED}" SEED=$SEED ADAPTER=ia3 IA3_SCALE="$SCALE" NO_COMPILE="$NO_COMPILE" STRIPDEF=1 STATIC=1 STEPS=20000 LR=1e-4 SNAP=5000 LIMIT=40 ELIMIT=20 \
  scripts/t2l_base_diag.sh google/gemma-2-2b-it "ia3_static_scale${SCALE}" "$GS" 16 > "$ST/s${SEED}_train.log" 2>&1 &
SPID=$!
if ! wait "$HPID"; then kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; exit 1; fi
wait "$SPID"
HYPER_SNAPSHOT="$HY/s${SEED}/snapshots/step20000.pt"
STATIC_SNAPSHOT="$ST/s${SEED}/snapshots/step20000.pt"
for SNAPSHOT in "$HYPER_SNAPSHOT" "$STATIC_SNAPSHOT"; do
  [ -f "$SNAPSHOT" ] || { echo "Expected checkpoint is missing: $SNAPSHOT" >&2; exit 1; }
done
.venv/bin/python scripts/t2p_eval_heldout_sni.py --interpreter google/gemma-2-2b-it \
  --adapter ia3 --ia3-scaling "$SCALE" --snapshot "$HYPER_SNAPSHOT" --static-snapshot "$STATIC_SNAPSHOT" \
  --limit 64 --batch-size 16 --n-desc 3 --device cuda:0 --out "$HY/s${SEED}/heldout_sni_ce_full21.jsonl"
.venv/bin/python scripts/t2p_eval_heldout_sni_acc.py --interpreter google/gemma-2-2b-it \
  --adapter ia3 --ia3-scaling "$SCALE" --snapshot "$HYPER_SNAPSHOT" --static-snapshot "$STATIC_SNAPSHOT" \
  --limit 48 --batch-size 16 --max-new-tokens 32 --device cuda:0 --out "$HY/s${SEED}/heldout_sni_acc.jsonl"
