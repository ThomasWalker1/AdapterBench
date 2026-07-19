#!/bin/bash
# Reproduce the T2L LoRA baseline for one seed.
#
# Trains the strip-def hypernetwork AND the same-shape static reference (multi-task LoRA) on the
# 479-task decontaminated SNI split (gemma-2-2b, plain SFT cross-entropy, definition stripped from the
# input so the description is the only route to the task), then scores matched - static on the 21
# held-out SNI validation tasks: teacher-forced CE (primary) + greedy-generation accuracy (corroborating).
#
# Usage: scripts/reproduce/task_t2l_lora.sh [SEED] [GPUS_HYPER] [GPUS_STATIC]
#   e.g. scripts/reproduce/task_t2l_lora.sh 3 0,1,2,3 4,5,6,7
set -uo pipefail
cd "$(dirname "$0")/../.."
SEED="${1:-777}"; GH="${2:-0,1,2,3}"; GS="${3:-4,5,6,7}"
export HF_HUB_OFFLINE=1   # model + datasets are cached; avoids the HF Hub 429 rate limit
HY=results/repro/t2l_base_diag/gemma2b_stripdef_hyper
ST=results/repro/t2l_base_diag/gemma2b_stripdef_static

echo "=== T2L LoRA baseline, seed $SEED ($(date)) ==="
SEED=$SEED STRIPDEF=1 STATIC=0 STEPS=20000 LR=1e-4 SNAP=5000 LIMIT=40 ELIMIT=20 \
  scripts/t2l_base_diag.sh google/gemma-2-2b-it gemma2b_stripdef_hyper "$GH" 16 > "$HY/s${SEED}_train.log" 2>&1 &
HPID=$!
SEED=$SEED STRIPDEF=1 STATIC=1 STEPS=20000 LR=1e-4 SNAP=5000 LIMIT=40 ELIMIT=20 \
  scripts/t2l_base_diag.sh google/gemma-2-2b-it gemma2b_stripdef_static "$GS" 16 > "$ST/s${SEED}_train.log" 2>&1 &
SPID=$!
wait "$HPID" "$SPID"
echo "=== training done ($(date)); running evals ==="
sleep 30

echo "--- CE eval (matched - static, PRIMARY) ---"
.venv/bin/python scripts/t2p_eval_heldout_sni.py --interpreter google/gemma-2-2b-it \
  --snapshot "$HY/s${SEED}/snapshots/step20000.pt" --static-snapshot "$ST/s${SEED}/snapshots/step20000.pt" \
  --limit 64 --batch-size 16 --n-desc 3 --device cuda:0 --out "$HY/s${SEED}/heldout_sni_ce_full21.jsonl"

echo "--- accuracy eval (corroborating) ---"
.venv/bin/python scripts/t2p_eval_heldout_sni_acc.py --interpreter google/gemma-2-2b-it \
  --snapshot "$HY/s${SEED}/snapshots/step20000.pt" --static-snapshot "$ST/s${SEED}/snapshots/step20000.pt" \
  --limit 48 --batch-size 16 --max-new-tokens 32 --device cuda:0 --out "$HY/s${SEED}/heldout_sni_acc.jsonl"
echo "=== DONE seed $SEED ($(date)) ==="
