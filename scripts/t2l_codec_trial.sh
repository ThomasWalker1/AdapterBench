#!/bin/bash
# One restart-safe T2L codec trial for the autoresearch protocol.  It trains the
# description-conditioned hypernetwork and the equal-shape static control together,
# then scores the held-out-SNI CE metric (and optionally generation accuracy).
#
# Usage: scripts/t2l_codec_trial.sh ADAPTER SCALE SEED GPUS_HYPER GPUS_STATIC ROOT
# Example:
#   STEPS=8000 bash scripts/t2l_codec_trial.sh ia3 1 1702 0,1 2,3 results/autoresearch/t2l/ia3/scale1
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:?adapter required}"; SCALE="${2:?scale required}"; SEED="${3:?seed required}"
GH="${4:?hyper GPU CSV required}"; GS="${5:?static GPU CSV required}"; ROOT="${6:?output root required}"
case "$ADAPTER" in lora|ia3|lokr|loha|fourierft) ;; *) echo "unsupported adapter: $ADAPTER" >&2; exit 2 ;; esac

STEPS="${STEPS:-8000}"; LR="${LR:-1e-4}"; SNAP="${SNAP:-$STEPS}"; LIMIT="${LIMIT:-40}"
PGB="${PGB:-16}"; CE_LIMIT="${CE_LIMIT:-24}"; ACC_LIMIT="${ACC_LIMIT:-0}"
NO_COMPILE="${NO_COMPILE:-1}"  # launcher-only fallback for hosts without a usable Triton; semantics unchanged
HY="$ROOT/hyper/s$SEED"; ST="$ROOT/static/s$SEED"
mkdir -p "$HY" "$ST"
export HF_HUB_OFFLINE=1

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  .venv/bin/adapterbench preflight --setting t2l --devices "$GH,$GS" --output "$HY"
fi

if [[ "$ADAPTER" == "lora" ]]; then
  LORA_SCALE="$SCALE"; IA3_SCALE="1.0"; LOKR_SCALE="1.0"; LOHA_SCALE="1.0"; FOURIER_SCALE="1.0"
elif [[ "$ADAPTER" == "ia3" ]]; then
  LORA_SCALE="-1.0"; IA3_SCALE="$SCALE"; LOKR_SCALE="1.0"; LOHA_SCALE="1.0"; FOURIER_SCALE="1.0"
elif [[ "$ADAPTER" == "lokr" ]]; then
  LORA_SCALE="-1.0"; IA3_SCALE="1.0"; LOKR_SCALE="$SCALE"; LOHA_SCALE="1.0"; FOURIER_SCALE="1.0"
elif [[ "$ADAPTER" == "loha" ]]; then
  LORA_SCALE="-1.0"; IA3_SCALE="1.0"; LOKR_SCALE="1.0"; LOHA_SCALE="$SCALE"; FOURIER_SCALE="1.0"
else
  LORA_SCALE="-1.0"; IA3_SCALE="1.0"; LOKR_SCALE="1.0"; LOHA_SCALE="1.0"; FOURIER_SCALE="$SCALE"
fi
echo "=== T2L trial adapter=$ADAPTER scale=$SCALE seed=$SEED steps=$STEPS ($(date)) ==="
OUT="$HY" SEED="$SEED" ADAPTER="$ADAPTER" LORA_SCALE="$LORA_SCALE" IA3_SCALE="$IA3_SCALE" LOKR_SCALE="$LOKR_SCALE" LOHA_SCALE="$LOHA_SCALE" FOURIER_SCALE="$FOURIER_SCALE" \
  NO_COMPILE="$NO_COMPILE" SKIP_INLINE_EVAL=1 STRIPDEF=1 STATIC=0 STEPS="$STEPS" LR="$LR" SNAP="$SNAP" LIMIT="$LIMIT" ELIMIT=4 \
  scripts/t2l_base_diag.sh google/gemma-2-2b-it "autoresearch_${ADAPTER}_scale${SCALE}_hyper" "$GH" "$PGB" \
  > "$ROOT/hyper_s${SEED}_train.log" 2>&1 &
HPID=$!
OUT="$ST" SEED="$SEED" ADAPTER="$ADAPTER" LORA_SCALE="$LORA_SCALE" IA3_SCALE="$IA3_SCALE" LOKR_SCALE="$LOKR_SCALE" LOHA_SCALE="$LOHA_SCALE" FOURIER_SCALE="$FOURIER_SCALE" \
  NO_COMPILE="$NO_COMPILE" SKIP_INLINE_EVAL=1 STRIPDEF=1 STATIC=1 STEPS="$STEPS" LR="$LR" SNAP="$SNAP" LIMIT="$LIMIT" ELIMIT=4 \
  scripts/t2l_base_diag.sh google/gemma-2-2b-it "autoresearch_${ADAPTER}_scale${SCALE}_static" "$GS" "$PGB" \
  > "$ROOT/static_s${SEED}_train.log" 2>&1 &
SPID=$!
if ! wait "$HPID"; then
  echo "hypernetwork training failed; stopping static control" >&2
  kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; exit 1
fi
if ! wait "$SPID"; then echo "static control training failed" >&2; exit 1; fi

HYPER_SNAPSHOT="$HY/snapshots/step${STEPS}.pt"
STATIC_SNAPSHOT="$ST/snapshots/step${STEPS}.pt"
for SNAPSHOT in "$HYPER_SNAPSHOT" "$STATIC_SNAPSHOT"; do
  [[ -f "$SNAPSHOT" ]] || { echo "expected checkpoint is missing: $SNAPSHOT" >&2; exit 1; }
done

.venv/bin/python scripts/t2p_eval_heldout_sni.py --interpreter google/gemma-2-2b-it \
  --adapter "$ADAPTER" --lora-scaling "$LORA_SCALE" --ia3-scaling "$IA3_SCALE" --lokr-scaling "$LOKR_SCALE" --loha-scaling "$LOHA_SCALE" --fourierft-scaling "$FOURIER_SCALE" \
  --snapshot "$HYPER_SNAPSHOT" --static-snapshot "$STATIC_SNAPSHOT" \
  --limit "$CE_LIMIT" --batch-size 16 --n-desc 3 --device cuda:0 --out "$HY/heldout_sni_ce_full21.jsonl"
if [[ "$ACC_LIMIT" != "0" ]]; then
  .venv/bin/python scripts/t2p_eval_heldout_sni_acc.py --interpreter google/gemma-2-2b-it \
    --adapter "$ADAPTER" --lora-scaling "$LORA_SCALE" --ia3-scaling "$IA3_SCALE" --lokr-scaling "$LOKR_SCALE" --loha-scaling "$LOHA_SCALE" --fourierft-scaling "$FOURIER_SCALE" \
    --snapshot "$HYPER_SNAPSHOT" --static-snapshot "$STATIC_SNAPSHOT" \
    --limit "$ACC_LIMIT" --batch-size 16 --max-new-tokens 32 --device cuda:0 --out "$HY/heldout_sni_acc.jsonl"
fi
echo "=== DONE adapter=$ADAPTER scale=$SCALE seed=$SEED ($(date)) ==="
