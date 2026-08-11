#!/bin/bash
# One restart-safe T2A codec trial for the autoresearch protocol.  It trains the
# description-conditioned hypernetwork and the equal-shape static control together,
# then scores the held-out-SNI CE metric (and optionally generation accuracy).
#
# Usage: scripts/t2a_codec_trial.sh ADAPTER SCALE SEED GPUS_HYPER GPUS_STATIC ROOT
# Example:
#   STEPS=8000 bash scripts/t2a_codec_trial.sh ia3 1 1702 0,1 2,3 results/autoresearch/t2a/ia3/scale1
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:?adapter required}"; SCALE="${2:?scale required}"; SEED="${3:?seed required}"
GH="${4:?hyper GPU CSV required}"; GS="${5:?static GPU CSV required}"; ROOT="${6:?output root required}"
case "$ADAPTER" in lora|ia3|lokr|fourierft|steering) ;; *) echo "unsupported adapter: $ADAPTER" >&2; exit 2 ;; esac

STEPS="${STEPS:-8000}"; LR="${LR:-1e-4}"; SNAP="${SNAP:-$STEPS}"; LIMIT="${LIMIT:-40}"
PGB="${PGB:-16}"; CE_LIMIT="${CE_LIMIT:-24}"; ACC_LIMIT="${ACC_LIMIT:-0}"
NO_COMPILE="${NO_COMPILE:-1}"  # launcher-only fallback for hosts without a usable Triton; semantics unchanged

# --- independent static control -------------------------------------------------------
# The corrected T2A protocol (AUTORESEARCH.md "T2A: metric, data split, and the independently
# selected control") selects the static reference on ITS OWN score, so the two roles routinely
# want DIFFERENT hyperparameters -- e.g. (IA)^3's hypernetwork peaks at lr 5e-5 while its static
# peaks at lr 8e-4, a rate at which the hypernetwork itself diverges. Training both at one
# (SCALE, LR) would yoke the control to the hypernetwork's choice and make a valid pair impossible.
#
# STATIC_SCALE / STATIC_LR default to the hypernetwork's values, so a command that omits them
# trains a matched pair.
STATIC_SCALE="${STATIC_SCALE:-$SCALE}"; STATIC_LR="${STATIC_LR:-$LR}"
# ROLES selects which halves to train: `both` (default), or one role so two single-role trials of
# DIFFERENT configurations can run concurrently on disjoint GPU sets. Prefer ROLES over
# STATIC_SCALE/STATIC_LR when the two roles belong to different sweep rungs, so each output
# directory keeps describing exactly one (role, scale, lr). Use STATIC_SCALE/STATIC_LR when you
# genuinely want a co-located pair at different hyperparameters, which is the confirmation case.
ROLES="${ROLES:-both}"
case "$ROLES" in both|hyper|static) ;; *) echo "ROLES must be both|hyper|static" >&2; exit 2 ;; esac
# The inline CE eval is an appendix figure and needs BOTH snapshots, so it is skipped
# automatically for a single-role trial. Score those with
# `scripts/t2a_score_checkpoints.py` (selection split, example offset, ROUGE-L) instead.
SKIP_EVAL="${SKIP_EVAL:-0}"
[[ "$ROLES" == "both" ]] || SKIP_EVAL=1

HY="$ROOT/hyper/s$SEED"; ST="$ROOT/static/s$SEED"
[[ "$ROLES" == "static" ]] || mkdir -p "$HY"
[[ "$ROLES" == "hyper"  ]] || mkdir -p "$ST"
export HF_HUB_OFFLINE=1

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  .venv/bin/adapterbench preflight --setting t2a --devices "$GH,$GS" --output "$HY"
fi

# The generic CODEC_SCALE/--codec-scaling override applies SCALE to whichever codec
# ADAPTER selects, so a new codec needs no scale-flag mapping here (or anywhere).
echo "=== T2A trial adapter=$ADAPTER roles=$ROLES seed=$SEED steps=$STEPS ($(date)) ==="
echo "===   hyper: scale=$SCALE lr=$LR   static: scale=$STATIC_SCALE lr=$STATIC_LR ==="
HPID=""; SPID=""
if [[ "$ROLES" != "static" ]]; then
  OUT="$HY" SEED="$SEED" ADAPTER="$ADAPTER" CODEC_SCALE="$SCALE" \
    NO_COMPILE="$NO_COMPILE" SKIP_INLINE_EVAL=1 STRIPDEF=1 STATIC=0 STEPS="$STEPS" LR="$LR" SNAP="$SNAP" LIMIT="$LIMIT" ELIMIT=4 \
    scripts/t2a_base_diag.sh google/gemma-2-2b-it "autoresearch_${ADAPTER}_scale${SCALE}_hyper" "$GH" "$PGB" \
    > "$ROOT/hyper_s${SEED}_train.log" 2>&1 &
  HPID=$!
fi
if [[ "$ROLES" != "hyper" ]]; then
  OUT="$ST" SEED="$SEED" ADAPTER="$ADAPTER" CODEC_SCALE="$STATIC_SCALE" \
    NO_COMPILE="$NO_COMPILE" SKIP_INLINE_EVAL=1 STRIPDEF=1 STATIC=1 STEPS="$STEPS" LR="$STATIC_LR" SNAP="$SNAP" LIMIT="$LIMIT" ELIMIT=4 \
    scripts/t2a_base_diag.sh google/gemma-2-2b-it "autoresearch_${ADAPTER}_scale${STATIC_SCALE}_static" "$GS" "$PGB" \
    > "$ROOT/static_s${SEED}_train.log" 2>&1 &
  SPID=$!
fi
if [[ -n "$HPID" ]] && ! wait "$HPID"; then
  echo "hypernetwork training failed; stopping static control" >&2
  [[ -n "$SPID" ]] && { kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; }
  exit 1
fi
if [[ -n "$SPID" ]] && ! wait "$SPID"; then echo "static control training failed" >&2; exit 1; fi

HYPER_SNAPSHOT="$HY/snapshots/step${STEPS}.pt"
STATIC_SNAPSHOT="$ST/snapshots/step${STEPS}.pt"
[[ "$ROLES" == "static" ]] || [[ -f "$HYPER_SNAPSHOT" ]] || { echo "expected checkpoint is missing: $HYPER_SNAPSHOT" >&2; exit 1; }
[[ "$ROLES" == "hyper"  ]] || [[ -f "$STATIC_SNAPSHOT" ]] || { echo "expected checkpoint is missing: $STATIC_SNAPSHOT" >&2; exit 1; }

# The inline evals rebuild BOTH modules from one --codec-scaling, so they are only meaningful when
# the pair shares a scale. A pair at different scales is scored by the re-scoring driver instead.
if [[ "$SKIP_EVAL" != "1" && "$STATIC_SCALE" == "$SCALE" ]]; then
  .venv/bin/python scripts/t2a_eval_heldout_sni.py --interpreter google/gemma-2-2b-it \
    --adapter "$ADAPTER" --codec-scaling "$SCALE" \
    --snapshot "$HYPER_SNAPSHOT" --static-snapshot "$STATIC_SNAPSHOT" \
    --limit "$CE_LIMIT" --batch-size 16 --n-desc 3 --device cuda:0 --out "$HY/heldout_sni_ce_full21.jsonl"
  if [[ "$ACC_LIMIT" != "0" ]]; then
    .venv/bin/python scripts/t2a_eval_heldout_sni_acc.py --interpreter google/gemma-2-2b-it \
      --adapter "$ADAPTER" --codec-scaling "$SCALE" \
      --snapshot "$HYPER_SNAPSHOT" --static-snapshot "$STATIC_SNAPSHOT" \
      --limit "$ACC_LIMIT" --batch-size 16 --max-new-tokens 32 --device cuda:0 --out "$HY/heldout_sni_acc.jsonl"
  fi
elif [[ "$SKIP_EVAL" != "1" ]]; then
  echo "skipping inline eval: hyper scale $SCALE != static scale $STATIC_SCALE; score with" \
       "scripts/t2a_score_checkpoints.py" >&2
fi
echo "=== DONE adapter=$ADAPTER roles=$ROLES seed=$SEED ($(date)) ==="
