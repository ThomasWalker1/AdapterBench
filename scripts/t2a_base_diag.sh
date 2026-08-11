#!/bin/bash
# T2A base-model capacity diagnostic. Holds the shipped DDP recipe FIXED (same as the Qwen3-0.6B
# leaderboard run: 479 decontam tasks, 128 descs, eff-batch 128, 62.5k steps, gte conditioning,
# q_proj/v_proj rank-8, strong junk-description adversarial control) and changes ONLY the base
# model. Purpose: is the near-null matched-control gap on Qwen3-0.6B a capacity floor? If the gap
# opens as the base grows (0.6B -> 2B -> 8B), the 0.6B row just carries that caveat; if it stays
# ~0, the lever is the conditioning objective/architecture, not scale.
#
# Usage: scripts/t2a_base_diag.sh <hf-interpreter-id> <tag> [GPU_CSV] [PER_GPU_BATCH]
#   e.g. scripts/t2a_base_diag.sh google/gemma-2-2b-it gemma2b 0,1,2,3 32
#        scripts/t2a_base_diag.sh meta-llama/Llama-3.1-8B-Instruct llama8b 0,1,2,3 8
set -euo pipefail
cd "$(dirname "$0")/.."
INTERP="${1:?need HF interpreter id}"; TAG="${2:?need short tag}"
GPUS="${3:-0,1,2,3}"; PGB="${4:-32}"
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
# tunables (env overrides): larger bases need a lower LR (1e-4 diverged on Mistral-7B; the paper
# uses 8e-5 for SFT). Convergence happens well before 60k steps, so STEPS can be cut; SNAP>0 saves
# model-only snapshots for a post-hoc training curve via scripts/t2a_eval_checkpoint.py.
STEPS="${STEPS:-62500}"; LR="${LR:-1e-4}"; SNAP="${SNAP:-0}"; SEED="${SEED:-777}"
# LIMIT<=0 trains on the full per-task datasets (paper-faithful; #2). CLAMBDA>0 turns on the
# mismatched-negative conditioning loss (#1); CMARGIN is its hinge target.
LIMIT="${LIMIT:-40}"; CLAMBDA="${CLAMBDA:-0.0}"; CMARGIN="${CMARGIN:-0.5}"; NJLAMBDA="${NJLAMBDA:-0.0}"
# STRIPDEF=1 drops the task definition from the input (task specified only via the description);
# STATIC=1 trains the multi-task-LoRA reference instead of the hypernetwork.
STRIP_FLAG=""; [ "${STRIPDEF:-0}" = "1" ] && STRIP_FLAG="--strip-task-def"
STATIC_FLAG=""; [ "${STATIC:-0}" = "1" ] && STATIC_FLAG="--static"
# This is a compilation-only escape hatch for CUDA environments without a compatible
# Triton installation. It changes neither the model forward nor any benchmark setting.
COMPILE_FLAG=""; [ "${NO_COMPILE:-0}" = "1" ] && COMPILE_FLAG="--no-compile"
INLINE_EVAL_FLAG=""; [ "${SKIP_INLINE_EVAL:-0}" = "1" ] && INLINE_EVAL_FLAG="--skip-inline-eval"
# The built-in arc/boolq final eval is a *self-describing* set that can't reveal conditioning under
# strip-def (the real eval is the held-out-SNI teacher-forced-CE script on snapshots). ELIMIT lets a
# strip-def run shrink that eval so it doesn't burn ~1h of autoregressive generation on the wrong instrument.
ELIMIT="${ELIMIT:-80}"; ETASKS="${ETASKS:-arc_easy,arc_challenge,hellaswag,boolq}"
OUT="${OUT:-results/repro/t2a_base_diag/$TAG/s$SEED}"
mkdir -p "$OUT"
ADAPTER="${ADAPTER:-lora}"; LORA_SCALE="${LORA_SCALE:--1.0}"; IA3_SCALE="${IA3_SCALE:-1.0}"; LOKR_SCALE="${LOKR_SCALE:-1.0}"; FOURIER_SCALE="${FOURIER_SCALE:-1.0}"
# CODEC_SCALE (empty = unset) is the generic per-adapter scale override — the preferred
# knob for new codecs; the per-codec env vars above are retained for recorded commands.
CODEC_SCALE="${CODEC_SCALE:-}"
CODEC_SCALE_FLAG=(); [ -n "$CODEC_SCALE" ] && CODEC_SCALE_FLAG=(--codec-scaling "$CODEC_SCALE")

echo "=== T2A base diag: interpreter=$INTERP tag=$TAG adapter=$ADAPTER lora_scale=$LORA_SCALE ia3_scale=$IA3_SCALE lokr_scale=$LOKR_SCALE fourierft_scale=$FOURIER_SCALE codec_scale=${CODEC_SCALE:-unset} gpus=$GPUS per_gpu_batch=$PGB steps=$STEPS lr=$LR snap=$SNAP limit=$LIMIT clambda=$CLAMBDA ($(date)) ==="
CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/torchrun --standalone --nproc_per_node="$NPROC" \
  scripts/t2a_train_ddp.py --interpreter "$INTERP" \
  --all-decontam-tasks --max-descriptions 128 --limit "$LIMIT" \
  --per-gpu-batch "$PGB" --steps "$STEPS" --learning-rate "$LR" --warmup-frac 0.1 \
  --max-grad-norm 1.0 --fixed-seq-len 512 --snapshot-every "$SNAP" \
  --adapter "$ADAPTER" --lora-scaling "$LORA_SCALE" --ia3-scaling "$IA3_SCALE" --lokr-scaling "$LOKR_SCALE" --fourierft-scaling "$FOURIER_SCALE" \
  "${CODEC_SCALE_FLAG[@]}" \
  $COMPILE_FLAG $INLINE_EVAL_FLAG \
  --contrastive-lambda "$CLAMBDA" --contrastive-margin "$CMARGIN" --neutral-junk-lambda "$NJLAMBDA" \
  $STRIP_FLAG $STATIC_FLAG \
  --eval-tasks "$ETASKS" --eval-limit "$ELIMIT" \
  --adversarial-control --seed "$SEED" --checkpoint-every 5000 \
  --output "$OUT"

echo "=== DONE $(date) — matched vs junk-description control: ==="
if [[ -f "$OUT/results.jsonl" ]]; then
  .venv/bin/python scripts/t2a_rigor_aggregate.py --results "$OUT/results.jsonl" 2>&1 | tail -15
else
  # A static equal-shape control deliberately has no conditioned evaluation rows; its
  # checkpoint is scored later alongside the hypernetwork snapshot.
  echo "static checkpoint saved; held-out paired evaluation is run by the caller."
fi
