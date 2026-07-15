#!/bin/bash
# Reproduce the T2L LoRA baseline at the SHIPPED full-scale config (leaderboards/task_conditioned_t2l.md).
#
# This is the fast, data-parallel training path (scripts/t2p_train_ddp.py): N-GPU DDP at a
# data-budget-matched effective batch of 128 (per-GPU 32), 62,500 steps = 8M example-visits =
# the same data/epochs as the single-GPU 1M x batch-8 run, in ~8h instead of ~2 days.
# torch.compile + fixed-seq-len 512 (static shapes) + persistent codec hooks.
#
# Trajectory shape differs from the batch-8 emergence curve BY DESIGN (see PROJECT_PLAN
# "CURRENT PHASE" / paper 4.1.1): the benchmark ships at this data-matched scale. The batch-8
# recipe (task_t2l_lora.sh) remains the emergence-curve reference.
#
# Restart-safe: re-run the identical command; each seed resumes from its last 5000-step checkpoint.
#
# Usage: scripts/reproduce/task_t2l_lora_ddp.sh [GPUS]
#   GPUS = comma-separated physical GPU ids (default "1,2,3,4"; leave cuda:0 for other runs).
#          The number of ids sets --nproc_per_node, so eff. batch = 32 * (#GPUS).
set -euo pipefail
cd "$(dirname "$0")/../.."

GPUS="${1:-1,2,3,4}"
NPROC="$(awk -F, '{print NF}' <<<"$GPUS")"
OUT=results/repro/task_t2l_lora_ddp

for SEED in 777 778 779; do
  CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/torchrun --standalone --nproc_per_node="$NPROC" \
    scripts/t2p_train_ddp.py --all-decontam-tasks --max-descriptions 128 --limit 40 \
    --per-gpu-batch 32 --steps 62500 --learning-rate 1e-4 --warmup-frac 0.1 \
    --max-grad-norm 1.0 --fixed-seq-len 512 \
    --eval-tasks arc_easy,arc_challenge,hellaswag,boolq --eval-limit 80 \
    --adversarial-control --seed "$SEED" --checkpoint-every 5000 \
    --output "$OUT/s$SEED"
done

# Aggregate the three seeds: t2p_rigor_aggregate reads one jsonl, so concatenate first
# (same pattern as results/_archive/t2p_cond_long/combined.jsonl for the 150K x3 run).
cat "$OUT"/s777/results.jsonl "$OUT"/s778/results.jsonl "$OUT"/s779/results.jsonl > "$OUT/combined.jsonl"
.venv/bin/python scripts/t2p_rigor_aggregate.py --results "$OUT/combined.jsonl"
