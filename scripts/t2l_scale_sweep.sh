#!/bin/bash
# Invariant-#2 best-of-scale sweep for T2L (leaderboards/task_conditioned_t2l.md), at the shipped
# DDP recipe. Single-seed (777) scale LOCATOR at a reduced 20k-step budget (conditioning emerges
# well before then; T2L's earlier under-powered sweep was ~flat, so this mainly confirms the
# default is near-optimal or finds a better scale). Confirm the winner at the full 62.5k multi-seed
# afterward (task_t2l_lora_ddp.sh with --lora-scaling <best>).
#
# Each run uses all given GPUs (DDP), so scales run SEQUENTIALLY. Launch only once the main DDP
# run has freed the GPUs. Usage: scripts/t2l_scale_sweep.sh [GPU_CSV]   (default "1,2,3,4")
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS="${1:-1,2,3,4}"; NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
SCALES=(2.83 5.66 11.31 22.63)   # around the LoRA codec default (alpha/sqrt(r)=5.66 for r=8)
OUT=results/repro/t2l_scale_sweep

for s in "${SCALES[@]}"; do
  echo "=== T2L scale $s ($(date)) ==="
  CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/torchrun --standalone --nproc_per_node="$NPROC" \
    scripts/t2p_train_ddp.py --all-decontam-tasks --max-descriptions 128 --limit 40 \
    --per-gpu-batch 32 --steps 20000 --learning-rate 1e-4 --warmup-frac 0.1 \
    --max-grad-norm 1.0 --fixed-seq-len 512 --lora-scaling "$s" \
    --eval-tasks arc_easy,arc_challenge,hellaswag,boolq --eval-limit 80 \
    --adversarial-control --seed 777 --checkpoint-every 5000 \
    --output "$OUT/scale$s"
done

echo "=== T2L SCALE SWEEP DONE $(date) ==="
for s in "${SCALES[@]}"; do
  echo "--- scale $s (mean matched-adversarial across families) ---"
  .venv/bin/python scripts/t2p_rigor_aggregate.py --results "$OUT/scale$s/results.jsonl" 2>&1 | tail -8
done
