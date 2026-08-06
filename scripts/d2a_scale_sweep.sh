#!/bin/bash
# Invariant-#2 best-of-scale sweep for D2A NIAH (leaderboards/document_niah_d2a.md).
#
# Single-seed (777) sweep of the LoRA output scale to LOCATE the retrieval optimum, using the
# shipped realistic recipe (train@256, eval in-distribution@256, 12k steps). The headline metric
# is in-distribution matched-control at 256; best-of-scale is reported on it. NOTE: D2A retrieval
# is a stochastic phase transition, so this single-seed pass is a coarse locator (fixed seed 777
# across scales for comparability) - confirm the winning scale multi-seed afterwards (re-run
# scripts/reproduce/document_niah_lora.sh with --lora-scaling <best>).
#
# Runs the scales across the given GPUs in waves (children of this script; wait-based, no pgrep).
# Usage: scripts/d2a_scale_sweep.sh [GPU_CSV]   (default "5,6,7")
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS="${1:-5,6,7}"; IFS=',' read -ra G <<< "$GPUS"; NG=${#G[@]}
SCALES=(11.31 22.63 45.25 67.88 90.5)   # 0.25x .. 2x the default 2*r^1.5 = 45.25
OUT=results/repro/d2a_scale_sweep

run(){ local s=$1 gpu=$2; mkdir -p "$OUT/scale$s"
  .venv/bin/adapterbench d2a-niah --adapters lora --needle-style realistic \
    --context-lengths 256 --eval-context-lengths 256 --num-train-documents 512 \
    --steps 12000 --eval-every 3000 --learning-rate 4e-5 --lora-scaling "$s" \
    --n-latents 208 --num-blocks 8 --eval-limit 32 --seed 777 --device "cuda:$gpu" \
    --output "$OUT/scale$s" > "$OUT/scale$s/run.log" 2>&1 & }

echo "=== D2A SCALE SWEEP start $(date) : scales=${SCALES[*]} on GPUs=$GPUS ==="
i=0; pids=()
for s in "${SCALES[@]}"; do
  run "$s" "${G[$((i % NG))]}"; pids+=("$!")
  i=$((i+1))
  if (( i % NG == 0 )); then wait "${pids[@]}"; pids=(); fi
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"

echo "=== D2A SCALE SWEEP DONE $(date) ==="
echo "scale      in-dist matched-control (niah_256, seed 777)"
for s in "${SCALES[@]}"; do
  v=$(.venv/bin/python scripts/d2a_niah_aggregate.py --root "$OUT/scale$s" --min-seeds 1 2>/dev/null \
        | awk '/[[:space:]]niah_256[[:space:]]/{print $6}')
  printf "  %-8s %s\n" "$s" "${v:-FAILED}"
done
echo "=== pick the best scale above; if it beats the default 45.25, confirm it multi-seed via:"
echo "===   scripts/reproduce/document_niah_lora.sh  (add --lora-scaling <best> to each run) ==="
