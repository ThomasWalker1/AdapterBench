#!/bin/bash
# Recovery/watchdog for the interrupted seed-3 T2L run, followed by the
# mechanism-free I2P prompt-conditioning scale probe.
#
# The first two arguments are the already-running torchrun parent PIDs.  This
# script never signals them; it only waits.  If they exit without both 20k
# snapshots, the official restart-safe reproduce wrapper is run detached by
# the caller and resumes from the latest checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

HPID="${1:?need current hyper torchrun PID}"
SPID="${2:?need current static torchrun PID}"
export HF_HUB_OFFLINE=1

HY="results/repro/t2l_base_diag/gemma2b_stripdef_hyper/s3"
ST="results/repro/t2l_base_diag/gemma2b_stripdef_static/s3"

echo "[$(date)] waiting for current T2L parents hyper=$HPID static=$SPID"
while kill -0 "$HPID" 2>/dev/null || kill -0 "$SPID" 2>/dev/null; do
  sleep 30
done

if [[ ! -f "$HY/snapshots/step20000.pt" || ! -f "$ST/snapshots/step20000.pt" ]]; then
  echo "[$(date)] a 20k snapshot is missing; resuming through the official wrapper"
  scripts/reproduce/task_t2l_lora.sh 3 0,1,2,3 4,5,6,7
fi

if [[ ! -f "$HY/heldout_sni_ce_full21.jsonl" ]]; then
  echo "[$(date)] running seed-3 held-out-SNI CE evaluation"
  .venv/bin/python scripts/t2p_eval_heldout_sni.py \
    --interpreter google/gemma-2-2b-it \
    --snapshot "$HY/snapshots/step20000.pt" \
    --static-snapshot "$ST/snapshots/step20000.pt" \
    --limit 64 --batch-size 16 --n-desc 3 --device cuda:0 \
    --out "$HY/heldout_sni_ce_full21.jsonl"
fi

if [[ ! -f "$HY/heldout_sni_acc.jsonl" ]]; then
  echo "[$(date)] running seed-3 held-out-SNI accuracy evaluation"
  .venv/bin/python scripts/t2p_eval_heldout_sni_acc.py \
    --interpreter google/gemma-2-2b-it \
    --snapshot "$HY/snapshots/step20000.pt" \
    --static-snapshot "$ST/snapshots/step20000.pt" \
    --limit 48 --batch-size 16 --max-new-tokens 32 --device cuda:0 \
    --out "$HY/heldout_sni_acc.jsonl"
fi

echo "[$(date)] seed-3 aggregate rows"
grep __aggregate__ "$HY/heldout_sni_ce_full21.jsonl"
grep __aggregate__ "$HY/heldout_sni_acc.jsonl"

echo "[$(date)] launching mechanism-free I2P scale probes"
PIDS=()
for SPEC in "0:0.5" "1:1" "2:2" "3:4"; do
  GPU="${SPEC%%:*}"
  SCALE="${SPEC##*:}"
  OUT="results/i2p_prompt_probe/scale${SCALE}_s777"
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/i2p_prompt_conditioning_probe.py \
    --device cuda:0 --steps 300 --batch-size 2 \
    --train-prompts 256 --eval-prompts 64 --eval-batch-size 4 --eval-seeds 1 \
    --rank 4 --scale "$SCALE" --latent-dim 256 --head-dim 256 \
    --checkpoint-every 25 --log-every 10 --seed 777 --output "$OUT" \
    > "$OUT/run.log" 2>&1 &
  PIDS+=("$!")
done
wait "${PIDS[@]}"
echo "[$(date)] all I2P prompt-conditioning scale probes finished"
