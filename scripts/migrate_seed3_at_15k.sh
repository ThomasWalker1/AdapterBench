#!/bin/bash
# Hand an in-flight seed-3 T2A run from Codex PTYs to the official detached
# reproduce wrapper at the durable step-15k checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

HPID="${1:?need current hyper torchrun PID}"
SPID="${2:?need current static torchrun PID}"
HY="results/repro/t2a_base_diag/gemma2b_stripdef_hyper/s3"
ST="results/repro/t2a_base_diag/gemma2b_stripdef_static/s3"
HSNAP="$HY/snapshots/step15000.pt"
SSNAP="$ST/snapshots/step15000.pt"

echo "[$(date)] waiting for both durable step-15k handoff points"
while [[ ! -f "$HSNAP" || ! -f "$SSNAP" ]]; do
  if ! kill -0 "$HPID" 2>/dev/null || ! kill -0 "$SPID" 2>/dev/null; then
    echo "[$(date)] an original torchrun parent exited before both 15k snapshots appeared" >&2
    exit 1
  fi
  sleep 30
done

# The rolling optimizer checkpoints are atomically replaced before the model-only
# snapshots are written.  Let both large snapshot writes settle before stopping
# the old parents, so the training-curve artifacts are valid too.
sleep 30
echo "[$(date)] step-15k snapshots present; stopping PTY-hosted torchrun parents"
kill -TERM "$HPID" "$SPID"
while kill -0 "$HPID" 2>/dev/null || kill -0 "$SPID" 2>/dev/null; do
  sleep 5
done
sleep 30

echo "[$(date)] restarting seed 3 in this detached tmux session"
scripts/reproduce/task_t2a_lora.sh 3 0,1,2,3 4,5,6,7
echo "[$(date)] detached seed-3 training and held-out evaluations complete"
