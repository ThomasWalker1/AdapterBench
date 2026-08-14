#!/bin/bash
# Run one single-role T2A DoRA trial and append its record to the codec's append-only
# ledger (results/autoresearch/t2a/dora/state.jsonl).
#
# Single-role by design: the corrected T2A protocol selects the static* control on its
# own score (AUTORESEARCH.md §"T2A: metric, data split, and the independently selected
# control"), so the hypernetwork and the static are separate rungs that may sit at
# different (scale, lr). Each role gets the released effective batch (4 GPUs x PGB 16 =
# 64), so two roles fill the machine when run concurrently on disjoint GPU sets.
#
# Selection metrics are NOT written here: the selector is selection-split ROUGE-L at
# --example-offset 40, produced by scripts/t2a_score_checkpoints.py over the snapshots
# this trial writes. The ledger row is the audit trail for the run itself.
#
# Usage: scripts/t2a_dora_trial.sh PHASE ROLE SCALE LR SEED GPUS ROOT [STEPS]
#   e.g. scripts/t2a_dora_trial.sh scale_locator hyper 1 1e-4 6702 0,1,2,3 \
#          results/autoresearch/t2a/dora/scale_locator/scale1_lr1e-4_hyper
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:?phase required}"; ROLE="${2:?role required (hyper|static)}"; SCALE="${3:?scale required}"
LR="${4:?learning rate required}"; SEED="${5:?seed required}"; GPUS="${6:?GPU CSV required}"
ROOT="${7:?artifact root required}"; STEPS="${8:-8000}"
case "$ROLE" in hyper|static) ;; *) echo "ROLE must be hyper|static" >&2; exit 2 ;; esac
PGB="${PGB:-16}"; LIMIT="${LIMIT:-40}"
LEDGER="${LEDGER:-results/autoresearch/t2a/dora/state.jsonl}"
mkdir -p "$(dirname "$LEDGER")" "$ROOT"

COMMAND="STEPS=$STEPS LR=$LR ROLES=$ROLE LIMIT=$LIMIT PGB=$PGB SKIP_EVAL=1 SKIP_PREFLIGHT=1 NO_COMPILE=1 bash scripts/t2a_codec_trial.sh dora $SCALE $SEED $GPUS $GPUS $ROOT"

if STEPS="$STEPS" LR="$LR" ROLES="$ROLE" LIMIT="$LIMIT" PGB="$PGB" SKIP_EVAL=1 SKIP_PREFLIGHT=1 NO_COMPILE=1 \
     bash scripts/t2a_codec_trial.sh dora "$SCALE" "$SEED" "$GPUS" "$GPUS" "$ROOT"; then
  STATUS=complete
  NOTES="Trained to the declared common budget; selection-split ROUGE-L is scored separately by scripts/t2a_score_checkpoints.py (--split selection --example-offset 40)."
else
  STATUS=numerically_invalid
  NOTES="Trial command failed or produced an invalid trajectory; retained explicitly and not eligible for selection."
fi

.venv/bin/python - "$LEDGER" "$ROOT" "$PHASE" "$ROLE" "$SCALE" "$SEED" "$LR" "$STEPS" "$PGB" "$STATUS" "$NOTES" "$COMMAND" <<'PY'
import json
import math
import sys
from pathlib import Path

(ledger, root, phase, role, scale, seed, lr, steps, batch, status, notes, command) = sys.argv[1:]
payload = {
    "phase": phase,
    "setting": "t2a",
    "codec": "dora",
    "role": role,
    "seed": int(seed),
    "free_hparams": {
        "scale": float(scale), "learning_rate": float(lr), "warmup_frac": 0.1,
        "steps": int(steps), "effective_batch": 4 * int(batch),
    },
    "command": command,
    "artifact_root": root,
    "status": status,
    "selection_metric": None,
    "control_metric": None,
    "helpfulness_metric": None,
    "notes": notes,
}
snapshot = Path(root) / role / f"s{seed}" / "snapshots" / f"step{int(steps)}.pt"
payload["snapshot"] = str(snapshot) if snapshot.exists() else None
meta = Path(root) / role / f"s{seed}" / "train_meta.json"
if meta.exists():
    losses = json.loads(meta.read_text()).get("losses") or []
    if losses:
        tail = losses[-5:]
        payload["final_loss"] = tail[-1]
        payload["converged"] = all(math.isfinite(v) for v in losses) and tail[-1] < losses[0]
        if not payload["converged"]:
            payload["status"] = "numerically_invalid"
            payload["notes"] = "Non-finite or non-decreasing training loss; divergence, not eligible for selection."
if payload["snapshot"] is None and payload["status"] == "complete":
    payload["status"] = "numerically_invalid"
    payload["notes"] = "Expected snapshot missing after a nominally successful run; rejected explicitly."
with Path(ledger).open("a") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
print(json.dumps({k: payload[k] for k in ("phase", "role", "free_hparams", "status", "final_loss")
                  if k in payload}, sort_keys=True))
PY

# A rejected rung is data for the bounded ladder, not a launcher failure: the append-only
# record above is the audit trail and the caller continues to the remaining rungs.
exit 0
