#!/bin/bash
# Run one paired T2A steering trial and append its controlled held-out CE result to
# the codec ledger.  The ledger is append-only: reruns are new observations, never
# replacements.  Mirrors scripts/t2a_lokr_record_trial.sh; scripts/t2a_scale_locator.sh
# accepts only lora|ia3, so each later codec brings its own thin launcher around the shared
# scripts/t2a_codec_trial.sh.  Nothing here touches the interpreter, conditioner,
# data, hook site, evaluator, control, or the codec's shape identity.
#
# Usage: scripts/t2a_steering_record_trial.sh PHASE SCALE SEED HYPER_GPUS STATIC_GPUS ROOT [LR] [STEPS] [CE_LIMIT] [ACC_LIMIT]
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:?phase required}"; SCALE="${2:?scale required}"; SEED="${3:?seed required}"
GH="${4:?hyper GPU CSV required}"; GS="${5:?static GPU CSV required}"; ROOT="${6:?artifact root required}"
LR="${7:-1e-4}"; STEPS="${8:-8000}"; CE_LIMIT="${9:-24}"; ACC_LIMIT="${10:-0}"
PGB="${PGB:-16}"; LEDGER="${LEDGER:-results/autoresearch/t2a/steering/state.jsonl}"
mkdir -p "$(dirname "$LEDGER")"
COMMAND="STEPS=$STEPS PGB=$PGB LR=$LR LIMIT=40 CE_LIMIT=$CE_LIMIT ACC_LIMIT=$ACC_LIMIT NO_COMPILE=1 bash scripts/t2a_codec_trial.sh steering $SCALE $SEED $GH $GS $ROOT"

if eval "$COMMAND"; then
  STATUS=complete
  NOTES="Held-out SNI CE; lower is better. Equal-shape static control and frozen helpfulness floor."
else
  STATUS=numerically_invalid
  NOTES="Trial command failed or produced an invalid trajectory; retained explicitly and not eligible for selection."
fi

.venv/bin/python - "$LEDGER" "$ROOT" "$PHASE" "$SCALE" "$SEED" "$LR" "$STEPS" "$PGB" "$CE_LIMIT" "$ACC_LIMIT" "$STATUS" "$NOTES" "$COMMAND" <<'PY'
import json
import math
import sys
from pathlib import Path

(ledger, root, phase, scale, seed, lr, steps, batch, ce_limit, acc_limit, status, notes, command) = sys.argv[1:]
payload = {
    "phase": phase,
    "setting": "t2a",
    "codec": "steering",
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
ce_path = Path(root) / "hyper" / f"s{seed}" / "heldout_sni_ce_full21.jsonl"
if status == "complete" and ce_path.exists():
    rows = [json.loads(line) for line in ce_path.read_text().splitlines() if line.strip()]
    aggregate = next((row for row in rows if row.get("task_id") == "__aggregate__"), None)
    values = [
        aggregate.get("matched_minus_static"), aggregate.get("ce_static"), aggregate.get("matched_minus_frozen"),
    ] if aggregate else []
    if aggregate is None or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        payload["status"] = "numerically_invalid"
        payload["notes"] = "Missing or non-finite controlled held-out CE aggregate; rejected explicitly."
    else:
        payload["selection_metric"], payload["control_metric"], payload["helpfulness_metric"] = values
        payload["ce_matched"] = aggregate.get("ce_matched")
        payload["ce_static"] = aggregate.get("ce_static")
        payload["ce_frozen"] = aggregate.get("ce_frozen")
        payload["matched_beats_static_tasks"] = aggregate.get("matched_beats_static")
else:
    payload["notes"] += " Held-out aggregate unavailable."
with Path(ledger).open("a") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
print(json.dumps({k: payload[k] for k in ("free_hparams", "status", "selection_metric", "helpfulness_metric")}, sort_keys=True))
PY

# A rejected point is data for the bounded ladder, not a launcher failure: its
# append-only record above is the audit trail and the caller continues to the
# remaining predeclared scales.
exit 0
