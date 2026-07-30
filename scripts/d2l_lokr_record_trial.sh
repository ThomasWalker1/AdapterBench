#!/bin/bash
# Run one restart-safe D2L LoKr trial and append its controlled retrieval result to
# the LoKr-only ledger. Reruns append a new observation; no prior trial is replaced.
#
# Usage: scripts/d2l_lokr_record_trial.sh PHASE SCALE SEED GPU STEPS ROOT [LR]
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:?phase required}"; SCALE="${2:?scale required}"; SEED="${3:?seed required}"
GPU="${4:?GPU required}"; STEPS="${5:?steps required}"; ROOT="${6:?artifact root required}"
LR="${7:-4e-5}"; LEDGER="results/autoresearch/d2l/lokr/state.jsonl"
WARMUP_STEPS="${WARMUP_STEPS:-1080}"
EVAL_EVERY="${EVAL_EVERY:-8000}"; EVAL_LIMIT="${EVAL_LIMIT:-32}"; EVAL_SEED="${EVAL_SEED:-2904}"
TRAIN_LENGTH="${TRAIN_LENGTH:-512}"; HARD_LENGTHS="${HARD_LENGTHS:-1024,2048,4096,8192,16384,32768}"
EVAL_LENGTHS="$TRAIN_LENGTH,$HARD_LENGTHS"
mkdir -p "$(dirname "$LEDGER")" "$ROOT"

COMMAND=".venv/bin/adapterbench d2p-niah --adapters lokr --lokr-scaling $SCALE --needle-style realistic_numeric_decoys --numeric-decoy-count 4 --context-lengths $TRAIN_LENGTH --eval-context-lengths $EVAL_LENGTHS --num-train-documents 512 --steps $STEPS --eval-every $EVAL_EVERY --learning-rate $LR --warmup-steps $WARMUP_STEPS --n-latents 208 --num-blocks 8 --eval-limit $EVAL_LIMIT --eval-seed $EVAL_SEED --seed $SEED --device cuda:$GPU --output $ROOT"
if eval "$COMMAND" > "$ROOT/run_${STEPS}.log" 2>&1; then
  STATUS=complete
  NOTES="Numeric-decoy D2L controlled retrieval. Selection is normalized log-length AUC after the 512-token matched-vs-frozen gate; context-swap must remain near zero."
else
  STATUS=numerically_invalid
  NOTES="Trial command failed or did not produce a valid controlled length curve; retained explicitly and not eligible for selection."
fi

.venv/bin/python - "$LEDGER" "$ROOT" "$PHASE" "$SCALE" "$SEED" "$GPU" "$LR" "$STEPS" "$WARMUP_STEPS" "$HARD_LENGTHS" "$TRAIN_LENGTH" "$STATUS" "$NOTES" "$COMMAND" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

(ledger, root, phase, scale, seed, gpu, lr, steps, warmup, hard_lengths, train_length,
 status, notes, command) = sys.argv[1:]
payload = {
    "phase": phase,
    "setting": "d2l",
    "codec": "lokr",
    "seed": int(seed),
    "free_hparams": {"scale": float(scale), "learning_rate": float(lr), "warmup_steps": int(warmup), "steps": int(steps)},
    "command": command,
    "artifact_root": root,
    "status": status,
    "selection_metric": None,
    "control_metric": None,
    "helpfulness_metric": None,
    "notes": notes,
}
results_path = Path(root) / "results.jsonl"
if status == "complete" and results_path.exists():
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    final = [row for row in rows if row.get("adapter") == "lokr" and row.get("metadata", {}).get("steps") == int(steps)]
    frozen = [row for row in rows if row.get("adapter") == "frozen_interpreter"]
    by_length = {int(row["task_id"].split("_", 1)[1]): row for row in final if row.get("task_id", "").startswith("niah_")}
    frozen_by_length = {int(row["task_id"].split("_", 1)[1]): row for row in frozen if row.get("task_id", "").startswith("niah_")}
    lengths = [int(item) for item in hard_lengths.split(",")]
    gate = int(train_length)
    try:
        deltas = [float(by_length[length]["metrics"]["accuracy"]) - float(by_length[length]["metrics"]["accuracy_ctxswap"]) for length in lengths]
        auc = sum((deltas[i] + deltas[i + 1]) / 2 for i in range(len(deltas) - 1)) / (len(deltas) - 1)
        matched = float(by_length[gate]["metrics"]["accuracy"])
        control = float(by_length[gate]["metrics"]["accuracy_ctxswap"])
        frozen_gate = float(frozen_by_length[gate]["metrics"]["accuracy"])
        values = [auc, matched, control, frozen_gate]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite metric")
        payload["selection_metric"] = auc
        payload["control_metric"] = control
        payload["helpfulness_metric"] = matched - frozen_gate
        payload["notes"] += f" gate matched={matched:.6f}, frozen={frozen_gate:.6f}, ctxswap={control:.6f}, hard_AUC={auc:.6f}."
        payload["artifact_sha256"] = hashlib.sha256(results_path.read_bytes()).hexdigest()
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        payload["status"] = "numerically_invalid"
        payload["notes"] = f"Missing/non-finite final controlled D2L curve: {exc}."
else:
    payload["notes"] += " Results file unavailable."
with Path(ledger).open("a") as handle:
    handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps(payload, sort_keys=True))
PY

# A rejected scale is still a bounded-ladder observation; let the launcher finish
# the other declared candidates and use the ledger for the decision.
exit 0
