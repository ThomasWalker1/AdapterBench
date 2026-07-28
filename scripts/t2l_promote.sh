#!/bin/bash
# Resume viable T2L locator points to one common larger checkpoint budget.  The
# trial root is unchanged, so optimizer, scheduler, and generated-adapter state
# are resumed rather than restarted.
#
# Usage: scripts/t2l_promote.sh ADAPTER SCALES_CSV SEED ROOT [STEPS] [FROM_STEPS]
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:?adapter required}"; SCALES_CSV="${2:?comma-separated scales required}"
SEED="${3:?seed required}"; ROOT="${4:?locator root required}"; STEPS="${5:-8000}"; FROM_STEPS="${6:-4000}"
case "$ADAPTER" in lora|ia3) ;; *) echo "unsupported adapter: $ADAPTER" >&2; exit 2 ;; esac
PGB="${PGB:-16}"; LR="${LR:-1e-4}"; LIMIT="${LIMIT:-40}"; CE_LIMIT="${CE_LIMIT:-24}"
GH="${GPUS_HYPER:-0,1,2,3}"; GS="${GPUS_STATIC:-4,5,6,7}"
IFS=',' read -ra SCALES <<< "$SCALES_CSV"

for SCALE in "${SCALES[@]}"; do
  TRIAL_ROOT="$ROOT/scale$SCALE"
  PRIOR="$TRIAL_ROOT/hyper/s$SEED/snapshots/step${FROM_STEPS}.pt"
  [[ -f "$PRIOR" ]] || { echo "missing required checkpoint: $PRIOR" >&2; exit 1; }
  CMD="STEPS=$STEPS PGB=$PGB LR=$LR LIMIT=$LIMIT CE_LIMIT=$CE_LIMIT bash scripts/t2l_codec_trial.sh $ADAPTER $SCALE $SEED $GH $GS $TRIAL_ROOT"
  echo "=== promotion: $CMD ==="
  STEPS="$STEPS" PGB="$PGB" LR="$LR" LIMIT="$LIMIT" CE_LIMIT="$CE_LIMIT" \
    bash scripts/t2l_codec_trial.sh "$ADAPTER" "$SCALE" "$SEED" "$GH" "$GS" "$TRIAL_ROOT"
  .venv/bin/python - "$ROOT/state.jsonl" "$ADAPTER" "$SCALE" "$SEED" "$STEPS" "$LR" "$TRIAL_ROOT" "$CMD" "$FROM_STEPS" <<'PY'
import json
import math
import sys
from pathlib import Path

state_path, codec, scale, seed, steps, lr, root, command, from_steps = sys.argv[1:]
rows = [json.loads(line) for line in (Path(root) / "hyper" / f"s{seed}" / "heldout_sni_ce_full21.jsonl").read_text().splitlines() if line.strip()]
summary = [row for row in rows if row["task_id"] == "__aggregate__" and row["step"] == int(steps)][-1]
finite = all(math.isfinite(float(summary[key])) for key in ("ce_matched", "ce_static", "ce_frozen", "matched_minus_static", "matched_minus_frozen"))
record = {
    "phase": "promotion", "setting": "t2l", "codec": codec, "seed": int(seed),
    "free_hparams": {"scale": float(scale), "learning_rate": float(lr), "steps": int(steps), "warmup_frac": 0.1},
    "command": command, "artifact_root": root, "status": "complete" if finite else "numerically_invalid",
    "selection_metric": summary["matched_minus_static"] if finite else None,
    "control_metric": summary["ce_static"] if finite else None,
    "helpfulness_metric": summary["matched_minus_frozen"] if finite else None,
    "notes": f"Resumed exactly from the {from_steps}-step checkpoint; held-out SNI CE, lower is better.",
}
with Path(state_path).open("a") as handle:
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps(record, sort_keys=True))
PY
done
