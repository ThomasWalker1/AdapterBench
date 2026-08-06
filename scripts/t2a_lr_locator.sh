#!/bin/bash
# Codec-specific T2A learning-rate locator at an already selected scale.
# Each point starts fresh: changing LR changes the optimizer trajectory and must
# not reuse a checkpoint trained with a different LR.
#
# Usage: scripts/t2a_lr_locator.sh ADAPTER SCALE LRS_CSV SEED ROOT [STEPS]
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:?adapter required}"; SCALE="${2:?scale required}"; LRS_CSV="${3:?comma-separated learning rates required}"
SEED="${4:?seed required}"; ROOT="${5:?output root required}"; STEPS="${6:-8000}"
case "$ADAPTER" in lora|ia3) ;; *) echo "unsupported adapter: $ADAPTER" >&2; exit 2 ;; esac
PGB="${PGB:-16}"; LIMIT="${LIMIT:-40}"; CE_LIMIT="${CE_LIMIT:-24}"
GH="${GPUS_HYPER:-0,1,2,3}"; GS="${GPUS_STATIC:-4,5,6,7}"
mkdir -p "$ROOT"
.venv/bin/adapterbench preflight --setting t2a --devices "$GH,$GS" --output "$ROOT"
IFS=',' read -ra LRS <<< "$LRS_CSV"

for LR in "${LRS[@]}"; do
  TRIAL_ROOT="$ROOT/lr$LR"
  mkdir -p "$TRIAL_ROOT"
  CMD="STEPS=$STEPS PGB=$PGB LR=$LR LIMIT=$LIMIT CE_LIMIT=$CE_LIMIT bash scripts/t2a_codec_trial.sh $ADAPTER $SCALE $SEED $GH $GS $TRIAL_ROOT"
  echo "=== LR locator: $CMD ==="
  STEPS="$STEPS" PGB="$PGB" LR="$LR" LIMIT="$LIMIT" CE_LIMIT="$CE_LIMIT" \
    bash scripts/t2a_codec_trial.sh "$ADAPTER" "$SCALE" "$SEED" "$GH" "$GS" "$TRIAL_ROOT"
  .venv/bin/python - "$ROOT/state.jsonl" "$ADAPTER" "$SCALE" "$SEED" "$STEPS" "$LR" "$TRIAL_ROOT" "$CMD" <<'PY'
import json
import math
import sys
from pathlib import Path

state_path, codec, scale, seed, steps, lr, root, command = sys.argv[1:]
rows = [json.loads(line) for line in (Path(root) / "hyper" / f"s{seed}" / "heldout_sni_ce_full21.jsonl").read_text().splitlines() if line.strip()]
summary = next(row for row in rows if row["task_id"] == "__aggregate__")
finite = all(math.isfinite(float(summary[key])) for key in ("ce_matched", "ce_static", "ce_frozen", "matched_minus_static", "matched_minus_frozen"))
record = {
    "phase": "learning_rate_locator", "setting": "t2a", "codec": codec, "seed": int(seed),
    "free_hparams": {"scale": float(scale), "learning_rate": float(lr), "steps": int(steps), "warmup_frac": 0.1},
    "command": command, "artifact_root": root, "status": "complete" if finite else "numerically_invalid",
    "selection_metric": summary["matched_minus_static"] if finite else None,
    "control_metric": summary["ce_static"] if finite else None,
    "helpfulness_metric": summary["matched_minus_frozen"] if finite else None,
    "notes": "Fresh optimizer trajectory at fixed selected scale; held-out SNI CE, lower is better.",
}
with Path(state_path).open("a") as handle:
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps(record, sort_keys=True))
PY
done
