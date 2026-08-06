#!/bin/bash
# Checkpointed, codec-neutral T2A scale locator.  Every point uses the released
# effective batch (4 GPUs x 16) for both the conditioned model and its equal-shape
# static control; it therefore consumes all eight GPUs and runs points sequentially.
#
# Usage: scripts/t2a_scale_locator.sh ADAPTER SCALES_CSV [SEED] [ROOT]
# Example: scripts/t2a_scale_locator.sh ia3 0.0625,0.25,1,4,16
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:?adapter required}"; SCALES_CSV="${2:?comma-separated scales required}"
SEED="${3:-1702}"; ROOT="${4:-results/autoresearch/t2a/$ADAPTER/scale_locator}"
case "$ADAPTER" in lora|ia3) ;; *) echo "unsupported adapter: $ADAPTER" >&2; exit 2 ;; esac

# The 4k first rung locates the viable scale region.  Promotion continues exact
# checkpoints to 8k; final confirmation is a separate, fresh-seed full-budget run.
STEPS="${STEPS:-4000}"; PGB="${PGB:-16}"; LR="${LR:-1e-4}"; LIMIT="${LIMIT:-40}"; CE_LIMIT="${CE_LIMIT:-24}"
GH="${GPUS_HYPER:-0,1,2,3}"; GS="${GPUS_STATIC:-4,5,6,7}"
mkdir -p "$ROOT"
.venv/bin/adapterbench preflight --setting t2a --devices "$GH,$GS" --output "$ROOT"

IFS=',' read -ra SCALES <<< "$SCALES_CSV"
for SCALE in "${SCALES[@]}"; do
  TRIAL_ROOT="$ROOT/scale$SCALE"
  mkdir -p "$TRIAL_ROOT"
  CMD="STEPS=$STEPS PGB=$PGB LR=$LR LIMIT=$LIMIT CE_LIMIT=$CE_LIMIT bash scripts/t2a_codec_trial.sh $ADAPTER $SCALE $SEED $GH $GS $TRIAL_ROOT"
  echo "=== locator: $CMD ==="
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
finite = all(math.isfinite(float(summary[key])) for key in (
    "ce_matched", "ce_static", "ce_frozen", "matched_minus_static", "matched_minus_frozen",
))
record = {
    "phase": "scale_locator", "setting": "t2a", "codec": codec, "seed": int(seed),
    "free_hparams": {"scale": float(scale), "learning_rate": float(lr), "steps": int(steps), "warmup_frac": 0.1},
    "command": command, "artifact_root": root, "status": "complete" if finite else "numerically_invalid",
    "selection_metric": summary["matched_minus_static"] if finite else None,
    "control_metric": summary["ce_static"] if finite else None,
    "helpfulness_metric": summary["matched_minus_frozen"] if finite else None,
    "notes": ("Held-out SNI CE; lower is better. Equal-shape static control and frozen helpfulness floor."
              if finite else "Numerically invalid (non-finite held-out metric); rejected from selection."),
}
with Path(state_path).open("a") as handle:
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps(record, sort_keys=True))
PY
done
