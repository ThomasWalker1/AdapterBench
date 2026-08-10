#!/bin/bash
# Verify the confirmation set, then measure the report split ONCE.
#
# The 11-task report split is a one-shot resource, so scoring is gated on an automated verification:
# every expected snapshot present, every ledger record marked complete, and no non-finite final
# training loss. If any check fails the script stops WITHOUT touching the split, leaving it intact.
#
# Scoring is restricted to the confirmation checkpoints via --checkpoint-filter, with
# --expect-checkpoints asserting the count: the driver selects by codec, not by phase, so without a
# filter it would score every in-scope checkpoint on held-out data and compromise any future
# re-selection.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
R=results/autoresearch/t2a/evaluation
L=results/autoresearch/t2a/confirmation_logs
C=lora,ia3,lokr,fourierft,steering
mkdir -p "$L"

{
echo "=== [$(date)] waiting for confirmation training to drain ==="
while pgrep -f t2a_train_ddp >/dev/null; do sleep 120; done

echo "=== [$(date)] verifying the confirmation set before spending the report split ==="
if ! .venv/bin/python scripts/t2a_verify_confirmation.py --codecs "$C"; then
  echo "!!! VERIFICATION FAILED -- report split NOT scored, still intact. Fix and rerun. !!!"
  exit 1
fi

echo "=== [$(date)] verification passed; indexing ==="
.venv/bin/python scripts/t2a_checkpoint_index.py --codecs "$C" >/dev/null

for spec in "generation report_scores gen" "ce report_scores_ce ce"; do
  set -- $spec
  echo "=== [$(date)] REPORT SPLIT $3 pass (one-shot) ==="
  for i in $(seq 0 7); do
    .venv/bin/python scripts/t2a_score_checkpoints.py \
      --shard "$i" --num-shards 8 --device "cuda:$i" --codecs "$C" \
      --split report --confirm-spend-report-split --example-offset 0 \
      --checkpoint-filter confirmation_v2 --expect-checkpoints 30 \
      --metric "$1" --limit 64 --n-desc 3 --batch-size 16 \
      --out-dir "$R/$2" > "$L/report_$3_$i.log" 2>&1 &
  done
  wait
  echo "=== [$(date)] $3 done: $(cat $R/$2/*.jsonl 2>/dev/null | wc -l) rows ==="
done

.venv/bin/python scripts/t2a_confirmation_result.py > "$L/confirmation_report.txt" 2>&1
echo "=== [$(date)] REPORT SPLIT SCORED -- confirmation numbers ready ==="
} >> "$L/score_driver.log" 2>&1
