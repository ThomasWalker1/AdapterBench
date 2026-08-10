#!/bin/bash
# Score the one-shot T2A report split for one or more codecs.
#
# Verifies every expected confirmation checkpoint converged before touching the report split.
# Pass a comma-separated codec list or a single codec name.
#
# Usage: scripts/reproduce/t2a_score_codec.sh CODEC[,CODEC...]
#   e.g. scripts/reproduce/t2a_score_codec.sh lora
#        scripts/reproduce/t2a_score_codec.sh lora,ia3,lokr,fourierft,steering
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1

CODECS="${1:?codec list required}"
R=results/autoresearch/t2a/evaluation
LOG_DIR=results/autoresearch/t2a/confirmation_logs
mkdir -p "$R/report_scores" "$R/report_scores_ce" "$LOG_DIR"

if ! .venv/bin/python scripts/t2a_verify_confirmation.py --codecs "$CODECS"; then
  echo "verification failed — report split NOT scored" >&2
  exit 1
fi

.venv/bin/python scripts/t2a_checkpoint_index.py --codecs "$CODECS" >/dev/null

for spec in "generation report_scores gen" "ce report_scores_ce ce"; do
  set -- $spec
  echo "=== report split $3 pass ($(date)) ==="
  IFS=',' read -ra CODEC_ARR <<< "$CODECS"
  EXPECT=$(( ${#CODEC_ARR[@]} * 6 ))  # 3 seeds x (hyper + static) per codec
  for i in $(seq 0 7); do
    .venv/bin/python scripts/t2a_score_checkpoints.py \
      --shard "$i" --num-shards 8 --device "cuda:$i" --codecs "$CODECS" \
      --split report --confirm-spend-report-split --example-offset 0 \
      --checkpoint-filter confirmation_v2 --expect-checkpoints "$EXPECT" \
      --metric "$1" --limit 64 --n-desc 3 --batch-size 16 \
      --out-dir "$R/$2" > "$LOG_DIR/report_${3}_$i.log" 2>&1 &
  done
  wait
done

.venv/bin/python scripts/t2a_confirmation_result.py
echo "=== report split scored for $CODECS ($(date)) ==="
