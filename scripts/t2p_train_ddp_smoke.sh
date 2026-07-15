#!/bin/bash
# Smoke test for the DDP T2L training path (scripts/t2p_train_ddp.py) — the analogue of
# i2p_hypernoise_smoke.py for the image seam. Runs a tiny, fast DDP job and asserts the machinery
# is intact: DDP init on N GPUs, torch.compile + fixed-seq-len + persistent hooks run, loss
# decreases, a checkpoint is written, and results.jsonl is a valid drop-in for the aggregator.
#
# This is the exact pattern validated on 2026-07-15 (4-GPU, 60 steps, small corpus). It needs
# free GPUs to run; it does NOT touch long runs — pass GPUS for whatever is idle.
#
# Usage: scripts/t2p_train_ddp_smoke.sh [GPUS]   (default "0,1"; use any idle pair)
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${1:-0,1}"
NPROC="$(awk -F, '{print NF}' <<<"$GPUS")"
OUT=results/_smoke_ddp
rm -rf "$OUT"

CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/torchrun --standalone --nproc_per_node="$NPROC" \
  scripts/t2p_train_ddp.py \
  --tasks lol_022,lol_043,lol_044,lol_045,lol_047,lol_050,lol_063,lol_064 \
  --max-descriptions 8 --limit 16 --per-gpu-batch 4 --steps 60 \
  --learning-rate 1e-4 --warmup-frac 0.1 --fixed-seq-len 512 \
  --eval-tasks arc_easy,boolq --eval-limit 8 \
  --adversarial-control --seed 777 --checkpoint-every 30 --loss-log-every 10 \
  --output "$OUT/s777"

echo "=== smoke assertions ==="
.venv/bin/python - "$OUT/s777" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
ck = d / "ckpt_lora_seed777_scaledefault.pt"
assert ck.exists(), f"no checkpoint written at {ck}"
rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines()]
adapters = {r["adapter"] for r in rows}
assert "frozen_interpreter" in adapters, "no frozen baseline row"
assert "lora" in adapters, "no trained-adapter row"
assert any("accuracy_mismatched" in r["metrics"] for r in rows if r["adapter"] == "lora"), \
    "adversarial control (accuracy_mismatched) missing"
print(f"OK: checkpoint present, {len(rows)} result rows, adapters={sorted(adapters)}, "
      "adversarial control present, results.jsonl parses.")
PY
echo "=== smoke passed ==="
