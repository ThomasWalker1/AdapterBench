#!/usr/bin/env python3
"""Append one `exploratory_screen` ledger record for a D2A DoRA screening point.

Status is always `exploratory`: these records exist for auditability, never as selection
evidence. A window located here is re-derived from scratch under AUTORESEARCH.md before it
can become a leaderboard row, and the record says so.

Records the gate (512-token in-distribution) accuracy at every evaluated rung, not only the
final one, because a transient peak that collapses is exactly what a too-large effective
step size looks like and is the thing the screen is trying to see.

Usage:
  d2a_dora_record_screen.py --run <run dir> --scale 4 --learning-rate 2e-5 --steps 8000 \
      --warmup-steps 960 --seed 902 --eval-lengths 512,1024 --eval-limit 12 \
      --ledger results/autoresearch/d2a/dora/state.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

GATE_LENGTH = 512
STEP_RE = re.compile(r"^\[step (\d+)\] loss=([^\s]+) (\{.*\})\s*$")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--scale", required=True)
    p.add_argument("--learning-rate", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--warmup-steps", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--eval-lengths", required=True)
    p.add_argument("--eval-limit", type=int, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    args = p.parse_args()

    rungs = []
    for log in sorted(args.run.glob("screen_*.log")):
        for line in log.read_text(errors="replace").splitlines():
            m = STEP_RE.match(line)
            if not m:
                continue
            try:
                loss = float(m.group(2))
            except ValueError:
                loss = float("nan")
            payload = ast.literal_eval(m.group(3))
            rungs.append({
                "step": int(m.group(1)),
                "loss": None if not math.isfinite(loss) else loss,
                "gate_accuracy": payload.get(f"niah_{GATE_LENGTH}", {}).get("accuracy"),
                "gate_ctxswap": payload.get(f"niah_{GATE_LENGTH}", {}).get("accuracy_ctxswap"),
                "accuracy_by_length": {k.split("_")[1]: v["accuracy"] for k, v in payload.items()},
            })

    gates = [r["gate_accuracy"] for r in rungs if r["gate_accuracy"] is not None]
    diverged = any(r["loss"] is None for r in rungs)
    record = {
        "phase": "exploratory_screen",
        "setting": "d2a",
        "codec": "dora",
        "seed": args.seed,
        "free_hparams": {
            "scale": float(args.scale), "learning_rate": float(args.learning_rate),
            "steps": args.steps, "warmup_steps": args.warmup_steps,
        },
        "command": (
            f"bash scripts/d2a_dora_explore.sh <gpus> \"{args.scale}:{args.learning_rate}\" {args.steps}"
        ),
        "artifact_root": str(args.run),
        "status": "exploratory",
        "selection_metric": None,
        "control_metric": max((r["gate_ctxswap"] for r in rungs if r["gate_ctxswap"] is not None), default=None),
        "helpfulness_metric": max(gates, default=None),
        "peak_gate_accuracy": max(gates, default=None),
        "final_gate_accuracy": gates[-1] if gates else None,
        "diverged": diverged,
        "rungs": rungs,
        "notes": (
            "EXPLORATORY SCREEN, NOT SELECTION EVIDENCE. Wide/short/low-resolution sweep over the free "
            f"axes to find out whether any operating window exists at all, scored at lengths "
            f"{args.eval_lengths} with {args.eval_limit} examples/bin on the DEV eval seed 1802 and the "
            "scout seed. The truncated eval instrument makes hard-length AUC incomparable with the "
            "declared-ladder rungs by design; only the 512-token gate column is comparable. The "
            "confirmation seeds and the held-out eval seed are untouched. Any window located here is "
            "re-derived from scratch under AUTORESEARCH.md (declared ladder, 3 selection seeds, "
            "stability gate, 5 fresh confirmation seeds) before it can become a leaderboard row."
        ),
    }
    with args.ledger.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    peak = record["peak_gate_accuracy"]
    print(f"  screen scale={args.scale:>6} lr={args.learning_rate:>6} "
          f"peak_gate={peak if peak is not None else float('nan'):.3f} "
          f"final_gate={record['final_gate_accuracy'] if record['final_gate_accuracy'] is not None else float('nan'):.3f} "
          f"{'DIVERGED' if diverged else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
