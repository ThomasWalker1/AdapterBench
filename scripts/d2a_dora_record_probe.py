#!/usr/bin/env python3
"""Append one append-only ledger record per D2A DoRA boundary-probe point.

Reads each probe run's eval lines, records the 512-token in-distribution gate, the
per-length accuracies, the maximum context-swap seen, and the normalized hard-length
log-AUC (the setting's declared selection statistic), and classifies the point:

  * ``complete``            - trained to budget with finite loss
  * ``numerically_invalid`` - non-finite loss at any eval, i.e. divergence

Nothing here selects: it records. A probe that diverged is retained explicitly, since a
narrow stable band is itself the finding this probe exists to establish.

Usage:
  d2a_dora_record_probe.py --root results/autoresearch/d2a/dora/scale_boundary \
      --scales 64,256 --seed 902 --learning-rate 4e-5 --steps 32000 \
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
HARD_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768)
STEP_RE = re.compile(r"^\[step (\d+)\] loss=([^\s]+) (\{.*\})\s*$")


def normalized_log_auc(deltas: dict[int, float]) -> float | None:
    lengths = [l for l in HARD_LENGTHS if l in deltas]
    if len(lengths) < 2:
        return None
    xs = [math.log2(l) for l in lengths]
    ys = [deltas[l] for l in lengths]
    area = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1))
    return area / (xs[-1] - xs[0])


def read_evals(log: Path) -> list[tuple[int, float, dict]]:
    out = []
    for line in log.read_text(errors="replace").splitlines():
        m = STEP_RE.match(line)
        if not m:
            continue
        step, raw_loss, payload = int(m.group(1)), m.group(2), m.group(3)
        try:
            loss = float(raw_loss)
        except ValueError:  # nan/inf render as text
            loss = float("nan")
        out.append((step, loss, ast.literal_eval(payload)))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scales", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--learning-rate", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    # The same recorder serves the boundary probe and the refinement: they differ only in
    # phase name and on-disk layout, never in what is measured.
    p.add_argument("--phase", default="scale_boundary_extension")
    p.add_argument("--log-glob", default="probe_*.log")
    p.add_argument("--scale-dir", default=None,
                   help="run directory name under --root (default: scale<SCALE>)")
    args = p.parse_args()

    records = []
    for scale in args.scales.split(","):
        run = args.root / (args.scale_dir or f"scale{scale}") / f"s{args.seed}"
        logs = sorted(run.glob(args.log_glob))
        evals = read_evals(logs[-1]) if logs else []
        record = {
            "phase": args.phase,
            "setting": "d2a",
            "codec": "dora",
            "seed": args.seed,
            "free_hparams": {
                "scale": float(scale), "learning_rate": float(args.learning_rate),
                "steps": args.steps, "warmup_steps": 960,
            },
            "command": (
                f"bash scripts/d2a_dora_boundary_probe.sh (--codec-scaling {scale}, "
                f"--learning-rate {args.learning_rate}, --steps {args.steps})"
            ),
            "artifact_root": str(run),
            "status": "complete",
            "selection_metric": None,
            "control_metric": None,
            "helpfulness_metric": None,
        }
        if not evals:
            record["status"] = "numerically_invalid"
            record["notes"] = ("No evaluation reached; the probe produced no scorable rung. Retained "
                              "explicitly as a boundary observation rather than retried.")
            records.append(record)
            continue

        final_step, final_loss, final = evals[-1]
        by_length = {int(k.split("_")[1]): v for k, v in final.items()}
        gate = by_length.get(GATE_LENGTH, {}).get("accuracy")
        deltas = {l: v["accuracy"] - v["accuracy_ctxswap"] for l, v in by_length.items()}
        diverged = any(not math.isfinite(loss) for _, loss, _ in evals)
        # A transient peak that later collapses is exactly what a too-large scale looks like,
        # so record the best gate value any rung reached, not only the final one.
        peaked = max(
            (payload[f"niah_{GATE_LENGTH}"]["accuracy"]
             for _, _, payload in evals if f"niah_{GATE_LENGTH}" in payload),
            default=0.0,
        )
        record.update({
            "status": "numerically_invalid" if diverged else "complete",
            "final_step": final_step,
            "final_loss": None if not math.isfinite(final_loss) else final_loss,
            "gate_accuracy": gate,
            "peak_gate_accuracy": peaked,
            "accuracy_by_length": {str(l): v["accuracy"] for l, v in by_length.items()},
            "control_by_length": {str(l): v["accuracy_ctxswap"] for l, v in by_length.items()},
            "selection_metric": normalized_log_auc(deltas),
            "control_metric": max((v["accuracy_ctxswap"] for v in by_length.values()), default=None),
            "helpfulness_metric": gate,
            "notes": (
                ("Non-finite loss at one or more evals: divergence, ineligible for selection. "
                 if diverged else "")
                + ("Upward boundary probe above the declared ladder's top endpoint (16), at the "
                   "ladder's own x4 spacing and the substrate-default LR. "
                   if args.phase == "scale_boundary_extension" else
                   "Protocol refinement around the operating window located by the exploratory screen, "
                   "at the full declared eval instrument and the committed 32k budget. ")
                + "Dev eval seed 1802 at 12 examples/bin, so a single retrieved example is 0.0833 and "
                  "the document's 4 numeric decoys put the emit-an-arbitrary-in-document-number rate "
                  "near 0.2."
            ),
        })
        records.append(record)

    with args.ledger.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"{'scale':>8} {'status':>20} {'gate':>7} {'peak_gate':>10} {'hard_auc':>9} {'final_loss':>11}")
    for record in records:
        auc = record.get("selection_metric")
        loss = record.get("final_loss")
        print(f"{record['free_hparams']['scale']:>8g} {record['status']:>20} "
              f"{(record.get('gate_accuracy') if record.get('gate_accuracy') is not None else float('nan')):>7.3f} "
              f"{(record.get('peak_gate_accuracy') or 0.0):>10.3f} "
              f"{(auc if auc is not None else float('nan')):>9.3f} "
              f"{(loss if loss is not None else float('nan')):>11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
