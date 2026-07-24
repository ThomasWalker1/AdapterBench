#!/usr/bin/env python3
"""Select D2L candidates by controlled normalized log-length AUC.

Candidate directories have the restart-safe layout used by
``d2p_niah_multifidelity.sh``: ``<root>/scale<SCALE>/s<SEED>/results.jsonl``.
The 256-token result is a helpfulness/control gate. Selection then uses trapezoidal
area under matched-minus-context-swap over log2 context length for the hard bins.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


DEFAULT_HARD_LENGTHS = (512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True)
class Candidate:
    scale: str
    steps: int
    auc: float
    gate_delta: float
    frozen_256: float


def controlled_auc(points: dict[int, float], lengths: tuple[int, ...] = DEFAULT_HARD_LENGTHS) -> float:
    if any(length not in points for length in lengths):
        missing = [length for length in lengths if length not in points]
        raise ValueError(f"missing hard-bin results: {missing}")
    xs = [math.log2(length) for length in lengths]
    ys = [points[length] for length in lengths]
    area = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1))
    return area / (xs[-1] - xs[0])


def load_candidate(
    results_path: Path, adapter: str, hard_lengths: tuple[int, ...], gate_length: int = 256,
) -> Candidate:
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    points: dict[int, float] = {}
    gate_delta = None
    frozen_256 = None
    adapter_steps = None
    for row in rows:
        if row["split"] != "test" or not row["task_id"].startswith("niah_"):
            continue
        length = int(row["task_id"].split("_", 1)[1])
        metrics = row["metrics"]
        if row["adapter"] == adapter and "accuracy_ctxswap" in metrics:
            adapter_steps = int(row.get("metadata", {}).get("steps", -1))
            delta = float(metrics["accuracy"]) - float(metrics["accuracy_ctxswap"])
            points[length] = delta
            if length == gate_length:
                gate_delta = delta
        elif row["adapter"] == "frozen_interpreter" and length == gate_length:
            frozen_256 = float(metrics["accuracy"])
    if gate_delta is None or frozen_256 is None or adapter_steps is None:
        raise ValueError(f"missing {gate_length}-token adapter or frozen result")
    scale_dir = results_path.parent.parent.name
    if not scale_dir.startswith("scale"):
        raise ValueError(f"cannot infer scale from {results_path}")
    return Candidate(
        scale=scale_dir.removeprefix("scale"), steps=adapter_steps, auc=controlled_auc(points, hard_lengths),
        gate_delta=gate_delta, frozen_256=frozen_256,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--top", type=int, default=1)
    parser.add_argument("--hard-lengths", default="512,1024,2048,4096,8192")
    parser.add_argument("--gate-length", type=int, default=256,
                        help="shortest in-distribution length used for helpfulness/control gating")
    parser.add_argument("--steps", type=int, help="only consider candidates evaluated at this completed step budget")
    parser.add_argument(
        "--retain-all-if-no-gate",
        action="store_true",
        help="when no candidate has learned the helpfulness gate yet, emit every completed candidate for a later rung",
    )
    parser.add_argument("--format", choices=("table", "scales"), default="table")
    args = parser.parse_args()
    lengths = tuple(int(value) for value in args.hard_lengths.split(","))
    if len(lengths) < 2 or tuple(sorted(lengths)) != lengths:
        raise SystemExit("--hard-lengths must contain at least two increasing lengths")
    completed = []
    for path in sorted(args.root.glob("scale*/s*/results.jsonl")):
        candidate = load_candidate(path, args.adapter, lengths, args.gate_length)
        if args.steps is None or candidate.steps == args.steps:
            completed.append(candidate)
    candidates = [candidate for candidate in completed if candidate.gate_delta > candidate.frozen_256]
    retained_without_signal = False
    if not candidates:
        if not args.retain_all_if_no_gate or not completed:
            raise SystemExit(f"no candidate passed the {args.gate_length}-token helpfulness gate")
        # A zero-signal early rung cannot distinguish candidates.  Retain every one rather
        # than making an arbitrary AUC tie-break look like a selection decision; the gate
        # remains mandatory once any candidate produces a helpful controlled signal.
        candidates = completed
        retained_without_signal = True
    candidates.sort(key=lambda candidate: (-candidate.auc, float(candidate.scale)))
    selected = candidates if retained_without_signal else candidates[:args.top]
    if args.format == "scales":
        print(" ".join(candidate.scale for candidate in selected))
    else:
        print("scale  steps  normalized_log_auc  gate_delta  frozen_256  promoted")
        for candidate in candidates:
            print(
                f"{candidate.scale:>5}  {candidate.steps:>5}  {candidate.auc:>18.6f}  {candidate.gate_delta:>10.6f}  "
                f"{candidate.frozen_256:>10.6f}  {'yes' if candidate in selected else ''}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
