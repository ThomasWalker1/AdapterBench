#!/usr/bin/env python3
"""Append one append-only ledger record per completed D2A steering trial at a rung.

The D2A locator itself is the shared, codec-neutral
``scripts/d2a_niah_numeric_decoy_dev.sh`` launcher (steering adds no per-codec
launcher and no per-codec scale flag: the generic ``--codec-scaling`` carries the
swept scale). That launcher writes artifacts but no ledger, so this reads the
artifacts it produced and emits the ``AUTORESEARCH.md`` trial records, exactly
reconstructing the command the launcher ran for each trial.

Records are never overwritten: a rerun appends a new observation.

Usage:
  d2a_steering_record_rung.py --root RESULTS_ROOT --steps N --phase PHASE \
      [--seed 902] [--learning-rate 4e-5] [--ledger PATH] [--notes TEXT]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# The locked numeric-decoy dev locator's evaluation grid (see
# scripts/d2a_niah_numeric_decoy_dev.sh): 512 is the shortest in-distribution
# helpfulness/control gate, the rest are the declared doubled hard bins.
GATE_LENGTH = 512
HARD_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768)


def controlled_log_auc(points: dict[int, float], lengths: tuple[int, ...]) -> float:
    """Trapezoidal area of matched-minus-control over log2 length, normalized to a mean."""
    xs = [math.log2(length) for length in lengths]
    ys = [points[length] for length in lengths]
    area = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1))
    return area / (xs[-1] - xs[0])


def command_for(scale: str, steps: int, seed: int, learning_rate: str, output: Path) -> str:
    """Reconstruct the exact d2a-niah invocation the shared launcher ran for this trial.

    The device is intentionally omitted from the reconstruction only when unknown;
    every other argument is fixed by the dev locator's environment.
    """
    return (
        f".venv/bin/adapterbench d2a-niah --adapters steering --codec-scaling {scale} "
        "--needle-style realistic_numeric_decoys --numeric-decoy-count 4 "
        "--context-lengths 512 --eval-context-lengths 512,1024,2048,4096,8192,16384,32768 "
        f"--num-train-documents 512 --steps {steps} --eval-every 8000 "
        f"--learning-rate {learning_rate} --warmup-steps 960 "
        "--n-latents 208 --num-blocks 8 --eval-limit 12 --eval-seed 1802 "
        f"--seed {seed} --output {output}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="locator root holding scale*/s*/results.jsonl")
    parser.add_argument("--steps", type=int, required=True, help="completed step budget of the rung to record")
    parser.add_argument("--phase", required=True, help="AUTORESEARCH phase label, e.g. scale_locator_rung8")
    parser.add_argument("--seed", type=int, default=902)
    parser.add_argument("--learning-rate", default="4e-5")
    parser.add_argument("--ledger", type=Path, default=Path("results/autoresearch/d2a/steering/state.jsonl"))
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    written = 0
    for results_path in sorted(args.root.glob("scale*/s*/results.jsonl")):
        rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
        matched: dict[int, float] = {}
        control: dict[int, float] = {}
        frozen_gate = None
        seen_steps = False
        for row in rows:
            if row["split"] != "test" or not row["task_id"].startswith("niah_"):
                continue
            length = int(row["task_id"].split("_", 1)[1])
            metrics = row["metrics"]
            if row["adapter"] == "steering" and "accuracy_ctxswap" in metrics:
                if int(row.get("metadata", {}).get("steps", -1)) != args.steps:
                    continue
                seen_steps = True
                matched[length] = float(metrics["accuracy"])
                control[length] = float(metrics["accuracy_ctxswap"])
            elif row["adapter"] == "frozen_interpreter" and length == GATE_LENGTH:
                frozen_gate = float(metrics["accuracy"])
        if not seen_steps:
            continue

        scale = results_path.parent.parent.name.removeprefix("scale")
        history = json.loads((results_path.parent / "history.json").read_text())
        entries = [entry for entry in history.get("steering", []) if entry["step"] == args.steps]
        loss = float(entries[-1]["loss"]) if entries else float("nan")

        have_all = all(length in matched for length in (GATE_LENGTH, *HARD_LENGTHS))
        if not have_all:
            # The launcher writes history.json before results.jsonl finishes, so a
            # recorder run that races the writer sees a partial bin set. That is a read
            # race, not an invalid trajectory: refuse to record rather than libel the
            # trial as numerically invalid. Re-run once the writer has finished.
            missing = [length for length in (GATE_LENGTH, *HARD_LENGTHS) if length not in matched]
            raise SystemExit(
                f"{results_path} is missing bins {missing} at step {args.steps}; "
                "the run is still writing its results. Re-run this recorder when it completes."
            )
        finite = math.isfinite(loss) and all(
            math.isfinite(value) for value in (*matched.values(), *control.values())
        )
        deltas = {length: matched[length] - control[length] for length in matched} if have_all else {}
        auc = controlled_log_auc(deltas, HARD_LENGTHS) if have_all and finite else None
        gate_delta = deltas.get(GATE_LENGTH)
        gate_passed = (
            None
            if not have_all or frozen_gate is None
            else bool(matched[GATE_LENGTH] > frozen_gate and math.isclose(control[GATE_LENGTH], 0.0, abs_tol=1e-9))
        )
        record = {
            "phase": args.phase,
            "setting": "d2a",
            "codec": "steering",
            "seed": args.seed,
            "free_hparams": {
                "scale": float(scale),
                "learning_rate": float(args.learning_rate),
                "steps": args.steps,
                "warmup_steps": 960,
            },
            "command": command_for(scale, args.steps, args.seed, args.learning_rate, results_path.parent),
            "artifact_root": str(results_path.parent),
            "status": "complete" if (have_all and finite) else "numerically_invalid",
            "selection_metric": auc,
            "control_metric": max(control.values()) if control else None,
            "helpfulness_metric": matched.get(GATE_LENGTH),
            "gate_accuracy": matched.get(GATE_LENGTH),
            "gate_delta": gate_delta,
            "frozen_gate_accuracy": frozen_gate,
            "train_loss": loss if math.isfinite(loss) else None,
            "gate_passed": gate_passed,
            "converged": bool(have_all and finite),
            "matched_by_length": {str(k): matched[k] for k in sorted(matched)},
            "control_by_length": {str(k): control[k] for k in sorted(control)},
            "notes": args.notes
            or (
                "Controlled hard-length AUC = normalized trapezoidal mean of "
                "matched-minus-context-swap over log2 length across 1024..32768; "
                "512 is the shortest in-distribution helpfulness/control gate. "
                "Selection never uses language-model loss."
            ),
        }
        with args.ledger.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        written += 1
        print(json.dumps({k: record[k] for k in ("free_hparams", "status", "selection_metric", "gate_accuracy", "gate_passed", "train_loss")}, sort_keys=True))
    if written == 0:
        raise SystemExit(f"no completed {args.steps}-step trial found under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
