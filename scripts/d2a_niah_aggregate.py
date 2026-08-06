#!/usr/bin/env python3
"""Aggregate completed D2A base-retrieval runs across seeds.

Each seed directory must contain the ``results.jsonl`` and ``history.json`` emitted by
``adapterbench d2a-niah``. The headline is the per-seed matched accuracy minus the
context-swap control; training loss is read only to locate the retrieval transition.

Example:
    .venv/bin/python scripts/d2a_niah_aggregate.py \
        --root results/repro/document_niah_lora --min-seeds 3
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeedResult:
    run: str
    adapter: str
    task_id: str
    matched: float
    control: float
    transition_step: int | None

    @property
    def headline(self) -> float:
        return self.matched - self.control


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, std


def _transition_step(history: dict, adapter: str, task_id: str) -> int | None:
    for record in history.get(adapter, []):
        accuracy = record.get(task_id, {}).get("accuracy")
        if accuracy is not None and accuracy >= 0.5:
            return int(record["step"])
    return None


def load_seed_results(run_dir: Path) -> list[SeedResult]:
    results_path = run_dir / "results.jsonl"
    history_path = run_dir / "history.json"
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    history = json.loads(history_path.read_text()) if history_path.exists() else {}

    seed_results = []
    for row in rows:
        if row["adapter"] == "frozen_interpreter":
            continue
        metrics = row["metrics"]
        if "accuracy" not in metrics or "accuracy_ctxswap" not in metrics:
            continue
        adapter = row["adapter"]
        task_id = row["task_id"]
        seed_results.append(
            SeedResult(
                run=run_dir.name,
                adapter=adapter,
                task_id=task_id,
                matched=float(metrics["accuracy"]),
                control=float(metrics["accuracy_ctxswap"]),
                transition_step=_transition_step(history, adapter, task_id),
            )
        )
    return seed_results


def find_run_dirs(root: Path) -> list[Path]:
    if (root / "results.jsonl").exists():
        return [root]
    return sorted(path.parent for path in root.glob("*/results.jsonl"))


def aggregate(root: Path) -> dict[tuple[str, str], list[SeedResult]]:
    grouped: dict[tuple[str, str], list[SeedResult]] = defaultdict(list)
    for run_dir in find_run_dirs(root):
        for result in load_seed_results(run_dir):
            grouped[(result.adapter, result.task_id)].append(result)
    return dict(grouped)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/repro/document_niah_lora"))
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--train-length", type=int, default=None,
                        help="training context length; if set, report the length-generalization "
                             "crossover (largest eval length with mean matched>=0.5) and its ratio to it")
    args = parser.parse_args()

    grouped = aggregate(args.root)
    if not grouped:
        raise SystemExit(f"no completed adapter results found under {args.root}")

    insufficient = False
    # per-adapter {eval_length: mean matched} for the length-generalization curve/crossover
    curve: dict[str, dict[int, float]] = defaultdict(dict)
    print("adapter  task       seeds  matched       ctxswap       matched-control  transition")
    for (adapter, task_id), results in sorted(grouped.items()):
        matched_mean, matched_std = mean_std([result.matched for result in results])
        control_mean, control_std = mean_std([result.control for result in results])
        headline_mean, headline_std = mean_std([result.headline for result in results])
        transitions = [result.transition_step for result in results if result.transition_step is not None]
        transition = "-"
        if transitions:
            transition_mean, transition_std = mean_std([float(step) for step in transitions])
            transition = f"{transition_mean:.0f}±{transition_std:.0f}"
        print(
            f"{adapter:<8} {task_id:<10} {len(results):>5}  "
            f"{matched_mean:.3f}±{matched_std:.3f}  "
            f"{control_mean:.3f}±{control_std:.3f}  "
            f"{headline_mean:+.3f}±{headline_std:.3f}       {transition}"
        )
        if task_id.startswith("niah_"):
            try:
                curve[adapter][int(task_id.split("_")[1])] = matched_mean
            except (ValueError, IndexError):
                pass
        if len(results) < args.min_seeds:
            insufficient = True
            print(f"  missing {args.min_seeds - len(results)} seed(s): have {', '.join(r.run for r in results)}")

    # Length-generalization summary (invariant #3): the crossover length is the largest
    # eval length whose mean matched retrieval is still >= 0.5; the ratio to the training
    # length is the scalar the leaderboard reports alongside the curve.
    multi_len = {a: c for a, c in curve.items() if len(c) > 1}
    if multi_len:
        print("\nlength-generalization (mean matched by eval length):")
        for adapter, c in sorted(multi_len.items()):
            lengths = sorted(c)
            print(f"  {adapter}: " + "  ".join(f"{L}:{c[L]:.2f}" for L in lengths))
            crossover = max((L for L in lengths if c[L] >= 0.5), default=None)
            if crossover is not None and args.train_length:
                print(f"    crossover(0.5)={crossover} tokens  = {crossover/args.train_length:.0f}x train length "
                      f"({args.train_length})")
            elif crossover is not None:
                print(f"    crossover(0.5)={crossover} tokens (pass --train-length for the ratio)")
            else:
                print("    never crosses 0.5 (no retrieval at any evaluated length)")

    return 1 if insufficient else 0


if __name__ == "__main__":
    raise SystemExit(main())
