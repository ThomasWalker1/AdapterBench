#!/usr/bin/env python3
"""Aggregate canonical T2A per-seed CE and accuracy files without training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def aggregate(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0


def read_aggregate(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return next(row for row in rows if row["task_id"] == "__aggregate__")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/repro/t2a_base_diag/gemma2b_stripdef_hyper"))
    parser.add_argument("--seeds", default="777,2,3")
    args = parser.parse_args()
    ce_rows, acc_rows = [], []
    for seed in args.seeds.split(","):
        directory = args.root / f"s{seed}"
        ce_rows.append(read_aggregate(directory / "heldout_sni_ce_full21.jsonl"))
        acc_rows.append(read_aggregate(directory / "heldout_sni_acc.jsonl"))
    ce, ce_std = aggregate([row["matched_minus_static"] for row in ce_rows])
    acc, acc_std = aggregate([row["matched_minus_static"] for row in acc_rows])
    wins = sum(round(row["frac_tasks_matched_lt_static"] * row["n_tasks"]) for row in ce_rows)
    print(f"T2A matched-static CE: {ce:+.3f} ± {ce_std:.3f} nats; wins {wins}/{sum(row['n_tasks'] for row in ce_rows)}")
    print(f"T2A matched-static accuracy: {acc:+.4f} ± {acc_std:.4f}")


if __name__ == "__main__":
    main()
