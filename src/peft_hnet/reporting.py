from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

from .contracts import EvaluationResult


def write_results(results: Iterable[EvaluationResult], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with (output_dir / "results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")
    flat = []
    metric_keys: set[str] = set()
    for row in rows:
        metrics = row.pop("metrics")
        metadata = row.pop("metadata")
        metric_columns = {f"metric.{k}": v for k, v in metrics.items()}
        metric_keys.update(metric_columns)
        flat.append({**row, **metric_columns, "metadata": json.dumps(metadata, default=str)})
    if flat:
        # Different task families report different metric names (e.g. "accuracy" for
        # multiple-choice tasks vs "exact_match" for gsm8k), so fieldnames must be the
        # union across all rows, not just the first row's keys - a fixed-width schema
        # from row 0 raises ValueError as soon as a differently-keyed row appears.
        base_fields = [key for key in flat[0] if not key.startswith("metric.") and key != "metadata"]
        fieldnames = base_fields + sorted(metric_keys) + ["metadata"]
        with (output_dir / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(flat)

