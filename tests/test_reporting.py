import csv
import json
from pathlib import Path

from adapterbench.contracts import EvaluationResult
from adapterbench.reporting import write_results


def _result(task_id: str, adapter: str, metrics: dict) -> EvaluationResult:
    return EvaluationResult(
        trial_id="trial-1",
        task_id=task_id,
        split="test",
        adapter=adapter,
        metrics=metrics,
        generated_parameter_count=100,
        generation_seconds=0.1,
        inference_seconds=1.0,
        metadata={"note": "x"},
    )


def test_write_results_handles_heterogeneous_metric_keys_across_rows(tmp_path):
    # Multiple-choice tasks report "accuracy"; gsm8k reports "exact_match". A fixed
    # CSV schema derived from only the first row previously raised ValueError once a
    # differently-keyed row appeared (regression: reporting.write_results).
    results = [
        _result("arc_easy", "lora", {"accuracy": 0.7, "n_examples": 100.0}),
        _result("gsm8k", "lora", {"exact_match": 0.13, "n_examples": 100.0}),
    ]

    write_results(results, tmp_path)

    lines = (tmp_path / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["metrics"] == {"accuracy": 0.7, "n_examples": 100.0}
    assert json.loads(lines[1])["metrics"] == {"exact_match": 0.13, "n_examples": 100.0}

    with (tmp_path / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {"metric.accuracy", "metric.exact_match", "metric.n_examples"} <= set(rows[0])
    assert rows[0]["metric.accuracy"] == "0.7"
    assert rows[0]["metric.exact_match"] == ""
    assert rows[1]["metric.exact_match"] == "0.13"
    assert rows[1]["metric.accuracy"] == ""


def test_write_results_is_idempotent_when_called_incrementally(tmp_path):
    first = [_result("arc_easy", "lora", {"accuracy": 0.7, "n_examples": 100.0})]
    write_results(first, tmp_path)
    second = first + [_result("gsm8k", "lora", {"exact_match": 0.13, "n_examples": 100.0})]
    write_results(second, tmp_path)

    lines = (tmp_path / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
