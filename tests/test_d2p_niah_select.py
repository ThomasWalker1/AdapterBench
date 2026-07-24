from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "d2p_niah_select.py"
SPEC = importlib.util.spec_from_file_location("d2p_niah_select", SCRIPT)
assert SPEC and SPEC.loader
selector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


def _row(adapter: str, length: int, accuracy: float, control: float | None = None, steps: int = 8000) -> dict:
    metrics = {"accuracy": accuracy}
    if control is not None:
        metrics["accuracy_ctxswap"] = control
    return {
        "adapter": adapter,
        "split": "test",
        "task_id": f"niah_{length}",
        "metrics": metrics,
        "metadata": {"steps": steps} if adapter != "frozen_interpreter" else {},
    }


def test_controlled_auc_uses_trapezoids_on_log_length_axis():
    points = {512: 0.0, 1024: 0.25, 2048: 0.5, 4096: 0.75, 8192: 1.0}
    assert selector.controlled_auc(points) == pytest.approx(0.5)


def test_load_candidate_uses_context_swap_and_checkpoint_step(tmp_path):
    results = tmp_path / "scale10" / "s902" / "results.jsonl"
    results.parent.mkdir(parents=True)
    rows = [_row("frozen_interpreter", 256, 0.0), _row("lora", 256, 1.0, 0.0)]
    rows += [_row("lora", length, value, 0.0) for length, value in {
        512: 0.0, 1024: 0.25, 2048: 0.5, 4096: 0.75, 8192: 1.0,
    }.items()]
    results.write_text("".join(json.dumps(row) + "\n" for row in rows))
    candidate = selector.load_candidate(results, "lora", selector.DEFAULT_HARD_LENGTHS)
    assert candidate.scale == "10"
    assert candidate.steps == 8000
    assert candidate.gate_delta == 1.0
    assert candidate.auc == pytest.approx(0.5)


def test_main_can_retain_all_candidates_when_an_early_rung_has_no_gate_signal(tmp_path, monkeypatch, capsys):
    for scale in ("1", "10"):
        results = tmp_path / f"scale{scale}" / "s902" / "results.jsonl"
        results.parent.mkdir(parents=True)
        rows = [_row("frozen_interpreter", 256, 0.0), _row("lora", 256, 0.0, 0.0)]
        rows += [_row("lora", length, 0.0, 0.0) for length in selector.DEFAULT_HARD_LENGTHS]
        results.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(sys, "argv", [
        "d2p_niah_select.py", "--root", str(tmp_path), "--adapter", "lora",
        "--retain-all-if-no-gate", "--format", "scales",
    ])
    assert selector.main() == 0
    assert capsys.readouterr().out.strip() == "1 10"


def test_load_candidate_supports_a_nondefault_in_distribution_gate(tmp_path):
    results = tmp_path / "scale10" / "s902" / "results.jsonl"
    results.parent.mkdir(parents=True)
    rows = [_row("frozen_interpreter", 512, 0.0), _row("lora", 512, 1.0, 0.0)]
    rows += [_row("lora", length, 0.5, 0.0) for length in (1024, 2048)]
    results.write_text("".join(json.dumps(row) + "\n" for row in rows))
    candidate = selector.load_candidate(results, "lora", (1024, 2048), gate_length=512)
    assert candidate.gate_delta == 1.0
    assert candidate.frozen_256 == 0.0
