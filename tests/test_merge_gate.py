"""Unit tests for the correctness merge gate (CPU, no GPU).

Covers the guarantees that keep the panel honest: the path guard forces substrate changes down a
separate path, the shape-identity lint enforces matched capacity, and the gate orchestration is
correctness-only and fails on any red check. The real `make_codec` is used for the capacity number.
"""

from __future__ import annotations

from adapterbench.merge_gate import (
    check_path_guard, check_shape_identity, generated_output_size, run_gate,
)
from adapterbench.schema import AdapterManifest, ParameterBudget


def _manifest(r=8, budget=ParameterBudget(reference_dim=2048, max_output_size=40000)):
    return AdapterManifest(
        schema_version=1, name=f"lora_r{r}", family="lora", implementation="peft",
        output_structure="LoRA A/B", target_modules=["q_proj", "v_proj"],
        hyperparameters={"r": r, "lora_alpha": 16}, compatible_objectives=["downstream"],
        supports_batched_generation=True, parameter_budget=budget,
    )


# ── path guard ───────────────────────────────────────────────────────────────────────────────
def test_path_guard_allows_codec_unit_paths():
    ok = check_path_guard([
        "src/adapterbench/t2p/codecs.py",
        "configs/adapters/newshape.yaml",
        "src/adapterbench/schema.py",
        "results/leaderboard/records.jsonl",
    ])
    assert ok.passed


def test_path_guard_rejects_substrate_changes():
    bad = check_path_guard(["src/adapterbench/t2p/codecs.py",
                            "src/adapterbench/t2p/sft_trainer.py"])  # trainer = substrate
    assert not bad.passed and "sft_trainer" in bad.detail


def test_path_guard_rejects_evaluator_and_conditioner():
    for substrate in ("src/adapterbench/t2p/live_evaluator.py",
                      "src/adapterbench/t2p/hypernetwork.py",
                      "src/adapterbench/cli/live_sft.py"):
        assert not check_path_guard([substrate]).passed


# ── shape-identity lint ──────────────────────────────────────────────────────────────────────
def test_shape_identity_passes_within_budget():
    # r=8 @ 2048 -> 8*(2048+2048)=32768 <= 40000
    assert generated_output_size("lora", {"r": 8}, 2048) == 32768
    assert check_shape_identity(_manifest(r=8)).passed


def test_shape_identity_rejects_over_budget():
    # r=16 @ 2048 -> 65536 > 40000 : exceeds the matched-capacity band
    check = check_shape_identity(_manifest(r=16))
    assert not check.passed and "65536" in check.detail


def test_shape_identity_requires_declared_budget():
    assert not check_shape_identity(_manifest(budget=None)).passed


# ── gate orchestration (correctness only, fails on any red) ──────────────────────────────────
def test_run_gate_all_green():
    report = run_gate(_manifest(), ["configs/adapters/lora_r8.yaml"],
                      runner=lambda argv: (0, "ok"), run_gpu_smoke=True)
    assert report.passed
    assert {c.name for c in report.checks} >= {"path-guard", "shape-identity", "validate", "catalog",
                                               "pytest", "gpu-smoke"}


def test_run_gate_fails_on_red_command():
    def runner(argv):
        return (1, "boom") if "pytest" in argv else (0, "ok")
    report = run_gate(_manifest(), ["configs/adapters/lora_r8.yaml"], runner=runner)
    assert not report.passed
    assert any(c.name == "pytest" and not c.passed for c in report.checks)


def test_run_gate_fails_on_substrate_path_even_if_commands_green():
    report = run_gate(_manifest(), ["src/adapterbench/t2p/sft_trainer.py"],
                      runner=lambda argv: (0, "ok"))
    assert not report.passed  # correctness commands green, but path guard is red
    assert any(c.name == "path-guard" and not c.passed for c in report.checks)
