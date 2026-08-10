from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from adapterbench.results import ResultValidationError, check_rendered_documents, load_records, validate_record


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_records_validate_and_match_verified_headlines():
    records = load_records(ROOT / "canonical_results")
    assert {record["setting"] for record in records} == {"T2A", "D2A"}
    by_identity = {(record["setting"], record["codec"]): record for record in records}
    # T2A headlines are ROUGE-L `matched - static*` on the 11 genuinely held-out tasks, re-derived
    # under the corrected protocol. Higher is better, so these are positive where the superseded
    # CE headlines were negative; CE and exact match are now appendix figures under `summary`.
    assert by_identity[("T2A", "lora_r8")]["metric"] == "held_out_rouge_l"
    assert by_identity[("T2A", "lora_r8")]["headline"]["direction"] == "higher_is_better"
    assert by_identity[("T2A", "lora_r8")]["headline"]["value"] == pytest.approx(0.04920358256274665)
    assert by_identity[("T2A", "lora_r8")]["headline"]["variation"] == pytest.approx(0.026636555964900128)
    assert by_identity[("T2A", "ia3")]["headline"]["value"] == pytest.approx(0.12781665374081072)
    assert by_identity[("T2A", "ia3")]["headline"]["variation"] == pytest.approx(0.04924238506468373)
    assert by_identity[("T2A", "ia3")]["selection_trail"]["protocol"] == "AUTORESEARCH.md"
    # The static control is selected on its own score, so its configuration is recorded separately
    # and need not equal the hypernetwork's -- for (IA)^3 it is the LR the hypernetwork diverges at.
    assert by_identity[("T2A", "ia3")]["free_hyperparameters"]["static_control"]["learning_rate"] == "8e-4"
    assert by_identity[("T2A", "ia3")]["free_hyperparameters"]["learning_rate"] == "5e-5"
    # steering now has a confirmed T2A row; its search previously stopped at a declaration.
    assert ("T2A", "steering") in by_identity
    assert by_identity[("D2A", "lora_r8")]["headline"]["value"] == pytest.approx(0.55625)
    assert by_identity[("D2A", "lora_r8")]["headline"]["control"] == 0.0
    assert by_identity[("D2A", "lora_r8")]["summary"]["crossover_length"] == 8192
    assert by_identity[("D2A", "ia3")]["headline"]["value"] == pytest.approx(0.7375)
    assert by_identity[("D2A", "ia3")]["headline"]["control"] == 0.0
    assert by_identity[("D2A", "ia3")]["summary"]["crossover_length"] == 32768


def test_canonical_validation_rejects_a_non_difference_headline():
    record = deepcopy(load_records(ROOT / "canonical_results")[0])
    record["headline"]["value"] = 123.0
    with pytest.raises(ResultValidationError, match="matched minus control"):
        validate_record(record)


def test_rendered_result_tables_are_in_lockstep_with_canonical_records():
    assert check_rendered_documents(load_records(ROOT / "canonical_results"), ROOT) == []
