from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from adapterbench.results import ResultValidationError, check_rendered_documents, load_records, validate_record


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_records_validate_and_match_verified_headlines():
    records = load_records(ROOT / "canonical_results")
    assert {record["setting"] for record in records} == {"T2L", "D2L"}
    by_identity = {(record["setting"], record["codec"]): record for record in records}
    assert by_identity[("T2L", "lora_r8")]["headline"]["value"] == pytest.approx(-0.7232315675488087)
    assert by_identity[("T2L", "lora_r8")]["headline"]["variation"] == pytest.approx(0.16191601739129335)
    assert by_identity[("D2L", "lora_r8")]["headline"]["value"] == pytest.approx(0.8875)
    assert by_identity[("D2L", "lora_r8")]["headline"]["control"] == 0.0
    assert by_identity[("D2L", "lora_r8")]["summary"]["crossover_length"] == 4096
    assert by_identity[("D2L", "ia3")]["headline"]["value"] == 1.0
    assert by_identity[("D2L", "ia3")]["headline"]["control"] == 0.0


def test_canonical_validation_rejects_a_non_difference_headline():
    record = deepcopy(load_records(ROOT / "canonical_results")[0])
    record["headline"]["value"] = 123.0
    with pytest.raises(ResultValidationError, match="matched minus control"):
        validate_record(record)


def test_rendered_result_tables_are_in_lockstep_with_canonical_records():
    assert check_rendered_documents(load_records(ROOT / "canonical_results"), ROOT) == []
