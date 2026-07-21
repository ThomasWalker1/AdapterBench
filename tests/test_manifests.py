from pathlib import Path

import pytest

from adapterbench.catalog import build_matrix, load_catalog
from adapterbench.peft_support import make_peft_config
from adapterbench.schema import ObjectiveSpec, make_trial


ROOT = Path(__file__).resolve().parents[1]


def test_catalog_and_trial_ids_are_complete_and_stable():
    setups, adapters = load_catalog(ROOT / "configs")
    assert set(setups) == {
        "document_to_peft_qwen06b_niah",
        "text_to_peft_gemma2b_sft",
    }
    assert {setup.conditioning.kind for setup in setups.values()} == {
        "document",
        "task_description",
    }
    assert set(adapters) == {"ia3", "lora_r8"}
    # Every registered adapter supports the downstream objective, so a downstream setup
    # builds a trial against all of them - matching how `adapterbench validate`/
    # build_matrix actually behaves.
    trials = build_matrix(setups["text_to_peft_gemma2b_sft"], list(adapters.values()))
    assert len(trials) == 2
    assert len({trial.trial_id for trial in trials}) == len(trials)
    repeated = build_matrix(setups["text_to_peft_gemma2b_sft"], list(adapters.values()))
    assert [trial.trial_id for trial in repeated] == [trial.trial_id for trial in trials]

    document_trials = build_matrix(
        setups["document_to_peft_qwen06b_niah"], list(adapters.values())
    )
    assert len(document_trials) == 2


def test_make_trial_rejects_an_objective_the_adapter_does_not_support():
    setups, adapters = load_catalog(ROOT / "configs")
    # An adapter that declares compatible_objectives=[downstream] only must not build a
    # trial against a reconstruction-objective setup. No reconstruction setup and no
    # downstream-only adapter ship in the LoRA-only catalog, so synthesize both from real
    # manifests to exercise the guard (this is exactly how a newly added codec that
    # restricts its objectives would be gated).
    reconstruction_setup = setups["text_to_peft_gemma2b_sft"].model_copy(
        update={
            "objective": ObjectiveSpec(
                kind="reconstruction",
                downstream_weight=0.0,
                reconstruction_weight=1.0,
                oracle_adapter_source="synthetic",
            )
        }
    )
    downstream_only = next(iter(adapters.values())).model_copy(update={"compatible_objectives": ["downstream"]})
    with pytest.raises(ValueError, match="does not support the reconstruction objective"):
        make_trial(reconstruction_setup, downstream_only)


def test_all_upstream_peft_manifests_build_configs():
    _, adapters = load_catalog(ROOT / "configs")
    for adapter in adapters.values():
        if adapter.implementation == "peft":
            config = make_peft_config(adapter)
            assert config.peft_type is not None


def test_custom_adapter_is_not_silently_mapped_to_upstream_peft():
    _, adapters = load_catalog(ROOT / "configs")
    # No custom-implementation adapter ships in the LoRA-only catalog (LoRA is
    # implementation="peft"), so synthesize one to exercise the guard - a newly added
    # codec with a hand-rolled implementation must not be silently routed to HF PEFT.
    custom = next(iter(adapters.values())).model_copy(update={"implementation": "custom"})
    with pytest.raises(ValueError, match="custom implementation"):
        make_peft_config(custom)
