from pathlib import Path

import pytest

from adapterbench.catalog import build_matrix, load_catalog
from adapterbench.peft_support import make_peft_config
from adapterbench.schema import ObjectiveSpec, make_trial


ROOT = Path(__file__).resolve().parents[1]


def test_catalog_and_trial_ids_are_complete_and_stable():
    setups, adapters = load_catalog(ROOT / "configs")
    assert set(setups) == {
        "text_to_peft_gemma2b_sft",
        "text_to_peft_sft_pilot",
    }
    assert len(adapters) == 7
    # Every registered adapter supports the downstream objective, so a downstream setup
    # builds a trial against all of them - matching how `adapterbench validate`/
    # build_matrix actually behaves.
    trials = build_matrix(setups["text_to_peft_gemma2b_sft"], list(adapters.values()))
    assert len(trials) == 7
    assert len({trial.trial_id for trial in trials}) == len(trials)
    repeated = build_matrix(setups["text_to_peft_gemma2b_sft"], list(adapters.values()))
    assert [trial.trial_id for trial in repeated] == [trial.trial_id for trial in trials]


def test_activation_steering_is_incompatible_with_reconstruction_objective():
    setups, adapters = load_catalog(ROOT / "configs")
    # activation_steering declares compatible_objectives=[downstream] only, so it must
    # not build a trial against a reconstruction-objective setup. No reconstruction setup
    # ships in the catalog, so synthesize one from a real setup to exercise the guard.
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
    with pytest.raises(ValueError, match="does not support the reconstruction objective"):
        make_trial(reconstruction_setup, adapters["activation_steering"])


def test_all_upstream_peft_manifests_build_configs():
    _, adapters = load_catalog(ROOT / "configs")
    for adapter in adapters.values():
        if adapter.implementation == "peft":
            config = make_peft_config(adapter)
            assert config.peft_type is not None


def test_custom_adapter_is_not_silently_mapped_to_upstream_peft():
    _, adapters = load_catalog(ROOT / "configs")
    with pytest.raises(ValueError, match="custom implementation"):
        make_peft_config(adapters["freeze_a_lora_r8"])
