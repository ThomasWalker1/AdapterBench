from pathlib import Path

import pytest

from peft_hnet.catalog import build_matrix, load_catalog
from peft_hnet.peft_support import make_peft_config


ROOT = Path(__file__).resolve().parents[1]


def test_catalog_and_trial_ids_are_complete_and_stable():
    setups, adapters = load_catalog(ROOT / "configs")
    assert set(setups) == {
        "text_to_peft_gemma2b_reconstruction",
        "text_to_peft_gemma2b_sft",
        "text_to_peft_mistral7b_reconstruction_pilot",
        "text_to_peft_sft_pilot",
    }
    assert len(adapters) == 8
    # activation_steering has no weight-space delta (dense_delta raises) and only
    # declares compatible_objectives=[downstream], so it deliberately can't build a
    # trial against a reconstruction-objective setup — restrict to the adapters that
    # can, matching how `peft-hnet validate`/build_matrix actually behaves.
    reconstruction_compatible = [
        adapter for adapter in adapters.values() if "reconstruction" in adapter.compatible_objectives
    ]
    trials = build_matrix(setups["text_to_peft_gemma2b_reconstruction"], reconstruction_compatible)
    assert len(trials) == 7
    assert len({trial.trial_id for trial in trials}) == len(trials)
    repeated = build_matrix(setups["text_to_peft_gemma2b_reconstruction"], reconstruction_compatible)
    assert [trial.trial_id for trial in repeated] == [trial.trial_id for trial in trials]


def test_activation_steering_is_incompatible_with_reconstruction_objective():
    setups, adapters = load_catalog(ROOT / "configs")
    with pytest.raises(ValueError, match="does not support the reconstruction objective"):
        build_matrix(setups["text_to_peft_gemma2b_reconstruction"], [adapters["activation_steering"]])


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
