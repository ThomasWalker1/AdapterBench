"""Versioned manifests defining comparable hypernetwork experiments."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceRef(StrictModel):
    name: str
    uri: str
    revision: str | None = None


class ModelRef(StrictModel):
    model_id: str
    revision: str | None = None
    frozen: bool = True
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"


class ConditioningSpec(StrictModel):
    kind: Literal["task_description", "spec_and_pseudoprogram", "document", "task_id"]
    encoder: str
    pooling: str | None = None
    fields: list[str]


class DatasetSpec(StrictModel):
    source: Literal["huggingface", "upstream", "local"]
    dataset_id: str
    revision: str | None = None
    train_split: str | None = None
    validation_split: str | None = None
    test_splits: list[str]
    task_id_field: str
    family_field: str | None = None
    condition_field: str
    input_field: str
    target_field: str
    holdout_unit: Literal["task", "task_family", "document"]


class ObjectiveSpec(StrictModel):
    kind: Literal["downstream", "reconstruction", "hybrid"]
    downstream_weight: float = Field(ge=0)
    reconstruction_weight: float = Field(ge=0)
    oracle_adapter_source: str | None = None

    @model_validator(mode="after")
    def validate_weights(self):
        if self.downstream_weight + self.reconstruction_weight <= 0:
            raise ValueError("at least one objective weight must be positive")
        if self.reconstruction_weight and not self.oracle_adapter_source:
            raise ValueError("reconstruction objectives require oracle_adapter_source")
        return self


class EvaluationSpec(StrictModel):
    primary_metric: str
    metrics: list[str]
    baselines: list[str]
    report_by: list[str]
    generation: dict[str, Any] = Field(default_factory=dict)


class RuntimeSpec(StrictModel):
    launcher: Literal["single", "torchrun", "accelerate"]
    num_processes: int = Field(ge=1)
    mixed_precision: Literal["no", "fp16", "bf16"]
    gradient_checkpointing: bool = False


class SetupManifest(StrictModel):
    schema_version: Literal[1]
    name: str
    description: str
    protocol: Literal["paw", "text_to_lora", "doc_to_lora", "generic_text_to_adapter"]
    availability: Literal["runnable", "artifacts_only", "specification_only"]
    runner: str
    sources: list[SourceRef]
    models: dict[str, ModelRef]
    conditioning: ConditioningSpec
    dataset: DatasetSpec
    objective: ObjectiveSpec
    evaluation: EvaluationSpec
    runtime: RuntimeSpec
    fixed_controls: list[str]
    notes: list[str] = Field(default_factory=list)


class AdapterManifest(StrictModel):
    schema_version: Literal[1]
    name: str
    family: Literal["lora", "fourierft", "lokr", "ia3", "prefix_tuning", "activation_steering"]
    implementation: Literal["peft", "custom"]
    output_structure: str
    target_modules: list[str]
    hyperparameters: dict[str, Any]
    compatible_objectives: list[Literal["downstream", "reconstruction", "hybrid"]]
    supports_batched_generation: bool
    notes: list[str] = Field(default_factory=list)


class TrialManifest(StrictModel):
    schema_version: Literal[1] = 1
    trial_id: str
    setup: SetupManifest
    adapter: AdapterManifest


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain a mapping: {path}")
    return value


def load_setup(path: str | Path) -> SetupManifest:
    return SetupManifest.model_validate(load_yaml(path))


def load_adapter(path: str | Path) -> AdapterManifest:
    return AdapterManifest.model_validate(load_yaml(path))


def make_trial(setup: SetupManifest, adapter: AdapterManifest) -> TrialManifest:
    if setup.objective.kind not in adapter.compatible_objectives:
        raise ValueError(f"{adapter.name} does not support the {setup.objective.kind} objective")
    payload = {"setup": setup.model_dump(mode="json"), "adapter": adapter.model_dump(mode="json")}
    digest = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return TrialManifest(trial_id=f"{setup.name}--{adapter.name}--{digest}", setup=setup, adapter=adapter)

