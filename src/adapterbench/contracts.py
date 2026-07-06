"""Interfaces allowing paper-specific hypernetworks to share one evaluator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class TaskExample:
    task_id: str
    condition: str
    input_text: str
    target_text: str
    family: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterArtifact:
    task_id: str
    adapter: str
    path: Path
    format: str
    generated_parameter_count: int
    generation_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    trial_id: str
    task_id: str
    split: str
    adapter: str
    metrics: Mapping[str, float]
    generated_parameter_count: int
    generation_seconds: float
    inference_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class HypernetworkBackend(ABC):
    """Paper-specific compiler: condition text/context -> portable adapter artifact."""

    @abstractmethod
    def generate(self, conditions: Mapping[str, str], output_dir: Path) -> Mapping[str, AdapterArtifact]: ...


class DownstreamEvaluator(ABC):
    """Frozen interpreter evaluation, deliberately independent of generator architecture."""

    @abstractmethod
    def evaluate(
        self, artifacts: Mapping[str, AdapterArtifact], examples: Iterable[TaskExample], split: str
    ) -> list[EvaluationResult]: ...

