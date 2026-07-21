"""Minimal, self-contained data types for the released-checkpoint prompt ablation.

Vendored deliberately so this negative-results verification does NOT import the active
``adapterbench`` package: the whole point of ``scripts/negative_results/`` is that it is
evidence *about* settings we rejected, not a runnable benchmark setting. Keeping it
decoupled means the benchmark code can be refactored or the setting deleted without
touching this record, and nobody can mistake it for a registered codec/setting.

The three fields here mirror the shapes the recovered upstream-bridge modules expect
(``task_examples.py``, ``hf_downstream_evaluator.py``), copied from the pre-removal
``adapterbench.contracts`` at git ``a8867de^``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


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


class DownstreamEvaluator:  # marker base; kept for parity with the recovered evaluator
    pass
