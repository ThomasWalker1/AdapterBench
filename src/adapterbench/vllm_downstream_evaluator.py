"""DownstreamEvaluator that scores generated LoRA adapters via vLLM, out-of-process under
``upstream/text-to-lora/.venv`` (same subprocess-bridge pattern as
``text_to_lora_backend.ReleasedTextToLoRABackend``, for the same reason: vLLM is not, and
should never be, a dependency of ``adapterbench`` itself).

Use this over ``HFDownstreamEvaluator`` whenever every adapter under test is a plain LoRA.
It exists because of a real, confirmed gap (see PROJECT_PLAN.md's Phase 5.5 follow-up):
scoring the *exact same* prompts, already-generated LoRA adapters, and scoring functions as
``HFDownstreamEvaluator``, with only the generation backend swapped from ``transformers`` to
``vllm==0.5.4`` (upstream's own pinned eval backend), took Mistral-7B-Instruct-v0.2's LoRA
numbers from actively disagreeing with the paper's published results to matching them
closely on 4/5 tasks. It cannot serve FourierFT/IA3/LoKr or any other non-LoRA adapter
format, so ``HFDownstreamEvaluator`` remains required for those.

A single vLLM engine load is shared across every family passed to one ``iter_evaluate``/
``iter_evaluate_frozen`` call (reloading per family would dominate wall time at this model
scale) via a persistent subprocess that streams one result back per family as it
completes - see ``scripts/vllm_generate.py``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import subprocess
import time
from typing import Iterable, Iterator, Mapping

from .contracts import AdapterArtifact, DownstreamEvaluator, EvaluationResult, TaskExample
from .hf_downstream_evaluator import (
    build_prefill_by_family,
    get_binary_accuracy,
    get_choice_accuracy,
    get_gsm8k_accuracy,
    load_faithful_tokenizer,
    render_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_T2L_DIR = REPO_ROOT / "upstream" / "text-to-lora"
UPSTREAM_T2L_PYTHON = UPSTREAM_T2L_DIR / ".venv" / "bin" / "python"
GENERATE_SCRIPT = REPO_ROOT / "scripts" / "vllm_generate.py"
RESULT_PREFIX = "##ADAPTERBENCH_RESULT## "

_SCORERS = {"gsm8k": get_gsm8k_accuracy, "boolq": get_binary_accuracy}


class VLLMDownstreamEvaluator(DownstreamEvaluator):
    def __init__(
        self,
        model_id: str,
        chat_template_path: str | Path,
        trial_id: str,
        device: str = "cuda:0",
        max_new_tokens: int = 512,
        use_icl: bool = False,
        gpu_memory_utilization: float = 0.85,
        python: str | Path = UPSTREAM_T2L_PYTHON,
    ):
        self.model_id = model_id
        self.trial_id = trial_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.python = Path(python)
        if not self.python.exists():
            raise FileNotFoundError(
                f"upstream text-to-lora interpreter not found at {self.python}; "
                "see SETUP.md to provision the upstream/text-to-lora uv venv"
            )
        # Only used to render prompts (chat template) - never loads model weights or
        # touches the GPU, so this class is cheap to construct in the main process.
        self.tokenizer = load_faithful_tokenizer(model_id, chat_template_path)
        self._prefill_by_family = build_prefill_by_family(use_icl)

    @staticmethod
    def _group_by_family(examples: Iterable[TaskExample]) -> dict[str, list[TaskExample]]:
        by_family: dict[str, list[TaskExample]] = defaultdict(list)
        for example in examples:
            by_family[example.family].append(example)
        return by_family

    def _prompts_for(self, family: str, examples: list[TaskExample]) -> list[str]:
        prefill = self._prefill_by_family.get(family, "")
        return [render_prompt(self.tokenizer, ex.input_text, prefill) for ex in examples]

    @staticmethod
    def _score(family: str, generated: str, target_text: str) -> bool:
        scorer = _SCORERS.get(family, get_choice_accuracy)
        return scorer(generated, target_text)

    def _stream_groups(self, groups: list[dict]) -> Iterator[tuple[str, list[str], float]]:
        """Run every group through one vLLM engine load, yielding (name, generations,
        seconds) as each group finishes - see ``scripts/vllm_generate.py``'s module
        docstring for why stdout lines are sentinel-prefixed rather than plain JSON."""
        manifest_path = Path(f"/tmp/adapterbench_vllm_manifest_{id(self)}_{time.time_ns()}.json")
        manifest_path.write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                    "max_new_tokens": self.max_new_tokens,
                    "groups": groups,
                }
            )
        )
        try:
            proc = subprocess.Popen(
                [str(self.python), str(GENERATE_SCRIPT.resolve()), "--manifest", str(manifest_path)],
                cwd=UPSTREAM_T2L_DIR,
                stdout=subprocess.PIPE,
                text=True,
            )
            seen = 0
            for line in proc.stdout:
                if not line.startswith(RESULT_PREFIX):
                    continue
                record = json.loads(line[len(RESULT_PREFIX) :])
                seen += 1
                yield record["name"], record["generations"], record["seconds"]
            returncode = proc.wait()
            if returncode != 0:
                raise RuntimeError(f"scripts/vllm_generate.py exited with code {returncode}")
            if seen != len(groups):
                raise RuntimeError(f"expected {len(groups)} group result(s), got {seen}")
        finally:
            manifest_path.unlink(missing_ok=True)

    def _evaluate_groups(
        self,
        split: str,
        family_examples: dict[str, list[TaskExample]],
        groups: list[dict],
        artifacts: Mapping[str, AdapterArtifact],
    ) -> Iterator[EvaluationResult]:
        for family, generations, seconds in self._stream_groups(groups):
            examples = family_examples[family]
            artifact = artifacts.get(family)
            correct = sum(
                1 for ex, gen in zip(examples, generations) if self._score(family, gen, ex.target_text)
            )
            metric_name = "exact_match" if family == "gsm8k" else "accuracy"
            adapter = artifact.adapter if artifact is not None else "frozen_interpreter"
            yield EvaluationResult(
                trial_id=self.trial_id,
                task_id=family,
                split=split,
                adapter=adapter,
                metrics={metric_name: correct / len(examples), "n_examples": float(len(examples))},
                generated_parameter_count=artifact.generated_parameter_count if artifact is not None else 0,
                generation_seconds=artifact.generation_seconds if artifact is not None else 0.0,
                inference_seconds=seconds,
                metadata={
                    "model_id": self.model_id,
                    "backend": "vllm",
                    "condition_variant": examples[0].metadata.get("condition_variant") if examples else None,
                    **({"artifact_metadata": dict(artifact.metadata)} if artifact is not None else {}),
                },
            )

    def iter_evaluate(
        self, artifacts: Mapping[str, AdapterArtifact], examples: Iterable[TaskExample], split: str
    ) -> Iterator[EvaluationResult]:
        family_examples = self._group_by_family(examples)
        groups = [
            {
                "name": family,
                "adapter_dir": str(artifacts[family].path) if family in artifacts else None,
                "prompts": self._prompts_for(family, family_ex),
            }
            for family, family_ex in family_examples.items()
        ]
        yield from self._evaluate_groups(split, family_examples, groups, artifacts)

    def iter_evaluate_frozen(self, examples: Iterable[TaskExample], split: str) -> Iterator[EvaluationResult]:
        family_examples = self._group_by_family(examples)
        groups = [
            {"name": family, "adapter_dir": None, "prompts": self._prompts_for(family, family_ex)}
            for family, family_ex in family_examples.items()
        ]
        yield from self._evaluate_groups(split, family_examples, groups, {})

    def evaluate(
        self, artifacts: Mapping[str, AdapterArtifact], examples: Iterable[TaskExample], split: str
    ) -> list[EvaluationResult]:
        return list(self.iter_evaluate(artifacts, examples, split))

    def evaluate_frozen(self, examples: Iterable[TaskExample], split: str) -> list[EvaluationResult]:
        return list(self.iter_evaluate_frozen(examples, split))
