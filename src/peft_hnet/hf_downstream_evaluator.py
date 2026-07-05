"""DownstreamEvaluator that scores generated adapters with plain transformers+peft.

Deliberately avoids vLLM: it is not installed in this environment, and non-LoRA
representations (FourierFT/IA3/LoKr) this benchmark ultimately compares are not
vLLM-servable anyway, so a custom evaluator is required regardless.

Replicates upstream Text-to-LoRA's tokenizer setup (padding side, pad token, chat
template, truncation side - see ``hyper_llm_modulator.utils.model_loading.get_tokenizer``)
so generated adapters are scored under the same input formatting they were evaluated
against upstream. The task condition text is never shown to the interpreter - only
used to generate the adapter - matching upstream's default ``system_message=""`` eval mode.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import re
import time
from typing import Iterable, Mapping

import torch

from .contracts import AdapterArtifact, DownstreamEvaluator, EvaluationResult, TaskExample

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def load_faithful_tokenizer(model_id: str, chat_template_path: str | Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    chat_template = Path(chat_template_path).read_text()
    chat_template = chat_template.replace("    ", "").replace("\n", "")
    tokenizer.chat_template = chat_template
    tokenizer.add_eos_token = False
    tokenizer.truncation_side = "left"
    return tokenizer


class HFDownstreamEvaluator(DownstreamEvaluator):
    def __init__(
        self,
        model_id: str,
        chat_template_path: str | Path,
        trial_id: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 256,
    ):
        from transformers import AutoModelForCausalLM

        self.model_id = model_id
        self.trial_id = trial_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = load_faithful_tokenizer(model_id, chat_template_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        self.model.eval()
        self.peft_model = None
        self._loaded_adapter_names: set[str] = set()

    def _ensure_adapter_loaded(self, name: str, path: Path) -> None:
        from peft import PeftModel

        if name in self._loaded_adapter_names:
            return
        if self.peft_model is None:
            self.peft_model = PeftModel.from_pretrained(self.model, str(path), adapter_name=name)
        else:
            self.peft_model.load_adapter(str(path), adapter_name=name)
        self._loaded_adapter_names.add(name)

    @contextmanager
    def _active(self, adapter_name: str | None):
        if adapter_name is None:
            if self.peft_model is None:
                yield self.model
            else:
                with self.peft_model.disable_adapter():
                    yield self.peft_model
        else:
            self.peft_model.set_adapter(adapter_name)
            yield self.peft_model

    def _prompt(self, input_text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _loglikelihood(self, model, prompt: str, continuation: str) -> float:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        continuation_text = continuation if continuation.startswith(" ") else " " + continuation
        continuation_ids = self.tokenizer(continuation_text, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([prompt_ids + continuation_ids], device=self.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
        target_ids = input_ids[0, 1:]
        n_continuation = len(continuation_ids)
        token_log_probs = log_probs[-n_continuation:].gather(-1, target_ids[-n_continuation:].unsqueeze(-1))
        return float(token_log_probs.sum()) / max(n_continuation, 1)

    def _score_multiple_choice(self, model, example: TaskExample) -> bool:
        choices = example.metadata["choices"]
        answer_index = example.metadata["answer_index"]
        prompt = self._prompt(example.input_text)
        scores = [self._loglikelihood(model, prompt, choice) for choice in choices]
        predicted = max(range(len(scores)), key=lambda i: scores[i])
        return predicted == answer_index

    def _score_gsm8k(self, model, example: TaskExample) -> bool:
        prompt = self._prompt(example.input_text)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = self.tokenizer.decode(output_ids[0, input_ids["input_ids"].shape[1] :], skip_special_tokens=True)
        matches = _NUMBER_RE.findall(generated)
        if not matches:
            return False
        predicted = matches[-1].replace(",", "")
        try:
            return abs(float(predicted) - float(example.target_text)) < 1e-4
        except ValueError:
            return False

    def _score(self, model, example: TaskExample) -> bool:
        if example.family == "gsm8k":
            return self._score_gsm8k(model, example)
        return self._score_multiple_choice(model, example)

    def _evaluate_group(
        self,
        family: str,
        examples: list[TaskExample],
        split: str,
        artifact: AdapterArtifact | None,
    ) -> EvaluationResult:
        adapter_name = None
        if artifact is not None:
            adapter_name = f"{family}__{artifact.representation}"
            self._ensure_adapter_loaded(adapter_name, artifact.path)

        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self._active(adapter_name) as model:
            correct = sum(1 for example in examples if self._score(model, example))
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        inference_seconds = time.perf_counter() - started

        metric_name = "exact_match" if family == "gsm8k" else "accuracy"
        representation = artifact.representation if artifact is not None else "frozen_interpreter"
        return EvaluationResult(
            trial_id=self.trial_id,
            task_id=family,
            split=split,
            representation=representation,
            metrics={metric_name: correct / len(examples), "n_examples": float(len(examples))},
            generated_parameter_count=artifact.generated_parameter_count if artifact is not None else 0,
            generation_seconds=artifact.generation_seconds if artifact is not None else 0.0,
            inference_seconds=inference_seconds,
            metadata={
                "model_id": self.model_id,
                "peak_gpu_memory_bytes_inference": (
                    torch.cuda.max_memory_allocated(self.device) if self.device.startswith("cuda") else None
                ),
                "condition_variant": examples[0].metadata.get("condition_variant") if examples else None,
                **({"artifact_metadata": dict(artifact.metadata)} if artifact is not None else {}),
            },
        )

    @staticmethod
    def _group_by_family(examples: Iterable[TaskExample]) -> dict[str, list[TaskExample]]:
        by_family: dict[str, list[TaskExample]] = defaultdict(list)
        for example in examples:
            by_family[example.family].append(example)
        return by_family

    def iter_evaluate(
        self, artifacts: Mapping[str, AdapterArtifact], examples: Iterable[TaskExample], split: str
    ):
        """Yield one EvaluationResult per task family as it completes.

        Prefer this over ``evaluate()`` for long runs: it lets a caller print
        progress and persist partial results incrementally, so a long
        multi-task/multi-example run doesn't lose everything if interrupted.
        """
        for family, family_examples in self._group_by_family(examples).items():
            yield self._evaluate_group(family, family_examples, split, artifacts.get(family))

    def iter_evaluate_frozen(self, examples: Iterable[TaskExample], split: str):
        for family, family_examples in self._group_by_family(examples).items():
            yield self._evaluate_group(family, family_examples, split, None)

    def evaluate(
        self, artifacts: Mapping[str, AdapterArtifact], examples: Iterable[TaskExample], split: str
    ) -> list[EvaluationResult]:
        return list(self.iter_evaluate(artifacts, examples, split))

    def evaluate_frozen(self, examples: Iterable[TaskExample], split: str) -> list[EvaluationResult]:
        return list(self.iter_evaluate_frozen(examples, split))
