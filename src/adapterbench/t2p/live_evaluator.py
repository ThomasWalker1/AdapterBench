"""DownstreamEvaluator counterpart for live-hook-trained hypernetworks.

`hf_downstream_evaluator.py::HFDownstreamEvaluator` activates a generated adapter via
`peft.PeftModel.load_adapter`/`set_adapter` — fundamentally tied to materialized,
PEFT-native adapter directories, which a hook-based adapter (activation steering,
or any adapter trained via `sft_trainer.py`) doesn't produce. This evaluator
activates an adapter the same way `sft_trainer.py` trains it: run the hypernetwork
forward, then `hypernetwork.apply(layers, generated)` to hook it live into the real
interpreter's forward pass. Every adapter — weight- or activation-based — is
evaluated identically, through the exact same code path it was trained through.

The scoring routines below (`_score_multiple_choice`, `_score_gsm8k`, `_group_by_family`)
are intentionally near-verbatim from `HFDownstreamEvaluator`: they only ever call
`model(...)`/`model.generate(...)` on whatever `model` object they're handed, so they
needed no changes — only the activation mechanism around them differs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import time

import torch
from torch import Tensor, nn

from ..contracts import EvaluationResult, TaskExample
from ..hf_downstream_evaluator import _NUMBER_RE
from .hypernetwork import TextToPeftHypernetwork


class HypernetworkDownstreamEvaluator:
    def __init__(
        self,
        interpreter: nn.Module,
        layers: nn.ModuleList,
        hypernetwork: TextToPeftHypernetwork,
        tokenizer,
        trial_id: str,
        device: str = "cuda:0",
        max_new_tokens: int = 256,
    ):
        self.interpreter = interpreter
        self.layers = layers
        self.hypernetwork = hypernetwork
        self.tokenizer = tokenizer
        self.trial_id = trial_id
        self.device = device
        self.max_new_tokens = max_new_tokens

    @contextmanager
    def _active(self, condition_embedding: Tensor | None):
        if condition_embedding is None:
            yield self.interpreter
            return
        with torch.no_grad():
            generated = self.hypernetwork(condition_embedding.unsqueeze(0))
        with self.hypernetwork.apply(self.layers, generated):
            yield self.interpreter

    def _prompt(self, input_text: str) -> str:
        # enable_thinking=False matters for reasoning-capable interpreters (Qwen3): without
        # it, the model expects to emit its own <think>...</think> block before answering,
        # so scoring a continuation immediately after the prompt is badly out-of-distribution
        # (confirmed directly: measured near content-independent log-likelihoods, e.g. "no"
        # scored higher than "yes" for "Is Paris the capital of France?"). Must match
        # lol_data.py::format_prompt_response's training-time prompt format exactly, or the
        # frozen/adapted comparison isn't apples-to-apples. No-op for non-Qwen3 tokenizers.
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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

    @staticmethod
    def _group_by_family(examples: Iterable[TaskExample]) -> dict[str, list[TaskExample]]:
        by_family: dict[str, list[TaskExample]] = defaultdict(list)
        for example in examples:
            by_family[example.family].append(example)
        return by_family

    def _evaluate_group(
        self, family: str, examples: list[TaskExample], split: str, condition_embedding: Tensor | None
    ) -> EvaluationResult:
        started = time.perf_counter()
        with self._active(condition_embedding) as model:
            correct = sum(1 for example in examples if self._score(model, example))
        inference_seconds = time.perf_counter() - started

        metric_name = "exact_match" if family == "gsm8k" else "accuracy"
        adapter = self.hypernetwork.adapter if condition_embedding is not None else "frozen_interpreter"
        return EvaluationResult(
            trial_id=self.trial_id,
            task_id=family,
            split=split,
            adapter=adapter,
            metrics={metric_name: correct / len(examples), "n_examples": float(len(examples))},
            generated_parameter_count=(
                self.hypernetwork.generated_parameter_count() if condition_embedding is not None else 0
            ),
            generation_seconds=0.0,
            inference_seconds=inference_seconds,
            metadata={"condition_variant": examples[0].metadata.get("condition_variant") if examples else None},
        )

    def iter_evaluate(self, condition_embeddings: Mapping[str, Tensor], examples: Iterable[TaskExample], split: str):
        """Yield one EvaluationResult per task family as it completes — prefer this
        over ``evaluate()`` for anything long-running (same reasoning as
        ``HFDownstreamEvaluator.iter_evaluate``)."""
        for family, family_examples in self._group_by_family(examples).items():
            yield self._evaluate_group(family, family_examples, split, condition_embeddings.get(family))

    def iter_evaluate_frozen(self, examples: Iterable[TaskExample], split: str):
        for family, family_examples in self._group_by_family(examples).items():
            yield self._evaluate_group(family, family_examples, split, None)

    def evaluate(
        self, condition_embeddings: Mapping[str, Tensor], examples: Iterable[TaskExample], split: str
    ) -> list[EvaluationResult]:
        return list(self.iter_evaluate(condition_embeddings, examples, split))

    def evaluate_frozen(self, examples: Iterable[TaskExample], split: str) -> list[EvaluationResult]:
        return list(self.iter_evaluate_frozen(examples, split))
