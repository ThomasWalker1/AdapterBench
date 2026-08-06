"""DownstreamEvaluator for live-hook-trained hypernetworks.

A hook-based adapter (activation steering, or any adapter trained via
`sft_trainer.py`) never produces a materialized, PEFT-native adapter directory to load
via `peft.PeftModel.load_adapter`/`set_adapter`. This evaluator instead activates an
adapter the same way `sft_trainer.py` trains it: run the hypernetwork forward, then
`hypernetwork.apply(layers, generated)` to hook it live into the real interpreter's
forward pass. Every adapter — weight- or activation-based — is evaluated identically,
through the exact same code path it was trained through.

Answer extraction (`get_choice_accuracy`/`get_binary_accuracy`) is imported from
`scoring.py` rather than reimplemented so scoring stays faithful to upstream
Text-to-LoRA's own generation+extraction eval protocol (see `scoring.py`'s module
docstring).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import time

import torch
from torch import Tensor, nn

from ..contracts import EvaluationResult, TaskExample
from ..scoring import _NUMBER_RE, get_binary_accuracy, get_choice_accuracy
from .hypernetwork import TextToPeftHypernetwork
from .niah_data import NiahExample, build_query_for_example, encode_context


def _greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int, device) -> str:
    """Greedy-decode the continuation of ``prompt`` and return only the newly generated text."""
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_ids[0, input_ids["input_ids"].shape[1] :], skip_special_tokens=True)


class HypernetworkDownstreamEvaluator:
    def __init__(
        self,
        interpreter: nn.Module,
        layers: nn.ModuleList,
        hypernetwork: TextToPeftHypernetwork,
        tokenizer,
        trial_id: str,
        device: str = "cuda:0",
        max_new_tokens: int = 512,  # matches upstream's vllm.SamplingParams(max_tokens=2**9)
        use_icl: bool = False,
    ):
        self.interpreter = interpreter
        self.layers = layers
        self.hypernetwork = hypernetwork
        self.tokenizer = tokenizer
        self.trial_id = trial_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._prefill_by_family = {"gsm8k": "Let's think step by step."}
        if use_icl:
            for family in ("arc_easy", "arc_challenge", "hellaswag", "boolq"):
                self._prefill_by_family[family] = "Answer:"

    @contextmanager
    def _active(self, condition_embedding: Tensor | None):
        if condition_embedding is None:
            yield self.interpreter
            return
        with torch.no_grad():
            generated = self.hypernetwork(condition_embedding.unsqueeze(0))
        with self.hypernetwork.apply(self.layers, generated):
            yield self.interpreter

    def _prompt(self, input_text: str, prefill: str = "") -> str:
        # enable_thinking=False matters for reasoning-capable interpreters (Qwen3): without
        # it, the model expects to emit its own <think>...</think> block before answering,
        # so scoring a continuation immediately after the prompt is badly out-of-distribution
        # (confirmed directly: measured near content-independent log-likelihoods, e.g. "no"
        # scored higher than "yes" for "Is Paris the capital of France?"). Must match
        # lol_data.py::format_prompt_response's training-time prompt format exactly, or the
        # frozen/adapted comparison isn't apples-to-apples. No-op for non-Qwen3 tokenizers.
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return chat_prompt + prefill

    def _generate(self, model, prompt: str) -> str:
        return _greedy_generate(model, self.tokenizer, prompt, self.max_new_tokens, self.device)

    def _score_choice(self, model, example: TaskExample) -> bool:
        prefill = self._prefill_by_family.get(example.family, "")
        generated = self._generate(model, self._prompt(example.input_text, prefill))
        return get_choice_accuracy(generated, example.target_text)

    def _score_boolq(self, model, example: TaskExample) -> bool:
        prefill = self._prefill_by_family.get(example.family, "")
        generated = self._generate(model, self._prompt(example.input_text, prefill))
        return get_binary_accuracy(generated, example.target_text)

    def _score_gsm8k(self, model, example: TaskExample) -> bool:
        prefill = self._prefill_by_family.get("gsm8k", "")
        generated = self._generate(model, self._prompt(example.input_text, prefill))
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
        if example.family == "boolq":
            return self._score_boolq(model, example)
        return self._score_choice(model, example)

    @staticmethod
    def _group_by_family(examples: Iterable[TaskExample]) -> dict[str, list[TaskExample]]:
        by_family: dict[str, list[TaskExample]] = defaultdict(list)
        for example in examples:
            by_family[example.family].append(example)
        return by_family

    def _evaluate_group(
        self, family: str, examples: list[TaskExample], split: str, condition_embedding: Tensor | None,
        mismatched_embedding: Tensor | None = None,
    ) -> EvaluationResult:
        started = time.perf_counter()
        with self._active(condition_embedding) as model:
            correct = sum(1 for example in examples if self._score(model, example))
        inference_seconds = time.perf_counter() - started

        metric_name = "exact_match" if family == "gsm8k" else "accuracy"
        metrics = {metric_name: correct / len(examples), "n_examples": float(len(examples))}
        # Mismatched-description control (the T2A analogue of D2A's context-swap, invariant #1):
        # score this family with an adapter generated from a *different* task's description. A
        # genuine task-conditioned adapter scores near this family's frozen/chance level here;
        # only matched >> mismatched is evidence the hypernetwork uses the description, not a
        # task-independent bias. Never headline raw accuracy - headline matched - mismatched.
        if mismatched_embedding is not None:
            with self._active(mismatched_embedding) as model:
                mm_correct = sum(1 for example in examples if self._score(model, example))
            metrics[f"{metric_name}_mismatched"] = mm_correct / len(examples)

        adapter = self.hypernetwork.adapter if condition_embedding is not None else "frozen_interpreter"
        return EvaluationResult(
            trial_id=self.trial_id,
            task_id=family,
            split=split,
            adapter=adapter,
            metrics=metrics,
            generated_parameter_count=(
                self.hypernetwork.generated_parameter_count() if condition_embedding is not None else 0
            ),
            generation_seconds=0.0,
            inference_seconds=inference_seconds,
            metadata={"condition_variant": examples[0].metadata.get("condition_variant") if examples else None},
        )

    def iter_evaluate(
        self, condition_embeddings: Mapping[str, Tensor], examples: Iterable[TaskExample], split: str,
        mismatched_embeddings: Mapping[str, Tensor] | None = None,
    ):
        """Yield one EvaluationResult per task family as it completes — prefer this
        over ``evaluate()`` for anything long-running, so a caller can print progress
        and persist partial results and an interrupted run keeps what finished. When
        ``mismatched_embeddings`` is given, each result also carries the mismatched-description
        control (``accuracy_mismatched``/``exact_match_mismatched``)."""
        for family, family_examples in self._group_by_family(examples).items():
            yield self._evaluate_group(
                family, family_examples, split, condition_embeddings.get(family),
                mismatched_embeddings.get(family) if mismatched_embeddings else None,
            )

    def iter_evaluate_frozen(self, examples: Iterable[TaskExample], split: str):
        for family, family_examples in self._group_by_family(examples).items():
            yield self._evaluate_group(family, family_examples, split, None)

    def evaluate(
        self, condition_embeddings: Mapping[str, Tensor], examples: Iterable[TaskExample], split: str
    ) -> list[EvaluationResult]:
        return list(self.iter_evaluate(condition_embeddings, examples, split))

    def evaluate_frozen(self, examples: Iterable[TaskExample], split: str) -> list[EvaluationResult]:
        return list(self.iter_evaluate_frozen(examples, split))


class DocumentHypernetworkDownstreamEvaluator:
    """Document-conditioned counterpart to `HypernetworkDownstreamEvaluator`, for
    scoring held-out NIAH (`niah_data.py`) examples. The activation mechanism is
    identical in spirit (hook the generated adapter live via
    `hypernetwork.apply(...)`, never `peft.PeftModel.load_adapter`) but the
    conditioning input is a whole per-example document rather than one embedding
    shared by an entire task family - `HypernetworkDownstreamEvaluator._active` hooks
    once per family (one condition_embedding serves every example in that family);
    this evaluator's `_active` instead runs the conditioner's `prepare_condition` (a real
    interpreter forward pass) and hooks a fresh adapter once *per example*, since every
    NIAH document is distinct. Scoring is exact-match on the needle's 4-digit answer
    (substring containment in the generated continuation), not
    `get_choice_accuracy`/`get_binary_accuracy` (those are for multiple-choice/boolean
    answers, not a short numeric string) and not a word-level ROUGE-L - exact-match is
    appropriate here since the generated answer is always meant to be exactly the
    needle's digits, not a free-form span ROUGE-L would be needed to fuzzily match.
    """

    def __init__(
        self,
        interpreter: nn.Module,
        layers: nn.ModuleList,
        hypernetwork: TextToPeftHypernetwork | None,
        tokenizer,
        trial_id: str,
        device: str = "cuda:0",
        max_new_tokens: int = 16,  # the answer is always exactly 4 digits - far less generation budget than free-form tasks
        max_context_len: int | None = None,
    ):
        self.interpreter = interpreter
        self.layers = layers
        self.hypernetwork = hypernetwork
        self.tokenizer = tokenizer
        self.trial_id = trial_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_context_len = max_context_len

    @contextmanager
    def _active(self, example: NiahExample | None):
        if example is None or self.hypernetwork is None:
            yield self.interpreter
            return
        # Tokenize the context exactly as training did for this needle_style.
        # Do not silently cap an eval bin below its requested length: D2A's length
        # curve is meaningful only when an 8192-token document actually conditions the
        # generated adapter. Callers may still supply an explicit safety cap.
        # (generic -> chat-wrapped; topic -> raw), then let the conditioner produce its
        # own raw_condition via `prepare_condition` - `EarlyExitPerceiverConditioner`
        # returns the full per-layer activation stack, `EarlyExitPerceiverConditioner`
        # returns its early-exit-encoded latents - so this evaluator is conditioner-agnostic
        # (the codec seam and generate_per_layer contract are identical for both).
        encoded = encode_context(self.tokenizer, example, max_length=self.max_context_len)
        input_ids = torch.tensor([encoded["input_ids"]], device=self.device)
        attention_mask = torch.tensor(
            [encoded.get("attention_mask", [1] * len(encoded["input_ids"]))], device=self.device
        )
        with torch.no_grad():
            raw_condition = self.hypernetwork.conditioner.prepare_condition(
                self.interpreter, input_ids, attention_mask
            )
            generated = self.hypernetwork.generate_per_layer(raw_condition)
        with self.hypernetwork.apply(self.layers, generated):
            yield self.interpreter

    def _generate(self, model, prompt: str) -> str:
        return _greedy_generate(model, self.tokenizer, prompt, self.max_new_tokens, self.device)

    def _score(self, model, example: NiahExample) -> bool:
        prompt = build_query_for_example(self.tokenizer, example)
        generated = self._generate(model, prompt)
        return example.digits in generated

    def _evaluate_group(self, family: str, examples: list[NiahExample], split: str, active: bool) -> EvaluationResult:
        started = time.perf_counter()
        n = len(examples)
        correct = 0
        swap_hits = 0
        for i, example in enumerate(examples):
            with self._active(example if active else None) as model:
                correct += int(self._score(model, example))
            # Context-swap control: same query, but the adapter is generated from the WRONG
            # document (the next example's). Genuine document-dependent retrieval must NOT
            # surface this example's digits here, so accuracy_ctxswap should sit near chance.
            # The matched-minus-swapped gap is the real retrieval signal - exact-digit CE/loss
            # is not (a model can drive response CE to ~0 by learning digit priors + the
            # teacher-forced continuation without ever routing the document; see PROJECT_PLAN).
            if active and n > 1:
                with self._active(examples[(i + 1) % n]) as model:
                    swap_hits += int(self._score(model, example))
        inference_seconds = time.perf_counter() - started

        adapter = self.hypernetwork.adapter if (active and self.hypernetwork is not None) else "frozen_interpreter"
        metrics = {"accuracy": correct / n, "n_examples": float(n)}
        if active and self.hypernetwork is not None and n > 1:
            metrics["accuracy_ctxswap"] = swap_hits / n
        return EvaluationResult(
            trial_id=self.trial_id,
            task_id=family,
            split=split,
            adapter=adapter,
            metrics=metrics,
            generated_parameter_count=(
                self.hypernetwork.generated_parameter_count() if (active and self.hypernetwork is not None) else 0
            ),
            generation_seconds=0.0,
            inference_seconds=inference_seconds,
            metadata={"depths": [example.depth for example in examples]},
        )

    def iter_evaluate(self, examples_by_family: Mapping[str, list[NiahExample]], split: str):
        """Yield one EvaluationResult per context-length family as it completes -
        same reasoning as `HypernetworkDownstreamEvaluator.iter_evaluate`."""
        for family, examples in examples_by_family.items():
            yield self._evaluate_group(family, examples, split, active=True)

    def iter_evaluate_frozen(self, examples_by_family: Mapping[str, list[NiahExample]], split: str):
        for family, examples in examples_by_family.items():
            yield self._evaluate_group(family, examples, split, active=False)

    def evaluate(self, examples_by_family: Mapping[str, list[NiahExample]], split: str) -> list[EvaluationResult]:
        return list(self.iter_evaluate(examples_by_family, split))

    def evaluate_frozen(self, examples_by_family: Mapping[str, list[NiahExample]], split: str) -> list[EvaluationResult]:
        return list(self.iter_evaluate_frozen(examples_by_family, split))
