"""DownstreamEvaluator that scores generated adapters with plain transformers+peft.

Deliberately avoids vLLM: it is not installed in this environment, and non-LoRA
adapters (FourierFT/IA3/LoKr) this benchmark ultimately compares are not
vLLM-servable anyway, so a custom evaluator is required regardless.

Replicates upstream Text-to-LoRA's tokenizer setup (padding side, pad token, chat
template, truncation side - see ``hyper_llm_modulator.utils.model_loading.get_tokenizer``)
so generated adapters are scored under the same input formatting they were evaluated
against upstream. The task condition text is never shown to the interpreter - only
used to generate the adapter - matching upstream's default ``system_message=""`` eval mode.

Scoring itself (this module's ``get_choice``/``get_binary_accuracy``) is generation +
answer-extraction, not log-likelihood-over-choices: upstream's own eval harness
(``hyper_llm_modulator.vllm_eval``/``utils.eval_tasks``) generates free text against a
templated prompt (see ``task_examples.py``) and extracts a leading choice letter/digit
or a loose true/false keyword. A log-likelihood scorer answers a different question
("which full choice text is most probable") than what the paper's published numbers
measure, so it isn't a substitute for this protocol if the goal is reproducing them.
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

# Matches upstream's hyper_llm_modulator.vllm_eval.eval() determinism settings (also
# applied in scripts/generate_t2l_adapter.py, where their absence was confirmed to make
# adapter generation non-deterministic across process invocations - see that script's
# comment). Applied here too so a fixed adapter's scored accuracy doesn't depend on
# GPU kernel-selection non-determinism either.
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

# Ordered so single-digit "10" doesn't get misparsed as leading-"1"; matches upstream's
# `hyper_llm_modulator.utils.eval_tasks.get_choice` CHOICES list.
_CHOICE_ORDER = [*"abcdefghijklmnopqrstuvwxyz", *"0123456789", "10"]


def get_choice(text: str) -> str | None:
    """Extract the leading answer choice (letter or digit) from generated text,
    matching upstream's ``get_choice`` exactly."""
    stripped = str(text).strip().strip(":`'\"(.) ").lower()
    for choice in _CHOICE_ORDER:
        if stripped.startswith(choice):
            return choice
    return None


def get_choice_accuracy(generated_text: str, target_text: str) -> bool:
    return get_choice(generated_text) == get_choice(target_text)


def get_bool_value(text: str) -> bool | None:
    """Loose true/false extraction, matching upstream's ``get_bool_value_from_text``:
    checks digits before words, and returns None (never-correct) if no boolean-ish
    token appears at all."""
    text = str(text)
    if "1" in text:
        return True
    if "0" in text:
        return False
    lowered = text.lower()
    for word, value in (
        ("yes", True),
        ("no", False),
        ("true", True),
        ("false", False),
        ("positive", True),
        ("negative", False),
        ("valid", True),
        ("invalid", False),
    ):
        if word in lowered:
            return value
    return None


def get_binary_accuracy(generated_text: str, target_text: str) -> bool:
    predicted = get_bool_value(generated_text)
    target = get_bool_value(target_text)
    if predicted is None or target is None:
        return False
    return predicted == target


def get_gsm8k_accuracy(generated_text: str, target_text: str) -> bool:
    matches = _NUMBER_RE.findall(generated_text)
    if not matches:
        return False
    predicted = matches[-1].replace(",", "")
    try:
        return abs(float(predicted) - float(target_text)) < 1e-4
    except ValueError:
        return False


def build_prefill_by_family(use_icl: bool) -> dict[str, str]:
    """GSM8K always gets a "Let's think step by step." nudge (matches upstream's
    ``eval_gsm8k``, unconditional on ``use_icl``); multiple-choice/boolq tasks only get
    the "Answer:" nudge when reproducing the ICL-enabled tables (e.g. the paper's Gemma
    table, which uses ICL for every method) - callers must pass the same ``use_icl``
    value used to build the ``TaskExample``s."""
    prefill = {"gsm8k": "Let's think step by step."}
    if use_icl:
        for family in ("arc_easy", "arc_challenge", "hellaswag", "boolq"):
            prefill[family] = "Answer:"
    return prefill


def render_prompt(tokenizer, input_text: str, prefill: str = "") -> str:
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": input_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return chat_prompt + prefill


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
        max_new_tokens: int = 512,  # matches upstream's vllm.SamplingParams(max_tokens=2**9)
        use_icl: bool = False,
    ):
        from transformers import AutoModelForCausalLM, set_seed

        set_seed(42)  # matches upstream's vllm_eval.eval() (also LLM(..., seed=42))

        self.model_id = model_id
        self.trial_id = trial_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = load_faithful_tokenizer(model_id, chat_template_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        self.model.eval()
        self.peft_model = None
        self._loaded_adapter_names: set[str] = set()
        self._prefill_by_family = build_prefill_by_family(use_icl)

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

    def _prompt(self, input_text: str, prefill: str = "") -> str:
        return render_prompt(self.tokenizer, input_text, prefill)

    def _generate(self, model, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(output_ids[0, input_ids["input_ids"].shape[1] :], skip_special_tokens=True)

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
        return get_gsm8k_accuracy(generated, example.target_text)

    def _score(self, model, example: TaskExample) -> bool:
        if example.family == "gsm8k":
            return self._score_gsm8k(model, example)
        if example.family == "boolq":
            return self._score_boolq(model, example)
        return self._score_choice(model, example)

    def _evaluate_group(
        self,
        family: str,
        examples: list[TaskExample],
        split: str,
        artifact: AdapterArtifact | None,
    ) -> EvaluationResult:
        adapter_name = None
        if artifact is not None:
            adapter_name = f"{family}__{artifact.adapter}"
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
        adapter = artifact.adapter if artifact is not None else "frozen_interpreter"
        return EvaluationResult(
            trial_id=self.trial_id,
            task_id=family,
            split=split,
            adapter=adapter,
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
