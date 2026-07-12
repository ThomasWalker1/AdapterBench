"""Generation-based answer-extraction and scoring utilities.

These are the pure scoring routines shared by every evaluator: given a model's
free-text generation and a target, extract a leading choice letter/digit, a loose
true/false keyword, or a final number, and decide correctness. They match upstream
Text-to-LoRA's own generation+extraction eval protocol
(``hyper_llm_modulator.utils.eval_tasks``) rather than log-likelihood-over-choices, so
a scorer built on them measures the same thing the paper's published numbers do.

The module-level ``torch.backends`` determinism settings match upstream's
``hyper_llm_modulator.vllm_eval.eval()`` (also applied in their
``scripts/generate_t2l_adapter.py``, where their absence was confirmed to make
generation non-deterministic across process invocations); importing this module
applies them so a fixed adapter's scored accuracy doesn't depend on GPU
kernel-selection non-determinism.
"""

from __future__ import annotations

import re

import torch

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
