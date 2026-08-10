"""Generation-based answer-extraction and scoring utilities.

These are the pure scoring routines shared by every evaluator: given a model's
free-text generation and a target, extract a leading choice letter/digit, a loose
true/false keyword, or a final number, and decide correctness. They match upstream
Text-to-LoRA's own generation+extraction eval protocol
(``hyper_llm_modulator.utils.eval_tasks``) rather than log-likelihood-over-choices, so
a scorer built on them measures the same thing the paper's published numbers do.

``rouge_l`` is Super-NaturalInstructions' own aggregate metric (see
``natural-instructions/eval/automatic/evaluation.py``), reimplemented here because the
``rouge_score`` package is not vendored: LCS F-measure over ``rouge_score``'s tokenization
with ``use_stemmer=False``, which is the setting SNI's evaluator uses. It reads off the
same generations as exact match, so a generation pass yields both metrics.

The module-level ``torch.backends`` determinism settings match upstream's
``hyper_llm_modulator.vllm_eval.eval()`` (whose absence was confirmed to make
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


# --- ROUGE-L (SNI's aggregate metric) ------------------------------------------------

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_ALNUM_TOKEN_RE = re.compile(r"^[a-z0-9]+$")


def rouge_tokenize(text: str) -> list[str]:
    """Tokenize exactly as ``rouge_score.tokenize.tokenize(text, stemmer=None)`` does:
    lowercase, collapse every non-``[a-z0-9]`` run to a space, split, drop empties.

    No stemming, because SNI's evaluator constructs its scorer with
    ``use_stemmer=False``; adding one here would silently diverge from the metric the
    upstream numbers are reported under.
    """
    lowered = _NON_ALNUM_RE.sub(" ", str(text).lower())
    return [tok for tok in lowered.split() if _ALNUM_TOKEN_RE.match(tok)]


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence, via the standard row-rolled DP."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l(generated_text: str, target_text: str) -> float:
    """ROUGE-L F-measure: LCS-based precision/recall harmonic mean, in [0, 1].

    Matches ``rouge_score``'s ``rougeL`` (not ``rougeLsum``: no sentence splitting) with
    ``use_stemmer=False``, which is what SNI's ``rouge()`` helper computes. An empty
    prediction or empty target scores 0.0, as upstream's ``_score_lcs`` does.
    """
    gen_tokens = rouge_tokenize(generated_text)
    tgt_tokens = rouge_tokenize(target_text)
    if not gen_tokens or not tgt_tokens:
        return 0.0
    lcs = lcs_length(tgt_tokens, gen_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(gen_tokens)
    recall = lcs / len(tgt_tokens)
    return 2.0 * precision * recall / (precision + recall)


def rouge_l_max(generated_text: str, target_texts) -> float:
    """SNI scores an instance against every acceptable reference and keeps the best.

    T2A's vendored `lol_` tasks carry a single answer per example, so this reduces to
    ``rouge_l`` there; it exists so a multi-reference task set does not need a second
    scorer. An empty reference list scores 0.0.
    """
    scores = [rouge_l(generated_text, target) for target in target_texts]
    return max(scores) if scores else 0.0
