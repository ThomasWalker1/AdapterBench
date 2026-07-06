"""Exact-match scoring for the lightweight synthetic setting. No chat template, no text
parsing, no loglikelihood scoring over natural-language choices (contrast
``live_evaluator.py``) — every synthetic example has a fixed-length prompt and a
fixed-length, exactly-known correct continuation, so scoring is a direct greedy-decode +
token comparison. Kept function-based rather than a class, proportionate to how little
logic remains once there's no tokenizer/chat-template coupling to manage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import random

import torch
from torch import Tensor, nn

from .hypernetwork import TextToPeftHypernetwork
from .synthetic_tasks import TASK_FAMILIES, build_prompt_ids, sample_digits
from .tiny_interpreter import DIGIT_BASE, PAD_ID


@contextmanager
def _active(
    hypernetwork: TextToPeftHypernetwork | None,
    layers: nn.ModuleList,
    condition_embedding: Tensor | None,
    batch_size: int,
):
    if condition_embedding is None:
        yield
        return
    # codecs require one generated adapter per interpreter-batch element (see
    # codecs.py::GeneratedUpdateCodec._check) — expand the single condition embedding to
    # match the eval batch of `batch_size` examples scored together for this family.
    with torch.no_grad():
        generated = hypernetwork(condition_embedding.unsqueeze(0).expand(batch_size, -1))
    with hypernetwork.apply(layers, generated):
        yield


def evaluate_family(
    hypernetwork: TextToPeftHypernetwork | None,
    interpreter: nn.Module,
    layers: nn.ModuleList,
    family_name: str,
    condition_embedding: Tensor | None,
    *,
    seq_len: int = 5,
    num_examples: int = 20,
    device: str = "cpu",
    seed: int = 1234,
) -> float:
    """Exact-match accuracy over ``num_examples`` fresh random inputs for one task
    family. ``condition_embedding=None`` scores the frozen interpreter (no hook)."""
    family = TASK_FAMILIES[family_name]
    family_index = list(TASK_FAMILIES).index(family_name)
    rng = random.Random(seed * 7_919 + family_index * 999_983)
    prompts, targets = [], []
    for _ in range(num_examples):
        digits = sample_digits(rng, seq_len)
        prompts.append(build_prompt_ids(digits))
        targets.append(family.transform(digits))

    input_ids = torch.tensor(prompts, device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    with _active(hypernetwork, layers, condition_embedding, num_examples):
        with torch.no_grad():
            generated = interpreter.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=seq_len + 1,
                do_sample=False,
                pad_token_id=PAD_ID,
            )

    predicted = (generated[:, prompt_len : prompt_len + seq_len] - DIGIT_BASE).tolist()
    correct = sum(1 for pred, target in zip(predicted, targets) if pred == target)
    return correct / num_examples


def evaluate_families(
    hypernetwork: TextToPeftHypernetwork | None,
    interpreter: nn.Module,
    layers: nn.ModuleList,
    condition_embeddings: Mapping[str, Tensor] | None,
    family_names: Iterable[str],
    *,
    seq_len: int = 5,
    num_examples: int = 20,
    device: str = "cpu",
    seed: int = 1234,
) -> dict[str, float]:
    """``condition_embeddings=None`` scores the frozen interpreter baseline for every
    family (no hook applied)."""
    return {
        family_name: evaluate_family(
            hypernetwork,
            interpreter,
            layers,
            family_name,
            condition_embeddings[family_name] if condition_embeddings is not None else None,
            seq_len=seq_len,
            num_examples=num_examples,
            device=device,
            seed=seed,
        )
        for family_name in family_names
    }
