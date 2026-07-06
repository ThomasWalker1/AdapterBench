"""Synthetic algorithmic tasks for the lightweight setting: a small hard-coded registry
of task families with a pure-Python, exactly-known transform and a handful of
hand-written natural-language description paraphrases each — no external data, no
description-generation step to track down later (contrast ``lol_data.py``'s unresolved
GPT-4o-mini provenance gap). Ground truth means a representation's live-SFT loss and
downstream accuracy can be checked against "does this codec even have the capacity to
represent this transform," decoupled from real-world task difficulty/noise.

Examples are encoded directly as integer ids over the fixed 16-symbol vocabulary defined
in ``tiny_interpreter.py`` (``PAD/BOS/EOS/SEP`` plus digits ``0``-``9``) — no HF tokenizer
involved anywhere in this setting.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import Dataset

from .tiny_interpreter import BOS_ID, DIGIT_BASE, EOS_ID, SEP_ID


@dataclass(frozen=True)
class TaskFamily:
    name: str
    transform: Callable[[list[int]], list[int]]
    descriptions: list[str]


def _copy(digits: list[int]) -> list[int]:
    return list(digits)


def _reverse(digits: list[int]) -> list[int]:
    return list(reversed(digits))


def _increment(digits: list[int]) -> list[int]:
    return [(d + 1) % 10 for d in digits]


def _sort(digits: list[int]) -> list[int]:
    return sorted(digits)


def _constant(digits: list[int]) -> list[int]:
    return [9] * len(digits)


TASK_FAMILIES: dict[str, TaskFamily] = {
    "copy": TaskFamily(
        "copy",
        _copy,
        [
            "Copy the input sequence exactly.",
            "Repeat the digits unchanged.",
            "Output exactly what you were given.",
            "Return the same sequence as the input.",
        ],
    ),
    "reverse": TaskFamily(
        "reverse",
        _reverse,
        [
            "Reverse the order of the digits.",
            "Output the digits in reverse order.",
            "Flip the sequence back to front.",
            "Write the digits backwards.",
        ],
    ),
    "increment": TaskFamily(
        "increment",
        _increment,
        [
            "Add one to every digit, wrapping nine to zero.",
            "Increment each digit by one, modulo ten.",
            "Output each digit plus one, wrapping around after nine.",
            "Shift every digit up by one, cycling nine back to zero.",
        ],
    ),
    "sort": TaskFamily(
        "sort",
        _sort,
        [
            "Sort the digits in ascending order.",
            "Output the digits from smallest to largest.",
            "Arrange the sequence in increasing order.",
            "Rearrange the digits so they are non-decreasing.",
        ],
    ),
    "constant": TaskFamily(
        "constant",
        _constant,
        [
            "Ignore the input and output all nines.",
            "Regardless of the input, respond with nines only.",
            "Output a sequence of nines no matter what you are given.",
            "Disregard the input digits and always answer with nines.",
        ],
    ),
}

_FAMILY_ORDER = list(TASK_FAMILIES)


def sample_digits(rng: random.Random, seq_len: int) -> list[int]:
    return [rng.randrange(10) for _ in range(seq_len)]


def encode_digits(digits: Sequence[int]) -> list[int]:
    return [DIGIT_BASE + d for d in digits]


def build_prompt_ids(input_digits: Sequence[int]) -> list[int]:
    return [BOS_ID] + encode_digits(input_digits) + [SEP_ID]


def build_target_ids(output_digits: Sequence[int]) -> list[int]:
    return encode_digits(output_digits) + [EOS_ID]


def encode_example(input_digits: Sequence[int], output_digits: Sequence[int]) -> dict:
    """Response-only supervision, same prompt/``-100`` shape as
    ``lol_data.py::tokenize_prompt_response`` but at the symbol-id level."""
    prompt_ids = build_prompt_ids(input_digits)
    target_ids = build_target_ids(output_digits)
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class SyntheticSFTDataset(Dataset):
    """Generates examples on the fly via a per-``(seed, family, index)`` RNG — no
    dataset download, fully reproducible. Mirrors ``LolSFTDataset``'s random (train) /
    deterministic (eval) condition-embedding draw convention."""

    def __init__(
        self,
        family_name: str,
        condition_embeddings: Tensor,
        *,
        seq_len: int = 5,
        size: int = 200,
        training: bool = True,
        seed: int = 0,
    ):
        self.family = TASK_FAMILIES[family_name]
        self.family_name = family_name
        self.condition_embeddings = condition_embeddings
        self.seq_len = seq_len
        self.size = size
        self.training = training
        self.seed = seed
        self._family_index = _FAMILY_ORDER.index(family_name)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        seed = self.seed * 10_000_019 + self._family_index * 104_729 + index
        rng = random.Random(seed)
        digits = sample_digits(rng, self.seq_len)
        output = self.family.transform(digits)
        example = encode_example(digits, output)
        num_descriptions = len(self.condition_embeddings)
        emb_index = rng.randrange(num_descriptions) if self.training else index % num_descriptions
        example["condition_embedding"] = self.condition_embeddings[emb_index]
        example["task_id"] = self.family_name
        return example
