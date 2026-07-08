"""Synthetic needle-in-a-haystack (NIAH) data for document-conditioned live SFT
(Setting 2's document-conditioning variant): a haystack of cycled filler sentences with
exactly one needle sentence inserted at a controlled depth, paired with a query/answer
about that needle. Reuses `lol_data.py::tokenize_prompt_response` for response-only
supervision on the query/answer side (that helper is a pure tokenizer/prompt/response
string utility with no Lots-of-LoRAs-specific content, so there is nothing
NIAH-specific to reimplement there).

Deliberately synthetic rather than read from `ctx_to_lora`'s own on-disk
`ctx_magic_number_<lo>_<hi>` bins (used by Setting 1's `run-d2l-niah`): this setting
trains its own from-scratch hypernetwork under a live SFT loop, which needs many fresh
documents per training step, not a fixed released-checkpoint eval split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .lol_data import tokenize_prompt_response

DEFAULT_CONTEXT_LENGTHS: tuple[int, ...] = (256, 512, 1024, 2048)

# Cycled (not repeated verbatim) so the haystack isn't trivially compressible in a way
# that would let a model "solve" NIAH by pattern-matching "the one sentence that isn't
# identical to its neighbors" rather than actually reading content.
_FILLER_SENTENCES: tuple[str, ...] = (
    "The grass is green.",
    "The sky is blue.",
    "Trees have leaves.",
    "Water flows downhill.",
    "The sun rises in the east.",
    "Birds can fly through the air.",
    "Rivers run into the sea.",
    "Mountains are made of rock.",
    "Snow falls in the winter.",
    "Flowers bloom in the spring.",
)

_TOPICS: tuple[str, ...] = (
    "engineering", "history", "chemistry", "music", "finance", "biology",
    "astronomy", "geography", "philosophy", "medicine", "aviation", "cooking",
)


def assert_context_fits_in_one_pass(context_length: int, model_max_position_embeddings: int) -> None:
    """D2L's own recipe splits an over-length document across multiple chunks and
    composes each chunk's generated LoRA rank together (`combine_lora`). This
    implementation deliberately does NOT replicate that - documents are packed into one
    context window per example instead (documented simplification, see
    PROJECT_PLAN.md). That choice is only valid as long as every configured context
    length actually fits in one forward pass: Qwen3-0.6B's `max_position_embeddings` is
    40960, comfortably larger than any length bin used by `DEFAULT_CONTEXT_LENGTHS`
    (largest: 2048), so no real chunking utility is implemented here. This assertion is
    the safety net for a future caller configuring a longer bin than the interpreter
    can actually take in one pass - it should fail loudly rather than silently
    truncating the document out from under the needle.
    """
    if context_length > model_max_position_embeddings:
        raise ValueError(
            f"context_length={context_length} exceeds model_max_position_embeddings="
            f"{model_max_position_embeddings} - multi-chunk splitting (D2L's own "
            "combine_lora rank composition across chunks) is not implemented here; "
            "see assert_context_fits_in_one_pass's docstring."
        )


@dataclass(frozen=True)
class NiahExample:
    context_text: str
    topic: str
    digits: str
    depth: float  # requested (not necessarily exact - see make_niah_example) fractional depth in [0, 1]


def make_niah_example(
    tokenizer,
    context_length: int,
    *,
    depth: float | None = None,
    rng: random.Random | None = None,
) -> NiahExample:
    """Build one haystack+needle document of approximately `context_length` tokens.

    `depth` is the needle sentence's fractional position among the filler sentences
    (0.0 = before any filler sentence, at the very start; 1.0 = at the very end); sampled
    uniformly in [0, 1] per example when not given explicitly (training), or pinned to
    a specific value (eval, to score depth-stratified accuracy). The needle's *actual*
    position can only land on a filler-sentence boundary, and the filler count is
    itself derived from `context_length` via each sentence's average token count - so
    the realized depth is an approximation of the requested one, not exact; see
    `tests/test_niah_data.py` for the tolerance this is checked at.
    """
    rng = rng or random.Random()
    resolved_depth = rng.random() if depth is None else depth
    if not 0.0 <= resolved_depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {resolved_depth}")
    topic = rng.choice(_TOPICS)
    digits = f"{rng.randrange(10000):04d}"
    needle = f"The special magic number for {topic} is {digits}."

    filler_pool_tokens = [
        len(tokenizer(sentence, add_special_tokens=False)["input_ids"]) for sentence in _FILLER_SENTENCES
    ]
    needle_tokens = len(tokenizer(needle, add_special_tokens=False)["input_ids"])
    avg_filler_tokens = sum(filler_pool_tokens) / len(filler_pool_tokens)
    target_filler_sentences = max(1, round((context_length - needle_tokens) / avg_filler_tokens))
    needle_index = min(target_filler_sentences, round(resolved_depth * target_filler_sentences))

    sentences = [_FILLER_SENTENCES[i % len(_FILLER_SENTENCES)] for i in range(target_filler_sentences)]
    sentences.insert(needle_index, needle)
    context_text = " ".join(sentences)
    realized_depth = needle_index / target_filler_sentences if target_filler_sentences else 0.0
    return NiahExample(context_text=context_text, topic=topic, digits=digits, depth=realized_depth)


def build_niah_eval_examples(
    tokenizer,
    context_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
    *,
    examples_per_bin: int = 20,
    seed: int = 778,  # deliberately different default from DocSFTDataset's 777 - held-out documents, not the training ones
    fixed_depth: float | None = None,
) -> dict[str, list[NiahExample]]:
    """Build held-out NIAH eval examples grouped by context-length bin (mirrors
    `task_examples.py::build_all_task_examples`'s family -> list[TaskExample] shape, for
    `live_evaluator.py::DocumentHypernetworkDownstreamEvaluator` to group results by the
    same "family" concept the pooled-vector evaluator already uses). Family names are
    `f"niah_{length}"`, one per requested context length bin.
    """
    rng = random.Random(seed)
    examples_by_family: dict[str, list[NiahExample]] = {}
    for length in context_lengths:
        examples_by_family[f"niah_{length}"] = [
            make_niah_example(tokenizer, length, depth=fixed_depth, rng=rng) for _ in range(examples_per_bin)
        ]
    return examples_by_family


def build_query_prompt(tokenizer, topic: str) -> str:
    """Mirrors `lol_data.py::format_prompt_response`'s chat-template handling
    (`enable_thinking=False` matters for Qwen3 - see that function's docstring) for
    this dataset's own fixed query template. Kept as a standalone helper (rather than
    reusing `format_prompt_response` itself, which is parameterized around
    `task_def`/`problem`/`user_prompt_template` for Lots-of-LoRAs' schema) since NIAH's
    query has no task-definition/user-template structure to plug into that.
    """
    user_content = f"What is the special magic number for {topic}?"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@dataclass(frozen=True)
class DocSFTBatch:
    """Parallel to `sft_trainer.py::SFTBatch`: `context_input_ids`/`context_attention_mask`
    are the haystack+needle document (fed through `capture_document_activations` -
    never through the hypernetwork-hooked interpreter directly); `input_ids`/
    `attention_mask`/`labels` are the query+answer, response-only-supervised exactly
    like `SFTBatch`, fed through the hooked interpreter for the actual training loss.
    """

    context_input_ids: Tensor  # (batch, context_seq)
    context_attention_mask: Tensor  # (batch, context_seq)
    input_ids: Tensor  # (batch, seq)
    attention_mask: Tensor  # (batch, seq)
    labels: Tensor  # (batch, seq), -100 on prompt tokens and padding

    def to(self, device: torch.device | str) -> "DocSFTBatch":
        return DocSFTBatch(
            context_input_ids=self.context_input_ids.to(device),
            context_attention_mask=self.context_attention_mask.to(device),
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )


class DocSFTDataset(Dataset):
    """Freshly-generated (not read from disk) synthetic NIAH training examples: one
    haystack+needle document plus one query/answer pair each."""

    def __init__(
        self,
        tokenizer,
        *,
        num_examples: int,
        context_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
        max_query_len: int = 64,
        seed: int = 777,
        fixed_depth: float | None = None,
    ):
        self.tokenizer = tokenizer
        rng = random.Random(seed)
        self.examples: list[dict] = []
        for _ in range(num_examples):
            context_length = rng.choice(list(context_lengths))
            example = make_niah_example(tokenizer, context_length, depth=fixed_depth, rng=rng)
            # `truncation=True, max_length=context_length * 2` is a generous safety cap
            # (sentence-boundary insertion in make_niah_example already targets
            # `context_length` closely - see that function's docstring on realized vs.
            # requested depth/length) rather than the primary length control.
            context_ids = tokenizer(
                example.context_text, add_special_tokens=False, truncation=True, max_length=context_length * 2
            )["input_ids"]
            prompt = build_query_prompt(tokenizer, example.topic)
            response = " " + example.digits + (tokenizer.eos_token or "")
            tokenized_query = tokenize_prompt_response(tokenizer, prompt, response, max_query_len)
            self.examples.append(
                {
                    "context_input_ids": context_ids,
                    "context_attention_mask": [1] * len(context_ids),
                    **tokenized_query,
                    "topic": example.topic,
                    "digits": example.digits,
                    "depth": example.depth,
                    "context_length": context_length,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def _pad_to_batch(batch: list[dict], ids_key: str, mask_key: str, pad_token_id: int) -> tuple[Tensor, Tensor]:
    max_len = max(len(item[ids_key]) for item in batch)
    ids, mask = [], []
    for item in batch:
        pad_len = max_len - len(item[ids_key])
        ids.append(item[ids_key] + [pad_token_id] * pad_len)
        mask.append(item[mask_key] + [0] * pad_len)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def doc_collate_fn(batch: list[dict], pad_token_id: int) -> DocSFTBatch:
    context_input_ids, context_attention_mask = _pad_to_batch(
        batch, "context_input_ids", "context_attention_mask", pad_token_id
    )
    input_ids, attention_mask = _pad_to_batch(batch, "input_ids", "attention_mask", pad_token_id)
    max_len = input_ids.shape[1]
    labels = torch.stack(
        [
            torch.tensor(item["labels"] + [-100] * (max_len - len(item["labels"])), dtype=torch.long)
            for item in batch
        ]
    )
    return DocSFTBatch(
        context_input_ids=context_input_ids,
        context_attention_mask=context_attention_mask,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
