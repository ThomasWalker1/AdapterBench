"""Synthetic needle-in-a-haystack (NIAH) data for document-conditioned live SFT
(the document-conditioning variant): a haystack of cycled filler sentences with
exactly one needle sentence inserted at a controlled depth, paired with a query/answer
about that needle. Reuses `lol_data.py::tokenize_prompt_response` for response-only
supervision on the query/answer side (that helper is a pure tokenizer/prompt/response
string utility with no Lots-of-LoRAs-specific content, so there is nothing
NIAH-specific to reimplement there).

Deliberately synthetic rather than read from any fixed on-disk
`ctx_magic_number_<lo>_<hi>` bins: this setting trains its own from-scratch
hypernetwork under a live SFT loop, which needs many fresh documents per training
step, not a fixed eval split.
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

# Doc-to-LoRA's own NIAH data format (upstream `data/generate_ctx_magic_number.py`): a
# single topic-free needle sentence, a fixed noise block repeated as filler, and a fixed
# topic-free query. This is the `needle_style="generic"` path - the D2L-parity format that
# empirically learns held-out NIAH retrieval (see PROJECT_PLAN.md's D2P section). The
# `"realistic"` style (below) reuses the same needle/query/scoring but swaps the repeated
# noise block for real prose. (An original topic-conditioned `"topic"` style was removed.)
_GENERIC_NOISE_BLOCK = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
_GENERIC_NEEDLE_TPL = "The special magic number is {digits}."
_GENERIC_QUERY = "What is the special magic number? Reply with only the number."

# needle_style values that share the D2L-parity query/answer/tokenization path (topic-free
# generic needle, chat-wrapped context, exact-digit answer). "realistic" differs from
# "generic" ONLY in the haystack: real English prose instead of a repeated noise block.
# Because the needle and answer stay the synthetic magic number, the exact-match scoring
# and the context-swap control remain exactly as clean as "generic".
_GENERIC_STYLES = ("generic", "realistic")

# Lazily built, process-cached pool of real sentences for needle_style="realistic".
_REALISTIC_POOL: tuple[tuple[str, ...], tuple[int, ...]] | None = None


def _realistic_filler_pool(tokenizer) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Build (once, cached) a pool of real English sentences to use as realistic NIAH
    haystack filler, with their tokenized lengths. Source: the `passage` field of BoolQ
    (real Wikipedia prose), read from the local HF cache (offline) — the same cache the
    rest of the pipeline already depends on, so no new download and nothing vendored. The
    pool is only *distractor* text; the needle/answer are still the synthetic magic number,
    so there is zero answer leakage and the control stays clean. Deterministic order (fixed
    shuffle seed), so documents are reproducible across runs. Assumes a single tokenizer per
    process (token counts are cached for it)."""
    global _REALISTIC_POOL
    if _REALISTIC_POOL is not None:
        return _REALISTIC_POOL
    import os
    import re as _re

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="train")
    seen: set[str] = set()
    sents: list[str] = []
    for passage in ds["passage"]:
        for s in _re.split(r"(?<=[.!?])\s+", passage.strip()):
            s = s.strip()
            if 30 <= len(s) <= 220 and s[0].isupper() and s.endswith((".", "!", "?")) and not s.isupper() and s not in seen:
                seen.add(s)
                sents.append(s)
    sents.sort()
    random.Random(0).shuffle(sents)
    sents = sents[:8000]
    counts = tuple(len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in sents)
    _REALISTIC_POOL = (tuple(sents), counts)
    return _REALISTIC_POOL


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
    needle_style: str = "generic"  # "generic" (D2L-parity) or "realistic" (real-prose haystack)


def make_niah_example(
    tokenizer,
    context_length: int,
    *,
    depth: float | None = None,
    rng: random.Random | None = None,
    needle_style: str = "generic",
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

    `needle_style` selects the haystack (both share the topic-free needle/query/answer, so
    the exact-match scoring and context-swap control are identical):
    - `"generic"` (default, Doc-to-LoRA parity): the needle ("The special magic number is
      {digits}.") among repeated copies of upstream's fixed noise block. The D2L-parity
      format that empirically learns held-out retrieval - see PROJECT_PLAN.md's D2P section.
    - `"realistic"`: the same needle among real English prose (BoolQ Wikipedia sentences),
      a genuine-distractor haystack instead of a repeated noise block.
    `topic` is always "" (a vestigial field from a removed topic-conditioned style).
    """
    rng = rng or random.Random()
    resolved_depth = rng.random() if depth is None else depth
    if not 0.0 <= resolved_depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {resolved_depth}")
    digits = f"{rng.randrange(10000):04d}"

    if needle_style == "generic":
        topic = ""
        needle = _GENERIC_NEEDLE_TPL.format(digits=digits)
        filler_block = _GENERIC_NOISE_BLOCK
        block_tokens = len(tokenizer(filler_block, add_special_tokens=False)["input_ids"])
        needle_tokens = len(tokenizer(needle, add_special_tokens=False)["input_ids"])
        target_blocks = max(1, round((context_length - needle_tokens) / max(block_tokens, 1)))
        needle_index = min(target_blocks, round(resolved_depth * target_blocks))
        blocks = [filler_block for _ in range(target_blocks)]
        blocks.insert(needle_index, needle)
        # Upstream joins blocks with a newline separator.
        context_text = "\n".join(blocks)
        realized_depth = needle_index / target_blocks if target_blocks else 0.0
        return NiahExample(
            context_text=context_text, topic=topic, digits=digits, depth=realized_depth, needle_style="generic"
        )

    if needle_style == "realistic":
        # Real-prose haystack + the same synthetic generic needle/answer as "generic".
        topic = ""
        needle = _GENERIC_NEEDLE_TPL.format(digits=digits)
        needle_tokens = len(tokenizer(needle, add_special_tokens=False)["input_ids"])
        pool, counts = _realistic_filler_pool(tokenizer)
        n = len(pool)
        target = max(1, context_length - needle_tokens)
        # Walk a contiguous window from a per-example random start through the (shuffled,
        # so consecutive entries are unrelated) pool, accumulating real sentences until the
        # token budget is met. rng-driven => deterministic per (seed, example).
        start = rng.randrange(n)
        idxs: list[int] = []
        total = 0
        i = 0
        while total < target and i < n:
            j = (start + i) % n
            idxs.append(j)
            total += counts[j]
            i += 1
        sentences = [pool[j] for j in idxs]
        num_filler = len(sentences)
        needle_index = min(num_filler, round(resolved_depth * num_filler))
        sentences.insert(needle_index, needle)
        context_text = " ".join(sentences)
        realized_depth = needle_index / num_filler if num_filler else 0.0
        return NiahExample(
            context_text=context_text, topic=topic, digits=digits, depth=realized_depth, needle_style="realistic"
        )

    raise ValueError(f"needle_style must be 'generic' or 'realistic', got {needle_style!r}")


def build_niah_eval_examples(
    tokenizer,
    context_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
    *,
    examples_per_bin: int = 20,
    seed: int = 778,  # deliberately different default from DocSFTDataset's 777 - held-out documents, not the training ones
    fixed_depth: float | None = None,
    needle_style: str = "generic",
) -> dict[str, list[NiahExample]]:
    """Build held-out NIAH eval examples grouped by context-length bin (mirrors
    `task_examples.py::build_all_task_examples`'s family -> list[TaskExample] shape, for
    `live_evaluator.py::DocumentHypernetworkDownstreamEvaluator` to group results by the
    same "family" concept the pooled-vector evaluator already uses). Family names are
    `f"niah_{length}"`, one per requested context length bin. `needle_style` is forwarded
    to `make_niah_example` (see it for "generic" vs "realistic").
    """
    rng = random.Random(seed)
    examples_by_family: dict[str, list[NiahExample]] = {}
    for length in context_lengths:
        examples_by_family[f"niah_{length}"] = [
            make_niah_example(tokenizer, length, depth=fixed_depth, rng=rng, needle_style=needle_style)
            for _ in range(examples_per_bin)
        ]
    return examples_by_family


def build_generic_query_prompt(tokenizer) -> str:
    """The topic-free upstream NIAH query, chat-wrapped (`enable_thinking=False` matters
    for Qwen3 - see `lol_data.py::format_prompt_response`). Used by both the `generic` and
    `realistic` needle styles (they share the same query/answer)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": _GENERIC_QUERY}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_query_for_example(tokenizer, example: "NiahExample") -> str:
    """The query prompt for an example - the one lookup both `DocSFTDataset` and
    `live_evaluator.DocumentHypernetworkDownstreamEvaluator` use so training and eval never
    build the query differently. Both shipped styles (`generic`, `realistic`) are
    topic-free, so this is always the generic query."""
    return build_generic_query_prompt(tokenizer)


def encode_context(tokenizer, example: "NiahExample", *, max_length: int | None = None) -> dict:
    """Tokenize a document's context so training and eval encode it identically: wrap the
    document as a chat user message (upstream feeds the context as a chat turn, D2L-parity)
    and add no extra special tokens on top of the template. Returns a dict with `input_ids`
    (and, for a real tokenizer, `attention_mask`).
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example.context_text}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    add_special = False
    kwargs = {"add_special_tokens": add_special}
    if max_length is not None:
        kwargs["truncation"] = True
        kwargs["max_length"] = max_length
    return tokenizer(text, **kwargs)


@dataclass(frozen=True)
class DocSFTBatch:
    """Parallel to `sft_trainer.py::SFTBatch`: `context_input_ids`/`context_attention_mask`
    are the haystack+needle document (fed through the conditioner-agnostic early-exit capture -
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
        needle_style: str = "generic",
    ):
        self.tokenizer = tokenizer
        rng = random.Random(seed)
        self.examples: list[dict] = []
        for _ in range(num_examples):
            context_length = rng.choice(list(context_lengths))
            example = make_niah_example(
                tokenizer, context_length, depth=fixed_depth, rng=rng, needle_style=needle_style
            )
            # `truncation=True, max_length=context_length * 2` is a generous safety cap
            # (sentence-boundary insertion in make_niah_example already targets
            # `context_length` closely - see that function's docstring on realized vs.
            # requested depth/length) rather than the primary length control.
            context_ids = encode_context(tokenizer, example, max_length=context_length * 2)["input_ids"]
            prompt = build_query_for_example(tokenizer, example)
            # Generic (D2L-parity) response is the bare number (matches upstream + the
            # validated probe); topic keeps its historical leading space.
            answer = example.digits if needle_style in _GENERIC_STYLES else " " + example.digits
            response = answer + (tokenizer.eos_token or "")
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
                    "needle_style": needle_style,
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
