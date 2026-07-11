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

# Doc-to-LoRA's own NIAH data format (upstream `data/generate_ctx_magic_number.py`): a
# single generic needle sentence (no topic), a fixed noise block repeated as filler, and
# a fixed topic-free query. This is the `needle_style="generic"` path, distinct from the
# original topic-conditioned `needle_style="topic"` above. The generic style is the one
# that empirically learns held-out NIAH retrieval in the D2L-parity recipe (see
# PROJECT_PLAN.md's D2P section); the topic style is retained for back-compat and because
# every existing niah_data test asserts against it.
_GENERIC_NOISE_BLOCK = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
_GENERIC_NEEDLE_TPL = "The special magic number is {digits}."
_GENERIC_QUERY = "What is the special magic number? Reply with only the number."


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
    needle_style: str = "topic"  # "topic" (original) or "generic" (D2L-parity) - see make_niah_example


def make_niah_example(
    tokenizer,
    context_length: int,
    *,
    depth: float | None = None,
    rng: random.Random | None = None,
    needle_style: str = "topic",
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

    `needle_style` selects the data format:
    - `"topic"` (default, original): one topic-tagged needle ("The special magic number
      for {topic} is {digits}.") among a cycled pool of short filler sentences; the query
      names that topic.
    - `"generic"` (Doc-to-LoRA parity): one topic-free needle ("The special magic number
      is {digits}.") among repeated copies of upstream's fixed noise block; the query is
      topic-free. `topic` is set to "" (unused). This is the format the D2L-parity recipe
      retrieves under - see PROJECT_PLAN.md's D2P section.
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

    if needle_style != "topic":
        raise ValueError(f"needle_style must be 'topic' or 'generic', got {needle_style!r}")
    topic = rng.choice(_TOPICS)
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
    needle_style: str = "topic",
) -> dict[str, list[NiahExample]]:
    """Build held-out NIAH eval examples grouped by context-length bin (mirrors
    `task_examples.py::build_all_task_examples`'s family -> list[TaskExample] shape, for
    `live_evaluator.py::DocumentHypernetworkDownstreamEvaluator` to group results by the
    same "family" concept the pooled-vector evaluator already uses). Family names are
    `f"niah_{length}"`, one per requested context length bin. `needle_style` is forwarded
    to `make_niah_example` (see it for "topic" vs "generic").
    """
    rng = random.Random(seed)
    examples_by_family: dict[str, list[NiahExample]] = {}
    for length in context_lengths:
        examples_by_family[f"niah_{length}"] = [
            make_niah_example(tokenizer, length, depth=fixed_depth, rng=rng, needle_style=needle_style)
            for _ in range(examples_per_bin)
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


def build_generic_query_prompt(tokenizer) -> str:
    """`needle_style="generic"` (D2L-parity) counterpart to `build_query_prompt`: the
    topic-free upstream query, chat-wrapped identically (`enable_thinking=False`)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": _GENERIC_QUERY}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_query_for_example(tokenizer, example: "NiahExample") -> str:
    """Dispatch the right query prompt for an example's `needle_style` - the one lookup
    both `DocSFTDataset` and `live_evaluator.DocumentHypernetworkDownstreamEvaluator` use
    so training and eval never build the query differently for a given style."""
    if example.needle_style == "generic":
        return build_generic_query_prompt(tokenizer)
    return build_query_prompt(tokenizer, example.topic)


def encode_context(tokenizer, example: "NiahExample", *, max_length: int | None = None) -> dict:
    """Tokenize a document's context the way its `needle_style` expects, so training and
    eval encode it identically. `"generic"` (D2L-parity) wraps the document as a chat user
    message (upstream feeds the context as a chat turn) and adds no extra special tokens
    on top of the template; `"topic"` tokenizes the raw context as-is (original behavior).
    Returns a dict with `input_ids` (and, for a real tokenizer, `attention_mask`).
    """
    if example.needle_style == "generic":
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": example.context_text}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        add_special = False
    else:
        text = example.context_text
        add_special = False
    kwargs = {"add_special_tokens": add_special}
    if max_length is not None:
        kwargs["truncation"] = True
        kwargs["max_length"] = max_length
    return tokenizer(text, **kwargs)


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
        needle_style: str = "topic",
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
            answer = example.digits if needle_style == "generic" else " " + example.digits
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
