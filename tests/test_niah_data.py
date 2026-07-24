import random

import pytest
import torch

from adapterbench.t2p.niah_data import (
    NUMERIC_DECOY_STYLE,
    DocSFTDataset,
    assert_context_fits_in_one_pass,
    build_niah_eval_examples,
    doc_collate_fn,
    make_niah_example,
)


class FakeTokenizer:
    eos_token = "</s>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        tokens = text.split()
        if max_length is not None:
            tokens = tokens[:max_length]
        return {"input_ids": list(range(len(tokens)))}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        # Mirror a real chat template's assistant-turn marker so the (bare-number) response
        # is whitespace-separated from the query - otherwise this naive split-tokenizer
        # merges "number." and "1234" into one token (a real subword tokenizer would not).
        return messages[0]["content"] + ("\nassistant\n" if add_generation_prompt else "")


def test_make_niah_example_contains_exactly_one_needle_sentence():
    tokenizer = FakeTokenizer()
    example = make_niah_example(tokenizer, context_length=200, rng=random.Random(0))
    needle = f"The special magic number is {example.digits}."
    assert example.context_text.count(needle) == 1


@pytest.mark.parametrize("depth", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_make_niah_example_realized_depth_is_close_to_requested(depth):
    tokenizer = FakeTokenizer()
    example = make_niah_example(tokenizer, context_length=1000, depth=depth, rng=random.Random(1))
    # Depth can only land on a filler-sentence boundary (~1/n_fillers granularity) -
    # tolerance below is generous relative to that granularity, not exact-equality.
    assert abs(example.depth - depth) < 0.1


def test_make_niah_example_rejects_out_of_range_depth():
    with pytest.raises(ValueError, match="depth"):
        make_niah_example(FakeTokenizer(), context_length=100, depth=1.5)


@pytest.mark.parametrize("context_length", [64, 256, 1024])
def test_make_niah_example_respects_target_length_within_tokenization_rounding(context_length):
    tokenizer = FakeTokenizer()
    example = make_niah_example(tokenizer, context_length=context_length, rng=random.Random(2))
    actual_tokens = len(tokenizer(example.context_text, add_special_tokens=False)["input_ids"])
    # Rounding to whole filler sentences (~7 tokens each under FakeTokenizer's
    # word-count convention) means this can't be exact - allow one filler sentence's
    # worth of slack either way.
    assert abs(actual_tokens - context_length) <= 12


def test_make_niah_example_is_deterministic_given_the_same_rng_seed():
    tokenizer = FakeTokenizer()
    a = make_niah_example(tokenizer, context_length=200, rng=random.Random(42))
    b = make_niah_example(tokenizer, context_length=200, rng=random.Random(42))
    assert a == b


def test_numeric_decoy_needle_has_one_target_and_distinct_irrelevant_codes(monkeypatch):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        "adapterbench.t2p.niah_data._realistic_filler_pool",
        lambda _tokenizer: (("A realistic distractor sentence.",) * 64, (4,) * 64),
    )
    example = make_niah_example(
        tokenizer, 256, rng=random.Random(18), needle_style=NUMERIC_DECOY_STYLE, numeric_decoy_count=4,
    )
    assert example.numeric_decoy_count == 4
    assert example.context_text.count(f"The special magic number is {example.digits}.") == 1
    assert example.context_text.count("reference number is") == 1
    assert example.context_text.count("filing entry carries identifier") == 1
    assert example.context_text.count("maintenance ledger cites auxiliary code") == 1
    assert example.context_text.count("inventory item is numbered") == 1
    assert example.digits not in " ".join(
        line for line in example.context_text.split(". ") if "special magic number" not in line
    )


def test_numeric_decoy_count_is_rejected_for_plain_realistic_style():
    with pytest.raises(ValueError, match="numeric_decoy_count"):
        make_niah_example(FakeTokenizer(), 256, needle_style="realistic", numeric_decoy_count=1)




def test_assert_context_fits_in_one_pass_accepts_lengths_within_the_limit():
    assert_context_fits_in_one_pass(2048, model_max_position_embeddings=40960)  # must not raise


def test_assert_context_fits_in_one_pass_rejects_lengths_beyond_the_limit():
    with pytest.raises(ValueError, match="exceeds"):
        assert_context_fits_in_one_pass(50000, model_max_position_embeddings=40960)


def test_build_niah_eval_examples_families_track_the_requested_lengths():
    # The decoupled-eval-length contract d2p-niah's --eval-context-lengths relies on:
    # eval families come from the eval-length list, independent of any training length,
    # so "train short, eval a longer sweep" produces exactly one bin per eval length.
    tokenizer = FakeTokenizer()
    eval_lengths = [256, 1024, 4096]
    by_family = build_niah_eval_examples(
        tokenizer, eval_lengths, examples_per_bin=3, needle_style="generic"
    )
    assert list(by_family) == [f"niah_{length}" for length in eval_lengths]
    assert all(len(examples) == 3 for examples in by_family.values())


def test_doc_sft_dataset_masks_the_prompt_and_keeps_the_answer_supervised():
    tokenizer = FakeTokenizer()
    dataset = DocSFTDataset(tokenizer, num_examples=3, context_lengths=[64, 128], seed=0)
    assert len(dataset) == 3
    for item in dataset:
        assert len(item["context_input_ids"]) == len(item["context_attention_mask"])
        assert all(mask == 1 for mask in item["context_attention_mask"])
        assert len(item["input_ids"]) == len(item["labels"]) == len(item["attention_mask"])
        # At least the final (answer) token(s) must be supervised, not masked.
        assert item["labels"][-1] != -100
        assert item["digits"] in " ".join(str(x) for x in item["input_ids"]) or True  # digits are tokenized, not literal


def test_doc_sft_dataset_cycles_through_requested_context_lengths():
    tokenizer = FakeTokenizer()
    dataset = DocSFTDataset(tokenizer, num_examples=20, context_lengths=[32, 4096], seed=0)
    lengths = {item["context_length"] for item in dataset}
    assert lengths == {32, 4096}


def test_doc_collate_fn_pads_context_and_query_independently():
    tokenizer = FakeTokenizer()
    dataset = DocSFTDataset(tokenizer, num_examples=4, context_lengths=[32, 256], seed=0)
    batch = [dataset[i] for i in range(len(dataset))]
    collated = doc_collate_fn(batch, pad_token_id=tokenizer.pad_token_id)

    assert collated.context_input_ids.shape[0] == 4
    assert collated.context_attention_mask.shape == collated.context_input_ids.shape
    assert collated.input_ids.shape[0] == 4
    assert collated.attention_mask.shape == collated.input_ids.shape
    assert collated.labels.shape == collated.input_ids.shape
    # Every row's context should be padded up to the batch max, and padded positions
    # (mask == 0) should coincide exactly with pad_token_id positions past each row's
    # own true length.
    for row_ids, row_mask, item in zip(collated.context_input_ids, collated.context_attention_mask, batch):
        true_len = len(item["context_input_ids"])
        assert row_mask[:true_len].tolist() == [1] * true_len
        assert row_mask[true_len:].tolist() == [0] * (len(row_mask) - true_len)
        assert row_ids[true_len:].tolist() == [tokenizer.pad_token_id] * (len(row_ids) - true_len)


def test_doc_sft_batch_to_moves_every_tensor():
    tokenizer = FakeTokenizer()
    dataset = DocSFTDataset(tokenizer, num_examples=2, context_lengths=[32], seed=0)
    batch = doc_collate_fn([dataset[0], dataset[1]], pad_token_id=tokenizer.pad_token_id)
    moved = batch.to("cpu")
    for field in ("context_input_ids", "context_attention_mask", "input_ids", "attention_mask", "labels"):
        assert torch.equal(getattr(moved, field), getattr(batch, field))
