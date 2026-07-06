import torch

from peft_hnet.t2p.lol_data import lol_collate_fn
from peft_hnet.t2p.synthetic_tasks import (
    TASK_FAMILIES,
    SyntheticSFTDataset,
    build_prompt_ids,
    build_target_ids,
    encode_example,
)
from peft_hnet.t2p.tiny_interpreter import BOS_ID, DIGIT_BASE, EOS_ID, SEP_ID


def test_family_transforms_are_exactly_known():
    digits = [3, 1, 4, 1, 5]
    assert TASK_FAMILIES["copy"].transform(digits) == [3, 1, 4, 1, 5]
    assert TASK_FAMILIES["reverse"].transform(digits) == [5, 1, 4, 1, 3]
    assert TASK_FAMILIES["increment"].transform([9, 0, 5]) == [0, 1, 6]
    assert TASK_FAMILIES["sort"].transform(digits) == [1, 1, 3, 4, 5]
    assert TASK_FAMILIES["constant"].transform(digits) == [9, 9, 9, 9, 9]


def test_every_family_has_multiple_description_paraphrases():
    for family in TASK_FAMILIES.values():
        assert len(family.descriptions) >= 4
        assert len(set(family.descriptions)) == len(family.descriptions)


def test_build_prompt_and_target_ids_use_the_fixed_symbol_vocabulary():
    prompt = build_prompt_ids([1, 2])
    assert prompt == [BOS_ID, DIGIT_BASE + 1, DIGIT_BASE + 2, SEP_ID]
    target = build_target_ids([3, 4])
    assert target == [DIGIT_BASE + 3, DIGIT_BASE + 4, EOS_ID]


def test_encode_example_masks_prompt_and_supervises_target_only():
    example = encode_example([1, 2], [2, 1])
    prompt_len = len(build_prompt_ids([1, 2]))
    assert example["labels"][:prompt_len] == [-100] * prompt_len
    assert example["labels"][prompt_len:] == build_target_ids([2, 1])
    assert example["input_ids"] == build_prompt_ids([1, 2]) + build_target_ids([2, 1])
    assert len(example["attention_mask"]) == len(example["input_ids"])


def test_synthetic_sft_dataset_training_draws_random_description_embedding():
    embeddings = torch.eye(3)
    dataset = SyntheticSFTDataset("copy", embeddings, size=50, training=True, seed=0)
    drawn = {tuple(dataset[i]["condition_embedding"].tolist()) for i in range(50)}
    assert len(drawn) > 1  # random draw actually varies across examples


def test_synthetic_sft_dataset_eval_draws_deterministic_description_embedding():
    embeddings = torch.eye(3)
    dataset = SyntheticSFTDataset("copy", embeddings, size=6, training=False, seed=0)
    for index in range(6):
        expected = embeddings[index % 3]
        assert torch.equal(dataset[index]["condition_embedding"], expected)


def test_synthetic_sft_dataset_is_reproducible_across_instances():
    embeddings = torch.eye(2)
    a = SyntheticSFTDataset("sort", embeddings, size=10, training=True, seed=42)
    b = SyntheticSFTDataset("sort", embeddings, size=10, training=True, seed=42)
    for i in range(10):
        assert a[i]["input_ids"] == b[i]["input_ids"]
        assert a[i]["labels"] == b[i]["labels"]


def test_synthetic_sft_dataset_collates_via_lol_collate_fn():
    embeddings = torch.eye(4)
    dataset = SyntheticSFTDataset("reverse", embeddings, seq_len=5, size=8, training=True, seed=1)
    batch = [dataset[i] for i in range(8)]
    result = lol_collate_fn(batch, pad_token_id=0)
    assert result.condition_embeddings.shape == (8, 4)
    assert result.input_ids.shape[0] == 8
