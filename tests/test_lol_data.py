import pytest
import torch
import yaml

from adapterbench.t2p.lol_data import (
    lol_collate_fn,
    load_decontaminated_train_task_ids,
    load_task_metadata,
    preprocess_lol_example,
    tokenize_prompt_response,
    validate_training_tasks,
)


def test_preprocess_lol_example_extracts_task_def_problem_and_answer():
    example = {
        "input": (
            "Definition: Say yes or no.\n\nPositive Example 1 -\nInput: foo\nOutput: bar\n\n"
            "Now complete the following example -\nInput: real problem text\nOutput:"
        ),
        "output": ["yes"],
    }
    result = preprocess_lol_example(example)
    assert result["task_def"] == "Say yes or no. Please complete the task without any explanation."
    assert result["problem"] == "real problem text"
    assert result["answer"] == "yes"


def test_preprocess_lol_example_notes_multiple_acceptable_answers():
    example = {
        "input": "Definition: X\n\nPositive Example 1 -\n\nNow complete the following example -\nInput: p\nOutput:",
        "output": ["a", "b"],
    }
    result = preprocess_lol_example(example)
    assert "comma-separated list" in result["task_def"]
    assert result["answer"] == "a, b"


class FakeTokenizer:
    eos_token = "</s>"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        tokens = text.split()
        if max_length is not None:
            tokens = tokens[:max_length]
        return {"input_ids": list(range(len(tokens)))}


def test_tokenize_prompt_response_masks_prompt_tokens():
    result = tokenize_prompt_response(FakeTokenizer(), "a b c", " d e", max_len=100)
    assert result["labels"][:3] == [-100, -100, -100]
    assert result["labels"][3:] == result["input_ids"][3:]
    assert len(result["input_ids"]) == len(result["labels"]) == len(result["attention_mask"])


def test_load_task_metadata_reads_yaml_and_truncates_descriptions(tmp_path):
    task_dir = tmp_path / "lol_999"
    task_dir.mkdir()
    (task_dir / "metadata.yaml").write_text(
        yaml.dump(
            {
                "ds_kwargs": {"path": "Lots-of-LoRAs/task999_example", "split": "train[:10]", "name": "default"},
                "descriptions": ["d0", "d1", "d2"],
                "user_prompt_template": "{task_def}\n\n{problem}",
                "response_field": "answer",
            }
        )
    )
    metadata = load_task_metadata(tmp_path, "lol_999", max_descriptions=2)
    assert metadata.dataset_id == "Lots-of-LoRAs/task999_example"
    assert metadata.split == "train[:10]"
    assert metadata.descriptions == ["d0", "d1"]


def _write_decontam_yaml(tmp_path):
    path = tmp_path / "decontam.yaml"
    path.write_text(
        yaml.dump(
            {
                "train_ds_names": ["lol_022", "lol_043", "lol_044"],
                "eval_ds_info": {"lol_035": {}, "lol_039": {}},
            }
        )
    )
    return path


def test_load_decontaminated_train_task_ids_reads_train_ds_names(tmp_path):
    path = _write_decontam_yaml(tmp_path)
    assert load_decontaminated_train_task_ids(path) == {"lol_022", "lol_043", "lol_044"}


def test_validate_training_tasks_accepts_tasks_in_the_decontaminated_list(tmp_path):
    path = _write_decontam_yaml(tmp_path)
    validate_training_tasks(["lol_022", "lol_044"], path)  # must not raise


def test_validate_training_tasks_rejects_held_out_validation_tasks(tmp_path):
    path = _write_decontam_yaml(tmp_path)
    with pytest.raises(ValueError, match="lol_035"):
        validate_training_tasks(["lol_022", "lol_035"], path)


def test_validate_training_tasks_rejects_contamination_removed_tasks(tmp_path):
    path = _write_decontam_yaml(tmp_path)
    with pytest.raises(ValueError, match="lol_033"):
        validate_training_tasks(["lol_033"], path)


def test_validate_training_tasks_ignores_non_lol_task_ids(tmp_path):
    path = _write_decontam_yaml(tmp_path)
    validate_training_tasks(["boolq", "arc_easy"], path)  # must not raise


def test_lol_collate_fn_pads_dynamically_and_stacks_condition_embeddings():
    batch = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [-100, -100, 3],
            "condition_embedding": torch.tensor([1.0, 2.0]),
        },
        {
            "input_ids": [4, 5],
            "attention_mask": [1, 1],
            "labels": [-100, 5],
            "condition_embedding": torch.tensor([3.0, 4.0]),
        },
    ]
    result = lol_collate_fn(batch, pad_token_id=0)
    assert result.input_ids.shape == (2, 3)
    assert result.input_ids[1].tolist() == [4, 5, 0]
    assert result.attention_mask[1].tolist() == [1, 1, 0]
    assert result.labels[1].tolist() == [-100, 5, -100]
    assert result.condition_embeddings.shape == (2, 2)
