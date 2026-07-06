"""Build small TaskExample sets for the Text-to-LoRA Gemma-2-2B benchmark tasks.

Condition text (used only to condition the hypernetwork's generation, never shown
to the interpreter at eval time - matching upstream Text-to-LoRA's default
``system_message=""`` evaluation mode) is sourced from a released checkpoint's own
``args.yaml`` (``eval_ds_info``), so generation uses the exact task-description
phrasing the released hypernetwork was evaluated against upstream. One description
variant per task is used, chosen deterministically, and recorded in each example's
metadata.
"""

from __future__ import annotations

from pathlib import Path
import re

import yaml

from .contracts import TaskExample

# (dataset_id, config_name, split)
DATASET_SOURCES: dict[str, tuple[str, str | None, str]] = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "boolq": ("google/boolq", None, "validation"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
}

_GSM8K_ANSWER_RE = re.compile(r"####\s*([\-0-9,\.]+)")


def load_task_descriptions(args_yaml_path: str | Path) -> dict[str, list[str]]:
    payload = yaml.safe_load(Path(args_yaml_path).read_text())
    return {task: info["descriptions"] for task, info in payload["eval_ds_info"].items()}


def build_conditions(
    descriptions: dict[str, list[str]], task_ids: list[str], variant: int = 0
) -> dict[str, str]:
    return {task_id: descriptions[task_id][variant] for task_id in task_ids}


def _arc_examples(task_id: str, condition: str, rows, limit: int, variant: int) -> list[TaskExample]:
    examples = []
    for i, row in enumerate(rows):
        if len(examples) >= limit:
            break
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        try:
            answer_index = labels.index(row["answerKey"])
        except ValueError:
            continue
        examples.append(
            TaskExample(
                task_id=f"{task_id}::{i}",
                condition=condition,
                input_text=row["question"],
                target_text=choices[answer_index],
                family=task_id,
                metadata={"choices": choices, "answer_index": answer_index, "condition_variant": variant},
            )
        )
    return examples


def _boolq_examples(task_id: str, condition: str, rows, limit: int, variant: int) -> list[TaskExample]:
    examples = []
    for i, row in enumerate(rows):
        if len(examples) >= limit:
            break
        answer_index = 1 if row["answer"] else 0
        examples.append(
            TaskExample(
                task_id=f"{task_id}::{i}",
                condition=condition,
                input_text=f"{row['passage']}\n\nQuestion: {row['question']}",
                target_text=["no", "yes"][answer_index],
                family=task_id,
                metadata={"choices": ["no", "yes"], "answer_index": answer_index, "condition_variant": variant},
            )
        )
    return examples


def _hellaswag_examples(task_id: str, condition: str, rows, limit: int, variant: int) -> list[TaskExample]:
    examples = []
    for i, row in enumerate(rows):
        if len(examples) >= limit:
            break
        choices = row["endings"]
        try:
            answer_index = int(row["label"])
        except (TypeError, ValueError):
            continue
        if not (0 <= answer_index < len(choices)):
            continue
        examples.append(
            TaskExample(
                task_id=f"{task_id}::{i}",
                condition=condition,
                input_text=row["ctx"],
                target_text=choices[answer_index],
                family=task_id,
                metadata={"choices": choices, "answer_index": answer_index, "condition_variant": variant},
            )
        )
    return examples


def _gsm8k_examples(task_id: str, condition: str, rows, limit: int, variant: int) -> list[TaskExample]:
    examples = []
    for i, row in enumerate(rows):
        if len(examples) >= limit:
            break
        match = _GSM8K_ANSWER_RE.search(row["answer"])
        if not match:
            continue
        examples.append(
            TaskExample(
                task_id=f"{task_id}::{i}",
                condition=condition,
                input_text=row["question"],
                target_text=match.group(1).replace(",", ""),
                family=task_id,
                metadata={"condition_variant": variant},
            )
        )
    return examples


_BUILDERS = {
    "arc_easy": _arc_examples,
    "arc_challenge": _arc_examples,
    "boolq": _boolq_examples,
    "hellaswag": _hellaswag_examples,
    "gsm8k": _gsm8k_examples,
}


def build_task_examples(
    task_id: str, descriptions: dict[str, list[str]], limit: int, variant: int = 0
) -> list[TaskExample]:
    from datasets import load_dataset

    if task_id not in DATASET_SOURCES:
        raise ValueError(f"unknown task_id: {task_id}")
    condition = descriptions[task_id][variant]
    dataset_id, config, split = DATASET_SOURCES[task_id]
    rows = load_dataset(dataset_id, config, split=split) if config else load_dataset(dataset_id, split=split)
    return _BUILDERS[task_id](task_id, condition, rows, limit, variant)


def build_all_task_examples(
    task_ids: list[str], descriptions: dict[str, list[str]], limit: int, variant: int = 0
) -> dict[str, list[TaskExample]]:
    return {task_id: build_task_examples(task_id, descriptions, limit, variant) for task_id in task_ids}
