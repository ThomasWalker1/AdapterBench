"""Training data for live SFT: Super-NaturalInstructions via its per-task
``Lots-of-LoRAs/task*`` HF Hub mirror — confirmed to be the only SNI access path used
anywhere in Sakana's own Text-to-LoRA repo (no raw/original SNI dataset reference exists
there). Ports the exact preprocessing/tokenization convention their pipeline uses
(``hyper_llm_modulator/utils/preprocessing.py``, ``data.py``), scoped down per project
decision: point at the existing per-task ``tasks/*/metadata.yaml`` descriptions as-is (a
small prefix, not the paper's full 128) rather than reproducing how those descriptions
were generated.

TODO: Sakana's paper claims these descriptions are GPT-4o-mini-generated replacements
for SNI's own inconsistent default templates, but nothing in the cloned upstream repo
documents or implements that generation step (no provenance comment found anywhere) —
come back to this if reproducing the paper's exact description quality matters later.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset

from .sft_trainer import SFTBatch


@dataclass(frozen=True)
class LolTaskMetadata:
    task_id: str
    dataset_id: str
    split: str
    config_name: str | None
    descriptions: list[str]
    user_prompt_template: str
    response_field: str


def load_task_metadata(tasks_root: Path, task_id: str, *, max_descriptions: int | None = None) -> LolTaskMetadata:
    """Read ``tasks/<task_id>/metadata.yaml`` (already cloned under upstream/text-to-lora)."""
    payload = yaml.safe_load((tasks_root / task_id / "metadata.yaml").read_text())
    ds_kwargs = payload["ds_kwargs"]
    descriptions = payload["descriptions"]
    if max_descriptions is not None:
        descriptions = descriptions[:max_descriptions]
    return LolTaskMetadata(
        task_id=task_id,
        dataset_id=ds_kwargs["path"],
        split=ds_kwargs["split"],
        config_name=ds_kwargs.get("name"),
        descriptions=descriptions,
        user_prompt_template=payload["user_prompt_template"],
        response_field=payload["response_field"],
    )


def preprocess_lol_example(example: dict) -> dict:
    """Port of upstream ``preprocessing.py``'s ``lol_``-prefixed branch: the raw
    ``Lots-of-LoRAs/task###_...`` dataset's ``input`` column embeds the task definition
    and the actual problem behind fixed literal markers; ``output`` is a list of one or
    more acceptable answers."""
    raw_input = example["input"]
    task_def = raw_input.split("Definition: ")[1].split("\n\nPositive Example")[0].strip()
    task_def += " Please complete the task without any explanation."
    problem = (
        raw_input.split("Now complete the following example -")[1].split("Input: ", 1)[1].split("\nOutput:")[0]
    ).strip()
    outputs = example["output"]
    if len(outputs) > 1:
        task_def += "\nThe answer should be a comma-separated list of possible completions."
    return {"task_def": task_def, "problem": problem, "answer": ", ".join(outputs)}


def format_prompt_response(tokenizer, task_def: str, problem: str, answer: str, user_prompt_template: str) -> tuple[str, str]:
    """``enable_thinking=False`` matters for reasoning-capable interpreters (e.g. Qwen3):
    without it, ``add_generation_prompt=True`` leaves the model expected to emit its own
    ``<think>...</think>`` block before answering, so training it to emit ``answer``
    immediately after the prompt teaches it to always skip reasoning — an unintended
    training signal. Passing ``enable_thinking=False`` explicitly selects Qwen3's
    documented non-thinking mode (inserts an empty ``<think>\\n\\n</think>\\n\\n`` block into
    the prompt itself). Harmless no-op for tokenizers whose chat template doesn't
    reference ``enable_thinking`` (confirmed for Gemma-2/Mistral)."""
    user_content = user_prompt_template.format(task_def=task_def, problem=problem)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    response = " " + answer + (tokenizer.eos_token or "")
    return prompt, response


def tokenize_prompt_response(tokenizer, prompt: str, response: str, max_len: int) -> dict:
    """Response-only supervision (``label=-100`` on prompt tokens). Simplified from
    upstream's sequence-pair-tokenization + ``sequence_ids()`` masking (which depends on
    tokenizer-specific pair-tokenization behavior not guaranteed for every causal-LM
    tokenizer) to a plain length-based split: tokenize the prompt alone to find its
    token length, tokenize prompt+response as one string, mask the first that many
    positions. Equivalent supervision, more portable across tokenizers."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + response, add_special_tokens=False, truncation=True, max_length=max_len)[
        "input_ids"
    ]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class LolSFTDataset(Dataset):
    """One task's tokenized training examples, each paired with a randomly (train) or
    deterministically (eval) drawn description embedding — port of upstream's
    ``PerTaskEmbSFTDataset``."""

    def __init__(
        self,
        tokenizer,
        metadata: LolTaskMetadata,
        condition_embeddings: Tensor,
        *,
        max_len: int = 512,
        limit: int | None = None,
        training: bool = True,
    ):
        self.metadata = metadata
        self.condition_embeddings = condition_embeddings
        self.training = training
        raw = load_dataset(path=metadata.dataset_id, split=metadata.split, name=metadata.config_name)
        if limit is not None:
            raw = raw.select(range(min(limit, len(raw))))
        self.examples = []
        for row in raw:
            processed = preprocess_lol_example(row)
            prompt, response = format_prompt_response(
                tokenizer, processed["task_def"], processed["problem"], processed["answer"], metadata.user_prompt_template
            )
            self.examples.append(tokenize_prompt_response(tokenizer, prompt, response, max_len))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = dict(self.examples[index])
        num_descriptions = len(self.condition_embeddings)
        emb_index = random.randrange(num_descriptions) if self.training else index % num_descriptions
        example["condition_embedding"] = self.condition_embeddings[emb_index]
        example["task_id"] = self.metadata.task_id
        return example


def lol_collate_fn(batch: list[dict], pad_token_id: int) -> SFTBatch:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels, condition_embeddings = [], [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
        condition_embeddings.append(item["condition_embedding"])
    return SFTBatch(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        condition_embeddings=torch.stack(condition_embeddings),
    )
