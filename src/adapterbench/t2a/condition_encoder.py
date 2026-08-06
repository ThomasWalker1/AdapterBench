"""In-process task-description embedding via Alibaba-NLP/gte-large-en-v1.5.

Replicates upstream Text-to-LoRA's own embedding recipe
(``hyper_llm_modulator/utils/model_loading.py::get_emb_model_and_fns`` +
``pooling.py::cls_pool`` + ``preprocessing.py::add_full_stop``) so embeddings here are
comparable to what Text-to-LoRA's own checkpoints were conditioned on. This runs in-process:
gte-large-en-v1.5's config and tokenizer load cleanly under ``adapterbench``'s own
pinned ``transformers`` (only needs
``trust_remote_code=True``, no incompatible pin), so no cross-venv subprocess bridge is
needed here.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_CONDITION_ENCODER = "Alibaba-NLP/gte-large-en-v1.5"


def load_condition_encoder(
    model_id: str = DEFAULT_CONDITION_ENCODER, device: str = "cpu"
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model = AutoModel.from_pretrained(
        model_id, dtype=torch.float32, trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def _add_full_stop(text: str) -> str:
    text = text.strip()
    if text and text[-1].isalpha():
        text += "."
    return text


@torch.no_grad()
def embed_task_descriptions(
    descriptions: Sequence[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
) -> Tensor:
    """Return (len(descriptions), hidden_size) CLS-pooled embeddings, float32."""
    formatted = [_add_full_stop(text) for text in descriptions]
    device = next(model.parameters()).device
    embeddings = []
    for start in range(0, len(formatted), batch_size):
        batch = formatted[start : start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt", padding_side="right"
        ).to(device)
        outputs = model(**encoded)
        right_padding = encoded["attention_mask"][:, 0].sum() == encoded["attention_mask"].shape[0]
        if not right_padding:
            raise ValueError("CLS pooling requires right padding")
        embeddings.append(outputs.last_hidden_state[:, 0].float().cpu())
    return torch.cat(embeddings, dim=0)
