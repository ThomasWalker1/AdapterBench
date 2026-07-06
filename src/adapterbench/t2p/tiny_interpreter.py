"""A tiny, freshly initialized real ``transformers`` causal LM for the lightweight
synthetic setting — no download, no pretrained checkpoint, no network dependency
(``LlamaForCausalLM(config)``, not ``.from_pretrained(...)``). Llama's decoder-layer
submodule naming (``self_attn.{q,k,v,o}_proj``, ``mlp.{gate,up,down}_proj``) matches what
``hypernetwork.py::_resolve_target``/``cli.py::_PILOT_DEFAULT_TARGET_MODULES`` already
expect, and it has no "thinking mode" chat-template behavior to worry about (see
``PROJECT_PLAN.md`` gotcha #18) — irrelevant here anyway since the synthetic setting never
tokenizes natural language for the interpreter, only fixed integer symbol ids.
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

# Fixed 16-symbol vocabulary shared with synthetic_tasks.py: PAD/BOS/EOS/SEP plus digits
# 0-9 (2 ids reserved). No HF tokenizer is needed anywhere in this setting.
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3
DIGIT_BASE = 4
VOCAB_SIZE = 16


def build_tiny_interpreter(
    *,
    vocab_size: int = VOCAB_SIZE,
    hidden_size: int = 32,
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    intermediate_size: int = 64,
    max_position_embeddings: int = 32,
    seed: int = 777,
) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=PAD_ID,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
    )
    return LlamaForCausalLM(config)
