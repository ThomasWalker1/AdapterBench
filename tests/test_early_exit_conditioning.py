"""Tests for the Doc-to-LoRA-parity early-exit + Perceiver-IO generation path
(EarlyExitPerceiverConditioner / capture_early_exit_representation) and the generic-needle
NIAH data it trains on. The mechanism (this path learns held-out NIAH retrieval where the
per-layer conditioner did not) is validated end-to-end by the d2a-niah run; these are the
unit-level shape/wiring/gradient checks."""
import random

import pytest
import torch
from torch import nn

from adapterbench.t2a.document_conditioning import (
    EarlyExitPerceiverConditioner,
    EarlyExitRepresentation,
    capture_early_exit_representation,
)
from adapterbench.t2a.hypernetwork import TextToPeftHypernetwork
from adapterbench.t2a.niah_data import (
    DocSFTDataset,
    build_generic_query_prompt,
    build_query_for_example,
    encode_context,
    make_niah_example,
)


class _Tok:
    eos_token = "</s>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        tokens = text.split()
        if max_length is not None:
            tokens = tokens[:max_length]
        return {"input_ids": list(range(len(tokens)))}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "<chat>" + messages[0]["content"]


def test_capture_early_exit_representation_uses_only_the_first_exit_layer_layers_and_restores():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    hidden_size, num_layers = 16, 4
    interpreter = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32, hidden_size=hidden_size, num_hidden_layers=num_layers, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=32, max_position_embeddings=32,
        )
    )
    for p in interpreter.parameters():
        p.requires_grad = False
    input_ids = torch.randint(0, 32, (2, 7))
    attention_mask = torch.ones(2, 7, dtype=torch.long)

    rep = capture_early_exit_representation(interpreter, input_ids, attention_mask, exit_layer=1)
    assert rep.shape == (2, 7, hidden_size)
    assert not rep.requires_grad
    # The interpreter's decoder-layer list must be restored to its full length afterwards.
    assert len(interpreter.model.layers) == num_layers
    # Early-exit at layer 1 must differ from the full-depth last hidden state.
    full = capture_early_exit_representation(interpreter, input_ids, attention_mask, exit_layer=num_layers)
    assert not torch.allclose(rep, full)


def test_early_exit_conditioner_shapes_and_layer_awareness():
    torch.manual_seed(0)
    hidden_size, num_layers, task_dim = 8, 3, 6
    cond = EarlyExitPerceiverConditioner(
        hidden_size=hidden_size, task_dim=task_dim, num_layers=num_layers, exit_layer=1,
        n_latents=5, num_blocks=2, latent_dim=8, num_heads=2,
    )
    rep = EarlyExitRepresentation(
        hidden_states=torch.randn(4, 5, hidden_size), attention_mask=torch.ones(4, 5, dtype=torch.long)
    )
    latents = cond.encode(rep)
    assert latents.shape == (4, 5, 8)
    summary = cond(latents, layer_index=None)
    assert summary.shape == (4, task_dim)
    per_layer_0 = cond(latents, layer_index=0)
    per_layer_2 = cond(latents, layer_index=2)
    assert per_layer_0.shape == (4, task_dim)
    assert not torch.allclose(per_layer_0, per_layer_2)  # distinct output queries per layer


def test_early_exit_padding_is_ignored_by_encode():
    torch.manual_seed(0)
    cond = EarlyExitPerceiverConditioner(
        hidden_size=8, task_dim=6, num_layers=2, exit_layer=1, n_latents=4, num_blocks=2, latent_dim=8, num_heads=2
    )
    cond.eval()
    hidden = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, dtype=torch.long)
    mask[:, 4:] = 0
    base = cond.encode(EarlyExitRepresentation(hidden, mask))
    corrupted = hidden.clone()
    corrupted[:, 4:, :] = torch.randn_like(corrupted[:, 4:, :]) * 100.0
    other = cond.encode(EarlyExitRepresentation(corrupted, mask))
    torch.testing.assert_close(base, other, atol=1e-4, rtol=1e-3)


class _TupleLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return (self.mlp.down_proj(hidden_states), "extra")


def test_early_exit_conditioner_wires_into_hypernetwork_and_is_differentiable():
    torch.manual_seed(0)
    hidden_size, num_layers, latent_dim = 8, 2, 32
    layers = nn.ModuleList([_TupleLayer(hidden_size) for _ in range(num_layers)])
    cond = EarlyExitPerceiverConditioner(
        hidden_size=hidden_size, task_dim=latent_dim // 2, num_layers=num_layers, exit_layer=1,
        n_latents=4, num_blocks=2, latent_dim=16, num_heads=2,
    )
    hyper = TextToPeftHypernetwork(
        module_shapes={"down_proj": (hidden_size, hidden_size)}, num_layers=num_layers,
        adapter="lora", latent_dim=latent_dim, head_dim=16, rank=2, conditioner=cond,
    )
    latents = cond.encode(
        EarlyExitRepresentation(torch.randn(3, 5, hidden_size), torch.ones(3, 5, dtype=torch.long))
    )
    generated = hyper.generate_per_layer(latents)
    inputs = torch.randn(3, 4, hidden_size)
    with hyper.apply(layers, generated):
        hidden, _ = layers[0](inputs)
        output, _ = layers[1](hidden)
    output.square().mean().backward()
    assert any(p.grad is not None for p in cond.parameters())


def test_make_niah_example_generic_style_has_topic_free_needle():
    ex = make_niah_example(_Tok(), context_length=200, rng=random.Random(0), needle_style="generic")
    assert ex.needle_style == "generic"
    assert ex.topic == ""
    assert f"The special magic number is {ex.digits}." in ex.context_text
    assert "magic number for" not in ex.context_text  # not the topic template


def test_make_niah_example_rejects_unknown_needle_style():
    with pytest.raises(ValueError, match="needle_style"):
        make_niah_example(_Tok(), context_length=100, needle_style="bogus")


def test_generic_query_is_topic_free_and_dispatch_matches_style():
    tok = _Tok()
    generic = make_niah_example(tok, 200, rng=random.Random(1), needle_style="generic")
    assert "for" not in build_generic_query_prompt(tok).replace("There and back", "")
    assert build_query_for_example(tok, generic) == build_generic_query_prompt(tok)


def test_encode_context_generic_is_chat_wrapped():
    tok = _Tok()
    ex = make_niah_example(tok, 200, rng=random.Random(2), needle_style="generic")
    # _Tok.apply_chat_template prefixes "<chat>", so a chat-wrapped context is longer than raw.
    assert encode_context(tok, ex)["input_ids"]  # non-empty


def test_doc_sft_dataset_generic_style_builds_valid_examples_and_records_style():
    tok = _Tok()
    ds = DocSFTDataset(tok, num_examples=3, context_lengths=[128], seed=0, needle_style="generic")
    assert len(ds) == 3
    for item in ds:
        assert item["needle_style"] == "generic"
        assert len(item["context_input_ids"]) == len(item["context_attention_mask"]) > 0
        assert len(item["input_ids"]) == len(item["labels"]) == len(item["attention_mask"])


def test_hypernetwork_requires_condition_dim_or_conditioner():
    with pytest.raises(ValueError, match="condition_dim"):
        TextToPeftHypernetwork(
            module_shapes={"q_proj": (8, 8)}, num_layers=2, adapter="lora", latent_dim=16, head_dim=16,
        )
