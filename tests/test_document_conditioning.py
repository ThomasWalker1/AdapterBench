import pytest
import torch
from torch import nn

from adapterbench.t2p.document_conditioning import (
    DocumentActivations,
    DocumentPerceiverConditioner,
    capture_document_activations,
)
from adapterbench.t2p.hypernetwork import TextToPeftHypernetwork


def _random_activations(batch, num_layers, seq_len, hidden_size, pad_from=None):
    hidden_states = torch.randn(batch, num_layers, seq_len, hidden_size)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    if pad_from is not None:
        attention_mask[:, pad_from:] = 0
    return DocumentActivations(hidden_states=hidden_states, attention_mask=attention_mask)


def test_forward_with_layer_index_none_returns_pooled_batch_by_task_dim():
    torch.manual_seed(0)
    conditioner = DocumentPerceiverConditioner(hidden_size=8, task_dim=6, num_layers=3, latent_dim=8)
    raw = _random_activations(batch=4, num_layers=3, seq_len=5, hidden_size=8)
    out = conditioner(raw, layer_index=None)
    assert out.shape == (4, 6)
    assert torch.isfinite(out).all()


def test_forward_layer_returns_pooled_batch_by_task_dim():
    torch.manual_seed(0)
    conditioner = DocumentPerceiverConditioner(hidden_size=8, task_dim=6, num_layers=3, latent_dim=8)
    raw = _random_activations(batch=4, num_layers=3, seq_len=5, hidden_size=8)
    out = conditioner(raw, layer_index=1)
    assert out.shape == (4, 6)
    assert torch.isfinite(out).all()


def test_forward_layer_uses_that_layers_own_activations_not_a_shared_summary():
    """Layer-index-awareness check: forward_layer(x, i) for two different layers with
    different underlying token activations should generally produce different vectors -
    unlike PooledVectorConditioner, which is byte-identical regardless of layer_index."""
    torch.manual_seed(0)
    conditioner = DocumentPerceiverConditioner(hidden_size=8, task_dim=6, num_layers=3, latent_dim=8)
    raw = _random_activations(batch=4, num_layers=3, seq_len=5, hidden_size=8)
    out_layer_0 = conditioner(raw, layer_index=0)
    out_layer_2 = conditioner(raw, layer_index=2)
    assert not torch.allclose(out_layer_0, out_layer_2)


def test_padded_positions_are_ignored():
    torch.manual_seed(0)
    conditioner = DocumentPerceiverConditioner(hidden_size=8, task_dim=6, num_layers=2, latent_dim=8)
    conditioner.eval()  # disable dropout - this test checks masking, not stochastic regularization
    raw = _random_activations(batch=2, num_layers=2, seq_len=6, hidden_size=8, pad_from=4)
    baseline = conditioner(raw, layer_index=0)

    # Corrupt only the padded positions (index 4:) - the mask should make this a no-op.
    corrupted_hidden = raw.hidden_states.clone()
    corrupted_hidden[:, :, 4:, :] = torch.randn_like(corrupted_hidden[:, :, 4:, :]) * 100.0
    corrupted = DocumentActivations(hidden_states=corrupted_hidden, attention_mask=raw.attention_mask)
    corrupted_out = conditioner(corrupted, layer_index=0)

    torch.testing.assert_close(baseline, corrupted_out, atol=1e-5, rtol=1e-4)


def test_capture_document_activations_shape_and_no_grad_leak_into_interpreter():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    hidden_size, num_layers = 16, 3
    interpreter = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32, hidden_size=hidden_size, num_hidden_layers=num_layers, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=32, max_position_embeddings=32,
        )
    )
    for parameter in interpreter.parameters():
        parameter.requires_grad = False

    input_ids = torch.randint(0, 32, (2, 7))
    attention_mask = torch.ones(2, 7, dtype=torch.long)
    activations = capture_document_activations(interpreter, input_ids, attention_mask)

    assert activations.shape == (2, num_layers, 7, hidden_size)
    assert not activations.requires_grad

    conditioner = DocumentPerceiverConditioner(hidden_size=hidden_size, task_dim=6, num_layers=num_layers, latent_dim=8)
    raw = DocumentActivations(hidden_states=activations, attention_mask=attention_mask)
    out = conditioner(raw, layer_index=None)
    out.square().mean().backward()

    assert any(p.grad is not None for p in conditioner.parameters())
    assert all(p.grad is None for p in interpreter.parameters())


class TupleReturningDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return (self.linear(hidden_states), "extra")


def test_document_conditioner_wired_into_hypernetwork_and_hooked():
    """Analogous to test_dynamic_t2p.py's TupleReturningDecoderLayer case, but with the
    document conditioner instead of a plain pooled-vector Tensor - the whole
    conditioner -> trunk -> heads -> hook chain must remain differentiable end to end,
    and gradients must reach the conditioner's own parameters."""
    torch.manual_seed(0)
    hidden_size, num_layers = 8, 2
    layers = nn.ModuleList([TupleReturningDecoderLayer(hidden_size) for _ in range(num_layers)])
    latent_dim = 32
    task_dim = latent_dim // 2  # must match TextToPeftHypernetwork's own task_dim split - see its __init__
    conditioner = DocumentPerceiverConditioner(
        hidden_size=hidden_size, task_dim=task_dim, num_layers=num_layers, latent_dim=8
    )
    hypernetwork = TextToPeftHypernetwork(
        module_shapes={"block": (hidden_size, hidden_size)},
        num_layers=num_layers,
        adapter="activation_steering",
        latent_dim=latent_dim,
        head_dim=32,
        conditioner=conditioner,
    )
    nn.init.normal_(hypernetwork.heads["block"].weight, std=0.01)

    raw = _random_activations(batch=3, num_layers=num_layers, seq_len=4, hidden_size=hidden_size)
    inputs = torch.randn(3, 4, hidden_size)
    generated = hypernetwork(raw)
    with hypernetwork.apply(layers, generated):
        hidden, extra = layers[0](inputs)
        output, extra2 = layers[1](hidden)
    assert extra == "extra" and extra2 == "extra"

    output.square().mean().backward()
    assert hypernetwork.heads["block"].weight.grad is not None
    assert any(p.grad is not None for p in conditioner.parameters())


def test_generate_per_layer_outputs_differ_meaningfully_across_layers():
    """Regression test for the pooled-broadcast gap: `TextToPeftHypernetwork.forward`
    (layer_index=None) mean-pools DocumentPerceiverConditioner's per-layer cross-attention
    into one summary vector broadcast to every layer, so `forward(raw)[name]` is nearly
    identical across the layer dimension (differing only via `depth_embedding`).
    `generate_per_layer` must instead route each layer through its own
    `forward_layer(raw, layer_index=i)` call, so outputs for genuinely different
    per-layer document activations must differ far more than the broadcast case does."""
    torch.manual_seed(0)
    hidden_size, num_layers, latent_dim = 8, 3, 32
    task_dim = latent_dim // 2
    conditioner = DocumentPerceiverConditioner(
        hidden_size=hidden_size, task_dim=task_dim, num_layers=num_layers, latent_dim=8
    )
    hypernetwork = TextToPeftHypernetwork(
        module_shapes={"q_proj": (hidden_size, hidden_size)},
        num_layers=num_layers,
        adapter="lora",
        latent_dim=latent_dim,
        head_dim=32,
        rank=2,
        conditioner=conditioner,
    )
    # The head is zero-initialized by default (see TextToPeftHypernetwork.__init__'s
    # `nn.init.zeros_(heads[name].weight)`) - with a zero weight, the head's output is
    # just its (layer-invariant) bias regardless of what the trunk/conditioner produce,
    # which would make both `broadcast` and `per_layer` trivially layer-invariant too and
    # defeat this test's purpose. Randomize it, same as
    # test_document_conditioner_wired_into_hypernetwork_and_hooked does.
    nn.init.normal_(hypernetwork.heads["q_proj"].weight, std=0.05)
    # `depth_embedding` is itself layer-varying (independent of the conditioner) and would
    # otherwise contaminate a layer-to-layer comparison of `forward` vs `generate_per_layer`
    # with a source of difference neither path's conditioning is responsible for. Zero its
    # embedding weight so any remaining layer-to-layer difference is attributable only to
    # the conditioner: `nn.LayerNorm` on an all-zero vector is itself all-zero (mean/std
    # both 0, so the normalized numerator is 0 regardless of `eps`), so this makes
    # `depth_embedding`'s output identically zero for every layer index.
    with torch.no_grad():
        hypernetwork.depth_embedding[0].weight.zero_()
    # Distinct per-layer token activations (not just noise around one shared value) so a
    # layer-faithful conditioner has genuinely different signal to pick up per layer.
    hidden_states = torch.stack(
        [torch.full((2, 5, hidden_size), float(layer)) + 0.01 * torch.randn(2, 5, hidden_size) for layer in range(num_layers)],
        dim=1,
    )
    raw = DocumentActivations(hidden_states=hidden_states, attention_mask=torch.ones(2, 5, dtype=torch.long))

    hypernetwork.eval()  # disable dropout so this comparison isn't confounded by dropout noise
    broadcast = hypernetwork(raw)["q_proj"]  # (num_layers, batch, output_size) - forward(), layer_index=None
    per_layer = hypernetwork.generate_per_layer(raw)["q_proj"]  # same shape, via forward_layer loop

    assert per_layer.shape == broadcast.shape == (num_layers, 2, hypernetwork.heads["q_proj"].out_features)

    # forward()'s mean-pooled-broadcast path: with depth_embedding zeroed out, every layer
    # now shares byte-identical conditioning (same pooled task vector, same zero depth, same
    # type vector) - so its output is exactly layer-invariant, the concrete manifestation of
    # the documented gap this fix addresses.
    torch.testing.assert_close(broadcast[0], broadcast[-1])
    # generate_per_layer's path: each layer conditions on its own document activations via
    # forward_layer - with genuinely different per-layer hidden_states, this must NOT be
    # layer-invariant.
    assert not torch.allclose(per_layer[0], per_layer[-1], atol=1e-3)
    per_layer_layer_gap = (per_layer[0] - per_layer[-1]).abs().mean().item()
    assert per_layer_layer_gap > 1e-3


def test_generate_per_layer_matches_forward_layer_and_hypernetwork_apply_hooks_correctly():
    """`generate_per_layer`'s assembled dict must be a drop-in replacement for
    `forward`'s: each `generated[name][i]` must equal `forward_layer(raw, i)[name]`
    exactly, and `hypernetwork.apply(layers, generated)` must hook it correctly into a
    real forward pass (same TupleReturningDecoderLayer pattern as
    test_document_conditioner_wired_into_hypernetwork_and_hooked, and
    test_dynamic_t2p.py's own TupleReturningDecoderLayer case)."""
    torch.manual_seed(0)
    hidden_size, num_layers = 8, 2
    layers = nn.ModuleList([TupleReturningDecoderLayer(hidden_size) for _ in range(num_layers)])
    latent_dim = 32
    task_dim = latent_dim // 2
    conditioner = DocumentPerceiverConditioner(
        hidden_size=hidden_size, task_dim=task_dim, num_layers=num_layers, latent_dim=8
    )
    hypernetwork = TextToPeftHypernetwork(
        module_shapes={"block": (hidden_size, hidden_size)},
        num_layers=num_layers,
        adapter="activation_steering",
        latent_dim=latent_dim,
        head_dim=32,
        conditioner=conditioner,
    )
    nn.init.normal_(hypernetwork.heads["block"].weight, std=0.01)
    hypernetwork.eval()  # disable dropout so forward_layer(raw, i) is exactly reproducible per call

    raw = _random_activations(batch=3, num_layers=num_layers, seq_len=4, hidden_size=hidden_size)
    generated = hypernetwork.generate_per_layer(raw)

    for layer_index in range(num_layers):
        expected = hypernetwork.forward_layer(raw, layer_index)["block"]
        torch.testing.assert_close(generated["block"][layer_index], expected)

    inputs = torch.randn(3, 4, hidden_size)
    with hypernetwork.apply(layers, generated):
        hidden, extra = layers[0](inputs)
        output, extra2 = layers[1](hidden)
    assert extra == "extra" and extra2 == "extra"

    output.square().mean().backward()
    assert hypernetwork.heads["block"].weight.grad is not None
    assert any(p.grad is not None for p in conditioner.parameters())


def test_hypernetwork_requires_condition_dim_or_conditioner():
    with pytest.raises(ValueError, match="condition_dim"):
        TextToPeftHypernetwork(
            module_shapes={"q_proj": (8, 8)}, num_layers=2, adapter="lora", latent_dim=16, head_dim=16,
        )
