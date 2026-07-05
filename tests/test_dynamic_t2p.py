import pytest
import torch
from torch import nn

from peft_hnet.t2p.codecs import make_codec
from peft_hnet.t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes


@pytest.mark.parametrize("name", ["lora", "freeze_a_lora", "fourierft", "lokr", "ia3", "activation_steering"])
def test_generated_representation_is_differentiable(name):
    torch.manual_seed(0)
    codec = make_codec(
        name,
        8,
        8,
        num_layers=2,
        rank=2,
        alpha=2,
        n_frequency=4,
    )
    inputs = torch.randn(3, 5, 8)
    base = torch.randn(3, 5, 8)
    generated = torch.randn(3, codec.output_size, requires_grad=True)
    output = codec.apply(inputs, base, generated, layer_index=1)
    assert output.shape == base.shape
    output.square().mean().backward()
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()


@pytest.mark.parametrize("name", ["lora", "freeze_a_lora", "fourierft", "lokr"])
def test_dense_delta_matches_apply(name):
    torch.manual_seed(0)
    codec = make_codec(name, 8, 8, num_layers=2, rank=2, alpha=2, n_frequency=4)
    inputs = torch.randn(3, 5, 8)
    base = torch.zeros(3, 5, 8)
    generated = torch.randn(3, codec.output_size)
    output = codec.apply(inputs, base, generated, layer_index=1)
    delta_weight = codec.dense_delta(generated, layer_index=1)
    assert delta_weight.shape == (3, 8, 8)
    expected = torch.einsum("bsi,boi->bso", inputs.to(delta_weight.dtype), delta_weight)
    torch.testing.assert_close(output.to(expected.dtype), expected, atol=1e-4, rtol=1e-4)


def test_ia3_has_no_dense_delta():
    codec = make_codec("ia3", 8, 8, num_layers=2)
    with pytest.raises(NotImplementedError):
        codec.dense_delta(torch.randn(3, codec.output_size), layer_index=0)


def test_activation_steering_has_no_dense_delta():
    codec = make_codec("activation_steering", 8, 8, num_layers=2)
    with pytest.raises(NotImplementedError):
        codec.dense_delta(torch.randn(3, codec.output_size), layer_index=0)


def test_activation_steering_requires_square_site():
    with pytest.raises(ValueError, match="square"):
        make_codec("activation_steering", 8, 4, num_layers=2)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)

    def forward(self, inputs):
        return self.q_proj(inputs)


def test_hypernetwork_hooks_apply_per_example_adapters_and_backpropagate():
    layers = nn.ModuleList([TinyLayer(), TinyLayer()])
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=6,
        module_shapes={"q_proj": (8, 8)},
        num_layers=2,
        representation="lora",
        latent_dim=32,
        head_dim=32,
        rank=2,
    )
    # Move away from the deliberately identity-preserving zero initialization.
    nn.init.normal_(hypernetwork.heads["q_proj"].weight, std=0.01)
    conditions = torch.randn(3, 6)
    inputs = torch.randn(3, 4, 8)
    generated = hypernetwork(conditions)
    with hypernetwork.apply(layers, generated):
        output = layers[1](layers[0](inputs))
    output.square().mean().backward()
    assert hypernetwork.heads["q_proj"].weight.grad is not None
    assert hypernetwork.generated_parameter_count() == 64


class TupleReturningDecoderLayer(nn.Module):
    """Stand-in for a real HF decoder layer, whose forward returns (hidden_states, ...)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return (self.linear(hidden_states), "extra_output_the_hook_must_preserve")


def test_hypernetwork_hooks_a_whole_layer_via_block_sentinel_and_backpropagates():
    hidden_size = 8
    layers = nn.ModuleList([TupleReturningDecoderLayer(hidden_size), TupleReturningDecoderLayer(hidden_size)])
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=6,
        module_shapes={"block": (hidden_size, hidden_size)},
        num_layers=2,
        representation="activation_steering",
        latent_dim=32,
        head_dim=32,
    )
    nn.init.normal_(hypernetwork.heads["block"].weight, std=0.01)
    conditions = torch.randn(3, 6)
    inputs = torch.randn(3, 4, hidden_size)
    generated = hypernetwork(conditions)
    with hypernetwork.apply(layers, generated):
        hidden, extra = layers[0](inputs)
        output, extra2 = layers[1](hidden)
    assert extra == "extra_output_the_hook_must_preserve"
    assert extra2 == "extra_output_the_hook_must_preserve"
    output.square().mean().backward()
    assert hypernetwork.heads["block"].weight.grad is not None
    assert torch.isfinite(hypernetwork.heads["block"].weight.grad).all()


def test_infer_module_shapes_uses_hidden_size_for_non_linear_targets():
    layers = nn.ModuleList([TupleReturningDecoderLayer(8)])
    shapes = infer_module_shapes(layers, ["block"], hidden_size=8)
    assert shapes == {"block": (8, 8)}


def test_infer_module_shapes_raises_without_hidden_size_for_non_linear_targets():
    layers = nn.ModuleList([TupleReturningDecoderLayer(8)])
    with pytest.raises(TypeError, match="hidden_size"):
        infer_module_shapes(layers, ["block"])


def test_infer_module_shapes_still_resolves_linear_submodules():
    shapes = infer_module_shapes(nn.ModuleList([TinyLayer()]), ["q_proj"])
    assert shapes == {"q_proj": (8, 8)}


@pytest.mark.parametrize("name", ["lora", "lokr"])
def test_bilinear_codecs_have_a_nonzero_initial_bias(name):
    codec = make_codec(name, 8, 8, num_layers=2, rank=2, alpha=2)
    bias = codec.initial_bias()
    assert bias is not None
    assert bias.shape == (codec.output_size,)
    assert bias.abs().max() > 0


@pytest.mark.parametrize("name", ["ia3", "fourierft", "activation_steering"])
def test_non_bilinear_codecs_have_no_initial_bias(name):
    codec = make_codec(name, 8, 8, num_layers=2, n_frequency=4)
    assert codec.initial_bias() is None


@pytest.mark.parametrize("name", ["lora", "lokr"])
def test_bilinear_codec_contributes_zero_at_init_but_has_nonzero_gradient(name):
    """Regression test for a real bug found while smoke-testing live SFT training:
    LoRA/LoKr split `generated` into two factors that are multiplied together (B@A,
    Kronecker product) — bilinear in both factors. With the hypernetwork's default
    all-zero head init, BOTH factors are zero simultaneously, and the gradient w.r.t.
    each factor is proportional to the *other* factor — so both gradients vanish too,
    a dead saddle point where training never moves (confirmed directly: the live SFT
    smoke test's loss repeated bit-for-bit across steps before this fix). The codec's
    `initial_bias()` override (randomizing one factor, leaving the other at zero) must
    fix this: zero contribution at init, but nonzero gradient so training can start.
    """
    torch.manual_seed(0)
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=4,
        module_shapes={"q_proj": (8, 8)},
        num_layers=1,
        representation=name,
        latent_dim=16,
        head_dim=16,
        rank=2,
    )
    conditions = torch.randn(3, 4)
    generated = hypernetwork(conditions)["q_proj"][0]
    inputs = torch.randn(3, 5, 8)
    base = torch.randn(3, 5, 8)

    output = hypernetwork.codecs["q_proj"].apply(inputs, base, generated, layer_index=0)
    torch.testing.assert_close(output, base)  # adapter contributes exactly zero at init

    output.square().mean().backward()
    assert hypernetwork.heads["q_proj"].weight.grad is not None
    assert hypernetwork.heads["q_proj"].weight.grad.abs().max() > 0

