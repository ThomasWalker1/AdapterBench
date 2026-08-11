import pytest
import torch
from torch import nn

from adapterbench.t2a.codecs import make_codec
from adapterbench.t2a.hypernetwork import TextToPeftHypernetwork, infer_module_shapes


@pytest.mark.parametrize("name", ["lora", "ia3", "lokr", "fourierft", "steering"])
def test_generated_adapter_is_differentiable(name):
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


@pytest.mark.parametrize("name", ["lora", "lokr", "fourierft"])
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


def test_make_codec_rejects_unregistered_shape():
    with pytest.raises(ValueError, match="unsupported differentiable adapter"):
        make_codec("lok r", 8, 8, num_layers=2)


def test_ia3_geometry_and_identity_initialization():
    codec = make_codec("ia3", 5, 8, num_layers=2, ia3_scaling=2.0)
    assert codec.output_size == 8
    generated = torch.zeros(3, codec.output_size)
    inputs = torch.randn(3, 4, 5)
    base = torch.randn(3, 4, 8)
    torch.testing.assert_close(codec.apply(inputs, base, generated, layer_index=0), base)
    assert codec.dense_delta(generated, layer_index=0).shape == (3, 8, 5)
    assert codec.initial_bias() is None


def test_ia3_apply_scales_each_output_channel():
    codec = make_codec("ia3", 5, 3, num_layers=1, ia3_scaling=0.5)
    inputs = torch.randn(2, 4, 5)
    base = torch.tensor([[[2.0, -3.0, 4.0]], [[-1.0, 6.0, 8.0]]]).expand(-1, 4, -1)
    generated = torch.tensor([[2.0, -1.0, 0.0], [-2.0, 1.0, 3.0]])
    expected = base * (1 + 0.5 * generated).unsqueeze(1)
    torch.testing.assert_close(codec.apply(inputs, base, generated, layer_index=0), expected)


def test_lokr_geometry_and_identity_initialization():
    codec = make_codec("lokr", 8, 8, num_layers=2, lokr_scaling=2.0)
    # 8 = 2 * 4, so ΔW = L(2,2) ⊗ R(4,4) and emits 4 + 16 scalars.
    assert (codec.out_factor_1, codec.out_factor_2) == (2, 4)
    assert (codec.in_factor_1, codec.in_factor_2) == (2, 4)
    assert codec.output_size == 20
    generated = torch.zeros(3, codec.output_size)
    inputs = torch.randn(3, 4, 8)
    base = torch.randn(3, 4, 8)
    torch.testing.assert_close(codec.apply(inputs, base, generated, layer_index=0), base)
    assert codec.dense_delta(generated, layer_index=0).shape == (3, 8, 8)
    # T2A's fixed Gemma-2 q_proj/v_proj budget is declared in the manifest.
    assert make_codec("lokr", 2304, 2048, num_layers=26).output_size == 4608
    assert make_codec("lokr", 2304, 1024, num_layers=26).output_size == 3072


def test_lokr_dense_delta_is_the_kronecker_product():
    codec = make_codec("lokr", 4, 4, num_layers=1, lokr_scaling=1.0)
    # 4 = 2 * 2.  The first four entries are L and the final four are R.
    generated = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    expected = torch.kron(generated[0, :4].reshape(2, 2), generated[0, 4:].reshape(2, 2))
    torch.testing.assert_close(codec.dense_delta(generated, layer_index=0)[0], expected)


def test_lokr_live_contraction_matches_dense_delta():
    torch.manual_seed(7)
    codec = make_codec("lokr", 8, 8, num_layers=1, lokr_scaling=0.75)
    inputs = torch.randn(2, 3, 8)
    generated = torch.randn(2, codec.output_size)
    base = torch.zeros(2, 3, 8)
    expected = torch.einsum("bsi,boi->bso", inputs, codec.dense_delta(generated, layer_index=0))
    torch.testing.assert_close(codec.apply(inputs, base, generated, layer_index=0), expected, atol=1e-5, rtol=1e-5)


def test_fourierft_geometry_matches_the_locked_lora_scalar_budget():
    codec = make_codec("fourierft", 8, 8, num_layers=2, rank=2, fourierft_scaling=2.0)
    assert codec.n_freqs == 32
    assert codec.output_size == make_codec("lora", 8, 8, num_layers=2, rank=2).output_size
    assert codec.dct_out.shape == (8, 8)
    assert codec.dct_in.shape == (8, 8)
    assert codec.row_frequencies.shape == (2, 32)
    assert codec.column_frequencies.shape == (2, 32)
    pairs = codec.row_frequencies[0] * 8 + codec.column_frequencies[0]
    assert pairs.unique().numel() == codec.output_size
    assert not tuple(codec.parameters())
    assert make_codec("fourierft", 2304, 2048, num_layers=26, rank=8).output_size == 34816
    assert make_codec("fourierft", 2304, 1024, num_layers=26, rank=8).output_size == 26624


def test_fourierft_zero_coefficients_are_identity_with_nonzero_gradient():
    codec = make_codec("fourierft", 8, 8, num_layers=2, rank=2, fourierft_scaling=0.75)
    inputs = torch.randn(3, 5, 8)
    base = torch.randn(3, 5, 8)
    generated = torch.zeros(3, codec.output_size, requires_grad=True)
    output = codec.apply(inputs, base, generated, layer_index=1)
    torch.testing.assert_close(output, base)
    output.square().mean().backward()
    assert codec.initial_bias() is None
    assert generated.grad is not None
    assert generated.grad.abs().max() > 0
    assert torch.isfinite(generated.grad).all()


def test_steering_geometry_and_identity_initialization():
    codec = make_codec("steering", 5, 8, num_layers=2, steering_scaling=2.0)
    assert codec.output_size == 8
    assert codec.scaling == 2.0
    generated = torch.zeros(3, codec.output_size)
    inputs = torch.randn(3, 4, 5)
    base = torch.randn(3, 4, 8)
    # zero vector => exact identity (frozen model), like every linear codec
    torch.testing.assert_close(codec.apply(inputs, base, generated, layer_index=0), base)
    assert codec.dense_delta(generated, layer_index=0).shape == (3, 8, 5)
    assert codec.initial_bias() is None
    assert not tuple(codec.parameters())


def test_steering_apply_adds_the_scaled_vector_at_every_position_per_example():
    codec = make_codec("steering", 8, 8, num_layers=1, steering_scaling=0.5)
    base = torch.randn(2, 4, 8)
    inputs = torch.randn(2, 4, 8)
    generated = torch.randn(2, 8)
    output = codec.apply(inputs, base, generated, layer_index=0)
    # each example's own vector, broadcast over its sequence positions
    torch.testing.assert_close(output, base + 0.5 * generated.unsqueeze(1))


def test_steering_zero_head_is_identity_with_nonzero_gradient():
    inputs = torch.randn(3, 5, 8)
    base = torch.randn(3, 5, 8)
    codec = make_codec("steering", 8, 8, num_layers=1)
    generated = torch.zeros(3, codec.output_size, requires_grad=True)
    output = codec.apply(inputs, base, generated, layer_index=0)
    torch.testing.assert_close(output, base)
    output.square().mean().backward()
    assert generated.grad is not None
    assert generated.grad.abs().max() > 0


def test_set_codec_scaling_applies_one_scale_to_every_codec():
    from adapterbench.t2a.codecs import set_codec_scaling

    codecs = {
        "q_proj": make_codec("lora", 8, 8, num_layers=1, rank=2, alpha=2),
        "block": make_codec("steering", 8, 8, num_layers=1),
    }
    set_codec_scaling(codecs, 3.5)
    assert all(codec.scaling == 3.5 for codec in codecs.values())


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
        adapter="lora",
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


def test_ia3_hypernetwork_hook_applies_per_example_adapters_and_backpropagates():
    layers = nn.ModuleList([TinyLayer(), TinyLayer()])
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=6, module_shapes={"q_proj": (8, 8)}, num_layers=2,
        adapter="ia3", latent_dim=32, head_dim=32, ia3_scaling=1.0,
    )
    nn.init.normal_(hypernetwork.heads["q_proj"].weight, std=0.01)
    conditions = torch.randn(3, 6)
    inputs = torch.randn(3, 4, 8)
    generated = hypernetwork(conditions)
    with hypernetwork.apply(layers, generated):
        output = layers[1](layers[0](inputs))
    output.square().mean().backward()
    assert hypernetwork.generated_parameter_count() == 16
    assert hypernetwork.heads["q_proj"].weight.grad is not None
    assert torch.isfinite(hypernetwork.heads["q_proj"].weight.grad).all()


def test_lokr_hypernetwork_hook_applies_per_example_adapters_and_backpropagates():
    layers = nn.ModuleList([TinyLayer(), TinyLayer()])
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=6, module_shapes={"q_proj": (8, 8)}, num_layers=2,
        adapter="lokr", latent_dim=32, head_dim=32, lokr_scaling=1.0,
    )
    conditions = torch.randn(3, 6)
    inputs = torch.randn(3, 4, 8)
    generated = hypernetwork(conditions)
    with hypernetwork.apply(layers, generated):
        output = layers[1](layers[0](inputs))
    output.square().mean().backward()
    assert hypernetwork.generated_parameter_count() == 40
    assert hypernetwork.heads["q_proj"].weight.grad is not None
    assert hypernetwork.heads["q_proj"].weight.grad.abs().max() > 0
    assert torch.isfinite(hypernetwork.heads["q_proj"].weight.grad).all()


def test_fourierft_hypernetwork_hook_applies_per_example_adapters_and_backpropagates():
    layers = nn.ModuleList([TinyLayer(), TinyLayer()])
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=6, module_shapes={"q_proj": (8, 8)}, num_layers=2,
        adapter="fourierft", latent_dim=32, head_dim=32, rank=2, fourierft_scaling=1.0,
    )
    conditions = torch.randn(3, 6)
    inputs = torch.randn(3, 4, 8)
    generated = hypernetwork(conditions)
    with hypernetwork.apply(layers, generated):
        output = layers[1](layers[0](inputs))
    output.square().mean().backward()
    assert hypernetwork.generated_parameter_count() == 64
    assert hypernetwork.heads["q_proj"].weight.grad is not None
    assert hypernetwork.heads["q_proj"].weight.grad.abs().max() > 0
    assert torch.isfinite(hypernetwork.heads["q_proj"].weight.grad).all()


def test_ia3_zero_head_is_identity_but_has_a_nonzero_gradient():
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=4, module_shapes={"q_proj": (8, 8)}, num_layers=1,
        adapter="ia3", latent_dim=16, head_dim=16,
    )
    generated = hypernetwork(torch.randn(3, 4))["q_proj"][0]
    inputs, base = torch.randn(3, 5, 8), torch.randn(3, 5, 8)
    output = hypernetwork.codecs["q_proj"].apply(inputs, base, generated, layer_index=0)
    torch.testing.assert_close(output, base)
    output.square().mean().backward()
    assert hypernetwork.heads["q_proj"].weight.grad is not None
    assert hypernetwork.heads["q_proj"].weight.grad.abs().max() > 0


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
        adapter="lora",
        latent_dim=32,
        head_dim=32,
        rank=2,
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


def test_steering_hypernetwork_hooks_the_residual_stream_and_backpropagates():
    hidden_size = 8
    layers = nn.ModuleList([TupleReturningDecoderLayer(hidden_size), TupleReturningDecoderLayer(hidden_size)])
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=6,
        module_shapes={"block": (hidden_size, hidden_size)},
        num_layers=2,
        adapter="steering",
        latent_dim=32,
        head_dim=32,
        steering_scaling=1.0,
    )
    nn.init.normal_(hypernetwork.heads["block"].weight, std=0.01)
    conditions = torch.randn(3, 6)
    inputs = torch.randn(3, 4, hidden_size)
    generated = hypernetwork(conditions)
    with hypernetwork.apply(layers, generated):
        hidden, extra = layers[0](inputs)
        output, _ = layers[1](hidden)
    assert extra == "extra_output_the_hook_must_preserve"
    output.square().mean().backward()
    assert hypernetwork.generated_parameter_count() == 2 * hidden_size
    assert hypernetwork.heads["block"].weight.grad is not None
    assert hypernetwork.heads["block"].weight.grad.abs().max() > 0
    assert torch.isfinite(hypernetwork.heads["block"].weight.grad).all()


def test_hypernetwork_forwards_codec_kwargs_to_make_codec():
    # Codec-specific kwargs (ia3_scaling, lora_scaling, steering_scaling, ...) reach the
    # codec through the constructor's **codec_kwargs passthrough, so registering a new
    # codec never needs a new named hypernetwork parameter.
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=4, module_shapes={"q_proj": (8, 8)}, num_layers=1,
        adapter="ia3", latent_dim=16, head_dim=16, ia3_scaling=2.5,
    )
    assert hypernetwork.codecs["q_proj"].scaling == 2.5
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=4, module_shapes={"q_proj": (8, 8)}, num_layers=1,
        adapter="lora", latent_dim=16, head_dim=16, rank=2, lora_scaling=7.0,
    )
    assert hypernetwork.codecs["q_proj"].scaling == 7.0


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


@pytest.mark.parametrize("name", ["lora", "lokr"])
def test_bilinear_codec_contributes_zero_at_init_but_has_nonzero_gradient(name):
    """Regression test for a real bug found while smoke-testing live SFT training:
    Factorized codecs have a zero-gradient saddle when all generated factors start at
    zero. LoRA/LoKr are bilinear. The codec's `initial_bias()` must retain exact
    identity at init while leaving a path for the head to train.
    """
    torch.manual_seed(0)
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=4,
        module_shapes={"q_proj": (8, 8)},
        num_layers=1,
        adapter=name,
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
