import torch

from peft_hnet.t2p.hypernetwork import infer_module_shapes
from peft_hnet.t2p.model_utils import get_decoder_layers
from peft_hnet.t2p.tiny_interpreter import build_tiny_interpreter


def test_build_tiny_interpreter_constructs_offline_with_expected_shapes():
    model = build_tiny_interpreter(vocab_size=16, hidden_size=32, num_layers=2)
    input_ids = torch.tensor([[1, 4, 5, 6, 3]])
    with torch.no_grad():
        output = model(input_ids=input_ids)
    assert output.logits.shape == (1, 5, 16)


def test_get_decoder_layers_and_infer_module_shapes_resolve_against_it():
    model = build_tiny_interpreter(hidden_size=32, num_layers=3, num_heads=4, num_kv_heads=2)
    layers = get_decoder_layers(model)
    assert len(layers) == 3
    shapes = infer_module_shapes(layers, ["q_proj", "v_proj"])
    assert shapes["q_proj"] == (32, 32)
    assert shapes["v_proj"] == (32, 16)  # num_kv_heads=2 * head_dim=8
    block_shapes = infer_module_shapes(layers, ["block"], hidden_size=model.config.hidden_size)
    assert block_shapes == {"block": (32, 32)}


def test_generate_produces_requested_number_of_new_tokens():
    model = build_tiny_interpreter()
    model.eval()
    input_ids = torch.tensor([[1, 4, 5, 6, 3]])
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=6, do_sample=False, pad_token_id=0)
    assert output.shape == (1, input_ids.shape[1] + 6)


def test_build_tiny_interpreter_is_deterministic_given_the_same_seed():
    a = build_tiny_interpreter(seed=123)
    b = build_tiny_interpreter(seed=123)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
