import torch

from peft_hnet.state import AdapterStateLayout


def test_adapter_state_round_trip_preserves_named_tensor_shapes():
    state = {"b": torch.randn(3), "a": torch.randn(2, 4)}
    layout = AdapterStateLayout.from_state_dict(state)
    vector = layout.flatten(state)
    restored = layout.unflatten(vector)
    assert layout.parameter_count == 11
    assert list(restored) == ["a", "b"]
    assert torch.equal(restored["a"], state["a"])
    assert torch.equal(restored["b"], state["b"])
