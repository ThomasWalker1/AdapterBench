import torch

from peft_hnet.generators import BasisStateGenerator
from peft_hnet.reconstruction import train_reconstruction_generator
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


def test_basis_generator_reconstruction_training_reduces_loss():
    torch.manual_seed(0)
    conditions = torch.randn(32, 6)
    mapping = torch.randn(6, 10)
    states = conditions @ mapping
    generator = BasisStateGenerator(condition_dim=6, state_dim=10, hidden_dim=24, num_bases=10)
    stats = train_reconstruction_generator(
        generator,
        conditions[:24],
        states[:24],
        conditions[24:],
        states[24:],
        steps=100,
        batch_size=8,
        learning_rate=1e-2,
        seed=0,
    )
    assert stats.final_loss < stats.initial_loss
    assert stats.nonfinite_steps == 0

