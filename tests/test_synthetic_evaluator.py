import torch

from adapterbench.t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes
from adapterbench.t2p.model_utils import get_decoder_layers
from adapterbench.t2p.synthetic_evaluator import evaluate_families
from adapterbench.t2p.synthetic_tasks import TASK_FAMILIES
from adapterbench.t2p.tiny_interpreter import build_tiny_interpreter


def _setup():
    interpreter = build_tiny_interpreter(vocab_size=16, hidden_size=16, num_layers=2, num_heads=2, num_kv_heads=1)
    interpreter.eval()
    for parameter in interpreter.parameters():
        parameter.requires_grad = False
    layers = get_decoder_layers(interpreter)
    module_shapes = infer_module_shapes(layers, ["block"], hidden_size=interpreter.config.hidden_size)
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=4,
        module_shapes=module_shapes,
        num_layers=len(layers),
        adapter="activation_steering",
        latent_dim=8,
        head_dim=8,
    )
    return interpreter, layers, hypernetwork


def test_evaluate_families_returns_a_fraction_per_requested_family():
    interpreter, layers, hypernetwork = _setup()
    condition_embeddings = {name: torch.randn(4) for name in TASK_FAMILIES}
    accuracies = evaluate_families(
        hypernetwork, interpreter, layers, condition_embeddings, ["copy", "reverse"], num_examples=8
    )
    assert set(accuracies) == {"copy", "reverse"}
    for accuracy in accuracies.values():
        assert 0.0 <= accuracy <= 1.0


def test_evaluate_families_frozen_baseline_needs_no_hypernetwork():
    interpreter, layers, _ = _setup()
    accuracies = evaluate_families(None, interpreter, layers, None, ["copy"], num_examples=8)
    assert set(accuracies) == {"copy"}


def test_evaluate_family_is_reproducible_given_the_same_seed():
    interpreter, layers, hypernetwork = _setup()
    condition_embeddings = {name: torch.randn(4) for name in TASK_FAMILIES}
    first = evaluate_families(hypernetwork, interpreter, layers, condition_embeddings, ["sort"], num_examples=10, seed=99)
    second = evaluate_families(hypernetwork, interpreter, layers, condition_embeddings, ["sort"], num_examples=10, seed=99)
    assert first == second
