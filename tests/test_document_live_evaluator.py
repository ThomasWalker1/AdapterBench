from types import SimpleNamespace

import torch
from torch import nn

from adapterbench.t2p.document_conditioning import DocumentPerceiverConditioner
from adapterbench.t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes
from adapterbench.t2p.live_evaluator import DocumentHypernetworkDownstreamEvaluator
from adapterbench.t2p.niah_data import NiahExample


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return (self.linear(hidden_states),)


class FakeCausalLM(nn.Module):
    """Like test_live_evaluator.py's FakeCausalLM, extended with output_hidden_states
    support so it can stand in for capture_document_activations's interpreter arg."""

    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([FakeDecoderLayer(hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        hidden = self.embed(input_ids)
        hidden_states = [hidden] if output_hidden_states else None
        for layer in self.layers:
            (hidden,) = layer(hidden)
            if output_hidden_states:
                hidden_states.append(hidden)
        return SimpleNamespace(
            logits=self.lm_head(hidden), hidden_states=tuple(hidden_states) if output_hidden_states else None
        )

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, do_sample=False, pad_token_id=None):
        pad = torch.zeros((input_ids.shape[0], max_new_tokens), dtype=input_ids.dtype)
        return torch.cat([input_ids, pad], dim=1)


class _BatchEncoding(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = [ord(char) % 16 for char in text][:8] or [1]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return _BatchEncoding(
                input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)
            )
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "0042 some text"  # always "contains" a 4-digit needle for a deterministic score


def _setup(vocab_size=16, hidden_size=8, num_layers=2):
    torch.manual_seed(0)
    interpreter = FakeCausalLM(vocab_size, hidden_size, num_layers)
    conditioner = DocumentPerceiverConditioner(hidden_size=hidden_size, task_dim=8, num_layers=num_layers, latent_dim=8)
    module_shapes = infer_module_shapes(interpreter.layers, ["block"], hidden_size=hidden_size)
    hypernetwork = TextToPeftHypernetwork(
        module_shapes=module_shapes,
        num_layers=num_layers,
        adapter="activation_steering",
        latent_dim=16,
        head_dim=16,
        conditioner=conditioner,
    )
    return interpreter, hypernetwork


def _example(digits="0042", topic="biology"):
    return NiahExample(context_text="filler filler filler needle filler", topic=topic, digits=digits, depth=0.5)


def test_evaluate_frozen_never_captures_document_activations():
    interpreter, hypernetwork = _setup()
    evaluator = DocumentHypernetworkDownstreamEvaluator(
        interpreter, interpreter.layers, hypernetwork, FakeTokenizer(), trial_id="t", device="cpu"
    )
    results = evaluator.evaluate_frozen({"niah_256": [_example()]}, split="test")
    assert len(results) == 1
    result = results[0]
    assert result.adapter == "frozen_interpreter"
    assert result.generated_parameter_count == 0
    assert result.metrics["accuracy"] == 1.0  # FakeTokenizer.decode always "contains" 0042


def test_evaluate_hooks_a_fresh_adapter_per_example_and_reports_parameter_count():
    interpreter, hypernetwork = _setup()
    evaluator = DocumentHypernetworkDownstreamEvaluator(
        interpreter, interpreter.layers, hypernetwork, FakeTokenizer(), trial_id="t", device="cpu"
    )
    results = evaluator.evaluate({"niah_256": [_example(), _example(digits="1234")]}, split="test")
    assert len(results) == 1
    result = results[0]
    assert result.adapter == "activation_steering"
    assert result.generated_parameter_count == hypernetwork.generated_parameter_count()
    assert result.metrics["n_examples"] == 2.0


def test_hooks_are_removed_after_each_example_no_leakage_between_calls():
    interpreter, hypernetwork = _setup()
    evaluator = DocumentHypernetworkDownstreamEvaluator(
        interpreter, interpreter.layers, hypernetwork, FakeTokenizer(), trial_id="t", device="cpu"
    )
    evaluator.evaluate({"niah_256": [_example(), _example()]}, split="test")
    for layer in interpreter.layers:
        assert layer._forward_hooks == {}
