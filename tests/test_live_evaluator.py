from types import SimpleNamespace

import torch
from torch import nn

from adapterbench.contracts import TaskExample
from adapterbench.t2p.hypernetwork import TextToPeftHypernetwork
from adapterbench.t2p.live_evaluator import HypernetworkDownstreamEvaluator


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return (self.linear(hidden_states),)


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([FakeDecoderLayer(hidden_size)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            (hidden,) = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = [ord(char) % 16 for char in text][:8] or [1]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


def _setup(vocab_size=16, hidden_size=8):
    torch.manual_seed(0)
    interpreter = FakeCausalLM(vocab_size, hidden_size)
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=4,
        module_shapes={"block": (hidden_size, hidden_size)},
        num_layers=1,
        adapter="activation_steering",
        latent_dim=16,
        head_dim=16,
    )
    return interpreter, hypernetwork


def _example(family="arc_easy"):
    return TaskExample(
        task_id=f"{family}_1",
        condition="ignored",
        input_text="pick one",
        target_text="0",
        family=family,
        metadata={"choices": ["a", "b"], "answer_index": 0},
    )


def test_evaluate_frozen_scores_without_any_hypernetwork_involvement():
    interpreter, hypernetwork = _setup()
    evaluator = HypernetworkDownstreamEvaluator(
        interpreter, interpreter.layers, hypernetwork, FakeTokenizer(), trial_id="t", device="cpu"
    )
    results = evaluator.evaluate_frozen([_example()], split="test")
    assert len(results) == 1
    result = results[0]
    assert result.adapter == "frozen_interpreter"
    assert result.generated_parameter_count == 0
    assert 0.0 <= result.metrics["accuracy"] <= 1.0


def test_evaluate_hooks_the_generated_adapter_and_reports_its_parameter_count():
    interpreter, hypernetwork = _setup()
    evaluator = HypernetworkDownstreamEvaluator(
        interpreter, interpreter.layers, hypernetwork, FakeTokenizer(), trial_id="t", device="cpu"
    )
    condition_embeddings = {"arc_easy": torch.randn(4)}
    results = evaluator.evaluate(condition_embeddings, [_example()], split="test")
    assert len(results) == 1
    result = results[0]
    assert result.adapter == "activation_steering"
    assert result.generated_parameter_count == hypernetwork.generated_parameter_count()


def test_hooks_are_removed_after_each_group_no_leakage_between_calls():
    interpreter, hypernetwork = _setup()
    evaluator = HypernetworkDownstreamEvaluator(
        interpreter, interpreter.layers, hypernetwork, FakeTokenizer(), trial_id="t", device="cpu"
    )
    evaluator.evaluate({"arc_easy": torch.randn(4)}, [_example()], split="test")
    for layer in interpreter.layers:
        assert layer._forward_hooks == {}
