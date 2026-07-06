from types import SimpleNamespace

import torch
from torch import nn

from adapterbench.t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes
from adapterbench.t2p.model_utils import get_decoder_layers
from adapterbench.t2p.sft_trainer import SFTBatch, compute_sft_loss, train_downstream_hypernetwork
from adapterbench.t2p.tiny_interpreter import build_tiny_interpreter


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return (self.linear(hidden_states),)


class TinyCausalLM(nn.Module):
    """Minimal stand-in for a real HF causal LM: embeds tokens, runs them through a
    small stack of tuple-returning decoder layers (so `TextToPeftHypernetwork.apply`'s
    "block"-hook path is exercised the same way it would be on a real model), projects
    to vocab logits."""

    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([TinyDecoderLayer(hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            (hidden,) = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _toy_setup(seed=0):
    torch.manual_seed(seed)
    vocab_size, hidden_size, num_layers = 16, 8, 2
    interpreter = TinyCausalLM(vocab_size, hidden_size, num_layers)
    for parameter in interpreter.parameters():
        parameter.requires_grad = False
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=6,
        module_shapes={"block": (hidden_size, hidden_size)},
        num_layers=num_layers,
        adapter="activation_steering",
        latent_dim=16,
        head_dim=16,
    )
    return interpreter, interpreter.layers, hypernetwork, vocab_size


def _make_batch(batch_size, seq_len, vocab_size, condition_dim, target_token):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.full((batch_size, seq_len), -100)
    labels[:, -1] = target_token  # only the last position is supervised, easy to learn
    return SFTBatch(
        input_ids=input_ids,
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        labels=labels,
        condition_embeddings=torch.randn(batch_size, condition_dim),
    )


def test_compute_sft_loss_is_finite_and_differentiable():
    interpreter, layers, hypernetwork, vocab_size = _toy_setup()
    batch = _make_batch(batch_size=3, seq_len=5, vocab_size=vocab_size, condition_dim=6, target_token=2)
    loss = compute_sft_loss(batch, interpreter, hypernetwork, layers)
    assert torch.isfinite(loss)
    loss.backward()
    assert hypernetwork.heads["block"].weight.grad is not None
    assert all(parameter.grad is None for parameter in interpreter.parameters())


def test_train_downstream_hypernetwork_reduces_loss_on_an_easy_target():
    interpreter, layers, hypernetwork, vocab_size = _toy_setup()
    nn.init.normal_(hypernetwork.heads["block"].weight, std=0.05)
    batch = _make_batch(batch_size=4, seq_len=6, vocab_size=vocab_size, condition_dim=6, target_token=3)

    stats = train_downstream_hypernetwork(
        hypernetwork, interpreter, layers, [batch], steps=100, learning_rate=1e-2
    )

    assert stats.steps == 100
    assert len(stats.losses) == 100
    assert all(torch.isfinite(torch.tensor(loss)) for loss in stats.losses)
    assert stats.final_loss < stats.initial_loss


def test_train_downstream_hypernetwork_works_against_a_real_tiny_transformers_model():
    """Same training loop as the toy-model tests above, but the interpreter is a real
    (if tiny) `transformers` LlamaForCausalLM (t2p/tiny_interpreter.py) — the lightweight
    synthetic setting's interpreter — proving it's a drop-in replacement for the
    hand-rolled `TinyCausalLM` stand-in, not just superficially similar."""
    torch.manual_seed(0)
    interpreter = build_tiny_interpreter(vocab_size=16, hidden_size=32, num_layers=2)
    for parameter in interpreter.parameters():
        parameter.requires_grad = False
    layers = get_decoder_layers(interpreter)
    module_shapes = infer_module_shapes(layers, ["q_proj", "v_proj"])
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=6,
        module_shapes=module_shapes,
        num_layers=len(layers),
        adapter="lora",
        latent_dim=16,
        head_dim=16,
        rank=2,
    )
    batch = _make_batch(batch_size=4, seq_len=6, vocab_size=16, condition_dim=6, target_token=5)

    stats = train_downstream_hypernetwork(hypernetwork, interpreter, layers, [batch], steps=100, learning_rate=1e-2)

    assert all(torch.isfinite(torch.tensor(loss)) for loss in stats.losses)
    assert stats.final_loss < stats.initial_loss
