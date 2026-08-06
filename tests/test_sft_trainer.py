from types import SimpleNamespace

import pytest
import torch
from torch import nn

from adapterbench.t2a.hypernetwork import TextToPeftHypernetwork, infer_module_shapes
from adapterbench.t2a.model_utils import get_decoder_layers
from adapterbench.t2a.sft_trainer import (
    SFTBatch,
    compute_sft_loss,
    train_downstream_hypernetwork,
    train_step,
    train_with_checkpoints,
)


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
        adapter="lora",
        latent_dim=16,
        head_dim=16,
        rank=2,
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


def test_train_with_checkpoints_returns_one_stats_entry_per_checkpoint_with_correct_deltas():
    interpreter, layers, hypernetwork, vocab_size = _toy_setup()
    batch = _make_batch(batch_size=4, seq_len=6, vocab_size=vocab_size, condition_dim=6, target_token=3)

    stats_by_checkpoint = train_with_checkpoints(
        hypernetwork, interpreter, layers, [batch], checkpoint_steps=[30, 50, 100], learning_rate=1e-2
    )

    assert list(stats_by_checkpoint) == [30, 50, 100]
    assert stats_by_checkpoint[30].steps == 30
    assert stats_by_checkpoint[50].steps == 20  # delta since checkpoint 30, not cumulative
    assert stats_by_checkpoint[100].steps == 50  # delta since checkpoint 50
    for stats in stats_by_checkpoint.values():
        assert all(torch.isfinite(torch.tensor(loss)) for loss in stats.losses)


def test_train_with_checkpoints_matches_one_continuous_run_of_the_same_total_steps():
    """Checkpointing shouldn't perturb the trajectory - training to [50, 100] on a given
    hypernetwork should produce bit-for-bit the same loss curve as one continuous 100-step
    run from the same starting weights, since both use a single persistent optimizer over
    the same deterministic batch cycle - no optimizer-reset discontinuity at the checkpoint
    boundary. Resets the *same* module instance's weights in place between the two runs
    (rather than comparing against a separately-constructed second instance) so this isn't
    confounded by the floating-point non-associativity two distinct module instances can
    introduce even from identical copied weights."""
    interpreter, layers, hypernetwork, vocab_size = _toy_setup()
    batch = _make_batch(batch_size=4, seq_len=6, vocab_size=vocab_size, condition_dim=6, target_token=3)
    pristine_state = {k: v.clone() for k, v in hypernetwork.state_dict().items()}

    torch.manual_seed(123)  # hypernetwork has internal dropout - pin the RNG both runs consume
    continuous = train_downstream_hypernetwork(hypernetwork, interpreter, layers, [batch], steps=100, learning_rate=1e-2)

    hypernetwork.load_state_dict(pristine_state)
    torch.manual_seed(123)
    checkpointed = train_with_checkpoints(
        hypernetwork, interpreter, layers, [batch], checkpoint_steps=[50, 100], learning_rate=1e-2
    )
    stitched_losses = checkpointed[50].losses + checkpointed[100].losses

    assert stitched_losses == continuous.losses


def test_train_with_checkpoints_rejects_non_ascending_checkpoints():
    interpreter, layers, hypernetwork, vocab_size = _toy_setup()
    batch = _make_batch(batch_size=4, seq_len=6, vocab_size=vocab_size, condition_dim=6, target_token=3)

    with pytest.raises(ValueError):
        train_with_checkpoints(hypernetwork, interpreter, layers, [batch], checkpoint_steps=[100, 100], learning_rate=1e-2)


def test_train_step_with_grad_accumulation_matches_one_step_on_the_concatenated_batch():
    """Accumulating gradients over two equal-size micro-batches should produce the same
    gradient as one step on their concatenation, since `masked_cross_entropy` already
    per-example-averages within a batch - averaging two equal-size per-batch means equals
    the true overall mean. This is the actual guarantee grad_accum_steps is supposed to
    provide (same update as a bigger batch, less memory), not just "it runs"."""
    interpreter, layers, hypernetwork, vocab_size = _toy_setup()
    batch_a = _make_batch(batch_size=4, seq_len=6, vocab_size=vocab_size, condition_dim=6, target_token=3)
    batch_b = _make_batch(batch_size=4, seq_len=6, vocab_size=vocab_size, condition_dim=6, target_token=5)
    concatenated = SFTBatch(
        input_ids=torch.cat([batch_a.input_ids, batch_b.input_ids]),
        attention_mask=torch.cat([batch_a.attention_mask, batch_b.attention_mask]),
        labels=torch.cat([batch_a.labels, batch_b.labels]),
        condition_embeddings=torch.cat([batch_a.condition_embeddings, batch_b.condition_embeddings]),
    )

    pristine_state = {k: v.clone() for k, v in hypernetwork.state_dict().items()}
    # The hypernetwork has internal dropout, which draws a differently-shaped random mask
    # for a (4, ...) batch than a (8, ...) one even from the same RNG state - breaking the
    # exact equivalence this test checks for reasons unrelated to grad_accum_steps itself.
    # Disable it so this isolates the accumulation math, not dropout's batch-shape coupling.
    hypernetwork.eval()

    optimizer_accum = torch.optim.AdamW(hypernetwork.parameters(), lr=1e-2)
    train_step([batch_a, batch_b], interpreter, hypernetwork, layers, optimizer_accum)
    accumulated_grads = {name: p.grad.clone() for name, p in hypernetwork.named_parameters() if p.grad is not None}

    # train_step's optimizer.step() moved the weights - reset to the same starting point
    # before the second call, or its gradients would reflect a different (post-step) state.
    hypernetwork.load_state_dict(pristine_state)
    optimizer_single = torch.optim.AdamW(hypernetwork.parameters(), lr=1e-2)
    train_step([concatenated], interpreter, hypernetwork, layers, optimizer_single)
    single_batch_grads = {name: p.grad.clone() for name, p in hypernetwork.named_parameters() if p.grad is not None}

    assert accumulated_grads.keys() == single_batch_grads.keys()
    for name in accumulated_grads:
        assert torch.allclose(accumulated_grads[name], single_batch_grads[name], atol=1e-5), name


def test_train_downstream_hypernetwork_respects_grad_accum_steps():
    interpreter, layers, hypernetwork, vocab_size = _toy_setup()
    batch = _make_batch(batch_size=4, seq_len=6, vocab_size=vocab_size, condition_dim=6, target_token=3)

    stats = train_downstream_hypernetwork(
        hypernetwork, interpreter, layers, [batch], steps=10, learning_rate=1e-2, grad_accum_steps=3
    )

    # 10 optimizer steps requested, regardless of how many micro-batches each consumes.
    assert stats.steps == 10
    assert len(stats.losses) == 10


def test_warmup_ramps_lr_linearly_then_holds_constant():
    interpreter, layers, hypernetwork, vocab_size = _toy_setup()
    batch = _make_batch(batch_size=4, seq_len=6, vocab_size=vocab_size, condition_dim=6, target_token=3)

    optimizer = torch.optim.AdamW(hypernetwork.parameters(), lr=1e-2)
    from adapterbench.t2a.sft_trainer import _linear_warmup_then_constant

    scheduler = _linear_warmup_then_constant(optimizer, warmup_steps=4)
    lrs = []
    for _ in range(6):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[0] == pytest.approx(1e-2 * 1 / 4)
    assert lrs[1] == pytest.approx(1e-2 * 2 / 4)
    assert lrs[3] == pytest.approx(1e-2)
    assert lrs[5] == pytest.approx(1e-2)  # held constant past warmup, no decay


def test_train_downstream_hypernetwork_works_against_a_real_tiny_transformers_model():
    """Same training loop as the toy-model tests above, but the interpreter is a real
    (if tiny, freshly initialized, never downloaded) `transformers` `LlamaForCausalLM` -
    proving the training loop is a drop-in replacement for the hand-rolled `TinyCausalLM`
    stand-in, not just superficially similar."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    interpreter = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=64, max_position_embeddings=32,
        )
    )
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
