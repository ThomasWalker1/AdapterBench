import torch
from torch import nn

from adapterbench.t2p.document_conditioning import DocumentPerceiverConditioner
from adapterbench.t2p.document_sft_trainer import (
    compute_doc_sft_loss,
    doc_train_step,
    train_doc_downstream_hypernetwork,
)
from adapterbench.t2p.hypernetwork import TextToPeftHypernetwork
from adapterbench.t2p.model_utils import get_decoder_layers
from adapterbench.t2p.niah_data import DocSFTBatch


def _toy_setup(seed=0, adapter="lora", target_modules=("q_proj",)):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    vocab_size, hidden_size, num_layers = 16, 8, 2
    interpreter = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_layers, num_attention_heads=2,
            num_key_value_heads=1, intermediate_size=16, max_position_embeddings=64,
        )
    )
    for parameter in interpreter.parameters():
        parameter.requires_grad = False
    layers = get_decoder_layers(interpreter)
    from adapterbench.t2p.hypernetwork import infer_module_shapes

    module_shapes = infer_module_shapes(layers, list(target_modules), hidden_size=hidden_size)
    latent_dim, task_dim = 16, 8
    conditioner = DocumentPerceiverConditioner(hidden_size=hidden_size, task_dim=task_dim, num_layers=num_layers, latent_dim=8)
    hypernetwork = TextToPeftHypernetwork(
        module_shapes=module_shapes,
        num_layers=num_layers,
        adapter=adapter,
        latent_dim=latent_dim,
        head_dim=16,
        rank=2,
        conditioner=conditioner,
    )
    return interpreter, layers, hypernetwork, vocab_size


def _make_doc_batch(batch_size, context_len, query_len, vocab_size, target_token):
    context_input_ids = torch.randint(0, vocab_size, (batch_size, context_len))
    query_input_ids = torch.randint(0, vocab_size, (batch_size, query_len))
    labels = torch.full((batch_size, query_len), -100)
    labels[:, -1] = target_token
    return DocSFTBatch(
        context_input_ids=context_input_ids,
        context_attention_mask=torch.ones(batch_size, context_len, dtype=torch.long),
        input_ids=query_input_ids,
        attention_mask=torch.ones(batch_size, query_len, dtype=torch.long),
        labels=labels,
    )


def test_compute_doc_sft_loss_is_finite_and_differentiable_and_isolates_interpreter_grad():
    interpreter, layers, hypernetwork, vocab_size = _toy_setup()
    batch = _make_doc_batch(batch_size=3, context_len=6, query_len=5, vocab_size=vocab_size, target_token=2)

    loss = compute_doc_sft_loss(batch, interpreter, hypernetwork, layers)
    assert torch.isfinite(loss)
    loss.backward()

    assert any(p.grad is not None for p in hypernetwork.parameters())
    assert all(p.grad is None for p in interpreter.parameters())


def test_doc_train_step_updates_hypernetwork_parameters():
    interpreter, layers, hypernetwork, vocab_size = _toy_setup()
    batch = _make_doc_batch(batch_size=2, context_len=6, query_len=5, vocab_size=vocab_size, target_token=2)
    pristine = {k: v.clone() for k, v in hypernetwork.state_dict().items()}

    optimizer = torch.optim.AdamW(hypernetwork.parameters(), lr=1e-2)
    loss = doc_train_step([batch], interpreter, hypernetwork, layers, optimizer)

    assert isinstance(loss, float)
    moved = any(not torch.equal(pristine[k], v) for k, v in hypernetwork.state_dict().items())
    assert moved


def test_train_doc_downstream_hypernetwork_reduces_loss_on_an_easy_target():
    interpreter, layers, hypernetwork, vocab_size = _toy_setup(adapter="activation_steering", target_modules=("block",))
    nn.init.normal_(hypernetwork.heads["block"].weight, std=0.05)
    batch = _make_doc_batch(batch_size=4, context_len=6, query_len=6, vocab_size=vocab_size, target_token=3)

    stats = train_doc_downstream_hypernetwork(
        hypernetwork, interpreter, layers, [batch], steps=60, learning_rate=1e-2
    )

    assert stats.steps == 60
    assert len(stats.losses) == 60
    assert all(torch.isfinite(torch.tensor(loss)) for loss in stats.losses)
    assert stats.final_loss < stats.initial_loss
