"""Live end-to-end SFT training for document-conditioned hypernetworks (NIAH): the
generated adapter is hooked into a real frozen interpreter's forward pass on real
query/answer examples, exactly like `sft_trainer.py`'s task-description-conditioned
training loop, but conditioned on `capture_document_activations`'s own per-layer
document token activations instead of a pooled task-description embedding.

`sft_trainer.py`'s `train_step`/`train_downstream_hypernetwork` are hardcoded to call
`compute_sft_loss(batch: SFTBatch, ...)` directly rather than accepting a
loss-computing callable, and this project's Setting-2 integration constraints say not
to modify those functions (even to add an optional `loss_fn` parameter with a
default that would preserve every existing call site's behavior) - so this module
reimplements the grad-accumulation/warmup-scheduling *loop shape* for `DocSFTBatch`
rather than parameterizing `sft_trainer.py`'s loop functions. It does reuse the two
conditioning-agnostic pieces those functions delegate to (`masked_cross_entropy`,
`_linear_warmup_then_constant`) instead of duplicating their logic, and returns the
same `SFTTrainStats` shape, so downstream callers (the `d2p-sft-pilot` CLI command,
tests) see an interface identical to `sft_trainer.py`'s.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .document_conditioning import DocumentActivations, capture_document_activations
from .hypernetwork import TextToPeftHypernetwork
from .niah_data import DocSFTBatch
from .sft_trainer import SFTTrainStats, _linear_warmup_then_constant, masked_cross_entropy


def compute_doc_sft_loss(
    batch: DocSFTBatch,
    interpreter: nn.Module,
    hypernetwork: TextToPeftHypernetwork,
    layers: nn.ModuleList,
    *,
    l2_reg_generated_w: float = 0.0,
    equally_weight_sample: bool = True,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Document-conditioned counterpart to `sft_trainer.py::compute_sft_loss`.

    Captures the frozen interpreter's own per-layer activations on
    `batch.context_input_ids`/`batch.context_attention_mask` (the haystack+needle
    document - a `@torch.no_grad()` pass, see `capture_document_activations`),
    conditions the hypernetwork on those instead of a pooled task-description
    embedding, then hooks the generated adapter into a *second*, gradient-tracked
    interpreter forward pass over `batch.input_ids` (the query) - the same
    hook-then-forward pattern `compute_sft_loss` uses, just with a different
    conditioning input and two interpreter passes (context capture, then query
    scoring) instead of one.

    Uses `hypernetwork.generate_per_layer(raw_condition)` (not the batched
    `hypernetwork(raw_condition)`) so a layer-index-aware conditioner (e.g.
    `DocumentPerceiverConditioner`) actually conditions each layer's generated adapter
    on that layer's own document-activation cross-attention, rather than every layer
    sharing one cross-layer-pooled summary vector - see `TextToPeftHypernetwork.
    generate_per_layer`/`forward_layer`'s docstrings for why this differs from
    `forward`. Costs `num_layers` trunk forward passes per training step instead of 1.
    """
    doc_activations = capture_document_activations(interpreter, batch.context_input_ids, batch.context_attention_mask)
    raw_condition = DocumentActivations(hidden_states=doc_activations, attention_mask=batch.context_attention_mask)
    generated = hypernetwork.generate_per_layer(raw_condition)
    with hypernetwork.apply(layers, generated):
        outputs = interpreter(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    loss = masked_cross_entropy(
        outputs.logits, batch.labels, equally_weight_sample=equally_weight_sample, label_smoothing=label_smoothing
    )
    if l2_reg_generated_w:
        # Port of upstream's `--l2_reg_generated_w` (see sft_trainer.py::compute_sft_loss) -
        # a weight-decay-like penalty on the generated output's own magnitude.
        reg = torch.stack([value.float().pow(2).mean() for value in generated.values()]).mean()
        loss = loss + l2_reg_generated_w * reg
    return loss


def doc_train_step(
    batches: list[DocSFTBatch],
    interpreter: nn.Module,
    hypernetwork: TextToPeftHypernetwork,
    layers: nn.ModuleList,
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float = 1.0,
    l2_reg_generated_w: float = 0.0,
) -> float:
    """Document-conditioned counterpart to `sft_trainer.py::train_step` - identical
    grad-accumulation shape (loss divided by ``len(batches)`` before each
    ``backward()``, so accumulated gradients are averaged, not summed), calling
    `compute_doc_sft_loss` instead of `compute_sft_loss`."""
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for batch in batches:
        loss = compute_doc_sft_loss(batch, interpreter, hypernetwork, layers, l2_reg_generated_w=l2_reg_generated_w)
        (loss / len(batches)).backward()
        losses.append(float(loss.detach()))
    torch.nn.utils.clip_grad_norm_(hypernetwork.parameters(), max_grad_norm)
    optimizer.step()
    return sum(losses) / len(losses)


def train_doc_downstream_hypernetwork(
    hypernetwork: TextToPeftHypernetwork,
    interpreter: nn.Module,
    layers: nn.ModuleList,
    train_batches: Iterable[DocSFTBatch],
    *,
    steps: int,
    learning_rate: float,
    max_grad_norm: float = 1.0,
    l2_reg_generated_w: float = 0.0,
    grad_accum_steps: int = 1,
    warmup_steps: int = 0,
) -> SFTTrainStats:
    """Document-conditioned counterpart to
    `sft_trainer.py::train_downstream_hypernetwork` - identical step-budget/
    grad-accumulation/warmup-scheduling shape (reuses `_linear_warmup_then_constant`
    unchanged), operating on `DocSFTBatch` via `doc_train_step` instead of `SFTBatch`
    via `train_step`. Interpreter params must already be frozen by the caller, exactly
    as `train_downstream_hypernetwork` requires."""
    optimizer = torch.optim.AdamW(hypernetwork.parameters(), lr=learning_rate)
    scheduler = _linear_warmup_then_constant(optimizer, warmup_steps)
    batch_iter = itertools.cycle(train_batches)
    losses = []
    for _ in range(steps):
        micro_batches = list(itertools.islice(batch_iter, grad_accum_steps))
        losses.append(
            doc_train_step(
                micro_batches, interpreter, hypernetwork, layers, optimizer,
                max_grad_norm=max_grad_norm, l2_reg_generated_w=l2_reg_generated_w,
            )
        )
        scheduler.step()
    return SFTTrainStats(initial_loss=losses[0], final_loss=losses[-1], steps=steps, losses=tuple(losses))
