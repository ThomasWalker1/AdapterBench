"""Live end-to-end SFT training for document-conditioned hypernetworks (NIAH): the
generated adapter is hooked into a real frozen interpreter's forward pass on real
query/answer examples, exactly like `sft_trainer.py`'s task-description-conditioned
training loop, but conditioned on the conditioner's own early-exit
document token activations instead of a pooled task-description embedding.

`sft_trainer.py`'s `train_step`/`train_downstream_hypernetwork` are hardcoded to call
`compute_sft_loss(batch: SFTBatch, ...)` directly rather than accepting a
loss-computing callable, and this project's integration constraints say not
to modify those functions (even to add an optional `loss_fn` parameter with a
default that would preserve every existing call site's behavior) - so this module
reimplements the grad-accumulation/warmup-scheduling *loop shape* for `DocSFTBatch`
rather than parameterizing `sft_trainer.py`'s loop functions. It does reuse the two
conditioning-agnostic pieces those functions delegate to (`masked_cross_entropy`,
`_linear_warmup_then_constant`) instead of duplicating their logic, and returns the
same `SFTTrainStats` shape, so downstream callers (the `d2p-niah` CLI command,
tests) see an interface identical to `sft_trainer.py`'s.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn

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
    document - a `@torch.no_grad()` pass, see `capture_early_exit_representation`),
    conditions the hypernetwork on those instead of a pooled task-description
    embedding, then hooks the generated adapter into a *second*, gradient-tracked
    interpreter forward pass over `batch.input_ids` (the query) - the same
    hook-then-forward pattern `compute_sft_loss` uses, just with a different
    conditioning input and two interpreter passes (context capture, then query
    scoring) instead of one.

    Uses `hypernetwork.generate_per_layer(raw_condition)` (not the batched
    `hypernetwork(raw_condition)`) so a layer-index-aware conditioner (e.g.
    `EarlyExitPerceiverConditioner`) actually conditions each layer's generated adapter
    on that layer's own document-activation cross-attention, rather than every layer
    sharing one cross-layer-pooled summary vector - see `TextToPeftHypernetwork.
    generate_per_layer`/`forward_layer`'s docstrings for why this differs from
    `forward`. Costs `num_layers` trunk forward passes per training step instead of 1.
    """
    raw_condition = hypernetwork.conditioner.prepare_condition(
        interpreter, batch.context_input_ids, batch.context_attention_mask
    )
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


def train_doc_niah_checkpointed(
    hypernetwork: TextToPeftHypernetwork,
    interpreter: nn.Module,
    layers: nn.ModuleList,
    train_items: list,
    *,
    collate,
    device,
    batch_size: int,
    steps: int,
    eval_every: int,
    learning_rate: float,
    evaluate: "callable",
    checkpoint_path,
    max_grad_norm: float = 1.0,
    l2_reg_generated_w: float = 0.0,
    grad_accum_steps: int = 1,
    warmup_steps: int = 0,
    log=print,
) -> list[dict]:
    """Restart-safe document-conditioned NIAH training: train `steps` optimizer steps,
    running `evaluate(hypernetwork, step)` every `eval_every` steps (and once at the end),
    and checkpointing **model + optimizer + scheduler + step + eval history** to
    `checkpoint_path` after every eval. If `checkpoint_path` already exists, resume from it
    (skipping the completed prefix) - so a killed multi-hour run picks up where it left off
    rather than restarting (see PROJECT_PLAN.md's D2P step 3: session teardowns repeatedly
    killed long runs). `evaluate` must return a JSON-serializable dict of metrics (e.g.
    per-family exact-digit `accuracy` and `accuracy_ctxswap`) - it is logged and stored in
    the returned/checkpointed history, never used to gate training (NIAH loss is not a
    retrieval signal - gotcha #9). Returns the eval-history list.

    `train_items` are the raw per-document dicts (a `DocSFTDataset`), NOT pre-collated
    batches: batches are reformed from a **document-level reshuffle every epoch**
    (deterministic per-epoch seed, so resume is exact), then `collate`d and moved to
    `device`. This matters - freezing one shuffle into fixed batches (materialising a
    DataLoader once) starves gradient diversity enough to noticeably delay NIAH's sharp
    retrieval phase transition; reforming groupings each epoch matches upstream / the
    validated probe.
    """
    import random as _random

    import torch

    n = len(train_items)
    n_batches = (n + batch_size - 1) // batch_size

    def _epoch_batches(epoch: int) -> list:
        order = list(range(n))
        _random.Random(10_000 + epoch).shuffle(order)
        return [
            collate([train_items[i] for i in order[b : b + batch_size]]).to(device)
            for b in range(0, n, batch_size)
        ]

    _cache: dict = {}

    def _micro_batch(micro_idx: int):
        epoch, pos = divmod(micro_idx, n_batches)
        if epoch not in _cache:
            _cache.clear()  # keep only the current epoch's batches resident on `device`
            _cache[epoch] = _epoch_batches(epoch)
        return _cache[epoch][pos]

    checkpoint_path = Path(checkpoint_path)
    optimizer = torch.optim.AdamW(hypernetwork.parameters(), lr=learning_rate)
    scheduler = _linear_warmup_then_constant(optimizer, warmup_steps)
    history: list[dict] = []
    start_step = 0
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hypernetwork.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        history = ckpt["history"]
        log(f"[resume] loaded checkpoint at step {start_step} from {checkpoint_path}", flush=True)

    def _save(step):
        tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(
            {
                "model": hypernetwork.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "history": history,
            },
            tmp,
        )
        tmp.replace(checkpoint_path)  # atomic - a kill mid-save can't corrupt the live checkpoint

    hypernetwork.train()
    last_loss = float("nan")
    for step in range(start_step, steps):
        micro_batches = [_micro_batch(step * grad_accum_steps + k) for k in range(grad_accum_steps)]
        last_loss = doc_train_step(
            micro_batches, interpreter, hypernetwork, layers, optimizer,
            max_grad_norm=max_grad_norm, l2_reg_generated_w=l2_reg_generated_w,
        )
        scheduler.step()
        done = step + 1
        if done % eval_every == 0 or done == steps:
            hypernetwork.eval()
            metrics = evaluate(hypernetwork, done)
            hypernetwork.train()
            record = {"step": done, "loss": last_loss, **metrics}
            history.append(record)
            log(f"[step {done}] loss={last_loss:.4f} {metrics}", flush=True)
            _save(done)
    return history
