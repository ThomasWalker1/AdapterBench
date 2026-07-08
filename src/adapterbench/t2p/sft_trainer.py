"""Live end-to-end SFT training: the hypernetwork's generated output is hooked into a
real frozen interpreter's forward pass on real training examples, and next-token
cross-entropy loss is backpropagated through the hook into the hypernetwork. No oracle
adapters, no reconstruction matching — mirrors Sakana's own actual training method
(``hyper_llm_modulator/sft_trainer.py``/``hooks.py``), built on our generalized
``TextToPeftHypernetwork.apply()`` (hookable at either a named linear submodule or a
whole decoder layer, see ``codecs.py``/``hypernetwork.py``) rather than their
LoRA-specific hook code, so any adapter trains the same way.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hypernetwork import TextToPeftHypernetwork


@dataclass(frozen=True)
class SFTBatch:
    input_ids: Tensor  # (batch, seq)
    attention_mask: Tensor  # (batch, seq)
    labels: Tensor  # (batch, seq), -100 on prompt tokens and padding
    condition_embeddings: Tensor  # (batch, condition_dim)

    def to(self, device: torch.device | str) -> "SFTBatch":
        return SFTBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
            condition_embeddings=self.condition_embeddings.to(device),
        )


@dataclass(frozen=True)
class SFTTrainStats:
    initial_loss: float
    final_loss: float
    steps: int
    losses: tuple[float, ...]


def masked_cross_entropy(
    logits: Tensor, labels: Tensor, *, equally_weight_sample: bool = True, label_smoothing: float = 0.0
) -> Tensor:
    """Standard next-token CE with ``label=-100`` masking. Port of upstream's
    ``sft_trainer.compute_loss``: with ``equally_weight_sample`` (default, matches
    upstream), loss is averaged per-*example* (sum of that example's token losses over
    its own non-masked token count) then averaged across the batch — not per-token —
    since a batch mixes tasks with very different response lengths."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    batch, seq_len, vocab = shift_logits.shape
    per_token = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
        label_smoothing=label_smoothing,
    ).view(batch, seq_len)
    mask = (shift_labels != -100).float()
    if equally_weight_sample:
        per_example = per_token.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return per_example.mean()
    return per_token.sum() / mask.sum().clamp_min(1.0)


def compute_sft_loss(
    batch: SFTBatch,
    interpreter: nn.Module,
    hypernetwork: TextToPeftHypernetwork,
    layers: nn.ModuleList,
    *,
    l2_reg_generated_w: float = 0.0,
    equally_weight_sample: bool = True,
    label_smoothing: float = 0.0,
) -> Tensor:
    generated = hypernetwork(batch.condition_embeddings)
    with hypernetwork.apply(layers, generated):
        outputs = interpreter(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    loss = masked_cross_entropy(
        outputs.logits, batch.labels, equally_weight_sample=equally_weight_sample, label_smoothing=label_smoothing
    )
    if l2_reg_generated_w:
        # Port of upstream's `--l2_reg_generated_w`: a weight-decay-like penalty on the
        # generated output's own magnitude, not a comparison against any target.
        reg = torch.stack([value.float().pow(2).mean() for value in generated.values()]).mean()
        loss = loss + l2_reg_generated_w * reg
    return loss


def train_step(
    batches: list[SFTBatch],
    interpreter: nn.Module,
    hypernetwork: TextToPeftHypernetwork,
    layers: nn.ModuleList,
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float = 1.0,
    l2_reg_generated_w: float = 0.0,
) -> float:
    """One optimizer step, accumulating gradients over every micro-batch in ``batches``
    (a single-element list is the plain, non-accumulated case). Loss is divided by
    ``len(batches)`` before each ``backward()`` so accumulated gradients are averaged, not
    summed, across micro-batches - matches upstream's own ``grad_accum_steps`` semantics.
    Returns the mean per-micro-batch loss (each micro-batch is already per-example-averaged
    by ``masked_cross_entropy``, so this mean-of-means equals the true overall mean when
    every micro-batch is the same size)."""
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for batch in batches:
        loss = compute_sft_loss(batch, interpreter, hypernetwork, layers, l2_reg_generated_w=l2_reg_generated_w)
        (loss / len(batches)).backward()
        losses.append(float(loss.detach()))
    torch.nn.utils.clip_grad_norm_(hypernetwork.parameters(), max_grad_norm)
    optimizer.step()
    return sum(losses) / len(losses)


def _linear_warmup_then_constant(optimizer: torch.optim.Optimizer, warmup_steps: int):
    """LR ramps linearly from ~0 to the optimizer's base LR over ``warmup_steps``, then
    stays constant - matches upstream's ``warmup_frac`` (a fraction of total steps spent
    ramping, then flat; no decay schedule is specified in their own training config, so
    none is added here). A no-op (constant LR throughout) when ``warmup_steps <= 0``."""

    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_downstream_hypernetwork(
    hypernetwork: TextToPeftHypernetwork,
    interpreter: nn.Module,
    layers: nn.ModuleList,
    train_batches: Iterable[SFTBatch],
    *,
    steps: int,
    learning_rate: float,
    max_grad_norm: float = 1.0,
    l2_reg_generated_w: float = 0.0,
    grad_accum_steps: int = 1,
    warmup_steps: int = 0,
) -> SFTTrainStats:
    """Interpreter params must already be frozen (``requires_grad=False``) by the
    caller — the optimizer is built over ``hypernetwork.parameters()`` only, so nothing
    else trains regardless, but skipping the freeze wastes memory on unused grad
    buffers. ``train_batches`` may be a one-shot iterable (e.g. a small in-memory list
    for a pilot) — cycled via `itertools.cycle` to run exactly ``steps`` optimizer steps,
    each over ``grad_accum_steps`` micro-batches (1 = no accumulation)."""
    optimizer = torch.optim.AdamW(hypernetwork.parameters(), lr=learning_rate)
    scheduler = _linear_warmup_then_constant(optimizer, warmup_steps)
    batch_iter = itertools.cycle(train_batches)
    losses = []
    for _ in range(steps):
        micro_batches = list(itertools.islice(batch_iter, grad_accum_steps))
        losses.append(
            train_step(
                micro_batches, interpreter, hypernetwork, layers, optimizer,
                max_grad_norm=max_grad_norm, l2_reg_generated_w=l2_reg_generated_w,
            )
        )
        scheduler.step()
    return SFTTrainStats(initial_loss=losses[0], final_loss=losses[-1], steps=steps, losses=tuple(losses))


def train_with_checkpoints(
    hypernetwork: TextToPeftHypernetwork,
    interpreter: nn.Module,
    layers: nn.ModuleList,
    train_batches: Iterable[SFTBatch],
    *,
    checkpoint_steps: Iterable[int],
    learning_rate: float,
    max_grad_norm: float = 1.0,
    l2_reg_generated_w: float = 0.0,
    grad_accum_steps: int = 1,
    warmup_steps: int = 0,
) -> dict[int, SFTTrainStats]:
    """Train to each of ``checkpoint_steps`` (ascending, cumulative - e.g. [100, 200, 400]
    trains 100 steps, then 100 *more* to reach 200 total, not 200 from scratch) under one
    persistent ``AdamW`` optimizer (and one persistent LR schedule, so warmup isn't
    restarted at a checkpoint boundary either), so a caller comparing held-out performance
    across step budgets (e.g. a step-budget sweep) doesn't confound the comparison by
    restarting Adam's momentum/bias-correction warmup at every checkpoint boundary the way
    calling ``train_downstream_hypernetwork`` once per budget would. Returns one
    ``SFTTrainStats`` per checkpoint covering only the steps *since the previous
    checkpoint* (deltas, not cumulative) - a caller wanting the full curve up to a
    checkpoint should concatenate."""
    checkpoint_steps = sorted(checkpoint_steps)
    optimizer = torch.optim.AdamW(hypernetwork.parameters(), lr=learning_rate)
    scheduler = _linear_warmup_then_constant(optimizer, warmup_steps)
    batch_iter = itertools.cycle(train_batches)
    stats_by_checkpoint: dict[int, SFTTrainStats] = {}
    previous = 0
    for checkpoint in checkpoint_steps:
        delta = checkpoint - previous
        if delta <= 0:
            raise ValueError(f"checkpoint_steps must be strictly ascending and positive, got {checkpoint_steps}")
        losses = []
        for _ in range(delta):
            micro_batches = list(itertools.islice(batch_iter, grad_accum_steps))
            losses.append(
                train_step(
                    micro_batches, interpreter, hypernetwork, layers, optimizer,
                    max_grad_norm=max_grad_norm, l2_reg_generated_w=l2_reg_generated_w,
                )
            )
            scheduler.step()
        stats_by_checkpoint[checkpoint] = SFTTrainStats(
            initial_loss=losses[0], final_loss=losses[-1], steps=delta, losses=tuple(losses)
        )
        previous = checkpoint
    return stats_by_checkpoint
