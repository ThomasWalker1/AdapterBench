"""Live end-to-end SFT training: the hypernetwork's generated output is hooked into a
real frozen interpreter's forward pass on real training examples, and next-token
cross-entropy loss is backpropagated through the hook into the hypernetwork. No oracle
adapters, no reconstruction matching — mirrors Sakana's own actual training method
(``hyper_llm_modulator/sft_trainer.py``/``hooks.py``), built on our generalized
``TextToPeftHypernetwork.apply()`` (hookable at either a named linear submodule or a
whole decoder layer, see ``codecs.py``/``hypernetwork.py``) rather than their
LoRA-specific hook code, so any representation trains the same way.
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
    batch: SFTBatch,
    interpreter: nn.Module,
    hypernetwork: TextToPeftHypernetwork,
    layers: nn.ModuleList,
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float = 1.0,
    l2_reg_generated_w: float = 0.0,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = compute_sft_loss(batch, interpreter, hypernetwork, layers, l2_reg_generated_w=l2_reg_generated_w)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(hypernetwork.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.detach())


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
) -> SFTTrainStats:
    """Interpreter params must already be frozen (``requires_grad=False``) by the
    caller — the optimizer is built over ``hypernetwork.parameters()`` only, so nothing
    else trains regardless, but skipping the freeze wastes memory on unused grad
    buffers. ``train_batches`` may be a one-shot iterable (e.g. a small in-memory list
    for a pilot) — cycled via `itertools.cycle` to run exactly ``steps`` steps."""
    optimizer = torch.optim.AdamW(hypernetwork.parameters(), lr=learning_rate)
    losses = [
        train_step(
            batch, interpreter, hypernetwork, layers, optimizer,
            max_grad_norm=max_grad_norm, l2_reg_generated_w=l2_reg_generated_w,
        )
        for batch in itertools.islice(itertools.cycle(train_batches), steps)
    ]
    return SFTTrainStats(initial_loss=losses[0], final_loss=losses[-1], steps=steps, losses=tuple(losses))
