"""Representation-neutral Text-to-LoRA-style reconstruction training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .generators import reconstruction_loss


@dataclass(frozen=True)
class ReconstructionStats:
    initial_loss: float
    final_loss: float
    best_validation_loss: float
    steps: int
    nonfinite_steps: int


def train_reconstruction_generator(
    generator: nn.Module,
    train_conditions: Tensor,
    train_states: Tensor,
    validation_conditions: Tensor,
    validation_states: Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> ReconstructionStats:
    if train_states.shape[-1] != validation_states.shape[-1]:
        raise ValueError("train and validation adapters use different state layouts")
    device = next(generator.parameters()).device
    train_conditions, train_states = train_conditions.to(device), train_states.to(device)
    validation_conditions, validation_states = validation_conditions.to(device), validation_states.to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate)
    random = torch.Generator(device=device).manual_seed(seed)
    losses = []
    best = float("inf")
    nonfinite = 0
    for step in range(steps):
        indices = torch.randint(len(train_conditions), (batch_size,), generator=random, device=device)
        loss = reconstruction_loss(generator(train_conditions[indices]), train_states[indices])
        if not torch.isfinite(loss):
            nonfinite += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 25 == 0 or step + 1 == steps:
            with torch.no_grad():
                value = reconstruction_loss(generator(validation_conditions), validation_states)
            best = min(best, float(value))
    if not losses:
        raise RuntimeError("reconstruction training produced no finite updates")
    return ReconstructionStats(losses[0], losses[-1], best, steps, nonfinite)

