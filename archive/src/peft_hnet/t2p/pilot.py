"""Phase 3 pilot: leave-one-task-out dense-ΔW reconstruction against real oracle LoRAs.

Trains a ``TextToPeftHypernetwork`` (any non-LoRA-only representation, e.g. FourierFT) to
regress each layer's generated adapter onto the real oracle LoRA's ``ΔW = B @ A`` via
``LinearUpdateCodec.dense_delta``, using upstream's own loss convention
(``F.l1_loss(deltaW, target_deltaW * delta_w_scaling)``, see ``recon_trainer.py``). No
forward pass through the interpreter model is needed: this is pure parameter-space
regression against tensors already on disk.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .hypernetwork import TextToPeftHypernetwork


@dataclass(frozen=True)
class PilotFoldStats:
    held_out_task_id: str
    train_task_ids: tuple[str, ...]
    initial_train_loss: float
    final_train_loss: float
    held_out_loss: float
    zero_baseline_loss: float
    steps: int


def _stack_targets(
    task_ids: Sequence[str],
    oracle_targets: Mapping[str, Mapping[str, Tensor]],
    module_names: Sequence[str],
    *,
    device: torch.device,
    delta_w_scaling: float,
) -> dict[str, Tensor]:
    """Return {module: Tensor(num_layers, num_tasks, out_features, in_features)}, scaled and on ``device``.

    Callers are expected to keep ``oracle_targets`` resident on CPU between folds and
    call this once per fold (not once per training step) — at real Mistral-7B dimensions
    even one fold's stack is large (7 tasks x 32 layers x 4096x4096 float32 for q_proj
    alone is ~15GB), so moving it to an accelerator on every step would repeatedly
    re-transfer identical data for no benefit.
    """
    return {
        module: torch.stack([oracle_targets[task_id][module] for task_id in task_ids], dim=1).to(device)
        * delta_w_scaling
        for module in module_names
    }


def _mean_l1_loss(
    hypernetwork: TextToPeftHypernetwork, condition_embeddings: Tensor, target_stack: Mapping[str, Tensor]
) -> Tensor:
    outputs = hypernetwork(condition_embeddings)
    per_layer_losses = []
    for module, codec in hypernetwork.codecs.items():
        generated = outputs[module]  # (num_layers, num_tasks, output_size)
        target = target_stack[module]  # (num_layers, num_tasks, out_features, in_features), pre-scaled
        for layer_index in range(hypernetwork.num_layers):
            predicted = codec.dense_delta(generated[layer_index], layer_index)
            per_layer_losses.append(F.l1_loss(predicted, target[layer_index]))
    return torch.stack(per_layer_losses).mean()


def _train_step(
    hypernetwork: TextToPeftHypernetwork,
    condition_embeddings: Tensor,
    target_stack: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    """One full-batch training step, computing and backpropagating one layer at a time.

    Dense-ΔW codecs (FourierFT in particular) materialize an (out_features, in_features)
    tensor per task per layer — at real Mistral-7B dimensions (q_proj: 4096x4096) that's
    ~1GB per layer just for the complex spectrum, times gradients. Computing every layer
    via one batched ``hypernetwork(condition_embeddings)`` call and backpropagating a
    single combined loss (the natural way to write this) keeps every layer's dense-ΔW
    graph alive simultaneously for one shared backward pass — confirmed via profiling to
    blow past 50GB for one training step on the real 32-layer/4096-dim problem, and
    ``retain_graph=True`` on a shared-then-split graph does *not* fix this (it disables
    buffer freeing for the entire graph reachable from each backward call, including that
    layer's own large FFT buffers, not just the shared trunk). The actual fix is
    ``TextToPeftHypernetwork.forward_layer``: each layer gets its own independent forward
    pass (recomputing the cheap trunk once per layer instead of once total), so ordinary
    (non-retained) backward frees that layer's graph completely before the next layer is
    even computed — bounds peak memory to roughly one layer's worth regardless of
    ``num_layers``.
    """
    optimizer.zero_grad(set_to_none=True)
    num_layers = hypernetwork.num_layers
    total_loss = 0.0
    for layer_index in range(num_layers):
        # All modules for this layer share one forward_layer() call/graph, so they must
        # be combined into a single backward() — backwarding per module here would try
        # to traverse the shared trunk graph a second time (already freed by the first).
        layer_outputs = hypernetwork.forward_layer(condition_embeddings, layer_index)
        module_losses = [
            F.l1_loss(codec.dense_delta(layer_outputs[module], layer_index), target_stack[module][layer_index])
            for module, codec in hypernetwork.codecs.items()
        ]
        layer_loss = torch.stack(module_losses).mean()
        (layer_loss / num_layers).backward()
        total_loss += float(layer_loss.detach()) / num_layers
    torch.nn.utils.clip_grad_norm_(hypernetwork.parameters(), 1.0)
    optimizer.step()
    return total_loss


def _zero_baseline_loss(target_stack: Mapping[str, Tensor]) -> float:
    losses = [target.abs().mean() for target in target_stack.values()]
    return float(torch.stack(losses).mean())


def train_dense_delta_hypernetwork(
    hypernetwork: TextToPeftHypernetwork,
    train_task_ids: Sequence[str],
    condition_embeddings: Mapping[str, Tensor],
    oracle_targets: Mapping[str, Mapping[str, Tensor]],
    *,
    steps: int,
    learning_rate: float,
    delta_w_scaling: float,
) -> tuple[float, float]:
    """Full-batch train ``hypernetwork`` on ``train_task_ids``. Returns (initial, final) loss."""
    module_names = tuple(hypernetwork.codecs)
    embeddings = torch.stack([condition_embeddings[task_id] for task_id in train_task_ids], dim=0)
    target_stack = _stack_targets(
        train_task_ids, oracle_targets, module_names, device=embeddings.device, delta_w_scaling=delta_w_scaling
    )
    optimizer = torch.optim.AdamW(hypernetwork.parameters(), lr=learning_rate)
    losses = [_train_step(hypernetwork, embeddings, target_stack, optimizer) for _ in range(steps)]
    return losses[0], losses[-1]


@torch.no_grad()
def evaluate_held_out_task(
    hypernetwork: TextToPeftHypernetwork,
    held_out_task_id: str,
    condition_embeddings: Mapping[str, Tensor],
    oracle_targets: Mapping[str, Mapping[str, Tensor]],
    *,
    delta_w_scaling: float,
) -> tuple[float, float]:
    """Returns (hypernetwork_loss, zero_baseline_loss) for the held-out task."""
    module_names = tuple(hypernetwork.codecs)
    embeddings = condition_embeddings[held_out_task_id].unsqueeze(0)
    target_stack = _stack_targets(
        [held_out_task_id], oracle_targets, module_names, device=embeddings.device, delta_w_scaling=delta_w_scaling
    )
    hypernetwork_loss = float(_mean_l1_loss(hypernetwork, embeddings, target_stack))
    zero_loss = _zero_baseline_loss(target_stack)
    return hypernetwork_loss, zero_loss


def run_leave_one_out_pilot(
    hypernetwork_factory: Callable[[], TextToPeftHypernetwork],
    task_ids: Sequence[str],
    condition_embeddings: Mapping[str, Tensor],
    oracle_targets: Mapping[str, Mapping[str, Tensor]],
    *,
    steps: int,
    learning_rate: float,
    delta_w_scaling: float,
    held_out_task_ids: Sequence[str] | None = None,
) -> list[PilotFoldStats]:
    """Train+evaluate one fold per task in ``held_out_task_ids`` (default: all of ``task_ids``).

    Each fold's "other 7" train split is always drawn from the full ``task_ids`` universe
    regardless of which subset is requested here — this parameter only selects which
    fold(s) *this call* computes, so independent folds can be split across separate
    processes/GPUs (each given a different, disjoint slice of ``held_out_task_ids`` but
    the same full ``task_ids``) and their results merged afterward.
    """
    stats = []
    for held_out in held_out_task_ids if held_out_task_ids is not None else task_ids:
        train_ids = tuple(t for t in task_ids if t != held_out)
        hypernetwork = hypernetwork_factory()
        initial_loss, final_loss = train_dense_delta_hypernetwork(
            hypernetwork,
            train_ids,
            condition_embeddings,
            oracle_targets,
            steps=steps,
            learning_rate=learning_rate,
            delta_w_scaling=delta_w_scaling,
        )
        held_out_loss, zero_loss = evaluate_held_out_task(
            hypernetwork, held_out, condition_embeddings, oracle_targets, delta_w_scaling=delta_w_scaling
        )
        stats.append(
            PilotFoldStats(
                held_out_task_id=held_out,
                train_task_ids=train_ids,
                initial_train_loss=initial_loss,
                final_train_loss=final_loss,
                held_out_loss=held_out_loss,
                zero_baseline_loss=zero_loss,
                steps=steps,
            )
        )
    return stats
