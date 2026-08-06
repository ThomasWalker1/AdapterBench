"""Document-conditioned counterpart to `condition_encoder.py`'s pooled task-description
embedding: instead of a small frozen encoder producing one vector per example, this reads
the frozen *interpreter's own* representation of a raw document and produces the
(batch, task_dim) vector `PooledVectorConditioner` would have produced, via a small
hand-rolled Perceiver-IO cross-attention stack. No `transformers` VLM-specific Perceiver
import (e.g. Idefics2) - just `nn.MultiheadAttention` cross/self-attention blocks.

The shipped path is the early-exit + Perceiver-IO design
(`capture_early_exit_representation` + `EarlyExitPerceiverConditioner`): the document is
run through the interpreter's first `L//4` decoder layers under no_grad, and the resulting
latents are fed as a plain tensor into the trainable hypernetwork - the same "frozen
encoder under no_grad, output fed as a tensor" pattern `condition_encoder.py` uses for task
descriptions. (An earlier full-depth per-layer conditioner was tried and removed; it did
not learn held-out NIAH retrieval - see PROJECT_PLAN.md's D2A section.)
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


class EarlyExitRepresentation(NamedTuple):
    """The raw_condition container for `EarlyExitPerceiverConditioner`: a document's
    *single* early-exit representation (one hidden state per token, from a shallow slice
    of the frozen interpreter) plus its padding mask. It has no `num_layers` axis
    - the whole point of the early-exit encoder is one
    representation, not a per-decoder-layer stack (see `capture_early_exit_representation`).
    """

    hidden_states: Tensor  # (batch, seq_len, hidden_size)
    attention_mask: Tensor  # (batch, seq_len), 1 = real token, 0 = padding


@torch.no_grad()
def capture_early_exit_representation(
    interpreter: nn.Module, input_ids: Tensor, attention_mask: Tensor, exit_layer: int
) -> Tensor:
    """Run only the frozen interpreter's first `exit_layer` decoder layers on a document
    and return the resulting single `(batch, seq_len, hidden_size)` hidden state, cast to
    float32. This is AdapterBench's port of Doc-to-LoRA's `ctx_encoder.EarlyExit` (its
    NIAH recipe uses `ctx_encoder_type=early_exit` at `layer_idx = num_layers // 4`) - a
    much cheaper, and empirically load-bearing, context encoder than the full per-layer
    activation stack the earlier (removed) full-depth capture produces (see PROJECT_PLAN.md's D2A
    section: the early-exit + Perceiver-IO generation path is what finally learns held-out
    NIAH retrieval, where the per-layer conditioner did not).

    Temporarily truncates the interpreter's decoder-layer list to `[:exit_layer]` for one
    forward pass, then restores it in a `finally` (so a raised exception can't leave the
    interpreter permanently shortened). `output.last_hidden_state` is the post-final-norm
    output after `exit_layer` layers. Same `@torch.no_grad()` + float32-cast + `.detach()`
    contract as the earlier (removed) full-depth capture (the interpreter is frozen; the returned
    tensor carries no autograd history and matches the from-scratch conditioner's float32).
    """
    base_model = interpreter.model if hasattr(interpreter, "model") else interpreter
    saved_layers = base_model.layers
    try:
        base_model.layers = saved_layers[:exit_layer]
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        base_model.layers = saved_layers
    return outputs.last_hidden_state.float().detach()


class _PerceiverIOBlock(nn.Module):
    """One Perceiver-IO cross-attention block: pre-normed cross-attention (a set of
    queries attend onto key/value tokens, masked at padded positions) + residual, then a
    pre-normed residual feed-forward. Used for both the encoder blocks (learned latents
    attend onto the document's early-exit tokens) and the decoder block (per-layer output
    queries attend onto the refined latents). Unlike a plain latent-self-attention + same-width cross-attention block (which pairs a
    self-attention among latents with a cross-attention onto same-width token activations),
    this projects the key/value width to the latent width up front so the document tokens
    and latents can differ in dimensionality - and it drops the latent self-attention
    (upstream's NIAH recipe sets `num_self_attn_per_block=0`).
    """

    def __init__(self, latent_dim: int, num_heads: int, dropout: float = 0.05):
        super().__init__()
        self.q_norm = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * latent_dim, latent_dim)
        )

    def forward(self, queries: Tensor, kv: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        attended, _ = self.attn(self.q_norm(queries), kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        queries = queries + attended
        return queries + self.ff(self.ff_norm(queries))


class EarlyExitPerceiverConditioner(nn.Module):
    """Doc-to-LoRA-parity document conditioner: an early-exit context encoder + a
    Perceiver-IO cross-attention stack, producing the same `(batch, task_dim)` per-layer
    output `PooledVectorConditioner` produces, so it slots
    into `TextToPeftHypernetwork` (via `conditioner=`) and the existing trunk/heads/codec
    seam unchanged so additional codecs plug in without conditioner-specific plumbing.

    This is the generation path that empirically learns held-out NIAH retrieval where an
    earlier full-depth per-layer conditioner (since removed) did not (see PROJECT_PLAN.md's
    D2A section). It differs from that earlier conditioner in two load-bearing ways, both
    ported from upstream's `scripts/niah/1-train.sh` recipe:

    1. **Early-exit context encoder** (`ctx_encoder_type=early_exit`, `layer_idx=L//4`):
       the document is encoded by the frozen interpreter's first `exit_layer` decoder
       layers into one `(batch, seq_len, hidden)` representation
       (`capture_early_exit_representation`), not the full per-decoder-layer stack.
    2. **Larger Perceiver-IO** (`n_latents=208`, `num_blocks=8`): `n_latents` learned
       latent queries cross-attend onto the early-exit document tokens over `num_blocks`
       blocks (the `encode` step, done once per document); then per-layer output queries
       cross-attend onto the refined latents (the `forward` step, done per layer) to
       produce each layer's `(batch, task_dim)` conditioning vector.

    Call convention (matches the other conditioners so `TextToPeftHypernetwork.
    generate_per_layer` needs no special-casing): the `raw_condition` threaded into
    `generate_per_layer` -> `self.conditioner(raw_condition, layer_index=i)` is the
    *already-encoded* latent tensor `(batch, n_latents, latent_dim)` that `encode`
    produced, NOT the raw document. `prepare_condition` runs the (expensive) early-exit
    capture + `encode` exactly once and returns those latents; `forward(latents,
    layer_index=i)` then only runs the (cheap) per-layer decoder cross-attention. This
    keeps the expensive cross-attention over document tokens out of
    `generate_per_layer`'s per-layer loop, which would otherwise redo it `num_layers`
    times per call.
    """

    def __init__(
        self,
        hidden_size: int,
        task_dim: int,
        num_layers: int,
        *,
        exit_layer: int,
        n_latents: int = 208,
        num_blocks: int = 8,
        latent_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,  # Doc-to-LoRA-parity: no dropout (upstream lora_dropout=0.0); the validated probe used 0.0
        seed: int = 777,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.exit_layer = exit_layer
        self.latent_dim = latent_dim
        generator = torch.Generator().manual_seed(seed)
        self.latents = nn.Parameter(torch.randn(n_latents, latent_dim, generator=generator) * latent_dim**-0.5)
        self.output_queries = nn.Parameter(
            torch.randn(num_layers, latent_dim, generator=generator) * latent_dim**-0.5
        )
        self.input_projection = nn.Linear(hidden_size, latent_dim)
        self.encoder_blocks = nn.ModuleList(
            [_PerceiverIOBlock(latent_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.decoder_block = _PerceiverIOBlock(latent_dim, num_heads, dropout)
        self.output_projection = nn.Sequential(nn.Linear(latent_dim, task_dim), nn.LayerNorm(task_dim))

    def encode(self, representation: "EarlyExitRepresentation") -> Tensor:
        """Cross-attend `n_latents` learned latents onto the document's early-exit token
        representation over every encoder block, returning the refined latents
        `(batch, n_latents, latent_dim)`. Trainable (the latents/blocks are hypernetwork
        parameters); the early-exit capture upstream of it is frozen + no_grad. Run once
        per document by `prepare_condition`, not per layer."""
        kv = self.input_projection(representation.hidden_states)
        # nn.MultiheadAttention key_padding_mask: True = ignore (inverse of 1=real mask).
        key_padding_mask = ~representation.attention_mask.bool()
        batch = kv.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1).contiguous()
        for block in self.encoder_blocks:
            latents = block(latents, kv, key_padding_mask)
        return latents

    def forward(self, latents: Tensor, layer_index: int | None = None) -> Tensor:
        """Decode the (already-encoded) latents into conditioning vectors. `layer_index=i`
        (the `generate_per_layer`/`forward_layer` path) uses layer `i`'s own output query,
        returning `(batch, task_dim)`; `layer_index=None` (the `forward` path) decodes
        every layer's output query at once and mean-pools across layers into one summary
        `(batch, task_dim)` - the same broadcast-vs-per-layer split
        the earlier (removed) per-layer conditioner documents, so the batched `forward` stays a valid
        (cheaper, layer-invariant) path and `generate_per_layer` is the layer-faithful one.
        """
        batch = latents.shape[0]
        if layer_index is None:
            queries = self.output_queries.unsqueeze(0).expand(batch, -1, -1)  # (batch, num_layers, latent_dim)
            decoded = self.decoder_block(queries, latents, None)
            return self.output_projection(decoded.mean(dim=1))
        query = self.output_queries[layer_index].unsqueeze(0).unsqueeze(1).expand(batch, -1, -1)  # (batch, 1, dim)
        decoded = self.decoder_block(query, latents, None)
        return self.output_projection(decoded.squeeze(1))

    def prepare_condition(self, interpreter: nn.Module, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Produce the `raw_condition` (already-encoded latents) that this conditioner's
        `forward` expects: run the frozen early-exit capture, then `encode`. Called once
        per document by `document_sft_trainer`/`live_evaluator` (parallel to
        `EarlyExitPerceiverConditioner.prepare_condition`), so the per-layer
        `generate_per_layer` loop only re-runs the cheap decoder cross-attention, never
        the expensive document encode."""
        representation = capture_early_exit_representation(interpreter, input_ids, attention_mask, self.exit_layer)
        return self.encode(EarlyExitRepresentation(hidden_states=representation, attention_mask=attention_mask))
