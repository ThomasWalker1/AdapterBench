"""Document-conditioned counterpart to `condition_encoder.py`'s pooled task-description
embedding: instead of a small frozen encoder producing one vector per example, this reads
the frozen *interpreter's own* per-layer token activations on a raw document and produces
the (batch, task_dim) vector `PooledVectorConditioner` would have produced, via a small
hand-rolled Perceiver-IO-style cross-attention stack (a learned per-layer latent query
cross-attending onto that layer's own token activations). No `transformers` VLM-specific
Perceiver import (e.g. Idefics2) - just `nn.MultiheadAttention` cross/self-attention
blocks, since those add no dependency and are all this needs.

This module has no dependency on a live `interpreter` beyond
`capture_document_activations`'s single forward pass - `DocumentPerceiverConditioner`
itself only ever sees the resulting tensor, exactly the same "small frozen encoder called
under no_grad, output fed as a plain tensor into the trainable hypernetwork" pattern
`condition_encoder.py` already uses for task descriptions.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


class DocumentActivations(NamedTuple):
    """The raw_condition container threaded through
    `TextToPeftHypernetwork`/`DocumentPerceiverConditioner`: a document's per-layer token
    activations plus the padding mask needed to ignore padded positions during
    cross-attention. Bundled into one NamedTuple (rather than passing the mask as a
    separate positional/keyword argument to every conditioner call site) so
    `TextToPeftHypernetwork.forward`/`forward_layer` can pass `condition_embeddings`
    through to `self.conditioner(condition_embeddings, layer_index=...)` completely
    opaquely, without knowing whether the conditioner needs a mask at all.
    """

    hidden_states: Tensor  # (batch, num_layers, seq_len, hidden_size)
    attention_mask: Tensor  # (batch, seq_len), 1 = real token, 0 = padding


@torch.no_grad()
def capture_document_activations(interpreter: nn.Module, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """Run the frozen interpreter on a document and return its per-decoder-layer token
    activations, stacked as `(batch, num_layers, seq_len, hidden_size)`.

    `output_hidden_states=True` on a causal LM returns `num_hidden_layers + 1` tensors:
    `hidden_states[0]` is the raw embedding-layer output (before any decoder layer has
    run), and `hidden_states[i]` for `i >= 1` is decoder layer `i - 1`'s output. This
    drops `hidden_states[0]` and keeps only the `num_hidden_layers` decoder-layer outputs,
    so `num_layers` here matches `len(get_decoder_layers(interpreter))` exactly - the same
    `num_layers` convention `TextToPeftHypernetwork`/`infer_module_shapes` already use
    elsewhere (one generated adapter slice per decoder layer, not per embedding-plus-decoder
    layer). This is a documented simplification: D2L's own interpreter-conditioning recipe
    may pool the embedding layer's output in too, but including it here would silently make
    `hidden_states.shape[1] != num_layers`, which every layer-indexed call site
    (`hypernetwork.apply`, `DocumentPerceiverConditioner.forward(..., layer_index=i)`)
    implicitly assumes is one-to-one with decoder layers.

    Called under `@torch.no_grad()` and every parameter upstream of this call is frozen
    (`interpreter.parameters()` all have `requires_grad=False` in every call site here) -
    the returned tensor carries no autograd history back into `interpreter` regardless;
    `no_grad` is belt-and-suspenders (also saves the activation memory during this pass).
    See `tests/test_document_conditioning.py`'s gradient-isolation test for a direct check.

    Cast to float32 before returning - the interpreter itself typically runs in bf16
    (`_d2p_sft_pilot_command` loads it with `dtype=torch.bfloat16`), but
    `DocumentPerceiverConditioner`'s own parameters are plain float32 (the hypernetwork
    is trained from scratch, not loaded in a reduced-precision dtype), and
    `nn.MultiheadAttention` requires its query/key/value to share one dtype. Same
    float32-cast convention `condition_encoder.py::embed_task_descriptions` already
    uses for its own frozen encoder's output.
    """
    outputs = interpreter(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    decoder_layer_states = outputs.hidden_states[1:]
    return torch.stack(decoder_layer_states, dim=1).float().detach()


class EarlyExitRepresentation(NamedTuple):
    """The raw_condition container for `EarlyExitPerceiverConditioner`: a document's
    *single* early-exit representation (one hidden state per token, from a shallow slice
    of the frozen interpreter) plus its padding mask. Parallel to `DocumentActivations`
    but with no `num_layers` axis - the whole point of the early-exit encoder is one
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
    float32. This is AdapterBench's port of Doc-to-LoRA's `ctx_encoder.EarlyExit` (D2L's
    NIAH recipe uses `ctx_encoder_type=early_exit` at `layer_idx = num_layers // 4`) - a
    much cheaper, and empirically load-bearing, context encoder than the full per-layer
    activation stack `capture_document_activations` produces (see PROJECT_PLAN.md's D2P
    section: the early-exit + Perceiver-IO generation path is what finally learns held-out
    NIAH retrieval, where the per-layer conditioner did not).

    Temporarily truncates the interpreter's decoder-layer list to `[:exit_layer]` for one
    forward pass, then restores it in a `finally` (so a raised exception can't leave the
    interpreter permanently shortened). `output.last_hidden_state` is the post-final-norm
    output after `exit_layer` layers. Same `@torch.no_grad()` + float32-cast + `.detach()`
    contract as `capture_document_activations` (the interpreter is frozen; the returned
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


class _PerceiverBlock(nn.Module):
    """One cross-attention block: latents self-attend among themselves, then cross-attend
    onto the document's own token activations (masked at padded positions), each followed
    by a small residual feed-forward - the same shape as a single Perceiver-IO layer."""

    def __init__(self, latent_dim: int, hidden_size: int, num_heads: int, dropout: float = 0.05):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(latent_dim)
        self.cross_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True, kdim=hidden_size, vdim=hidden_size
        )
        self.cross_attn_norm = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4 * latent_dim, latent_dim)
        )
        self.ff_norm = nn.LayerNorm(latent_dim)

    def forward(self, latents: Tensor, tokens: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        attended, _ = self.self_attn(latents, latents, latents, need_weights=False)
        latents = self.self_attn_norm(latents + attended)
        attended, _ = self.cross_attn(latents, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False)
        latents = self.cross_attn_norm(latents + attended)
        return self.ff_norm(latents + self.ff(latents))


class DocumentPerceiverConditioner(nn.Module):
    """Cross-attention conditioner over a document's own per-layer token activations.

    Implements the same call convention as `PooledVectorConditioner.forward(raw_condition,
    layer_index)` (see `hypernetwork.py`), but `raw_condition` here is a
    `DocumentActivations(hidden_states=(batch, num_layers, seq_len, hidden_size),
    attention_mask=(batch, seq_len))` rather than a plain pooled vector.

    Architecture: one learned latent query per decoder layer
    (`self.query`, shape `(num_layers, n_latent_queries, latent_dim)`), `num_blocks`
    stacked `_PerceiverBlock`s (self-attention among that layer's latents, then
    cross-attention onto that layer's own token activations, masked at padded
    positions), mean-pooled over the latent-query dimension, then linearly projected
    down to `task_dim` - the same output shape `PooledVectorConditioner` produces, so it
    slots into `TextToPeftHypernetwork` unchanged.

    Layer-index awareness (see `TextToPeftHypernetwork.forward` vs `forward_layer`, and
    `PooledVectorConditioner`'s docstring for contrast):

    - `forward_layer(raw_condition, layer_index=i)` is called once per layer by
      `TextToPeftHypernetwork.forward_layer`, so this conditioner uses *that layer's own*
      token activations (`hidden_states[:, i]`) and *that layer's own* learned latent
      query slice (`self.query[i]`) - a genuinely different `(batch, task_dim)` vector
      per layer, richer than `PooledVectorConditioner` (byte-identical regardless of
      `layer_index`, since it only ever has one externally-pooled vector to begin with).
    - `forward(raw_condition, layer_index=None)` is called exactly once by
      `TextToPeftHypernetwork.forward`, which then broadcasts ONE `(batch, task_dim)`
      vector across every layer via `depth_embedding` (`tasks.unsqueeze(0).expand(
      self.num_layers, ...)`). To keep that broadcast contract intact without
      restructuring `forward`'s loop, `layer_index=None` runs the same per-layer
      cross-attention independently for every layer (batched over the layer dimension),
      then mean-pools the resulting per-layer latents *across layers* before the final
      `task_dim` projection - trading away per-layer specificity for a single summary
      vector, in exchange for reusing `forward`'s existing single-task-vector +
      depth-embedding split unchanged. `forward_layer`'s per-layer path is where this
      conditioner's layer-awareness actually pays off; `forward`'s batched path is
      intentionally the same "single pooled-ish vector, depth embedding carries layer
      identity" shape `PooledVectorConditioner` already uses.

      **Fixed (2026-07-07):** both document-conditioned call sites now use the
      layer-faithful path instead of the mean-pooled-broadcast one -
      `document_sft_trainer.py::compute_doc_sft_loss` and
      `live_evaluator.py::DocumentHypernetworkDownstreamEvaluator._active` both call
      `TextToPeftHypernetwork.generate_per_layer(raw_condition)` (which loops
      `forward_layer(raw_condition, layer_index=i)` over every layer and stacks the
      results into `forward`'s own `dict[name -> Tensor(num_layers, batch,
      output_size)]` shape) rather than the batched `hypernetwork(raw_condition)` (i.e.
      `forward`, `layer_index=None`). So every codec trained or scored against a
      document now conditions on that layer's own cross-attention result, not a
      cross-layer-pooled summary shared by every layer - at the cost of `num_layers`
      trunk forward passes per call instead of 1 (see `generate_per_layer`'s own
      docstring). `sft_trainer.py::compute_sft_loss` (the pooled-vector Text-to-LoRA
      path) still calls the batched `forward` and should keep doing so -
      `PooledVectorConditioner` ignores `layer_index` entirely, so for that conditioner
      the per-layer loop and the batched call are numerically equivalent and the
      batched call is strictly cheaper.
    """

    def __init__(
        self,
        hidden_size: int,
        task_dim: int,
        num_layers: int,
        *,
        n_latent_queries: int = 4,
        latent_dim: int | None = None,
        num_heads: int = 4,
        num_blocks: int = 2,
        seed: int = 777,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.n_latent_queries = n_latent_queries
        self.latent_dim = latent_dim if latent_dim is not None else hidden_size
        generator = torch.Generator().manual_seed(seed)
        self.query = nn.Parameter(
            torch.randn(num_layers, n_latent_queries, self.latent_dim, generator=generator) * (self.latent_dim**-0.5)
        )
        self.blocks = nn.ModuleList(
            [_PerceiverBlock(self.latent_dim, hidden_size, num_heads) for _ in range(num_blocks)]
        )
        self.output_projection = nn.Sequential(nn.Linear(self.latent_dim, task_dim), nn.LayerNorm(task_dim))

    def _run_layer(self, tokens: Tensor, key_padding_mask: Tensor | None, layer_index: int) -> Tensor:
        """Cross-attend one layer's own latent query onto that layer's own token
        activations. Returns `(batch, n_latent_queries, latent_dim)`."""
        batch = tokens.shape[0]
        latents = self.query[layer_index].unsqueeze(0).expand(batch, -1, -1).contiguous()
        for block in self.blocks:
            latents = block(latents, tokens, key_padding_mask)
        return latents

    def forward(self, raw_condition: DocumentActivations, layer_index: int | None = None) -> Tensor:
        hidden_states, attention_mask = raw_condition.hidden_states, raw_condition.attention_mask
        # nn.MultiheadAttention's key_padding_mask convention: True means "ignore this
        # position" - the inverse of our 1=real/0=padding attention_mask.
        key_padding_mask = ~attention_mask.bool()

        if layer_index is not None:
            latents = self._run_layer(hidden_states[:, layer_index], key_padding_mask, layer_index)
            return self.output_projection(latents.mean(dim=1))

        per_layer_summary = torch.stack(
            [self._run_layer(hidden_states[:, i], key_padding_mask, i).mean(dim=1) for i in range(self.num_layers)],
            dim=0,
        )  # (num_layers, batch, latent_dim)
        return self.output_projection(per_layer_summary.mean(dim=0))

    def prepare_condition(self, interpreter: nn.Module, input_ids: Tensor, attention_mask: Tensor) -> DocumentActivations:
        """Produce the `raw_condition` this conditioner's `forward`/`forward_layer`
        expect, from a live interpreter + document token ids. For this conditioner that is
        the full per-decoder-layer activation stack (`capture_document_activations`) bundled
        with its mask - byte-identical to what `document_sft_trainer`/`live_evaluator` built
        inline before `prepare_condition` existed, so behavior is unchanged. Both
        document-conditioned call sites now go through `conditioner.prepare_condition(...)`
        so the early-exit conditioner (which needs a different, cheaper capture) can slot in
        without either call site knowing which conditioner it holds - see
        `EarlyExitPerceiverConditioner.prepare_condition`.
        """
        activations = capture_document_activations(interpreter, input_ids, attention_mask)
        return DocumentActivations(hidden_states=activations, attention_mask=attention_mask)


class _PerceiverIOBlock(nn.Module):
    """One Perceiver-IO cross-attention block: pre-normed cross-attention (a set of
    queries attend onto key/value tokens, masked at padded positions) + residual, then a
    pre-normed residual feed-forward. Used for both the encoder blocks (learned latents
    attend onto the document's early-exit tokens) and the decoder block (per-layer output
    queries attend onto the refined latents). Unlike `_PerceiverBlock` above (which pairs a
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
    output `PooledVectorConditioner`/`DocumentPerceiverConditioner` produce, so it slots
    into `TextToPeftHypernetwork` (via `conditioner=`) and the existing trunk/heads/codec
    seam unchanged so additional codecs plug in without conditioner-specific plumbing.

    This is the generation path that empirically learns held-out NIAH retrieval where the
    per-layer `DocumentPerceiverConditioner` did not (see PROJECT_PLAN.md's D2P section).
    It differs from `DocumentPerceiverConditioner` in two load-bearing ways, both ported
    from upstream's `scripts/niah/1-train.sh` recipe:

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
        dropout: float = 0.0,  # D2L-parity: no dropout (upstream lora_dropout=0.0); the validated probe used 0.0
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
        `DocumentPerceiverConditioner` documents, so the batched `forward` stays a valid
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
        `DocumentPerceiverConditioner.prepare_condition`), so the per-layer
        `generate_per_layer` loop only re-runs the cheap decoder cross-attention, never
        the expensive document encode."""
        representation = capture_early_exit_representation(interpreter, input_ids, attention_mask, self.exit_layer)
        return self.encode(EarlyExitRepresentation(hidden_states=representation, attention_mask=attention_mask))
