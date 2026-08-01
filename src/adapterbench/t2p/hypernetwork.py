"""One shared Text-to-PEFT hypernetwork shell with swappable output codecs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from operator import attrgetter
from typing import Any

import torch
from torch import Tensor, nn

from .codecs import GeneratedUpdateCodec, make_codec


class PooledVectorConditioner(nn.Module):
    """Default conditioner - reproduces `TextToPeftHypernetwork`'s original,
    pre-refactor behavior exactly: a single pooled per-example vector (e.g. a
    task-description embedding from `condition_encoder.py`), linearly projected and
    layer-normed into `task_dim`. Ignores `layer_index` entirely - every layer sees
    byte-identical conditioning; only `depth_embedding` (see
    `TextToPeftHypernetwork.forward`/`forward_layer`) differentiates one layer's
    generated output from another's. Contrast with
    `document_conditioning.py::EarlyExitPerceiverConditioner`, which *is`
    layer-index-aware (it has genuinely different per-layer activations available to
    condition on, not just one pooled vector) - see that class's docstring.
    """

    def __init__(self, condition_dim: int, task_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(condition_dim, task_dim), nn.LayerNorm(task_dim))

    def forward(self, raw_condition: Tensor, layer_index: int | None = None) -> Tensor:
        return self.encoder(raw_condition)


class ResidualMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 4 * width),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(4 * width, width),
            nn.SiLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.block(inputs)


class TextToPeftHypernetwork(nn.Module):
    """Generate one adapter per task, layer, and target-module type."""

    def __init__(
        self,
        *,
        condition_dim: int | None = None,
        module_shapes: Mapping[str, tuple[int, int]],
        num_layers: int,
        adapter: str,
        latent_dim: int = 512,
        head_dim: int = 2048,
        rank: int = 8,
        alpha: float = 16.0,
        n_frequency: int = 1000,
        steering_scale: float = 1.0,
        ia3_scaling: float = 1.0,
        lokr_scaling: float = 1.0,
        loha_scaling: float = 1.0,
        seed: int = 777,
        conditioner: nn.Module | None = None,
    ):
        super().__init__()
        self.module_names = tuple(module_shapes)
        self.num_layers = num_layers
        self.adapter = adapter
        task_dim, depth_dim, type_dim = latent_dim // 2, latent_dim // 4, latent_dim // 4
        if conditioner is not None:
            # A caller-supplied conditioner must itself output `(batch, task_dim)` with
            # this exact `task_dim = latent_dim // 2` - `self.trunk`'s input width is
            # `task_dim + depth_dim + type_dim == latent_dim` regardless of which
            # conditioner produced the `task_dim` slice. Construct e.g.
            # EarlyExitPerceiverConditioner(..., task_dim=latent_dim // 2, ...) to match.
            self.conditioner = conditioner
        else:
            if condition_dim is None:
                raise ValueError("condition_dim is required when conditioner is not provided")
            self.conditioner = PooledVectorConditioner(condition_dim, task_dim)
        self.depth_embedding = nn.Sequential(nn.Embedding(num_layers, depth_dim), nn.LayerNorm(depth_dim))
        self.type_embedding = nn.Sequential(nn.Embedding(len(module_shapes), type_dim), nn.LayerNorm(type_dim))
        self.trunk = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.SiLU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.SiLU(),
            ResidualMLP(latent_dim),
            ResidualMLP(latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, head_dim),
            nn.SiLU(),
        )
        codecs = {}
        heads = {}
        for index, (name, (in_features, out_features)) in enumerate(module_shapes.items()):
            codec = make_codec(
                adapter,
                in_features,
                out_features,
                num_layers=num_layers,
                rank=rank,
                alpha=alpha,
                n_frequency=n_frequency,
                steering_scale=steering_scale,
                ia3_scaling=ia3_scaling,
                lokr_scaling=lokr_scaling,
                loha_scaling=loha_scaling,
                seed=seed + index,
            )
            codecs[name] = codec
            heads[name] = nn.Linear(head_dim, codec.output_size)
            nn.init.zeros_(heads[name].weight)
            initial_bias = codec.initial_bias()
            if initial_bias is None:
                nn.init.zeros_(heads[name].bias)
            else:
                with torch.no_grad():
                    heads[name].bias.copy_(initial_bias)
        self.codecs = nn.ModuleDict(codecs)
        self.heads = nn.ModuleDict(heads)

    def generated_parameter_count(self) -> int:
        return self.num_layers * sum(codec.output_size for codec in self.codecs.values())

    def _generate_from_tasks_per_layer(self, tasks_per_layer: Tensor) -> dict[str, Tensor]:
        """Shared batched trunk/heads computation used by both ``forward`` and
        ``generate_per_layer``: takes an already layer-differentiated (or
        layer-broadcast) tasks tensor, shape ``(num_layers, batch, task_dim)``, and runs
        the depth/type embedding concatenation + ``self.trunk`` + ``self.heads`` exactly
        once across the whole ``num_layers`` dimension in a single batched call - this is
        the (cheap) part that must never be redone per layer; see ``generate_per_layer``'s
        docstring for why looping this per layer used to cost ~28x more trunk-MLP compute
        for zero benefit.
        """
        batch = tasks_per_layer.shape[1]
        depths = self.depth_embedding(torch.arange(self.num_layers, device=tasks_per_layer.device))
        outputs = {}
        for type_index, name in enumerate(self.module_names):
            types = self.type_embedding(torch.tensor(type_index, device=tasks_per_layer.device)).expand(
                self.num_layers, -1
            )
            features = torch.cat(
                (
                    tasks_per_layer,
                    depths.unsqueeze(1).expand(-1, batch, -1),
                    types.unsqueeze(1).expand(-1, batch, -1),
                ),
                dim=-1,
            )
            outputs[name] = self.heads[name](self.trunk(features))
        return outputs

    def forward(self, condition_embeddings: Any) -> dict[str, Tensor]:
        # `condition_embeddings` is whatever raw form `self.conditioner` expects (a
        # plain (batch, condition_dim) Tensor for the default PooledVectorConditioner;
        # the early-exit latents container for EarlyExitPerceiverConditioner) -
        # `batch` is derived from the conditioner's *output* (always (batch, task_dim))
        # rather than from `condition_embeddings` directly, since the latter's shape is
        # conditioner-specific.
        tasks = self.conditioner(condition_embeddings, layer_index=None)
        batch = tasks.shape[0]
        tasks_per_layer = tasks.unsqueeze(0).expand(self.num_layers, batch, -1)
        return self._generate_from_tasks_per_layer(tasks_per_layer)

    def forward_layer(self, condition_embeddings: Any, layer_index: int) -> dict[str, Tensor]:
        """Generate one layer's parameters only (numerically equal to ``forward(...)[name][layer_index]``
        for `PooledVectorConditioner`; see below for why that equality doesn't generally
        hold for a layer-index-aware conditioner).

        Use this directly (rather than `generate_per_layer`) when a caller actually wants
        to do a per-layer *immediate* ``backward()``: dense-ΔW codecs (e.g. FourierFT)
        materialize an ``(out_features, in_features)`` tensor per task per layer, which is
        large at real model dimensions, and backpropagating through ``forward``'s single
        batched call would keep every layer's dense-ΔW graph alive simultaneously for one
        shared backward pass. Calling this per layer instead — each with its own immediate
        ``backward()`` — lets autograd free one layer's graph before the next layer is even
        computed, at the cost of recomputing the (cheap) trunk forward once per layer
        instead of once total. Neither existing call site actually does this (see
        `generate_per_layer`'s docstring), so `generate_per_layer` no longer calls this
        method — it exists for that future per-layer-backward use case and for
        layer-faithful conditioning (see next paragraph); prefer `generate_per_layer` when
        you want every layer's generated output at once without an immediate backward.

        `layer_index` is forwarded to `self.conditioner` (unlike `forward`, which always
        passes `layer_index=None`) - `PooledVectorConditioner` ignores it (so this method
        stays numerically equal to `forward(...)[name][layer_index]` for that conditioner,
        as documented above), but `EarlyExitPerceiverConditioner` uses it to condition on
        that one layer's own document-token activations instead of a cross-layer-pooled
        summary - see that class's docstring for the full reasoning. That means for a
        layer-index-aware conditioner, `forward_layer(x, i)` and `forward(x)[name][i]` are
        *not* numerically equal in general - `forward_layer` is deliberately the more
        layer-faithful of the two.
        """
        tasks = self.conditioner(condition_embeddings, layer_index=layer_index)
        batch = tasks.shape[0]
        depth = self.depth_embedding(torch.tensor([layer_index], device=tasks.device))
        outputs = {}
        for type_index, name in enumerate(self.module_names):
            types = self.type_embedding(torch.tensor(type_index, device=tasks.device)).unsqueeze(0)
            features = torch.cat(
                (
                    tasks.unsqueeze(0),
                    depth.unsqueeze(1).expand(-1, batch, -1),
                    types.unsqueeze(1).expand(-1, batch, -1),
                ),
                dim=-1,
            )
            outputs[name] = self.heads[name](self.trunk(features))[0]
        return outputs

    def generate_per_layer(self, condition_embeddings: Any) -> dict[str, Tensor]:
        """Drop-in, layer-faithful replacement for ``forward(condition_embeddings)``:
        produces the same ``dict[name -> Tensor(num_layers, batch, output_size)]`` shape
        ``forward`` does, so the result works unchanged anywhere ``forward``'s return
        value is consumed (in particular, ``hypernetwork.apply(layers, generated)``).

        Only `self.conditioner(condition_embeddings, layer_index=i)` is called once per
        layer here - that part genuinely needs to be per-layer for a layer-index-aware
        conditioner (e.g. `document_conditioning.py::EarlyExitPerceiverConditioner`) to
        produce genuinely different per-layer conditioning (`forward` broadcasts one
        cross-layer-pooled vector to every layer instead). The depth/type embedding
        concatenation + `self.trunk` + `self.heads` computation is then run exactly once,
        batched across all `num_layers` conditioner outputs at once (via
        `_generate_from_tasks_per_layer`, the same batched logic `forward` itself uses) -
        *not* once per layer. This method previously called `forward_layer` in a loop,
        which recomputed the trunk from scratch for every layer (~28x wasted trunk-MLP
        compute for the default Qwen3-0.6B interpreter's 28 layers) even though neither
        real call site (`document_sft_trainer.py::compute_doc_sft_loss`,
        `live_evaluator.py::DocumentHypernetworkDownstreamEvaluator._active`) ever does a
        per-layer immediate backward that would have justified that cost - see
        `forward_layer`'s own docstring for when a per-layer immediate backward is
        actually the right tool.

        For `PooledVectorConditioner` this is numerically equal to ``forward(...)`` (up
        to dropout noise), since that conditioner ignores `layer_index` and produces the
        same vector regardless.
        """
        tasks_per_layer = torch.stack(
            [self.conditioner(condition_embeddings, layer_index=i) for i in range(self.num_layers)], dim=0
        )
        return self._generate_from_tasks_per_layer(tasks_per_layer)

    def apply(self, layers: list[nn.Module] | nn.ModuleList, generated: Mapping[str, Tensor]):
        return _apply_codec_hooks(layers, self.codecs, generated)


@contextmanager
def _apply_codec_hooks(
    layers: list[nn.Module] | nn.ModuleList, codecs: Mapping[str, GeneratedUpdateCodec], generated: Mapping[str, Tensor]
):
    """Register one forward hook per (layer, codec) that applies the per-example generated
    adapter live during the frozen interpreter's forward, and remove every hook on exit.

    Shared by ``TextToPeftHypernetwork.apply`` (condition-generated adapters) and
    ``StaticAdapter.apply`` (a single directly-optimized adapter): both attach the identical
    codec at the identical site, differing only in where ``generated`` comes from.
    """
    handles = []
    for layer_index, layer in enumerate(layers):
        for name, codec in codecs.items():
            module = _resolve_target(layer, name)
            parameters = generated[name][layer_index]

            def hook(module, args, output, *, codec=codec, parameters=parameters, layer_index=layer_index):
                # A decoder layer's forward (resolved via "block") returns a tuple
                # (hidden_states, ...); a linear submodule's forward returns a bare
                # tensor. Unwrap/rewrap so codec.apply() only ever sees a tensor.
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output
                updated = codec.apply(args[0], hidden, parameters, layer_index)
                return (updated, *output[1:]) if is_tuple else updated

            handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _resolve_target(layer: nn.Module, name: str) -> nn.Module:
    """Resolve a hook site by name: ``"block"`` means the whole decoder layer (the
    residual stream — for activation-space codecs), anything else is a named linear
    submodule (weight-space codecs).

    The candidate list covers the causal-LM decoder paths used by the two active language
    settings. Names remain dot-free so they are valid ``nn.ModuleDict`` keys."""
    if name == "block":
        return layer
    candidates = (name, f"self_attn.{name}", f"mlp.{name}")
    for candidate in candidates:
        try:
            return attrgetter(candidate)(layer)
        except AttributeError:
            pass
    raise AttributeError(f"could not resolve target module {name!r} in {type(layer).__name__}")


class StaticAdapter(nn.Module):
    """A single directly-optimized adapter of the codec's shape, broadcast across the batch and
    independent of any condition -- the "multi-task LoRA" reference for the ``matched - static``
    metric. It holds its own parameters but reuses the (parameter-free) codecs for their output
    size, saddle-escaping init, and ``apply`` hooks, and is trained by the same SFT loss as the
    hypernetwork, so it isolates the *generic* help a shape gives with no conditioning. It exposes
    the same ``forward``/``apply`` surface the downstream evaluator expects, so a trained static
    adapter drops straight into the existing eval path (its "matched" score, computed for any
    condition, is the static reference)."""

    def __init__(self, codecs: nn.ModuleDict, num_layers: int):
        super().__init__()
        self.codecs = codecs  # parameter-free; shared with (or rebuilt like) the hypernetwork's
        self.num_layers = num_layers
        self.params = nn.ParameterDict()
        for name, codec in codecs.items():
            p = torch.zeros(num_layers, codec.output_size)
            bias = codec.initial_bias()
            if bias is not None:
                p = p + bias.unsqueeze(0)  # broadcast the codec's saddle-escape init across layers
            else:
                nn.init.normal_(p, std=0.01)
            self.params[name] = nn.Parameter(p)

    def forward(self, condition_or_batch) -> dict[str, Tensor]:
        # same adapter for every example: accepts a batch size (training) or a (batch, dim)
        # condition tensor whose shape[0] is used (eval), and ignores the condition's content.
        batch = condition_or_batch if isinstance(condition_or_batch, int) else condition_or_batch.shape[0]
        return {name: p.unsqueeze(1).expand(-1, batch, -1) for name, p in self.params.items()}

    def apply(self, layers: list[nn.Module] | nn.ModuleList, generated: Mapping[str, Tensor]):
        return _apply_codec_hooks(layers, self.codecs, generated)


def infer_module_shapes(
    layers: list[nn.Module] | nn.ModuleList, target_modules: list[str], *, hidden_size: int | None = None
) -> dict[str, tuple[int, int]]:
    """Infer (in_features, out_features) per target. Non-``nn.Linear`` targets (e.g. the
    whole decoder layer, resolved via ``"block"``) have no such pair to introspect — pass
    ``hidden_size`` (typically ``model.config.hidden_size``) to use for those instead."""
    first = layers[0]
    shapes = {}
    for name in target_modules:
        module = _resolve_target(first, name)
        if isinstance(module, nn.Linear):
            shapes[name] = (module.in_features, module.out_features)
        elif hidden_size is not None:
            shapes[name] = (hidden_size, hidden_size)
        else:
            raise TypeError(f"{name} is {type(module).__name__}; pass hidden_size for non-nn.Linear targets")
    return shapes
