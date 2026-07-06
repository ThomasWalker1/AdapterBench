"""One shared Text-to-PEFT hypernetwork shell with swappable output codecs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from operator import attrgetter

import torch
from torch import Tensor, nn

from .codecs import GeneratedUpdateCodec, make_codec


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
        condition_dim: int,
        module_shapes: Mapping[str, tuple[int, int]],
        num_layers: int,
        adapter: str,
        latent_dim: int = 512,
        head_dim: int = 2048,
        rank: int = 8,
        alpha: float = 16.0,
        n_frequency: int = 1000,
        steering_scale: float = 1.0,
        seed: int = 777,
    ):
        super().__init__()
        self.module_names = tuple(module_shapes)
        self.num_layers = num_layers
        self.adapter = adapter
        task_dim, depth_dim, type_dim = latent_dim // 2, latent_dim // 4, latent_dim // 4
        self.task_encoder = nn.Sequential(nn.Linear(condition_dim, task_dim), nn.LayerNorm(task_dim))
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

    def forward(self, condition_embeddings: Tensor) -> dict[str, Tensor]:
        batch = condition_embeddings.shape[0]
        tasks = self.task_encoder(condition_embeddings)
        depths = self.depth_embedding(torch.arange(self.num_layers, device=tasks.device))
        outputs = {}
        for type_index, name in enumerate(self.module_names):
            types = self.type_embedding(torch.tensor(type_index, device=tasks.device)).expand(self.num_layers, -1)
            features = torch.cat(
                (
                    tasks.unsqueeze(0).expand(self.num_layers, batch, -1),
                    depths.unsqueeze(1).expand(-1, batch, -1),
                    types.unsqueeze(1).expand(-1, batch, -1),
                ),
                dim=-1,
            )
            outputs[name] = self.heads[name](self.trunk(features))
        return outputs

    def forward_layer(self, condition_embeddings: Tensor, layer_index: int) -> dict[str, Tensor]:
        """Generate one layer's parameters only (numerically equal to ``forward(...)[name][layer_index]``).

        Dense-ΔW codecs (e.g. FourierFT) materialize an ``(out_features, in_features)``
        tensor per task per layer, which is large at real model dimensions. Backpropagating
        through ``forward``'s single batched call would keep every layer's dense-ΔW graph
        alive simultaneously for one shared backward pass. Calling this per layer instead —
        each with its own immediate ``backward()`` — lets autograd free one layer's graph
        before the next layer is even computed, at the cost of recomputing the (cheap)
        trunk forward once per layer instead of once total.
        """
        batch = condition_embeddings.shape[0]
        tasks = self.task_encoder(condition_embeddings)
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

    @contextmanager
    def apply(self, layers: list[nn.Module] | nn.ModuleList, generated: Mapping[str, Tensor]):
        handles = []
        for layer_index, layer in enumerate(layers):
            for name, codec in self.codecs.items():
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
    submodule (weight-space codecs)."""
    if name == "block":
        return layer
    candidates = (name, f"self_attn.{name}", f"mlp.{name}")
    for candidate in candidates:
        try:
            return attrgetter(candidate)(layer)
        except AttributeError:
            pass
    raise AttributeError(f"could not resolve target module {name!r} in {type(layer).__name__}")


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

