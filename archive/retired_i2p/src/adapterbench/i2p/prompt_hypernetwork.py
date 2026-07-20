"""Prompt-conditioned weight-space adapters for the I2P boundary probe.

Unlike :mod:`adapterbench.i2p.hypernoise`, this module applies generated LoRA
updates directly to the denoising UNet.  The prompt is encoded once by
SD-Turbo's frozen text encoder, pooled, and passed through the same trunk used
by T2L/D2L.

SD-Turbo's 128 attention projections are heterogeneous, so the language
hypernetwork's rectangular ``layer x module-type`` layout is not a natural
fit.  This adapter instead batches targets through one shared trunk and shares
an output head among targets with the same ``(in_features, out_features)``
shape.  Learned target/role embeddings still allow a distinct adapter at every
projection.  This avoids a separate large output head per Linear while
preserving the exact codec seam and per-instance generated output.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Mapping

import torch
from torch import Tensor, nn

from ..t2p.codecs import GeneratedUpdateCodec, make_codec
from ..t2p.hypernetwork import PooledVectorConditioner, ResidualMLP


def _make_hypernetwork_trunk(latent_dim: int, head_dim: int) -> nn.Sequential:
    """Archived copy of the trunk constructor used by this experiment.

    The active T2P module no longer carries the image-probe-only constructor, so
    keeping it here makes the retired snapshot internally complete.
    """
    return nn.Sequential(
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


def _key(name: str) -> str:
    return name.replace(".", "__")


def _attention_role(name: str) -> str:
    """Return a stable role such as ``attn2:to_k`` from a diffusers path."""
    pieces = name.split(".")
    attention = next((piece for piece in pieces if piece in {"attn1", "attn2"}), "attention")
    projection = pieces[-2] if pieces[-1].isdigit() else pieces[-1]
    return f"{attention}:{projection}"


def _shape_key(module: nn.Linear) -> str:
    return f"i{module.in_features}_o{module.out_features}"


class PromptConditionedUNetAdapter(nn.Module):
    """Generate one per-example codec vector for every UNet attention Linear."""

    def __init__(
        self,
        targets: list[tuple[str, nn.Linear]],
        *,
        condition_dim: int = 1024,
        codec_name: str = "lora",
        latent_dim: int = 256,
        head_dim: int = 256,
        rank: int = 4,
        alpha: float = 8.0,
        scaling: float | None = 1.0,
        seed: int = 777,
    ):
        super().__init__()
        if not targets:
            raise ValueError("at least one target Linear is required")
        if latent_dim % 4:
            raise ValueError("latent_dim must be divisible by 4")

        self.target_names = tuple(name for name, _ in targets)
        self._modules_by_name = {name: module for name, module in targets}
        self.codec_name = codec_name

        prompt_dim, target_dim, role_dim = latent_dim // 2, latent_dim // 4, latent_dim // 4
        roles = sorted({_attention_role(name) for name, _ in targets})
        role_to_index = {role: index for index, role in enumerate(roles)}
        self.conditioner = PooledVectorConditioner(condition_dim, prompt_dim)
        self.target_embedding = nn.Sequential(nn.Embedding(len(targets), target_dim), nn.LayerNorm(target_dim))
        self.role_embedding = nn.Sequential(nn.Embedding(len(roles), role_dim), nn.LayerNorm(role_dim))
        self.trunk = _make_hypernetwork_trunk(latent_dim, head_dim)
        self.register_buffer(
            "role_indices",
            torch.tensor([role_to_index[_attention_role(name)] for name, _ in targets], dtype=torch.long),
            persistent=True,
        )

        codecs: dict[str, GeneratedUpdateCodec] = {}
        heads: dict[str, nn.Linear] = {}
        groups: dict[str, list[int]] = defaultdict(list)
        init_biases: dict[str, list[Tensor]] = defaultdict(list)
        for index, (name, module) in enumerate(targets):
            key = _key(name)
            shape = _shape_key(module)
            codec = make_codec(
                codec_name,
                module.in_features,
                module.out_features,
                num_layers=1,
                rank=rank,
                alpha=alpha,
                lora_scaling=scaling,
                seed=seed + index,
            )
            codecs[key] = codec
            groups[shape].append(index)
            initial = codec.initial_bias()
            init_biases[shape].append(
                initial.clone() if initial is not None else torch.zeros(codec.output_size)
            )
            if shape not in heads:
                head = nn.Linear(head_dim, codec.output_size)
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                heads[shape] = head

        self.codecs = nn.ModuleDict(codecs)
        self.heads = nn.ModuleDict(heads)
        self._shape_groups = {shape: tuple(indices) for shape, indices in groups.items()}
        for shape, biases in init_biases.items():
            self.register_buffer(f"_initial__{shape}", torch.stack(biases), persistent=True)

    def generated_parameter_count(self) -> int:
        """Numbers emitted per prompt (the quantity compared across codec shapes)."""
        return sum(codec.output_size for codec in self.codecs.values())

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, condition_embeddings: Tensor) -> dict[str, Tensor]:
        """Return ``target_name -> (batch, codec.output_size)`` generated vectors."""
        prompts = self.conditioner(condition_embeddings)
        batch = prompts.shape[0]
        target_indices = torch.arange(len(self.target_names), device=prompts.device)
        targets = self.target_embedding(target_indices)
        roles = self.role_embedding(self.role_indices)
        features = torch.cat(
            (
                prompts.unsqueeze(0).expand(len(self.target_names), -1, -1),
                targets.unsqueeze(1).expand(-1, batch, -1),
                roles.unsqueeze(1).expand(-1, batch, -1),
            ),
            dim=-1,
        )
        hidden = self.trunk(features)

        generated: dict[str, Tensor] = {}
        for shape, indices_tuple in self._shape_groups.items():
            indices = torch.tensor(indices_tuple, device=hidden.device)
            group = self.heads[shape](hidden.index_select(0, indices))
            initial = getattr(self, f"_initial__{shape}").unsqueeze(1)
            group = group + initial
            for local_index, target_index in enumerate(indices_tuple):
                generated[self.target_names[target_index]] = group[local_index]
        return generated

    @contextmanager
    def apply(self, generated: Mapping[str, Tensor]):
        """Apply one generated adapter per batch element directly in weight space."""
        handles = []
        for name in self.target_names:
            codec = self.codecs[_key(name)]
            parameters = generated[name]
            module = self._modules_by_name[name]

            def hook(module, args, output, *, codec=codec, parameters=parameters):
                return codec.apply(args[0], output, parameters, 0)

            handles.append(module.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
