"""Differentiable generated-parameter codecs. A codec defines an adapter's *output
structure* (how many generated numbers, reshaped how) and how that output is applied at
a hook site. The hook site is determined by the `target_modules`/`module_shapes` name
passed to `TextToPeftHypernetwork` — `"block"` resolves to the whole decoder layer (see
`hypernetwork.py::_resolve_target`), anything else resolves to a named `nn.Linear`
submodule.

**LoRA is the only codec in the current baseline** (see PROJECT_PLAN.md's session
handoff and roadmap). The whole point of the framework is that a new adapter shape is just a new
`GeneratedUpdateCodec` subclass + one entry in `make_codec` + an adapter manifest;
additional shapes (FreezeALoRA, LoKr, FourierFT, IA3, activation steering, …) are
reintroduced one at a time as committed codecs, each with its own leaderboard entry, not carried here
speculatively. `GeneratedUpdateCodec` deliberately keeps the full contract
(`dense_delta`, `initial_bias`, the `"block"` residual-stream hook path) so those shapes
plug back in without framework changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch
from torch import Tensor, nn


class GeneratedUpdateCodec(nn.Module, ABC):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    @property
    @abstractmethod
    def output_size(self) -> int: ...

    @abstractmethod
    def apply(self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int) -> Tensor: ...

    @abstractmethod
    def dense_delta(self, generated: Tensor, layer_index: int) -> Tensor:
        """Return the (batch, out_features, in_features) weight-space update ΔW."""

    def initial_bias(self) -> Tensor | None:
        """Optional fixed nonzero bias for the generated head's output.

        Default ``None`` means "an all-zero initial output is fine" — true for any
        codec whose ``apply()`` is linear in ``generated`` (IA3, activation steering,
        FourierFT): zero output still means zero contribution, and the gradient w.r.t.
        ``generated`` is nonzero at that point since it doesn't depend on `generated`
        itself. Codecs that split ``generated`` into two factors multiplied together
        (LoRA's ``B @ A``, LoKr's Kronecker factors) are bilinear — at an all-zero
        starting point, the gradient w.r.t. *both* factors vanishes simultaneously (each
        factor's gradient is proportional to the other, which is also zero), a dead
        saddle point standard LoRA practice avoids by initializing one factor non-zero.
        Those codecs override this to return a bias pattern with one factor's slice
        randomized and the other's left at zero — the adapter still contributes exactly
        zero at init (since the zero factor still zeroes the product), but gradient
        flows to the zero factor immediately (proportional to the non-zero one).
        """
        return None

    def _check_output_size(self, generated: Tensor) -> None:
        if generated.shape[-1] != self.output_size:
            raise ValueError(f"expected {self.output_size} generated values, got {generated.shape[-1]}")

    def _check(self, inputs: Tensor, generated: Tensor) -> None:
        self._check_output_size(generated)
        if inputs.shape[0] != generated.shape[0]:
            raise ValueError("one generated adapter is required per batch element")


class LoRACodec(GeneratedUpdateCodec):
    def __init__(
        self, in_features: int, out_features: int, rank: int, alpha: float, use_rslora: bool = True,
        scaling: float | None = None, seed: int = 777
    ):
        super().__init__(in_features, out_features)
        self.rank = rank
        # `scaling`, if given, overrides the alpha-derived value - Doc-to-LoRA's NIAH recipe
        # applies `lora_alpha = 2*r^1.5` DIRECTLY as the scale (~45.25 at r=8; see upstream
        # lora_layer.lora_forward), ~8x larger than rslora's alpha/sqrt(r). That larger update
        # is load-bearing for the generated adapter to override the frozen model on NIAH.
        self.scaling = scaling if scaling is not None else (alpha / math.sqrt(rank) if use_rslora else alpha / rank)
        self.seed = seed

    @property
    def output_size(self) -> int:
        return self.rank * (self.in_features + self.out_features)

    def _split(self, generated: Tensor) -> tuple[Tensor, Tensor]:
        split = self.rank * self.in_features
        a = generated[:, :split].reshape(-1, self.rank, self.in_features)
        b = generated[:, split:].reshape(-1, self.out_features, self.rank)
        return a, b

    def initial_bias(self) -> Tensor:
        # A's slice gets a small random nonzero init (breaks the bilinear zero-gradient
        # saddle point — see GeneratedUpdateCodec.initial_bias); B's slice stays zero so
        # the adapter still contributes exactly zero at initialization (ΔW = B @ A = 0
        # whenever B = 0, regardless of A).
        split = self.rank * self.in_features
        generator = torch.Generator().manual_seed(self.seed)
        bias = torch.zeros(self.output_size)
        bias[:split] = torch.randn(split, generator=generator) / math.sqrt(self.in_features)
        return bias

    def apply(self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int) -> Tensor:
        self._check(inputs, generated)
        a, b = self._split(generated)
        low_rank = torch.einsum("bsi,bri->bsr", inputs.to(a.dtype), a)
        delta = torch.einsum("bsr,bor->bso", low_rank, b)
        return base_output + self.scaling * delta.to(base_output.dtype)

    def dense_delta(self, generated: Tensor, layer_index: int) -> Tensor:
        self._check_output_size(generated)
        a, b = self._split(generated)
        return self.scaling * torch.einsum("bor,bri->boi", b, a)


def make_codec(
    name: str,
    in_features: int,
    out_features: int,
    *,
    num_layers: int,
    rank: int = 8,
    alpha: float = 16.0,
    lora_scaling: float | None = None,
    seed: int = 777,
    # Accepted-and-ignored so call sites (and future codecs) can pass a
    # uniform kwarg set without every caller special-casing which codec is registered.
    **_unused_codec_kwargs,
) -> GeneratedUpdateCodec:
    """Build the codec named `name`. LoRA is the only registered shape in the current
    baseline; a new adapter shape adds one entry to `constructors` (plus its
    `GeneratedUpdateCodec` subclass above and its manifest), added one at a time with a leaderboard entry.
    """
    constructors = {
        "lora": lambda: LoRACodec(in_features, out_features, rank, alpha, scaling=lora_scaling, seed=seed),
    }
    try:
        return constructors[name]()
    except KeyError as error:
        raise ValueError(
            f"unsupported differentiable adapter: {name!r} "
            f"(baseline registers only {sorted(constructors)}; new shapes arrive via the pipeline)"
        ) from error
