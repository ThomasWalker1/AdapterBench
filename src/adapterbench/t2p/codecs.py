"""Differentiable generated-parameter codecs. A codec defines an adapter's *output
structure* (how many generated numbers, reshaped how) and how that output is applied at
a hook site. The hook site is determined by the `target_modules`/`module_shapes` name
passed to `TextToPeftHypernetwork` — `"block"` resolves to the whole decoder layer (see
`hypernetwork.py::_resolve_target`), anything else resolves to a named `nn.Linear`
submodule.

The whole point of the framework is that a new adapter shape is just a new
`GeneratedUpdateCodec` subclass + one entry in `make_codec` + an adapter manifest;
additional shapes arrive one at a time with their own leaderboard entry. `GeneratedUpdateCodec`
deliberately keeps the full contract
(`dense_delta`, `initial_bias`, the `"block"` residual-stream hook path) so those shapes
plug back in without framework changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch
from torch import Tensor, nn


def _saddle_bias(output_size: int, seed: int, filled: list[tuple[int, int, float]]) -> Tensor:
    """Build a length-``output_size`` bias that is zero everywhere except the given
    ``[start, end)`` slices, each filled with a fan-in-normalized normal draw.

    Shared by the bilinear codecs' ``initial_bias`` (LoRA/LoKr/LoHa): the factor(s) whose
    slice is left zero keep the initial product exactly zero, while the seeded slice(s)
    break the dead all-zero saddle so gradient flows immediately. Draws come from a single
    local ``Generator`` in the order ``filled`` is given, so callers preserve their exact
    initialization by listing slices in their original draw order.
    """
    generator = torch.Generator().manual_seed(seed)
    bias = torch.zeros(output_size)
    for start, end, fan_in in filled:
        bias[start:end] = torch.randn(end - start, generator=generator) / math.sqrt(fan_in)
    return bias


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
        return _saddle_bias(self.output_size, self.seed, [(0, split, self.in_features)])

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


class IA3Codec(GeneratedUpdateCodec):
    """Output-channel (IA)^3 scaling for a hooked linear projection.

    The generated vector ``v`` has one scalar per output channel and acts as
    ``W -> diag(1 + scale * v) W``.  At the live hook this is implemented as
    a multiplicative transform of the projection output, which is exact for the
    bias-free projection hooks used by both benchmark settings and preserves the
    identity mapping at the all-zero generated initialization.
    """

    def __init__(self, in_features: int, out_features: int, scaling: float = 1.0):
        super().__init__(in_features, out_features)
        self.scaling = scaling

    @property
    def output_size(self) -> int:
        return self.out_features

    def apply(self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int) -> Tensor:
        self._check(inputs, generated)
        multiplier = 1 + self.scaling * generated.to(base_output.dtype)
        return base_output * multiplier.unsqueeze(1)

    def dense_delta(self, generated: Tensor, layer_index: int) -> Tensor:
        self._check_output_size(generated)
        # IA3 is multiplicative rather than an additive, generated weight update:
        # ΔW = diag(scale * v) W depends on the frozen target module's W, which is
        # intentionally not part of this codec's generated state.  There is therefore
        # no standalone additive ΔW to materialize through this interface.  Return the
        # correctly shaped zero additive component; `apply()` is the authoritative live
        # implementation of the nonzero transform.
        return generated.new_zeros(generated.shape[0], self.out_features, self.in_features)


class LoKrCodec(GeneratedUpdateCodec):
    """Kronecker-factored additive update for a hooked linear projection.

    The generated update is ``scale * (L \u2297 R)``.  Rather than introduce a
    tunable internal rank, the factor geometry is a deterministic, balanced exact
    factorization of each frozen projection dimension.  For example, a 2048 by
    2048 projection uses ``(32 x 32) \u2297 (64 x 64)`` and emits 5,120 scalars.
    This is part of the codec's fixed shape identity, not a sweep parameter.

    ``L`` and ``R`` are bilinear generated factors.  ``initial_bias`` initializes
    R but leaves L exactly zero: the initial live update is identity while the
    head can immediately receive a gradient through L.
    """

    def __init__(self, in_features: int, out_features: int, scaling: float = 1.0, seed: int = 777):
        super().__init__(in_features, out_features)
        self.out_factor_1, self.out_factor_2 = self._balanced_factors(out_features)
        self.in_factor_1, self.in_factor_2 = self._balanced_factors(in_features)
        self.scaling = scaling
        self.seed = seed

    @staticmethod
    def _balanced_factors(size: int) -> tuple[int, int]:
        """Return the exact divisor pair closest to ``sqrt(size)``.

        This keeps neither Kronecker factor arbitrarily privileged and works for
        all positive projection widths, including prime dimensions (where the
        only exact shape is ``1 x size``).
        """
        for first in range(math.isqrt(size), 0, -1):
            if size % first == 0:
                return first, size // first
        raise AssertionError("positive dimensions always have a factorization")

    @property
    def output_size(self) -> int:
        return self.out_factor_1 * self.in_factor_1 + self.out_factor_2 * self.in_factor_2

    def _split(self, generated: Tensor) -> tuple[Tensor, Tensor]:
        split = self.out_factor_1 * self.in_factor_1
        left = generated[:, :split].reshape(-1, self.out_factor_1, self.in_factor_1)
        right = generated[:, split:].reshape(-1, self.out_factor_2, self.in_factor_2)
        return left, right

    def initial_bias(self) -> Tensor:
        # See GeneratedUpdateCodec.initial_bias: initializing both Kronecker
        # factors at zero is a dead bilinear saddle.  Keep L zero (so L \u2297 R is
        # exactly zero) and seed R with a fan-in-normalized, deterministic draw.
        split = self.out_factor_1 * self.in_factor_1
        return _saddle_bias(self.output_size, self.seed, [(split, self.output_size, self.in_factor_2)])

    def apply(self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int) -> Tensor:
        self._check(inputs, generated)
        left, right = self._split(generated)
        # Do not materialize ΔW in the live path: contract the two Kronecker
        # factors directly against x[j1, j2].  This preserves the generated
        # autograd graph while avoiding a per-example dense projection tensor.
        factored_inputs = inputs.to(left.dtype).reshape(
            inputs.shape[0], inputs.shape[1], self.in_factor_1, self.in_factor_2
        )
        # Stage the contractions rather than submitting all three operands in one
        # einsum.  The latter can choose a full five-index contraction; these two
        # exact contractions exploit the Kronecker separability and keep the live
        # path practical at the benchmark's sequence and projection dimensions.
        intermediate = torch.einsum("bsij,bai->bsaj", factored_inputs, left)
        delta = torch.einsum("bsaj,bcj->bsac", intermediate, right)
        return base_output + self.scaling * delta.reshape(
            inputs.shape[0], inputs.shape[1], self.out_features
        ).to(base_output.dtype)

    def dense_delta(self, generated: Tensor, layer_index: int) -> Tensor:
        self._check_output_size(generated)
        left, right = self._split(generated)
        delta = torch.einsum("bai,bcj->bacij", left, right).reshape(
            generated.shape[0], self.out_features, self.in_features
        )
        return self.scaling * delta


class LoHaCodec(GeneratedUpdateCodec):
    """Hadamard product of two low-rank additive updates.

    ``ΔW = scale * (B₁ @ A₁) ⊙ (B₂ @ A₂)``.  The public shape identity
    retains the repository's rank-8 LoRA scalar budget: LoHa uses fixed internal
    rank 4, so its four factors emit ``2 * 4 * (in + out) == 8 * (in + out)``
    values.  The internal rank is deliberately derived from the locked base rank,
    never exposed as a sweep parameter.

    A dense per-example ΔW would be prohibitive at a live hook.  ``apply`` uses
    the equivalent rank-squared contraction, preserving the generated tensors'
    autograd graph through the frozen interpreter forward.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, scaling: float = 1.0, seed: int = 777):
        super().__init__(in_features, out_features)
        if rank <= 0:
            raise ValueError("LoHa rank must be positive")
        self.rank = rank
        self.scaling = scaling
        self.seed = seed

    @property
    def output_size(self) -> int:
        return 2 * self.rank * (self.in_features + self.out_features)

    def _split(self, generated: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self._check_output_size(generated)
        a_size = self.rank * self.in_features
        b_size = self.rank * self.out_features
        a1 = generated[:, :a_size].reshape(-1, self.rank, self.in_features)
        a2 = generated[:, a_size : 2 * a_size].reshape(-1, self.rank, self.in_features)
        b1 = generated[:, 2 * a_size : 2 * a_size + b_size].reshape(-1, self.out_features, self.rank)
        b2 = generated[:, 2 * a_size + b_size :].reshape(-1, self.out_features, self.rank)
        return a1, a2, b1, b2

    def initial_bias(self) -> Tensor:
        # LoHa is fourth-order in its factors.  Initializing every factor at zero
        # is a dead saddle; merely initializing one factor is insufficient because
        # every gradient still contains another zero factor.  Keep B1 zero so the
        # whole update is exactly zero, while initialize A1 and the complete second
        # branch B2@A2 nonzero.  The immediate B1 gradient is then nonzero.
        a_size = self.rank * self.in_features
        b_size = self.rank * self.out_features
        # Seed A1, A2 and B2 (in that draw order); B1's slice [2*a_size : 2*a_size + b_size]
        # is left zero, so the whole Hadamard update starts at exactly zero.
        return _saddle_bias(
            self.output_size,
            self.seed,
            [
                (0, a_size, self.in_features),
                (a_size, 2 * a_size, self.in_features),
                (2 * a_size + b_size, self.output_size, self.rank),
            ],
        )

    def apply(self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int) -> Tensor:
        self._check(inputs, generated)
        a1, a2, b1, b2 = self._split(generated)
        # (B1@A1) ⊙ (B2@A2) has rank at most r². Form only its small input/output
        # rank-pair factors, never a (batch, out, in) generated tensor in the live
        # path. Both staged contractions keep `generated` live for backward.
        input_pairs = a1.unsqueeze(2) * a2.unsqueeze(1)      # (batch, r, r, in)
        output_pairs = b1.unsqueeze(3) * b2.unsqueeze(2)     # (batch, out, r, r)
        features = torch.einsum("bsi,brqi->bsrq", inputs.to(a1.dtype), input_pairs)
        delta = torch.einsum("bsrq,borq->bso", features, output_pairs)
        return base_output + self.scaling * delta.to(base_output.dtype)

    def dense_delta(self, generated: Tensor, layer_index: int) -> Tensor:
        a1, a2, b1, b2 = self._split(generated)
        first = torch.einsum("bor,bri->boi", b1, a1)
        second = torch.einsum("bor,bri->boi", b2, a2)
        return self.scaling * first * second


def make_codec(
    name: str,
    in_features: int,
    out_features: int,
    *,
    num_layers: int,
    rank: int = 8,
    alpha: float = 16.0,
    lora_scaling: float | None = None,
    ia3_scaling: float = 1.0,
    lokr_scaling: float = 1.0,
    loha_scaling: float = 1.0,
    seed: int = 777,
    # Accepted-and-ignored so call sites (and future codecs) can pass a
    # uniform kwarg set without every caller special-casing which codec is registered.
    **_unused_codec_kwargs,
) -> GeneratedUpdateCodec:
    """Build a registered generated-update codec."""

    def _loha() -> LoHaCodec:
        if rank < 2 or rank % 2:
            raise ValueError("LoHa requires an even base rank >= 2 so its fixed internal rank is rank/2")
        return LoHaCodec(in_features, out_features, rank=rank // 2, scaling=loha_scaling, seed=seed)

    constructors = {
        "lora": lambda: LoRACodec(in_features, out_features, rank, alpha, scaling=lora_scaling, seed=seed),
        "ia3": lambda: IA3Codec(in_features, out_features, scaling=ia3_scaling),
        "lokr": lambda: LoKrCodec(in_features, out_features, scaling=lokr_scaling, seed=seed),
        # Four LoHa factor groups at rank r/2 have exactly the scalar count of the
        # repository's fixed rank-r LoRA budget. r=8 is locked by the substrate.
        "loha": _loha,
    }
    try:
        return constructors[name]()
    except KeyError as error:
        raise ValueError(
            f"unsupported differentiable adapter: {name!r} "
            f"(registered codecs: {sorted(constructors)})"
        ) from error
