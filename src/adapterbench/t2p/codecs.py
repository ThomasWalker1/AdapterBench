"""Differentiable generated-parameter codecs, hookable at either a named linear
submodule (weight-space adapters: LoRA, FreezeALoRA, LoKr, FourierFT) or a whole
decoder layer's output / residual stream (activation-space adapters: IA3,
activation steering). The hook site is determined entirely by the `target_modules`/
`module_shapes` name passed to `TextToPeftHypernetwork` — `"block"` resolves to the
layer itself (see `hypernetwork.py::_resolve_target`), anything else resolves to a named
`nn.Linear` submodule. A codec that ignores `inputs` (IA3, activation steering) works at
either site unchanged; one that uses `inputs` as a linear map's pre-activation input
(LoRA and friends) is only meaningful at a linear-submodule site.
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
        self, in_features: int, out_features: int, rank: int, alpha: float, use_rslora: bool = True, seed: int = 777
    ):
        super().__init__(in_features, out_features)
        self.rank = rank
        self.scaling = alpha / math.sqrt(rank) if use_rslora else alpha / rank
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


class FreezeALoRACodec(GeneratedUpdateCodec):
    def __init__(
        self, in_features: int, out_features: int, rank: int, alpha: float, num_layers: int, seed: int
    ):
        super().__init__(in_features, out_features)
        self.rank = rank
        self.scaling = alpha / math.sqrt(rank)
        generator = torch.Generator().manual_seed(seed)
        fixed_a = torch.randn(num_layers, rank, in_features, generator=generator) / math.sqrt(in_features)
        self.register_buffer("fixed_a", fixed_a)

    @property
    def output_size(self) -> int:
        return self.out_features * self.rank

    def apply(self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int) -> Tensor:
        self._check(inputs, generated)
        a = self.fixed_a[layer_index].to(inputs.dtype)
        b = generated.reshape(-1, self.out_features, self.rank)
        low_rank = torch.einsum("bsi,ri->bsr", inputs, a)
        delta = torch.einsum("bsr,bor->bso", low_rank, b)
        return base_output + self.scaling * delta.to(base_output.dtype)

    def dense_delta(self, generated: Tensor, layer_index: int) -> Tensor:
        self._check_output_size(generated)
        a = self.fixed_a[layer_index]
        b = generated.reshape(-1, self.out_features, self.rank)
        return self.scaling * torch.einsum("bor,ri->boi", b, a.to(b.dtype))


class IA3Codec(GeneratedUpdateCodec):
    @property
    def output_size(self) -> int:
        return self.out_features

    def apply(self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int) -> Tensor:
        self._check(inputs, generated)
        scale = 1.0 + generated.unsqueeze(1).to(base_output.dtype)
        return base_output * scale

    def dense_delta(self, generated: Tensor, layer_index: int) -> Tensor:
        raise NotImplementedError(
            "IA3 is multiplicative on activations (base_output * (1 + generated)), "
            "not additive in weight space — it has no ΔW to regress onto."
        )


class ActivationSteeringCodec(GeneratedUpdateCodec):
    """Additive activation steering: ``h <- h + scale * generated``.

    Ignores ``inputs`` entirely (like IA3Codec) but is additive rather than
    multiplicative, and is meant to be hooked at a whole decoder layer's output (the
    residual stream, ``target_modules=["block"]``) rather than a specific linear
    submodule — requires a square site (``in_features == out_features``, e.g.
    ``hidden_size``) since it adds directly to the layer's output activations.
    """

    def __init__(self, in_features: int, out_features: int, scale: float = 1.0):
        super().__init__(in_features, out_features)
        if in_features != out_features:
            raise ValueError("activation steering requires a square residual-stream site")
        self.scale = scale

    @property
    def output_size(self) -> int:
        return self.out_features

    def apply(self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int) -> Tensor:
        self._check_output_size(generated)
        return base_output + self.scale * generated.unsqueeze(1).to(base_output.dtype)

    def dense_delta(self, generated: Tensor, layer_index: int) -> Tensor:
        raise NotImplementedError(
            "activation steering is additive on activations (base_output + scale * generated), "
            "not additive in weight space — it has no ΔW to regress onto."
        )


def _factor_pair(value: int) -> tuple[int, int]:
    first = int(math.sqrt(value))
    while first > 1 and value % first:
        first -= 1
    return first, value // first


class LoKrCodec(GeneratedUpdateCodec):
    def __init__(self, in_features: int, out_features: int, alpha: float = 1.0, seed: int = 777):
        super().__init__(in_features, out_features)
        self.a_out, self.b_out = _factor_pair(out_features)
        self.a_in, self.b_in = _factor_pair(in_features)
        self.scaling = alpha
        self.seed = seed

    @property
    def output_size(self) -> int:
        return self.a_out * self.a_in + self.b_out * self.b_in

    def initial_bias(self) -> Tensor:
        # Same bilinear zero-gradient saddle as LoRA (torch.kron(a, b) is bilinear too)
        # — randomize the `a` slice, leave `b` at zero so the adapter still contributes
        # exactly zero at init but gradient reaches `b` immediately.
        split = self.a_out * self.a_in
        generator = torch.Generator().manual_seed(self.seed)
        bias = torch.zeros(self.output_size)
        bias[:split] = torch.randn(split, generator=generator) / math.sqrt(max(self.a_in, 1))
        return bias

    def dense_delta(self, generated: Tensor, layer_index: int) -> Tensor:
        self._check_output_size(generated)
        split = self.a_out * self.a_in
        a = generated[:, :split].reshape(-1, self.a_out, self.a_in)
        b = generated[:, split:].reshape(-1, self.b_out, self.b_in)
        delta_weight = torch.stack([torch.kron(ai, bi) for ai, bi in zip(a, b)])
        return self.scaling * delta_weight

    def apply(self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int) -> Tensor:
        self._check(inputs, generated)
        delta_weight = self.dense_delta(generated, layer_index)
        delta = torch.einsum("bsi,boi->bso", inputs.to(delta_weight.dtype), delta_weight)
        return base_output + delta.to(base_output.dtype)


class FourierFTCodec(GeneratedUpdateCodec):
    def __init__(self, in_features: int, out_features: int, n_frequency: int, scaling: float, seed: int):
        super().__init__(in_features, out_features)
        if n_frequency > in_features * out_features:
            raise ValueError("n_frequency exceeds the matrix size")
        generator = torch.Generator().manual_seed(seed)
        locations = torch.randperm(in_features * out_features, generator=generator)[:n_frequency]
        self.register_buffer("locations", locations)
        self.n_frequency = n_frequency
        self.scaling = scaling

    @property
    def output_size(self) -> int:
        return self.n_frequency

    def dense_delta(self, generated: Tensor, layer_index: int) -> Tensor:
        self._check_output_size(generated)
        batch = generated.shape[0]
        spectrum = torch.zeros(
            batch,
            self.out_features * self.in_features,
            dtype=torch.complex64,
            device=generated.device,
        )
        spectrum[:, self.locations] = generated.float().to(torch.complex64)
        # norm="forward" (unnormalized ifft, all scaling in `self.scaling`): the default
        # norm="backward" divides by out_features*in_features, which is negligible at the
        # tiny dimensions unit tests use but is a ~1/16.7M attenuation at real Mistral-7B
        # dimensions (4096x4096) — enough to make `scaling` unable to compensate and the
        # generated head's gradient underflow to a practically untrainable magnitude.
        # `scaling` should be a dimension-independent knob, matching every other codec's
        # `alpha`-style scaling; norm="backward"'s hidden 1/N dependence broke that.
        delta_weight = torch.fft.ifft2(
            spectrum.reshape(batch, self.out_features, self.in_features), norm="forward"
        ).real
        return self.scaling * delta_weight

    def apply(self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int) -> Tensor:
        self._check(inputs, generated)
        delta_weight = self.dense_delta(generated, layer_index)
        delta = torch.einsum("bsi,boi->bso", inputs.float(), delta_weight)
        return base_output + delta.to(base_output.dtype)


def make_codec(
    name: str,
    in_features: int,
    out_features: int,
    *,
    num_layers: int,
    rank: int = 8,
    alpha: float = 16.0,
    n_frequency: int = 1000,
    fourier_scaling: float = 300.0,
    steering_scale: float = 1.0,
    seed: int = 777,
) -> GeneratedUpdateCodec:
    constructors = {
        "lora": lambda: LoRACodec(in_features, out_features, rank, alpha, seed=seed),
        "freeze_a_lora": lambda: FreezeALoRACodec(
            in_features, out_features, rank, alpha, num_layers, seed
        ),
        "ia3": lambda: IA3Codec(in_features, out_features),
        "lokr": lambda: LoKrCodec(in_features, out_features, alpha=alpha / max(rank, 1), seed=seed),
        "fourierft": lambda: FourierFTCodec(
            in_features, out_features, n_frequency, fourier_scaling, seed
        ),
        "activation_steering": lambda: ActivationSteeringCodec(in_features, out_features, steering_scale),
    }
    try:
        return constructors[name]()
    except KeyError as error:
        raise ValueError(f"unsupported differentiable adapter: {name}") from error

