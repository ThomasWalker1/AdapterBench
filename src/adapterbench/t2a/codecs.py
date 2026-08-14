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
from torch.utils.checkpoint import checkpoint


def _saddle_bias(output_size: int, seed: int, filled: list[tuple[int, int, float]]) -> Tensor:
    """Build a length-``output_size`` bias that is zero everywhere except the given
    ``[start, end)`` slices, each filled with a fan-in-normalized normal draw.

    Shared by the bilinear codecs' ``initial_bias`` (LoRA/LoKr): the factor(s) whose
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

    def apply_at(
        self, module: nn.Module, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int
    ) -> Tensor:
        """Hook-site-aware entry point, called by the forward hooks in
        ``hypernetwork.py::_apply_codec_hooks`` (and the persistent-hook variant in
        ``scripts/t2a_train_ddp.py``) with the resolved hook-site module itself.

        The default implementation ignores ``module`` and delegates to ``apply``, so every
        codec whose update is a function of the generated values alone — LoRA, (IA)³, LoKr,
        FourierFT, steering — is unchanged, bit for bit. Override it only for a codec whose
        definition references the *frozen* weight at its hook site: ``DoRACodec`` decomposes
        ``W0`` into magnitude and direction, so it needs read access to ``W0`` rather than
        just ``W0 @ x`` (``base_output``), and no amount of generated state can recover it.

        The frozen weight is read, never written, and is not part of the generated state or
        the optimizer's parameters — the interpreter stays frozen, and ``output_size`` is
        unaffected.
        """
        return self.apply(inputs, base_output, generated, layer_index)

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


class DoRACodec(GeneratedUpdateCodec):
    """Weight-decomposed low-rank adaptation (Liu et al., 2024).

    DoRA splits the frozen projection into a per-output-channel **magnitude** and a
    **direction**, adapts the direction with a LoRA-style low-rank update, and renormalizes:

        W' = m ⊙_row (W0 + s·B@A) / ||W0 + s·B@A||_row

    where ``||·||_row`` is the vector norm over each output row's input fan-in, ``m`` is one
    magnitude scalar per output channel, and ``s`` is this codec's uniform ``.scaling``.
    Both parts are generated: the emitted vector is LoRA's ``A``/``B`` slices followed by
    ``out_features`` magnitude scalars, which are a **delta on** the frozen row norms
    (``m = ||W0||_row + s·g_m``), so the all-zero generated output is exactly the frozen
    projection.

    Why it is interesting for this benchmark: DoRA is the one registered shape whose update
    is defined *relative to the frozen weight it edits*. Every other codec here emits a
    self-contained function of its generated values, so the hypernetwork must discover the
    scale of a useful edit from data; DoRA hands it a decomposition in which magnitude and
    direction are separately addressable and already normalized by the host weight. If the
    obstacle to one-shot adapter prediction is partly *calibration* rather than expressivity,
    this shape should show it — and it is a direct test of whether LoRA's reported deficit
    versus full fine-tuning, which DoRA was introduced to close, has any counterpart when the
    adapter is predicted rather than optimized.

    Implementation notes:

    - **The denominator is detached**, exactly as in the reference implementation (and PEFT's
      ``DoraLinearLayer``): the paper treats ``||V + ΔV||_c`` as a constant in the backward
      pass, which is part of the method, not an optimization. The numerator ``m`` stays
      differentiable, so gradient reaches the magnitude slice.
    - The row norms are computed **without materializing** ``W0 + s·B@A`` (a per-example
      ``(out, in)`` tensor at every layer would dominate the step): expanding
      ``||W0 + s·BA||²_row = ||W0||²_row + 2s·⟨W0, BA⟩_row + s²·||BA||²_row`` needs only
      ``A@W0ᵀ`` and the ``r × r`` Gram matrix ``A@Aᵀ``. This is exact, not an approximation —
      ``tests/test_dynamic_t2a.py`` pins it against the materialized reference.
    - ``apply`` is bilinear in the ``A``/``B`` slices, so ``initial_bias`` seeds ``A`` and
      leaves ``B`` (and the magnitude slice) at zero, exactly as ``LoRACodec`` does.
    - The hook site must be a linear projection: the magnitude/direction decomposition is
      defined on a weight matrix, so ``"block"`` (the residual stream) has nothing to
      decompose.
    """

    def __init__(
        self, in_features: int, out_features: int, rank: int, *, scaling: float = 1.0, seed: int = 777,
        epsilon: float = 1e-6,
    ):
        super().__init__(in_features, out_features)
        self.rank = rank
        # One knob for the whole intervention (the benchmark's uniform `--codec-scaling`
        # sweep axis): it scales the directional update AND the generated magnitude delta,
        # so `scaling` moves the strength of the edit without changing its shape. At
        # scaling -> 0 the codec is exactly the frozen projection for any generated values.
        self.scaling = scaling
        self.seed = seed
        # Guards the sqrt/divide when a large directional update lands anti-parallel to a
        # frozen row; never reached at any usable scale, but a NaN here would be silent.
        self.epsilon = epsilon

    @property
    def output_size(self) -> int:
        # LoRA's rank-r budget plus one magnitude scalar per output channel. The magnitude
        # vector is DoRA's shape identity, not a tunable extra: it is the +d_out overhead
        # the method is defined by (5.9% over rank-8 LoRA at gemma-2-2b's q_proj).
        return self.rank * (self.in_features + self.out_features) + self.out_features

    def _split(self, generated: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        a_end = self.rank * self.in_features
        b_end = a_end + self.rank * self.out_features
        a = generated[:, :a_end].reshape(-1, self.rank, self.in_features)
        b = generated[:, a_end:b_end].reshape(-1, self.out_features, self.rank)
        magnitude = generated[:, b_end:]
        return a, b, magnitude

    def initial_bias(self) -> Tensor:
        # Same bilinear saddle-escape as LoRACodec (see GeneratedUpdateCodec.initial_bias):
        # seed A, leave B at zero so the directional update is exactly zero at init. The
        # magnitude slice also stays zero, which makes m = ||W0||_row and therefore
        # W' = W0 exactly — DoRA's own identity initialization.
        return _saddle_bias(self.output_size, self.seed, [(0, self.rank * self.in_features, self.in_features)])

    def _row_norms(self, base_weight: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(||W0||_row, ||W0 + scaling*B@A||_row)`` in float32.

        Both are detached: the frozen weight carries no gradient, and DoRA's backward pass
        treats the adapted norm as a constant. Computed via the expansion documented in the
        class docstring, so no per-example dense ``(out, in)`` tensor is ever formed.
        """
        weight = base_weight.detach().float()
        base_squared = weight.pow(2).sum(dim=1)  # (out,)
        a, b = a.detach().float(), b.detach().float()
        # <W0[o], (B@A)[o]> without forming B@A: (A @ W0^T) is (batch, rank, out).
        projected = torch.einsum("bri,oi->bro", a, weight)
        cross = torch.einsum("bor,bro->bo", b, projected)
        # ||(B@A)[o]||^2 = b[o] @ (A A^T) @ b[o]
        gram = torch.einsum("bri,bqi->brq", a, a)
        delta_squared = (torch.einsum("bor,brq->boq", b, gram) * b).sum(dim=-1)
        adapted_squared = base_squared.unsqueeze(0) + 2 * self.scaling * cross + self.scaling**2 * delta_squared
        return base_squared.clamp_min(self.epsilon).sqrt(), adapted_squared.clamp_min(self.epsilon).sqrt()

    def _magnitude_norm_scale(self, base_weight: Tensor, a: Tensor, b: Tensor, magnitude: Tensor) -> Tensor:
        """The per-example, per-output-channel factor ``m / ||W0 + s·B@A||_row``."""
        base_norm, adapted_norm = self._row_norms(base_weight, a, b)
        # Generated magnitudes are a delta on the frozen row norms, so zero => m = ||W0||_row
        # and (with B = 0) the ratio is exactly 1: the frozen projection, bit for bit.
        return (base_norm.unsqueeze(0) + self.scaling * magnitude.float()) / adapted_norm

    def apply(
        self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int,
        *, base_weight: Tensor | None = None,
    ) -> Tensor:
        self._check(inputs, generated)
        if base_weight is None:
            raise ValueError(
                "DoRACodec needs the frozen hook-site weight (it decomposes W0 into magnitude "
                "and direction): call apply_at(module, ...) — which the benchmark's forward "
                "hooks do — or pass base_weight explicitly."
            )
        a, b, magnitude = self._split(generated)
        low_rank = torch.einsum("bsi,bri->bsr", inputs.to(a.dtype), a)
        directional = torch.einsum("bsr,bor->bso", low_rank, b)
        scale = self._magnitude_norm_scale(base_weight, a, b, magnitude).unsqueeze(1).to(base_output.dtype)
        return scale * (base_output + self.scaling * directional.to(base_output.dtype))

    def apply_at(
        self, module: nn.Module, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int
    ) -> Tensor:
        weight = getattr(module, "weight", None)
        if weight is None or weight.shape != (self.out_features, self.in_features):
            found = "none" if weight is None else f"shape {tuple(weight.shape)}"
            raise ValueError(
                f"dora hooks a linear projection whose frozen weight it decomposes, and needs "
                f"a ({self.out_features}, {self.in_features}) weight there; {type(module).__name__} "
                f"has {found}. The 'block' residual-stream site cannot carry a weight-decomposed "
                "codec."
            )
        return self.apply(inputs, base_output, generated, layer_index, base_weight=weight)

    def dense_delta(
        self, generated: Tensor, layer_index: int, *, base_weight: Tensor | None = None
    ) -> Tensor:
        """``W' - W0`` for the hooked projection.

        Unlike the additive codecs, DoRA's update is defined relative to the weight it edits,
        so the frozen weight is required. This materializes the dense ``(batch, out, in)``
        update and is a diagnostic/testing path — ``apply`` is the live implementation and
        never forms it.
        """
        self._check_output_size(generated)
        if base_weight is None:
            raise ValueError(
                "DoRACodec.dense_delta needs the frozen hook-site weight: DoRA's update is "
                "W' - W0 = m ⊙ (W0 + s·B@A)/||W0 + s·B@A|| - W0, which is not a function of "
                "the generated values alone."
            )
        a, b, magnitude = self._split(generated)
        weight = base_weight.detach().float()
        adapted = weight.unsqueeze(0) + self.scaling * torch.einsum("bor,bri->boi", b, a).float()
        scale = self._magnitude_norm_scale(base_weight, a, b, magnitude)
        return (scale.unsqueeze(-1) * adapted - weight.unsqueeze(0)).to(generated.dtype)


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


class SteeringCodec(GeneratedUpdateCodec):
    """Additive residual-stream steering: ``h -> h + scale * v`` with one generated
    vector ``v`` per (layer, example), broadcast across sequence positions.

    This is the benchmark's first activation-space codec: its hook site is the whole
    decoder layer (``"block"`` — see ``hypernetwork.py::_resolve_target``), not a linear
    projection, so it edits the residual stream directly and performs no weight update
    at all. The function-vector/task-vector literature shows a single residual-stream
    vector can install an in-context task, which is exactly the T2A question; for D2A
    the vector must carry the conditioned document's needle. Budget is ``d_model``
    scalars per layer — the smallest registered shape alongside (IA)³.

    ``apply()`` is linear in ``generated``: the all-zero initialization is exactly the
    frozen model with a nonzero gradient, so the default ``initial_bias() -> None``
    is correct (no bilinear saddle).
    """

    def __init__(self, in_features: int, out_features: int, scaling: float = 1.0):
        super().__init__(in_features, out_features)
        self.scaling = scaling

    @property
    def output_size(self) -> int:
        return self.out_features

    def apply(self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int) -> Tensor:
        self._check(inputs, generated)
        return base_output + self.scaling * generated.to(base_output.dtype).unsqueeze(1)

    def dense_delta(self, generated: Tensor, layer_index: int) -> Tensor:
        self._check_output_size(generated)
        # Steering is an input-independent additive shift of the hooked activation
        # (a bias-space edit), not a linear map of the input: there is no ΔW with
        # ΔW @ x equal to it. As with IA3, return the correctly shaped zero additive
        # component; `apply()` is the authoritative live implementation.
        return generated.new_zeros(generated.shape[0], self.out_features, self.in_features)


class FourierFTCodec(GeneratedUpdateCodec):
    """Sparse coefficients over a fixed, orthonormal two-dimensional DCT-II basis.

    Every generated scalar multiplies one outer product of fixed input/output DCT
    columns.  Unlike factorized updates this representation has no change-of-basis
    gauge symmetry and is linear in the generated values.  The bases are frozen
    buffers, with an independently sampled set of distinct frequencies per layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_layers: int,
        n_freqs: int,
        scaling: float = 1.0,
        seed: int = 777,
    ):
        super().__init__(in_features, out_features)
        if n_freqs <= 0:
            raise ValueError("FourierFT n_freqs must be positive")
        if n_freqs > in_features * out_features:
            raise ValueError(
                f"FourierFT n_freqs={n_freqs} exceeds the {out_features}x{in_features} "
                "2D DCT spectrum"
            )
        self.n_freqs = n_freqs
        self.num_layers = num_layers
        self.scaling = scaling

        generator = torch.Generator().manual_seed(seed)
        row_frequencies_by_layer = []
        column_frequencies_by_layer = []
        for _ in range(num_layers):
            flat_frequencies = torch.randperm(
                out_features * in_features, generator=generator
            )[:n_freqs]
            row_frequencies = torch.div(flat_frequencies, in_features, rounding_mode="floor")
            column_frequencies = flat_frequencies.remainder(in_features)
            row_frequencies_by_layer.append(row_frequencies)
            column_frequencies_by_layer.append(column_frequencies)

        # Store each complete orthonormal 1D basis once, rather than duplicating
        # repeated columns for every sampled pair and layer (which would consume
        # several GB at benchmark dimensions). Indexing these buffers materializes
        # exactly the Phi_out/Phi_in columns in the codec definition.
        def _dct_basis(size: int) -> Tensor:
            positions = torch.arange(size, dtype=torch.float32).unsqueeze(1)
            frequencies = torch.arange(size).unsqueeze(0)
            normalization = torch.full((size,), math.sqrt(2 / size))
            normalization[0] = math.sqrt(1 / size)
            return normalization * torch.cos(
                math.pi * (2 * positions + 1) * frequencies / (2 * size)
            )

        self.register_buffer("dct_out", _dct_basis(out_features))
        self.register_buffer("dct_in", _dct_basis(in_features))
        self.register_buffer("row_frequencies", torch.stack(row_frequencies_by_layer))
        self.register_buffer("column_frequencies", torch.stack(column_frequencies_by_layer))

    @property
    def output_size(self) -> int:
        return self.n_freqs

    def _delta(self, inputs: Tensor, generated: Tensor, layer_index: int) -> Tensor:
        # Transform once into the complete input-frequency basis, then gather the
        # sampled columns. Many of the n_freqs pairs share a column, so this is much
        # cheaper than multiplying x by a repeated (in_features, n_freqs) matrix.
        projected_all = torch.einsum("bsi,ic->bsc", inputs.to(self.dct_in.dtype), self.dct_in)
        # Place the <1%-dense learned diagonal into a temporary frequency-domain
        # operator and use one batched GEMM. This never forms weight-space ΔW and is
        # substantially faster than thousands of small gather/scatter kernels. The
        # surrounding activation checkpoint keeps this workspace out of the saved
        # per-layer backward state.
        flat_indices = (
            self.column_frequencies[layer_index] * self.out_features
            + self.row_frequencies[layer_index]
        )
        spectral_weights = generated.new_zeros(
            generated.shape[0], self.in_features * self.out_features
        ).scatter(1, flat_indices.unsqueeze(0).expand(generated.shape[0], -1), generated)
        spectral_weights = spectral_weights.reshape(
            generated.shape[0], self.in_features, self.out_features
        )
        spectral_out = torch.bmm(projected_all, spectral_weights)
        delta = torch.einsum("bsr,or->bso", spectral_out, self.dct_out)
        return delta

    def apply(self, inputs: Tensor, base_output: Tensor, generated: Tensor, layer_index: int) -> Tensor:
        self._check(inputs, generated)
        if torch.is_grad_enabled() and (inputs.requires_grad or generated.requires_grad):
            # Recompute the DCT intermediates during backward instead of retaining
            # them for every hooked layer. This is exact activation checkpointing;
            # it changes memory/compute only, not the model forward or gradients.
            delta = checkpoint(
                lambda layer_inputs, layer_generated: self._delta(
                    layer_inputs, layer_generated, layer_index
                ),
                inputs,
                generated,
                use_reentrant=False,
            )
        else:
            delta = self._delta(inputs, generated, layer_index)
        return base_output + self.scaling * delta.to(base_output.dtype)

    def dense_delta(self, generated: Tensor, layer_index: int) -> Tensor:
        self._check_output_size(generated)
        phi_in = self.dct_in[:, self.column_frequencies[layer_index]]
        phi_out = self.dct_out[:, self.row_frequencies[layer_index]]
        return self.scaling * torch.einsum("ok,bk,ik->boi", phi_out, generated, phi_in)


def make_codec(
    name: str,
    in_features: int,
    out_features: int,
    *,
    num_layers: int,
    rank: int = 8,
    alpha: float = 16.0,
    lora_scaling: float | None = None,
    dora_scaling: float = 1.0,
    ia3_scaling: float = 1.0,
    lokr_scaling: float = 1.0,
    fourierft_scaling: float = 1.0,
    steering_scaling: float = 1.0,
    seed: int = 777,
    # Accepted-and-ignored so call sites (and future codecs) can pass a
    # uniform kwarg set without every caller special-casing which codec is registered.
    **_unused_codec_kwargs,
) -> GeneratedUpdateCodec:
    """Build a registered generated-update codec."""

    constructors = {
        "lora": lambda: LoRACodec(in_features, out_features, rank, alpha, scaling=lora_scaling, seed=seed),
        # Weight-decomposed: the same rank-r directional factors plus one generated magnitude
        # scalar per output channel, renormalized by the frozen row norms it reads at the
        # hook site (the only registered codec that reads W0 — see `apply_at`).
        "dora": lambda: DoRACodec(in_features, out_features, rank, scaling=dora_scaling, seed=seed),
        "ia3": lambda: IA3Codec(in_features, out_features, scaling=ia3_scaling),
        "lokr": lambda: LoKrCodec(in_features, out_features, scaling=lokr_scaling, seed=seed),
        # Activation-space: one steering vector per layer added to the residual
        # stream; hook at "block", not a projection. d_model scalars per layer.
        "steering": lambda: SteeringCodec(in_features, out_features, scaling=steering_scaling),
        # One coefficient per generated scalar in the locked rank-r LoRA budget.
        "fourierft": lambda: FourierFTCodec(
            in_features,
            out_features,
            num_layers=num_layers,
            n_freqs=rank * (in_features + out_features),
            scaling=fourierft_scaling,
            seed=seed,
        ),
    }
    try:
        return constructors[name]()
    except KeyError as error:
        raise ValueError(
            f"unsupported differentiable adapter: {name!r} "
            f"(registered codecs: {sorted(constructors)})"
        ) from error


def set_codec_scaling(codecs, scaling: float) -> None:
    """Apply one swept output scale to every codec in a ``codecs`` mapping.

    Every registered codec exposes a uniform scalar ``.scaling``, so the invariant-#2
    best-of-scale sweep needs exactly one mechanism: entry points expose a single
    ``--codec-scaling`` flag and call this, instead of each new codec threading its own
    ``--<name>-scaling`` flag through every trainer/evaluator (the per-codec flags are
    retained for recorded reproduction commands, but new codecs should not add more).
    """
    for codec in codecs.values():
        codec.scaling = scaling
