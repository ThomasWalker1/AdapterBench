"""HyperNoise image setting in the AdapterBench codec seam (see PROJECT_PLAN.md, "Image domain").

Reimplements "Noise Hypernetworks" (Eyring et al., 2025; github.com/ExplainableML/HyperNoise)
as an AdapterBench setting: an adapter on a frozen distilled generator (SD-Turbo)
modulates the *initial noise* to maximize a reward, trained end-to-end by the tractable
noise-space objective `L(φ) = reg·mean(Δx₀²) − r(g_θ(x₀+Δx₀))`. Unlike from-scratch personalization
(ruled out by the de-risk), reward-tilting is something a small adapter demonstrably CAN do, so the
setting has signal — and the adapter is a LoRA, so AdapterBench's shape question applies natively.

The adapter is a **directly-trained** codec (`DirectCodecAdapter`), matching HyperNoise's
optimizer-fit LoRA — the degenerate "identity hypernetwork" case of the seam (the generated
numbers ARE trainable parameters). Other codec shapes drop in through `make_codec` unchanged;
a per-condition *generated* upgrade is a documented future direction (see PROJECT_PLAN.md).

Two deviations from the reference impl, both documented:
- **f_φ via the difference form** `unet_adapted(x₀) − unet_base(x₀)` (exactly 0 at init by the codec's
  `initial_bias` contract), instead of their single-pass `conv_out`-delta-only patch. Codec-agnostic
  and correct for any hook site; costs one extra (no_grad) UNet pass.
- **Hook site = attention linears** (attn1/attn2 q/k/v/out) rather than every Conv+Linear, so the
  existing Linear-oriented codec seam is reused verbatim.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor, nn

from .image_generator import FrozenGenerator
from ..t2p.codecs import make_codec

F_PHI_TIMESTEP = 999.0  # HyperNoise runs the noise network at max timestep (x0 is pure noise)


def find_attention_linears(unet: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Broad hook set: every Linear inside a cross/self attention (`attn1.`/`attn2.` q/k/v/out).
    Matches HyperNoise's broad LoRA coverage (minus conv layers), for enough modulation capacity."""
    targets = []
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear) and (".attn1." in name or ".attn2." in name):
            targets.append((name, module))
    return targets


class DirectCodecAdapter(nn.Module):
    """A directly-trained adapter over a fixed set of target Linear modules, one codec each.

    The trainable parameter per target IS the codec's generated-output vector (initialised from
    `codec.initial_bias()`, so LoRA starts at B=0 → exactly-zero contribution at init). This is the
    unconditional/identity-hypernetwork instantiation of the AdapterBench seam: swap `codec_name`
    to compare shapes; nothing else changes.
    """

    def __init__(self, targets: list[tuple[str, nn.Linear]], *, codec_name: str = "lora",
                 rank: int = 16, alpha: float = 32.0, scaling: float | None = None, seed: int = 777):
        super().__init__()
        self.target_names = [name for name, _ in targets]
        self._modules_by_name = {name: module for name, module in targets}
        codecs, deltas = {}, {}
        for i, (name, module) in enumerate(targets):
            key = name.replace(".", "__")  # ModuleDict keys can't contain '.'
            # scaling (if given) is the swept LoRA-scale axis (invariant #2), applied
            # directly to every codec so a scale sweep is not confounded by rank/alpha.
            codec = make_codec(codec_name, module.in_features, module.out_features,
                               num_layers=1, rank=rank, alpha=alpha, lora_scaling=scaling, seed=seed + i)
            codecs[key] = codec
            init = codec.initial_bias()
            deltas[key] = nn.Parameter(init.clone() if init is not None else torch.zeros(codec.output_size))
        self.codecs = nn.ModuleDict(codecs)
        self.deltas = nn.ParameterDict(deltas)
        self.codec_name = codec_name

    def generated_parameter_count(self) -> int:
        return sum(p.numel() for p in self.deltas.values())

    @contextmanager
    def apply(self, batch: int):
        """Hook every target module so its output becomes `codec.apply(input, output, delta)`, with
        the single trainable delta broadcast across the batch. Mirrors `hypernetwork.apply`."""
        handles = []
        for name in self.target_names:
            key = name.replace(".", "__")
            codec, delta, module = self.codecs[key], self.deltas[key], self._modules_by_name[name]

            def hook(module, args, output, *, codec=codec, delta=delta):
                generated = delta.unsqueeze(0).expand(batch, -1)
                return codec.apply(args[0], output, generated, 0)

            handles.append(module.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


def noise_transform(gen: FrozenGenerator, adapter: DirectCodecAdapter, x0: Tensor,
                    encoder_hidden_states: Tensor) -> Tensor:
    """f_φ(x₀): Δx₀ = unet_adapted(x₀,t=999,c) − unet_base(x₀,t=999,c). Base pass is adapter-free
    (no_grad, since it does not depend on the adapter); the adapted pass carries the gradient. Δx₀=0
    at init (LoRA B=0), matching HyperNoise's initialization and the L2/KL-small assumption."""
    timestep = torch.tensor(F_PHI_TIMESTEP, device=x0.device)
    with torch.no_grad():
        base = gen.unet(x0, timestep, encoder_hidden_states=encoder_hidden_states).sample
    with adapter.apply(batch=x0.shape[0]):
        adapted = gen.unet(x0, timestep, encoder_hidden_states=encoder_hidden_states).sample
    return adapted - base


def generate_from_latents(gen: FrozenGenerator, latents: Tensor, encoder_hidden_states: Tensor,
                          *, num_steps: int = 1) -> Tensor:
    """Clean g_θ: few-step SD-Turbo generation from a GIVEN initial latent (the modulated noise),
    adapters OFF, guidance-free. Gradient flows through g_θ back to the modulated latent (and hence
    the adapter), so the VAE decode here is deliberately grad-capable (an eval-only `@torch.no_grad`
    decode would sever the reward's gradient path to the adapter)."""
    scheduler = gen.scheduler
    scheduler.set_timesteps(num_steps, device=gen.device)
    latents = latents * scheduler.init_noise_sigma
    for timestep in scheduler.timesteps:
        model_input = scheduler.scale_model_input(latents, timestep)
        predicted = gen.unet(model_input, timestep, encoder_hidden_states=encoder_hidden_states).sample
        latents = scheduler.step(predicted, timestep, latents).prev_sample
    images = gen.vae.decode(latents / gen.vae.config.scaling_factor).sample  # frozen VAE, but differentiable in latents
    return (images / 2 + 0.5).clamp(0, 1)


def hypernoise_loss(gen: FrozenGenerator, adapter: DirectCodecAdapter, x0: Tensor,
                    encoder_hidden_states: Tensor, reward_fn, prompts: list[str], *,
                    reg_weight: float = 0.5, num_steps: int = 1):
    """The HyperNoise objective: `reg·mean(Δx₀²) − mean(reward)`. `reward_fn(images01, prompts)`
    returns a per-image reward to maximize (see `i2p/rewards.py`). Returns (loss, metrics)."""
    delta = noise_transform(gen, adapter, x0, encoder_hidden_states)
    x_hat = x0 + delta
    images = generate_from_latents(gen, x_hat, encoder_hidden_states, num_steps=num_steps)
    reward = reward_fn(images, prompts).mean()
    reg = (delta ** 2).mean()
    loss = reg_weight * reg - reward
    return loss, {"reward": float(reward.detach()), "reg": float(reg.detach()),
                  "delta_rms": float(delta.detach().pow(2).mean().sqrt()), "loss": float(loss.detach())}
