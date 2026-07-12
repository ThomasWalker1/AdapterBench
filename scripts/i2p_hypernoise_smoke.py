"""Seam smoke for the reward-tilting image setting: does the two-pass noise-transform run,
is Δx₀=0 at init (zero-init contract), does the reward backprop to the adapter (and NOT the
frozen UNet), and does one step make Δx₀≠0? Uses the cheap redness reward.

Run: .venv/bin/python scripts/i2p_hypernoise_smoke.py --device cuda:0
"""

from __future__ import annotations

import argparse

import torch

from adapterbench.i2p.hypernoise import (
    DirectCodecAdapter, find_attention_linears, generate_from_latents, hypernoise_loss, noise_transform,
)
from adapterbench.i2p.image_generator import embed_prompt, load_frozen_generator
from adapterbench.i2p.rewards import redness_reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    print("[1/5] load frozen SD-Turbo ...", flush=True)
    gen = load_frozen_generator(device=dev, dtype=torch.float32)
    targets = find_attention_linears(gen.unet)
    print(f"      hooked attention linears: {len(targets)}")
    adapter = DirectCodecAdapter(targets, codec_name="lora", rank=args.rank).to(dev)
    print(f"      adapter trainable params: {adapter.generated_parameter_count():,}")

    x0 = torch.randn(args.batch, gen.unet.config.in_channels, 64, 64, device=dev)
    ehs = embed_prompt(gen, ["a photo"] * args.batch)

    print("[2/5] Δx₀ at init (must be ~0: zero-init contract) ...", flush=True)
    with torch.no_grad():
        delta0 = noise_transform(gen, adapter, x0, ehs)
    print(f"      ||Δx₀||_rms at init = {delta0.pow(2).mean().sqrt().item():.2e}")
    assert delta0.abs().max().item() < 1e-4, "Δx₀ not zero at init — initial_bias/zero-init broken"

    print("[3/5] loss + backward → adapter grads (frozen UNet none) ...", flush=True)
    opt = torch.optim.SGD(adapter.parameters(), lr=1e-2)
    opt.zero_grad(set_to_none=True)
    loss, metrics = hypernoise_loss(gen, adapter, x0, ehs, redness_reward, ["a photo"] * args.batch, reg_weight=0.5)
    loss.backward()
    nonzero = [n for n, p in adapter.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0]
    unet_grad = [n for n, p in gen.unet.named_parameters() if p.grad is not None]
    print(f"      metrics={metrics}")
    print(f"      adapter params w/ nonzero finite grad: {len(nonzero)}/{len(list(adapter.parameters()))}; frozen UNet grads: {len(unet_grad)}")
    assert len(nonzero) > 0, "no adapter parameter received a finite non-zero gradient"
    assert not unet_grad, "frozen UNet received gradients"

    print("[4/5] optimizer step → Δx₀ must become nonzero ...", flush=True)
    opt.step()
    with torch.no_grad():
        delta1 = noise_transform(gen, adapter, x0, ehs)
    print(f"      ||Δx₀||_rms after 1 step = {delta1.pow(2).mean().sqrt().item():.2e}")
    assert delta1.abs().max().item() > 1e-6, "Δx₀ still zero after a step — adapter not training"

    print("[5/5] generate + redness reward is finite ...", flush=True)
    with torch.no_grad():
        img = generate_from_latents(gen, x0, ehs, num_steps=1)
        r = redness_reward(img)
    print(f"      image {tuple(img.shape)} range [{img.min():.3f},{img.max():.3f}]  redness={r.mean().item():+.4f}")
    assert img.shape[-1] == 512 and torch.isfinite(img).all()

    print("\nHYPERNOISE SMOKE: PASS")


if __name__ == "__main__":
    main()
