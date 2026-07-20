# Image domain — reward tilting (SD-Turbo, ImageReward)

**Task.** An adapter modulates the initial noise of a frozen distilled text-to-image
generator (`stabilityai/sd-turbo`) so the sampled image scores higher under a fixed, differentiable
reward (ImageReward). Trained end-to-end by `L(φ) = λ·‖Δx₀‖² − r(g(x₀+Δx₀))`.

**This is the unconditional (directly-optimized) member.** Unlike T2L/D2L, the adapter is *not*
generated from a per-instance condition — the trainable parameter *is* the codec output, shared
across all prompts (the "identity-hypernetwork" case of the seam). So the question here is whether a
codec *shape* supports a noise edit that raises a *specific* reward, not whether it installs
prompt-specific content. Consistent with that, the single optimized edit is prompt-**generic** (the
gain survives a prompt-swap probe) — an expected, characterized property, not a control the row must
pass. The operative control is **reward-swap** (below). A prompt-*conditioned* hypernetwork for this
setting is a documented future direction; a preliminary probe generated a working reward-tilting
adapter (+0.14 IR) but did not become prompt-specific.

**Metric — `matched − control`.** Matched = ImageReward gain of the adapter over the frozen
generator. Control = **reward-swap**: an adapter trained for a near-orthogonal reward (image
redness) must *not* raise ImageReward. Headline = matched IR-gain − reward-swap IR-gain. We also
report the CLIP image–text fidelity drop (a hacked gain shows up as collapsed fidelity); CLIP uses a
different encoder than the reward so the check is not circular.

**Free HPs:** scale, `λ` (reg weight), learning rate, steps. **Fixed:** rank (shape identity);
reward target, batch size, eval protocol (substrate). Optimizer is SGD+momentum.

Baseline commit: `1ec5951`.

| Shape | scale | λ | steps | lr | seeds | matched IR-gain | reward-swap ctrl | **matched − control** | CLIP-T drop |
|-------|------:|-----:|------:|-----:|:-----:|----------------:|-----------------:|----------------------:|------------:|
| LoRA (r=16) | 4 | 0.25 | 3000 | 1e-3 | 3 | +0.160 ± 0.015 | −3.34 ± 0.01 | **+3.50** | +0.004 |

Scale is a sharp inverted-U (λ=0.25, single seed): 2:+0.185 · 3:+0.079 · 4:+0.170 · 6:+0.106 ·
8:−0.107 · 16:−1.20 · 32:−0.36 — optimum at 2–4; beyond it the noise edit destroys the image.

**Reproduce:** `scripts/reproduce/image_lora.sh [DEVICE]` (runs matched + reward-swap control over
3 seeds, then aggregates). The per-seed command it wraps:

```bash
.venv/bin/adapterbench i2p-hypernoise --reward imagereward \
  --scale 4 --reg-weight 0.25 --steps 3000 --eval-every 500 \
  --rank 16 --batch-size 2 --n-seeds 2 --seed 777 \
  --output results/repro/image_lora/imagereward_s777
```
