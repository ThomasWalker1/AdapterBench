# Language domain — document-conditioned generation (Doc-to-LoRA, NIAH)

**Task.** The hypernetwork cross-attends over a frozen `Qwen3-0.6B`'s own early-exit activations of
a target document (Perceiver-IO: 208 latents, 8 blocks) and emits an adapter that retrieves from
that one document. Evaluated on synthetic needle-in-a-haystack: a topic-free needle
(`The special magic number is NNNN.`) hidden in noise; scoring is exact four-digit match.

**Metric — `matched − control`.** Matched = held-out retrieval accuracy. Control =
**context-swap**: an adapter generated from the *wrong* document must fall to chance
(`accuracy_ctxswap`). Headline = accuracy − accuracy_ctxswap. A result counts only when accuracy is
high **and** the control is ~0 (rules out digit priors, memorization, format-only learning).

**Free HPs:** scale, learning rate, warmup, steps. **Fixed (substrate):** early-exit depth, latent
count (208), cross-attention blocks (8), training documents, context length. **Fixed:** rank.
Retrieval is a hard, stochastic phase transition (onset ~1.6k–4.5k steps across seeds), and loss
saturates near-0 well before accuracy leaves the floor — so multi-seed and matched−control (never
loss) are essential here.

Baseline commit: `1ec5951`.

| Shape | scale | lr | steps | seeds | accuracy | ctxswap | **matched − control** |
|-------|:-----:|-------:|------:|:-----:|---------:|--------:|----------------------:|
| LoRA (r=8) | ≈45.25 (default) | 4e-5 | 6000 | 1 | 1.00 | 0.00 | **1.00** |

**Difficulty knob — length generalization:** train on 128–256-token documents, evaluate out to
8192; report the retrieval-vs-length curve and the eval/train length ratio at which retrieval
crosses 0.5.

**Reproduce:** `scripts/reproduce/document_niah_lora.sh [DEVICE]` (3 seeds; the length-gen variant
is a commented block in that script). The per-seed command it wraps:

```bash
.venv/bin/adapterbench d2p-niah --adapters lora --needle-style generic \
  --context-lengths 384 --num-train-documents 512 --steps 6000 --eval-every 500 \
  --learning-rate 4e-5 --n-latents 208 --num-blocks 8 --eval-limit 32 \
  --seed 777 --output results/repro/document_niah_lora/s777
```

> **Gap:** the recorded number is currently single-seed; the script runs 3 seeds to fill the
> multi-seed invariant. Length-gen is already 6-seed (`d2p_lengthgen_lora_s777..782`).
