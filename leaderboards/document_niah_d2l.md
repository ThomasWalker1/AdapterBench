# Language domain — document-conditioned generation (Doc-to-LoRA, NIAH)

**Task.** The hypernetwork cross-attends over a frozen `Qwen3-0.6B`'s own early-exit activations of
a target document (Perceiver-IO: 208 latents, 8 blocks) and emits an adapter that retrieves from
that one document. Evaluated on needle-in-a-haystack: a topic-free needle
(`The special magic number is NNNN.`) hidden among **real Wikipedia prose** (`--needle-style
realistic`, BoolQ passages); scoring is exact four-digit match. One unified run per seed trains at
256 tokens and evaluates at 256 (in-distribution, the headline) **and** a length sweep to 8192 (the
difficulty knob) — the old separate base-retrieval and length-gen runs are consolidated into this.

**Metric — `matched − control`.** Matched = held-out retrieval accuracy. Control =
**context-swap**: an adapter generated from the *wrong* document must fall to chance
(`accuracy_ctxswap`). Headline = accuracy − accuracy_ctxswap. A result counts only when accuracy is
high **and** the control is ~0 (rules out digit priors, memorization, format-only learning).

**Free HPs:** scale, learning rate, warmup, steps. **Fixed (substrate):** early-exit depth, latent
count (208), cross-attention blocks (8), training documents, context length. **Fixed:** rank.
Retrieval is a hard, stochastic phase transition (onset ~1.6k–4.5k steps across seeds), and loss
saturates near-0 well before accuracy leaves the floor — so multi-seed and matched−control (never
loss) are essential here.

Headline row is the **in-distribution** point (train length = eval length = 256), 5 seeds:

| Shape | scale | lr | steps | seeds | accuracy | ctxswap | **matched − control** |
|-------|:-----:|-------:|------:|:-----:|---------:|--------:|----------------------:|
| LoRA (r=8) | ≈45.25 (default) | 4e-5 | 12000 | 5 | 0.887 ± 0.143 | 0.000 | **+0.887 ± 0.143** |

**Difficulty knob — length generalization (same run, eval sweep):** mean matched−control by eval
length (5 seeds, ctxswap 0.000 at every length):

| eval len | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| matched−control | 0.887 | 0.844 | 0.806 | 0.806 | 0.762 | 0.369 |

**Crossover(0.5) = 4096 tokens = 16× the 256-token training length.** Retrieval is a hard, stochastic
phase transition (onset ~3.7k–5.4k steps across seeds; per-seed in-distribution: 0.97/0.84/1.0/0.63/1.0),
which is why 5 seeds + the transition-step spread are reported, not a single lucky number.

**Reproduce:** `scripts/reproduce/document_niah_lora.sh [DEVICE]` (5 seeds). The per-seed command:

```bash
.venv/bin/adapterbench d2p-niah --adapters lora --needle-style realistic \
  --context-lengths 256 --eval-context-lengths 256,512,1024,2048,4096,8192 \
  --num-train-documents 512 --steps 12000 --eval-every 1000 \
  --learning-rate 4e-5 --n-latents 208 --num-blocks 8 --eval-limit 32 \
  --seed 777 --output results/repro/document_niah_realistic/s777
```
