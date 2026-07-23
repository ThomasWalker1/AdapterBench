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
count (208), cross-attention blocks (8), training documents, context length. **Fixed:** each
codec's shape-defining parameter budget.
Retrieval can be a stochastic phase transition, and loss can saturate near-0 before accuracy leaves
the floor — so multi-seed and matched−control (never loss) are essential here.

Every headline row is the **in-distribution** point (train length = eval length = 256), with its
listed confirmation seeds:

<!-- canonical-results:d2l-markdown:start -->
| Shape | scale | lr | steps | seeds | accuracy | ctxswap | **matched − control** |
|---|:---:|---:|---:|:---:|---:|---:|:---:|
| LoRA (r=8) | 45.25 | 4e-5 | 12,000 | 5 | 0.887 ± 0.143 | 0.000 | **+0.887 ± 0.143** |
| (IA)³ | 64 | 2e-5 | 36,000 | 3 | 1.000 ± 0.000 | 0.000 | **+1.000 ± 0.000** |

| eval len | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| matched − control (LoRA (r=8)) | 0.887 | 0.844 | 0.806 | 0.806 | 0.762 | 0.369 |
| matched − control ((IA)³) | 1.000 | 1.000 | 1.000 | 0.990 | 0.833 | 0.354 |
<!-- canonical-results:d2l-markdown:end -->

**Difficulty knob — length generalization (same run, eval sweep):** mean matched−control by eval
length (the listed confirmation seeds; ctxswap 0.000 at every length):

The canonical records above are the source of these curves; the context-swap value is `0.000`
at every evaluated length for both codecs.

**Crossover(0.5) = 4096 tokens = 16× the 256-token training length** for both canonical rows.

**Reproduce:** `scripts/reproduce/document_niah_lora.sh [DEVICE]` (5 seeds) and
`scripts/reproduce/document_niah_ia3.sh [DEVICE]` (3 seeds). Per-seed commands:

```bash
.venv/bin/adapterbench d2p-niah --adapters lora --needle-style realistic \
  --context-lengths 256 --eval-context-lengths 256,512,1024,2048,4096,8192 \
  --num-train-documents 512 --steps 12000 --eval-every 1000 \
  --learning-rate 4e-5 --n-latents 208 --num-blocks 8 --eval-limit 32 \
  --seed 777 --output results/repro/document_niah_realistic/s777
```

```bash
.venv/bin/adapterbench d2p-niah --adapters ia3 --ia3-scaling 64 --needle-style realistic \
  --context-lengths 256 --eval-context-lengths 256,512,1024,2048,4096,8192 \
  --num-train-documents 512 --steps 36000 --eval-every 1000 --learning-rate 2e-5 --warmup-frac 0.03 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 --seed 786 --device cuda:0 \
  --output results/repro/document_niah_ia3/scale64/s786
```
