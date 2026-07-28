# Language domain — document-conditioned generation (Doc-to-LoRA, NIAH)

**Task.** The hypernetwork cross-attends over a frozen `Qwen3-0.6B`'s own early-exit activations of
a target document (Perceiver-IO: 208 latents, 8 blocks) and emits an adapter that retrieves from
that one document. The locked NIAH document contains one queried topic-free needle
(`The special magic number is NNNN.`) plus four distinct, explicitly irrelevant four-digit records
among real Wikipedia prose (`--needle-style realistic_numeric_decoys`, BoolQ passages). The frozen
interpreter receives only the fixed query; scoring is exact four-digit match. Every confirmation
run trains at 512 tokens and evaluates 512–32768 tokens.

**Metric — `matched − control`.** Matched = held-out retrieval accuracy. Control =
**context-swap**: an adapter generated from the *wrong* document must fall to chance
(`accuracy_ctxswap`). Headline = accuracy − accuracy_ctxswap. A result counts only when accuracy is
high **and** the control is ~0 (rules out digit priors, memorization, format-only learning).

**Free HPs:** scale, learning rate, warmup, steps. **Fixed (substrate):** early-exit depth, latent
count (208), cross-attention blocks (8), training documents, context length. **Fixed:** each
codec's shape-defining parameter budget.
Retrieval can be a stochastic phase transition, and loss can saturate near-0 before accuracy leaves
the floor — so multi-seed and matched−control (never loss) are essential here.

Every headline row is the **in-distribution** point (train length = eval length = 512), with five
held-out confirmation seeds:

<!-- canonical-results:d2l-markdown:start -->
| Shape | scale | lr | steps | seeds | accuracy | ctxswap | **matched − control** |
|---|:---:|---:|---:|:---:|---:|---:|:---:|
| LoRA (r=8) | 100 | 4e-5 | 32,000 | 5 | 0.556 ± 0.327 | 0.000 | **+0.556 ± 0.327** |
| (IA)³ | 64 | 2e-5 | 32,000 | 5 | 0.738 ± 0.327 | 0.000 | **+0.738 ± 0.327** |

| eval len | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|
| matched − control (LoRA (r=8)) | 0.556 | 0.619 | 0.594 | 0.550 | 0.506 | 0.456 | 0.425 |
| matched − control ((IA)³) | 0.738 | 0.725 | 0.738 | 0.719 | 0.725 | 0.681 | 0.688 |
<!-- canonical-results:d2l-markdown:end -->

**Difficulty knob — length generalization (same run, eval sweep):** mean matched−control by eval
length (the listed confirmation seeds; ctxswap 0.000 at every length). The normalized log-length
tail AUC (1024–32768) is **0.526 ± 0.328** for LoRA and **0.714 ± 0.356** for IA³.

The canonical records above are the source of these curves; the context-swap value is `0.000`
at every evaluated length for both codecs.

**Crossover(0.5):** LoRA reaches 8192 tokens (16× the 512-token training length); IA³ reaches
32768 tokens (64×).

**Reproduce:** `scripts/reproduce/document_niah_numeric_decoy_lora.sh [DEVICE]` and
`scripts/reproduce/document_niah_numeric_decoy_ia3.sh [DEVICE]` (five seeds each). The shared
implementation is `scripts/reproduce/document_niah_numeric_decoy.sh`; set `SEEDS=<one seed>` to
schedule one seed in parallel. Per-seed commands:

```bash
.venv/bin/adapterbench d2p-niah --adapters lora --lora-scaling 100 \
  --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
  --context-lengths 512 --eval-context-lengths 512,1024,2048,4096,8192,16384,32768 \
  --num-train-documents 512 --steps 32000 --eval-every 8000 --learning-rate 4e-5 --warmup-steps 960 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 --eval-seed 2904 --seed 2901 \
  --output results/repro/d2l_numeric_decoy/lora/s2901
```

```bash
.venv/bin/adapterbench d2p-niah --adapters ia3 --ia3-scaling 64 \
  --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
  --context-lengths 512 --eval-context-lengths 512,1024,2048,4096,8192,16384,32768 \
  --num-train-documents 512 --steps 32000 --eval-every 8000 --learning-rate 2e-5 --warmup-steps 960 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 --eval-seed 2904 --seed 2901 --device cuda:0 \
  --output results/repro/d2l_numeric_decoy/ia3/s2901
```
