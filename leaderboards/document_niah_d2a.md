# Language domain — document-conditioned generation (Doc-to-Adapter, NIAH)

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

Every headline row is the **in-distribution** point (train length = eval length = 512), with
five held-out confirmation seeds:

<!-- canonical-results:d2a-markdown:start -->
| Shape | scale | lr | steps | seeds | accuracy | ctxswap | **matched − control** |
|---|:---:|---:|---:|:---:|---:|---:|:---:|
| LoRA (r=8) | 100 | 4e-5 | 32,000 | 5 | 0.556 ± 0.327 | 0.000 | **+0.556 ± 0.327** |
| DoRA | 64 | 2e-5 | 32,000 | 5 | 0.956 ± 0.073 | 0.000 | **+0.956 ± 0.073** |
| FourierFT | 4 | 2e-5 | 36,000 | 5 | 0.656 ± 0.352 | 0.000 | **+0.656 ± 0.352** |
| (IA)³ | 64 | 2e-5 | 32,000 | 5 | 0.738 ± 0.327 | 0.000 | **+0.738 ± 0.327** |
| LoKr | 16 | 2e-5 | 36,000 | 5 | 0.981 ± 0.037 | 0.000 | **+0.981 ± 0.037** |
| steering | 32 | 2e-5 | 32,000 | 5 | 0.881 ± 0.050 | 0.000 | **+0.881 ± 0.050** |

| eval len | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|
| matched − control (LoRA (r=8)) | 0.556 | 0.619 | 0.594 | 0.550 | 0.506 | 0.456 | 0.425 |
| matched − control (DoRA) | 0.956 | 0.994 | 0.969 | 0.988 | 0.975 | 0.969 | 0.894 |
| matched − control (FourierFT) | 0.656 | 0.688 | 0.700 | 0.650 | 0.631 | 0.562 | 0.581 |
| matched − control ((IA)³) | 0.738 | 0.725 | 0.738 | 0.719 | 0.725 | 0.681 | 0.688 |
| matched − control (LoKr) | 0.981 | 0.994 | 0.981 | 0.994 | 0.975 | 0.981 | 0.925 |
| matched − control (steering) | 0.881 | 0.925 | 0.944 | 0.906 | 0.963 | 0.912 | 0.800 |

**Selection trail.**
- **DoRA** — The operating point was located by a joint scale x learning-rate sweep, which DoRA needs because its two free axes interact rather than separating: at the substrate-default lr 4e-5 no scale on the identity-centred ladder retrieves, while at lr 2e-5 scale 64 reaches the 512-token in-distribution gate by 8k steps. The sweep ran on the scout seed 902 with the dev eval instrument (eval seed 1802) and varied only free axes, so it never touched the held-out instrument. The point was then re-derived on protocol at the full declared instrument and the 32000-step budget the committed rows use: scales 32, 64 and 128 at lr 2e-5 all reach the gate ceiling of 1.000 with normalized hard-length AUC 0.983, 1.000 and 0.983, so they are a tied set of three and scale 64 is promoted as the AUC argmax without claiming a resolved optimum among them; the upper endpoint 128 does not beat 64, so no further extension is licensed, and the LR axis closes interior at 2e-5 (4e-5 does not retrieve, 1e-5 collapses to 0.083). The section 3b stability gate passed 3/3 on selection seeds 6201/6202/6203 (gates 1.000, 0.500, 1.000; divergence rate 0/3) - the same gate that ruled steering's analogous scale-64 point ineligible at 1/3 - and five fresh confirmation seeds on the held-out eval instrument retained the result. Every rung, including the ones that did not retrieve, is retained in the append-only ledger hashed above. State ledger: `results/autoresearch/d2a/dora/state.jsonl`.
- **FourierFT** — The scale ladder at seed 5201 promoted scale 4 at 16k steps. Its lower geometric LR neighbor 2e-5 beat 4e-5 and 8e-5, then improved to tail AUC 0.659 at the 36k rung. The three-selection-seed stability gate compared LR 2e-5 with 4e-5: 2e-5 had the better mean tail AUC (0.778 vs 0.743) and much lower population SD (0.070 vs 0.359), with 3/3 finite runs for both. Five disjoint confirmations (5301–5305) were finite; seed 5305 exposed substantial retrieval-phase variance rather than numerical divergence. State ledger: `results/autoresearch/d2a/fourierft/state.jsonl`.
- **LoKr** — A LoKr-identity-centered scale ladder (0.0625, 0.25, 1, 4, 16) was zero through 4; scale 16 was the only helpful point. The geometric upper boundary at 64 was zero at both 8k and 16k, closing the scale choice. At scale 16, the setting-default LR 4e-5 and upper neighbor 8e-5 were weak/zero, while 2e-5 won at 16k; its restart-safe 36k promotion improved hard-length AUC to 0.675. Five fresh 36k confirmations in the clean repro path (seeds 3003–3007) retained every result. State ledger: `results/autoresearch/d2a/lokr/state.jsonl`.
- **steering** — Steering's ladder was centred on its own identity convention (h -> h + scale*v from a zero-initialised head, so every scale is exactly the frozen model at initialization and the manifest default 1.0 is the natural centre), never on another codec's scale. The declared x4 ladder 0.0625-16 was retrieval-zero at 8k and 16k for every point; at 32k only the top endpoint 16 passed the 512-token gate, and weakly (hard-length AUC 0.158; re-scoring its checkpoint at 32 examples/bin gave 0.125, establishing a real but weak effect rather than sampling noise). Extending outward per protocol, scale 64 retrieved strongly on the scout (gate 1.000, AUC 0.867) but converged on only 1 of 3 selection seeds and is therefore INELIGIBLE under the stability gate; 128 and 256 (the declared safety limit) both diverged after a strong transient peak, and both scale-64 LR neighbours failed (2e-5 diverged at the final rung, 8e-5 never retrieved). Because the gate split the ladder, a predeclared x2 refinement tested the untested midpoint: scale 32 clears the gate 3/3 with divergence rate 0/3 at both tested learning rates. The LR axis then closed with an interior optimum at 2e-5 (mean selection-seed AUC 0.886 at 1e-5, 0.911 at 2e-5, 0.725 at 4e-5, ineligible at 8e-5). An earlier declared 32k->64k step extension was abandoned at 48k once scale 64 showed the apparent late onset was an artifact of an under-scaled operating point rather than of the step budget. Steering's stable band is narrow: one x2 step above the selected point drops convergence to 1 of 3 and two steps destroys training, consistent with an activation-space gain acting on a residual stream whose RMS grows with depth. Every selection number is a mean over the three declared selection seeds 6101/6102/6103 at the common 32k budget, never a scout peak; five fresh confirmation seeds retained the result. State ledger: `results/autoresearch/d2a/steering/state.jsonl`.
<!-- canonical-results:d2a-markdown:end -->

**Difficulty knob — length generalization (same run, eval sweep):** mean matched−control by eval
length (the listed confirmation seeds; ctxswap 0.000 at every length). The normalized log-length
tail AUC (1024–32768) is **0.526 ± 0.328** for LoRA, **0.636 ± 0.338** for FourierFT,
**0.714 ± 0.356** for IA³, **0.975 ± 0.022** for LoKr, and **0.969 ± 0.042** for DoRA.

The canonical records above are the source of these curves; the context-swap value is `0.000`
at every evaluated length for every codec.

**Crossover(0.5):** LoRA reaches 8192 tokens (16× the 512-token training length); FourierFT, IA³,
LoKr, steering and DoRA reach 32768 tokens (64×).

**Reproduce:** `scripts/reproduce/document_niah_numeric_decoy_lora.sh [DEVICE]` and
`scripts/reproduce/document_niah_numeric_decoy_ia3.sh [DEVICE]` (five seeds each), or
`scripts/reproduce/document_niah_numeric_decoy_lokr_all.sh [DEVICE]` or
`scripts/reproduce/document_niah_numeric_decoy_fourierft_all.sh [DEVICE]` (five seeds). The shared
implementation is `scripts/reproduce/document_niah_numeric_decoy.sh`; set `SEEDS=<one seed>` to
schedule one seed in parallel. Per-seed commands:

```bash
.venv/bin/adapterbench d2a-niah --adapters lora --lora-scaling 100 \
  --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
  --context-lengths 512 --eval-context-lengths 512,1024,2048,4096,8192,16384,32768 \
  --num-train-documents 512 --steps 32000 --eval-every 8000 --learning-rate 4e-5 --warmup-steps 960 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 --eval-seed 2904 --seed 2901 \
  --output results/repro/d2a_numeric_decoy/lora/s2901
```

```bash
.venv/bin/adapterbench d2a-niah --adapters ia3 --ia3-scaling 64 \
  --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
  --context-lengths 512 --eval-context-lengths 512,1024,2048,4096,8192,16384,32768 \
  --num-train-documents 512 --steps 32000 --eval-every 8000 --learning-rate 2e-5 --warmup-steps 960 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 --eval-seed 2904 --seed 2901 --device cuda:0 \
  --output results/repro/d2a_numeric_decoy/ia3/s2901
```

```bash
.venv/bin/adapterbench d2a-niah --adapters lokr --lokr-scaling 16 \
  --needle-style realistic_numeric_decoys --numeric-decoy-count 4 \
  --context-lengths 512 --eval-context-lengths 512,1024,2048,4096,8192,16384,32768 \
  --num-train-documents 512 --steps 36000 --eval-every 8000 --learning-rate 2e-5 --warmup-steps 1080 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 --eval-seed 2904 --seed 3003 --device cuda:0 \
  --output results/repro/d2a_numeric_decoy/lokr/s3003
```
