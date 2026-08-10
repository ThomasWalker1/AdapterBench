# AdapterBench — user guide

AdapterBench asks one question: **does the *shape* of a hypernetwork-generated PEFT
adapter matter?** Text-to-LoRA and Doc-to-LoRA both have a hypernetwork emit a LoRA —
but neither tests that choice of representation against alternatives under a fixed
generation-and-evaluation protocol. AdapterBench holds the hypernetwork shell, training
loop, data, and evaluator constant and varies **only** the generated representation (the
*codec*), so any difference in the scored result is attributable to shape.

This guide covers: the active settings, how a result is scored (the metric and its
controls), how to add a new codec, and how to run and reproduce each setting. For
environment setup see [SETUP.md](SETUP.md); for the interface contract see
[BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md).

---

## 1. The active settings

Each setting fixes a frozen interpreter and the same hypernetwork shell, and differs only
in **what the hypernetwork is conditioned on** and **how the adapter is scored**. The
interpreter is per-setting (`gemma-2-2b` for T2A and `Qwen3-0.6B` for D2A); within a
setting it is fixed and only the codec varies.

| Setting | Conditioned on | The adapter should… | Behavioral metric | Control (what a non-conditioning adapter can't pass) |
|---------|----------------|---------------------|-------------------|------------------------------------------------------|
| **T2A** — task-conditioned | pooled embedding of a free-text *task description* | install a whole task (definition stripped from the input) | held-out-SNI **ROUGE-L** (primary) + exact match and CE as appendix | **matched − static\***: must beat an *independently selected* same-shape static multi-task adapter |
| **D2A** — document-conditioned | cross-attention over the interpreter's own activations for a *document* | retrieve info from that one document | needle-in-a-haystack exact-match | **context-swap**: adapter from the *wrong* document must fall to chance |

The settings are never pooled — their absolute scores use different conditioning,
objectives, and evaluators. The cross-setting question is whether a codec's *relative*
behavior repeats across protocols.

Retired investigations are not hidden: [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md)
collates settings that failed a condition-shuffle or behavioral-capacity gate. This
includes standard input-visible Text-to-LoRA; it is distinct from the definition-stripped
T2A setting above.

---

## 2. How a result is scored — the metric and invariants

Every setting reports **matched − control**, never raw performance, and satisfies four
invariants (see the paper's protocol section):

1. **Behavioral metric with a built-in control.** A control that an adapter exploiting
   priors, memorization, or format-only cues cannot pass. The headline is always
   `matched − control`.
2. **Scale is swept, best-of reported** — so a shape claim reads "even at its own best
   scale, shape *X* underperforms," not an artifact of a fixed default scale.
3. **A graded difficulty knob** (e.g. D2A's eval context length or T2A's task-family
   headroom) so codecs spread instead of all saturating.
4. **Multi-seed** — at least three seeds; the spread is reported.

### The T2A metric: matched − static (definition-stripped)

Standard T2A SFT concatenates the task *definition* and the problem in the prompt, so the
frozen interpreter reads the task straight from its input and the description-conditioned
adapter is **redundant** — every codec then scores ≈0 conditioning, masking the shape
differences the benchmark exists to measure. AdapterBench removes this redundancy: the
definition is **stripped** from the input (`--strip-task-def`), so the task is specifiable
*only* through the description → hypernetwork → adapter (the D2A "context is necessary"
principle applied to T2A). Conditioning is still over the task-level description; only the
interpreter's input changes.

The control is **matched − static**: the conditioned adapter is scored against a *static
reference* — a single adapter of the **same shape**, directly optimized on the same SFT
data (a multi-task LoRA, not emitted by any hypernetwork), which absorbs all task-*generic*
help. `eval_ds_info` holds 21 `lol_` tasks but only **11 are absent from `train_ds_names`**, so the
two splits are disjoint on task *and* example: hyperparameters are chosen on the 10 in-distribution
tasks at `--example-offset 40` (training consumes the leading 40 examples of the same split), and the
11 genuinely held-out tasks are scored exactly once by the confirmation seeds:

- **ROUGE-L (primary):** Super-NaturalInstructions' own aggregate metric — LCS F-measure over
  greedy generations, averaged over the 3 held-out description variants. Higher is better;
  `matched − static* > 0` means conditioning helps. Scored by
  `scripts/t2a_eval_heldout_sni_acc.py`, which reads ROUGE-L and exact match off one decode pass.
- **exact match (appendix):** normalized exact match from the *same* generations. Stricter and
  unambiguous, so it catches ROUGE-L partial-credit inflation, but near-meaningless on the
  open-ended tasks — never the selector.
- **CE (appendix):** teacher-forced cross-entropy, scored by `scripts/t2a_eval_heldout_sni.py`.
  It is the training objective, so it keeps two jobs — divergence detector and eligibility gate —
  and loses selection. On the held-out split CE and ROUGE-L rank the shapes at Spearman −0.50.

`matched − static*` **subsumes the helpfulness floor** (static\* ≥ frozen) and needs **no junk
descriptions**; `matched − frozen` is reported alongside to confirm the adapter helps at all. It is
**not non-gameable**: it cannot be gamed by sabotaging a wrong-condition case, but it *can* be
inflated by handicapping the control's optimization — a control driven toward a no-op makes the
difference arbitrarily large. That is why the control is selected independently, on its own score,
and why a control approaching the frozen model marks a control failure rather than a good result.

---

## 3. Adding a new codec

A new adapter shape needs exactly two things: an **output structure** (a differentiable
codec — how many numbers the hypernetwork emits and how they become an update) and a
**hook site** (a named linear submodule for weight-space adapters, or a whole layer /
the residual stream for activation-space ones). Concretely:

1. **Subclass `GeneratedUpdateCodec`** in `src/adapterbench/t2a/codecs.py`:
   - `output_size` (property) — how many scalars the hypernetwork head emits per
     (layer, module).
   - `apply(inputs, base_output, generated, layer_index)` — fold the generated scalars
     into the hooked module's output.
   - `dense_delta(generated, layer_index)` — the `(batch, out, in)` weight-space ΔW.
   - `initial_bias()` — override only if `apply()` is **bilinear** in `generated` (two
     factors multiplied, like LoRA's `B@A` or LoKr's Kronecker factors); return a bias
     that randomizes one factor's slice so the head escapes the all-zero dead saddle.
     Linear codecs (IA³, activation steering, FourierFT) can keep the default `None`.
2. **Register it** — add one entry to the `constructors` dict in `make_codec`
   (`codecs.py`) and its default hook site in `PILOT_DEFAULT_TARGET_MODULES`
   (`cli/_shared.py`) + `DOC_TO_LORA_PARITY_TARGET_MODULES` (`cli/live_sft.py`). Registered
   shapes: `lora`, `ia3`, `lokr`, `loha`, `fourierft`, `steering`.
3. **Add a manifest** under `configs/adapters/` so the CLI can select it (and add the
   family to `schema.py`'s `AdapterManifest.family` literal).
4. **Run the setting CLIs** (Section 4) at the codec's own best free hyperparameters and
   **record a leaderboard row**. Scale sweeps use the single generic `--codec-scaling`
   flag (every codec exposes a uniform `.scaling`); no per-codec flag is needed.

The training loop, data pipeline, hypernetwork trunk, evaluator, and controls are
identical for every codec that plugs in this way — that is the whole point.

### The hyperparameter partition (fair comparison)

Comparing codecs under one shared recipe only measures them *under that recipe*. Each
shape is reported **at its own best configuration**, under a strict partition:

- **Shared substrate** (never tuned per shape): training data, hypernetwork trunk,
  evaluator, control.
- **Free** (chosen per shape to maximize matched − control over ≥3 seeds): output scale,
  learning rate, warmup, step budget.
- **Shape identity** (defines the adapter): e.g. LoRA's rank.

---

## 4. Running and reproducing each setting

Environment first (see [SETUP.md](SETUP.md)):

```bash
uv venv .venv --python 3.11 && uv pip install -e ".[dev]"
uv run adapterbench doctor --require-cuda && uv run pytest -q
```

### T2A — task-conditioned

Trains the strip-def hypernetwork **and** the same-shape static reference (data-parallel,
gemma-2-2b), then scores `matched − static*` ROUGE-L on the 11 genuinely held-out SNI tasks
(with exact match and CE as appendix figures).
Evals require `HF_HUB_OFFLINE=1` (the model and datasets are cached; this avoids the HF Hub
rate limit) — the reproduce script sets it for you.

```bash
# reproduce the LoRA baseline for one seed: trains hyper+static, then both evals
scripts/reproduce/task_t2a_lora.sh 777 0,1,2,3 4,5,6,7    # SEED GPUS_HYPER GPUS_STATIC

# results (per-task + __aggregate__ rows):
#   .../heldout_sni_acc.jsonl                     (ROUGE-L primary + exact match, one decode pass)
#   .../heldout_sni_ce_full21.jsonl               (CE: appendix figure, divergence/eligibility gate)
#   rescore/report_scores/*.jsonl                 (the one-shot 11-task report-split measurement)
```

Helpers: `scripts/t2a_base_diag.sh <hf-interpreter> <tag> <gpus> <per-gpu-batch>` runs the
recipe with a swappable base model and env knobs (`SEED`, `STEPS`, `LR`, `SNAP`, `LIMIT`,
`ELIMIT`, `STRIPDEF`, `STATIC`); the 21 held-out tasks' metadata is vendored (once) by
`scripts/vendor_heldout_sni_metadata.py`. Baseline: `matched − static* = +0.049 ± 0.027` ROUGE-L
(59/63 task-seed pairs, 3 seeds).

### D2A — document-conditioned (NIAH)

The locked numeric-decoy setting trains at a fixed 512-token context and evaluates
in-distribution plus a length-generalization sweep out to 32768 tokens:

```bash
scripts/reproduce/document_niah_numeric_decoy_lora.sh cuda:0   # 5 seeds, realistic haystack + decoys
.venv/bin/python scripts/d2a_niah_aggregate.py --root <run> --min-seeds 5 --train-length 512
```

(`scripts/reproduce/document_niah_lora.sh` is the earlier decoy-free 256-token recipe,
retained for the historical realistic-haystack diagnostic.)

---

## 5. Where results live

- **`leaderboards/*.md`** — one file per setting: rendered baseline rows with the shape,
  free hyperparameters, `matched − control ± variation`, seed count, and exact command.
- **`canonical_results/`** — versioned aggregate records with seed values, provenance,
  source-artifact hashes, exact commands, and difficulty curves. These are the source of
  truth for rendered result tables.
- **`scripts/reproduce/`** — one script per setting's baseline; each runs a preflight and
  is restart-safe.
- **`results/repro/`** — the leaderboard-backing run outputs (`results.jsonl` + metrics).

A leaderboard row is produced by: run the reproduce script → aggregate → update the
canonical record → render/check the structured tables with `adapterbench results check`.
