# AdapterBench — user guide

AdapterBench asks one question: **does the *shape* of a hypernetwork-generated PEFT
adapter matter?** Text-to-LoRA, Program-as-Weights, and Doc-to-LoRA all have a
hypernetwork emit a LoRA — but none tests that choice of representation against
alternatives under a fixed generation-and-evaluation protocol. AdapterBench holds the
hypernetwork shell, training loop, data, and evaluator constant and varies **only** the
generated representation (the *codec*), so any difference in the scored result is
attributable to shape.

This guide covers: the three settings, how a result is scored (the metric and its
controls), how to add a new codec, and how to run and reproduce each setting. For
environment setup see [SETUP.md](SETUP.md); for the interface contract see
[BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md).

---

## 1. The three settings

Each setting shares the same frozen interpreter (`Qwen3-0.6B`) and the same
hypernetwork shell, and differs only in **what the hypernetwork is conditioned on** and
**how the adapter is scored**.

| Setting | Conditioned on | The adapter should… | Behavioral metric | Control (what a non-conditioning adapter can't pass) |
|---------|----------------|---------------------|-------------------|------------------------------------------------------|
| **T2L** — task-conditioned | pooled embedding of a free-text *task description* | install a whole task | held-out answer accuracy | **junk-description** adapter must not match, **and** matched must beat frozen (helpfulness floor) |
| **D2L** — document-conditioned | cross-attention over the interpreter's own activations for a *document* | retrieve info from that one document | needle-in-a-haystack exact-match | **context-swap**: adapter from the *wrong* document must fall to chance |
| **I2P** — image reward-tilting | (a fixed reward) on a frozen SD-Turbo generator | raise a differentiable reward | ImageReward gain over frozen | **reward-swap**: adapter trained for a near-orthogonal reward must not raise ImageReward |

The settings are never pooled — their absolute scores use different conditioning,
objectives, and evaluators. The cross-setting question is whether a codec's *relative*
behavior repeats across protocols.

---

## 2. How a result is scored — the metric and invariants

Every setting reports **matched − control**, never raw performance, and satisfies four
invariants (see the paper's protocol section):

1. **Behavioral metric with a built-in control.** A control that an adapter exploiting
   priors, memorization, or format-only cues cannot pass. The headline is always
   `matched − control`.
2. **Scale is swept, best-of reported** — so a shape claim reads "even at its own best
   scale, shape *X* underperforms," not an artifact of a fixed default scale.
3. **A graded difficulty knob** (e.g. D2L's eval context length, I2P's edit-regularization
   weight) so codecs spread instead of all saturating.
4. **Multi-seed** — at least three seeds; the spread is reported.

### The T2L two-part metric (matched − junk **and** matched − frozen)

For T2L specifically, `matched − control` alone is **not sufficient**. A control that is a
gap between the matched adapter and a *wrong-description* adapter can be inflated by an
adapter that **sabotages** the wrong-description case (drives it far below the frozen
model) without the matched adapter ever helping. Such a model scores a large
matched − junk while never installing the task.

AdapterBench therefore requires **two** conditions for genuine T2L conditioning, both
reported by `scripts/t2p_rigor_aggregate.py`:

- **matched − junk > 0** — the description changes the adapter (adversarial control:
  the junk description is maximally dissimilar/meaningless, e.g. `dogs;cats;bananas`), and
- **matched − frozen ≥ 0** — the adapter genuinely *helps* the task (helpfulness floor).

The aggregator prints a verdict — `NO CONDITIONING` / `SABOTAGE` / `WEAK-GENERIC` /
`GENUINE CONDITIONING` — with a noise floor tied to the eval sample size.

---

## 3. Adding a new codec

A new adapter shape needs exactly two things: an **output structure** (a differentiable
codec — how many numbers the hypernetwork emits and how they become an update) and a
**hook site** (a named linear submodule for weight-space adapters, or a whole layer /
the residual stream for activation-space ones). Concretely:

1. **Subclass `GeneratedUpdateCodec`** in `src/adapterbench/t2p/codecs.py`:
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
   (`codecs.py`). LoRA is the only registered shape on `main`.
3. **Add a manifest** under `configs/adapters/` so the CLI can select it.
4. **Run the setting CLIs** (Section 4) at the codec's own best free hyperparameters and
   **record a leaderboard row**.

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

### T2L — task-conditioned

Data-parallel training at the shipped effective-batch-128 config, held-out eval with the
adversarial control, then the two-part aggregate:

```bash
# reproduce the LoRA baseline (3 seeds, DDP across the given GPUs)
scripts/reproduce/task_t2l_lora_ddp.sh 0,1,2,3,4,5,6,7

# aggregate one run's results.jsonl -> matched-junk AND matched-frozen verdict
.venv/bin/python scripts/t2p_rigor_aggregate.py --results <run>/results.jsonl
```

Helpers: `scripts/t2l_base_diag.sh <hf-interpreter> <tag> <gpus> <per-gpu-batch>` runs the
shipped recipe with a swappable base model and optional training-objective knobs
(`STEPS`, `LR`, `SNAP`, `LIMIT`, `CLAMBDA`, `NJLAMBDA` env vars); `--snapshot-every` +
`scripts/t2p_eval_checkpoint.py` score any checkpoint (any eval-task set, `--avg-descriptions`)
without retraining.

### D2L — document-conditioned (NIAH)

Trains at a fixed 256-token context and evaluates in-distribution plus a length-generalization
sweep:

```bash
scripts/reproduce/document_niah_lora.sh                       # 5 seeds, realistic haystack
.venv/bin/python scripts/d2p_niah_aggregate.py --root <run> --min-seeds 5 --train-length 256
```

### I2P — image reward-tilting

```bash
scripts/reproduce/image_lora.sh
```

---

## 5. Where results live

- **`leaderboards/*.md`** — one file per setting: the committed baseline row(s) with the
  shape, its free hyperparameters, `matched − control ± std`, seed count, and the exact
  reproduce command. This is the source of truth for the benchmark's headline numbers.
- **`scripts/reproduce/`** — one script per setting's baseline; each is self-contained.
- **`results/repro/`** — the leaderboard-backing run outputs (`results.jsonl` + metrics).

A leaderboard row is produced by: run the reproduce script → aggregate → paste the
headline (with its control and seed spread) and the reproduce command into the setting's
leaderboard file.
