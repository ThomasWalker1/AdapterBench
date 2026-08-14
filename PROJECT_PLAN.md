# AdapterBench: Completion and Release Plan

## What this is

AdapterBench asks one question: **when a hypernetwork generates a
parameter-efficient adapter, does the adapter's shape matter?**

The benchmark holds the conditioning path, hypernetwork trunk, training data,
optimizer, frozen interpreter, hook sites, and evaluation protocol fixed within each
setting. It varies only the generated representation through one `codec` + `hook site`
seam.

The benchmark contains exactly two genuinely inference-time adaptive language settings:

- **T2A — task-description conditioning.** A task description is available only to the
  hypernetwork. The frozen interpreter receives the problem without the task definition.
- **D2A — document conditioning.** A document is available only through the
  document-conditioned hypernetwork. The frozen interpreter answers a query without the
  document in its ordinary input.

LoRA is the rigorous baseline that validates both settings; alternative shapes are
registered and evaluated one at a time against it. Registered codecs: `lora`, `dora`,
`ia3`, `lokr`, `fourierft`, and `steering` (the first activation-space shape, hooked at
the residual stream).

## Current status (2026-08-04)

Both language settings pass controls that require genuine condition dependence.

<!-- canonical-results:release-summary-markdown:start -->
| setting | frozen interpreter | primary result | condition control |
|---|---|---|---|
| T2A | gemma-2-2b | LoRA `matched − static* = +0.049 ± 0.027`; FourierFT `matched − static* = +0.068 ± 0.044`; (IA)³ `matched − static* = +0.128 ± 0.049`; LoKr `matched − static* = +0.077 ± 0.033`; Steering `matched − static* = +0.119 ± 0.038` ROUGE-L on 11 held-out SNI tasks (3 confirmation seeds each) | independently selected same-shape static control: LoRA wins 3/3 confirmation seeds; FourierFT wins 3/3 confirmation seeds; (IA)³ wins 3/3 confirmation seeds; LoKr wins 3/3 confirmation seeds; Steering wins 3/3 confirmation seeds |
| D2A | Qwen3-0.6B | LoRA (r=8): `matched − context-swap = +0.556 ± 0.327`; FourierFT: `matched − context-swap = +0.656 ± 0.352`; (IA)³: `matched − context-swap = +0.738 ± 0.327`; LoKr: `matched − context-swap = +0.981 ± 0.037`; steering: `matched − context-swap = +0.881 ± 0.050` exact-match | LoRA (r=8) control `0.000`; FourierFT control `0.000`; (IA)³ control `0.000`; LoKr control `0.000`; steering control `0.000` |
<!-- canonical-results:release-summary-markdown:end -->

T2A's selection metric and headline are **ROUGE-L**, Super-NaturalInstructions' own aggregate
metric, measured on the 11 genuinely held-out `lol_` tasks against an independently selected
same-shape static control. Cross-entropy is an appendix figure: it stays a divergence detector and
an eligibility gate and never selects. On the report split CE and ROUGE-L rank the shapes at
Spearman −0.50, so a CE-selected point is not the behaviourally best point.

The five shapes are **not ranked**. Only `(IA)³ > LoRA` (+0.079, 2.4× SE) and
`Steering > LoRA` (+0.069, 2.6× SE) separate; every adjacent pair does not, and `LoKr` vs `(IA)³`
would need ~90 seeds per codec. Read the table as five independent measurements of conditioning
against five independent controls.

D2A's locked numeric-decoy NIAH setting trains at 512 tokens and tests through 32768.
LoRA crosses 0.5 through 8192 (16×); FourierFT, IA³, and LoKr cross through 32768 (64×).

Canonical results and exact commands are in:

- `leaderboards/task_conditioned_t2a.md`
- `leaderboards/document_niah_d2a.md`

Negative investigations are retained as results rather than as dormant settings.
`NEGATIVE_RESULTS.md` records the standard input-visible Text-to-LoRA negative result —
the setting produces helpful adapters but fails a condition control — with its
matched-control measurements, a verification on Sakana's own released checkpoints
(standalone, under `scripts/negative_results/t2l_released_prompt_ablation/`, outside the
benchmark code), and reproduction instructions.

## Where the project stands

The **scientific substrate is complete** and the project is no longer in
setting-discovery or substrate-building mode:

- the active scope is frozen to T2A and D2A;
- both conditions reach the frozen interpreter only through the generated adapter path;
- both settings have behavioral controls, a helpfulness floor, difficulty axes, multi-seed
  LoRA results, and canonical reproduction scripts;
- the shared codec/hook seam, training paths, evaluators, checkpointing, manifests, tests,
  leaderboards, and negative-results record are implemented;
- the standard-T2A conditioning failure is verified on Sakana's released checkpoints
  (`scripts/negative_results/t2l_released_prompt_ablation/`), not just our own diagnostics;
- image and planning infrastructure is intentionally absent.

LoRA is deliberately the validated reference, not the answer to the benchmark's
question. (IA)³, LoKr, and FourierFT now have complete two-setting rows beside it; steering
and DoRA are registered with their protocol evaluations outstanding. Everything
above exists so that question can be asked, which makes **codec exploration the one
substantive remaining phase**.

## Next phase: codec exploration

AdapterBench's reason to exist is the comparison the substrate now makes fair: *does the
generated adapter's shape matter?* Answering it means populating both leaderboards with
the non-LoRA codecs already described in the paper (Section "Codecs"), one at a time,
through the stable extension path in "Stable extension interface" below. None of this
changes the shared substrate.

Candidate codecs to add (each is a `GeneratedUpdateCodec` subclass + a `make_codec` entry
+ a `configs/adapters/<name>.yaml` + unit tests):

| codec | update | budget / site | reparam. symmetry | why it is interesting | status |
|---|---|---|---|---|---|
| (IA)³ | `W ↦ diag(1+v) W` | `d_out` | none | can a tiny, symmetry-free shape carry conditioning at all? | both rows complete |
| LoKr | `ΔW = B ⊗ A` | factor-dependent | scaling only | full-rank reach from few scalars; different budget/expressivity trade-off | both rows complete |
| FourierFT | `ΔW = F⁻¹(sparse coeffs)` | `n` (chosen), size-independent | none | fixed global basis removes the low-rank rotation symmetry entirely | both rows complete |
| steering (`steering`) | `h ↦ h + s·v` | `d_model` / residual stream (`block`) | scaling | adaptation with no weight edit; hook site is the residual stream, not a projection | registered; autoresearch pending |
| DoRA (`dora`) | `W ↦ m ⊙ (W₀+s·BA)/‖W₀+s·BA‖_row` | rank-8 LoRA budget + `d_out` | direction-only rotation (magnitude is gauge-fixed) | the only shape defined *relative to the weight it edits*: magnitude and direction are separately addressable and pre-normalized by the host weight, so it tests whether one-shot prediction is limited by calibration rather than expressivity | registered; autoresearch in flight |

The open empirical question is whether the low-rank rotation symmetry LoRA carries
(`BA = (BG)(G⁻¹A)`) is a real obstacle to one-shot prediction, i.e. whether a
symmetry-free or lower-budget shape conditions *better* than LoRA under identical training.
For each codec the deliverable is a leaderboard row in **both** T2A and D2A: `matched −
control` at the codec's best swept scale, at least three seeds, with the same conditioner,
trunk, data, evaluator, and controls as the LoRA reference. Losing shapes stay on the
leaderboards — a shape that fails to condition is itself a result.

Per-codec worklist (repeat the "Stable extension interface" steps):

1. Implement the `GeneratedUpdateCodec` subclass (geometry, initialization, `apply`,
   `dense_delta`, `initial_bias` if bilinear) and register it.
2. Add its manifest/config and unit tests (geometry, init, hook application).
3. Run codec-specific scale selection (D2A's useful scale is far above PEFT defaults;
   sweep, don't assume).
4. Run ≥3 seeds in each setting; record aggregates as compact canonical metrics.
5. Add a row to each leaderboard with exact commands and artifacts.

## Remaining release prerequisites

Mechanical, and independent of codec exploration — none require another research phase:

- [ ] clean-environment installation and every documented command verified;
- [ ] GPU smoke tests and full LoRA reproductions on release hardware (the sandbox has no
      CUDA-visible GPU, the only expected local failure);
- [ ] confirm/commit the final T2A scale-selection evidence and each codec's D2A sweep summary as
      compact records;
- [ ] TeX/PDF build and visual proof of the paper and website;
- [ ] license decision (`CITATION.cff`, version `0.1.0`, `CHANGELOG.md` are present).

Already done: versioned canonical T2A/D2A aggregates with a drift check, provenance
hashes and model revisions, smoke/full reproduction entry points, preflight diagnostics,
and manifest/catalog/drift checks passing locally.

Repository policy: experiment outputs under `results/` are scratch unless explicitly
force-added as compact canonical metrics. Checkpoints, adapters, logs, and large artifacts
are never committed. The human drives commits.

## Architecture

### Shared contracts and codec seam

- `src/adapterbench/contracts.py` — `HypernetworkBackend.generate()`,
  `DownstreamEvaluator.evaluate()`, and shared task/result objects.
- `src/adapterbench/t2a/codecs.py` — `GeneratedUpdateCodec` plus the registered codecs
  (`LoRACodec`, `DoRACodec`, `IA3Codec`, `LoKrCodec`, `FourierFTCodec`, `SteeringCodec`).
  A new shape implements:
  - `output_size`;
  - `apply(...)`;
  - `dense_delta(...)`;
  - `initial_bias()` when required by a bilinear parameterization.
- `src/adapterbench/t2a/hypernetwork.py` — the shared hypernetwork shell, per-layer heads,
  codec application, and hook lifecycle.
- `src/adapterbench/t2a/model_utils.py` — frozen interpreter loading and layer discovery.
- `src/adapterbench/t2a/live_evaluator.py` — hook-based downstream evaluation for both
  settings.

The generated adapter is applied live during the frozen interpreter's forward pass.
Training backpropagates ordinary next-token cross-entropy through the hook into the
hypernetwork. Adapters are not materialized through `peft.PeftModel` during training or
evaluation.

### T2A: task-description conditioning

- `src/adapterbench/t2a/lol_data.py` — vendored Lots-of-LoRAs/SNI task loading,
  decontaminated task validation, and definition stripping.
- `src/adapterbench/t2a/condition_encoder.py` — pooled task-description conditioner.
- `src/adapterbench/t2a/sft_trainer.py` — fixed-budget and checkpointed T2A SFT,
  gradient accumulation, warmup, and static-adapter training.
- `scripts/t2a_train_ddp.py` — data-parallel T2A training.
- `scripts/t2a_eval_heldout_sni_acc.py` — greedy generation scored for **ROUGE-L and exact match off
  one decode pass**: the primary metric and its cross-check. `--example-offset` skips the examples
  training consumed; `--split` is selected by the re-scoring driver below.
- `scripts/t2a_eval_heldout_sni.py` — held-out teacher-forced CE, now an appendix figure and the
  divergence/eligibility gate, never the selector.
- `scripts/t2a_score_checkpoints.py` — the scoring driver for both splits, resumable, with the
  report split guarded behind `--split report --confirm-spend-report-split --checkpoint-filter`.
- `scripts/t2a_sweep_axes.py` / `t2a_selection_seeds.py` / `t2a_confirmation_seeds.py` — the
  scout, §3b selection-seed and §4 confirmation-seed phases of the re-derivation.
- `scripts/reproduce/task_t2a_lora.sh` — canonical one-seed LoRA reproduction.

Training tasks are vendored under `data/t2a/`. The loader refuses tasks outside T2A's
479-task decontaminated training split and excludes all held-out validation tasks.

### D2A: document conditioning

- `src/adapterbench/t2a/document_conditioning.py` —
  `capture_early_exit_representation` plus `EarlyExitPerceiverConditioner`.
- `src/adapterbench/t2a/niah_data.py` — deterministic needle/haystack examples,
  realistic-prose distractors, packing guards, and evaluation examples.
- `src/adapterbench/t2a/document_sft_trainer.py` — restart-safe document-conditioned
  training with atomic model/optimizer/scheduler checkpoints.
- `src/adapterbench/t2a/live_evaluator.py` —
  `DocumentHypernetworkDownstreamEvaluator`, including the context-swap control.
- `scripts/d2a_niah_aggregate.py` — multi-seed and length-generalization aggregation.
- `scripts/reproduce/document_niah_lora.sh` — canonical D2A LoRA reproduction.

Each document is packed into one interpreter context. The implementation deliberately
does not reproduce Doc-to-LoRA's multi-chunk `combine_lora`; the one-pass constraint is
enforced explicitly.

## Why the settings are genuinely adaptive

### T2A

Standard task-conditioned SFT often gives the frozen interpreter the same task definition
that conditions the hypernetwork. In that design, the adapter is redundant.

AdapterBench strips the definition from the interpreter input. The task is available only
through:

`task description → conditioner → hypernetwork → generated adapter`

The primary comparison is against a directly optimized static adapter of the same shape
trained on the same multi-task data. `matched − static < 0` CE therefore isolates
task-specific conditioning from generic adapter help.

### D2A

The query does not contain the document or needle. The only document path is:

`document → frozen early-exit features → perceiver conditioner → hypernetwork → adapter`

At evaluation, the context-swap control uses the adapter generated from a different
document. Its exact-match accuracy is zero while the matched adapter retrieves the held-out
needle, ruling out answer priors and context-independent formatting behavior.

## Benchmark invariants

Every leaderboard entry must satisfy all four:

1. **Behavioral metric with a condition control.** Report `matched − control`, never raw
   quality alone.
2. **Best-of-scale comparison.** Sweep each codec's scale and report its best valid
   operating point.
3. **Graded difficulty.** T2A reports task-level behavior and D2A reports context-length
   generalization rather than a single saturated point.
4. **Multi-seed evidence.** Use at least three seeds and report variation.

Within a setting, comparisons must keep fixed:

- condition features and conditioner architecture;
- hypernetwork trunk and per-layer head budget;
- training examples and optimizer protocol;
- frozen interpreter;
- hook sites where semantically compatible;
- evaluator and controls;
- generated-scalar budget.

## T2A result

The rigorous T2A baseline uses ROUGE-L on the **11 genuinely held-out SNI tasks** (report
split), with an independently selected same-shape static control. Confirmation seeds are
disjoint from the selection seeds that chose the operating point.

| codec | matched − static* ROUGE-L | confirmation seeds |
|---|---|---|
| LoRA | **+0.049 ± 0.027** | 1741, 1742, 1743 |
| FourierFT | +0.068 ± 0.044 | 5041, 5042, 5043 |
| (IA)³ | +0.128 ± 0.049 | 1751, 1752, 1753 |
| LoKr | +0.077 ± 0.033 | 2741, 2742, 2743 |
| Steering | +0.119 ± 0.038 | 4741, 4742, 4743 |

CE and exact match are appendix figures off the same decode pass; neither selects. On this split
CE and ROUGE-L rank the shapes at Spearman −0.50.

Canonical records: `canonical_results/t2a_*.json`. Source artifacts:
`results/autoresearch/t2a/<codec>/confirmation_v2/report_split_s*.jsonl`.

## D2A result

The shipped setting uses realistic Wikipedia-prose haystacks, one topic-free four-digit
needle, four explicitly irrelevant four-digit decoys, Qwen3-0.6B, early-exit document
features, a perceiver conditioner, and a generated codec on the selected projection.

Across five seeds at the 512-token training length:

- LoRA (r=8) matched exact-match: `0.556 ± 0.327`; context-swap: `0.000`; matched minus control: `+0.556 ± 0.327`.
- (IA)³ matched exact-match: `0.738 ± 0.327`; context-swap: `0.000`; matched minus control: `+0.738 ± 0.327`.

The length-generalization crossover is 8192 tokens (16×) for LoRA and 32768 (64×) for FourierFT, IA³, and LoKr.

## How to run

Install and validate:

```bash
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
uv run adapterbench doctor --require-cuda
uv run adapterbench validate
uv run adapterbench catalog
uv run pytest -q
```

Reproduce one T2A confirmation seed:

```bash
bash scripts/reproduce/task_t2a_lora.sh 1741 0,1,2,3 4,5,6,7
```

Full three-seed headline for LoRA:

```bash
bash scripts/reproduce/t2a_reproduce_all.sh lora 0,1,2,3 4,5,6,7
```

Inspect the canonical aggregate:

```bash
.venv/bin/python scripts/t2a_confirmation_result.py
cat results/autoresearch/t2a/evaluation/confirmation_result.json
```

Run the D2A numeric-decoy NIAH confirmation:

```bash
bash scripts/reproduce/document_niah_numeric_decoy_lora.sh cuda:0
```

Run the D2A scale locator:

```bash
bash scripts/d2a_numeric_decoy_dev.sh
```

The held-out-SNI evaluations must run with `HF_HUB_OFFLINE=1` after metadata is vendored.
The reproduction script sets it automatically.

## Gotchas

### Shared

- Freeze every interpreter parameter. Train only the conditioner, hypernetwork, generated
  representation, or explicitly selected static reference.
- Do not detach generated adapter tensors before the hooked forward pass.
- Remove every hook in `finally`; leaked hooks silently contaminate later evaluations.
- A codec with bilinear generated factors must use a nonzero saddle-breaking
  initialization.
- Keep generated-parameter counts and hook sites explicit in manifests.
- A positive raw score without a condition control is not an adaptive result.

### T2A

- Definition stripping is load-bearing. If the interpreter sees the task definition,
  conditioning becomes redundant.
- The static reference must use the same codec shape, data, hook sites, and training loss.
- Use the vendored 479-task decontaminated split. Do not silently substitute a generic
  SuperNI split.
- The held-out metadata is local; online model metadata checks can trigger Hugging Face
  rate limits even when all model/data files are cached.
- CE and behaviour answer different questions, and on genuinely held-out tasks they *anti-correlate*
  (Spearman −0.50). ROUGE-L selects and headlines; CE is a divergence detector and eligibility gate.
  A CE-selected operating point is reliably not the behaviourally best one.
- The 10 `lol_` eval tasks that appear in `train_ds_names` must be scored at `--example-offset 40`.
  Training consumes the leading 40 examples of the same `train[:10000]` split, so offset 0 replays
  training examples: it inflates matched, inflates the *static* more, and biases the scale argmax high.

### D2A

- Early-exit features plus the perceiver conditioner are load-bearing. The removed
  full-depth conditioner did not learn retrieval.
- Use the realistic-prose haystack for the canonical row. Repeated synthetic noise is a
  useful diagnostic but an easier task.
- The context-swap control must remain exactly zero or near chance.
- Ensure the full document fits in one interpreter pass; do not silently truncate.
- Strong L1 regularization can collapse the generated LoRA to zero. The upstream-style
  coefficient `1.5` failed; `0` or `0.1` reproduced retrieval.
- D2A's useful LoRA scale is much larger than ordinary PEFT defaults. Scale must be swept,
  not assumed.

## Stable extension interface

There is one leaderboard per active setting:

- `leaderboards/task_conditioned_t2a.md`
- `leaderboards/document_niah_d2a.md`

Settings are never pooled because their metrics differ. The cross-setting question is
whether a codec's relative behavior repeats.

A codec is added through the stable extension path (this is exactly the "Next phase:
codec exploration" worklist):

1. subclass `GeneratedUpdateCodec`;
2. register it in `make_codec`;
3. add `configs/adapters/<name>.yaml`;
4. add unit tests for geometry, initialization, and hook application;
5. run codec-specific scale selection;
6. run at least three seeds in both settings;
7. add a row to each leaderboard with exact commands and artifacts.

Losing shapes remain in the leaderboards: a shape that fails to condition is a result, not
a failure. Additional *settings* (beyond T2A/D2A) are out of scope and would be considered
only if they passed a preregistered matched-control audit; no image or planning setting is
planned.

## Reference: prior art

The benchmark is motivated by:

- Text-to-LoRA and Lots-of-LoRAs;
- Doc-to-LoRA;
- PEFT representations such as LoRA, IA3, LoKr, FourierFT, and activation steering.

AdapterBench's contribution is the controlled comparison: the adapter is a
hypernetwork output, live end-to-end training is shared, and only representation shape
changes.
