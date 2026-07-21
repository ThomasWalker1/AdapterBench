# AdapterBench: Completion and Release Plan

## What this is

AdapterBench asks one question: **when a hypernetwork generates a
parameter-efficient adapter, does the adapter's shape matter?**

The benchmark holds the conditioning path, hypernetwork trunk, training data,
optimizer, frozen interpreter, hook sites, and evaluation protocol fixed within each
setting. It varies only the generated representation through one `codec` + `hook site`
seam.

The benchmark contains exactly two genuinely inference-time adaptive language settings:

- **T2L — task-description conditioning.** A task description is available only to the
  hypernetwork. The frozen interpreter receives the problem without the task definition.
- **D2L — document conditioning.** A document is available only through the
  document-conditioned hypernetwork. The frozen interpreter answers a query without the
  document in its ordinary input.

LoRA is currently the only registered codec; it is the rigorous baseline that validates
both settings before alternative shapes are added.

## Current status (2026-07-21)

Both language settings pass controls that require genuine condition dependence.

<!-- canonical-results:release-summary-markdown:start -->
| setting | frozen interpreter | primary result | condition control |
|---|---|---|---|
| T2L | gemma-2-2b | `matched − static = −0.723 ± 0.162` nats CE over 3 seeds | matched beats a same-shape static multi-task LoRA on 59/63 task-seed pairs |
| D2L | Qwen3-0.6B | `matched − context-swap = +0.887 ± 0.143` exact-match over 5 seeds | wrong-document adapters score `0.000` |
<!-- canonical-results:release-summary-markdown:end -->

T2L generation accuracy corroborates the CE result:
`matched − static = +0.0317 ± 0.0060`; `matched − frozen = +0.235`.

D2L generalizes beyond its 256-token training contexts. Its accuracy crosses 0.5 at
4096 tokens, or 16× the training length.

Canonical results and exact commands are in:

- `leaderboards/task_conditioned_t2l.md`
- `leaderboards/document_niah_d2l.md`

Negative investigations are retained as results rather than as dormant settings.
`NEGATIVE_RESULTS.md` records the standard input-visible Text-to-LoRA negative result —
the setting produces helpful adapters but fails a condition control — with its
matched-control measurements, a verification on Sakana's own released checkpoints
(standalone, under `scripts/negative_results/t2l_released_prompt_ablation/`, outside the
benchmark code), and reproduction instructions.

## Where the project stands

The **scientific substrate is complete** and the project is no longer in
setting-discovery or substrate-building mode:

- the active scope is frozen to T2L and D2L;
- both conditions reach the frozen interpreter only through the generated adapter path;
- both settings have behavioral controls, a helpfulness floor, difficulty axes, multi-seed
  LoRA results, and canonical reproduction scripts;
- the shared codec/hook seam, training paths, evaluators, checkpointing, manifests, tests,
  leaderboards, and negative-results record are implemented;
- the standard-T2L conditioning failure is verified on Sakana's released checkpoints
  (`scripts/negative_results/t2l_released_prompt_ablation/`), not just our own diagnostics;
- image and planning infrastructure is intentionally absent.

**LoRA is the only registered codec.** It is deliberately the validated reference, not the
answer to the benchmark's question. Everything above exists so that question can now be
asked, which makes **codec exploration the one substantive remaining phase**.

## Next phase: codec exploration

AdapterBench's reason to exist is the comparison the substrate now makes fair: *does the
generated adapter's shape matter?* Answering it means populating both leaderboards with
the non-LoRA codecs already described in the paper (Section "Codecs"), one at a time,
through the stable extension path in "Stable extension interface" below. None of this
changes the shared substrate.

Candidate codecs to add (each is a `GeneratedUpdateCodec` subclass + a `make_codec` entry
+ a `configs/adapters/<name>.yaml` + unit tests):

| codec | update | budget / site | reparam. symmetry | why it is interesting |
|---|---|---|---|---|
| (IA)³ | `W ↦ diag(1+v) W` | `d_out` | none | can a tiny, symmetry-free shape carry conditioning at all? |
| LoKr | `ΔW = B ⊗ A` | factor-dependent | scaling only | full-rank reach from few scalars; different budget/expressivity trade-off |
| FourierFT | `ΔW = F⁻¹(sparse coeffs)` | `n` (chosen), size-independent | none | fixed global basis removes the low-rank rotation symmetry entirely |
| activation steering | `h ↦ h + s·v` | `d_model` | scaling | adaptation with no weight edit; hook site is the residual stream, not a projection |

The open empirical question is whether the low-rank rotation symmetry LoRA carries
(`BA = (BG)(G⁻¹A)`) is a real obstacle to one-shot prediction, i.e. whether a
symmetry-free or lower-budget shape conditions *better* than LoRA under identical training.
For each codec the deliverable is a leaderboard row in **both** T2L and D2L: `matched −
control` at the codec's best swept scale, at least three seeds, with the same conditioner,
trunk, data, evaluator, and controls as the LoRA reference. Losing shapes stay on the
leaderboards — a shape that fails to condition is itself a result.

Per-codec worklist (repeat the "Stable extension interface" steps):

1. Implement the `GeneratedUpdateCodec` subclass (geometry, initialization, `apply`,
   `dense_delta`, `initial_bias` if bilinear) and register it.
2. Add its manifest/config and unit tests (geometry, init, hook application).
3. Run codec-specific scale selection (D2L's useful scale is far above PEFT defaults;
   sweep, don't assume).
4. Run ≥3 seeds in each setting; record aggregates as compact canonical metrics.
5. Add a row to each leaderboard with exact commands and artifacts.

## Remaining release prerequisites

Mechanical, and independent of codec exploration — none require another research phase:

- [ ] clean-environment installation and every documented command verified;
- [ ] GPU smoke tests and full LoRA reproductions on release hardware (the sandbox has no
      CUDA-visible GPU, the only expected local failure);
- [ ] confirm/commit the final T2L scale-selection evidence and D2L's full sweep summary as
      a compact record (D2L's selected operating scale is 45.25);
- [ ] TeX/PDF build and visual proof of the paper and website;
- [ ] license decision (`CITATION.cff`, version `0.1.0`, `CHANGELOG.md` are present).

Already done: versioned canonical T2L/D2L LoRA aggregates with a drift check, provenance
hashes and model revisions, smoke/full reproduction entry points, preflight diagnostics,
and all 94 unit tests + manifest/catalog/drift checks passing locally.

Repository policy: experiment outputs under `results/` are scratch unless explicitly
force-added as compact canonical metrics. Checkpoints, adapters, logs, and large artifacts
are never committed. The human drives commits.

## Architecture

### Shared contracts and codec seam

- `src/adapterbench/contracts.py` — `HypernetworkBackend.generate()`,
  `DownstreamEvaluator.evaluate()`, and shared task/result objects.
- `src/adapterbench/t2p/codecs.py` — `GeneratedUpdateCodec` plus the registered
  `LoRACodec`. A new shape implements:
  - `output_size`;
  - `apply(...)`;
  - `dense_delta(...)`;
  - `initial_bias()` when required by a bilinear parameterization.
- `src/adapterbench/t2p/hypernetwork.py` — the shared hypernetwork shell, per-layer heads,
  codec application, and hook lifecycle.
- `src/adapterbench/t2p/model_utils.py` — frozen interpreter loading and layer discovery.
- `src/adapterbench/t2p/live_evaluator.py` — hook-based downstream evaluation for both
  settings.

The generated adapter is applied live during the frozen interpreter's forward pass.
Training backpropagates ordinary next-token cross-entropy through the hook into the
hypernetwork. Adapters are not materialized through `peft.PeftModel` during training or
evaluation.

### T2L: task-description conditioning

- `src/adapterbench/t2p/lol_data.py` — vendored Lots-of-LoRAs/SNI task loading,
  decontaminated task validation, and definition stripping.
- `src/adapterbench/t2p/condition_encoder.py` — pooled task-description conditioner.
- `src/adapterbench/t2p/sft_trainer.py` — fixed-budget and checkpointed T2L SFT,
  gradient accumulation, warmup, and static-adapter training.
- `scripts/t2p_train_ddp.py` — data-parallel T2L training.
- `scripts/t2p_eval_heldout_sni.py` — held-out teacher-forced CE, the primary metric.
- `scripts/t2p_eval_heldout_sni_acc.py` — held-out generation accuracy.
- `scripts/reproduce/task_t2l_lora.sh` — canonical one-seed LoRA reproduction.

Training tasks are vendored under `data/t2l/`. The loader refuses tasks outside T2L's
479-task decontaminated training split and excludes all held-out validation tasks.

### D2L: document conditioning

- `src/adapterbench/t2p/document_conditioning.py` —
  `capture_early_exit_representation` plus `EarlyExitPerceiverConditioner`.
- `src/adapterbench/t2p/niah_data.py` — deterministic needle/haystack examples,
  realistic-prose distractors, packing guards, and evaluation examples.
- `src/adapterbench/t2p/document_sft_trainer.py` — restart-safe document-conditioned
  training with atomic model/optimizer/scheduler checkpoints.
- `src/adapterbench/t2p/live_evaluator.py` —
  `DocumentHypernetworkDownstreamEvaluator`, including the context-swap control.
- `scripts/d2p_niah_aggregate.py` — multi-seed and length-generalization aggregation.
- `scripts/reproduce/document_niah_lora.sh` — canonical D2L LoRA reproduction.

Each document is packed into one interpreter context. The implementation deliberately
does not reproduce Doc-to-LoRA's multi-chunk `combine_lora`; the one-pass constraint is
enforced explicitly.

## Why the settings are genuinely adaptive

### T2L

Standard task-conditioned SFT often gives the frozen interpreter the same task definition
that conditions the hypernetwork. In that design, the adapter is redundant.

AdapterBench strips the definition from the interpreter input. The task is available only
through:

`task description → conditioner → hypernetwork → generated adapter`

The primary comparison is against a directly optimized static adapter of the same shape
trained on the same multi-task data. `matched − static < 0` CE therefore isolates
task-specific conditioning from generic adapter help.

### D2L

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
3. **Graded difficulty.** T2L reports task-level behavior and D2L reports context-length
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

## T2L result

The rigorous T2L baseline trains the hypernetwork and same-shape static reference on the
same 479-task corpus, strips definitions from interpreter inputs, and evaluates on all 21
held-out SNI tasks.

| seed | matched − static CE | matched beats static | accuracy matched − static |
|---:|---:|---:|---:|
| 777 | approximately −0.9 nats | 20/21 tasks | positive |
| 2 | approximately −0.7 nats | 20/21 tasks | positive |
| 3 | −0.544 nats | 19/21 tasks | +0.0248 |
| aggregate | **−0.723 ± 0.162** | **59/63 task-seed pairs** | **+0.0317 ± 0.0060** |

CE is primary because it is non-saturating and matches the training objective. Greedy
exact-match accuracy is corroborating: it saturates on easy tasks and can understate
adapter differences.

Result paths:

`results/repro/t2l_base_diag/gemma2b_stripdef_hyper/s{777,2,3}/`

with:

- `heldout_sni_ce_full21.jsonl`
- `heldout_sni_acc.jsonl`

## D2L result

The shipped setting uses realistic Wikipedia-prose haystacks, a topic-free four-digit
needle, Qwen3-0.6B, early-exit document features, a perceiver conditioner, and a generated
rank-8 LoRA on the selected projection.

Across five seeds at the 256-token training length:

- matched exact-match: `0.887 ± 0.143`;
- context-swap exact-match: `0.000`;
- matched minus control: `+0.887 ± 0.143`.

The length-generalization crossover is 4096 tokens, 16× the training length.

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

Reproduce one T2L seed:

```bash
bash scripts/reproduce/task_t2l_lora.sh 777 0,1,2,3 4,5,6,7
```

Inspect T2L aggregate rows:

```bash
tail -1 results/repro/t2l_base_diag/gemma2b_stripdef_hyper/s777/heldout_sni_ce_full21.jsonl
tail -1 results/repro/t2l_base_diag/gemma2b_stripdef_hyper/s777/heldout_sni_acc.jsonl
```

Run the D2L realistic-NIAH baseline:

```bash
bash scripts/reproduce/document_niah_lora.sh cuda:0
```

Run the D2L scale locator:

```bash
bash scripts/d2l_scale_sweep.sh
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

### T2L

- Definition stripping is load-bearing. If the interpreter sees the task definition,
  conditioning becomes redundant.
- The static reference must use the same codec shape, data, hook sites, and training loss.
- Use the vendored 479-task decontaminated split. Do not silently substitute a generic
  SuperNI split.
- The held-out metadata is local; online model metadata checks can trigger Hugging Face
  rate limits even when all model/data files are cached.
- CE and accuracy answer different questions. Keep CE primary and accuracy corroborating.

### D2L

- Early-exit features plus the perceiver conditioner are load-bearing. The removed
  full-depth conditioner did not learn retrieval.
- Use the realistic-prose haystack for the canonical row. Repeated synthetic noise is a
  useful diagnostic but an easier task.
- The context-swap control must remain exactly zero or near chance.
- Ensure the full document fits in one interpreter pass; do not silently truncate.
- Strong L1 regularization can collapse the generated LoRA to zero. The upstream-style
  coefficient `1.5` failed; `0` or `0.1` reproduced retrieval.
- D2L's useful LoRA scale is much larger than ordinary PEFT defaults. Scale must be swept,
  not assumed.

## Stable extension interface

There is one leaderboard per active setting:

- `leaderboards/task_conditioned_t2l.md`
- `leaderboards/document_niah_d2l.md`

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
a failure. Additional *settings* (beyond T2L/D2L) are out of scope and would be considered
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
