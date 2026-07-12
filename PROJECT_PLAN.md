# AdapterBench: Project Plan

## What this is

AdapterBench answers one question: **when a hypernetwork generates a parameter-efficient
adapter instead of an optimizer fitting one directly, does the adapter's *shape* matter?**
Text-to-LoRA, Program-as-Weights, and Doc-to-LoRA all generate LoRA specifically, without
testing it against simpler alternatives in this generation setting. AdapterBench holds the
hypernetwork, training procedure, and evaluation protocol fixed and varies only the
generated representation — LoRA, FreezeALoRA, LoKr, FourierFT, IA3, and activation
steering all plug into the same `codec` (output structure) + `hook site` (attachment
point) seam, so a new adapter needs no adapter-specific plumbing.

The setting is **live end-to-end SFT** — hook a hypernetwork's generated output directly
into a real frozen `Qwen3-0.6B` interpreter's forward pass on real training examples,
backprop ordinary next-token cross-entropy through the hook, and evaluate on real held-out
benchmarks. The hypernetwork is trained entirely from scratch (no released checkpoint) —
this is where the six representations are actually compared head to head.

## Architecture

- `src/adapterbench/contracts.py` — `HypernetworkBackend.generate()`,
  `DownstreamEvaluator.evaluate()`, `TaskExample`/`AdapterArtifact`/`EvaluationResult`.
  Every setting implements these two interfaces; nothing else needs to know which setting
  produced a result.
- `task_examples.py` — builds `TaskExample`s for arc_easy/arc_challenge/boolq/
  hellaswag/gsm8k from public HF datasets, using upstream's own prompt templates and
  3-shot in-context examples verbatim. Scoring is generation + answer-extraction, not
  log-likelihood-over-choices — an answer is generated and a leading choice letter/digit
  (or a loose true/false keyword) extracted, rather than choice log-likelihoods scored.
- **Live SFT:**
  - `t2p/codecs.py` — adapter-agnostic differentiable codecs (`GeneratedUpdateCodec`
    base), each hookable at either a named linear submodule (LoRA, FreezeALoRA, LoKr,
    FourierFT) or a whole decoder layer's residual stream (IA3, activation steering).
  - `t2p/hypernetwork.py::TextToPeftHypernetwork` — the shared hypernetwork shell;
    `apply()`'s hook closure handles both bare-tensor and tuple decoder outputs.
  - `t2p/lol_data.py` — Lots-of-LoRAs/SNI training data. `load_decontaminated_train_task_ids`
    reads T2L's own 479-task training split from `hyper_lora_decontam_lol_tasks.yaml`;
    `validate_training_tasks` refuses any task outside it (never one of T2L's
    contamination-removed or held-out-validation tasks).
  - `t2p/sft_trainer.py` — the training loop. `train_downstream_hypernetwork` (fixed step
    budget) and `train_with_checkpoints` (several step budgets under one persistent
    optimizer, for step-budget sweeps) both support `grad_accum_steps` and `warmup_steps`
    — matching upstream's own recipe (`batch_size=4, grad_accum_steps=64` → effective
    batch 256, `lr=1e-5`, `warmup_frac=0.1`, 10 epochs over the full 479-task corpus).
  - `t2p/live_evaluator.py::HypernetworkDownstreamEvaluator` — activates a generated
    adapter via `hypernetwork.apply(...)` (the same mechanism training uses) rather than
    `peft.PeftModel.load_adapter`, so hook-based adapters that were never materialized as
    PEFT adapters (activation steering, or anything trained here) can be scored at all.
  - `cli.py`'s `t2p-sft` (single adapter), `t2p-sft-pilot` (train + compare several
    adapters, optionally on the full 479-task split via `--all-decontam-tasks`, optionally
    across several seeds via `--seeds`), and `t2p-sft-sweep` (checkpointed step-budget
    sweep under one persistent optimizer, to tell "still converging" apart from "already
    past the point where held-out generalization peaks").
  - **Document-conditioning variant (Doc-to-LoRA-style, integrated
    2026-07-07):** same live-SFT loop, same six codecs, same
    `TextToPeftHypernetwork.apply()` hook mechanism - only the conditioning input
    changes, from a pooled task-description embedding to a frozen interpreter's own
    per-layer token activations on a synthetic needle-in-a-haystack (NIAH) document.
    - `t2p/hypernetwork.py::PooledVectorConditioner` - the pre-existing pooled-vector
      behavior, extracted verbatim into a pluggable `conditioner` module
      (`TextToPeftHypernetwork(..., conditioner=None)` still builds this by default,
      so every pre-existing call site is byte-for-byte unchanged).
    - `t2p/document_conditioning.py::DocumentPerceiverConditioner` - a hand-rolled
      (no HF VLM-specific Perceiver import) cross-attention stack: one learned latent
      query per decoder layer, cross-attending onto that layer's own document-token
      activations (`capture_document_activations`, a `@torch.no_grad()` forward pass
      through the frozen interpreter with `output_hidden_states=True`). Layer-index-
      aware in a way `PooledVectorConditioner` structurally cannot be - see that
      class's docstring for the `forward` (cross-layer-pooled summary, to keep
      `TextToPeftHypernetwork.forward`'s single-task-vector broadcast contract intact)
      vs. `forward_layer` (genuinely per-layer) design split. Both document-conditioned
      call sites (`document_sft_trainer.py::compute_doc_sft_loss`,
      `live_evaluator.py::DocumentHypernetworkDownstreamEvaluator`) use
      `TextToPeftHypernetwork.generate_per_layer` (fixed 2026-07-07), which loops
      `forward_layer` across every layer instead of the cross-layer-pooled-broadcast
      `forward` - so training/eval both condition each layer's generated adapter on
      that layer's own cross-attention result, at the cost of `num_layers` trunk
      forward passes instead of 1 per call.
    - `t2p/niah_data.py` - synthetic haystack/needle generation (`make_niah_example`,
      cycled filler sentences + one `"The special magic number for {topic} is
      {digits}."` needle at a controlled depth), `DocSFTDataset`/`doc_collate_fn`/
      `DocSFTBatch` (parallel to `lol_data.py`'s `LolSFTDataset`/`lol_collate_fn`/
      `sft_trainer.py`'s `SFTBatch`), and `build_niah_eval_examples` for held-out
      scoring. **Documented simplification:** every document is packed into one
      context window per example - this does NOT replicate Doc-to-LoRA's own
      multi-chunk rank composition (`combine_lora`, splitting a LoRA's rank across
      chunks when a document exceeds the interpreter's context window); see
      `assert_context_fits_in_one_pass`'s docstring. Valid here because Qwen3-0.6B's
      `max_position_embeddings` (40960) comfortably exceeds every configured length bin
      (default up to 2048).
    - `t2p/document_sft_trainer.py::compute_doc_sft_loss`/`doc_train_step`/
      `train_doc_downstream_hypernetwork` - a document-conditioned sibling of
      `sft_trainer.py`'s training loop (reuses that module's `masked_cross_entropy`/
      `_linear_warmup_then_constant` rather than duplicating them, but reimplements the
      grad-accumulation/step-loop shape itself, since `sft_trainer.py`'s own loop
      functions are hardcoded to `SFTBatch`/`compute_sft_loss` and were left unmodified
      per this integration's own scope decision - see that module's docstring).
    - `t2p/live_evaluator.py::DocumentHypernetworkDownstreamEvaluator` - same
      hook-then-score pattern as `HypernetworkDownstreamEvaluator`, but hooks a fresh
      adapter *per example* (every NIAH document is distinct, unlike one
      condition_embedding shared by a whole task family) and scores via exact 4-digit
      substring match rather than `get_choice_accuracy`/ROUGE-L.
    - `cli.py`'s `d2p-sft-pilot` - mirrors `t2p-sft-pilot`'s structure
      (`--adapters`/`--seeds`/`--steps`/`--grad-accum-steps`/`--warmup-frac`/
      `--eval-limit`/`--device`/`--output`), with `--context-lengths`/
      `--num-train-documents` replacing the task-selection flags.

## Doc-to-LoRA NIAH: reproducing the from-scratch training recipe

Doc-to-LoRA's headline claim is near-perfect needle-in-a-haystack (NIAH) retrieval far
beyond both its 32–256-token training contexts and the base Gemma-2-2B's 8K window. None
of the four released checkpoints are NIAH-trained (every one's `train_ds_names` is QA
data), so reproducing that claim requires running the upstream NIAH
training recipe from scratch.

**Scope note (2026-07-10):** the released QA checkpoints are out of scope for the
doc-conditioned work. The earlier generalization test (scoring the QA `gemma_demo`
checkpoint on NIAH — it retrieved at 1–2K but collapsed to 0 by 7–8K) has been dropped;
doc-conditioned NIAH is now purely (a) this from-scratch recipe reproduction and (b) our
own six-codec `d2p-sft-pilot` framework (below).

**Reproduction attempt of the NIAH training recipe itself (2026-07-08): FAILED to
converge, root cause diagnosed.** We ran `scripts/niah/1-train.sh` verbatim (same seed=1,
same data generator at the paper's 640K-sample scale, upstream's own pinned venv; only
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` added after a first attempt died to
allocator fragmentation at step 471/1482). Training completed all 1482 steps
(`train_outputs/runs/Jul08_06-45-39_*/checkpoint-1482/`), but the model never learned
retrieval: in-training `prefix_matchings` rose 0.002 → 0.15 mid-run, then collapsed to a
bit-identical 0.0536 for every subsequent eval, while `per_token_accs` sat at 0.82-0.83
(the fixed response template) and eval loss froze at 0.504. The mechanism is visible in
`gen_lora_l1_norm`: it fell from 0.015 at init to ~0.0004 almost immediately — the NIAH
recipe's `gen_lora_l1_reg_coef=1.5` (15x the main experiment's 0.1) crushed the
context-dependent LoRA to ~zero, leaving only the context-independent bias-LoRA, which can
express the template but not the needle. Downstream eval of this checkpoint scores
near-chance everywhere (0.0-0.17 across 1K-32K bins).

**L1 ablation (2026-07-09): confirmed — the coefficient alone separates collapse from
reproduction.** Two retrains changing only `gen_lora_l1_reg_coef` (everything else
byte-identical to `1-train.sh`, same seed=1, same data), evaluated identically
(n=30/bin, test split, 1024-token chunks; `results/d2l_niah_ours_l1_0p{1,0}/`):

| bin | shipped (L1=1.5) | L1=0.1 | L1=0.0 | paper (D2L) |
|---|---|---|---|---|
| 1024–2048 | 0.00 | 1.00 | 1.00 | ~1.0 |
| 7168–8192 | 0.13 | 0.97 | 1.00 | 1.0 |
| 16384–20480 | 0.17 | 0.50 | 0.97 | — |
| 28672–32768 | 0.07 | 0.10 | 0.67 | 0.997 |

Both ablations converged to `prefix_matchings=1.0` on the training distribution
(vs. the shipped recipe's frozen 0.0536), and the effect on downstream length
generalization is monotone in the coefficient: L1=0.0 is the closest reproduction —
perfect retrieval through 8K (matching the paper exactly, at 32-256-token training
contexts and a base model with an 8K window), 0.97 at 16-20K, degrading to 0.67 at
28-32K where the paper reports 0.997. So the paper's qualitative claim (near-perfect
retrieval far beyond training length and the base context window) reproduces at L1=0
or 0.1, the exact 28-32K figure does not, and the shipped config's L1=1.5 does not
train a working model at all in our environment. Whether upstream's own successful runs
used a different effective coefficient (or the collapse is seed/hardware-contingent) is
an open question for upstream.

## Results: comparing representations under live SFT

Qwen3-0.6B, hypernetwork trained entirely from scratch, full 479-task decontaminated
corpus (`--all-decontam-tasks`, 20 examples/task ≈ 9,580 examples), matching upstream's own
recipe (`--grad-accum-steps 64 --warmup-frac 0.1 --learning-rate 1e-5`, 380 optimizer
steps ≈ 10 epochs), 3 seeds per adapter (777/778/779), scored against held-out
boolq/hellaswag (n=60/family).

<!-- RESULTS: results/t2p_sft_full_rerun_{lora_freeze,ia3_lokr,fourierft_steering}/results.jsonl -->

Held-out accuracy (%), mean ± std across 3 seeds (777/778/779). **Re-run 2026-07-10
post-`eval()` fix** (dropout disabled during scoring, gotcha #7):

| adapter | boolq | hellaswag |
|---|---|---|
| frozen_interpreter | 75.0 | 23.3 |
| lora | 72.8 ± 11.1 | 39.4 ± 2.5 |
| freeze_a_lora | 74.4 ± 3.5 | 35.0 ± 8.7 |
| ia3 | 75.0 ± 0.0 | 37.2 ± 4.2 |
| lokr | 75.6 ± 3.5 | 38.3 ± 5.0 |
| fourierft | 30.6 ± 15.4 | 0.0 ± 0.0 |
| activation_steering | 70.0 ± 5.0 | 30.0 ± 1.7 |

The discriminative signal is hellaswag (boolq barely separates — most adapters sit near
frozen's 75.0). There the four low-rank weight adapters cluster well above frozen — lora
39.4, lokr 38.3, ia3 37.2, freeze_a_lora 35.0 vs frozen 23.3 — with **no single clear
winner** (lora nominally highest and lowest-variance, but within noise of lokr/ia3).
Activation steering beats frozen more modestly (30.0). **FourierFT is a genuine,
seed-independent catastrophic failure**: exactly 0.0 on hellaswag in all 3 seeds, wild
boolq swings (± 15.4). That failure — not a single-adapter champion — is the robust result.

**Correction vs. the pre-fix (2026-07-08) numbers:** disabling eval-time dropout raised
hellaswag for every low-rank adapter (lora 29.4→39.4, freeze 25.6→35.0, lokr 32.2→38.3,
ia3 35.6→37.2), which *changed the secondary conclusion*: the earlier "IA3 is strongest and
most stable" no longer holds — IA3 is now one of a cluster, with lora nominally ahead. The
headline (FourierFT collapses while the low-rank family all beat frozen) is unchanged and
stronger. The old caveat is resolved: these are the post-fix numbers.

**Why 3 seeds, and why the full corpus:** an earlier pass at this comparison (8-task
subset, single seed, 400 steps) found first that a contaminated task split made LoRA look
like it actively harmed generalization, then — after fixing that — found that a 3x larger
step budget made *every* adapter's held-out accuracy collapse while training loss kept
improving. Chasing that down turned up a real bug: none of the live-SFT commands actually
seeded the `DataLoader` shuffle or the hypernetwork's own weight init/dropout, so
nominally-identical runs landed on different trajectories — the "more training causes
collapse" framing was an artifact of that, not a property of training length. Both are
fixed now (see Gotchas below); this run is the first one to test whether results are
actually stable once seeding is real.

## Results: document-conditioning variant (NIAH) — SOLVED (D2L-parity recipe)

**STATUS (2026-07-11): the from-scratch six-codec D2P framework NOW learns held-out NIAH
retrieval.** A D2L-parity recipe was validated cheaply, then integrated into the framework
(codec seam intact). On Qwen3-0.6B, generic-needle NIAH, the generated LoRA reaches
**held-out exact-digit accuracy = 1.0 with the context-swap control = 0.0** — genuine
document-dependent retrieval, not memorization or a digit prior. This supersedes the earlier
"does NOT yet learn / diagnosis + integration plan" framing.

### The fix: what was load-bearing
The earlier diagnosis correctly ruled out conditioner capacity and data scale, and identified
the recipe mismatches. The combination that flips 0 -> retrieval:

- **Generic-needle, chat-tokenized data** (upstream `ctx_magic_number` format): one topic-free
  needle "The special magic number is NNNN." among repeated noise blocks joined by newlines, the
  document wrapped as a chat user message; topic-free query; answer = the bare 4 digits. (The old
  topic-tagged needle/query was replaced; `niah_data.py` now has `needle_style="generic"`.)
- **`down_proj` hook** (not q_proj/v_proj) and **LoRA scale = 2*r^1.5 = 45.25** applied directly
  (~8x rslora's default) — the two most load-bearing knobs.
- **lr 4e-5, reg 0, no dropout.**
- **Early-exit context encoder** (frozen interpreter's first `L//4 = 7` decoder layers, single
  `last_hidden_state`) + **Perceiver-IO** (208 latent queries, 8 cross-attention blocks) ->
  per-layer output-query decode -> the existing trunk/heads/codec seam.

Cheap validation (standalone probe, 512 docs, ctx~384): a clean phase transition — held-out
matched 0.00 (step 1200) -> 0.56 (step 1600) -> 1.00 (step 2000+), ctxswap pinned at 0.00, train
and held-out rising together (general copy algorithm, not memorization). Overfit control (32
docs) hit 1.0. Only 512 documents suffice at this context length.

### Integration + a real gotcha found while reproducing it in-framework
Integrated (tests green at 122): `capture_early_exit_representation` +
`EarlyExitPerceiverConditioner` in `document_conditioning.py` (both conditioners expose
`prepare_condition`, so trainer/evaluator are conditioner-agnostic — codec seam and
`generate_per_layer` unchanged); `needle_style="generic"` in `niah_data.py`;
`document_sft_trainer.py::train_doc_niah_checkpointed` (restart-safe: atomic
model+optimizer+scheduler+step+history checkpoint every eval, resume from it — validated on real
kills); `cli d2p-niah` (the parity command; `--adapters` for the comparison,
`--scale-weight-codecs` for the fairness control below).

**Gotcha (new, gotcha #11): freezing the batch grouping starves NIAH's phase transition.** The
first in-framework runs converged ~3-4x slower than the probe and looked broken. A chain of
controlled isolations established there was **no code bug** — the conditioner classes are
numerically bit-identical to the probe's, and the recipe matched. The culprit: the CLI
materialised the `DataLoader` into a fixed list of batches once, so every epoch cycled the *same*
8-doc groupings, whereas the probe reshuffled documents into fresh groupings each epoch. That
reduced gradient diversity enough to badly delay the sharp retrieval transition (which is itself
init/seed-sensitive, so the transition step already varies run-to-run, ~1600-2500+). Fix:
`train_doc_niah_checkpointed` takes the raw per-doc items and reforms batches from a
**document-level reshuffle every epoch** (deterministic per-epoch seed -> resume stays exact);
runs use a generous step budget so even a late transition completes.

### Six-codec comparison under document conditioning (Qwen3-0.6B, 512 docs, 6000 steps, ctx~384)
<!-- RESULTS: results/d2p_niah_{lora,freeze_a_lora,ia3,lokr,fourierft,activation_steering} -->

Held-out exact-digit accuracy / context-swap (n=32 docs), final step. Every codec uses the same
recipe/hook (`down_proj`, except activation_steering on the residual `block`); **only LoRA
received the load-bearing 45.25 scale — the others use their own default scaling** (see caveat):

| codec | held-out accuracy | ctxswap | train loss @ 6000 |
|---|---|---|---|
| frozen_interpreter | 0.00 | — | — |
| **lora** | **1.00** | 0.00 | 0.00 |
| freeze_a_lora | 0.06 | 0.00 | 0.10 |
| ia3 | 0.06 | 0.00 | 0.06 |
| lokr | 0.06 | 0.00 | 0.24 |
| fourierft | 0.22 | 0.00 | 0.21 |
| activation_steering | 0.00 | 0.00 | 0.23 |

LoRA generalizes perfectly; every other codec drives **training** loss low (0.06-0.24) but
does **not** generalize to held-out documents (accuracy stays low/noisy, ctxswap 0 everywhere) —
the "loss is not retrieval" pattern (gotcha #9) playing out per-codec: they memorize trained
docs without learning the general copy algorithm.

**Scale-matched control — both scale AND shape matter.** Because only LoRA got the ~8x-larger
45.25 scale the frozen model needs to be overridden (the load-bearing knob from the diagnosis),
the table above is scale-confounded. Re-running the two other LoRA-family weight codecs at the
SAME 45.25 scale (`--scale-weight-codecs`, `results/d2p_niah_{freeze_a_lora,lokr}_hi`) disentangles
it:

| codec | held-out acc @ default scale | held-out acc @ 45.25 | ctxswap |
|---|---|---|---|
| lora | (n/a — 45.25 is its recipe) | **1.00** | 0.00 |
| freeze_a_lora | 0.06 | **0.34** (peaked 0.44; train loss -> 0.001) | 0.00 |
| lokr | 0.06 | **0.06** (no lift) | 0.00 |

So **scale is load-bearing** (freeze_a_lora jumps 0.06 -> 0.34 purely from the scale match) **and
shape matters on top of it** (scale-matched, LoRA 1.0 ≫ freeze_a_lora 0.34 ≫ lokr 0.06 — fully
generating both low-rank factors beats generating only B over a fixed random A, which beats the
Kronecker factorization). freeze_a_lora fits *training* to loss ~0 but only partially generalizes,
a genuine shape-driven generalization gap vs LoRA. IA3 (multiplicative) / FourierFT (spectral) /
activation-steering (additive) have different scaling semantics and still need per-mechanism
tuning for a fully fair panel — the remaining open item. The robust headline stands: the
D2L-parity recipe makes the framework learn NIAH, LoRA retrieves perfectly, and among the
scale-matched low-rank family the generated adapter's *shape* measurably changes generalization.

## Length-generalization benchmark (step 1): the first shape comparison under the four invariants

**STATUS (2026-07-11): done.** This is the first genuinely benchmark-shaped result — it satisfies
all four invariants at once (control-gated behavioral metric, scale-matched, a graded difficulty
knob, multi-seed) and, in doing so, *overturned* the earlier single-seed scale-matched claim above.

`d2p-niah` now takes `--eval-context-lengths` (decouples train vs eval length;
`build_niah_eval_examples` already accepted a length list — the change is CLI glue + a
train/eval-union `assert_context_fits_in_one_pass` guard, tests at 123). Protocol: **train on
128+256-token documents, evaluate a length sweep 256 → 512 → 1K → 2K → 4K → 8K**, every codec
**scale-matched at 45.25** (`--scale-weight-codecs`, so this is a pure shape comparison), 512 docs,
lr 4e-5, n_latents 208 / num_blocks 8. LoRA on 6 seeds (777–782), freeze_a_lora/lokr on 3 (777–779);
LoRA at 6000 steps, freeze/lokr at 8000 (freeze transitions ~3000–4000 and needs the room).

<!-- RESULTS: results/d2p_lengthgen_{lora,freeze_a_lora,lokr}_s{777..782}; aggregate_lengthgen.py -->

Held-out exact-digit accuracy by eval length (mean ± std across seeds; **context-swap control = 0.00
at every bin, every seed** — all retrieval is genuinely document-dependent). Training length = 256,
so an eval bin at length L is an L/256× extrapolation:

| codec | 256 | 512 | 1K | 2K | 4K | 8K | transition rate | transition step | best reach | gen params |
|---|---|---|---|---|---|---|---|---|---|---|
| lora | 0.73±0.35 | 0.75±0.35 | 0.72±0.36 | 0.74±0.37 | 0.73±0.38 | 0.44±0.26 | **4/6** | ~1500–2500 | 8192 (**32×**) | 917K |
| freeze_a_lora | 0.83±0.10 | 0.84±0.14 | 0.86±0.10 | 0.84±0.07 | 0.85±0.12 | 0.56±0.04 | **3/3** | ~3000–4000 | 8192 (**32×**) | 229K |
| lokr | 0.41±0.28 | 0.41±0.22 | 0.36±0.14 | 0.39±0.18 | 0.35±0.18 | 0.12±0.05 | **1/3** | ~4500 | 4096 (16×) | 100K |

**The result is a vector, not a winner** — the shapes separate differently on each axis, which is the
whole point of reporting the vector (per "What each codec should be scored on"):

- **Reliability (transition rate):** freeze_a_lora (3/3) > lora (4/6) > lokr (1/3). Transitioning
  LoRA seeds are perfect (1.0 through 4096, 0.62 at 8192); the 2/6 that don't transition sit in the
  memorization basin (loss → ~0.02, retrieval ~0.2) and drag LoRA's *mean* below freeze_a_lora's.
- **Speed (transition step):** lora (~1500–2500) > freeze_a_lora (~3000–4000) > lokr (~4500). LoRA
  finds the copy algorithm fastest when it finds it at all.
- **Peak reach:** lora = freeze_a_lora = 8192 (32× training length) > lokr = 4096 (16×). At 8K,
  every codec degrades (the graded knob works — nothing saturates at 1.0 across the whole sweep).
- **Parameter efficiency:** freeze_a_lora reaches the *same* 32× peak as LoRA with **4× fewer
  generated parameters** (229K vs 917K — it generates only the B factor over a fixed random A), and
  a higher, tighter mean. On this task, generating both LoRA factors is *not* a good use of the
  hypernetwork's output budget: fixing A costs some sample efficiency (later transition) but nothing
  in ceiling, and it's markedly more reliable. lokr is smallest (100K) but weakest on every quality axis.

**Methodological headline — the protocol corrected an earlier wrong conclusion.** The single-seed
scale-matched control above reported freeze_a_lora at 0.34 ≪ LoRA 1.0 and called it a shape-driven
generalization gap. That gap was **largely a seed/budget artifact**: freeze_a_lora transitions late
(~3000–4000), so a single run at a shorter budget catches it mid-transition. Multi-seed + a generous
step budget (invariant #4 + adequate steps) shows freeze_a_lora is in fact *more reliable* than LoRA
and matches its peak reach. This is the strongest possible vindication of the invariants: the
single-number, single-seed comparison was not just imprecise, it was directionally wrong. Report only
numbers that survive the control, the scale match, **and** the seeds.

**Reproduce:** `aggregate_lengthgen.py` (log-based, works on in-progress or finished runs) emits this
table plus per-seed transition steps and crossover lengths; the grid launcher is `run_lengthgen_grid.sh`.


## How to run

```bash
# Full-corpus live SFT, all six adapters, three seeds
uv run adapterbench t2p-sft-pilot \
  --all-decontam-tasks --adapters lora,freeze_a_lora,ia3,lokr,fourierft,activation_steering \
  --seeds 777,778,779 --steps 380 --grad-accum-steps 64 --warmup-frac 0.1 \
  --learning-rate 1e-5 --eval-limit 60 --output results/t2p_sft_full

# Step-budget sweep (single seed, several step budgets, one persistent optimizer)
uv run adapterbench t2p-sft-sweep \
  --adapters lora,freeze_a_lora,ia3,lokr,fourierft,activation_steering \
  --checkpoint-steps 100,200,400,800,1200 --eval-limit 40 --output results/t2p_sft_sweep

# Document-conditioning variant: live SFT on synthetic NIAH documents
# (smoke scope - see the document-conditioning results section below for the real smoke-run numbers)
uv run adapterbench d2p-sft-pilot \
  --adapters lora --seeds 777 --steps 20 --grad-accum-steps 1 \
  --context-lengths 256 --num-train-documents 40 --eval-limit 10 \
  --device cuda:0 --output results/d2p_sft_smoke

# D2L-parity NIAH: the recipe that actually retrieves (LoRA -> held-out acc 1.0,
# ctxswap 0). Restart-safe (resume by re-running the same command); use .venv/bin for long runs.
# Pass several --adapters for the six-codec comparison; --scale-weight-codecs for the fair control.
.venv/bin/adapterbench d2p-niah \
  --adapters lora --needle-style generic --context-lengths 384 \
  --num-train-documents 512 --steps 6000 --eval-every 500 --learning-rate 4e-5 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 --device cuda:0 --output results/d2p_niah_lora

# Length-generalization benchmark (step 1): train short, eval a length sweep, scale-matched.
# One run per (codec, seed); restart-safe. run_lengthgen_grid.sh launches the whole grid; aggregate
# with aggregate_lengthgen.py. --eval-context-lengths decouples eval length from --context-lengths.
.venv/bin/adapterbench d2p-niah \
  --adapters freeze_a_lora --seed 777 --needle-style generic \
  --context-lengths 128,256 --eval-context-lengths 256,512,1024,2048,4096,8192 \
  --num-train-documents 512 --steps 8000 --eval-every 500 --learning-rate 4e-5 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 --scale-weight-codecs \
  --device cuda:0 --output results/d2p_lengthgen_freeze_a_lora_s777
```

```bash
uv run pytest -q   # 123 tests as of 2026-07-11 (+ eval-length decoupling contract for d2p-niah)
```

## Gotchas (read before touching the pipeline again)

1. **Neither the `DataLoader` shuffle nor `TextToPeftHypernetwork`'s own weight init/
   dropout was seeded**, despite every live-SFT command exposing `--seed` — the seed
   parameter only fed `initial_bias()` (gotcha #3 below). Fixed: `torch.manual_seed(seed)`
   once at the top of each command *and* again immediately before each adapter's
   `TextToPeftHypernetwork(...)` construction inside the per-adapter loop (so adapter N
   doesn't inherit RNG state from adapters trained before it), plus an explicit
   `generator=` on every `DataLoader`. This makes a *single* run reproducible given a fixed
   seed — it doesn't by itself establish whether results are stable *across* seeds.
2. **Qwen3's chat template defaults to "thinking" mode** — without
   `enable_thinking=False`, the model expects to emit its own `<think>...</think>` block
   before answering, so scoring a direct-answer continuation is badly out-of-distribution
   (confirmed: `frozen_interpreter` scored *below chance* on boolq without this fix).
   Applies to both training prompts (`lol_data.py::format_prompt_response`) and eval
   prompts (`live_evaluator.py::_prompt`) — harmless no-op for non-Qwen3 tokenizers.
3. **LoRA/LoKr have a dead zero-gradient saddle point at this hypernetwork's default
   all-zero head init.** Both split the head's output into two factors multiplied
   together — with both factors zero-initialized, the gradient w.r.t. *each* factor is
   proportional to the *other*, so both vanish simultaneously. Fixed at the codec level:
   `GeneratedUpdateCodec.initial_bias()` lets LoRA/LoKr override the head's bias with one
   factor's slice randomized (the product is still exactly zero at init, but gradient now
   reaches the zero factor immediately). Every other codec is linear in the generated
   output, so all-zero init is already fine for them.
4. **Moving this project's directory breaks every installed console script** (`.venv/bin/`
   entry points have an absolute-path shebang baked in at install time). Fix:
   `uv pip install --python .venv/bin/python --reinstall -e ".[dev]"` — regenerates the
   shims without recreating the venv. Renaming `pyproject.toml`'s `[project].name` can
   separately make `uv run` silently re-resolve `uv.lock` against the wrong interpreter on
   `PATH`; re-lock explicitly with `uv lock --python .venv/bin/python` if that happens.
5. **Document-conditioning live SFT does NOT replicate Doc-to-LoRA's own multi-chunk
    rank composition** (`combine_lora`, splitting a document across several context
    windows and composing each chunk's generated LoRA rank together when the document
    exceeds the interpreter's context length). Every document here is packed into one
    context window per example instead - a documented simplification, valid only as
    long as every configured `--context-lengths` bin fits under the interpreter's own
    `max_position_embeddings` (`niah_data.py::assert_context_fits_in_one_pass` is the
    safety net; `d2p-sft-pilot` now calls it automatically before building any dataset,
    fixed 2026-07-08 — previously it was implemented and tested but never wired in,
    so an over-long bin silently corrupted position encodings instead of erroring).
6. **The frozen interpreter itself runs in bf16 but every from-scratch hypernetwork
    parameter (including `DocumentPerceiverConditioner`'s) is plain float32** —
    `capture_document_activations` casts its captured activations to float32 before
    they reach the conditioner, or `nn.MultiheadAttention` raises a dtype-mismatch
    `RuntimeError` (confirmed directly while smoke-testing this integration against the
    real bf16 Qwen3-0.6B interpreter - see that function's docstring).
7. **Every live-SFT pilot/sweep command must call `hypernetwork.eval()` after training
    and before scoring** — `t2p-sft-pilot`, `d2p-sft-pilot`, and `t2p-sft-sweep` all
    trained a hypernetwork and then handed it straight to an evaluator without this,
    leaving the trunk's/conditioner's `nn.Dropout(0.05)` layers active during held-out
    scoring and adding non-determinism on top of genuine seed variance. Fixed 2026-07-08
    in all three commands. Any new pilot/sweep command needs the same call.
8. **`TextToPeftHypernetwork.generate_per_layer` used to recompute the shared trunk/heads
    once per layer** (28x for Qwen3-0.6B's 28 layers) to enable a per-layer-immediate-
    backward memory optimization that neither real caller (`document_sft_trainer.py`,
    `live_evaluator.py`) actually exercises — both defer or skip backward entirely, so the
    memory saving never materialized. Fixed 2026-07-08: per-layer work is now limited to
    the conditioner call itself; trunk/heads run once, batched across layers, same as
    `forward()`. `forward_layer` (single-layer, still useful for a future caller that does
    do per-layer backward) is unchanged.
9. **NIAH training loss is NOT a retrieval signal — never gate on it.** Response CE can be
    driven to ~0 while exact-digit retrieval is 0/N (proven 2026-07-11: parity mechanism
    check hit loss 0.000, retrieval 0/8 on trained docs). The answer is ~5 tokens and CE is
    dominated by the easy teacher-forced continuation; the one hard token (first digit,
    needing the adapter to carry doc identity) is never learned. Always evaluate with
    exact-digit generation **and** the context-swap control (`accuracy_ctxswap` in
    `DocumentHypernetworkDownstreamEvaluator`, added 2026-07-11) — genuine retrieval means
    matched `accuracy` high AND `accuracy_ctxswap` near chance.
10. **`LoRACodec` scale defaults to `alpha/sqrt(r)`=5.66, but D2L's NIAH recipe applies
    `2·r^1.5`=45.25 directly** (~8×). Pass `make_codec(..., lora_scaling=2*r**1.5)` (added
    2026-07-11) for D2L parity; the small default scale cannot override the frozen model to
    emit an unseen needle. See the D2P diagnosis section for the full mismatch table.
11. **Freezing the NIAH batch grouping starves the retrieval phase transition.** Materialising a
    `DataLoader` into a fixed list of batches once (then only reshuffling batch *order*) cycles the
    same document groupings every epoch; the from-scratch document-conditioned NIAH run converges
    ~3-4× slower that way and can look like it will never retrieve — even though the recipe is
    correct (confirmed by isolation: the conditioner classes are numerically bit-identical to the
    validated probe's). `train_doc_niah_checkpointed` reforms batches from a **document-level
    reshuffle every epoch** (deterministic per-epoch seed, so restart-safe resume stays exact) to
    avoid this. The transition step is also init/seed-sensitive (varies ~1600-2500+), so give NIAH
    runs a generous step budget rather than assuming a fixed transition point.

## Benchmark design: developing this into a real adapter benchmark

**Read this before adding tasks or "improving" numbers.** AdapterBench's job is to answer *does the
generated adapter's shape matter?* — so the benchmark's value is entirely in whether its
comparisons are **valid**, not in how many tasks it has. The dominant failure mode is producing
confident-but-meaningless leaderboards. This session hit two such traps directly, and both are
properties of the *protocol*, not the task:

- **Loss is not capability.** Response CE was driven to 0.000 with 0/N retrieval (gotcha #9).
  Anything that gates, early-stops, or ranks on train/val loss will produce a clean-looking table
  that measures nothing.
- **Scale dominates shape.** The same LoRA is 0.0 at scale 5.66 and 1.0 at 45.25; freeze_a_lora
  went 0.06 → 0.34 from a scale match alone. Comparing codecs at their *default* scales ranks
  scales, not shapes.

So the benchmark is defined by four protocol invariants. Any new task or setting MUST satisfy them.

### The four invariants (non-negotiable)

1. **Behavioral metric with a built-in control; headline = matched − control, never loss.**
   Every task needs a control that catches "cheating" (priors, memorization, format-only learning).
   The document/NIAH setting already has the gold-standard one: the **context-swap control**
   (generate the adapter from the *wrong* document, same query → must sit near chance;
   `DocumentHypernetworkDownstreamEvaluator.accuracy_ctxswap`). A finding only counts when matched
   accuracy is high AND the control is near chance. If a proposed task has no clean control, it is a
   *complement*, not the core.

2. **Scale is a swept axis, not a fixed choice.** Run a small **per-codec scale sweep and report
   best-of** (the "oracle-scale" comparison). Then "shape matters" means "even at its own best
   scale, shape X underperforms" — a claim that survives scrutiny. The confounded default-scale
   table (LoRA got 45.25, others their defaults) is exactly what NOT to publish as a shape result.
   `d2p-niah --scale-weight-codecs` was a first step (applies one scale to the LoRA family); the
   real need is a `--scale-sweep` that runs each codec over a small grid around its
   effective-magnitude and keeps the best.

3. **A graded difficulty knob so codecs actually spread.** A setting where everything saturates at
   1.0 (or all fail) teaches nothing. NIAH at a single length is nearly binary. **Length
   generalization is the ideal knob for the document setting** — train on short contexts, evaluate a
   length sweep (256 → 8K) — because it is continuous, cheap (Qwen3-0.6B's 40K window means no
   multi-chunk composition needed up to ~16K, so gotcha #5's `combine_lora` gap doesn't bite), and
   it is exactly where shapes should diverge (does a rank-8 update carry routing structure that
   *extrapolates*, or just memorize the training length?). Report a **curve per codec** plus a scalar
   summary (the eval/train length ratio at which retrieval crosses 0.5).

4. **Multi-seed, because the quantity being measured is stochastic.** The retrieval phase transition
   landed anywhere from step ~1600 to ~2500+ on *identical* recipes (gotcha #11). A single-seed
   number is partly luck. Use ≥3 seeds, report mean±std. The **transition step itself is a metric**
   (sample efficiency / trainability), not just noise to average away.

### What each codec should be scored on (a vector, not one number)
"Does shape matter" is multi-dimensional; collapsing to peak accuracy throws away the interesting
structure. Report per codec:
- **Peak held-out retrieval** (matched − control), best-of the scale sweep.
- **Length-extrapolation ratio** — eval/train length at which retrieval still holds (the D2L-style
  claim; the most discriminative axis).
- **Sample efficiency** — retrieval vs #training docs, or the phase-transition step.
- **Parameter efficiency** — retrieval per *generated* parameter. Codecs differ here by orders of
  magnitude (FourierFT's `n_frequency` vs LoRA's `rank·(dᵢₙ+dₒᵤₜ)` vs IA3's `dₒᵤₜ`), and this is
  directly the question "is this shape a good use of the hypernetwork's fixed output budget?".
  (The length-gen result already shows this axis biting: freeze_a_lora matches LoRA's 32× peak reach
  with 4× fewer generated parameters — see the length-generalization results section.)

### Per-codec autoresearch: generalizing the scale sweep (invariant #2, done right)
**DEFERRED below the image domain (2026-07-11) — spec retained for when it's picked up.**
Invariant #2 sweeps *one* hyperparameter (scale) per codec and reports best-of. But scale is not
special — it is simply the HP we caught being load-bearing first. Several HPs are neither the task
nor the adapter *shape* yet strongly move the result (lr, warmup, step budget, scale; the
length-gen run showed a codec's transition step spanning ~1500–4500). So the honest generalization is
a **per-codec autoresearch loop**: fix the shape, search its HPs to best-of, and only then compare.
This upgrades the claim from the weak form ("at one shared recipe, shapes differ" — a shape can lose
merely because the recipe suits LoRA) to the strong form ("even at its *own* tuned optimum, shape X
underperforms"). It is the right long-term backbone for the benchmark, but it only stays a *shape*
benchmark under a strict HP partition and three guardrails — get either wrong and the benchmark eats
itself.

**HP partition (decide per HP, empirically, before searching):**
1. **Shared substrate — identical across codecs, never tuned per-codec.** Task data, the
   conditioner/hypernetwork trunk (`n_latents`, `num_blocks`, `exit_layer`), eval protocol, and the
   control. This is what "fixed procedure" *means*; tuning the conditioner per codec stops the
   comparison being about the adapter.
2. **Free optimization HPs — the loop may tune these.** `lr`, `warmup`, `steps`, `scale`. Not task,
   not shape-identity, but empirically shape-sensitive.
3. **Shape-identity HPs — the trap; fix by definition or sweep only along the parameter-efficiency
   axis, never *maximize*.** `rank`, LoKr's factorization, FourierFT's `n_frequency`. A loop that
   freely maximizes these drives every shape toward "as dense as the budget allows" and "shape"
   dissolves.

**Three guardrails (each grounded in a failure this project already hit):**
- **Optimize `matched − control`, never loss.** Seeds reached loss ~0.02 with ~0.2 retrieval
  (memorization basin, gotcha #9). A loop pointed at loss tunes every codec into memorization. The
  objective is the behavioral, control-gated metric — discrete and flat-until-transition, a genuinely
  hard target.
- **Every config is multi-seed.** The transition is stochastic (LoRA 4/6 seeds, ~1500–2500; a single
  trial mostly measures seed luck). The inner objective must be transition-rate or best-of-k over ≥3
  seeds, which multiplies cost by k.
- **Equal search budget and search space per codec, both reported.** "Best-of-N trials over space S"
  makes N and S part of the result; an unequal budget/space smuggles the bias back in. Publish the
  tuned HPs — the tuned-HP table ("LoRA wants lr 4e-5 / scale 45; freeze_a_lora needs ~2× the steps;
  LoKr can't be tuned into reliable retrieval anywhere in S") is itself a richer artifact than a
  leaderboard.

**Cost and the pruning hazard.** Cost is `codecs × configs × seeds × steps` — a modest 12-config × 3-seed
loop is ~12× the current grid, feasible on Qwen3-0.6B (why it was chosen) but a week not a day of GPU.
ASHA/successive-halving helps because non-transitioning configs are flat at 0 through ~step 1500 and
`d2p-niah` is restart-safe/checkpointed — **but prune with care**: lokr_s777 first crossed 0.5 at
~step 4500, and two LoRA seeds transitioned only at ~2500–3000. Aggressive early-stopping would prune
the slow-but-real configs and falsely label a shape "incapable at any config." The late, stochastic
transition is exactly what makes naive HPO dangerous here.

**Staged rollout (keep every intermediate result publishable, never touch shape-identity):**
1. Finish the fixed-recipe, scale-matched length-gen result (**done** — see results section) as the
   honest baseline the loop must beat.
2. Implement the scale sweep (concrete step 2 below) *as* the minimal loop — one free HP, best-of,
   multi-seed — to shake out the objective/seed/pruning machinery.
3. Only then generalize the search to `{scale, lr, warmup, steps}` per codec with ASHA + equal budget
   on the frozen shared substrate, reporting the tuned HPs.

### The recommended core setting
Keep **both** conditioning modalities the framework already supports — a good adapter shape should
hold up on both, and they probe different things:
- **Document conditioning (NIAH), flagship discriminative axis = length generalization.** This is
  the strongest core because of its clean control and continuous difficulty knob. Run it under the
  scale-sweep + control + multi-seed protocol on Qwen3-0.6B (cheap enough to run the whole
  6-codec × lengths × seeds × scale grid; 512 docs / a few-thousand steps suffices).
- **Task-description conditioning (Text-to-LoRA-style), held-out tasks.** See the T2L notes below.

Optionally add **one semantic document task** (document QA — condition on a doc, answer a question
whose answer is in it; exact/span match) as a *complement* to NIAH, to check the shape ranking
isn't an artifact of the copy/retrieval nature of NIAH. D2L's own training data (`fw_qa_v2`,
`squad`, `ropes`, `drop_compact`) is a natural source. Caveat: QA's "swap the document" control is
fuzzier than NIAH's exact-digit one, so treat it as corroboration, not the rigorous core.

### Applying the same rigor to the Text-to-LoRA (task-description) setting
The `t2p-sft-pilot` setting (train a hypernetwork from scratch, condition on a task-description
embedding, score held-out boolq/hellaswag) already has multi-seed (777/778/779) and the eval-mode
fix (gotcha #7), but it is missing the other three invariants:
- **No control.** Add the analog of context-swap: score each held-out task with a **mismatched task
  description** (a different task's embedding). Genuine task-conditioning must collapse toward
  frozen/chance there; matched − mismatched is the real signal, exactly as for NIAH. Without it,
  "LoRA beats frozen on hellaswag" can't be distinguished from "the adapter learned a generic
  format/prior."
- **No scale sweep.** Same confound as the document setting — run the per-codec scale sweep here too.
- **Weak difficulty grading / saturation.** boolq barely separates (most adapters ≈ frozen's 75.0);
  hellaswag was the only discriminative task. Curate eval tasks where the frozen baseline leaves
  real headroom (drop near-saturated ones), and consider a difficulty knob analogous to length —
  e.g. number of in-context shots, or held-out *task-family* distance from the training split.
- FourierFT's seed-independent catastrophic 0.0 on hellaswag is a genuine result to keep, but re-check
  it under the scale sweep (it may be scale-starved rather than shape-broken, the same lesson as
  the document setting).

### Concrete first steps for a new agent (in order)
**Scope note (2026-07-11):** step 1 is done and the language document-conditioning line is considered
complete for the paper (see Roadmap). Steps 2–5 below are **deferred below the image domain** — they
remain the correct plan for deepening the *document* setting, but the next active phase is bringing the
benchmark to the image modality, not continuing here. Pick these up after (or alongside) the image work.
1. ~~Add `--eval-context-lengths` to `d2p-niah` ... run the low-rank family across a length curve at
   3 seeds, scale-matched.~~ **DONE (2026-07-11)** — see the "Length-generalization benchmark (step 1)"
   results section. Flag landed (`--eval-context-lengths`), low-rank family run (lora ×6 seeds,
   freeze_a_lora/lokr ×3), `aggregate_lengthgen.py` emits the per-codec vector. The result is a
   vector (reliability / speed / reach / parameter efficiency), and it overturned the earlier
   single-seed scale-matched claim (freeze_a_lora is *more reliable* than LoRA and matches its 32×
   reach at 4× fewer params — not the 0.34 ≪ 1.0 gap the single run reported).
2. Add a `--scale-sweep` (small per-codec grid, keep best) and make best-of-scale the reported
   number everywhere. **Build this as the minimal per-codec autoresearch loop** (one free HP,
   best-of, multi-seed) — see "Per-codec autoresearch" above; it is the machinery every later HP
   search reuses, and the guardrails (optimize matched−control, multi-seed inner eval, don't
   over-prune the late transition) must be right here first.
3. Add the **mismatched-description control** to `HypernetworkDownstreamEvaluator` and surface
   `accuracy_mismatched`, mirroring `accuracy_ctxswap`.
4. Add a metric-aggregation step that emits the per-codec vector (peak, length-ratio, sample
   efficiency, parameter efficiency) rather than only per-task accuracy rows. (`aggregate_lengthgen.py`
   is a first, NIAH-specific instance of this — generalize it.)
5. Only after 1–4 are solid: extend to IA3/FourierFT/activation-steering (each needs its own scale
   semantics tuned — multiplicative / spectral / additive — before it belongs in a fair panel), and
   optionally add the semantic QA complement. Then generalize step 2's single-HP loop to the full
   per-codec `{scale, lr, warmup, steps}` autoresearch search (ASHA, equal budget, frozen substrate).

Guiding principle throughout: a result is only worth reporting if it survives its control, its scale
sweep, and its seeds. Everything else is a diagnostic, not a benchmark number.

## Roadmap

**The language document-conditioning (Doc-to-LoRA) line is complete for the paper's current scope
(2026-07-11).** It comprises: the from-scratch NIAH reproduction (L1 ablation), the
six-codec D2P framework that learns held-out retrieval, and — the capstone — the
length-generalization shape benchmark under all four invariants (§ "Length-generalization benchmark
(step 1)"). That last is the first genuinely valid shape comparison: a per-codec vector (reliability /
speed / reach / parameter efficiency), and it corrected the earlier single-seed claim (freeze_a_lora is
*more reliable* than LoRA and matches its 32× reach at 4× fewer generated params — not the "0.34 ≪ 1.0"
gap a single run reported). No further document-setting runs are needed to have a publishable result.

**Next active phase: the image domain.** The framework's `codec` + `hook site` seam is modality-agnostic
by design (report.html §3.2 "Image Settings" / §3.3 "Latent-Space Planning" sketch the intent), but no
image-domain code exists yet — this is a build, not a tuning pass. The goal is to check whether the
adapter-shape findings from the language setting *transfer* to a visual generator (e.g. hypernetwork-
generated adapters on a frozen image/latent model, conditioned on an image or a task spec), reusing the
same four invariants: a behavioral metric with a built-in control, matched scale, a graded difficulty
knob, and multi-seed. A shape ranking that holds across modalities is a far stronger claim than one
measured on NIAH alone.

**Deferred (valuable, but explicitly below the image domain):**
- **Per-codec scale sweep + the autoresearch loop** — the generalization of invariant #2 (full spec +
  guardrails retained in "Per-codec autoresearch" under Benchmark design). Deferred by decision on
  2026-07-11: the fixed-recipe scale-matched length-gen result already stands as publishable, and the
  loop is a large compute/engineering investment better spent after the modality-transfer question.
- **Remaining document-setting codecs** (IA3 / FourierFT / activation-steering) — each needs its own
  scale semantics tuned before it joins the matched panel.
- **Mismatched-description control for the T2L task-description setting** (concrete step 3), and
  description-variant robustness (released checkpoints carry 3 paraphrases/task; only variant 0 used).
- **Multi-seed live-SFT at full 479-task scale, second independent replication** of the §5.2 pattern.

## Reference: prior art

- Text-to-LoRA — arXiv:2506.06105 (the system this benchmark generalizes beyond LoRA).
- Program-as-Weights — arXiv:2607.02512 (LoRA vs. prefix-tuning as hypernetwork targets;
  found LoRA ahead, on its own compiler/interpreter setting).
- Doc-to-LoRA — arXiv:2602.15902 (document-conditioned, NIAH-evaluated, structurally
  distinct from Text-to-LoRA's task-description conditioning). Its document-conditioning
  mechanism (cross-attention over a frozen interpreter's own per-layer activations) was
  ported into this benchmark on 2026-07-07 (`d2p-sft-pilot`), so all six codecs can now be
  compared head-to-head under both conditioning mechanisms, not just Text-to-LoRA's.
- HyperTuning (Phang et al.) — arXiv:2402.16817 (the one directly-comparable prior result
  that disagrees with LoRA-over-steering-tokens, on a different task distribution — the
  open question this project exists to help answer).
- LoRA, FourierFT (arXiv:2405.03003), KronA (arXiv:2212.10650), Compacter
  (arXiv:2106.04647) — the representations under comparison.
