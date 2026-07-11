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

Two settings, deliberately kept separate (different conditioning, objectives,
interpreters, and evaluators — their absolute scores are never pooled):

1. **Disk-artifact checkpoint reproduction** — load a released Text-to-LoRA checkpoint
   (Gemma-2-2B, Mistral-7B, Llama-3.1-8B, all from SakanaAI), generate a real LoRA for a
   benchmark task, and score it via vLLM (upstream's own inference backend) exactly the
   way the paper does. No training of our own — this is a pure reproduction check. A
   second project, Doc-to-LoRA (D2L, also SakanaAI), also has a Setting-1-shaped disk-
   artifact reproduction here: load a released D2L checkpoint and evaluate it on
   needle-in-a-haystack (NIAH, `ctx_magic_number_*`) tasks. Its conditioning is
   structurally different (cross-attention over a document's own per-layer activations,
   not a pooled task-description embedding) and its own eval API doesn't separate adapter
   generation from downstream scoring, so it gets its own backend module rather than
   reusing Text-to-LoRA's — see "Setting 1 (disk-artifact)" below.
2. **Live end-to-end SFT** — hook a hypernetwork's generated output directly into a real
   frozen `Qwen3-0.6B` interpreter's forward pass on real training examples, backprop
   ordinary next-token cross-entropy through the hook, and evaluate on real held-out
   benchmarks. The hypernetwork is trained entirely from scratch here (no released
   checkpoint) — this is where the six representations are actually compared head to head.

## Architecture

- `src/adapterbench/contracts.py` — `HypernetworkBackend.generate()`,
  `DownstreamEvaluator.evaluate()`, `TaskExample`/`AdapterArtifact`/`EvaluationResult`.
  Every setting implements these two interfaces; nothing else needs to know which setting
  produced a result.
- **Setting 1 (disk-artifact):**
  - `text_to_lora_backend.py::ReleasedTextToLoRABackend` — wraps a released `hypermod.pt`.
    Generation runs out-of-process under `upstream/text-to-lora/.venv` via
    `scripts/generate_t2l_adapter.py`, because `hyper_llm_modulator` pins a
    torch/transformers/peft stack incompatible with `adapterbench`'s own — **never import
    it directly from `adapterbench`.**
  - `vllm_downstream_evaluator.py::VLLMDownstreamEvaluator` — the evaluator this setting
    actually uses. Generates via vLLM (`vllm==0.5.4`, already installed in
    `upstream/text-to-lora/.venv` — no new dependency needed), out-of-process through
    `scripts/vllm_generate.py`, which loads the engine once and streams one result back
    per task family. LoRA-only (vLLM's LoRA serving support) — sufficient for this
    setting, since every setup here evaluates a released LoRA checkpoint.
  - `hf_downstream_evaluator.py::HFDownstreamEvaluator` — plain `transformers`+`peft`
    fallback, kept because it's the only evaluator that can score non-LoRA adapter
    formats if this setting ever needs one. Also the shared source of the
    generation+answer-extraction scoring primitives (`get_choice_accuracy`/
    `get_binary_accuracy`/`get_gsm8k_accuracy`) and prompt-building helpers
    (`build_prefill_by_family`/`render_prompt`/`load_faithful_tokenizer`) both evaluators
    use, so scoring never drifts between them.
  - `task_examples.py` — builds `TaskExample`s for arc_easy/arc_challenge/boolq/
    hellaswag/gsm8k from public HF datasets, using upstream's own prompt templates and
    3-shot in-context examples verbatim. Scoring is generation + answer-extraction, not
    log-likelihood-over-choices — the paper's numbers were produced by generating an
    answer and extracting a leading choice letter/digit (or a loose true/false keyword),
    not by scoring choice log-likelihoods.
  - `cli.py`'s `run` command — generate → evaluate → evaluate_frozen. `--evaluator
    {hf,vllm}` selects the evaluator (default `hf`; pass `vllm` for this setting).
  - `doc_to_lora_backend.py::ReleasedDocToLoRANIAHEvaluator` — wraps a released D2L
    `pytorch_model.bin`. Unlike `ReleasedTextToLoRABackend`/`VLLMDownstreamEvaluator`,
    this is a single class, not a `HypernetworkBackend`/`DownstreamEvaluator` pair:
    `ctx_to_lora.eval_utils.evaluate()` bakes per-document LoRA generation (never
    materialized as a portable adapter - it's merged straight into the running model's
    forward pass by a hand-rolled monkeypatch) and downstream scoring into one call, so
    there is no artifact to hand from a "generate" step to a separate "evaluate" step.
    Runs out-of-process under `upstream/doc-to-lora/.venv` via `scripts/run_d2l_eval.py`
    (one subprocess call per `evaluate`/`evaluate_frozen`, covering every requested
    `ctx_magic_number_<lo>_<hi>` dataset in that one call — D2L only loads the model once
    regardless, so there's no benefit to `VLLMDownstreamEvaluator`'s persistent-process
    streaming pattern here). `cli.py`'s `run-d2l-niah` command wires it up (a separate
    subcommand from `run`, not a variant of it — see that command's own comment for why).
- **Setting 2 (live SFT):**
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
  - **Document-conditioning variant (Doc-to-LoRA-style Setting 2, integrated
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

## Setting 1 results: reproducing Text-to-LoRA

Released checkpoints, n=1000/task (n=300 for gsm8k), scored via `VLLMDownstreamEvaluator`
(`adapterbench run --evaluator vllm`). Gemma uses `--use-icl` (matches the paper's Table 8,
which applies ICL to every method); Mistral and Llama don't (matches their own main
tables, where ICL is a separate baseline, not applied to the T2L(SFT) row).

<!-- RESULTS: results/vllm_repro/{gemma,mistral,llama}_{arc_easy,arc_challenge,boolq,hellaswag,gsm8k} -->

Accuracy / exact-match (%), frozen interpreter vs. released T2L LoRA (completed
2026-07-07; numbers verified against `results/vllm_repro/` on 2026-07-10):

| task | Gemma frozen | Gemma LoRA | Mistral frozen | Mistral LoRA | Llama frozen | Llama LoRA |
|---|---|---|---|---|---|---|
| arc_easy      | 88.9 | 91.0 | 76.6 | 88.8 | 92.7 | 94.7 |
| arc_challenge | 73.7 | 75.6 | 65.7 | 77.9 | 76.8 | 85.1 |
| boolq         | 82.1 | 81.6 | 75.1 | 84.2 | 82.0 | 85.5 |
| hellaswag     | 52.2 | 61.0 | 30.8 | 65.1 | 56.8 | 64.3 |
| gsm8k         | 65.3 | 58.3 | 43.3 | 46.3 | 84.0 | 81.3 |

The generated LoRA beats the frozen baseline on **12 of 15** model×task combinations
(the 3 exceptions: Gemma boolq, Gemma gsm8k, Llama gsm8k). Mistral shows the largest
LoRA-vs-frozen gap, driven mostly by its frozen baseline collapsing on hellaswag (30.8)
— Mistral-7B-Instruct's un-prompted continuations frequently ignore the answer format,
which the generated LoRA corrects. This reproduces T2L's published pattern (generated
LoRA ≳ frozen on the multiple-choice benchmarks). Report §5.1 renders the same table.

## Doc-to-LoRA NIAH: reproducing the from-scratch training recipe

Doc-to-LoRA's headline claim is near-perfect needle-in-a-haystack (NIAH) retrieval far
beyond both its 32–256-token training contexts and the base Gemma-2-2B's 8K window. None
of the four released checkpoints are NIAH-trained (every one's `train_ds_names` is QA
data — see Gotchas), so reproducing that claim requires running the upstream NIAH
training recipe from scratch.

**Scope note (2026-07-10):** the released QA checkpoints are out of scope for the
doc-conditioned work. The earlier generalization test (scoring the QA `gemma_demo`
checkpoint on NIAH — it retrieved at 1–2K but collapsed to 0 by 7–8K) has been dropped;
doc-conditioned NIAH is now purely (a) this from-scratch recipe reproduction and (b) our
own six-codec `d2p-sft-pilot` framework (Setting 2, below).

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

## Setting 2 results: comparing representations under live SFT

Qwen3-0.6B, hypernetwork trained entirely from scratch, full 479-task decontaminated
corpus (`--all-decontam-tasks`, 20 examples/task ≈ 9,580 examples), matching upstream's own
recipe (`--grad-accum-steps 64 --warmup-frac 0.1 --learning-rate 1e-5`, 380 optimizer
steps ≈ 10 epochs), 3 seeds per adapter (777/778/779), scored against held-out
boolq/hellaswag (n=60/family).

<!-- RESULTS: results/t2p_sft_full_rerun_{lora_freeze,ia3_lokr,fourierft_steering}/results.jsonl -->

Held-out accuracy (%), mean ± std across 3 seeds (777/778/779). **Re-run 2026-07-10
post-`eval()` fix** (dropout disabled during scoring, gotcha #14):

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

## Setting 2 results: document-conditioning variant (NIAH) — diagnosis + D2L integration plan

**STATUS (2026-07-11): the from-scratch six-codec D2P framework does NOT yet learn NIAH
retrieval. A full diagnostic pass established why; an upstream-parity integration is in
progress. Read this whole section before touching the D2P path — it supersedes the earlier
"generalization wall" framing.**

### What was tried and ruled out (all on Qwen3-0.6B unless noted)

Scaled up from the 2026-07-07 smoke across many configs. Held-out NIAH retrieval stayed at
**0** in every one:

| variation | held-out | rules out |
|---|---|---|
| conditioner: single-vector bottleneck (default) | 0 | — |
| conditioner: 8× capacity (32 latents / 4 blocks) | 0 | "make the conditioner bigger" |
| conditioner: per-slot Perceiver-IO (no bottleneck) | 0 | the bottleneck hypothesis |
| conditioner: faithful per-layer (`layer_to_layer`) | 0 | conditioner architecture generally |
| data scale: 3K → 60K → 120K docs | 0 (loss plateau ~1.54) | "just needs more data" in this range |
| Gemma-2-2B interpreter | 0/8 overfit — won't even converge | model swap as a quick fix (integration bug) |

The **overfit control always passes on Qwen** (memorizes 8 docs), so the mechanism is wired
correctly — it is *generalization* to held-out documents that never emerges.

### THE key finding: CE loss is NOT a retrieval signal

Every "loss decreasing" observation above says nothing about retrieval. Direct proof: an
upstream-parity mechanism check (scale 45.25, down_proj, 8000 steps, lr 4e-5) drove training
loss to **0.000 while retrieving 0/8 on the same documents**. The answer is ~5 tokens
(space + 4 digits + eos); teacher-forced CE is dominated by the easy continuation tokens, so
it rounds to ~0 while the one hard token — the first digit, which requires the adapter to
carry document identity — is never learned. **Never trust loss for NIAH; use exact-digit
generation + the context-swap control (see below).**

### Why upstream works and ours did not — verified recipe mismatches

AdapterBench's D2P was built as its own unified-codec framework, NOT a faithful D2L port.
Against upstream's actual NIAH recipe (`upstream/doc-to-lora/scripts/niah/1-train.sh` + its
config), load-bearing mismatches, each verified against the code:

| knob | AdapterBench default | upstream NIAH |
|---|---|---|
| LoRA scale | `alpha/sqrt(r)` = 5.66 | `lora_alpha = 2·r^1.5` = **45.25**, applied DIRECTLY (`lora_forward` uses `scaling=lora_alpha`; model_loading.py:171, lora_layer.py:520) — ~8× larger |
| hook site | `q_proj,v_proj` | `down_proj` only |
| learning rate | 1e-3 | 4e-5 (the big scale needs a low lr for stability) |
| doc encoder | per-layer / all-layer activations | **early-exit @ layer L//4**, single representation (ctx_encoder.py `EarlyExit`) |
| Perceiver | 2 blocks / 4 latent queries | **8 blocks / 208 latent queries**, `num_self_attn_per_block=0` |
| data | topic-needle, raw tok (`add_special_tokens=False`) | generic needle, context as a chat user message |
| regime | batch 4, no packing, +L2 penalty | batch 1 × grad_accum 16, sequence packing, per-context loss, `gen_lora_l1_reg_coef=1.5`* |

*Our own Setting-1 reproduction (above) found L1=1.5 **collapses**; use ~0. The two most
load-bearing mismatches are the 8×-too-small scale and the wrong hook site: a too-small
adapter on the wrong modules cannot override the frozen model to emit an unseen needle.

### Already integrated this session (2026-07-11, tests green)
- `DocumentHypernetworkDownstreamEvaluator`: **context-swap control** — each query is also
  scored against the WRONG document (the next example's); genuine doc-dependent retrieval must
  sit near chance there. Reported as `accuracy_ctxswap` beside exact-match `accuracy`. The
  matched-minus-swapped gap is the real signal.
- `LoRACodec` / `make_codec`: optional `scaling` / `lora_scaling` override so the framework can
  express upstream's 45.25 (was hardcoded to 5.66).

### Remaining integration — start fresh here ("make NIAH work at all", then vary the codec)
The six-codec comparison is meaningless until ONE config actually retrieves. So reproduce a
working recipe inside the framework first, then vary only the codec:
1. **Generation path**: add an early-exit context encoder (interpreter → layer L//4, single
   representation) + a larger Perceiver-IO (208 latents, 8 blocks) emitting per-(layer,module,
   rank) output queries → codec params, replacing the small per-layer conditioner + trunk/head.
   Keep the codec seam so all six codecs still plug in.
2. **Parity config path** (new CLI flags, e.g. a `--d2l-parity` preset): `down_proj` hook,
   `lora_scaling = 2·r^1.5`, lr 4e-5, generic-needle + chat-tokenized NIAH data, reg ~0.
3. **Full-scale run — MUST be restart-safe/checkpointed** (session teardowns repeatedly killed
   multi-hour runs this session; checkpoint model+optimizer every eval and resume). Train at
   upstream's regime; per eval, log exact-digit `accuracy` AND `accuracy_ctxswap`. Success =
   matched retrieval lifts off 0 while the context-swap control stays near chance.
4. Only then: the six-codec comparison under document conditioning.

The full parity recipe is specified in the mismatch table above (self-contained — rebuild from
it). A validating prototype was run this session but lived in a scratch dir (not committed).

## How to run

```bash
# Setting 1: reproduce a released checkpoint via vLLM
uv run adapterbench run \
  --setup text_to_peft_mistral7b_reconstruction_pilot \
  --adapter lora_r8_t2l \
  --checkpoint upstream/text-to-lora/trained_t2l/mistral_7b_t2l/hypermod.pt \
  --chat-template upstream/text-to-lora/chat_templates/mistralai/Mistral-7B-Instruct-v0.2/chat_template.jinja \
  --tasks boolq --limit 1000 --evaluator vllm --output results/vllm_repro/mistral_boolq

# Doc-to-LoRA NIAH: evaluate a FROM-SCRATCH NIAH checkpoint on the length bins.
# (Released QA checkpoints are out of scope - see the from-scratch reproduction section.)
uv run adapterbench run-d2l-niah \
  --setup doc_to_peft_gemma2b_reconstruction \
  --checkpoint <from-scratch NIAH run>/checkpoint-1482/pytorch_model.bin \
  --datasets ctx_magic_number_1024_2048,ctx_magic_number_7168_8192 \
  --limit 30 --split test --output results/d2l_niah_eval

# Setting 2: full-corpus live SFT, all six adapters, three seeds
uv run adapterbench t2p-sft-pilot \
  --all-decontam-tasks --adapters lora,freeze_a_lora,ia3,lokr,fourierft,activation_steering \
  --seeds 777,778,779 --steps 380 --grad-accum-steps 64 --warmup-frac 0.1 \
  --learning-rate 1e-5 --eval-limit 60 --output results/t2p_sft_full

# Step-budget sweep (single seed, several step budgets, one persistent optimizer)
uv run adapterbench t2p-sft-sweep \
  --adapters lora,freeze_a_lora,ia3,lokr,fourierft,activation_steering \
  --checkpoint-steps 100,200,400,800,1200 --eval-limit 40 --output results/t2p_sft_sweep

# Setting 2, document-conditioning variant: live SFT on synthetic NIAH documents
# (smoke scope - see "Setting 2 results" below for the real smoke-run numbers)
uv run adapterbench d2p-sft-pilot \
  --adapters lora --seeds 777 --steps 20 --grad-accum-steps 1 \
  --context-lengths 256 --num-train-documents 40 --eval-limit 10 \
  --device cuda:0 --output results/d2p_sft_smoke
```

```bash
uv run pytest -q   # 112 tests as of 2026-07-07 (per-layer document-conditioning fix)
```

`google/gemma-2-2b-it` and `meta-llama/Llama-3.1-8B-Instruct` are gated models — need
`hf auth login` with license acceptance on the same HF account (see SETUP.md).

## Gotchas (read before touching the pipeline again)

1. **Never import `hyper_llm_modulator` or `vllm` directly from `adapterbench`.** Both are
   pinned, incompatible stacks — always shell out to `upstream/text-to-lora/.venv` via a
   subprocess bridge (`text_to_lora_backend.py`, `scripts/generate_t2l_adapter.py`,
   `vllm_downstream_evaluator.py`, `scripts/vllm_generate.py`). Never `.resolve()` the
   upstream interpreter path itself (it's a symlink into the venv; resolving it collapses
   to the system interpreter and skips venv site-packages) — only resolve *data* paths.
2. **vLLM's per-LoRA tokenizer fallback (`get_lora_tokenizer`) only catches `OSError`,
   but a bare PEFT adapter directory doesn't reliably raise that type** — surfaces as a
   hard `ValueError` crash instead of the intended silent fallback. Fixed by saving a real
   copy of the base tokenizer into any adapter directory before vLLM ever calls
   `generate()` on it (`scripts/vllm_generate.py`) — every adapter here shares the
   interpreter's own tokenizer, so this is always correct, not a workaround.
3. **`use_rslora=True` adapters (r=8, alpha=16) need a backend that actually implements
   rslora scaling** (`alpha/sqrt(r)`, not the standard `alpha/r`) to match the paper's
   numbers — this is genuinely why matching upstream's own vLLM backend (not just "a"
   correct LoRA implementation) matters for reproduction; a naive-but-correct standard
   LoRA scaling would silently under-scale these adapters by ~2.8x.
4. **Neither the `DataLoader` shuffle nor `TextToPeftHypernetwork`'s own weight init/
   dropout was seeded**, despite every live-SFT command exposing `--seed` — the seed
   parameter only fed `initial_bias()` (gotcha #6 below). Fixed: `torch.manual_seed(seed)`
   once at the top of each command *and* again immediately before each adapter's
   `TextToPeftHypernetwork(...)` construction inside the per-adapter loop (so adapter N
   doesn't inherit RNG state from adapters trained before it), plus an explicit
   `generator=` on every `DataLoader`. This makes a *single* run reproducible given a fixed
   seed — it doesn't by itself establish whether results are stable *across* seeds.
5. **Qwen3's chat template defaults to "thinking" mode** — without
   `enable_thinking=False`, the model expects to emit its own `<think>...</think>` block
   before answering, so scoring a direct-answer continuation is badly out-of-distribution
   (confirmed: `frozen_interpreter` scored *below chance* on boolq without this fix).
   Applies to both training prompts (`lol_data.py::format_prompt_response`) and eval
   prompts (`live_evaluator.py::_prompt`) — harmless no-op for non-Qwen3 tokenizers.
6. **LoRA/LoKr have a dead zero-gradient saddle point at this hypernetwork's default
   all-zero head init.** Both split the head's output into two factors multiplied
   together — with both factors zero-initialized, the gradient w.r.t. *each* factor is
   proportional to the *other*, so both vanish simultaneously. Fixed at the codec level:
   `GeneratedUpdateCodec.initial_bias()` lets LoRA/LoKr override the head's bias with one
   factor's slice randomized (the product is still exactly zero at init, but gradient now
   reaches the zero factor immediately). Every other codec is linear in the generated
   output, so all-zero init is already fine for them.
7. **Moving this project's directory breaks every installed console script** (`.venv/bin/`
   entry points have an absolute-path shebang baked in at install time). Fix:
   `uv pip install --python .venv/bin/python --reinstall -e ".[dev]"` — regenerates the
   shims without recreating the venv. Renaming `pyproject.toml`'s `[project].name` can
   separately make `uv run` silently re-resolve `uv.lock` against the wrong interpreter on
   `PATH`; re-lock explicitly with `uv lock --python .venv/bin/python` if that happens.
8. **`google/gemma-2-2b-it` and `meta-llama/Llama-3.1-8B-Instruct` are gated** — `hf auth
   login` (the modern `hf` CLI) plus accepting each model's license on the same account.
9. **None of the four checkpoints on `SakanaAI/doc-to-lora` are NIAH-trained** — every
   `args.yaml`'s `train_ds_names` is QA data (self-gen `fw_qa_v2` + `pwc`/`squad`/`ropes`/
   `drop_compact`), never `ctx_magic_number`. There is no released NIAH-specific D2L
   checkpoint; `run-d2l-niah` against any of them is a generalization test, not a
   reproduction of a paper table for that exact checkpoint+dataset pairing. D2L's own
   from-scratch NIAH training recipe exists (`upstream/doc-to-lora/scripts/niah/*.sh`) but
   is Setting-2-shaped (train from scratch), not Setting 1.
10. **`ctx_to_lora`'s dataset loader never generates `ctx_magic_number_*` data lazily** —
    `data/raw_datasets/ctx_magic_number_<lo>_<hi>/{train,val,test}.jsonl` must already
    exist on disk (`uv run data/generate_ctx_magic_number.py`, see SETUP.md) before
    `run_eval`/`run-d2l-niah` can reference that dataset name; a missing bin fails with a
    `datasets`-library file-not-found error, not a clear "run the generator first" message.
11. **`ctx_to_lora`'s `Trainer` reports to `wandb` by default** — every checkpoint's
    `args.yaml` carries `report_to: [tensorboard, wandb]` from its original training run,
    and `eval_trainer_args` copies that field verbatim into the eval-time
    `Seq2SeqTrainingArguments`. Without `WANDB_MODE=disabled` (set inside
    `scripts/run_d2l_eval.py`, matching upstream's own `scripts/niah/2-eval.sh` launch
    convention), a plain eval run silently creates a real run under whatever wandb account
    happens to be logged in on the host - confirmed: the first standalone run of
    `run_d2l_eval.py` while developing this integration did exactly that before the env var
    was added.
12. **Document-conditioning live SFT does NOT replicate Doc-to-LoRA's own multi-chunk
    rank composition** (`combine_lora`, splitting a document across several context
    windows and composing each chunk's generated LoRA rank together when the document
    exceeds the interpreter's context length). Every document here is packed into one
    context window per example instead - a documented simplification, valid only as
    long as every configured `--context-lengths` bin fits under the interpreter's own
    `max_position_embeddings` (`niah_data.py::assert_context_fits_in_one_pass` is the
    safety net; `d2p-sft-pilot` now calls it automatically before building any dataset,
    fixed 2026-07-08 — previously it was implemented and tested but never wired in,
    so an over-long bin silently corrupted position encodings instead of erroring).
13. **The frozen interpreter itself runs in bf16 but every from-scratch hypernetwork
    parameter (including `DocumentPerceiverConditioner`'s) is plain float32** —
    `capture_document_activations` casts its captured activations to float32 before
    they reach the conditioner, or `nn.MultiheadAttention` raises a dtype-mismatch
    `RuntimeError` (confirmed directly while smoke-testing this integration against the
    real bf16 Qwen3-0.6B interpreter - see that function's docstring).
14. **Every live-SFT pilot/sweep command must call `hypernetwork.eval()` after training
    and before scoring** — `t2p-sft-pilot`, `d2p-sft-pilot`, and `t2p-sft-sweep` all
    trained a hypernetwork and then handed it straight to an evaluator without this,
    leaving the trunk's/conditioner's `nn.Dropout(0.05)` layers active during held-out
    scoring and adding non-determinism on top of genuine seed variance. Fixed 2026-07-08
    in all three commands. Any new pilot/sweep command needs the same call.
15. **`TextToPeftHypernetwork.generate_per_layer` used to recompute the shared trunk/heads
    once per layer** (28x for Qwen3-0.6B's 28 layers) to enable a per-layer-immediate-
    backward memory optimization that neither real caller (`document_sft_trainer.py`,
    `live_evaluator.py`) actually exercises — both defer or skip backward entirely, so the
    memory saving never materialized. Fixed 2026-07-08: per-layer work is now limited to
    the conditioner call itself; trunk/heads run once, batched across layers, same as
    `forward()`. `forward_layer` (single-layer, still useful for a future caller that does
    do per-layer backward) is unchanged.
16. **NIAH training loss is NOT a retrieval signal — never gate on it.** Response CE can be
    driven to ~0 while exact-digit retrieval is 0/N (proven 2026-07-11: parity mechanism
    check hit loss 0.000, retrieval 0/8 on trained docs). The answer is ~5 tokens and CE is
    dominated by the easy teacher-forced continuation; the one hard token (first digit,
    needing the adapter to carry doc identity) is never learned. Always evaluate with
    exact-digit generation **and** the context-swap control (`accuracy_ctxswap` in
    `DocumentHypernetworkDownstreamEvaluator`, added 2026-07-11) — genuine retrieval means
    matched `accuracy` high AND `accuracy_ctxswap` near chance.
17. **`LoRACodec` scale defaults to `alpha/sqrt(r)`=5.66, but D2L's NIAH recipe applies
    `2·r^1.5`=45.25 directly** (~8×). Pass `make_codec(..., lora_scaling=2*r**1.5)` (added
    2026-07-11) for D2L parity; the small default scale cannot override the frozen model to
    emit an unseen needle. See the D2P diagnosis section for the full mismatch table.

## Roadmap

- **Multi-seed live-SFT comparison at full 479-task scale** — in progress (see Results
  above); the direct next question once it lands is whether the pattern replicates a
  *second* time, not just across these 3 seeds but across independent reruns.
- Description-variant robustness (the released checkpoints' `args.yaml` carries 3
  paraphrased descriptions per benchmark task; only variant 0 is used anywhere so far).
- **D2P: integrate a working D2L-parity recipe, then compare the six codecs** — the active
  D2P workstream. The from-scratch six-codec framework does not yet learn NIAH; the full
  diagnosis, the verified upstream-recipe mismatch table, what's already integrated
  (context-swap diagnostic, `lora_scaling` override), and the remaining integration steps
  (early-exit + 208/8 per-slot generation path, `--d2l-parity` config, restart-safe
  full-scale run, then the six-codec comparison) are all in the **"Setting 2 results:
  document-conditioning variant (NIAH)"** section above. Start there.

## Reference: prior art

- Text-to-LoRA — arXiv:2506.06105 (the system Setting 1 reproduces and Setting 2
  generalizes beyond LoRA).
- Program-as-Weights — arXiv:2607.02512 (LoRA vs. prefix-tuning as hypernetwork targets;
  found LoRA ahead, on its own compiler/interpreter setting).
- Doc-to-LoRA — arXiv:2602.15902 (Setting 1 disk-artifact reproduction integrated on
  2026-07-07; document-conditioned, NIAH-evaluated, structurally distinct from
  Text-to-LoRA's task-description conditioning). Its document-conditioning mechanism
  (cross-attention over a frozen interpreter's own per-layer activations) was also
  ported into Setting 2 the same day (`d2p-sft-pilot`), so all six codecs can now be
  compared head-to-head under both conditioning mechanisms, not just Text-to-LoRA's.
- HyperTuning (Phang et al.) — arXiv:2402.16817 (the one directly-comparable prior result
  that disagrees with LoRA-over-steering-tokens, on a different task distribution — the
  open question this project exists to help answer).
- LoRA, FourierFT (arXiv:2405.03003), KronA (arXiv:2212.10650), Compacter
  (arXiv:2106.04647) — the representations under comparison.
