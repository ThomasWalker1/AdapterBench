# AdapterBench: Project Plan

## What this is

AdapterBench answers one question: **when a hypernetwork generates a parameter-efficient
adapter instead of an optimizer fitting one directly, does the adapter's *shape* matter?**
Text-to-LoRA, Program-as-Weights, and Doc-to-LoRA all generate LoRA specifically, without
testing it against alternative representations in this generation setting. AdapterBench
holds the hypernetwork, training procedure, and evaluation protocol fixed and varies only
the generated representation, which plugs into a single `codec` (output structure) +
`hook site` (attachment point) seam — so a new adapter needs no adapter-specific plumbing.

The setting is **live end-to-end SFT** — hook a hypernetwork's generated output directly
into a real frozen `Qwen3-0.6B` interpreter's forward pass on real training examples,
backprop ordinary next-token cross-entropy through the hook, and evaluate on real held-out
benchmarks. The hypernetwork is trained entirely from scratch (no released checkpoint).

**Current status (2026-07-12).** Three evaluation settings — two language, one image — are
set up and validated end-to-end with a **LoRA baseline codec**:
- **T2L** — task-description conditioning (Text-to-LoRA-style), scored on held-out
  benchmark tasks.
- **D2L** — document conditioning (Doc-to-LoRA-style), scored on held-out
  needle-in-a-haystack (NIAH) retrieval, including length generalization.
- **I2P** — image-domain reward-tilting (HyperNoise-style): a from-scratch adapter on a
  frozen SD-Turbo modulates the initial noise to maximize a reward (ImageReward headline),
  scored on held-out prompts under all four invariants (see § "Image domain: I2P").

LoRA is the only codec on `main`. It exists to prove the *evaluation pipeline* works with a
single, well-understood representation; any pipeline failure is then attributable to the
plumbing, not the codec. **Additional adapter shapes are added one at a time** — a codec subclass
+ `make_codec` entry + manifest — each landing with its own row in the relevant setting's
leaderboard (see § "The benchmark: per-setting leaderboards"). Previously-explored shapes and
their results live in git history.

**The modality-transfer question is now answered for LoRA**: the adapter-shape seam and the
four invariants port to a visual generator (I2P). The remaining goals are completing the LoRA
baseline results on every setting and then adding alternative shapes (see § "Roadmap").

## Session handoff (2026-07-14) — read this first

**Design (current).** The benchmark is **committed per-setting leaderboards** (`leaderboards/*.md`):
each entry is `(shape, free hyperparameters, matched−control ± std, #seeds, reproduce command)`,
reproducible by running the setting's CLI against the committed codec. There is **no** automated
search, proxies, or merge gate — that machinery was built, validated on LoRA, then removed as
over-engineering. What survives as *rules* (not code): the HP partition (shared substrate fixed,
only free HPs vary per entry, rank fixed) and the four invariants (matched−control never loss;
scale swept; difficulty knob; multi-seed). A new shape is a codec subclass + `make_codec` entry +
manifest, added one at a time, each with its leaderboard row. See § "The benchmark: per-setting
leaderboards" and `leaderboards/README.md`.

**Validated LoRA baselines** (recorded in the leaderboards): image reward-tilting matched−control
**+3.50** (IR gain +0.160 ± 0.015 vs −3.34 reward-swap; scale optimum 2–4); T2L conditioning
**+0.033** (150K × 3 seeds, all 4 families); D2L NIAH **+1.00** (acc 1.0, ctxswap 0.0). The paper
(`~/adapterbench-paper.tex`) is updated to this design with these numbers.

**⏳ RUNNING right now (do not kill unless intended):**
- **1M-step T2L run** (paper scale, the *task* setting — not document) — `t2p-sft-pilot` →
  `results/t2p_cond_1M/s777/`, single seed 777, on cuda:0, restart-safe (`--checkpoint-every 10000`),
  ~2 days. Recipe: 128 descriptions, batch 8, lr 2.5e-5, warmup 0.1, `--adversarial-control`,
  eval-limit 80. `results.jsonl` is written only at the end. **If it dies, re-run the identical
  command** (§ "T2L" IN-PROGRESS block) — it resumes from the last 10K checkpoint. When it finishes:
  `.venv/bin/python scripts/t2p_rigor_aggregate.py --results results/t2p_cond_1M/s777/results.jsonl`,
  extend the emergence curve (5K→20K→60K→150K→**1M**), decide multi-seed-at-1M.
- **3-seed D2L base-retrieval baseline** — seeds 777/778/779 on cuda:1/2/3 →
  `results/repro/document_niah_lora/s{777,778,779}/`, started 2026-07-14. All use the
  validated 384-token/512-document/6000-step recipe at scale 45.25 and are restart-safe.
  When they finish: `.venv/bin/python scripts/d2p_niah_aggregate.py --root
  results/repro/document_niah_lora --min-seeds 3`, then update the D2L leaderboard row.
- Other users (`sa86`) intermittently share the GPUs; `nvidia-smi` before launching, prefer 0%-util
  GPUs, never touch cuda:0's 1M run.

**Completion checklist — aggregate both running experiments when they finish:**
- [ ] **T2L 1M:** run `.venv/bin/python scripts/t2p_rigor_aggregate.py --results
  results/t2p_cond_1M/s777/results.jsonl`, then update the emergence curve, T2L leaderboard,
  and paper.
- [ ] **D2L 3-seed:** run `.venv/bin/python scripts/d2p_niah_aggregate.py --root
  results/repro/document_niah_lora --min-seeds 3`, then update the D2L leaderboard.

**LoRA results status + gaps** (the "complete the baseline" work):
- **Image (I2P): complete.** Operating point (scale 4, reg 0.25) × 3 seeds + reward-swap control ×
  3 seeds + full scale sweep (single-seed tails) + reg difficulty knob. Archived
  `results/_archive/i2p_hypernoise_v2/`.
- **Task (T2L): 150K × 3 done** (`results/_archive/t2p_cond_long/s{777,778,779}_150k`); **1M paper
  scale running**. Gap = finish 1M (+ optional multi-seed-at-1M).
- **Document (D2L): length-gen is 6-seed** (`d2p_lengthgen_lora_s777..782`) but **base NIAH
  retrieval is single-seed** (`d2p_niah_lora`). Gap = ≥3-seed base retrieval; confirm the scale
  operating point (≈45). **Three-seed completion run is now in progress** at the paths above.

**Next steps (priority order):**
1. **Finish the 1M T2L run** (running) → aggregate, extend the emergence curve, update the T2L
   leaderboard row and paper.
2. **Complete the D2L base-retrieval baseline** — the ≥3-seed operating-point run is in
   progress; aggregate it and update `leaderboards/document_niah_d2l.md` when it finishes.
3. **Codebase cleanup** — audit for anything orphaned by the machinery removal (dead scripts,
   stale docstrings); keep only the setting CLIs + aggregate/smoke helpers.
4. **Then add alternative shapes** (IA³, LoKr, FourierFT, activation-steering — in git history), one
   PR at a time, each with its leaderboard row.

**Repo hygiene:** `results/` is gitignored scratch (the LoRA baseline data is under
`results/_archive/`); checkpoints/`*.pt`/`*.log` never committed. Stage-ready: `leaderboards/`,
`data/`, modified `src/`/`scripts/`/docs. **Nothing has been committed — the human drives that.**
The T2L data is vendored to `data/t2l/` (13 MB) so no `upstream/` clone is needed.

## Architecture

- `src/adapterbench/contracts.py` — `HypernetworkBackend.generate()`,
  `DownstreamEvaluator.evaluate()`, `TaskExample`/`AdapterArtifact`/`EvaluationResult`.
  Every setting implements these two interfaces; nothing else needs to know which setting
  produced a result.
- `task_examples.py` — builds `TaskExample`s for arc_easy/arc_challenge/boolq/
  hellaswag/gsm8k from public HF datasets, using upstream's own prompt templates and
  3-shot in-context examples verbatim. Scoring is generation + answer-extraction, not
  log-likelihood-over-choices — an answer is generated and a leading choice letter/digit
  (or a loose true/false keyword) extracted.
- **The codec seam:**
  - `t2p/codecs.py` — differentiable generated-parameter codecs (`GeneratedUpdateCodec`
    base). **LoRA (`LoRACodec`) is the only registered codec in the baseline**; a new
    shape is a subclass + one `make_codec` entry + a manifest, added one at a time with its
    own leaderboard row (see § "The benchmark: per-setting leaderboards").
    The base contract (`dense_delta`, `initial_bias`) and the hypernetwork's `"block"`
    residual-stream hook path are deliberately kept general so weight-space and
    activation-space shapes both plug in without framework changes.
  - `t2p/hypernetwork.py::TextToPeftHypernetwork` — the shared hypernetwork shell;
    `apply()`'s hook closure handles both bare-tensor (linear-submodule) and tuple
    (whole-decoder-layer) outputs. `PooledVectorConditioner` (task-description) and the
    document conditioner both plug into the same trunk/heads/codec path via a pluggable
    `conditioner`.
  - `t2p/lol_data.py` — Lots-of-LoRAs/SNI training data, read from the **vendored** `data/t2l/`
    (per-task `metadata.yaml` + the split yaml; see `data/t2l/NOTICE.md`) so the setting needs
    no `upstream/` clone. `load_decontaminated_train_task_ids` reads T2L's own 479-task training
    split from `hyper_lora_decontam_lol_tasks.yaml`; `validate_training_tasks` refuses any task
    outside it (never one of T2L's contamination-removed or held-out-validation tasks).
  - `t2p/sft_trainer.py` — the T2L training loop. `train_downstream_hypernetwork` (fixed
    step budget) and `train_with_checkpoints` (several budgets under one persistent
    optimizer, for step-budget sweeps) both support `grad_accum_steps` and `warmup_steps`
    — matching upstream's recipe (`batch_size=4, grad_accum_steps=64` → effective batch
    256, `lr=1e-5`, `warmup_frac=0.1`, 10 epochs over the 479-task corpus).
  - `t2p/live_evaluator.py::HypernetworkDownstreamEvaluator` — activates a generated
    adapter via `hypernetwork.apply(...)` (the same mechanism training uses) rather than
    `peft.PeftModel.load_adapter`, so adapters trained here (never materialized as PEFT
    adapters) can be scored at all.
  - `cli/`'s `t2p-sft` (single adapter), `t2p-sft-pilot` (train + score, optionally on the
    full 479-task split via `--all-decontam-tasks`, optionally across seeds via `--seeds`),
    and `t2p-sft-sweep` (checkpointed step-budget sweep under one persistent optimizer, to
    tell "still converging" apart from "already past the point where held-out
    generalization peaks").
- **Document-conditioning (D2L) variant:** same live-SFT loop, same codec seam, only the
  conditioning input changes — from a pooled task-description embedding to a frozen
  interpreter's own per-layer token activations on a synthetic NIAH document.
  - `t2p/document_conditioning.py` — `capture_document_activations` (a `@torch.no_grad()`
    forward through the frozen interpreter with `output_hidden_states=True`),
    `capture_early_exit_representation` + `EarlyExitPerceiverConditioner` (the generation
    path that learns held-out retrieval — see § "D2L NIAH"), and the earlier
    `DocumentPerceiverConditioner`. All conditioners expose `prepare_condition`, so the
    trainer/evaluator are conditioner-agnostic; the codec seam and `generate_per_layer`
    are unchanged.
  - `t2p/niah_data.py` — synthetic haystack/needle generation (`make_niah_example`,
    `needle_style="generic"` = one topic-free `"The special magic number is NNNN."` needle
    among repeated noise, chat-tokenized), `DocSFTDataset`/`doc_collate_fn`/`DocSFTBatch`,
    `build_niah_eval_examples`. **Documented simplification:** every document is packed into
    one context window per example — this does NOT replicate Doc-to-LoRA's multi-chunk rank
    composition (`combine_lora`); `assert_context_fits_in_one_pass` is the guard. Valid
    because Qwen3-0.6B's `max_position_embeddings` (40960) exceeds every configured length.
  - `t2p/document_sft_trainer.py` — the document-conditioned training loop
    (`compute_doc_sft_loss`/`doc_train_step`/`train_doc_downstream_hypernetwork`, and
    `train_doc_niah_checkpointed`, restart-safe with atomic
    model+optimizer+scheduler+step+history checkpointing).
  - `t2p/live_evaluator.py::DocumentHypernetworkDownstreamEvaluator` — hooks a fresh
    adapter *per example* (every NIAH document is distinct) and scores via exact 4-digit
    substring match, with a **context-swap control** (`accuracy_ctxswap`).
  - `cli/`'s `d2p-sft-pilot` (mirrors `t2p-sft-pilot` with `--context-lengths`/
    `--num-train-documents`) and `d2p-niah` (the D2L-parity NIAH command, with
    `--eval-context-lengths` for the length-generalization sweep).

## D2L NIAH: the from-scratch recipe (why the setting is legitimate)

The D2L setting only means something if a from-scratch hypernetwork can actually learn NIAH
retrieval at all. Two pieces of work established that it can, and pinned exactly what is
load-bearing.

**Reproducing the upstream NIAH recipe (the L1 ablation).** Doc-to-LoRA's headline claim is
near-perfect NIAH retrieval far beyond its 32–256-token training contexts. None of the four
released checkpoints are NIAH-trained, so reproducing it required running the upstream recipe
from scratch. Running `scripts/niah/1-train.sh` verbatim (seed=1, paper-scale data) **failed
to converge** — retrieval collapsed to a frozen ~0.05, and `gen_lora_l1_norm` fell from 0.015
to ~0.0004 almost immediately: the recipe's `gen_lora_l1_reg_coef=1.5` (15× the main
experiment's 0.1) crushed the context-dependent LoRA to zero, leaving only a
context-independent bias-LoRA that expresses the template but not the needle. An ablation
changing *only* that coefficient confirmed it is the whole story:

| eval bin | shipped (L1=1.5) | L1=0.1 | L1=0.0 | paper (D2L) |
|---|---|---|---|---|
| 1024–2048 | 0.00 | 1.00 | 1.00 | ~1.0 |
| 7168–8192 | 0.13 | 0.97 | 1.00 | 1.0 |
| 16384–20480 | 0.17 | 0.50 | 0.97 | — |
| 28672–32768 | 0.07 | 0.10 | 0.67 | 0.997 |

The paper's qualitative claim (near-perfect retrieval far beyond training length and the base
context window) reproduces at L1=0 or 0.1; the exact 28–32K figure does not; the shipped
L1=1.5 trains no working model in our environment. Whether upstream's successful runs used a
different effective coefficient (or the collapse is seed/hardware-contingent) is an open
question for upstream.

**The in-framework recipe that retrieves (the "D2P" path).** Ported into the codec-seam
framework, the generated-LoRA D2L setting reaches **held-out exact-digit accuracy = 1.0 with
the context-swap control = 0.0** on Qwen3-0.6B — genuine document-dependent retrieval, not
memorization or a digit prior. What was load-bearing, beyond the L1 lesson:
- **Generic-needle, chat-tokenized data** (`needle_style="generic"`): one topic-free needle
  among repeated noise, document wrapped as a chat user message, answer = the bare 4 digits.
- **`down_proj` hook** and **LoRA scale = 2·r^1.5 = 45.25 applied directly** (~8× rslora's
  default) — the two most load-bearing knobs (see gotcha on scale below).
- **lr 4e-5, reg 0, no dropout.**
- **Early-exit context encoder** (frozen interpreter's first `L//4 = 7` decoder layers) +
  **Perceiver-IO** (208 latent queries, 8 cross-attention blocks) → per-layer output-query
  decode → the existing trunk/heads/codec seam.

Cheap validation showed a clean phase transition (held-out matched 0.00 → 0.56 → 1.00 across
steps, ctxswap pinned at 0.00, train and held-out rising together = a general copy algorithm,
not memorization); 512 documents suffice at ctx ~384.

**Length generalization (the graded difficulty knob).** Training on 128+256-token documents
and evaluating a length sweep (256 → 512 → 1K → 2K → 4K → 8K, context-swap control = 0.00 at
every bin), the generated LoRA retrieves reliably out to **8192 tokens — a 32× extrapolation
beyond training length** — degrading only at the far end (the knob is genuinely graded;
nothing saturates across the whole sweep). Per-seed the retrieval phase transition is
stochastic (lands anywhere ~1500–2500+ steps), which is why the protocol requires multiple
seeds (see § "The four invariants"). `aggregate_lengthgen.py` emits the per-run length curve
plus transition step and crossover length from the training logs.

## T2L: task-description setting (works end-to-end; control added — conditioning unproven)

`t2p-sft-pilot` trains a hypernetwork from scratch, conditions on a task-description
embedding, and scores held-out boolq/hellaswag (generation + answer-extraction, dropout
disabled during scoring per the eval-mode gotcha). With the LoRA baseline on the full
479-task decontaminated corpus (3 seeds, upstream's recipe), **LoRA clears the frozen
interpreter on the discriminative task** (hellaswag ≈ 39% vs frozen ≈ 23%); boolq barely
separates (most everything sits near frozen's 75%), which is itself a finding about the eval
set, not the adapter — see the difficulty-knob invariant.

**Rigor pass (2026-07-13) — the mismatched-description control changes the read.** The setting
now has the control it was missing: `live_evaluator.py` computes `accuracy_mismatched` (each
family scored with an adapter generated from a *different* family's description — the T2L
analogue of D2L's `accuracy_ctxswap`), `t2p-sft-pilot --scales` sweeps LoRA scale, and the
eval spans a difficulty spread (frozen headroom hellaswag 0.23 → boolq 0.70). Run first at a
first-pass recipe (full 479-task corpus, lr 1e-5, effective batch 256, **8** descriptions/task,
a few hundred steps — later found under-powered; see "The null was under-training" below),
**3 seeds, eval-limit 80** (`results/t2p_rigor_ms/`, via `scripts/t2p_rigor_aggregate.py`):

| family | frozen | matched | mismatched | matched − mismatched |
|---|---|---|---|---|
| arc_easy | 0.713 | 0.733 ± 0.05 | 0.750 | **−0.017** |
| arc_challenge | 0.475 | 0.500 ± 0.06 | 0.521 | **−0.021** |
| hellaswag | 0.225 | 0.325 ± 0.05 | 0.346 | **−0.021** |
| boolq | 0.700 | 0.729 ± 0.03 | 0.758 | **−0.029** |

The adapter **does** beat frozen (hellaswag +0.10, others +0.02–0.03), but the **mismatched
adapter gains just as much** — matched − mismatched is a consistent, tight ≈ **−0.02** across
all four families and three seeds. So at this recipe the T2L gain over frozen is **real but not
task-description-specific**: a wrong description works as well as the right one. This is exactly
the over-attribution the control exists to catch — the earlier "hellaswag 39% vs frozen 23%"
was measured *without* a control and credited conditioning this recipe does not exhibit.
(This turns out to be a property of the recipe, not the setting — the paper-matched recipe below
does condition. The value here is that the control flags the difference either way.)

**Strong control at the under-powered recipe — no conditioning (`results/t2p_strong/`).**
The weak swap above deranges among four similar QA descriptions, so a skeptic could argue the
descriptions were too close to tell apart. The strong control (`t2p-sft-pilot
--adversarial-control`) instead scores each family with an adapter generated from a maximally
*dissimilar / meaningless* description — Text-to-LoRA's own `additional_eval_descs`
(`"dogs;cats;bananas;"`, random noise, `"gggg…"`). Over a 3-scale × 3-seed grid it too showed
matched ≈ mismatched (per-family −0.056 … +0.031, all within the seed std) and a flat scale
sweep (2.83/5.66/11.31 → 0.598/0.571/0.570). So *at this recipe* the gain over frozen is not
description-specific — but this recipe is **far from T2L's actual SFT recipe**, and the null
turned out to be an artifact of it (next).

**The null was under-training, not the setting (`results/t2p_cond*/`, 2026-07-13).** Comparing
against the Text-to-LoRA paper, the runs above were doubly under-powered: **8 task descriptions
per task** (T2L uses 128, sampled online per example) and **a few hundred SGD steps** (T2L
scales to ~$10^6$). With only 8 descriptions the hypernetwork can minimize the SFT loss with a
task-*independent* adapter that never reads the description. Rerunning the **paper-matched
recipe** (128 descriptions, batch 8, no grad-accum, lr $2.5\times10^{-5}$, warmup 0.1) at a
step-budget sweep (single seed, eval-limit 40) shows description-conditioning *emerging with
training*: mean matched − adversarial = **+0.006 (5K) → +0.056 (20K) → +0.044 (60K)** — flat
noise at 5K, clearly positive by 20K.

**Confirmed positive at 150K × 3 seeds, eval-limit 80 (`results/t2p_cond_long/`):**

| family | frozen | matched | adversarial-mismatched | matched − adv |
|---|---|---|---|---|
| arc_easy | 0.713 | 0.700 ± 0.035 | 0.667 | **+0.033** |
| arc_challenge | 0.475 | 0.542 ± 0.050 | 0.504 | **+0.037** |
| hellaswag | 0.225 | 0.358 ± 0.062 | 0.338 | **+0.021** |
| boolq | 0.700 | 0.713 ± 0.080 | 0.671 | **+0.042** |

**All four families are positive** (mean matched − adversarial ≈ **+0.033**), consistent across
three seeds — the correct description beats pure junk on every family, and the gains over frozen
concentrate where there is headroom (hellaswag +0.13, arc_challenge +0.07). The effect is
**real but modest**: 150K steps is ~15% of T2L's budget, so this is plausibly still
training-limited (the emergence trajectory has not obviously saturated).

**⏳ IN PROGRESS — the paper-scale (1M-step) run (started 2026-07-14).** To test whether the
effect strengthens toward the paper's numbers at full scale, a **1M-step** run is training in
the background: single seed 777, same paper-matched recipe (128 descriptions, batch 8, lr
2.5e-5), `--adversarial-control`, eval-limit 80, restart-safe (`--checkpoint-every 10000`).
- Output: `results/t2p_cond_1M/s777/` (metrics `results.jsonl` written only at the end; live
  progress via `results/t2p_cond_1M/s777.log` loss + the `ckpt_lora_seed777_scaledefault.pt`
  checkpoint's step count).
- Single-GPU wall-clock is **~2 days** (~0.2 s/step). It is uncheckpointed-safe: if it dies,
  **re-run the exact same command** (below) and it resumes from the last 10K-step checkpoint.
- **When it finishes:** `python scripts/t2p_rigor_aggregate.py --results results/t2p_cond_1M/s777/results.jsonl`,
  then record the matched−adversarial numbers in the table above (extending the emergence
  trajectory 5K→20K→60K→150K→1M) and decide if multi-seed at 1M (2 more seeds) is warranted.
- Re-run / resume command:
  ```bash
  .venv/bin/adapterbench t2p-sft-pilot --device cuda:0 --all-decontam-tasks \
    --max-descriptions 128 --limit 40 --batch-size 8 \
    --eval-tasks arc_easy,arc_challenge,hellaswag,boolq --eval-limit 80 \
    --adversarial-control --seeds 777 --steps 1000000 --grad-accum-steps 1 \
    --warmup-frac 0.1 --learning-rate 2.5e-5 --max-grad-norm 1.0 --checkpoint-every 10000 \
    --output results/t2p_cond_1M/s777
  ```
A 3-seed run at 1M would need ~2 days each (parallelizable across GPUs) or a data-parallel (DDP)
rewrite to speed a single trajectory; deferred until the single-seed 1M point justifies it.

**Why this matters.** The control does double duty. First it *caught an over-claim*: the prior
uncontrolled "hellaswag ≈39% vs frozen ≈23%" credited task-conditioning that, at that recipe,
neither control supports. Then it *tracked the real thing*: once the recipe is fixed
(descriptions + training scale — both load-bearing, and both invisible to a loss-only view),
the same control cleanly registers conditioning emerging and turning consistently positive. The
harness — matched, weak + adversarial controls, scale sweep, multi-seed, step-budget sweep —
is in place (`--adversarial-control`, `--scales`, `--max-descriptions`,
`scripts/t2p_rigor_aggregate.py`).

## Image domain: I2P — reward-tilting (HyperNoise), validated

The image setting answers the modality-transfer question: does the adapter-shape seam + the
four invariants carry to a visual generator? It does. A de-risk first ruled out from-scratch
visual *concept injection* (personalization had ~zero signal — a small adapter on a frozen
generator cannot inject an unseen subject). Reward-*tilting*, by contrast, is something a
small adapter demonstrably CAN do, and it is exactly the AdapterBench shape question in
image form. The setting reimplements **Noise Hypernetworks** (Eyring et al., 2025): an
adapter on a frozen distilled generator (SD-Turbo) modulates the *initial noise* to maximize
a reward, trained end-to-end by the tractable noise-space objective
`L(φ) = reg·mean(Δx₀²) − r(g_θ(x₀+Δx₀))`.

- **The seam is native.** The adapter is a `DirectCodecAdapter` — one codec per hooked
  attention Linear, the trainable parameter *is* the codec's generated-output vector (the
  degenerate "identity hypernetwork" case). It reuses `t2p/codecs.py::make_codec` and
  `initial_bias` verbatim (LoRA zero-init → Δx₀=0 at init). Other codec shapes drop in by
  swapping `codec_name`; a per-condition *generated* upgrade (the hypernetwork emits the
  noise-adapter) is the deferred Path B.
- **Hook site:** the UNet's attention linears (attn1/attn2 q/k/v/out — 128 Linears on
  SD-Turbo), found by `i2p/hypernoise.py::find_attention_linears`.
- **f_φ via a codec-agnostic difference form** `unet_adapted(x₀) − unet_base(x₀)` (exactly 0
  at init), instead of the reference's single-pass `conv_out`-delta patch — correct for any
  hook site, costs one extra (no_grad) UNet pass.

**Modules** (`src/adapterbench/i2p/`): `image_generator.py` (frozen SD-Turbo loader +
prompt embed), `hypernoise.py` (`DirectCodecAdapter`, `noise_transform`, grad-capable
`generate_from_latents`, `hypernoise_loss`), `rewards.py` (differentiable **ImageReward**
headline + redness swap-partner, behind one `reward(images01, prompts)` protocol),
`image_scoring.py` (frozen CLIP-T fidelity control), `hypernoise_trainer.py` (restart-safe
checkpointed training + the four-invariant `evaluate_rewards`), `prompt_data.py` (disjoint
train/eval prompts). CLI: `i2p-hypernoise` (one atomic cell). Pipeline:
`scripts/i2p_hypernoise_pipeline.py` (8-GPU grid) + `scripts/i2p_hypernoise_aggregate.py`
(four-invariant summary). Seam smoke: `scripts/i2p_hypernoise_smoke.py`.

**Four-invariant mapping (all satisfied on the LoRA baseline):**
1. **Behavioral metric + control; headline = matched − control.** Metric = ImageReward gain
   (adapter − frozen). Control = **reward-swap**: the adapter trained for ImageReward must
   raise ImageReward, while an adapter trained for a near-orthogonal reward (**redness**) must
   not — the image analogue of the D2L context-swap. A within-run **prompt-swap** control
   (score ImageReward against a *mismatched* prompt; a genuine alignment gain must not
   transfer) adds a second, cheap control with huge dynamic range (frozen matched IR ≈ +1.1
   vs prompt-swapped ≈ −2.2).
2. **Scale is swept, not fixed.** A LoRA-scale sweep (`--scale`), best-of reported — comparing
   at default scale ranks scales, not shapes (gotcha #10 in language form).
3. **Graded difficulty knob = `reg_weight`** — the fidelity↔reward tradeoff (the paper's
   Fig. 2). Higher reg → smaller noise edit → lower reward gain but higher fidelity. Reported
   as a curve (ImageReward gain and CLIP-T drop vs reg).
4. **Multi-seed** — reward maximization is stochastic; ≥3 seeds, mean±std.

**Validated recipe:** frozen SD-Turbo (fp32) · `DirectCodecAdapter` LoRA rank 16 over the 128
attention linears · SGD+momentum lr 1e-3 · **LoRA scale ≈ 2–4** (the operating point; see
below) · `reg_weight` a secondary knob · f_φ difference form · 1-step generation · 3000 steps.
Redness is the zero-dependency swap-control partner (trivially hackable — the NIAH-needle
analogue); **ImageReward-v1.0 is the headline reward** (BLIP-based human preference, fidelity
intrinsic, hard to hack), backpropagated through the frozen reward model to the generator's
noise.

**Headline results (2026-07-13, LoRA codec, 3000 steps, `results/i2p_hypernoise_v2`).**
Produced by `i2p_hypernoise_pipeline.py` + `i2p_hypernoise_aggregate.py`. All four invariants
satisfied with a genuine positive matched signal (the first run's default scale=2/reg=0.5 was
under-powered — the matched gain was within noise; a scale sweep pinned the operating point).

1. **Reward-swap control (invariant #1) — decisive, and now carried by a real matched signal.**
   At the operating point (LoRA scale 4, reg 0.25): the **imagereward** adapter raises
   ImageReward by **+0.160 ± 0.015** (3 seeds: 0.17/0.17/0.14) with CLIP-T essentially preserved
   (drop +0.004), while the **red** adapter (wrong reward) drives IR gain to **−3.34 ± 0.01**
   and CLIP-T down ~0.11. Headline **matched − control = +3.50** — the matched side is a
   consistent, fidelity-preserving gain, not just the control collapsing.
2. **Scale sweep (invariant #2) — a clear optimum (inverted-U).** IR gain vs LoRA scale (reg
   0.25): 2:**+0.185** · 3:+0.079 · 4:+0.170 · 6:+0.106 · 8:−0.107 · 16:−1.20 · 32:−0.36.
   Best-of ≈ scale 2–4; above the optimum the noise edit grows large enough to wreck the image.
   Scale is the **dominant knob** — the image-domain echo of "scale dominates shape."
3. **Difficulty knob (invariant #3) — `reg_weight`, most active in the high-scale regime.** At
   the stable scale 4, reg 0.1→0.5 holds IR gain ≈ +0.14–0.16 with CLIP-T drop rising as reg
   falls; at scale 16 reg is the safety valve (reg 0.1 → −3.38, reg 0.25 → −1.20), i.e. it
   controls how destructive an over-large scale becomes. Fidelity↔reward tradeoff is real but
   scale-mediated.
4. **Multi-seed (invariant #4).** The matched gain is tight across 3 seeds (±0.015), so the
   +0.16 signal and its separation from the −3.34 control are both statistically clean.

**This is the closed-out I2P result: a genuine, multi-seed, fidelity-preserving ImageReward
gain with a razor-sharp reward-swap control and a scale optimum** — a full member of the codec
panel. (First-run artifact for the record: at scale=2/reg=0.5 the matched gain was ~0, which is
why an operating-point sweep was needed; the peak is at low scale, not high.)

## How to run

```bash
# I2P: image-domain reward-tilting, one cell (LoRA codec, ImageReward headline). Restart-safe.
# Operating point: LoRA scale ~2-4, reg ~0.25 (higher scale destroys the image - see § I2P).
.venv/bin/adapterbench i2p-hypernoise \
  --device cuda:0 --reward imagereward --scale 4 --reg-weight 0.25 \
  --steps 3000 --eval-every 500 --n-seeds 2 --output results/i2p_hypernoise/imagereward_s777

# I2P: the full four-invariant pipeline across GPUs (reward-swap x scale-sweep x reg-knob x seeds),
# then the derived summary. Restart-safe (skips cells whose results.jsonl exists).
nohup .venv/bin/python scripts/i2p_hypernoise_pipeline.py --out results/i2p_hypernoise &
.venv/bin/python scripts/i2p_hypernoise_aggregate.py --out results/i2p_hypernoise

# T2L: full-corpus live SFT, LoRA baseline, three seeds
uv run adapterbench t2p-sft-pilot \
  --all-decontam-tasks --adapters lora \
  --seeds 777,778,779 --steps 380 --grad-accum-steps 64 --warmup-frac 0.1 \
  --learning-rate 1e-5 --eval-limit 60 --output results/t2p_sft_full

# T2L: step-budget sweep (single seed, several budgets, one persistent optimizer)
uv run adapterbench t2p-sft-sweep \
  --adapters lora \
  --checkpoint-steps 100,200,400,800,1200 --eval-limit 40 --output results/t2p_sft_sweep

# D2L NIAH: the recipe that retrieves (LoRA -> held-out acc 1.0, ctxswap 0).
# Restart-safe (resume by re-running the same command); use .venv/bin for long runs.
.venv/bin/adapterbench d2p-niah \
  --adapters lora --needle-style generic --context-lengths 384 \
  --num-train-documents 512 --steps 6000 --eval-every 500 --learning-rate 4e-5 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 --device cuda:0 --output results/d2p_niah_lora

# D2L NIAH length generalization: train short, eval a length sweep. --eval-context-lengths
# decouples eval length from --context-lengths. Restart-safe; one run per seed.
.venv/bin/adapterbench d2p-niah \
  --adapters lora --seed 777 --needle-style generic \
  --context-lengths 128,256 --eval-context-lengths 256,512,1024,2048,4096,8192 \
  --num-train-documents 512 --steps 6000 --eval-every 500 --learning-rate 4e-5 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 \
  --device cuda:0 --output results/d2p_lengthgen_lora_s777
```

```bash
uv run pytest -q   # 101 tests (LoRA-only baseline)
```

## Gotchas (read before touching the pipeline again)

1. **Neither the `DataLoader` shuffle nor `TextToPeftHypernetwork`'s own weight init/
   dropout was seeded**, despite every live-SFT command exposing `--seed`. Fixed:
   `torch.manual_seed(seed)` once at the top of each command *and* again immediately before
   each `TextToPeftHypernetwork(...)` construction (so a later-trained adapter doesn't
   inherit RNG state from an earlier one), plus an explicit `generator=` on every
   `DataLoader`. This makes a *single* run reproducible; it doesn't by itself establish
   stability *across* seeds.
2. **Qwen3's chat template defaults to "thinking" mode** — without `enable_thinking=False`,
   the model expects to emit its own `<think>...</think>` block before answering, so scoring
   a direct-answer continuation is badly out-of-distribution (confirmed: `frozen_interpreter`
   scored *below chance* on boolq without this fix). Applies to both training
   (`lol_data.py::format_prompt_response`) and eval (`live_evaluator.py::_prompt`); harmless
   no-op for non-Qwen3 tokenizers.
3. **LoRA has a dead zero-gradient saddle point at this hypernetwork's default all-zero head
   init.** It splits the head's output into two factors multiplied together — with both
   zero-initialized, the gradient w.r.t. each is proportional to the other, so both vanish.
   Fixed at the codec level: `GeneratedUpdateCodec.initial_bias()` lets a bilinear codec
   override the head's bias with one factor's slice randomized (the product is still exactly
   zero at init, but gradient reaches the zero factor immediately). Codecs that are linear in
   the generated output need no such fix — a general fact about the seam, not LoRA-specific.
4. **Moving this project's directory breaks every installed console script** (`.venv/bin/`
   entry points bake in an absolute-path shebang). Fix:
   `uv pip install --python .venv/bin/python --reinstall -e ".[dev]"`. Renaming
   `pyproject.toml`'s `[project].name` can separately make `uv run` re-resolve `uv.lock`
   against the wrong interpreter on `PATH`; re-lock with `uv lock --python .venv/bin/python`.
5. **Document-conditioning does NOT replicate Doc-to-LoRA's multi-chunk rank composition**
   (`combine_lora`). Every document is packed into one context window instead — valid only
   while every `--context-lengths` bin fits under the interpreter's `max_position_embeddings`
   (`niah_data.py::assert_context_fits_in_one_pass` is the net; `d2p-*` commands call it
   automatically before building a dataset, so an over-long bin errors instead of silently
   corrupting position encodings).
6. **The frozen interpreter runs in bf16 but every from-scratch hypernetwork parameter is
   float32** — `capture_document_activations` casts captured activations to float32 before
   the conditioner, or `nn.MultiheadAttention` raises a dtype-mismatch `RuntimeError`.
7. **Every live-SFT pilot/sweep command must call `hypernetwork.eval()` after training and
   before scoring** — otherwise the trunk's/conditioner's `nn.Dropout(0.05)` stays active
   during held-out scoring, adding non-determinism on top of genuine seed variance. Any new
   pilot/sweep command needs the same call.
8. **`TextToPeftHypernetwork.generate_per_layer` no longer recomputes the shared trunk/heads
   per layer** (it used to, 28× for Qwen3-0.6B, to enable a per-layer-immediate-backward
   memory optimization neither real caller exercises). Per-layer work is now limited to the
   conditioner call; trunk/heads run once, batched across layers, same as `forward()`.
   `forward_layer` (single-layer, for a future per-layer-backward caller) is unchanged.
9. **NIAH training loss is NOT a retrieval signal — never gate on it.** Response CE can be
   driven to ~0 while exact-digit retrieval is 0/N: the answer is ~5 tokens and CE is
   dominated by the easy teacher-forced continuation, while the one hard token (first digit,
   needing the adapter to carry doc identity) is never learned. Always evaluate with
   exact-digit generation **and** the context-swap control (`accuracy_ctxswap`).
10. **`LoRACodec` scale defaults to `alpha/sqrt(r)`=5.66, but D2L's NIAH recipe applies
    `2·r^1.5`=45.25 directly** (~8×). Pass `make_codec(..., lora_scaling=2*r**1.5)` for D2L
    parity (`d2p-niah` does this); the small default scale cannot override the frozen model
    to emit an unseen needle.
11. **Freezing the NIAH batch grouping starves the retrieval phase transition.** Materialising
    a `DataLoader` into a fixed list of batches once (reshuffling only batch *order*) cycles
    the same document groupings every epoch and converges ~3–4× slower — can look like it
    will never retrieve even though the recipe is correct. `train_doc_niah_checkpointed`
    reforms batches from a **document-level reshuffle every epoch** (deterministic per-epoch
    seed, so restart-safe resume stays exact). The transition step is also init/seed-sensitive
    (~1600–2500+), so give NIAH runs a generous step budget.
12. **The image reward's gradient must reach the adapter — a `@torch.no_grad` VAE decode
    silently severs it.** In I2P the reward is applied to the *decoded* image, so
    `generate_from_latents` uses a grad-capable VAE decode; an eval-only no-grad decode in the
    training path zeroes every adapter gradient while the loss still looks fine. The seam smoke
    (`i2p_hypernoise_smoke.py`) asserts all 128 adapter params get a finite non-zero gradient
    and the frozen UNet gets none.
13. **`image-reward` / `clip-anytorch` pin the Python env; installing them naïvely breaks
    CUDA.** `uv pip install image-reward` let the resolver bump torch 2.5→2.13 (+CUDA 13),
    which the CUDA-12.4 driver cannot run, and left shadowing `nvidia-*-cu13` libs behind. The
    deps are pinned in `pyproject.toml` (`image-reward`, `clip-anytorch`, `setuptools<80` for
    `pkg_resources`) so `uv lock`/`uv sync` keep torch at 2.5.1 — never `uv pip install` these
    imperatively. ImageReward's bundled BLIP also imports three symbols
    (`apply_chunking_to_forward` etc.) from `transformers.modeling_utils` that current
    transformers moved to `transformers.pytorch_utils`; `i2p/rewards.py` shims them before
    importing `ImageReward` rather than pinning transformers down (which would risk the Qwen3
    language pipeline).

## Benchmark design: the four invariants

**Read this before adding tasks or "improving" numbers.** AdapterBench's value is entirely in
whether its comparisons are **valid**, not in how many tasks it has. The dominant failure mode
is a confident-but-meaningless leaderboard. Two traps hit directly, both properties of the
*protocol*, not the task:
- **Loss is not capability.** Response CE was driven to 0.000 with 0/N retrieval (gotcha #9).
  Anything that gates, early-stops, or ranks on train/val loss measures nothing.
- **Scale dominates shape.** The same LoRA is 0.0 at scale 5.66 and 1.0 at 45.25. Comparing
  codecs at their *default* scales ranks scales, not shapes.

So the benchmark is defined by four protocol invariants. Any new task or setting (including
the image domain) MUST satisfy them.

1. **Behavioral metric with a built-in control; headline = matched − control, never loss.**
   Every task needs a control that catches cheating (priors, memorization, format-only
   learning). The D2L/NIAH setting has the gold-standard one: the **context-swap control**
   (generate the adapter from the *wrong* document, same query → must sit near chance;
   `accuracy_ctxswap`). A finding counts only when matched accuracy is high AND the control is
   near chance. A task with no clean control is a *complement*, not a core.
2. **Scale is a swept axis, not a fixed choice.** Run a small **per-codec scale sweep and
   report best-of**. Then "shape matters" means "even at its own best scale, shape X
   underperforms" — a claim that survives scrutiny. Comparing at default scales is exactly
   what NOT to publish as a shape result. Each leaderboard entry records the winning scale (the
   contributor sweeps it manually; `d2p-niah --lora-scaling` / `i2p-hypernoise --scale` /
   `t2p-sft-pilot --scales` set it).
3. **A graded difficulty knob so codecs actually spread.** A setting where everything
   saturates at 1.0 (or all fail) teaches nothing. **Length generalization is the ideal knob
   for the document setting** — continuous, cheap, and exactly where a shape either
   extrapolates or just memorized the training length. Report a **curve per codec** plus a
   scalar summary (the eval/train length ratio at which retrieval crosses 0.5). The image
   setting will need its own analogous knob.
4. **Multi-seed, because the quantity being measured is stochastic.** The retrieval phase
   transition landed anywhere from ~1600 to ~2500+ steps on *identical* recipes (gotcha #11).
   A single-seed number is partly luck. Use ≥3 seeds, report mean±std. The **transition step
   itself is a metric** (sample efficiency / trainability), not just noise to average away.

**What a codec should be scored on (a vector, not one number).** "Does shape matter" is
multi-dimensional; collapsing to peak accuracy throws away the structure. Report per codec:
peak held-out retrieval (matched − control, best-of the scale sweep); the
length-extrapolation ratio (most discriminative axis); sample efficiency (retrieval vs
#training docs, or the transition step); and **parameter efficiency** — retrieval per
*generated* parameter, i.e. "is this shape a good use of the hypernetwork's fixed output
budget?". Different shapes differ here by orders of magnitude, which is precisely what makes
the shape question interesting.

## The benchmark: per-setting leaderboards (current design, 2026-07-14 later)

An autoresearch search driver, lightweight proxies, and a git-native merge gate were built,
validated on LoRA, then **removed** as over-engineering (see the session handoff). The benchmark is
now deliberately simple:

- **`leaderboards/*.md`** — one committed leaderboard per setting (image / T2L / D2L). Each entry is
  `(shape, free HPs, matched−control ± std, #seeds, reproduce command)`. See `leaderboards/README.md`
  for the format and the rules. LoRA baseline entries are recorded (image +3.50, T2L +0.033, D2L +1.00).
- **Eval scripts = the existing setting CLIs** (`i2p-hypernoise`, `d2p-niah`, `t2p-sft-pilot`), which
  already take HPs and emit matched−control. A leaderboard entry's "reproduce command" is exactly one
  of these invocations; re-running it against the committed codec regenerates the number.
- **No automated HP search, no proxies, no merge gate.** HP selection is manual (by hand or a sweep
  the contributor runs) and recorded with the entry. What survives as **rules** (not code): the HP
  partition (shared substrate fixed, only *free* HPs vary per entry, rank fixed) and the four
  invariants (matched−control never loss; scale swept; difficulty knob; multi-seed). A new shape is a
  codec subclass + `make_codec` entry + manifest, added one at a time, each with its leaderboard row.

## Roadmap

**The image domain is now built and validated on the LoRA baseline** (§ "Image domain: I2P").
The `codec` + `hook site` seam proved modality-agnostic: the same seam, four invariants, and
`initial_bias` zero-init that drive T2L/D2L drive the reward-tilting SD-Turbo setting
unchanged. A shape ranking that holds across modalities is a far stronger claim than one
measured on NIAH alone — so the image setting is now a full member of the codec panel, and any
new codec shape is scored on it too.

**I2P Path B (generated noise-adapter) — EXPLORED, negative result (2026-07-14).** The Path B
upgrade — have the hypernetwork *generate* the reward-tilting noise-adapter from the prompt
(single forward pass) rather than train one directly (Path A) — was built, de-risked on two
rewards, and then **removed** (the tree carries only the working Path A). The "generate vs
optimize" claim splits into two sub-claims, and the image domain cleanly separates them:
- **Generation feasibility — SUPPORTED.** A `GeneratedCodecAdapter` (Path A's flat 128-target
  structure, but each per-target delta emitted by a shared conditioner→trunk→per-shape-head
  hypernetwork off the pooled CLIP prompt embedding) reliably beat frozen SD-Turbo: ImageReward
  **+0.14** (vs Path A's optimizer-fit +0.16), CLIP-align up to +2.0, fidelity preserved. A
  hypernetwork *can* emit a working noise-adapter in one forward pass.
- **Conditional specialization — NOT SUPPORTED.** The native control (generate the adapter from a
  *mismatched* prompt, keep content matched — the image analogue of the D2L context-swap) did not
  stably separate: matched−condswap ≈ 0 for ImageReward and only transient glimpses (+0.45 at
  mid-training, within ±1 run-to-run noise) for a prompt-*discriminative* CLIP-alignment reward.
- **Why (unifying explanation).** Training only ever sees matched conditioning, so a prompt-specific
  adapter emerges only if maximizing the reward *requires* one — as task/document conditioning does
  in T2L/D2L. Reward-tilting a frozen 1-step generator is served well by a **prompt-generic** nudge;
  injecting genuinely prompt-specific content via a small noise edit is close to the from-scratch
  concept-injection capability a prior de-risk already measured at ~zero. So **noise-space adapters
  do generic reward-tilting but not prompt-specific content steering.** This sharpens the thesis
  (hypernetwork *conditioning* works for tasks/documents, not for reward-tilting noise edits) and
  leaves the core "does shape matter" contribution untouched — that runs on **Path A**, which is
  already validated in the image domain and does not need Path B.
- **Lesson if revisited (code is gone; in git history if resurrected):** the generated adapter is an
  ~82M-param hypernetwork; **AdamW diverges** on it (per-parameter normalization makes the effective
  step ≈ lr·√N ≈ 0.9 even with grad-clip-to-1.0, overshooting the sharp reward optimum → image
  destroyed → NaN). **Use SGD+momentum** (as Path A does): with grad-clipping its step norm is
  exactly `lr`, preserving the reg-vs-reward balance. SGD lr 1e-3 / scale 2 / reg 0.25 was stable.

**Next active phase: complete the LoRA baseline, then add alternative shapes.**
1. Finish the 1M-step T2L run (running) and update its leaderboard row + the emergence curve.
2. Finish and aggregate the running ≥3-seed D2L base-retrieval baseline at the operating point
   (length-gen is already 6-seed; image and T2L operating points are 3-seed).
3. Codebase cleanup after the machinery removal (dead scripts, stale docstrings).
4. **Then add alternative codec shapes** (IA³, LoKr, FourierFT, activation-steering — all in git
   history), one at a time, each landing with its own leaderboard row at its own best free-HP config,
   under the fixed substrate + four invariants. This is the "does shape matter" payload.

**Deferred (valuable, lower priority):**
- **Multi-seed at 1M for T2L** (a second/third full-scale trajectory) once the first 1M run lands.
- **A graded difficulty knob / eval-task curation for T2L** where the frozen baseline leaves more
  headroom (the current families are QA with similar descriptions).

## Reference: prior art

- Text-to-LoRA — arXiv:2506.06105 (the system this benchmark generalizes beyond LoRA;
  task-description conditioning).
- Program-as-Weights — arXiv:2607.02512 (LoRA vs. prefix-tuning as hypernetwork targets;
  found LoRA ahead, on its own compiler/interpreter setting).
- Doc-to-LoRA — arXiv:2602.15902 (document-conditioned, NIAH-evaluated, structurally distinct
  from Text-to-LoRA). Its document-conditioning mechanism (cross-attention over a frozen
  interpreter's own per-layer activations) is the basis of the D2L setting here.
- HyperTuning (Phang et al.) — arXiv:2402.16817 (the one directly-comparable prior result that
  disagrees with LoRA-over-steering-tokens, on a different task distribution — an open question
  this project exists to help answer).
- Noise Hypernetworks (Eyring et al., NeurIPS 2025) — the reward-tilting objective and the
  SD-Turbo LoRA setting the I2P image domain reimplements in the codec seam.
- ImageReward (Xu et al., 2023) — the BLIP-based human-preference model used as I2P's
  differentiable headline reward.
- Representations of interest for future pipeline codecs: FourierFT (arXiv:2405.03003), KronA
  (arXiv:2212.10650), Compacter (arXiv:2106.04647).
