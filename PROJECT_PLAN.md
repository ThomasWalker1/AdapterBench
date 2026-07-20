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
into a real frozen interpreter's forward pass on real training examples, backprop ordinary
next-token cross-entropy through the hook, and evaluate on real held-out benchmarks. The
interpreter is per-setting (gemma-2-2b for T2L and Qwen3-0.6B for D2L). The hypernetwork
is trained entirely from scratch (no released checkpoint).

**Current status (2026-07-20).** Two genuinely conditioned language settings are complete
end-to-end with a rigorous **LoRA baseline codec**:
- **T2L** — task-description conditioning (Text-to-LoRA-style), scored on held-out
  benchmark tasks.
- **D2L** — document conditioning (Doc-to-LoRA-style), scored on held-out
  needle-in-a-haystack (NIAH) retrieval, including length generalization.

The former I2P reward-tilting setting is **retired**. Its positive row came from a single
directly optimized adapter shared across prompts, and prompt-conditioned follow-ups did
not pass the condition-shuffle/selective-behavior controls. It therefore did not test the
benchmark's core premise. Historical code and evidence are in `archive/retired_i2p/`.
`IMAGE_DOMAIN_PLAN.md` defines the admission gates for a replacement.

LoRA is the only codec on `main`. It exists to prove the *evaluation pipeline* works with a
single, well-understood representation; any pipeline failure is then attributable to the
plumbing, not the codec. **Additional adapter shapes are added one at a time** — a codec subclass
+ `make_codec` entry + manifest — each landing with its own row in the relevant setting's
leaderboard (see § "The benchmark: per-setting leaderboards"). Previously-explored shapes and
their results live in git history.

The modality-transfer question remains open. No image row will be restored until the
hypernetwork-only condition causally changes behavior relative to both static and
condition-shuffled controls.

## Session handoff (2026-07-20) — read this first

**State.** The benchmark has **two rigorously controlled, genuinely conditioned LoRA
baselines**, T2L and D2L. There is still only one codec (LoRA); alternative shapes are
deferred. There is no active image setting.

**Baselines** (`leaderboards/*.md`, all `matched − control`):
- **T2L** — `matched − static = −0.72 ± 0.16` nats CE (59/63 held-out task-seed pairs, 3 seeds),
  accuracy `matched − static / matched − frozen = +0.032 / +0.235`. gemma-2-2b, definition
  **stripped** from the input, plain SFT CE. CE is primary (non-saturating); accuracy corroborates.
- **D2L** — `+0.887 ± 0.143` (NIAH, 5 seeds, ctxswap 0.000, realistic-prose haystack; length-gen
  crossover at 4096 tok = 16× train length).

**✅ Seed 3 integrated (2026-07-19).** Seed 3 finished at 20k and independently confirmed the result:
CE `matched − static = −0.544` (19/21 tasks) and accuracy `+0.0248`. Across seeds 777/2/3, CE is
`−0.723 ± 0.162` nats and accuracy is `+0.0317 ± 0.0060`; matched beats static by CE on 59/63
task-seed pairs. Results:
`results/repro/t2l_base_diag/gemma2b_stripdef_hyper/s{777,2,3}/heldout_sni_{ce_full21,acc}.jsonl`.

**NEXT:**
1. **Image-domain go/no-go research.** Follow `IMAGE_DOMAIN_PLAN.md`: begin with a
   reference-image-conditioned direct-optimization capacity oracle, then run the small
   hypernetwork causality probe only if the oracle passes. Do not add a CLI or leaderboard
   before both `matched − static` and `matched − shuffled-condition` pass.
2. **Sweep engineering** — make per-codec scale sweeps efficient enough to produce *best-of-scale*
   LoRA baselines cheaply (invariant #2), the prerequisite for the T2L strip-def scale sweep and for
   every future codec. Also confirm/close the D2L best-of-scale sweep. (The T2L default-scale row is
   acceptable until this lands.)
3. **Paper + docs coherence pass** — once the science is settled: check `README.md`, `SETUP.md`,
   `BENCHMARK_CONTRACT.md`, `AGENTS.md` for stale framing; ensure the paper has no placeholders
   beyond planning and its numbers match the leaderboards; **add tests for the new T2L eval path**
   (`strip_task_def`, `StaticAdapter`, `t2p_eval_heldout_sni{,_acc}.py`) — currently zero coverage.
4. **License + hosting** — add `LICENSE` + `CITATION`; stand up GitHub Pages for `docs/index.html` at
   adapterbench.github.io (Settings → Pages → `main` `/docs`; fill the placeholder GitHub link
   `href="https://github.com/"`); initial commit + push (human-driven).
5. **(Deferred — user)** the **planning** setting — returns later; not started.
6. **(Deferred — user)** **alternative codec shapes** — the real "does shape matter?" payload; the
   registry (`make_codec`) has only LoRA, so the benchmark's central claim is not yet exercised.
   Adding one: codec subclass + `make_codec` entry + `configs/adapters/<name>.yaml`, then a leaderboard
   row per applicable active setting.

**Eval gotcha — `HF_HUB_OFFLINE=1` is required** for the held-out-SNI evals: after the one-time
`vendor_heldout_sni_metadata.py` downloads, the HF Hub 429-rate-limits the model-metadata check; the
model and datasets are cached, so offline mode is both correct and necessary. The reproduce script
sets it for you.

**Key files.** `GUIDE.md` (user guide — coherent with the current T2L method) · `leaderboards/*.md`
(source-of-truth rows) · `docs/index.html` (landing page) · `scripts/reproduce/task_t2l_lora.sh`
(one-seed T2L baseline) · `scripts/t2p_eval_heldout_sni.py` (CE) + `_acc.py` (accuracy) ·
`scripts/vendor_heldout_sni_metadata.py` (recovers all 21 held-out tasks) · `adapterbench-paper.tex`
§sec:t2l.

**Repo hygiene.** `scripts/reproduce/task_t2l_lora_ddp.sh` is **superseded** (the old
matched-adversarial / arc-boolq pipeline) — review and delete when integrating seed-3. `results/` is
gitignored scratch. Checkpoints/`*.pt`/`*.log` are never committed. **Nothing is committed — the human
drives that.** T2L data is vendored to `data/t2l/` (no `upstream/` clone needed).

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
  interpreter's own early-exit representation of a synthetic NIAH document.
  - `t2p/document_conditioning.py` — `capture_early_exit_representation` (a `@torch.no_grad()`
    forward through the frozen interpreter's first `L//4` decoder layers) +
    `EarlyExitPerceiverConditioner` (the generation path that learns held-out retrieval — see
    § "D2L NIAH"), the **only** shipped conditioner. (An earlier full-depth per-layer
    conditioner + full-activation capture were removed 2026-07-15 — they did not learn
    retrieval.) The conditioner exposes `prepare_condition`, so the trainer/evaluator are
    conditioner-agnostic; the codec seam and `generate_per_layer` are unchanged.
  - `t2p/niah_data.py` — synthetic haystack/needle generation (`make_niah_example`).
    `needle_style="generic"` = one topic-free `"The special magic number is NNNN."` needle
    among a repeated noise block; `needle_style="realistic"` = the same needle among **real
    Wikipedia prose** (BoolQ passages, loaded offline), a genuine-distractor haystack. Both
    share the topic-free query + exact-4-digit answer + context-swap control (so scoring stays
    clean; the needle/answer are always the synthetic number). Also `DocSFTDataset`/
    `doc_collate_fn`/`DocSFTBatch`/`build_niah_eval_examples`. **Documented simplification:**
    every document is packed into one context window per example — does NOT replicate
    Doc-to-LoRA's multi-chunk `combine_lora`; `assert_context_fits_in_one_pass` is the guard.
  - `t2p/document_sft_trainer.py` — the document-conditioned training loop
    (`compute_doc_sft_loss`/`doc_train_step`/`train_doc_downstream_hypernetwork`, and
    `train_doc_niah_checkpointed`, restart-safe with atomic
    model+optimizer+scheduler+step+history checkpointing).
  - `t2p/live_evaluator.py::DocumentHypernetworkDownstreamEvaluator` — hooks a fresh
    adapter *per example* (every NIAH document is distinct) and scores via exact 4-digit
    substring match, with a **context-swap control** (`accuracy_ctxswap`).
  - `cli/`'s `d2p-niah` (the D2L NIAH command: `--needle-style generic|realistic`,
    `--context-lengths` train length, `--eval-context-lengths` for the unified in-distribution
    + length-generalization sweep). (The earlier `d2p-sft-pilot` command was removed 2026-07-15.)

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

## T2L: task-description setting (✅ conditioning PROVEN — strip-def + matched−static; see the COMPLETE subsection)

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
**⚡ FAST DDP TRAINING PATH — `scripts/t2p_train_ddp.py` (2026-07-15, now the benchmark default).**
The single-GPU batch-8 trajectory above is ~2 days; that is untenable when *every new codec shape*
needs its own T2L run. The DDP path fixes the **data budget** (not the step count) and reaches it far
faster:
- **4-GPU data-parallel** (frozen interpreter replicated per rank — no grad; only the hypernetwork's
  few-M params all-reduce, so comm is cheap), `DistributedSampler` shards the corpus, rank-0 does the
  held-out eval.
- **Effective batch 128** (world 4 × per-GPU 32) × **62,500 steps = 8M example-visits**, i.e. the
  *same* total example-visits and ~418 epochs as 1M×batch-8 — just fewer, larger optimizer steps.
- **`torch.compile` + `--fixed-seq-len 512` (static shapes) + persistent codec hooks** →
  **471 ms/step**. Measured alternatives at eff-batch 128: eager+dynamic-pad 755 ms, eager+fixed-512
  792 ms, **compile+fixed-512 471 ms** (1.6×). Full run **~8.2 h** (~5.8× vs the batch-8 1M run).
- **lr √-scaled 2.5e-5 → 1e-4** for the 16× batch (AdamW; free HP per §optimization — trajectory
  shape deliberately differs from the batch-8 emergence curve, which is fine: the benchmark ships at
  this data-matched scale, not at a fixed step count).
- Restart-safe (rank-0 `--checkpoint-every 5000`, re-run identical command to resume), reuses the
  pilot's exact eval + `ResultRecorder` so `results.jsonl` is a **verified drop-in** for
  `t2p_rigor_aggregate.py`.
- **GPU allocation:** each run uses 4 GPUs (cuda:1–4), so seeds run *sequentially*; cuda:0's batch-8
  1M run and cuda:5–7 are untouched. Launch/aggregate command in § "How to run".
- **When the seed-777 run finishes** (`results/t2p_cond_ddp/s777/`): aggregate, record the
  matched−adversarial number as the T2L leaderboard/paper row at this config, then run seeds 778/779
  (same command, `--seed`), and add alternative shapes each via one such run.
- Compile note: hard `mark_dynamic` raises `ConstraintViolationError` under variable-length inputs
  (transformers specializes the seq dim) — hence `--fixed-seq-len 512` (numerically identical: pad
  tokens are masked in attention and loss) rather than dynamic shapes. See gotcha #14.

**Why this matters.** The control does double duty. First it *caught an over-claim*: the prior
uncontrolled "hellaswag ≈39% vs frozen ≈23%" credited task-conditioning that, at that recipe,
neither control supports. Then it *tracked the real thing*: once the recipe is fixed
(descriptions + training scale — both load-bearing, and both invisible to a loss-only view),
the same control cleanly registers conditioning emerging and turning consistently positive. The
harness — matched, weak + adversarial controls, scale sweep, multi-seed, step-budget sweep —
is in place (`--adversarial-control`, `--scales`, `--max-descriptions`,
`scripts/t2p_rigor_aggregate.py`).

### T2L: the conditioning problem and the pipeline redesign — ✅ COMPLETE (2026-07-19)

**Status: DONE.** Baseline established (3-seed, CE + accuracy), reproducible eval scripts, paper §sec:t2l
+ results + Table 1 finalized. The three-line summary: standard T2L SFT gives near-null conditioning
because the prompt already states the task; strip the task definition so the description is the only
route to it; then a from-scratch LoRA hypernetwork learns genuinely description-conditioned adapters
(matched−static = −0.72±0.16 nats CE on 59/63 held-out task-seed pairs; +0.235 accuracy over frozen). It was a
pipeline problem, not a codec problem. Details below.


**The finding.** Standard T2L SFT produces a *generically helpful* adapter, not a
description-conditioned one. At the shipped config the adapter beats frozen (≈ +0.05) but a **junk
description** ("dogs;cats;bananas") lifts it just as much — matched ≈ junk. This is **not a capacity
problem**: a controlled base-model sweep (`scripts/t2l_base_diag.sh`, shipped recipe fixed, only the
interpreter changes) is flat across scale — mean matched−junk **Qwen3-0.6B +0.003 / gemma-2-2b
+0.022 / Mistral-7B −0.012**, all within the ±0.05 eval noise. The paper's own weight-space null
result (no correlation between adapter similarity and description similarity) agrees.

**The diagnosis: a pipeline problem, not a codec problem — and the root cause is one line.** The
standard T2L prompt template is `{task_def}\n\n{problem}` (`lol_data.py`): it puts the task
**definition in the input the frozen model sees**. So the description fed to the hypernetwork is
*structurally redundant* — the model reads the task from `task_def` and never needs the conditioned
adapter (confirmed: with the definition stripped, the prompt for an emotion-classification example
is just the bare sentence, no task cue). Compounding this, the eval benchmarks (arc/boolq/hellaswag)
are ones the frozen model already handles, so there is **no headroom** for any adapter to add value —
matched ≈ frozen ≈ static ≈ junk, all bunched. Under such a pipeline *every* codec reads ≈0
conditioning, masking the codec-shape differences AdapterBench exists to measure. The pipeline must
be fixed before any shape comparison is meaningful.

**The template that works — D2L.** D2L conditioning is huge (+0.887) because the needle is *only* in
the document: the query cannot reveal it, so the adapter **must** carry the context. Conditioning is
*necessary* there. T2L has to be made necessary the same way.

**The redesign — make conditioning necessary at the data level, then measure it cleanly:**
1. **Pipeline fix: strip the task definition from the input** (`--strip-task-def`, `lol_data.py`
   template `{problem}` only). The task is then specifiable *only* through the description →
   hypernetwork, so the adapter *must* carry it — the D2L "context is necessary" principle. This is
   the primary, most fundamental fix (no objective can make a redundant description matter; this
   removes the redundancy). **Validated** (CPU): stripping turns an emotion-classification prompt
   into the bare sentence with no task cue.
   - **Still *task* conditioning, not problem conditioning.** strip-def changes only what the frozen
     model reads in its *input*; the hypernetwork is still conditioned on the task-level
     **description** (it never sees the individual problem). Two task statements coexist normally —
     `task_def` (in the input) and `description` (in the condition embedding) — and they are
     redundant with each other; strip-def removes `task_def` so the description becomes load-bearing.
     This **keeps T2L's conditioning mechanism unchanged** (description → gte → hypernetwork → LoRA)
     and departs only from T2L's SFT/eval *input format*. It is the honest test of T2L's own claim
     ("a description of a new task yields a working LoRA zero-shot"), which the standard self-describing
     prompt never actually tests — and is why T2L's own reported gains are small ("pinch of salt").
2. **Metric: `matched − static`.** static = a directly-optimized single adapter of the same shape
   (`--static`, `StaticAdapter`; the "multi-task LoRA"), trained on the same SFT data — it absorbs
   all *generic* help. matched − static > 0 means the description-conditioned adapter beats the best
   static adapter of that shape. **Non-gameable** (no negative to sabotage), **subsumes the
   helpfulness floor** (static ≥ frozen), needs **no junk prompts**. `scripts/t2p_eval_checkpoint.py
   --static-snapshot` reports it; junk − static ≈ 0 is an optional description-driven check.
3. **Eval where conditioning can show — the held-out SNI tasks** (the **21** `lol_###` keys in
   `eval_ds_info`, distinct from the 479 train tasks; = T2L's own held-out validation set — the other
   10 `eval_ds_info` keys are standard benchmarks like boolq/arc/gsm8k). With the definition stripped,
   frozen is *helpless* on them (it can't infer the task) → large headroom, and a single static adapter
   can't serve all 21 diverse tasks → matched should beat static *if* the pipeline taught conditioning.
   Free-form tasks → score by **teacher-forced CE**, not answer-extraction. Each held-out task has a
   **vendored `data/t2l/tasks/<id>/metadata.yaml`** (ds_kwargs/split/template/response_field, same as
   the train tasks) and **3 published held-out descriptions** in `eval_ds_info[<id>]["descriptions"]`.
   **Built:** `scripts/t2p_eval_heldout_sni.py` loads each held-out task with `strip_task_def=True`,
   applies the matched (description-conditioned) / static / frozen adapters via the trainer's own
   persistent codec hooks, and reports per-task + aggregate `matched − static` (headline; negative CE
   delta = conditioning helps) and `matched − frozen` (helpfulness floor), plus the fraction of tasks
   where matched < static.
4. **Optionally, a training objective that rewards using the description** — the neutral-junk
   objective (`--neutral-junk-lambda`): `CE_matched + λ(CE_mismatched − CE_frozen)²` via a 3rd
   no-grad frozen forward. Complements (1); non-gameable, unlike the contrastive hinge (gotcha #16).

**Decisive test:** train the hypernetwork **and** a static reference with `--strip-task-def`, then
`matched − static` on the 21 held-out SNI tasks. A single static adapter can't specialize to a
diverse held-out task from a definition-free input; a description-conditioned one can — *if* the
pipeline taught it to.

**RESULT (2026-07-19, 3 seeds) — conditioning is GENUINE under the redesigned pipeline.** All
strip-def runs trained to 20k steps on gemma-2-2b with **standard CE** (no neutral/contrastive —
`matched − static` needs no junk control): `gemma2b_stripdef_hyper` (hypernetwork) and
`gemma2b_stripdef_static` (`StaticAdapter`), identical recipe (20k steps, eff-batch 64 = per-gpu-batch
16 × 4, LIMIT 40, snapshots every 5k). `scripts/t2p_eval_heldout_sni.py` on the step-20000 snapshots,
**all 21 held-out SNI tasks**, 64 ex × 3 descriptions, teacher-forced CE:

  - **`matched − static` = −0.72 ± 0.16 nats** over 3 seeds (seed 777: −0.858; seed 2: −0.768;
    seed 3: −0.544), and **matched < static on 59/63 task-seed pairs (94%)**. Biggest per-task gaps (seed 777): lol_039
    −6.79, lol_710 −2.28, lol_701 −1.76, lol_614 −1.35.
  - **`matched − frozen` = −10.8 nats** (frozen mean CE 13.3, up to ~21) — with the definition stripped
    the base model is genuinely helpless, so the adapter is *necessary*, not merely additive.
  - **Training curve (seed 777, `heldout_curve.jsonl`):** matched − static −0.67 (5k) → −0.74 (10k) →
    −0.60 (15k) → −0.86 (20k) — upward but noisy, not strictly monotonic.
  - Results: `.../gemma2b_stripdef_hyper/{s777,s2,s3}/heldout_sni_ce_full21.jsonl`. Seed set via `SEED=`.
  - **Accuracy corroboration** (greedy generation, normalized exact-match, `t2p_eval_heldout_sni_acc.py`,
    `heldout_sni_acc.jsonl`, 3 seeds): matched **0.508** / static 0.476 / frozen 0.273 → **matched−frozen
    = +0.235** (adaptation large), **matched−static = +0.032 ± 0.006** (conditioning positive,
    non-reversing, matched ≥ static on 47/63 task-seed pairs, 75%). Smaller than the CE signal because accuracy **saturates** where
    both adapters already succeed and because MC tasks (MMMLU) leak answer choices into the stripped
    input (frozen scores high on content: high CE but high accuracy). Big per-task acc wins where
    unsaturated: lol_275 +0.375, lol_636 +0.31, lol_1711 +0.15, lol_039 +0.15. **CE stays primary**
    (non-saturating, well-defined for open-ended tasks) — this is stated in the paper.

This is the decisive evidence the T2L setting needed: **the earlier near-null was a PIPELINE problem,
not a codec problem.** Once conditioning is made necessary (strip-def) and measured where there is
headroom (held-out SNI, teacher-forced CE, `matched − static`), the LoRA hypernetwork produces
description-conditioned adapters that generalize to unseen tasks and beat the best same-shape static
adapter almost everywhere. The **neutral-junk objective was NOT the lever** — plain CE + strip-def
suffices; the winning runs use `--neutral-junk-lambda 0`. Results: `results/repro/t2l_base_diag/
gemma2b_stripdef_hyper/s777/heldout_sni_ce_full21.jsonl`.

**Data recovery.** Of T2L's 21 held-out `lol_###` tasks, only 10 shipped a vendored `metadata.yaml`;
`scripts/vendor_heldout_sni_metadata.py` resolves the other 11 to their `Lots-of-LoRAs/task###_*` Hub
datasets (all found; 4 are MMMLU MC, plus MNLI/winogrande/qasc/glucose/etc.), verifies load+parse, and
writes schema-matching metadata so the full 21 are reproducible.

**Paper (done, 2026-07-19):** §sec:t2l rewritten to the method that worked — strip-def + plain CE +
`matched − static` on held-out SNI (the neutral-junk objective / matched−junk / arc-boolq narrative is
dropped; user decision "rewrite to what worked"). Results subsection, Table 1 row (`−0.72 ± 0.16 nats
CE, 59/63 task-seed pairs, 3 seeds`), table caption (CE sign convention: more-negative-is-better), and invariant #1
(static-reference control) all updated. `HF_HUB_OFFLINE=1` is now required for evals (the Hub 429-rate-
limited us after the vendoring downloads; everything is cached).

**Documented negative (kept):** the earlier neutral-junk run — clean training dynamics (matched CE
0.23 ≪ frozen 8.19) but a null arc/boolq eval — a self-describing, no-headroom eval set cannot reveal
conditioning regardless of how clean training looks. That negative is *why* the strip-def +
held-out-SNI redesign was needed, and why the neutral-junk objective is not in the paper.

**Metric infra:** `scripts/t2p_rigor_aggregate.py` reports **matched − junk AND matched − frozen**
with a verdict (`NO CONDITIONING`/`SABOTAGE`/`WEAK-GENERIC`/`GENUINE`) and a noise floor (arc/boolq
protocol). `scripts/t2p_eval_checkpoint.py --static-snapshot` adds `matched − static` on those
families. The **held-out-SNI CE eval (`scripts/t2p_eval_heldout_sni.py`) is now built** — it is the
decisive `matched − static` instrument on definition-free held-out tasks; pending its first run on the
strip-def snapshots.

**Other infra (2026-07-17):** `--snapshot-every N` (model-only step snapshots → training curve);
`scripts/t2p_eval_checkpoint.py` scores any snapshot (`--avg-descriptions N` for paper-parity
3-description averaging), validated against gemma to ±1 example; eval task set expanded with
**gsm8k / openbookqa / winogrande** (PIQA's repo is a rejected loading-script; HumanEval/MBPP need a
code sandbox — deferred). Memory: 2B/7B bases use **8 GPUs × batch 16 = eff-batch 128**; the
neutral-junk objective's 3rd forward fits at batch 8.

## Archived image-domain record: retired I2P / UnHype boundary experiments

> **Not an active setting.** The implementation, experiment drivers, former leaderboard,
> and curated conclusion moved to `archive/retired_i2p/` on 2026-07-20. The historical
> detail below is retained to document the decision, not as a current benchmark recipe.

The former image setting asked whether the adapter-shape seam and four invariants carry to
a visual generator. A de-risk first ruled out from-scratch
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

**Archived modules** (`archive/retired_i2p/src/adapterbench/i2p/`):
`image_generator.py` (frozen SD-Turbo loader +
prompt embed), `hypernoise.py` (`DirectCodecAdapter`, `noise_transform`, grad-capable
`generate_from_latents`, `hypernoise_loss`), `rewards.py` (differentiable **ImageReward**
headline + redness swap-partner, behind one `reward(images01, prompts)` protocol),
`image_scoring.py` (frozen CLIP-T fidelity control), `hypernoise_trainer.py` (restart-safe
checkpointed training + the four-invariant `evaluate_rewards`), `prompt_data.py` (disjoint
train/eval prompts). The former CLI and experiment drivers are archived alongside them.

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

**Historical result only:** this was a genuine multi-seed, fidelity-preserving
ImageReward gain and a valid generic reward-tilting experiment, but not a valid
inference-time-conditioned AdapterBench row. It has been removed from the active panel.

### Retired prompt-conditioned and UnHype-style probes (2026-07-19–20)

**Why.** The shipped I2P (above) is the *unconditional* identity-hypernetwork case: `DirectCodecAdapter`
is a learned parameter, not a function of any condition, so I2P does not exercise the benchmark's core
premise — a hypernetwork *generates* a *conditioned* adapter — the way T2L/D2L do. The reframe made this
honest; this plan makes it real.

**Architecture (mirror T2L exactly).**
- Conditioning input: the prompt embedding (SD-Turbo's text-encoder pooled output).
- prompt emb → hypernetwork trunk → codec heads → per-Linear LoRA for the UNet attention projections
  (attn1/attn2 q/k/v/out — the same 128 Linears `find_attention_linears` already targets), emitted in
  one forward pass. Reuses `make_codec` + the T2P hypernetwork trunk.
- **Apply the adapter to the generator directly (weight-space), NOT as a noise perturbation.** Drop the
  HyperNoise `Δx₀ = unet_adapted − unet_base` construction and its `λ‖Δx₀‖²` regularizer — that
  noise-space indirection biases toward a small *generic* nudge. A straight prompt-conditioned LoRA on
  the UNet, generate, score, backprop, is cleaner and makes I2P architecturally identical to T2L/D2L.
  (Keep a fidelity term to prevent reward-hacking; CLIP-T drop is already the anti-hack check.)
- Metric: **`matched − static`** — the prompt-conditioned adapter vs. a single static LoRA of the same
  shape (the existing `DirectCodecAdapter` *is* the static reference) — plus the reward-swap control.

**The crux (read before building).** Unlike T2L, **you cannot strip the prompt from a text-to-image
generator** — the prompt is its essential input, so the frozen generator *already sees it*. A
prompt-conditioned adapter is therefore structurally **redundant** (the exact T2L redundancy problem,
but unfixable by stripping). Plain reward-tilting will most likely give **matched ≈ static** —
conditioning adds nothing over a generic tilt; this is why the de-risk found prompt-specific steering
≈ 0. Building the hypernetwork is necessary but not sufficient — conditioning must be made *necessary*.

**Necessity mechanisms (the strip-def analogue), in preference order:**
1. **Per-prompt distinct hard requirements the base model fails** (rare compositions, counting, spatial
   relations, specific styles), curated so the reward-maximizing edit genuinely differs per prompt and
   one static edit can't satisfy all. Most faithful to "conditioning is necessary"; most curation effort.
2. **Reward-conditioning** (safe fallback, *guaranteed* real signal): the hypernetwork reads *which*
   reward to optimize (ImageReward / redness / blueness / …) and must emit a reward-specialized adapter;
   metric = `matched − reward-swap`. Sidesteps prompt-redundancy entirely, but answers "shape matters
   for *reward*-conditioned tilting," not prompt-conditioning.
3. **Degraded-prompt-to-generator** (literal strip-def analogue): give the generator a stripped prompt
   ("a photo") and the hypernetwork the full description; the adapter must carry the content. Cleanest
   analogy, but it is the content-injection path the de-risk measured at ≈ 0 — will most likely fail.

**Recommended first step.** Build the prompt-conditioned hypernetwork (weight-space, mechanism-free) and
**measure plain `matched − static` first** — the cheap go/no-go on whether reward-tilting is even
slightly prompt-specific. If ~0 (likely), layer on #1. #2 is the fallback that *guarantees* a positive
conditioning result if I2P must land positive. Prototype small (a few hundred prompts, short training)
before the full build.

**Prototype result (2026-07-19) — optimizer works; mechanism-free prompt specialization does not.**
The weight-space prototype is built in `i2p/prompt_hypernetwork.py` +
`scripts/i2p_prompt_conditioning_probe.py`: pooled SD-Turbo prompt embedding → the shared T2P trunk →
shape-shared codec heads → one per-example LoRA for each of the 128 UNet attention projections. A
same-shape `DirectCodecAdapter` is trained on identical prompts/noise/objective. The probe uses 256
ordinary train prompts, 64 held-out prompts, 300 paired steps, and reports both the planned
`matched − static` and the stricter `matched − shuffled-condition` control.

- **Optimizer de-risk was load-bearing.** An initial AdamW scale sweep collapsed every generated
  adapter to ImageReward ≈−2.28 by step 10 (the known Path-B failure below), while the static controls
  improved. Switching to the documented SGD+momentum recipe (lr 1e-3, grad clip 1) was stable.
- **SGD result (one seed, scale sweep):**

  | LoRA scale | matched − static ImageReward | matched − shuffled | static − frozen |
  |---:|---:|---:|---:|
  | 0.5 | +0.063 ± 0.026 SE | −0.001 ± 0.006 | +0.049 |
  | 1 | +0.153 ± 0.036 | +0.011 ± 0.008 | +0.068 |
  | 2 | +0.152 ± 0.036 | −0.000 ± 0.010 | +0.120 |
  | 4 | +0.200 ± 0.042 | +0.023 ± 0.012 | +0.168 |

  CLIP-T is preserved/slightly better for matched than static (+0.004–0.005), so this is not reward
  hacking. But **matched ≈ shuffled at every scale**: the hypernetwork's advantage over the separately
  optimized static adapter is a generic parameterization/optimization advantage, not evidence that
  it uses the instance prompt. `matched − static` alone would over-attribute that gain; the shuffled
  condition control catches it.
- **Go/no-go: NO-GO for a full mechanism-free prompt-conditioned build.** This is the anticipated
  boundary result: ordinary ImageReward tilting does not make prompt-specific adapters necessary.
  Keep the shipped unconditional Path A as the honest image row. If a truly prompt-conditioned image
  member is required, add mechanism #1 using prompt-indexed, mutually incompatible requirements with
  a reward that can actually score them; require both `matched − static > 0` and
  `matched − shuffled > 0`. Reward-ID conditioning (#2) remains a guaranteed-positive fallback but
  must be labeled as reward-conditioning, not prompt-conditioning.

**Necessity-mechanism prototype (2026-07-19) — conditioning works; literal erasure is not yet
uniform.** We adapted UnHype's selective concept-removal framing into a structurally non-redundant
probe: SD-Turbo receives the *same* two-object prompt and latent twice, while the hypernetwork alone
receives either `remove A` or `remove B`. Each scene therefore has two incompatible requested
interventions. `scripts/i2p_selective_erasure_probe.py` trains the prompt-conditioned LoRA and a
same-shape directly optimized static LoRA on identical paired examples. The frozen CLIP training
score is `sim(image, retained) − sim(image, erased)`; evaluation uses unseen scene templates, 32
scenes × both directions × 2 seeds, with the two directions paired as one experimental unit.

| LoRA scale | CLIP matched − static | CLIP matched − swapped | target suppression | retained change | CLIP-T change |
|---:|---:|---:|---:|---:|---:|
| 0.5 | +0.00094 ± 0.00020 | +0.00188 ± 0.00039 | +0.00187 | −0.00093 | −0.00029 |
| 1 | +0.00193 ± 0.00043 | +0.00385 ± 0.00085 | +0.00078 | +0.00114 | +0.00156 |
| 2 | +0.00317 ± 0.00086 | +0.00634 ± 0.00172 | +0.00136 | +0.00181 | +0.00029 |
| 4 | +0.00816 ± 0.00115 | +0.01632 ± 0.00230 | +0.00556 | +0.00260 | −0.00626 |

The result transfers to an independently trained scorer. At scale 2, ImageReward margin is
`matched − static = +0.0893 ± 0.0322` and `matched − swapped = +0.1787 ± 0.0645`; at scale 4 it is
`+0.1576 ± 0.0343` and `+0.3151 ± 0.0686`. This is a **GO for the necessity framing**: unlike plain
prompt tilting, the adapter changes in the requested semantic direction and condition swapping
reverses the advantage. Scale 2 is the cleaner operating point; scale 4 is stronger but loses 0.0063
CLIP-T.

Do **not** yet integrate this as a solved "object erasure" benchmark. Qualitative grids show a clear
selective deletion in some cases, but many examples satisfy the margin mostly by strengthening the
retained object or globally restaging the scene. The next full-build gate is an object-presence
objective/evaluator that separately enforces target absence and non-target preservation (preferably
an open-vocabulary detector), followed by the same static and swapped controls. Until that gate
passes, describe this result as **target-conditioned semantic steering**, not reliable unlearning.

**Detector gate (2026-07-20) — final NO-GO for integration as object erasure.** We added an
evaluation-only Grounding DINO Tiny scorer and restricted the gate to scene/seed pairs where the
frozen image contains *both* objects at confidence ≥0.25 (60/64 pairs, 93.8%). Passing requires:
significant detector-margin `matched − static` and `matched − swapped`, positive target suppression,
retained-confidence loss no worse than −0.02, and ≥10% selective success with a ≥5-point advantage
over static. Three progressively more task-aligned objectives were tested:

1. **Original relative CLIP margin.** Scale 2 significantly improves detector margin
   (`matched − static +0.0256 ± 0.0079`; `matched − swapped +0.0511 ± 0.0159`) and preserves the
   non-target (`−0.0014`), confirming a real but small semantic routing effect. However, only **0.8%**
   of valid tasks cross the actual selective-erasure threshold. Scale 4 reaches 3.3% but is not
   significant and loses more retention.
2. **Absolute CLIP suppression (erase:retain weight 2:1).** This makes the static baseline learn a
   genuine generic suppression adapter rather than cancel exactly. The strongest cell (scale 4,
   fidelity 1) lowers detector target confidence by 0.113 and beats static/swapped in detector margin,
   but retained confidence falls 0.071, CLIP-T falls 0.039, and selective success is only **6.7%**.
3. **UnHype-style frozen denoising target.** We directly train the generated/static LoRAs to move the
   full two-object prediction toward the retain-only prompt, leaving the detector evaluation-only.
   At step 400, scale 2 / negative-guidance 0.5 and 1.0 reach **11.7% / 17.5%** apparent target
   deletion, but suppress the retained object almost identically (target `−0.363 / −0.407`; retained
   `−0.361 / −0.403`), collapse CLIP-T (`−0.051 / −0.120`), and show no significant matched advantage
   over static or swapped. The planned 1000-step runs were stopped at the next safe point because
   additional optimization was amplifying broad suppression, not selectivity.

**Final decision (2026-07-20).** The separate erase condition can produce measurable
semantic steering, so the structural-necessity argument is sound. But under SD-Turbo +
attention-projection LoRA, neither a relative image-space objective, an absolute
suppression objective, nor the UnHype denoising target produced reliable selective object
erasure. Stronger objectives suppressed both the target and retained object. Mechanism-free
prompt conditioning also failed its shuffled-condition control. I2P is therefore retired,
not renamed or integrated. This is a legitimate boundary result.

The replacement must use a natural external condition that the frozen generator cannot
otherwise see. The recommended first candidate is reference-image-conditioned identity or
appearance transfer; its staged capacity and causality gates are in
`IMAGE_DOMAIN_PLAN.md`.

## How to run

```bash
# T2L: reproduce the LoRA baseline for one seed — trains the strip-def hypernetwork + the same-shape
# static reference (data-parallel, gemma-2-2b), then scores matched-static on the 21 held-out SNI
# tasks (CE + accuracy). Needs HF_HUB_OFFLINE=1 (set by the script). Args: SEED GPUS_HYPER GPUS_STATIC.
scripts/reproduce/task_t2l_lora.sh 777 0,1,2,3 4,5,6,7
# read the aggregate rows (matched-static CE = primary; accuracy = corroborating):
grep __aggregate__ results/repro/t2l_base_diag/gemma2b_stripdef_hyper/s777/heldout_sni_ce_full21.jsonl
grep __aggregate__ results/repro/t2l_base_diag/gemma2b_stripdef_hyper/s777/heldout_sni_acc.jsonl

# D2L NIAH: the SHIPPED unified recipe (realistic real-prose haystack; one run = in-distribution
# headline @256 + length-gen sweep to 8192). Restart-safe. The 5-seed baseline is the reproduce
# wrapper: scripts/reproduce/document_niah_lora.sh [DEVICE].
.venv/bin/adapterbench d2p-niah \
  --adapters lora --needle-style realistic --seed 777 \
  --context-lengths 256 --eval-context-lengths 256,512,1024,2048,4096,8192 \
  --num-train-documents 512 --steps 12000 --eval-every 1000 --learning-rate 4e-5 \
  --n-latents 208 --num-blocks 8 --eval-limit 32 \
  --device cuda:0 --output results/repro/document_niah_realistic/s777

# Invariant-#2 best-of-scale sweep for D2L (single-seed locator; confirm the winner multi-seed after):
bash scripts/d2l_scale_sweep.sh 5,6,7   # D2L: sweeps --lora-scaling, self-summarizes scale->matched-control
```

```bash
uv run pytest -q   # 91 tests (active T2L/D2L code; retired image tests live in the archive)
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
   float32** — `capture_early_exit_representation` casts the captured representation to float32
   before the conditioner, or `nn.MultiheadAttention` raises a dtype-mismatch `RuntimeError`.
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
12. **`torch.compile` on the frozen interpreter fights variable sequence lengths.** With
    per-batch dynamic padding the seq length changes every step; Dynamo then specializes a graph
    per length, blows through `cache_size_limit`, and silently falls back to eager (losing the
    speedup). Hard `torch._dynamo.mark_dynamic(input_ids, 1)` does **not** rescue it — transformers
    specializes the seq dim internally, so it raises `ConstraintViolationError` and the run dies.
    The fix (`scripts/t2p_train_ddp.py --fixed-seq-len 512`) is to pad every batch to a fixed length
    so shapes are static and one graph is traced and reused — **numerically identical** (pad tokens
    are masked in both attention and the CE loss) and it doubles as removing the DDP variable-length
    straggler (all ranks do equal work per step). Also register codec hooks **once** (persistent,
    updating the generated-param dict in place) rather than re-registering per step via
    `hypernetwork.apply()`'s context manager, which would re-trace every step. Measured on
    Qwen3-0.6B at eff-batch 128: compile+fixed-512 471 ms/step vs eager 755 ms (1.6×).
13. **The DDP LR (`1e-4`) is tuned for the small base and diverges on a 7B interpreter.** On
    Mistral-7B the run converged cleanly to loss ~0.004 by step 25k, then the loss *spiked to 1.59*
    around step 30k and only crawled back — a classic Adam divergence (bigger gradients through the
    larger frozen base; the `max_grad_norm=1.0` clip alone didn't catch it). The paper uses **8e-5**
    for SFT; `LR=8e-5 scripts/t2l_base_diag.sh …` is stable on Mistral-7B. Rule: scale the LR *down*
    with the base model, and don't trust a low-loss prefix — watch the whole trajectory for a spike.
14. **A "matched > mismatched" conditioning loss is gamed by *sabotaging the negative*.** The
    contrastive hinge `relu(margin + CE_matched − CE_mismatched)` (`--contrastive-lambda`) is
    satisfied two ways: make the matched adapter *help*, or make the mismatched adapter *hurt*. Since
    CE_matched is already floored (the adapter fits), the network takes the cheap route and drives
    CE_mismatched to ~10+ (a destructive wrong-description adapter). At eval this generalizes to
    "unfamiliar description → sabotage": matched−junk looks huge (+0.6) while matched never beats
    frozen — a degenerate pass. Reducing λ only makes the sabotage milder, not genuine. Two fixes:
    the **neutral-junk objective** (`(CE_mismatched − CE_frozen)²` — punishes sabotage *and*
    genericness, only rewards matched-helps) and the **matched − static metric** (no negative to
    sabotage). Also: log cross-rank-*reduced* CE terms — a per-rank contrast value read against the
    reduced loss falsely looked like "0 contrast" for a whole run.

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
   what NOT to publish as a shape result. Each leaderboard entry records the winning scale
   (the contributor sweeps it manually; `d2p-niah --lora-scaling` and
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

- **`leaderboards/*.md`** — one committed leaderboard per active setting (T2L / D2L). Each entry is
  `(shape, free HPs, matched−control ± std, #seeds, reproduce command)`. See `leaderboards/README.md`
  for the format and the rules. LoRA baseline entries are recorded (T2L matched−static
  `−0.72 ± 0.16` nats CE; D2L `+0.887 ± 0.143` in-distribution / crossover 16×).
- **Eval scripts = the existing setting CLIs** (`d2p-niah`, `t2p-sft-pilot`), which
  already take HPs and emit matched−control. A leaderboard entry's "reproduce command" is exactly one
  of these invocations; re-running it against the committed codec regenerates the number.
- **No automated HP search, no proxies, no merge gate.** HP selection is manual (by hand or a sweep
  the contributor runs) and recorded with the entry. What survives as **rules** (not code): the HP
  partition (shared substrate fixed, only *free* HPs vary per entry, rank fixed) and the four
  invariants (matched−control never loss; scale swept; difficulty knob; multi-seed). A new shape is a
  codec subclass + `make_codec` entry + manifest, added one at a time, each with its leaderboard row.

## Roadmap

**Active baseline phase:** T2L and D2L have rigorous, reproducible LoRA rows. I2P was
removed on 2026-07-20 because a directly optimized, prompt-generic adapter is not a
hypernetwork-conditioned benchmark member. Its prompt-conditioned and UnHype-style
follow-ups are archived boundary results, not hidden failures.

**Image research:** follow `IMAGE_DOMAIN_PLAN.md`. Start with a direct-optimization
capacity oracle for reference-image-conditioned identity/appearance transfer. If it
passes, run a small held-out hypernetwork probe whose primary controls are
`matched − static` and `matched − shuffled-condition`. Only a causal positive result earns
an active package, CLI, leaderboard, or paper headline. A negative result is acceptable;
reward-ID conditioning remains a labeled diagnostic rather than a guaranteed-positive
core task.

**Future (explicitly deferred — not this phase):**
- **Alternative codec shapes** (IA³, LoKr, FourierFT, activation-steering — all in git history and
  in git history — their exploratory archived results were pruned in the 2026-07-15 cleanup, so each
  is re-run fresh), added one at a time, each a subclass +
  `make_codec` entry + manifest + leaderboard row at its own best free-HPs. This is the
  "does shape matter" payload and resumes after the active LoRA baselines are locked.
- **Multi-seed at the batch-8 1M scale** for T2L (a second/third full trajectory) if the emergence
  endpoint warrants it.
- **A difficulty knob / eval-task curation for T2L** with more frozen-baseline headroom (current
  families are similar-description QA).

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
- Noise Hypernetworks (Eyring et al., NeurIPS 2025) — the reward-tilting objective used
  by the retired I2P experiment.
- ImageReward (Xu et al., 2023) — the BLIP-based human-preference model used by that
  retired experiment.
- Representations of interest for future pipeline codecs: FourierFT (arXiv:2405.03003), KronA
  (arXiv:2212.10650), Compacter (arXiv:2106.04647).
