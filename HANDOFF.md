# AdapterBench — handoff prompt for the next agent

You are picking up development of **AdapterBench**, a benchmark that asks: *when a hypernetwork
generates a parameter-efficient adapter instead of an optimizer fitting one, does the adapter's
**shape** matter, and which shape is best?* The design holds the hypernetwork, training loop,
and evaluation protocol fixed and varies only the generated representation through a `codec`
(output structure) + `hook-site` (attachment point) seam.

**Read `PROJECT_PLAN.md` first** — start with its "Session handoff (2026-07-14)" section, then
the "T2L" and "Image domain: I2P" sections. This file is the short version plus concrete next
steps. The paper draft is `~/adapterbench-paper.tex`.

## Orientation (repo state)

- Three settings are validated **on the LoRA baseline codec** (LoRA is the only codec on `main`,
  by design):
  - **T2L** — task-description conditioning (Text-to-LoRA-style).
  - **D2L** — document conditioning (Doc-to-LoRA-style), NIAH retrieval + length generalization.
  - **I2P** — image reward-tilting on frozen SD-Turbo (HyperNoise-style); a *controlled positive*
    (matched ImageReward +0.16 ± 0.015 vs a −3.34 reward-swap control; scale optimum ≈ 2–4).
- Every setting reports **matched − control**, never raw accuracy/loss (see the four invariants
  in PROJECT_PLAN). Controls: D2L context-swap, T2L mismatched-description (weak + adversarial),
  I2P reward-swap + prompt-swap + CLIP-T fidelity.
- **Self-contained:** T2L data is vendored to `data/t2l/` (13 MB) — `upstream/text-to-lora/` is
  no longer needed and can be deleted. Training *examples* stream from the HF Hub at runtime.
- **Environment is pinned and fragile** (`pyproject.toml` / `uv.lock`): torch 2.5.1 + CUDA 12.4;
  `image-reward` + `clip-anytorch` + `setuptools<80` for I2P. **Never `uv pip install` those
  imperatively** — it bumped torch to a CUDA-13 build and broke CUDA last time. Use `uv sync`.
  Run things with `.venv/bin/adapterbench …` or `.venv/bin/python …`.
- **Git hygiene:** `results/` is gitignored (scratch by default); checkpoints (`*.pt`), logs,
  and adapter weights are never committed. Stage-ready right now: `data/`, the modified
  `src/`/`scripts/`/docs, and this file. **Nothing has been committed** — the human drives that.
- **Shared GPUs:** another user (`sa86`) intermittently occupies GPUs. Always `nvidia-smi`
  before launching, and prefer GPUs at 0% util.

## RUNNING right now — do not kill unless intended

- **1M-step T2L run** (paper scale): `results/t2p_cond_1M/s777/`, on cuda:0, single seed 777,
  paper-matched recipe (128 descriptions, batch 8, lr 2.5e-5, warmup 0.1), `--adversarial-control`,
  eval-limit 80, restart-safe (`--checkpoint-every 10000`). ~2 days wall-clock (~0.2 s/step).
  - Live progress: `results/t2p_cond_1M/s777.log` (loss) and the step count inside
    `results/t2p_cond_1M/s777/ckpt_lora_seed777_scaledefault.pt`.
  - Metrics (`results.jsonl`) are written **only at the end** (the pilot evaluates post-training).
  - **If it dies, re-run the identical command** (in PROJECT_PLAN's T2L "IN PROGRESS" block) — it
    resumes from the last 10K-step checkpoint. To get an early read, load a mid-training
    checkpoint into a fresh hypernetwork and run the evaluator manually.

## Next steps (in priority order)

The theme: **finish the one long run in flight; then the two development thrusts are (a) the
generated-adapter upgrade for images — I2P Path B, the headline generation-vs-optimization test —
and (b) the benchmark's automation machinery (the autoresearch inner loop + git-native merge
mechanism).** The machinery in (b) needs **no new codecs** to build or prove — LoRA (optionally a
rank/scale variant as a stand-in "second entry") exercises every path; new shapes are the
*payload* for a pipeline that already works, not a prerequisite.

### 1. Finish and record the 1M T2L run
When `results/t2p_cond_1M/s777/results.jsonl` appears (~2 days; resume with the command in
PROJECT_PLAN's T2L "IN PROGRESS" block if it dies):
`.venv/bin/python scripts/t2p_rigor_aggregate.py --results results/t2p_cond_1M/s777/results.jsonl`.
Add the matched−adversarial numbers to the T2L trajectory table in PROJECT_PLAN, extending the
emergence curve 5K→20K→60K→150K→**1M**. Then judge whether the effect strengthened at full scale;
decide if **multi-seed at 1M** (2 more seeds, parallelizable) is worth it. This is the only
compute-bound item — kick everything below off in parallel while it runs.

### 2. I2P Path B — the *generated* noise-adapter (the generation-vs-optimization test for images)
Today's I2P (`i2p/hypernoise.py::DirectCodecAdapter`) trains one shared noise-adapter *directly* —
the identity-hypernetwork case. Path B makes a hypernetwork **generate** the noise-modulation
adapter from a condition (the prompt embedding, or a reward-target embedding) in a single forward
pass, so a different prompt yields a different adapter. This is AdapterBench's headline
"hypernetwork generates the adapter vs an optimizer fits one," now in the image domain.
- **Reuse the existing seam.** `TextToPeftHypernetwork` (shared trunk/heads/codec + a pluggable
  conditioner) is exactly this machinery — it's how T2L generates per-task adapters. Point its
  hook site at the UNet attention linears (`i2p/hypernoise.py::find_attention_linears`;
  `module_shapes` via `infer_module_shapes`), condition on a pooled prompt embedding (the frozen
  text encoder's output, or a separate encoder), and train end-to-end on the *same* reward
  objective (`hypernoise_loss`), sampling prompts per batch so the generated adapter must work
  across prompts rather than fit one.
- **The control gets stronger and native.** Because the adapter is now generated per prompt, the
  control is a genuine **prompt-swap on the generation side**: generate the adapter from a
  *mismatched* prompt and require the reward gain to vanish — the image analogue of D2L's
  context-swap and T2L's mismatched-description. Headline = matched − mismatched, plus the CLIP-T
  fidelity check, exactly as now.
- **Feasibility flag.** The de-risk this session showed from-scratch *concept injection*
  generation had ~zero signal. Reward-*tilting* generation is easier (a small noise perturbation,
  not a new concept) but is still from-scratch generation, so start small: confirm a generated
  per-prompt adapter beats frozen **and** beats the mismatched-prompt control on a handful of
  prompts before scaling. If it holds, I2P is the only setting testing generation *and* shape in
  the image domain; if it doesn't, that is itself a reportable boundary of adapter generation.

### 3. Develop the autoresearch inner loop — validated on LoRA
This is the per-codec **free-HP search** described in PROJECT_PLAN § "Optimization" and
§ "Per-codec autoresearch": *fix a codec's shape, search its free optimization hyperparameters to
best-of on the frozen shared substrate, multi-seed, scored on `matched − control` (never loss),
then report the codec at its own best config.* Build and test it with **LoRA as the only codec**:

- **A search driver** that takes a codec + setting and sweeps the *free* HPs — `scale`, `lr`,
  `warmup`, `steps` — over ≥3 seeds, selecting best-of on `matched − control`. Reuse the existing
  restart-safe runners (`d2p-niah` for NIAH — the cleanest control; `i2p-hypernoise` for images;
  `t2p-sft-pilot --scales` for T2L) rather than reinventing training. Start by wrapping the manual
  sweeps this session already did (I2P scale/reg grid via `scripts/i2p_hypernoise_pipeline.py`,
  T2L via `t2p-sft-pilot`) into one parameterized driver.
- **Enforce the HP partition** (the trap that makes or breaks a *shape* benchmark): *shared
  substrate* (task data, conditioner/trunk, evaluator, control) is identical across codecs and
  never tuned; *free* HPs are searched; *shape-identity* HPs (e.g. rank) are fixed or only moved
  along the parameter-efficiency axis, never maximized. Encode this partition explicitly so a
  future codec can't cheat by tuning the substrate.
- **Three guardrails, each grounded in a bug we hit:** optimize `matched − control` not loss;
  every config is multi-seed (transitions are stochastic); equal search budget + space per codec,
  both reported (publish the tuned-HP table, not just a scalar).
- **Test on LoRA:** run the driver with LoRA on D2L-NIAH (and/or I2P), confirm it recovers the
  known good operating points (D2L scale ≈ 45; I2P scale ≈ 2–4) and emits a per-codec result
  *vector* (peak matched−control, difficulty-knob curve, sample efficiency, parameter efficiency).
  If it finds LoRA's optimum unaided, the loop works. Consider ASHA/successive-halving for cost,
  but prune carefully — some NIAH seeds only transition at ~4500 steps (see gotcha in PROJECT_PLAN).

### 4. Build the git-native merging mechanism — validated on LoRA
This is PROJECT_PLAN § "Git-native benchmark". The pipeline treats a codec as a small mergeable
unit and cleanly separates *merged (code only)* from *measured (derived records)*. Build and
dry-run the whole flow using LoRA (or a trivial LoRA rank/scale variant registered as a second
"codec") as the guinea pig — you do not need a genuinely new shape to prove the mechanism:

- **Correctness merge-gate (CI):** a check that a codec PR is a *valid, deterministic, fairly
  comparable* panel member — `adapterbench validate` (manifest well-formed) + `adapterbench
  catalog` (registers) + `pytest -q` + a short GPU smoke proving it **generates → hooks →
  backprops** through `hypernetwork.apply(...)` (not `peft.load_adapter`) + a **shape-identity
  lint** rejecting codecs whose generated-parameter count exceeds the panel's declared budget
  band. The gate tests correctness, **never** quality (a codec that loses to LoRA still merges).
- **Path guard:** codec PRs may touch only `t2p/codecs.py`'s registry region + `configs/adapters/`
  + `schema.py`'s family `Literal` + `results/` — enforce via CODEOWNERS / a CI path check, so a
  substrate change is forced down a separate, more-scrutinized path.
- **Derived, provenance-stamped leaderboard:** post-merge, the autoresearch loop (#3) produces
  `EvaluationResult` records stamped with `(main_git_sha, trial_id, seed, data_split,
  search_budget)`, appended (never overwritten) under `results/leaderboard/`. The leaderboard is a
  **view regenerated** from those records, not hand-edited, and is never a merge gate. Note:
  `results/` is gitignored as scratch (see below) — decide whether leaderboard records are the
  intended exception to force-add (`git add -f`) or live in a side store / GH Releases.
- **The proposal loop wrapper:** propose (shape + a hypothesis about which invariant axis it
  should move) → implement → gate → merge → evaluate → report (outcome vs hypothesis; losers stay
  on `main` as negative results). Scaffold it and dry-run with LoRA so the loop is real before any
  new codec arrives.

### Secondary / deferred
- **Integrate results into `~/adapterbench-paper.tex`** (structure in place, numbers deferred):
  I2P from `results/i2p_hypernoise_v2/` (`scripts/i2p_hypernoise_aggregate.py`); T2L from
  `results/t2p_cond*/` (`scripts/t2p_rigor_aggregate.py`). Frame T2L honestly (conditioning
  emerges with the proper recipe; the control caught an over-claim *and* tracked the real effect).
- **New codec shapes** (activation-steering, IA³, LoKr, FourierFT — the actual "does shape matter"
  payload): defer until the pipeline (#3, #4) is proven on LoRA; then each is a small PR.
- **DDP** for single-trajectory scale — only if 1M-scale runs become routine.

## Handy commands

```bash
# tests (CPU; avoids GPU contention)
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q

# I2P one cell (operating point scale≈4, reg≈0.25; higher scale destroys the image)
.venv/bin/adapterbench i2p-hypernoise --device cuda:1 --reward imagereward --scale 4 \
  --reg-weight 0.25 --steps 3000 --eval-every 500 --n-seeds 2 --output results/i2p_demo
.venv/bin/python scripts/i2p_hypernoise_aggregate.py --out results/i2p_hypernoise_v2

# T2L rigor pilot (mismatched + adversarial control, scale sweep, multi-family, multi-seed)
.venv/bin/adapterbench t2p-sft-pilot --device cuda:1 --all-decontam-tasks \
  --max-descriptions 128 --batch-size 8 --grad-accum-steps 1 --learning-rate 2.5e-5 \
  --warmup-frac 0.1 --eval-tasks arc_easy,arc_challenge,hellaswag,boolq --eval-limit 80 \
  --adversarial-control --seeds 777 --steps 20000 --output results/t2p_demo
.venv/bin/python scripts/t2p_rigor_aggregate.py --results results/t2p_demo/results.jsonl
```

## Gotchas that already bit us (don't repeat)

- **Operating point matters more than "bigger."** I2P LoRA scale has a sharp optimum (~2–4);
  scale 16 destroys the image. T2L conditioning needs 128 descriptions **and** long training
  (8 descriptions / few-hundred steps gives a task-*independent* adapter that fools a loss-only
  view). Always sweep the operating point before concluding a setting "doesn't work."
- **Never conclude from a single seed / small eval.** T2L per-family noise at eval-limit 40 is
  ±0.075; use ≥3 seeds and eval-limit 80.
- **Loss is not capability** — gate and report on matched − control, never loss.
- **Long runs must be checkpointed** (`t2p-sft-pilot --checkpoint-every N`, `d2p-niah` is
  restart-safe, `i2p-hypernoise` checkpoints per cell).
