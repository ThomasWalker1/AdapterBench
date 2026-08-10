# AdapterBench

**AdapterBench** tests whether the *shape* of a hypernetwork-generated PEFT adapter
matters. Recent systems such as Text-to-LoRA and Doc-to-LoRA generate LoRA specifically,
without testing it against alternative representations under the same generation
protocol. AdapterBench holds the hypernetwork, training loop, and evaluator fixed and
varies only the generated representation. A new adapter here needs exactly two things:

1. an **output structure** (a differentiable codec — how many numbers, what shape), and
2. a **hook site** (a named linear submodule for weight-space adapters, or the
   whole decoder layer/residual stream for activation-space ones).

The training loop, data pipeline, and evaluator are the same for every adapter that plugs
in this way — that's the whole point.

> **Current state:** LoRA validates the two genuinely conditioned settings:
> task-description conditioning (T2A) and document conditioning (D2A). (IA)³, LoKr, and
> FourierFT have complete two-setting benchmark rows alongside LoRA; LoHa is registered
> and its protocol evaluation is in flight. **Steering** — the first activation-space
> codec, hooked at the residual stream (`"block"`) rather than a projection — is the
> newest registered shape; its two-setting autoresearch evaluation is pending. New
> shapes are added one at a time as a codec + registration entry + manifest, then
> evaluated with both setting CLIs and recorded in the per-setting leaderboards.
>
> **Current phase: codec exploration.** The core benchmark and LoRA reference results
> are implemented; the remaining research work is to compare the registered shapes under
> the same substrate.

## The setting

**Live end-to-end SFT** — a hypernetwork trained entirely from scratch, hooked directly
into a frozen interpreter's forward pass and scored on held-out behavior. T2A uses
`gemma-2-2b`; D2A uses `Qwen3-0.6B`. The baseline runs the LoRA codec, and the pipeline
adds shapes to compare head to head over time.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for architecture, active results, and hard-won
gotchas. [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) collates the investigated settings
that failed condition controls or behavioral capacity gates; they remain scientific
results without remaining active benchmark infrastructure.

Committed headline records live in [`canonical_results/`](canonical_results/). They, not
the rendered tables, are the source of truth; run `uv run adapterbench results check` to
verify leaderboards against them, then `uv run adapterbench website build` to refresh the
public site. The paper draft is maintained separately and is not part of the drift check.

## Layout

- `src/adapterbench/` — the installable package (`pip install -e .` → `adapterbench` CLI).
- `src/adapterbench/t2a/` — adapter-agnostic codecs (`codecs.py`), the hypernetwork shell
  + generalized hook (`hypernetwork.py`), the training loop (`sft_trainer.py`),
  Lots-of-LoRAs/SNI data loading (`lol_data.py`), and the hook-based downstream evaluator
  (`live_evaluator.py`).
- `leaderboards/` — the benchmark results and exact reproduction commands, one file per
  setting.
- `website/` — static benchmark site; build with `uv run adapterbench website build` (see
  [website/INTEGRATION.md](website/INTEGRATION.md) for personal-site deployment).
- `configs/setups/`, `configs/adapters/` — declarative setup and codec metadata.

Start with [SETUP.md](SETUP.md) for environment setup, then **[GUIDE.md](GUIDE.md)** for
the benchmark's active settings, metrics and controls, how to add a codec, and how to run
and reproduce each setting. **[CONTRIBUTING.md](CONTRIBUTING.md)** is the PR workflow for
landing a new codec and leaderboard row. [BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md) gives the
interface contract every setting implements, and [PROJECT_PLAN.md](PROJECT_PLAN.md) tracks
current status and results.

## Quickstart

```bash
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
uv run adapterbench doctor --require-cuda
uv run pytest -q
uv run adapterbench validate
uv run adapterbench catalog
uv run adapterbench results validate
uv run adapterbench results check
uv run adapterbench website build   # regenerate website/ from canonical_results/
```

## Live end-to-end SFT

Train one adapter and inspect its loss curve:

```bash
uv run adapterbench t2a-sft --device cuda:0 --tasks lol_022 \
  --adapter lora --target-modules q_proj,v_proj \
  --steps 60 --output results/t2a_sft/smoke_lol022_lora.json
```

Train the baseline on the full 479-task decontaminated corpus, across several seeds,
scored against real held-out benchmark examples (`--adapters` currently accepts `lora`;
newly committed codecs extend the list):

```bash
uv run adapterbench t2a-sft-pilot --device cuda:0 \
  --all-decontam-tasks --adapters lora \
  --seeds 777,778,779 --grad-accum-steps 64 --warmup-frac 0.1 --learning-rate 1e-5 \
  --output results/t2a_sft_full
```

Or sweep step budgets for a single seed, under one persistent optimizer (no Adam-restart
discontinuity at checkpoint boundaries):

```bash
uv run adapterbench t2a-sft-sweep --device cuda:0 \
  --adapters lora \
  --checkpoint-steps 100,200,400,800,1200 --output results/t2a_sft_sweep
```

The validated T2A and D2A recipes, scale sweeps, controls, and longer reproduction
commands are recorded in [PROJECT_PLAN.md](PROJECT_PLAN.md) and
[leaderboards/](leaderboards/README.md).
