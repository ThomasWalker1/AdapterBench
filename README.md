# AdapterBench

**AdapterBench** tests whether the *shape* of a hypernetwork-generated PEFT adapter
matters. Text-to-LoRA, Program-as-Weights, and Doc-to-LoRA all generate a LoRA
specifically, without testing it against alternative representations in this generation
setting. AdapterBench holds the hypernetwork, training loop, and evaluator fixed and
varies only the generated representation. A new adapter here needs exactly two things:

1. an **output structure** (a differentiable codec — how many numbers, what shape), and
2. a **hook site** (a named linear submodule for weight-space adapters, or the
   whole decoder layer/residual stream for activation-space ones).

The training loop, data pipeline, and evaluator are the same for every adapter that plugs
in this way — that's the whole point.

> **Current state: LoRA is the only codec on `main`.** It's the baseline used to develop
> and validate the benchmark's evaluation pipeline. Every other shape is (re)introduced
> one at a time through the **autoresearch git-merge pipeline** — propose a codec on a
> branch, gate the PR on correctness (not on winning), merge, then evaluate and append to
> the leaderboard. See PROJECT_PLAN.md § "Git-native benchmark" and § "LoRA-only baseline".

## The setting

**Live end-to-end SFT** — a hypernetwork trained entirely from scratch, hooked directly
into a real frozen `Qwen3-0.6B` interpreter's forward pass, scored on real held-out
benchmarks. The baseline runs the LoRA codec; the pipeline adds shapes to compare head to
head over time (see the note above).

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for architecture, results, and hard-won gotchas.

## Layout

- `src/adapterbench/` — the installable package (`pip install -e .` → `adapterbench` CLI).
- `src/adapterbench/t2p/` — adapter-agnostic codecs (`codecs.py`), the hypernetwork shell
  + generalized hook (`hypernetwork.py`), the training loop (`sft_trainer.py`),
  Lots-of-LoRAs/SNI data loading (`lol_data.py`), and the hook-based downstream evaluator
  (`live_evaluator.py`).
- `configs/setups/`, `configs/adapters/` — declarative manifests; `adapterbench
  catalog`/`validate`/`matrix` operate on these.

Start with [SETUP.md](SETUP.md) for environment setup, then
[BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md) for the interfaces every setting
implements, then [PROJECT_PLAN.md](PROJECT_PLAN.md) for current status and results.

## Quickstart

```bash
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
uv run adapterbench doctor --require-cuda
uv run pytest -q
uv run adapterbench validate
uv run adapterbench catalog
```

## Live end-to-end SFT

Train one adapter and inspect its loss curve:

```bash
uv run adapterbench t2p-sft --device cuda:0 --tasks lol_022 \
  --adapter lora --target-modules q_proj,v_proj \
  --steps 60 --output results/t2p_sft/smoke_lol022_lora.json
```

Train the baseline on the full 479-task decontaminated corpus, across several seeds,
scored against real held-out benchmark examples (`--adapters` currently accepts `lora`;
pipeline-added shapes extend the list):

```bash
uv run adapterbench t2p-sft-pilot --device cuda:0 \
  --all-decontam-tasks --adapters lora \
  --seeds 777,778,779 --grad-accum-steps 64 --warmup-frac 0.1 --learning-rate 1e-5 \
  --output results/t2p_sft_full
```

Or sweep step budgets for a single seed, under one persistent optimizer (no Adam-restart
discontinuity at checkpoint boundaries):

```bash
uv run adapterbench t2p-sft-sweep --device cuda:0 \
  --adapters lora \
  --checkpoint-steps 100,200,400,800,1200 --output results/t2p_sft_sweep
```
