# AdapterBench

**AdapterBench** benchmarks parameter-efficient fine-tuning (PEFT) adapters —
LoRA, IA3, FourierFT, LoKr, activation steering, and anything you add next — under one
controlled comparison: a shared hypernetwork **generates** the adapter from a
task description, hooks it live into a frozen interpreter's forward pass, and trains
end-to-end on real next-token supervision. No oracle adapters, no adapter-specific
training recipe. A new adapter needs exactly two things:

1. an **output structure** (a differentiable codec — how many numbers, what shape), and
2. a **hook site** (a named linear submodule for weight-space adapters, or the
   whole decoder layer/residual stream for activation-space ones).

The training loop, data pipeline, and evaluator are the same for every adapter
that plugs in this way — that's the whole point.

This generalizes the mechanism behind Sakana AI's
[Text-to-LoRA](https://github.com/SakanaAI/text-to-lora) beyond LoRA itself, and also
hosts earlier, disk-artifact-based comparisons (a released Text-to-LoRA checkpoint, a
self-trained reconstruction hypernetwork) as separate, clearly labeled settings — see
[PROJECT_PLAN.md](PROJECT_PLAN.md) for what's implemented vs. still a gap, real results,
and hard-won gotchas.

## Why "plug-and-play" is the actual claim, not just a tagline

Every adapter added so far — including a brand-new activation-based one
(`ActivationSteeringCodec`) — was wired in with zero changes to the training loop, data
pipeline, or evaluator. Proven end-to-end on a real model
(`adapterbench t2p-sft-pilot`, `Qwen/Qwen3-0.6B`, 8 real training tasks, real held-out
`boolq`/`hellaswag` examples): LoRA, IA3, and activation steering trained and scored
through the identical code path, producing real, differentiated results — including an
adapter (LoRA, in this small pilot) that measurably *hurt* downstream performance
relative to doing nothing. That's a substantive finding a rigged or LoRA-only harness
could never produce.

## Layout

- `src/adapterbench/` — the installable package (`pip install -e .` → `adapterbench` CLI).
- `src/adapterbench/t2p/` — the live end-to-end SFT mechanism: adapter-agnostic
  codecs (`codecs.py`), the hypernetwork shell + generalized hook (`hypernetwork.py`),
  the training loop (`sft_trainer.py`), Lots-of-LoRAs/SNI data loading (`lol_data.py`),
  and the hook-based downstream evaluator (`live_evaluator.py`).
- `configs/setups/`, `configs/adapters/` — declarative manifests for each benchmark
  setting and each adapter; `adapterbench catalog`/`validate`/`matrix` operate on
  these.
- Retired reconstruction-matching code (`t2p/pilot.py`, `t2p/oracle_targets.py`, and
  friends) isn't in the working tree — it's preserved in git history (see the repo's
  first two commits) rather than a live `archive/` directory. See
  [PROJECT_PLAN.md](PROJECT_PLAN.md) for what superseded it and why.

Start with [SETUP.md](SETUP.md) for environment setup, then
[BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md) for the interfaces every setting
implements, then [PROJECT_PLAN.md](PROJECT_PLAN.md) for current status and real results.

## Quickstart

```bash
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
uv run adapterbench doctor --require-cuda
uv run pytest -q
uv run adapterbench validate
uv run adapterbench catalog
```

## Live end-to-end SFT (the current main path)

Train one adapter and inspect its loss curve:

```bash
uv run adapterbench t2p-sft --device cuda:0 --tasks lol_022 \
  --adapter lora --target-modules q_proj,v_proj \
  --steps 60 --output results/t2p_sft/smoke_lol022_lora.json
```

Train and compare several adapters on a shared task split, scored against real
held-out benchmark examples:

```bash
uv run adapterbench t2p-sft-pilot --device cuda:0 --output results/t2p_sft_pilot
```

## Disk-artifact settings (released/self-trained checkpoints)

```bash
uv run adapterbench run \
  --setup text_to_peft_gemma2b_reconstruction \
  --adapter lora_r8_t2l \
  --checkpoint upstream/text-to-lora/trained_t2l/gemma_2b_t2l/hypermod.pt \
  --tasks arc_easy,arc_challenge,boolq,hellaswag,gsm8k \
  --limit 100 \
  --output results/text_to_peft_gemma2b_reconstruction_phase1
```

Add `--evaluator vllm` to score through upstream's own inference backend (vLLM) instead of
plain `transformers`+`peft` — LoRA-only (not FourierFT/IA3/LoKr), but reproduces the
paper's published numbers noticeably more closely; see PROJECT_PLAN.md's Phase 5.5.

Generate an immutable comparison matrix for any registered setup:

```bash
uv run adapterbench matrix \
  --setup text_to_peft_gemma2b_reconstruction \
  --adapters lora_r8_t2l,fourierft_1000,lokr_r8,ia3,prefix_tuning_64 \
  --output results/text_to_peft_gemma2b_reconstruction/trials.json
```
