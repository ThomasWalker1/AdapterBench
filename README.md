# AdapterBench

**AdapterBench** tests whether the *shape* of a hypernetwork-generated PEFT adapter
matters — LoRA, FreezeALoRA, LoKr, FourierFT, IA3, activation steering, and anything you
add next. Text-to-LoRA, Program-as-Weights, and Doc-to-LoRA all generate a LoRA
specifically, without testing it against simpler alternatives in this generation setting.
A new adapter here needs exactly two things:

1. an **output structure** (a differentiable codec — how many numbers, what shape), and
2. a **hook site** (a named linear submodule for weight-space adapters, or the
   whole decoder layer/residual stream for activation-space ones).

The training loop, data pipeline, and evaluator are the same for every adapter that plugs
in this way — that's the whole point.

## Two settings

1. **Disk-artifact checkpoint reproduction** — load a released Text-to-LoRA checkpoint
   (Gemma-2-2B, Mistral-7B, Llama-3.1-8B, all from SakanaAI), generate a real LoRA, and
   score it via vLLM exactly the way the paper does. No training of ours — a pure
   reproduction check.
2. **Live end-to-end SFT** — a hypernetwork trained entirely from scratch, hooked
   directly into a real frozen `Qwen3-0.6B` interpreter's forward pass, comparing all six
   representations head to head on real held-out benchmarks.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for architecture, results, and hard-won gotchas.

## Layout

- `src/adapterbench/` — the installable package (`pip install -e .` → `adapterbench` CLI).
- `src/adapterbench/vllm_downstream_evaluator.py`, `text_to_lora_backend.py` — Setting 1:
  generate + score a released checkpoint via vLLM, out-of-process under
  `upstream/text-to-lora/.venv`.
- `src/adapterbench/t2p/` — Setting 2: adapter-agnostic codecs (`codecs.py`), the
  hypernetwork shell + generalized hook (`hypernetwork.py`), the training loop
  (`sft_trainer.py`), Lots-of-LoRAs/SNI data loading (`lol_data.py`), and the hook-based
  downstream evaluator (`live_evaluator.py`).
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

## Setting 1: reproduce a released checkpoint

```bash
uv run adapterbench run \
  --setup text_to_peft_mistral7b_reconstruction_pilot \
  --adapter lora_r8_t2l \
  --checkpoint upstream/text-to-lora/trained_t2l/mistral_7b_t2l/hypermod.pt \
  --chat-template upstream/text-to-lora/chat_templates/mistralai/Mistral-7B-Instruct-v0.2/chat_template.jinja \
  --tasks boolq --limit 1000 --evaluator vllm \
  --output results/vllm_repro/mistral_boolq
```

`--evaluator vllm` scores through upstream's own inference backend instead of plain
`transformers`+`peft` — reproduces the paper's published numbers noticeably more closely
(see PROJECT_PLAN.md).

## Setting 2: live end-to-end SFT

Train one adapter and inspect its loss curve:

```bash
uv run adapterbench t2p-sft --device cuda:0 --tasks lol_022 \
  --adapter lora --target-modules q_proj,v_proj \
  --steps 60 --output results/t2p_sft/smoke_lol022_lora.json
```

Train and compare all six adapters on the full 479-task decontaminated corpus, across
several seeds, scored against real held-out benchmark examples:

```bash
uv run adapterbench t2p-sft-pilot --device cuda:0 \
  --all-decontam-tasks --adapters lora,freeze_a_lora,ia3,lokr,fourierft,activation_steering \
  --seeds 777,778,779 --grad-accum-steps 64 --warmup-frac 0.1 --learning-rate 1e-5 \
  --output results/t2p_sft_full
```

Or sweep step budgets for a single seed, under one persistent optimizer (no Adam-restart
discontinuity at checkpoint boundaries):

```bash
uv run adapterbench t2p-sft-sweep --device cuda:0 \
  --adapters lora,freeze_a_lora,ia3,lokr,fourierft,activation_steering \
  --checkpoint-steps 100,200,400,800,1200 --output results/t2p_sft_sweep
```
