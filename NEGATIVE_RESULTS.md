# Negative result: standard Text-to-LoRA does not condition

AdapterBench counts a setting as genuinely adaptive only when changing the condition
available to the adapter generator changes held-out behavior while the frozen
interpreter's input is held fixed. The decisive quantity is `matched − control`, where the
control substitutes a shuffled, junk, or wrong condition — a raw gain over the frozen model
is not enough. This document records the one language setting that motivated that rule:
**standard, input-visible Text-to-LoRA**, which produces helpful adapters but does not pass
a condition control.

## The interpreter can already read the task

Prior evaluations of generated adapters overwhelmingly report a raw gain over the frozen
model, or a gain over a single static adapter, and read that as *adaptation*. That conflates
two effects a benchmark must separate: **generic** adapter help, which any well-optimized
adapter supplies regardless of the condition, and **condition-specific** adaptation, the
only effect that justifies a hypernetwork at all.

Standard Text-to-LoRA cannot separate them, because its SFT template gives the frozen
interpreter `{task_definition}\n\n{problem}` while *also* handing a description of the same
task to the hypernetwork. The condition reaches the interpreter through two paths, so the
adapter is free to be redundant:

```text
task description ──> hypernetwork ──> generated LoRA ──> interpreter
task definition  ──────────────────────────────────────> interpreter
```

When we hold the shipped recipe fixed and vary **only** the hypernetwork's condition, the
apparent advantage collapses to within evaluation noise.

**Our diagnostic runs.** Holding the shipped DDP recipe fixed and changing only the base
interpreter, the mean matched-minus-junk accuracy was:

| interpreter | matched − junk |
|---|---:|
| Qwen3-0.6B | `+0.003` |
| Gemma-2-2B | `+0.022` |
| Mistral-7B | `−0.019` |

All three are within the approximately `±0.05` evaluation noise. An earlier under-powered
three-seed run made the issue especially clear: matched minus mismatched accuracy was
`−0.017`, `−0.021`, `−0.021`, and `−0.029` on ARC-Easy, ARC-Challenge, HellaSwag, and BoolQ.
The adapters improved raw performance; a wrong or meaningless description improved it just
as much.

**Verified on Sakana's released checkpoints.** To confirm this is a property of the setting
and not of our reimplementation, we ran the same ablation directly on Sakana's *released*
Text-to-LoRA hypernetworks. Only the text handed to the hypernetwork changed — `matched`
(the family's own task description), `shuffled` (a different family's real description), or
`junk` (meaningless strings) — while the frozen interpreter saw an identical prompt in every
case:

| released checkpoint | matched − frozen | matched − shuffled | matched − junk |
|---|---:|---:|---:|
| `gemma_2b_t2l` (google/gemma-2-2b-it) | `+0.011` | `+0.022` | `+0.029` |
| `mistral_7b_t2l` (mistralai/Mistral-7B-Instruct-v0.2) | `+0.130` | `−0.009` | `−0.001` |

Means over ARC-Easy, ARC-Challenge, BoolQ, and HellaSwag (200 examples/family). On Mistral
the generated adapter is genuinely and generically helpful (`matched − frozen = +0.130`),
yet giving it the *correct* per-task description buys nothing over a wrong task's description
or meaningless text. On Gemma every gap is within the same `±0.05` band. The published
checkpoints reproduce the diagnostic: the adapter helps, the per-prompt condition does not —
and the weaker matched-versus-static control used in prior work would have credited the gap
to adaptation.

## The active AdapterBench setting, by contrast

This is a negative result about the **standard input-visible setting**, not the active
AdapterBench T2A setting. AdapterBench strips the task definition from the interpreter input
so the task survives *only* through the generated adapter, and reports the harder-to-game
`matched − static` control against a same-shape static multi-task LoRA. That redesigned
setting passes: `matched − static = −0.571 ± 0.045` nats of held-out cross-entropy over three
seeds (see `leaderboards/task_conditioned_t2a.md`).

## Reproducing the released-checkpoint verification

The verification lives entirely outside the benchmark code, as a standalone record under
`scripts/negative_results/t2l_released_prompt_ablation/`. It is not a registered setting and
does not import the active `adapterbench` package. Its `README.md` has the full method; the
essential steps are below.

**1. Released checkpoints** (`SakanaAI/text-to-lora`, revision
`6e571eda2188b216f027263cd28c99c0fdcf2fa3`):

```bash
huggingface-cli download SakanaAI/text-to-lora \
  --include 'trained_t2l/gemma_2b_t2l/*' 'trained_t2l/mistral_7b_t2l/*' \
  --local-dir <CKPT_DIR>
```

**2. Upstream generation venv** (minimal — only the load/generate path is needed, no vLLM;
generation must run under `hyper_llm_modulator`, which pins an older torch/transformers stack
incompatible with `adapterbench`'s):

```bash
git clone --depth 1 https://github.com/SakanaAI/text-to-lora.git <UPSTREAM>
cd <UPSTREAM>
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python transformers==4.46.2 peft accelerate datasets \
  safetensors pyyaml numpy wandb einops sentencepiece protobuf inflect rouge-score torchmetrics
uv pip install --python .venv/bin/python -e . --no-deps
uv pip install --python .venv/bin/python -e src/fishfarm --no-deps
```

**3. Run the ablation** (once per checkpoint; generation uses the upstream venv, evaluation
uses this repo's `.venv`):

```bash
.venv/bin/python scripts/negative_results/t2l_released_prompt_ablation/run_ablation.py \
  --checkpoint <CKPT_DIR>/trained_t2l/gemma_2b_t2l/hypermod.pt \
  --interpreter google/gemma-2-2b-it \
  --chat-template <UPSTREAM>/chat_templates/google/gemma-2-2b-it/chat_template.jinja \
  --families arc_easy,arc_challenge,boolq,hellaswag --limit 200 \
  --gen-python <UPSTREAM>/.venv/bin/python --gen-cwd <UPSTREAM> \
  --gen-device cuda:0 --eval-device cuda:0 \
  --output results/negative_results/t2l_released_prompt_ablation/gemma_2b
```

Swap in `mistral_7b_t2l` / `mistralai/Mistral-7B-Instruct-v0.2` and its chat template for the
second checkpoint. Each run writes `ablation_summary.json` (per-family accuracies and the
three deltas) and `conditions_by_kind.json` (the exact matched/shuffled/junk texts used).

## Provenance

- **Released-checkpoint verification:**
  `results/negative_results/t2l_released_prompt_ablation/{gemma_2b,mistral_7b}/ablation_summary.json`.
  The disk-artifact "setting 1" code the standalone scripts were rebuilt from was removed
  from the active tree in git commits `a8867de` / `64de45e` and is recoverable from
  `a8867de^`.
- **Diagnostic training runs:** `results/repro/t2a_base_diag/`, `results/t2a_cond_ddp/`, and
  `results/_archive/t2a_cond_long/`.
