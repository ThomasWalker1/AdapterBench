# Released Text-to-LoRA prompt-conditioning ablation (negative-results verification)

**This is not a benchmark setting.** It is a standalone verification of
[`NEGATIVE_RESULTS.md`](../../../NEGATIVE_RESULTS.md) §1 ("Standard Text-to-LoRA: the
interpreter can already read the task"). Nothing here is registered as an AdapterBench
codec, setup, or leaderboard row, and it does not import the active `adapterbench`
package — the data types it needs are vendored in [`_contracts.py`](_contracts.py). It
exists so the negative result is reproducible from published artifacts without keeping
any retired benchmark infrastructure in the active tree.

## Question

The standard Text-to-LoRA setting hands the hypernetwork a task description and lets it
generate a per-task LoRA. But the held-out evaluation families (ARC, BoolQ, HellaSwag)
are *self-describing*: the interpreter's own prompt already states the task. So does the
correct per-task description actually matter, or is the generated adapter just
generically helpful?

For each family the frozen interpreter sees an identical prompt. Only the text given to
the hypernetwork changes:

| condition | text handed to the hypernetwork | source |
|---|---|---|
| `matched`  | the family's own task description        | released `args.yaml` → `eval_ds_info[family].descriptions[variant]` |
| `shuffled` | a **different** family's real description | rotate families by one |
| `junk`     | meaningless strings                       | released `args.yaml` → `additional_eval_descs` |

If `matched` is not reliably above `shuffled`/`junk`, the per-prompt condition is not
load-bearing. `matched − frozen` still measures the generated adapter's generic help.

## What runs where

Two environments that never share a process, communicating only through saved PEFT
adapters + a JSON manifest:

- **Generation** (`generate_t2l_adapter.py`) runs under the **upstream text-to-lora
  venv**, which has `hyper_llm_modulator` and loads the released `hypermod.pt`. It forces
  eager attention (the released `args.yaml` requests flash-attn, which this minimal venv
  omits; eager is also the only correct path for Gemma-2 soft-capping).
- **Evaluation** (`hf_downstream_evaluator.py`) runs under this repo's `.venv` with plain
  `transformers`+`peft`. Prompt templates, tokenizer setup, and answer-extraction are
  copied verbatim from upstream's own eval (see the module docstrings) so the numbers are
  comparable to the paper's protocol. Also forced to eager.

Attention implementation is held fixed across `matched`/`shuffled`/`junk`/`frozen`, so it
cannot bias the reported deltas.

## Provenance / setup

Released checkpoints (`SakanaAI/text-to-lora`, revision
`6e571eda2188b216f027263cd28c99c0fdcf2fa3`):

```bash
huggingface-cli download SakanaAI/text-to-lora \
  --include 'trained_t2l/gemma_2b_t2l/*' 'trained_t2l/mistral_7b_t2l/*' \
  --local-dir <CKPT_DIR>
```

Upstream generation venv (minimal, no vLLM — only the load/generate path is needed):

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

## Reproduce

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

Swap in `mistral_7b_t2l` / `mistralai/Mistral-7B-Instruct-v0.2` and its chat template for
the second checkpoint. Output: `ablation_summary.json` (per-family accuracies and the
three deltas) plus `conditions_by_kind.json` (the exact texts used).

## Result

Means over ARC-Easy, ARC-Challenge, BoolQ, HellaSwag (200 examples/family, variant 0):

| released checkpoint | matched − frozen | matched − shuffled | matched − junk |
|---|---:|---:|---:|
| `gemma_2b_t2l`  | `+0.011` | `+0.022` | `+0.029` |
| `mistral_7b_t2l` | `+0.130` | `−0.009` | `−0.001` |

On the released checkpoints, `matched` tracks `shuffled` and `junk` within evaluation
noise (`±0.05`), while `matched − frozen` stays positive (hugely so on Mistral). The
generated adapter helps generically, but the correct per-prompt description is not what
makes it help. See [`NEGATIVE_RESULTS.md`](../../../NEGATIVE_RESULTS.md) §1 for the full
write-up and the per-family JSON under
`results/negative_results/t2l_released_prompt_ablation/`.
