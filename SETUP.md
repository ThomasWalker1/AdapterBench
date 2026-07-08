# Environment and execution

## Harness environment

The tested host combination is Python 3.11, PyTorch 2.5.1 with CUDA 12.4,
Transformers 4.57.x, and PEFT 0.19. The NVIDIA 550.90.07 driver on this machine
supports this CUDA build. Do not use the pre-existing Transformers 5.7 build:
its float8 dtype expectations are incompatible with PyTorch 2.5 and PEFT 0.19.

`adapterbench` is managed with `uv`, independent of any pre-existing conda
environment on a given machine, so any researcher can reproduce it from
`pyproject.toml`/`uv.lock` alone:

```bash
cd /home/tw78/AdapterBench
uv venv .venv --python 3.11
uv pip install -e ".[dev]"     # or: uv sync, once uv.lock is committed
uv run adapterbench doctor --require-cuda
uv run pytest -q
uv run adapterbench validate
uv run adapterbench catalog
```

For Setting 2 (live end-to-end SFT — no upstream clone or gated-model access required,
`Qwen/Qwen3-0.6B` is ungated), see the Quickstart section of [README.md](README.md)
(`t2p-sft`/`t2p-sft-pilot`). The document-conditioning variant of Setting 2
(`d2p-sft-pilot`, conditioned on a frozen interpreter's own per-layer activations on a
synthetic needle-in-a-haystack document instead of a pooled task-description embedding
— see `PROJECT_PLAN.md`'s "Document-conditioning variant" section) has the same
no-upstream-clone/no-gated-model requirement, since it also trains from scratch against
`Qwen/Qwen3-0.6B` and generates its own synthetic training data rather than reading
`ctx_to_lora`'s on-disk `ctx_magic_number_*` bins:

```bash
uv run adapterbench d2p-sft-pilot \
  --adapters lora --seeds 777 --steps 20 --grad-accum-steps 1 \
  --context-lengths 256 --num-train-documents 40 --eval-limit 10 \
  --device cuda:0 --output results/d2p_sft_smoke
```

This is the recommended smoke-test scope (one adapter/seed, 20 steps, one context-length
bin) before scaling up `--adapters`/`--seeds`/`--steps`/`--num-train-documents`/
`--context-lengths` the same way `t2p-sft-pilot` scales up its own flags - see
PROJECT_PLAN.md's Roadmap for the next scale-up step. Everything below this point is
specific to Setting 1 (disk-artifact checkpoint reproduction).

`environment.yml` (conda) is kept only as a legacy/optional reference; it is
not the primary supported path and may drift from `pyproject.toml`.

If `nvidia-smi` shows GPUs but `doctor` reports zero, the process was launched
inside a device-isolated sandbox/container. Check that `/dev/nvidia0` and
`/dev/nvidiactl` exist in that process and launch it with GPU device access.

Set a Hugging Face token before downloading gated models:

```bash
huggingface-cli login
export HF_HOME=/path/with/sufficient/cache/space
```

**`google/gemma-2-2b-it` and `meta-llama/Llama-3.1-8B-Instruct` are gated models.** You
must accept each one's license on its Hugging Face model page with the same account as
your token, or every command below that loads that interpreter will fail with a 401/403
from the Hub. Mistral and the other datasets used by `adapterbench run` (ARC, BoolQ,
HellaSwag, GSM8K) are not gated.

## Build a comparison matrix

```bash
adapterbench matrix \
  --setup text_to_lora_gemma2b \
  --adapters lora_r8_t2l,fourierft_1000,lokr_r8,ia3,prefix_tuning_64 \
  --output runs/text_to_lora_gemma2b/trials.json
```

Trial IDs hash the complete setup and adapter manifests, so changing a model
revision, split, objective, or PEFT hyperparameter creates a new ID.

## GPU materialization check

This generates every adapter tensor from one condition, attaches it to the
frozen Qwen3 interpreter, and executes a downstream LM forward pass. The
generator is deliberately untrained; the command validates compatibility and
parameter accounting, not task quality.

```bash
adapterbench peft-smoke \
  --model Qwen/Qwen3-0.6B \
  --adapters lora_r8_t2l,fourierft_1000,lokr_r8,ia3,prefix_tuning_64 \
  --device cuda:0 \
  --output results/qwen3_06b_adapter_smoke.json
```

## Text-to-LoRA reference setup

Keep the upstream environment separate because its checked-in `uv.lock` and
FlashAttention wheel encode older Torch/CUDA assumptions:

```bash
git clone https://github.com/SakanaAI/text-to-lora.git upstream/text-to-lora
cd upstream/text-to-lora
uv venv --python 3.10 --seed
uv sync
uv pip install src/fishfarm
huggingface-cli download SakanaAI/text-to-lora --local-dir . --include 'trained_t2l/*'
```

The reconstruction protocol first trains task-specific oracle adapters with
`scripts/train_lora_baselines.sh`, then invokes `scripts/train_hyper_recon.py`.
The end-to-end SFT protocol uses the three `scripts/train_t2l_*.sh` launchers.
Our two Text-to-LoRA manifests keep those objectives separate.

### Generating and evaluating adapters from a released Text-to-LoRA checkpoint

`hyper_llm_modulator` (upstream's package) pins torch 2.4.0/transformers
4.46.2/peft 0.15.2, which conflicts with `adapterbench`'s own pins. It is
therefore never imported by `adapterbench` directly. `ReleasedTextToLoRABackend`
(`src/adapterbench/text_to_lora_backend.py`) instead shells out to
`scripts/generate_t2l_adapter.py`, run explicitly under
`upstream/text-to-lora/.venv/bin/python`, and only reads back the adapter
directories + `manifest.json` it writes. This is the same pattern
`scripts/measure_released_t2l.py` already used for a one-off Mistral
measurement; `adapterbench run` generalizes it into the benchmark contract
(`src/adapterbench/contracts.py`) for the released Gemma-2-2B checkpoint:

```bash
uv run adapterbench run \
  --setup text_to_peft_gemma2b_reconstruction \
  --adapter lora_r8_t2l \
  --checkpoint upstream/text-to-lora/trained_t2l/gemma_2b_t2l/hypermod.pt \
  --tasks arc_easy \
  --limit 10 \
  --output results/text_to_peft_gemma2b_reconstruction_smoke
```

This is the recommended smoke-test scope (1 task, 10 examples) before scaling
to the full `--tasks arc_easy,arc_challenge,boolq,hellaswag,gsm8k --limit 100`
run. `adapterbench run` writes both the `lora` (released checkpoint) and
`frozen_interpreter` baseline rows to `results.jsonl`/`results.csv` via
`src/adapterbench/reporting.py`. `HFDownstreamEvaluator`
(`src/adapterbench/hf_downstream_evaluator.py`) evaluates with plain
`transformers`+`peft` rather than vLLM (not installed here, and non-LoRA
adapters this benchmark ultimately compares are not vLLM-servable
anyway), replicating upstream's exact tokenizer setup (chat template,
padding/truncation sides) so results are comparable to the paper's own
evaluation formatting even though the underlying serving stack differs.

## Doc-to-LoRA reference setup

Keep the upstream environment separate for the same reason as Text-to-LoRA: its checked-in
`uv.lock`, prebuilt FlashAttention wheel, and pinned `vllm`/`deepspeed`/`transformers`
versions encode assumptions that must not leak into `adapterbench`'s own dependency set.

```bash
git clone https://github.com/SakanaAI/doc-to-lora.git upstream/doc-to-lora
cd upstream/doc-to-lora
uv venv --python 3.10 --seed
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --torch-backend=cu124
uv sync
uv pip install tokenizers==0.21.0
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
huggingface-cli download SakanaAI/doc-to-lora --local-dir trained_d2l --include 'gemma_demo/*'
```

This deliberately skips two steps from upstream's own `install.sh`: the
`flashinfer-python` install (a `vllm` speedup; `ctx_to_lora.eval_utils.run_eval` - the only
entry point `adapterbench` calls - never imports `vllm`) and the SQuAD download +
`build_{drop,pwc,ropes,squad}_compact.py` data-prep steps (needed only for D2L's "main
experiment" QA training/eval, not for the NIAH (`ctx_magic_number_*`) protocol this setup
uses). Add them back if you need D2L's own main-experiment scripts to run unmodified.

`SakanaAI/doc-to-lora` ships four checkpoint subfolders (`gemma_2b_d2l`, `gemma_demo`,
`mistral_7b_d2l`, `qwen_4b_d2l`), all gated behind the same HF license-acceptance flow as
`google/gemma-2-2b-it` (see above) - accept `SakanaAI/doc-to-lora`'s license on the same
account as your token. **All four are QA-trained** (their `args.yaml`'s `train_ds_names`
are self-generated QA data plus `pwc`/`squad`/`ropes`/`drop_compact` - never
`ctx_magic_number`); there is no released NIAH-specific D2L checkpoint. `gemma_demo`
(used here, ~1.3GB) is Gemma-2-2B-IT trained for 80k steps and is the checkpoint
README.md's own Python API example uses; `gemma_2b_d2l` is the same base model trained
for 20k steps under the same naming convention as `mistral_7b_d2l`/`qwen_4b_d2l` (D2L's
"main experiment" checkpoints). Evaluating any of them on `ctx_magic_number_*` is
therefore a generalization test, not upstream's own reported protocol for that checkpoint.

### Generating the NIAH (`ctx_magic_number`) data

`ctx_to_lora`'s dataset loader (`ctx_to_lora.data.definitions.DS_KWARGS`) expects
`data/raw_datasets/ctx_magic_number_<lo>_<hi>/{train,val,test}.jsonl` to already exist -
nothing downloads or generates them lazily. Generate just the bins you need
(`--only-first-n-bins` selects a prefix of the bin list in
`data/generate_ctx_magic_number.py`, in the order `(32,128), (128,256), (256,512),
(512,1024), (32,1024), (1024,2048), ...`; `--base-samples-per-bin` shrinks the unused
`train.jsonl` this also writes - val/test are always fixed at 1000 examples/bin
regardless):

```bash
cd upstream/doc-to-lora
uv run data/generate_ctx_magic_number.py --only-first-n-bins 5 --base-samples-per-bin 100
```

### Evaluating a released Doc-to-LoRA checkpoint on NIAH

`ctx_to_lora` (upstream's package) pins torch 2.6.0/transformers 4.51.3/peft 0.15.2/
vllm 0.8.5.post1, which conflicts with `adapterbench`'s own pins. It is therefore never
imported by `adapterbench` directly. Unlike Text-to-LoRA, D2L's own eval API
(`ctx_to_lora.eval_utils.evaluate`, wrapped by `run_eval`) bakes per-document adapter
generation and downstream scoring into one call - there is no separate "generate a
portable adapter" step to bridge, so there is no `HypernetworkBackend`/
`DownstreamEvaluator` split for this setting; see
`src/adapterbench/doc_to_lora_backend.py`'s module docstring for the full reasoning.
`scripts/run_d2l_eval.py`, run explicitly under `upstream/doc-to-lora/.venv/bin/python`,
imports `ctx_to_lora.eval_utils` directly (safe here - that import happens inside the
upstream venv's own process, never inside `adapterbench`'s) and writes a `manifest.json`
`adapterbench.doc_to_lora_backend.ReleasedDocToLoRANIAHEvaluator` reads back.

```bash
uv run adapterbench run-d2l-niah \
  --setup doc_to_peft_gemma2b_reconstruction \
  --checkpoint upstream/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --datasets ctx_magic_number_32_1024 \
  --limit 10 \
  --baseline \
  --output results/doc_to_peft_gemma2b_reconstruction_smoke
```

This is the recommended smoke-test scope (1 dataset bin, 10 examples, `--baseline` to also
confirm the frozen no-context interpreter scores near zero) before scaling to `--datasets`
covering more of `configs/setups/doc_to_peft_gemma2b_reconstruction.yaml`'s full
`test_splits` list (D2L's own NIAH eval sweep, `upstream/doc-to-lora/scripts/niah/2-eval.sh`)
at `--limit 1000`. Every larger bin needs its own `generate_ctx_magic_number.py` run first
(see above). Scoring is D2L's own: word-level ROUGE-L F1
(`ctx_to_lora.metrics.compute_rouge`) for `ctx_magic_number_*` specifically, because that
dataset family is not in `ctx_to_lora.data.definitions.CLOSED_QA_DATASETS` (which would
otherwise route it through the QA-F1 scorer used for squad/drop/ropes) - not a choice made
by `adapterbench`, faithfully reproduced from upstream.

## Eight-GPU launch policy

Use one process per A100 and BF16. Each setup manifest records its launcher and
process count. Typical launch forms are:

```bash
torchrun --standalone --nproc_per_node=8 TRAINING_ENTRYPOINT --config TRIAL_JSON
accelerate launch --num_processes 8 TRAINING_ENTRYPOINT --config TRIAL_JSON
```

Write one result directory per immutable trial ID. Never average task scores across
different setups together; compare adapter rankings and failure modes within each
setup.

