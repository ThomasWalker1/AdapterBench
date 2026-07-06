# Environment and execution

## Harness environment

The tested host combination is Python 3.11, PyTorch 2.5.1 with CUDA 12.4,
Transformers 4.57.x, and PEFT 0.19. The NVIDIA 550.90.07 driver on this machine
supports this CUDA build. Do not use the pre-existing Transformers 5.7 build:
its float8 dtype expectations are incompatible with PyTorch 2.5 and PEFT 0.19.

`peft_hnet` is managed with `uv`, independent of any pre-existing conda
environment on a given machine, so any researcher can reproduce it from
`pyproject.toml`/`uv.lock` alone:

```bash
cd /home/tw78/AdapterBench
uv venv .venv --python 3.11
uv pip install -e ".[dev]"     # or: uv sync, once uv.lock is committed
uv run peft-hnet doctor --require-cuda
uv run pytest -q
uv run peft-hnet validate
uv run peft-hnet catalog
```

For the live end-to-end SFT path (the current main benchmark mechanism — no upstream
clone or gated-model access required, `Qwen/Qwen3-0.6B` is ungated), see the Quickstart
section of [README.md](README.md) (`t2p-sft`/`t2p-sft-pilot`). Everything below this
point is specific to the disk-artifact settings (released/self-trained checkpoints).

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

**`google/gemma-2-2b-it` is a gated model.** You must accept Google's license
on its Hugging Face model page with the same account as your token, or every
command below that loads the interpreter will fail with a 401/403 from the
Hub. Mistral and the other datasets used by `peft-hnet run` (ARC, BoolQ,
HellaSwag, GSM8K) are not gated.

## Build a comparison matrix

```bash
peft-hnet matrix \
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
peft-hnet peft-smoke \
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
4.46.2/peft 0.15.2, which conflicts with `peft_hnet`'s own pins. It is
therefore never imported by `peft_hnet` directly. `ReleasedTextToLoRABackend`
(`src/peft_hnet/text_to_lora_backend.py`) instead shells out to
`scripts/generate_t2l_adapter.py`, run explicitly under
`upstream/text-to-lora/.venv/bin/python`, and only reads back the adapter
directories + `manifest.json` it writes. This is the same pattern
`scripts/measure_released_t2l.py` already used for a one-off Mistral
measurement; `peft-hnet run` generalizes it into the benchmark contract
(`src/peft_hnet/contracts.py`) for the released Gemma-2-2B checkpoint:

```bash
uv run peft-hnet run \
  --setup text_to_peft_gemma2b_reconstruction \
  --adapter lora_r8_t2l \
  --checkpoint upstream/text-to-lora/trained_t2l/gemma_2b_t2l/hypermod.pt \
  --tasks arc_easy \
  --limit 10 \
  --output results/text_to_peft_gemma2b_reconstruction_smoke
```

This is the recommended smoke-test scope (1 task, 10 examples) before scaling
to the full `--tasks arc_easy,arc_challenge,boolq,hellaswag,gsm8k --limit 100`
run. `peft-hnet run` writes both the `lora` (released checkpoint) and
`frozen_interpreter` baseline rows to `results.jsonl`/`results.csv` via
`src/peft_hnet/reporting.py`. `HFDownstreamEvaluator`
(`src/peft_hnet/hf_downstream_evaluator.py`) evaluates with plain
`transformers`+`peft` rather than vLLM (not installed here, and non-LoRA
representations this benchmark ultimately compares are not vLLM-servable
anyway), replicating upstream's exact tokenizer setup (chat template,
padding/truncation sides) so results are comparable to the paper's own
evaluation formatting even though the underlying serving stack differs.

## PAW reference setup

The released PAW checkpoint and verified evaluation data are public:

```bash
huggingface-cli download programasweights/paw-4b-qwen3-0.6b \
  --local-dir upstream/paw-4b-qwen3-0.6b
huggingface-cli download yuntian-deng/fuzzy_bench_verified \
  --repo-type dataset --local-dir data/fuzzy_bench_verified
```

The checkpoint metadata pins Qwen3-0.6B, rank 64, alpha 16, 64 bases, and all
attention/MLP projection targets. The public Python repository is an inference
SDK; until compiler-training code/data are released, alternative-PEFT PAW
training remains a specified experiment rather than a reproducible run.

## Eight-GPU launch policy

Use one process per A100 and BF16. Each setup manifest records its launcher and
process count. Typical launch forms are:

```bash
torchrun --standalone --nproc_per_node=8 TRAINING_ENTRYPOINT --config TRIAL_JSON
accelerate launch --num_processes 8 TRAINING_ENTRYPOINT --config TRIAL_JSON
```

Write one result directory per immutable trial ID. Never average PAW and
Text-to-LoRA task scores together; compare representation rankings and failure
modes within each setup.

