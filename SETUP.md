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

For live end-to-end SFT (no upstream clone or gated-model access required,
`Qwen/Qwen3-0.6B` is ungated), see the Quickstart section of [README.md](README.md)
(`t2p-sft`/`t2p-sft-pilot`). The document-conditioning variant
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
PROJECT_PLAN.md's Roadmap for the next scale-up step.

`environment.yml` (conda) is kept only as a legacy/optional reference; it is
not the primary supported path and may drift from `pyproject.toml`.

If `nvidia-smi` shows GPUs but `doctor` reports zero, the process was launched
inside a device-isolated sandbox/container. Check that `/dev/nvidia0` and
`/dev/nvidiactl` exist in that process and launch it with GPU device access.

## Build a comparison matrix

```bash
adapterbench matrix \
  --setup text_to_peft_gemma2b_sft \
  --adapters lora_r8_t2l \
  --output runs/text_to_peft_gemma2b_sft/trials.json
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
  --adapters lora_r8_t2l \
  --device cuda:0 \
  --output results/qwen3_06b_adapter_smoke.json
```

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

