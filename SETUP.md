# Environment and execution

AdapterBench supports exactly two language settings:

- T2L task-description conditioning;
- D2L document conditioning.

## Environment

The tested host combination is Python 3.11, PyTorch 2.5.1 with CUDA 12.4,
Transformers 4.57.x, and PEFT 0.19. The project uses `uv` and is reproducible from
`pyproject.toml` plus `uv.lock`.

```bash
cd /home/tw78/AdapterBench
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
uv run adapterbench doctor --require-cuda
uv run adapterbench validate
uv run adapterbench catalog
uv run adapterbench results validate
uv run adapterbench results check
uv run pytest -q
```

If `nvidia-smi` sees GPUs but `doctor` does not, the process is inside a
device-isolated sandbox or container. Confirm that `/dev/nvidia0` and
`/dev/nvidiactl` are visible.

`environment.yml` is a legacy optional reference. The `uv` environment is authoritative.

## T2L

T2L data is vendored under `data/t2l/`: per-task metadata, the 479-task decontaminated
training split, and held-out task descriptions. No upstream clone is required.

Small plumbing run:

```bash
uv run adapterbench t2p-sft \
  --device cuda:0 \
  --tasks lol_022 \
  --adapter lora \
  --steps 60 \
  --output results/t2p_sft/smoke_lol022_lora.json
```

Canonical one-seed baseline:

```bash
bash scripts/reproduce/task_t2l_lora.sh 777 0,1,2,3 4,5,6,7
```

Full reference reproduction (three sequential restart-safe seeds, then aggregation):

```bash
bash scripts/reproduce/task_t2l_lora_all.sh 0,1,2,3 4,5,6,7
```

The held-out-SNI evaluations use local cached model/data files and must run with
`HF_HUB_OFFLINE=1`; the reproduction script sets it.

## D2L

D2L creates deterministic synthetic needle-in-a-haystack examples locally and uses
realistic cached prose as distractors. No Doc-to-LoRA clone is required.

Small plumbing run:

```bash
uv run adapterbench d2p-niah \
  --adapters lora \
  --steps 20 \
  --grad-accum-steps 1 \
  --context-lengths 256 \
  --num-train-documents 40 \
  --eval-limit 10 \
  --device cuda:0 \
  --output results/d2p_niah_smoke
```

Canonical baseline:

```bash
bash scripts/reproduce/document_niah_lora.sh cuda:0
```

Both canonical scripts first check the installed environment, CUDA visibility, pinned
cached models, required data, output layout, and conflicting training jobs. The full
training runs require a GPU; no wall-clock estimates are stated because the repository
does not contain comparable timing measurements.

Scale locator:

```bash
bash scripts/d2l_scale_sweep.sh
```

## Manifest and GPU checks

Build a T2L trial matrix:

```bash
adapterbench matrix \
  --setup text_to_peft_gemma2b_sft \
  --adapters lora_r8 \
  --output runs/text_to_peft_gemma2b_sft/trials.json
```

Materialize a generated LoRA and execute one frozen-interpreter forward pass:

```bash
adapterbench peft-smoke \
  --model Qwen/Qwen3-0.6B \
  --adapters lora_r8 \
  --device cuda:0 \
  --output results/qwen3_06b_adapter_smoke.json
```

Trial IDs hash the complete setup and adapter manifests. Changing a model revision,
split, objective, or PEFT parameter creates a new ID.

For multi-GPU runs use one process per A100 and BF16. Write one result directory per
immutable trial ID, and never pool absolute task scores across T2L and D2L.
