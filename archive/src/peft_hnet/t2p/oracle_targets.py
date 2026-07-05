"""Build dense ΔW = B @ A reconstruction targets from real PEFT LoRA safetensors.

Port of the key-parsing + ``bmm(B, A)`` logic in
``hyper_llm_modulator/data.py::get_recon_train_data`` (upstream Text-to-LoRA), scoped
down to what this benchmark needs: a task's saved ``adapter_model.safetensors`` in, a
``{module_name: (num_layers, out_features, in_features)}`` dict out. Ported rather than
imported because ``hyper_llm_modulator`` pins an incompatible torch/transformers/peft
stack (see ``text_to_lora_backend.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file
from torch import Tensor

# Mistral-7B-Instruct-v0.2's LoRA target modules, as trained by Phase 2's oracle LoRAs
# (`scripts/train_lora_baselines.py`, r=8 on q_proj/v_proj) — confirmed against the real
# `adapter_model.safetensors` headers, not assumed from the paper/README.
MISTRAL7B_LORA_MODULE_SHAPES: dict[str, tuple[int, int]] = {"q_proj": (4096, 4096), "v_proj": (4096, 1024)}
MISTRAL7B_NUM_LAYERS = 32


def find_oracle_adapter_paths(oracle_root: Path, task_ids: Sequence[str]) -> dict[str, Path]:
    """Map each task id to its single-task oracle LoRA's ``adapter_model.safetensors``.

    Oracle LoRA runs are saved under timestamped directory names
    (``train_outputs/sft/oracle_lora/<timestamp>_<hash>/``), so the task→directory
    mapping isn't derivable from the path alone — resolved here by reading each run's
    ``args.yaml::train_ds_names``.
    """
    remaining = set(task_ids)
    found: dict[str, Path] = {}
    for run_dir in sorted(oracle_root.iterdir()):
        args_path = run_dir / "args.yaml"
        if not args_path.is_file():
            continue
        args = yaml.safe_load(args_path.read_text())
        train_ds_names = args.get("train_ds_names") or []
        if len(train_ds_names) != 1:
            continue
        task_id = train_ds_names[0]
        if task_id in remaining:
            found[task_id] = run_dir / "adapter_model.safetensors"
            remaining.discard(task_id)
    if remaining:
        raise FileNotFoundError(f"no single-task oracle LoRA found under {oracle_root} for tasks: {sorted(remaining)}")
    return found


def load_lora_delta_targets(
    adapter_path: Path, target_modules: Sequence[str], num_layers: int
) -> dict[str, Tensor]:
    """Load one task's oracle LoRA and return its per-module dense ΔW stack.

    ``adapter_path`` is a PEFT ``adapter_model.safetensors`` file with keys of the form
    ``base_model.model.model.layers.{i}.self_attn.{module}.lora_{A,B}.weight``. Returns
    ``{module_name: Tensor(num_layers, out_features, in_features)}``, i.e. the same shape
    and orientation ``LinearUpdateCodec.dense_delta`` produces.
    """
    state_dict = load_file(str(adapter_path))
    lora_a: dict[str, list[Tensor | None]] = {name: [None] * num_layers for name in target_modules}
    lora_b: dict[str, list[Tensor | None]] = {name: [None] * num_layers for name in target_modules}

    for key, value in state_dict.items():
        for module_name in target_modules:
            if f".{module_name}." not in key:
                continue
            layer_index = int(key.split("layers.")[-1].split(".")[0])
            if layer_index >= num_layers:
                continue
            if "lora_A" in key:
                lora_a[module_name][layer_index] = value
            elif "lora_B" in key:
                lora_b[module_name][layer_index] = value

    targets: dict[str, Tensor] = {}
    for module_name in target_modules:
        missing = [i for i, v in enumerate(lora_a[module_name]) if v is None]
        if missing:
            raise ValueError(f"{adapter_path}: missing lora_A for {module_name} at layers {missing}")
        a_stack = torch.stack(lora_a[module_name], dim=0)
        b_stack = torch.stack(lora_b[module_name], dim=0)
        targets[module_name] = torch.bmm(b_stack, a_stack).to(torch.float32)
    return targets


def build_task_oracle_targets(
    task_adapter_paths: Mapping[str, Path], target_modules: Sequence[str], num_layers: int
) -> dict[str, dict[str, Tensor]]:
    """Load dense ΔW targets for every task in ``task_adapter_paths``."""
    return {
        task_id: load_lora_delta_targets(adapter_path, target_modules, num_layers)
        for task_id, adapter_path in task_adapter_paths.items()
    }
