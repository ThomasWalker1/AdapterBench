"""Cheap, actionable checks run before an expensive canonical reproduction."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUIREMENTS = {
    "t2l": {
        "models": {
            "models--google--gemma-2-2b-it": "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
            "models--Alibaba-NLP--gte-large-en-v1.5": "104333d6af6f97649377c2afbde10a7704870c7b",
        },
        "data": ["data/t2l/hyper_lora_decontam_lol_tasks.yaml", "data/t2l/eval_ds_info.yaml", "data/t2l/tasks"],
        "jobs": ("t2p_train_ddp.py", "t2p_eval_heldout_sni"),
    },
    "d2l": {
        "models": {"models--Qwen--Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca"},
        "data": ["src/adapterbench/t2p/niah_data.py"],
        "jobs": ("d2p-niah",),
    },
}


def preflight(setting: str, devices: str, output: Path, require_cuda: bool = True) -> list[str]:
    if setting not in _REQUIREMENTS:
        raise ValueError(f"unknown setting {setting!r}; choose t2l or d2l")
    messages = []
    requirements = _REQUIREMENTS[setting]
    if not Path(".venv/bin/python").exists():
        raise RuntimeError("missing .venv/bin/python; run the installation commands in SETUP.md")
    for relative in requirements["data"]:
        if not (REPO_ROOT / relative).exists():
            raise RuntimeError(f"missing required data: {relative}; restore the tracked repository data before reproducing")
    cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    missing_models = [f"{name}@{revision}" for name, revision in requirements["models"].items() if not (cache_root / name / "snapshots" / revision).exists()]
    if missing_models:
        raise RuntimeError("missing pinned Hugging Face cache entries: " + ", ".join(missing_models) + ". Download them with network access before running the offline reproduction.")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not importable; reinstall with `uv pip install -e \".[dev]\"`") from error
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU is not available to this process. Check nvidia-smi, container device passthrough, and CUDA-compatible PyTorch before launching training.")
    requested = [item.removeprefix("cuda:") for item in devices.split(",") if item]
    if not requested or any(not item.isdigit() for item in requested):
        raise RuntimeError("devices must be a comma-separated list such as 0,1,2 or cuda:0")
    if len(requested) != len(set(requested)):
        raise RuntimeError("a CUDA device was assigned more than once; use disjoint device lists for concurrent T2L jobs")
    if torch.cuda.is_available() and any(int(item) >= torch.cuda.device_count() for item in requested):
        raise RuntimeError(f"requested CUDA device is outside the visible range 0..{torch.cuda.device_count() - 1}")
    if output.exists():
        messages.append(f"resume layout detected at {output}; existing atomic checkpoints will be reused")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(output.parent, os.W_OK):
            raise RuntimeError(f"output parent is not writable: {output.parent}")
        messages.append(f"output will be created at {output}")
    running = subprocess.run(["ps", "-eo", "args="], check=True, text=True, capture_output=True).stdout
    conflicts = [line.strip() for line in running.splitlines() if any(token in line for token in requirements["jobs"]) and "preflight" not in line]
    if conflicts:
        raise RuntimeError("conflicting reproduction/training job detected; wait for it or choose a different GPU/output: " + conflicts[0])
    messages.append(f"preflight passed for {setting}; requested device(s): {devices}")
    return messages
