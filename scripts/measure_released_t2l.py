"""Measure released Text-to-LoRA adapter generation without downstream decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint
from hyper_llm_modulator.utils import embed_texts, get_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    device_index = device.index or 0
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    torch.cuda.synchronize(device_index)
    load_started = time.perf_counter()
    (
        upstream_args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    ) = load_hypermod_checkpoint(args.checkpoint, device)
    torch.cuda.synchronize(device_index)
    load_seconds = time.perf_counter() - load_started

    layer_indices = torch.arange(len(get_layers(model)), device=device)
    generation_started = time.perf_counter()
    task_embedding = embed_texts(
        [args.description],
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
        device,
    )
    encoded = hypermod.task_encoder(task_embedding)["encoded_task_emb"]
    state = hypermod.gen_lora(layer_indices, encoded)
    torch.cuda.synchronize(device_index)
    generation_seconds = time.perf_counter() - generation_started
    generated_parameters = sum(value.numel() for value in state.values())
    generated_bytes = sum(value.numel() * value.element_size() for value in state.values())
    result = {
        "checkpoint": args.checkpoint,
        "base_model": upstream_args.model_dir,
        "adapter": "lora",
        "description": args.description,
        "model_load_seconds": load_seconds,
        "adapter_generation_seconds": generation_seconds,
        "generated_parameters": generated_parameters,
        "generated_bytes": generated_bytes,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device_index),
        "torch": torch.__version__,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
