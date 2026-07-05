"""Generate one or more Text-to-LoRA adapters from a trained hypernetwork checkpoint.

This script must run under the upstream Text-to-LoRA environment
(``upstream/text-to-lora/.venv/bin/python``), NOT the main ``peft_hnet`` uv venv:
``hyper_llm_modulator`` pins an older torch/transformers/peft stack that conflicts
with ``peft_hnet``'s own dependencies. ``peft_hnet.text_to_lora_backend`` invokes this
script via subprocess and only ever reads back the adapter directories + JSON manifest
it writes, so the two environments never need to share a process.

Loads the base model + hypernetwork once, then generates one saved PEFT LoRA adapter
per condition (avoids repeated ~2B-parameter model loads for multi-task batches).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint, save_lora
from hyper_llm_modulator.utils import embed_texts, get_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to hypermod.pt")
    parser.add_argument(
        "--conditions-json",
        required=True,
        help="JSON file mapping task_id -> condition text",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    conditions = json.loads(Path(args.conditions_json).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    device_index = device.index or 0
    torch.cuda.set_device(device_index)

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

    manifest: dict[str, dict] = {
        "checkpoint": args.checkpoint,
        "base_model": upstream_args.model_dir,
        "representation": "lora",
        "model_load_seconds": load_seconds,
        "torch": torch.__version__,
        "adapters": {},
    }

    for task_id, condition in conditions.items():
        torch.cuda.reset_peak_memory_stats(device_index)
        torch.cuda.synchronize(device_index)
        generation_started = time.perf_counter()
        task_embedding = embed_texts(
            [condition],
            emb_model,
            emb_tokenizer,
            task_desc_format_fn,
            pooling_fn,
            device,
        )
        encoded = hypermod.task_encoder(task_embedding)["encoded_task_emb"]
        lora_state_dict = hypermod.gen_lora(layer_indices, encoded)
        torch.cuda.synchronize(device_index)
        generation_seconds = time.perf_counter() - generation_started

        adapter_dir = output_dir / task_id
        save_lora(lora_state_dict, hypermod.peft_config, str(adapter_dir))

        generated_parameters = sum(v.numel() for v in lora_state_dict.values())
        generated_bytes = sum(
            v.numel() * v.element_size() for v in lora_state_dict.values()
        )
        manifest["adapters"][task_id] = {
            "condition": condition,
            "path": str(adapter_dir),
            "generation_seconds": generation_seconds,
            "generated_parameter_count": generated_parameters,
            "generated_bytes": generated_bytes,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device_index),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
