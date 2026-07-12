"""GPU integration check for the adapter artifact -> frozen interpreter boundary."""

from __future__ import annotations

from hashlib import sha256
import time

import torch

from .peft_support import count_adapter_parameters, make_peft_config
from .schema import AdapterManifest
from .state import AdapterStateLayout, trainable_state_dict


def _materialize_generated_state(model, condition: str) -> int:
    """Deterministic stand-in generator used only to test the artifact contract.

    It emits every adapter tensor as a function of condition and tensor name. It
    is intentionally untrained; downstream numbers from this check are not a
    benchmark result.
    """
    generated = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            digest = sha256(f"{condition}\0{name}".encode()).digest()
            seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
            generator = torch.Generator(device=parameter.device).manual_seed(seed)
            # Multiplicative adapters (e.g. IA3, via the pipeline) center weights at 1.0;
            # the LoRA-only baseline is additive, so weights center at 0.0.
            if "ia3" in name:
                parameter.copy_(1.0 + 1e-3 * torch.randn(parameter.shape, generator=generator, device=parameter.device))
            else:
                parameter.copy_(1e-3 * torch.randn(parameter.shape, generator=generator, device=parameter.device))
            generated += parameter.numel()
    return generated


def run_adapter_materialization_smoke(
    model_id: str, adapters: list[AdapterManifest], device: str, condition: str
) -> list[dict]:
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible; run outside a device-isolated sandbox")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoded = tokenizer("Input: I loved it.\nOutput: positive", return_tensors="pt").to(device)
    results = []
    for adapter in adapters:
        torch.cuda.reset_peak_memory_stats(device) if device.startswith("cuda") else None
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
        model.eval()
        peft_config = make_peft_config(adapter)
        # PEFT 0.19 enumerates float8 dtypes added after torch 2.5 when its
        # automatic adapter upcast is enabled. Adapter tensors are explicitly
        # generated below, so retaining their native dtype is appropriate.
        model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
        generated_count = _materialize_generated_state(model, condition)
        layout = AdapterStateLayout.from_state_dict(trainable_state_dict(model))
        if layout.parameter_count != generated_count:
            raise RuntimeError("adapter layout and generated parameter count disagree")
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**encoded, labels=encoded["input_ids"])
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "model": model_id,
                "adapter": adapter.name,
                "family": adapter.family,
                "status": "ok",
                "generated_parameters": generated_count,
                "trainable_parameters": count_adapter_parameters(model),
                "downstream_lm_loss": float(outputs.loss),
                "forward_seconds": elapsed,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None,
                "warning": "Untrained deterministic generator; this validates materialization and execution only.",
            }
        )
        del outputs, model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return results
