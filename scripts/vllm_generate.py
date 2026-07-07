"""Subprocess worker for ``adapterbench.vllm_downstream_evaluator.VLLMDownstreamEvaluator``:
generates against already-rendered prompts via vLLM (upstream's own inference backend,
``vllm==0.5.4`` pinned in ``upstream/text-to-lora/pyproject.toml``).

Must run under ``upstream/text-to-lora/.venv`` (vLLM is not, and should never be, a
dependency of ``adapterbench`` itself - see ``text_to_lora_backend.py``'s docstring for why
this project always shells out for anything upstream-pinned).

Loads the vLLM engine once, then generates one group at a time, writing one sentinel-prefixed
JSON line to stdout per completed group so a caller can stream results incrementally rather
than waiting for the whole manifest to finish, with only a single engine load no matter how
many groups there are. The sentinel prefix (rather than a plain JSON line) is required
because vLLM's own logger is configured to write its INFO/WARNING lines to stdout too (see
``vllm.logger.DEFAULT_LOGGING_CONFIG``) - the caller filters for the prefix and ignores
everything else on stdout.

Manifest JSON (read from --manifest): {"model_id": str, "gpu_memory_utilization": float,
"max_new_tokens": int, "groups": [{"name": str, "adapter_dir": str|null, "prompts": [str]}]}.
Prompts are passed in fully rendered (chat template + any prefill already applied) so this
script never re-derives templating logic that already lives once in
``hf_downstream_evaluator.py`` - only the generation backend differs from that module's
``_generate``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RESULT_PREFIX = "##ADAPTERBENCH_RESULT## "


def _ensure_lora_dir_has_tokenizer(adapter_dir: str, base_tokenizer) -> None:
    """Every adapter here shares the interpreter's own tokenizer (never a task-specific
    one), but a bare PEFT adapter directory (``adapter_config.json`` +
    ``adapter_model.safetensors`` only) has no tokenizer files at all. vLLM's own LoRA
    tokenizer resolution (``get_lora_tokenizer``) tries to load one from the adapter
    directory first and only falls back to the base tokenizer on ``OSError`` - but a bare
    directory with no ``config.json`` sometimes makes ``transformers`` raise ``ValueError``
    instead (confirmed: same root cause as this project's gotcha #8 in PROJECT_PLAN.md,
    there non-fatal because it only broke a training-side eval step; here it's fatal
    because scoring depends on the generate() call actually succeeding). Saving a real
    copy of the base tokenizer into the adapter directory once sidesteps the fallback path
    entirely instead of depending on which exception type this particular transformers/
    huggingface_hub version happens to raise.
    """
    if (Path(adapter_dir) / "tokenizer_config.json").exists():
        return
    base_tokenizer.save_pretrained(adapter_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    manifest = json.loads(open(args.manifest).read())

    import vllm
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    base_tokenizer = AutoTokenizer.from_pretrained(manifest["model_id"])
    for group in manifest["groups"]:
        if group["adapter_dir"] is not None:
            _ensure_lora_dir_has_tokenizer(group["adapter_dir"], base_tokenizer)

    llm = vllm.LLM(
        manifest["model_id"],
        seed=42,
        max_model_len=2**12,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=manifest.get("gpu_memory_utilization", 0.85),
    )
    sampling_params = vllm.SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=manifest.get("max_new_tokens", 512),
        repetition_penalty=1.0,
    )

    for i, group in enumerate(manifest["groups"]):
        lora_request = None
        if group["adapter_dir"] is not None:
            # int id must be unique and >0, per vllm.lora.request.LoRARequest's own
            # docstring ("cannot be 0 or it will be treated as None").
            lora_request = LoRARequest(group["name"], i + 1, group["adapter_dir"])
        started = time.perf_counter()
        outputs = llm.generate(group["prompts"], sampling_params, lora_request=lora_request)
        elapsed = time.perf_counter() - started
        record = {
            "name": group["name"],
            "generations": [o.outputs[0].text for o in outputs],
            "seconds": elapsed,
        }
        print(RESULT_PREFIX + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
