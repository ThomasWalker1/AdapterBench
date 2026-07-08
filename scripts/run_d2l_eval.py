"""Run Doc-to-LoRA's own NIAH evaluation against a released checkpoint (or, with
``--model-name-or-path``, a frozen-interpreter baseline) and write a manifest.json of
per-dataset scores, mirroring ``generate_t2l_adapter.py``'s manifest-writing convention.

This script must run under the upstream Doc-to-LoRA environment
(``upstream/doc-to-lora/.venv/bin/python``), NOT the main ``adapterbench`` uv venv:
``ctx_to_lora`` pins its own torch/transformers/peft/vllm stack (see SETUP.md).
``adapterbench.doc_to_lora_backend`` invokes this script via subprocess and only ever
reads back the JSON manifest it writes, so the two environments never need to share a
process. Unlike ``generate_t2l_adapter.py``, this script *imports* ``ctx_to_lora`` directly
(rather than only calling out to it) - that import only happens inside this script's own
process, which itself already runs under the upstream venv, so it never crosses into
``adapterbench``'s process the way the constraint in PROJECT_PLAN.md/SETUP.md forbids.

Why this shells out to ``ctx_to_lora.eval_utils.run_eval`` instead of re-implementing
anything: D2L's forward pass is a hand-rolled non-PEFT monkeypatch (Perceiver document
aggregator + ``combine_lora``/``lora_forward_packed``), and ``run_eval`` already
implements NIAH data loading, generation, and scoring (word-level ROUGE-L for the
``ctx_magic_number_*`` family specifically - it is not in ``CLOSED_QA_DATASETS``, so
``eval_generation`` scores it with ``compute_rouge`` rather than QA F1) end to end. See
``adapterbench.doc_to_lora_backend``'s module docstring for why this collapses the usual
``HypernetworkBackend``/``DownstreamEvaluator`` split into a single call here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Must be set before any CUDA context is created - matches generate_t2l_adapter.py's own
# determinism-flags-before-torch-import convention.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# ctx_to_lora's own Trainer reports to wandb by default (see args.yaml's report_to);
# every one of D2L's own launch scripts (e.g. scripts/niah/2-eval.sh) prefixes
# `WANDB_MODE=disabled` for exactly this reason - without it, a plain eval run creates a
# real wandb.ai run under whatever account is logged in on this machine.
os.environ.setdefault("WANDB_MODE", "disabled")

import torch

from ctx_to_lora.eval_utils import run_eval

# ctx_to_lora.eval_utils.run_eval() re-applies these itself (plus torch.manual_seed(42) via
# set_seed(42)) before evaluation, but setting them here too costs nothing and matches this
# project's other upstream worker scripts.
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _to_jsonable(value):
    """`run_eval`'s returned metrics mix python floats, numpy scalars (from np.mean in
    compute_rouge/compute_qa_f1_score), and the literal string "None" (a placeholder
    eval_generation writes for length bins with zero samples) - json.dumps chokes on the
    numpy types alone, so normalize everything before serializing.
    """
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        help="Path to a D2L checkpoint's pytorch_model.bin. Requires a sibling "
        "args.yaml two directories up (run_eval's own convention: "
        "`.../<run_dir>/checkpoint-N/pytorch_model.bin` needs `<run_dir>/args.yaml`).",
    )
    parser.add_argument(
        "--model-name-or-path",
        help="Evaluate the frozen base model instead of a checkpoint (baseline; "
        "mutually exclusive with --checkpoint-path).",
    )
    parser.add_argument(
        "--remove-context",
        action="store_true",
        help="Baseline-only: withhold the document entirely (no in-context document text, "
        "and (with --checkpoint-path) no internalized LoRA either).",
    )
    parser.add_argument("--datasets", nargs="+", required=True, help="e.g. ctx_magic_number_32_1024")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument(
        "--max-samples-per-ds",
        type=int,
        default=-1,
        help="-1 keeps run_eval's own default (from the checkpoint's args.yaml / 500 for base-model eval).",
    )
    parser.add_argument("--max-ctx-chunk-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--eval-batch-size-gen", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if bool(args.checkpoint_path) == bool(args.model_name_or_path):
        raise ValueError("exactly one of --checkpoint-path / --model-name-or-path is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_eval_kwargs = dict(
        datasets=args.datasets,
        split=args.split,
        eval_batch_size=args.eval_batch_size_gen,
        max_ctx_chunk_len=args.max_ctx_chunk_len,
        max_new_tokens=args.max_new_tokens,
        generative=True,
        remove_context=args.remove_context,
    )
    if args.split == "test":
        run_eval_kwargs["max_test_samples_per_ds"] = args.max_samples_per_ds
    else:
        run_eval_kwargs["max_val_samples_per_ds"] = args.max_samples_per_ds

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if args.checkpoint_path:
        raw_metrics = run_eval(checkpoint_path=args.checkpoint_path, **run_eval_kwargs)
    else:
        raw_metrics = run_eval(model_name_or_path=args.model_name_or_path, **run_eval_kwargs)
    torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - started

    manifest = {
        "checkpoint": args.checkpoint_path,
        "base_model": args.model_name_or_path,
        "remove_context": args.remove_context,
        "split": args.split,
        "datasets": args.datasets,
        "eval_seconds": eval_seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__,
        # Keyed by "{split}_{dataset}" (run_eval's own naming), one entry per requested
        # dataset - values are the raw per-dataset metrics dict eval_generation produced
        # (num_samples + rougeL.f1, both overall and per length bin).
        "metrics": _to_jsonable(raw_metrics),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
