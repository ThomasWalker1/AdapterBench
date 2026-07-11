"""Setting 1 (disk-artifact reproduction) commands: `run` (Text-to-LoRA released
checkpoints, scored via HF or vLLM) and `run-d2l-niah` (released Doc-to-LoRA
checkpoints on needle-in-a-haystack). Neither trains anything - both load a released
hypernetwork, generate an adapter, and score it."""

from __future__ import annotations

from pathlib import Path

from ..catalog import build_matrix, load_catalog
from ._shared import DEFAULT_CATALOG, REPO_ROOT, ResultRecorder


def _run_command(args) -> None:
    from ..task_examples import build_all_task_examples, build_conditions, load_task_descriptions
    from ..text_to_lora_backend import ReleasedTextToLoRABackend

    setups, adapters = load_catalog(args.root)
    setup = setups[args.setup]
    adapter = adapters[args.adapter]
    trial = build_matrix(setup, [adapter])[0]

    checkpoint = Path(args.checkpoint)
    descriptions = load_task_descriptions(checkpoint.parent / "args.yaml")
    task_ids = args.tasks.split(",")
    conditions = build_conditions(descriptions, task_ids, variant=args.condition_variant)

    print(f"[1/4] generating {len(task_ids)} adapter(s) via {args.adapter}...", flush=True)
    backend = ReleasedTextToLoRABackend(checkpoint=checkpoint, device=args.device)
    artifacts = backend.generate(conditions, Path(args.output) / "adapters")
    print(f"[1/4] generated: {list(artifacts)}", flush=True)

    print(f"[2/4] building up to {args.limit} example(s) per task for {task_ids}...", flush=True)
    examples_by_family = build_all_task_examples(
        task_ids, descriptions, args.limit, variant=args.condition_variant, use_icl=args.use_icl
    )
    all_examples = [example for group in examples_by_family.values() for example in group]
    print(f"[2/4] built {len(all_examples)} example(s) total", flush=True)

    print(
        f"[3/4] loading interpreter {setup.models['interpreter'].model_id} "
        f"(evaluator={args.evaluator})...",
        flush=True,
    )
    if args.evaluator == "vllm":
        from ..vllm_downstream_evaluator import VLLMDownstreamEvaluator

        evaluator = VLLMDownstreamEvaluator(
            model_id=setup.models["interpreter"].model_id,
            chat_template_path=args.chat_template,
            trial_id=trial.trial_id,
            device=args.device,
            use_icl=args.use_icl,
        )
    else:
        from ..hf_downstream_evaluator import HFDownstreamEvaluator

        evaluator = HFDownstreamEvaluator(
            model_id=setup.models["interpreter"].model_id,
            chat_template_path=args.chat_template,
            trial_id=trial.trial_id,
            device=args.device,
            use_icl=args.use_icl,
        )

    print("[4/4] evaluating (writing results.jsonl/.csv after each task)...", flush=True)
    recorder = ResultRecorder(args.output, lambda r: f"  {r.task_id:16} {r.adapter:20} {r.metrics}")
    for result in evaluator.iter_evaluate(artifacts, all_examples, split=args.split):
        recorder.record(result)
    for result in evaluator.iter_evaluate_frozen(all_examples, split=args.split):
        recorder.record(result)


def _run_d2l_command(args) -> None:
    from ..doc_to_lora_backend import ReleasedDocToLoRANIAHEvaluator

    setups, adapters = load_catalog(args.root)
    setup = setups[args.setup]
    adapter = adapters[args.adapter]
    trial = build_matrix(setup, [adapter])[0]

    datasets = args.datasets.split(",")
    output_dir = Path(args.output)

    evaluator = ReleasedDocToLoRANIAHEvaluator(
        checkpoint=args.checkpoint,
        trial_id=trial.trial_id,
        max_ctx_chunk_len=args.max_ctx_chunk_len,
        max_new_tokens=args.max_new_tokens,
        eval_batch_size_gen=args.eval_batch_size_gen,
    )

    recorder = ResultRecorder(output_dir, lambda r: f"  {r.task_id:28} {r.adapter:20} {r.metrics}")

    print(f"[1/2] evaluating released D2L checkpoint on {datasets}...", flush=True)
    for result in evaluator.evaluate(
        datasets, args.limit, output_dir / "d2l_eval", split=args.split, adapter=args.adapter
    ):
        recorder.record(result)

    if args.baseline:
        print("[2/2] evaluating frozen-interpreter baseline...", flush=True)
        for result in evaluator.evaluate_frozen(
            datasets,
            args.limit,
            output_dir / "d2l_eval_baseline",
            model_name_or_path=setup.models["interpreter"].model_id,
            split=args.split,
            remove_context=not args.baseline_icl,
        ):
            recorder.record(result)


def register(subparsers) -> None:
    run = subparsers.add_parser(
        "run", help="generate adapters via a HypernetworkBackend and score them with a DownstreamEvaluator"
    )
    run.add_argument("--root", default=DEFAULT_CATALOG, type=Path)
    run.add_argument("--setup", required=True)
    run.add_argument("--adapter", required=True)
    run.add_argument("--checkpoint", required=True, help="Path to a hypermod.pt checkpoint")
    run.add_argument(
        "--tasks", default="arc_easy,arc_challenge,boolq,hellaswag,gsm8k", help="comma-separated task ids"
    )
    run.add_argument("--limit", type=int, default=10, help="examples per task")
    run.add_argument("--condition-variant", type=int, default=0)
    run.add_argument("--split", default="test")
    run.add_argument("--device", default="cuda:0")
    run.add_argument(
        "--use-icl",
        action="store_true",
        help="prepend upstream Text-to-LoRA's 3-shot in-context examples to every prompt and force an "
        "'Answer:'/\"Let's think step by step.\" generation prefix, matching the paper's Gemma table "
        "(Table 8), which uses ICL for every method including the frozen baseline. Must be set "
        "consistently: it changes both the built TaskExamples and the evaluator's scoring prefill.",
    )
    run.add_argument(
        "--chat-template",
        default=str(
            REPO_ROOT / "upstream" / "text-to-lora" / "chat_templates" / "google" / "gemma-2-2b-it" / "chat_template.jinja"
        ),
    )
    run.add_argument(
        "--evaluator",
        choices=["hf", "vllm"],
        default="hf",
        help="'hf' (default): plain transformers+peft, works for any adapter format. "
        "'vllm': upstream's own inference backend (vllm==0.5.4, via a subprocess into "
        "upstream/text-to-lora/.venv) - LoRA-only (not FourierFT/IA3/LoKr), but reproduces "
        "the paper's published numbers noticeably more closely (confirmed for Mistral-7B, "
        "see PROJECT_PLAN.md's Phase 5.5 follow-up); prefer it whenever every adapter under "
        "test is a plain LoRA.",
    )
    run.add_argument("--output", default="results/text_to_peft_gemma2b_reconstruction")
    run.set_defaults(func=_run_command)

    # A separate subcommand rather than reusing `run`: D2L conditions on a raw document
    # plus a list of ctx_magic_number_<lo>_<hi> NIAH dataset names (not a per-task
    # condition string plus a built list of TaskExamples), and its own eval API bakes
    # adapter generation and downstream scoring into one call - see
    # adapterbench.doc_to_lora_backend's module docstring. `run`'s generate-then-evaluate
    # shape genuinely doesn't fit.
    run_d2l = subparsers.add_parser(
        "run-d2l-niah", help="evaluate a released Doc-to-LoRA checkpoint on needle-in-a-haystack (ctx_magic_number) tasks"
    )
    run_d2l.add_argument("--root", default=DEFAULT_CATALOG, type=Path)
    run_d2l.add_argument("--setup", required=True)
    run_d2l.add_argument("--adapter", default="lora_r8_d2l")
    run_d2l.add_argument("--checkpoint", required=True, help="Path to a D2L checkpoint's pytorch_model.bin")
    run_d2l.add_argument(
        "--datasets", required=True, help="comma-separated ctx_magic_number_<lo>_<hi> dataset names"
    )
    run_d2l.add_argument("--limit", type=int, default=500, help="examples per dataset")
    run_d2l.add_argument("--split", default="test", choices=["validation", "test"])
    run_d2l.add_argument("--max-ctx-chunk-len", type=int, default=1024)
    run_d2l.add_argument("--max-new-tokens", type=int, default=32)
    run_d2l.add_argument("--eval-batch-size-gen", type=int, default=4)
    run_d2l.add_argument(
        "--baseline",
        action="store_true",
        help="also score a frozen-interpreter baseline (same base model, no D2L checkpoint)",
    )
    run_d2l.add_argument(
        "--baseline-icl",
        action="store_true",
        help="baseline hands the document to the frozen interpreter directly in its prompt "
        "(interpreter_with_icl) instead of withholding it entirely (frozen_interpreter, the default)",
    )
    run_d2l.add_argument("--output", default="results/doc_to_peft_gemma2b_reconstruction")
    run_d2l.set_defaults(func=_run_d2l_command)
