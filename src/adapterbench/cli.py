from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import build_matrix, load_catalog
from .doctor import environment_report


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "configs"

# lol_022/043/044/045/047/050/063/064 are all confirmed present in T2L's own
# train_ds_names (upstream/text-to-lora/configs/hyper_lora_decontam_lol_tasks.yaml).
# The previous default (lol_022,033,034,035,039,043,044,045) trained on lol_033/034
# (two of T2L's 10 contamination-removed tasks) and lol_035/039 (two of T2L's own 11
# held-out zero-shot validation tasks) - exactly the leakage a training pilot should
# avoid. --decontam-config validates any --tasks value against this list at run time.
_DEFAULT_SFT_TRAIN_TASKS = "lol_022,lol_043,lol_044,lol_045,lol_047,lol_050,lol_063,lol_064"


def _catalog_command(args) -> None:
    setups, adapters = load_catalog(args.root)
    print("Setups:")
    for item in setups.values():
        print(f"  {item.name:32} {item.protocol:24} {item.availability}")
    print("Adapters:")
    for item in adapters.values():
        print(f"  {item.name:32} {item.family:24} {item.implementation}")


def _validate_command(args) -> None:
    setups, adapters = load_catalog(args.root)
    trials = []
    for setup in setups.values():
        for adapter in adapters.values():
            try:
                trials.extend(build_matrix(setup, [adapter]))
            except ValueError as error:
                print(f"SKIP {setup.name} x {adapter.name}: {error}")
    print(f"valid setups={len(setups)} adapters={len(adapters)} trials={len(trials)}")


def _matrix_command(args) -> None:
    setups, adapters = load_catalog(args.root)
    setup = setups[args.setup]
    selected = list(adapters.values()) if args.adapters == "all" else [adapters[name] for name in args.adapters.split(",")]
    trials = build_matrix(setup, selected)
    payload = [trial.model_dump(mode="json") for trial in trials]
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    for trial in trials:
        print(trial.trial_id)


def _doctor_command(args) -> None:
    report = environment_report()
    print(json.dumps(report, indent=2))
    if args.require_cuda and not report["cuda"]["available"]:
        raise SystemExit("CUDA is required but not visible to this process")


def _peft_smoke_command(args) -> None:
    from .hf_smoke import run_adapter_materialization_smoke

    _, adapters = load_catalog(args.root)
    selected = [adapters[name] for name in args.adapters.split(",")]
    results = run_adapter_materialization_smoke(args.model, selected, args.device, args.condition)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def _run_command(args) -> None:
    from .reporting import write_results
    from .task_examples import build_all_task_examples, build_conditions, load_task_descriptions
    from .text_to_lora_backend import ReleasedTextToLoRABackend

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
        from .vllm_downstream_evaluator import VLLMDownstreamEvaluator

        evaluator = VLLMDownstreamEvaluator(
            model_id=setup.models["interpreter"].model_id,
            chat_template_path=args.chat_template,
            trial_id=trial.trial_id,
            device=args.device,
            use_icl=args.use_icl,
        )
    else:
        from .hf_downstream_evaluator import HFDownstreamEvaluator

        evaluator = HFDownstreamEvaluator(
            model_id=setup.models["interpreter"].model_id,
            chat_template_path=args.chat_template,
            trial_id=trial.trial_id,
            device=args.device,
            use_icl=args.use_icl,
        )

    print("[4/4] evaluating (writing results.jsonl/.csv after each task)...", flush=True)
    results: list = []

    def _record(result) -> None:
        results.append(result)
        write_results(results, args.output)
        print(f"  {result.task_id:16} {result.adapter:20} {result.metrics}", flush=True)

    for result in evaluator.iter_evaluate(artifacts, all_examples, split=args.split):
        _record(result)
    for result in evaluator.iter_evaluate_frozen(all_examples, split=args.split):
        _record(result)


def _t2p_sft_command(args) -> None:
    import torch
    from functools import partial

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .t2p.condition_encoder import embed_task_descriptions, load_condition_encoder
    from .t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes
    from .t2p.lol_data import LolSFTDataset, load_task_metadata, lol_collate_fn, validate_training_tasks
    from .t2p.model_utils import get_decoder_layers
    from .t2p.sft_trainer import train_downstream_hypernetwork

    # Pins the hypernetwork's weight init (only its codec-specific initial_bias is seeded
    # by `TextToPeftHypernetwork(seed=...)` itself, not its Linear/Embedding layers) and
    # the DataLoader's shuffle order, both of which otherwise draw from whatever the global
    # RNG happens to be at construction time - see PROJECT_PLAN.md's step-budget-sweep
    # follow-up for why two nominally-identical runs landed on wildly different outcomes
    # without this.
    torch.manual_seed(args.seed)

    task_ids = args.tasks.split(",")
    target_modules = args.target_modules.split(",")
    tasks_dir = Path(args.tasks_dir)
    validate_training_tasks(task_ids, args.decontam_config)

    print(f"[1/5] loading interpreter {args.interpreter}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.interpreter)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    interpreter = AutoModelForCausalLM.from_pretrained(args.interpreter, dtype=torch.bfloat16).to(args.device)
    interpreter.eval()
    for parameter in interpreter.parameters():
        parameter.requires_grad = False
    layers = get_decoder_layers(interpreter)
    module_shapes = infer_module_shapes(layers, target_modules, hidden_size=interpreter.config.hidden_size)

    print(f"[2/5] embedding task descriptions via {args.condition_encoder}...", flush=True)
    encoder_model, encoder_tokenizer = load_condition_encoder(args.condition_encoder, device=args.device)
    metadata_by_task = {
        task_id: load_task_metadata(tasks_dir, task_id, max_descriptions=args.max_descriptions)
        for task_id in task_ids
    }
    embeddings_by_task = {
        task_id: embed_task_descriptions(metadata.descriptions, encoder_model, encoder_tokenizer).to(args.device)
        for task_id, metadata in metadata_by_task.items()
    }
    condition_dim = next(iter(embeddings_by_task.values())).shape[-1]
    del encoder_model

    print(f"[3/5] loading + tokenizing {len(task_ids)} task(s) from {args.tasks_dir}...", flush=True)
    datasets = [
        LolSFTDataset(tokenizer, metadata_by_task[task_id], embeddings_by_task[task_id], limit=args.limit)
        for task_id in task_ids
    ]
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(lol_collate_fn, pad_token_id=tokenizer.pad_token_id),
    )
    batches = [batch.to(args.device) for batch in dataloader]
    print(f"[3/5] built {len(dataset)} example(s) across {len(datasets)} task(s), {len(batches)} batch(es)/epoch", flush=True)

    print(f"[4/5] training hypernetwork (adapter={args.adapter}, target_modules={target_modules})...", flush=True)
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=condition_dim,
        module_shapes=module_shapes,
        num_layers=len(layers),
        adapter=args.adapter,
        seed=args.seed,
    ).to(args.device)
    stats = train_downstream_hypernetwork(
        hypernetwork,
        interpreter,
        layers,
        batches,
        steps=args.steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        l2_reg_generated_w=args.l2_reg_generated_w,
    )

    print(f"[5/5] initial_loss={stats.initial_loss:.4f} final_loss={stats.final_loss:.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "interpreter": args.interpreter,
        "adapter": args.adapter,
        "target_modules": target_modules,
        "tasks": task_ids,
        "steps": stats.steps,
        "initial_loss": stats.initial_loss,
        "final_loss": stats.final_loss,
        "losses": list(stats.losses),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}", flush=True)


_PILOT_DEFAULT_TARGET_MODULES = {
    "lora": ["q_proj", "v_proj"],
    "freeze_a_lora": ["q_proj", "v_proj"],
    "lokr": ["q_proj", "v_proj"],
    "fourierft": ["q_proj", "v_proj"],
    "ia3": ["k_proj", "v_proj", "down_proj"],
    "activation_steering": ["block"],
}


def _t2p_sft_pilot_command(args) -> None:
    import torch
    from functools import partial

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .reporting import write_results
    from .task_examples import build_all_task_examples, load_task_descriptions
    from .t2p.condition_encoder import embed_task_descriptions, load_condition_encoder
    from .t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes
    from .t2p.live_evaluator import HypernetworkDownstreamEvaluator
    from .t2p.lol_data import LolSFTDataset, load_task_metadata, lol_collate_fn, validate_training_tasks
    from .t2p.model_utils import get_decoder_layers
    from .t2p.sft_trainer import train_downstream_hypernetwork

    # See _t2p_sft_command's comment on why this is required for reproducibility, not
    # just tidiness: neither the DataLoader shuffle nor the hypernetwork's own weight
    # init/dropout is otherwise pinned.
    torch.manual_seed(args.seed)

    task_ids = args.tasks.split(",")
    adapters = args.adapters.split(",")
    validate_training_tasks(task_ids, args.decontam_config)
    eval_task_ids = args.eval_tasks.split(",")
    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] loading interpreter {args.interpreter}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.interpreter)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    interpreter = AutoModelForCausalLM.from_pretrained(args.interpreter, dtype=torch.bfloat16).to(args.device)
    interpreter.eval()
    for parameter in interpreter.parameters():
        parameter.requires_grad = False
    layers = get_decoder_layers(interpreter)

    print(f"[2/6] embedding task descriptions via {args.condition_encoder}...", flush=True)
    encoder_model, encoder_tokenizer = load_condition_encoder(args.condition_encoder, device=args.device)
    metadata_by_task = {
        task_id: load_task_metadata(tasks_dir, task_id, max_descriptions=args.max_descriptions)
        for task_id in task_ids
    }
    train_embeddings_by_task = {
        task_id: embed_task_descriptions(metadata.descriptions, encoder_model, encoder_tokenizer).to(args.device)
        for task_id, metadata in metadata_by_task.items()
    }
    condition_dim = next(iter(train_embeddings_by_task.values())).shape[-1]

    eval_descriptions = load_task_descriptions(args.eval_descriptions)
    eval_condition_embeddings = {
        family: embed_task_descriptions(
            [eval_descriptions[family][args.eval_variant]], encoder_model, encoder_tokenizer
        )[0].to(args.device)
        for family in eval_task_ids
    }
    del encoder_model

    print(f"[3/6] loading + tokenizing {len(task_ids)} training task(s) from {args.tasks_dir}...", flush=True)
    datasets = [
        LolSFTDataset(tokenizer, metadata_by_task[task_id], train_embeddings_by_task[task_id], limit=args.limit)
        for task_id in task_ids
    ]
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=partial(lol_collate_fn, pad_token_id=tokenizer.pad_token_id),
    )
    batches = [batch.to(args.device) for batch in dataloader]
    print(f"[3/6] built {len(dataset)} example(s) across {len(datasets)} task(s), {len(batches)} batch(es)/epoch", flush=True)

    print(f"[4/6] building {args.eval_limit} eval example(s) per family for {eval_task_ids}...", flush=True)
    eval_examples_by_family = build_all_task_examples(
        eval_task_ids, eval_descriptions, args.eval_limit, variant=args.eval_variant, use_icl=args.use_icl
    )
    all_eval_examples = [example for group in eval_examples_by_family.values() for example in group]

    results: list = []
    loss_curves: dict = {}

    def _record(result) -> None:
        results.append(result)
        write_results(results, output_dir)
        print(f"  {result.adapter:20} {result.task_id:16} {result.metrics}", flush=True)

    print("[5/6] scoring frozen interpreter baseline...", flush=True)
    frozen_evaluator = HypernetworkDownstreamEvaluator(
        interpreter,
        layers,
        None,
        tokenizer,
        trial_id="t2p_sft_pilot::frozen_interpreter",
        device=args.device,
        use_icl=args.use_icl,
    )
    for result in frozen_evaluator.iter_evaluate_frozen(all_eval_examples, split=args.eval_split):
        _record(result)

    for adapter in adapters:
        # Re-seeded per adapter (not just once at the top of this function) so each
        # adapter's weight init/dropout trajectory is independent of which adapters were
        # trained before it in this same process/loop position - otherwise adapter N's
        # result would silently depend on how much RNG state adapters 1..N-1 consumed.
        torch.manual_seed(args.seed)
        target_modules = _PILOT_DEFAULT_TARGET_MODULES[adapter]
        print(f"[6/6] adapter={adapter} target_modules={target_modules}: training...", flush=True)
        module_shapes = infer_module_shapes(layers, target_modules, hidden_size=interpreter.config.hidden_size)
        hypernetwork = TextToPeftHypernetwork(
            condition_dim=condition_dim,
            module_shapes=module_shapes,
            num_layers=len(layers),
            adapter=adapter,
            seed=args.seed,
        ).to(args.device)
        stats = train_downstream_hypernetwork(
            hypernetwork,
            interpreter,
            layers,
            batches,
            steps=args.steps,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            l2_reg_generated_w=args.l2_reg_generated_w,
        )
        print(f"  {adapter}: initial_loss={stats.initial_loss:.4f} final_loss={stats.final_loss:.4f}", flush=True)
        loss_curves[adapter] = {
            "target_modules": target_modules,
            "initial_loss": stats.initial_loss,
            "final_loss": stats.final_loss,
            "steps": stats.steps,
            "losses": list(stats.losses),
        }
        (output_dir / "loss_curves.json").write_text(json.dumps(loss_curves, indent=2) + "\n")

        print(f"[6/6] adapter={adapter}: evaluating...", flush=True)
        evaluator = HypernetworkDownstreamEvaluator(
            interpreter,
            layers,
            hypernetwork,
            tokenizer,
            trial_id=f"t2p_sft_pilot::{adapter}",
            device=args.device,
            use_icl=args.use_icl,
        )
        for result in evaluator.iter_evaluate(eval_condition_embeddings, all_eval_examples, split=args.eval_split):
            _record(result)

        del hypernetwork, evaluator
        torch.cuda.empty_cache()

    print(f"wrote {output_dir}/results.jsonl, {output_dir}/results.csv, {output_dir}/loss_curves.json", flush=True)


def _t2p_sft_sweep_command(args) -> None:
    """Checkpointed counterpart to `t2p-sft-pilot`: scores held-out accuracy at several
    step budgets per adapter under one persistent optimizer (`t2p.sft_trainer.
    train_with_checkpoints`), instead of only ever reporting a single fixed-step endpoint.

    Originally motivated by an apparent finding that longer training made held-out
    accuracy worse for every adapter (see PROJECT_PLAN.md's Phase 4 section) - but running
    this swept the ground out from under that framing: most adapters were already
    flat/collapsed by the *first* checkpoint, with no peak-then-decline curve at all, and
    the numbers didn't even match an earlier pilot run at the same nominal step count.
    Root cause: neither the DataLoader shuffle nor the hypernetwork's own weight
    init/dropout was seeded, so nominally-identical runs land on different trajectories
    (now fixed - see the `torch.manual_seed`/`generator=` calls below). This command is
    still the right tool for telling "still converging" apart from "already past the
    point where held-out generalization peaks" per adapter - just don't trust a single
    seed's answer to that question either, per the same finding.
    """
    import dataclasses

    import torch
    from functools import partial

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .reporting import write_results
    from .task_examples import build_all_task_examples, load_task_descriptions
    from .t2p.condition_encoder import embed_task_descriptions, load_condition_encoder
    from .t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes
    from .t2p.live_evaluator import HypernetworkDownstreamEvaluator
    from .t2p.lol_data import LolSFTDataset, load_task_metadata, lol_collate_fn, validate_training_tasks
    from .t2p.model_utils import get_decoder_layers
    from .t2p.sft_trainer import train_with_checkpoints

    torch.manual_seed(args.seed)

    task_ids = args.tasks.split(",")
    adapters = args.adapters.split(",")
    checkpoint_steps = sorted(int(s) for s in args.checkpoint_steps.split(","))
    validate_training_tasks(task_ids, args.decontam_config)
    eval_task_ids = args.eval_tasks.split(",")
    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] loading interpreter {args.interpreter}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.interpreter)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    interpreter = AutoModelForCausalLM.from_pretrained(args.interpreter, dtype=torch.bfloat16).to(args.device)
    interpreter.eval()
    for parameter in interpreter.parameters():
        parameter.requires_grad = False
    layers = get_decoder_layers(interpreter)

    print(f"[2/6] embedding task descriptions via {args.condition_encoder}...", flush=True)
    encoder_model, encoder_tokenizer = load_condition_encoder(args.condition_encoder, device=args.device)
    metadata_by_task = {
        task_id: load_task_metadata(tasks_dir, task_id, max_descriptions=args.max_descriptions)
        for task_id in task_ids
    }
    train_embeddings_by_task = {
        task_id: embed_task_descriptions(metadata.descriptions, encoder_model, encoder_tokenizer).to(args.device)
        for task_id, metadata in metadata_by_task.items()
    }
    condition_dim = next(iter(train_embeddings_by_task.values())).shape[-1]

    eval_descriptions = load_task_descriptions(args.eval_descriptions)
    eval_condition_embeddings = {
        family: embed_task_descriptions(
            [eval_descriptions[family][args.eval_variant]], encoder_model, encoder_tokenizer
        )[0].to(args.device)
        for family in eval_task_ids
    }
    del encoder_model

    print(f"[3/6] loading + tokenizing {len(task_ids)} training task(s) from {args.tasks_dir}...", flush=True)
    datasets = [
        LolSFTDataset(tokenizer, metadata_by_task[task_id], train_embeddings_by_task[task_id], limit=args.limit)
        for task_id in task_ids
    ]
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=partial(lol_collate_fn, pad_token_id=tokenizer.pad_token_id),
    )
    batches = [batch.to(args.device) for batch in dataloader]
    print(f"[3/6] built {len(dataset)} example(s) across {len(datasets)} task(s), {len(batches)} batch(es)/epoch", flush=True)

    print(f"[4/6] building {args.eval_limit} eval example(s) per family for {eval_task_ids}...", flush=True)
    eval_examples_by_family = build_all_task_examples(
        eval_task_ids, eval_descriptions, args.eval_limit, variant=args.eval_variant, use_icl=args.use_icl
    )
    all_eval_examples = [example for group in eval_examples_by_family.values() for example in group]

    results: list = []
    loss_curves: dict = {}

    def _record(result) -> None:
        results.append(result)
        write_results(results, output_dir)
        print(f"  step={result.metadata['checkpoint_step']:<5} {result.adapter:20} {result.task_id:16} {result.metrics}", flush=True)

    print("[5/6] scoring frozen interpreter baseline (checkpoint_step=0)...", flush=True)
    frozen_evaluator = HypernetworkDownstreamEvaluator(
        interpreter, layers, None, tokenizer, trial_id="t2p_sft_sweep::frozen_interpreter", device=args.device,
        use_icl=args.use_icl,
    )
    for result in frozen_evaluator.iter_evaluate_frozen(all_eval_examples, split=args.eval_split):
        _record(dataclasses.replace(result, metadata={**result.metadata, "checkpoint_step": 0}))

    for adapter in adapters:
        # Re-seeded per adapter, not just once at the top - see t2p-sft-pilot's identical
        # comment for why (each adapter's trajectory must be independent of loop position).
        torch.manual_seed(args.seed)
        target_modules = _PILOT_DEFAULT_TARGET_MODULES[adapter]
        print(f"[6/6] adapter={adapter} target_modules={target_modules}: sweeping checkpoints {checkpoint_steps}...", flush=True)
        module_shapes = infer_module_shapes(layers, target_modules, hidden_size=interpreter.config.hidden_size)
        hypernetwork = TextToPeftHypernetwork(
            condition_dim=condition_dim,
            module_shapes=module_shapes,
            num_layers=len(layers),
            adapter=adapter,
            seed=args.seed,
        ).to(args.device)
        stats_by_checkpoint = train_with_checkpoints(
            hypernetwork,
            interpreter,
            layers,
            batches,
            checkpoint_steps=checkpoint_steps,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            l2_reg_generated_w=args.l2_reg_generated_w,
        )
        loss_curves[adapter] = {
            "target_modules": target_modules,
            "checkpoints": {
                str(step): {"initial_loss": s.initial_loss, "final_loss": s.final_loss, "losses": list(s.losses)}
                for step, s in stats_by_checkpoint.items()
            },
        }
        (output_dir / "loss_curves.json").write_text(json.dumps(loss_curves, indent=2) + "\n")

        for step in checkpoint_steps:
            print(f"  {adapter} @ step {step}: final_loss={stats_by_checkpoint[step].final_loss:.4f} - evaluating...", flush=True)
            evaluator = HypernetworkDownstreamEvaluator(
                interpreter, layers, hypernetwork, tokenizer, trial_id=f"t2p_sft_sweep::{adapter}::step{step}",
                device=args.device, use_icl=args.use_icl,
            )
            for result in evaluator.iter_evaluate(eval_condition_embeddings, all_eval_examples, split=args.eval_split):
                _record(dataclasses.replace(result, metadata={**result.metadata, "checkpoint_step": step}))
            del evaluator

        del hypernetwork
        torch.cuda.empty_cache()

    print(f"wrote {output_dir}/results.jsonl, {output_dir}/results.csv, {output_dir}/loss_curves.json", flush=True)


def _t2p_synthetic_pilot_command(args) -> None:
    import torch
    from functools import partial

    from .contracts import EvaluationResult
    from .reporting import write_results
    from .t2p.condition_encoder import embed_task_descriptions, load_condition_encoder
    from .t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes
    from .t2p.lol_data import lol_collate_fn
    from .t2p.model_utils import get_decoder_layers
    from .t2p.sft_trainer import train_downstream_hypernetwork
    from .t2p.synthetic_evaluator import evaluate_families
    from .t2p.synthetic_tasks import TASK_FAMILIES, SyntheticSFTDataset
    from .t2p.tiny_interpreter import PAD_ID, build_tiny_interpreter

    torch.manual_seed(args.seed)

    adapters = args.adapters.split(",")
    family_names = args.families.split(",") if args.families else list(TASK_FAMILIES)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] building tiny interpreter (freshly initialized, no download)...", flush=True)
    interpreter = build_tiny_interpreter(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        seed=args.seed,
    ).to(args.device)
    interpreter.eval()
    for parameter in interpreter.parameters():
        parameter.requires_grad = False
    layers = get_decoder_layers(interpreter)

    print(f"[2/6] embedding task descriptions via {args.condition_encoder}...", flush=True)
    encoder_model, encoder_tokenizer = load_condition_encoder(args.condition_encoder, device=args.device)
    train_embeddings_by_family, eval_embeddings_by_family = {}, {}
    for family_name in family_names:
        descriptions = TASK_FAMILIES[family_name].descriptions
        train_descriptions = descriptions[: args.train_descriptions]
        eval_descriptions = descriptions[args.train_descriptions :]
        if not eval_descriptions:
            raise ValueError(f"{family_name} has no held-out descriptions left after --train-descriptions")
        train_embeddings_by_family[family_name] = embed_task_descriptions(
            train_descriptions, encoder_model, encoder_tokenizer
        ).to(args.device)
        eval_embeddings_by_family[family_name] = embed_task_descriptions(
            eval_descriptions, encoder_model, encoder_tokenizer
        )[0].to(args.device)
    condition_dim = next(iter(train_embeddings_by_family.values())).shape[-1]
    del encoder_model

    print(f"[3/6] building {args.examples_per_family} synthetic example(s) per family for {family_names}...", flush=True)
    datasets = [
        SyntheticSFTDataset(
            family_name,
            train_embeddings_by_family[family_name],
            seq_len=args.seq_len,
            size=args.examples_per_family,
            training=True,
            seed=args.seed,
        )
        for family_name in family_names
    ]
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=partial(lol_collate_fn, pad_token_id=PAD_ID),
    )
    batches = [batch.to(args.device) for batch in dataloader]
    print(f"[3/6] built {len(dataset)} example(s), {len(batches)} batch(es)/epoch", flush=True)

    results: list = []
    loss_curves: dict = {}

    def _record(adapter: str, trial_id: str, accuracies: dict, generated_parameter_count: int) -> None:
        for family_name, accuracy in accuracies.items():
            results.append(
                EvaluationResult(
                    trial_id=trial_id,
                    task_id=family_name,
                    split="synthetic_eval",
                    adapter=adapter,
                    metrics={"exact_match": accuracy, "n_examples": float(args.eval_examples_per_family)},
                    generated_parameter_count=generated_parameter_count,
                    generation_seconds=0.0,
                    inference_seconds=0.0,
                    metadata={},
                )
            )
        write_results(results, output_dir)
        for family_name, accuracy in accuracies.items():
            print(f"  {adapter:20} {family_name:12} exact_match={accuracy:.2f}", flush=True)

    print("[4/6] scoring frozen interpreter baseline...", flush=True)
    frozen_accuracies = evaluate_families(
        None, interpreter, layers, None, family_names,
        seq_len=args.seq_len, num_examples=args.eval_examples_per_family, device=args.device, seed=args.eval_seed,
    )
    _record("frozen_interpreter", "t2p_synthetic_pilot::frozen_interpreter", frozen_accuracies, 0)

    for adapter in adapters:
        # Re-seeded per adapter, not just once at the top - see t2p-sft-pilot's identical
        # comment for why (each adapter's trajectory must be independent of loop position).
        torch.manual_seed(args.seed)
        target_modules = _PILOT_DEFAULT_TARGET_MODULES[adapter]
        print(f"[5/6] adapter={adapter} target_modules={target_modules}: training...", flush=True)
        module_shapes = infer_module_shapes(layers, target_modules, hidden_size=interpreter.config.hidden_size)
        hypernetwork = TextToPeftHypernetwork(
            condition_dim=condition_dim,
            module_shapes=module_shapes,
            num_layers=len(layers),
            adapter=adapter,
            rank=args.rank,
            n_frequency=args.n_frequency,
            seed=args.seed,
        ).to(args.device)
        stats = train_downstream_hypernetwork(
            hypernetwork,
            interpreter,
            layers,
            batches,
            steps=args.steps,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            l2_reg_generated_w=args.l2_reg_generated_w,
        )
        print(f"  {adapter}: initial_loss={stats.initial_loss:.4f} final_loss={stats.final_loss:.4f}", flush=True)
        loss_curves[adapter] = {
            "target_modules": target_modules,
            "initial_loss": stats.initial_loss,
            "final_loss": stats.final_loss,
            "steps": stats.steps,
            "losses": list(stats.losses),
        }
        (output_dir / "loss_curves.json").write_text(json.dumps(loss_curves, indent=2) + "\n")

        print(f"[6/6] adapter={adapter}: evaluating...", flush=True)
        accuracies = evaluate_families(
            hypernetwork, interpreter, layers, eval_embeddings_by_family, family_names,
            seq_len=args.seq_len, num_examples=args.eval_examples_per_family, device=args.device, seed=args.eval_seed,
        )
        _record(adapter, f"t2p_synthetic_pilot::{adapter}", accuracies, hypernetwork.generated_parameter_count())

        del hypernetwork

    print(f"wrote {output_dir}/results.jsonl, {output_dir}/results.csv, {output_dir}/loss_curves.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PEFT adapters as hypernetwork outputs")
    parser.set_defaults(func=lambda _: parser.print_help())
    subparsers = parser.add_subparsers(dest="command")

    catalog = subparsers.add_parser("catalog", help="list registered setups and adapters")
    catalog.add_argument("--root", default=DEFAULT_CATALOG, type=Path)
    catalog.set_defaults(func=_catalog_command)

    validate = subparsers.add_parser("validate", help="validate all manifests and compatible trial combinations")
    validate.add_argument("--root", default=DEFAULT_CATALOG, type=Path)
    validate.set_defaults(func=_validate_command)

    matrix = subparsers.add_parser("matrix", help="materialize immutable trial manifests")
    matrix.add_argument("--root", default=DEFAULT_CATALOG, type=Path)
    matrix.add_argument("--setup", required=True)
    matrix.add_argument("--adapters", default="all", help="comma-separated adapter names or 'all'")
    matrix.add_argument("--output")
    matrix.set_defaults(func=_matrix_command)

    doctor = subparsers.add_parser("doctor", help="report package and accelerator availability")
    doctor.add_argument("--require-cuda", action="store_true")
    doctor.set_defaults(func=_doctor_command)

    smoke = subparsers.add_parser("peft-smoke", help="materialize generated PEFT state and execute a frozen HF model")
    smoke.add_argument("--root", default=DEFAULT_CATALOG, type=Path)
    smoke.add_argument("--model", default="Qwen/Qwen3-0.6B")
    smoke.add_argument("--adapters", default="lora_r8_t2l,fourierft_1000,lokr_r8,ia3,prefix_tuning_64")
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--condition", default="Normalize a sentiment statement to positive or negative.")
    smoke.add_argument("--output", default="results/hf_adapter_smoke.json")
    smoke.set_defaults(func=_peft_smoke_command)

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

    t2p_sft = subparsers.add_parser(
        "t2p-sft",
        help="live end-to-end SFT: hook the hypernetwork's generated output into a real interpreter's forward pass and train on real next-token loss (Phase 4)",
    )
    t2p_sft.add_argument("--tasks-dir", default=str(REPO_ROOT / "upstream" / "text-to-lora" / "tasks"))
    t2p_sft.add_argument("--tasks", default=_DEFAULT_SFT_TRAIN_TASKS)
    t2p_sft.add_argument(
        "--decontam-config",
        default=str(REPO_ROOT / "upstream" / "text-to-lora" / "configs" / "hyper_lora_decontam_lol_tasks.yaml"),
        help="T2L's own train_ds_names list - --tasks is validated against it so a training run "
        "can't silently include one of T2L's contamination-removed or held-out-validation tasks",
    )
    t2p_sft.add_argument("--interpreter", default="Qwen/Qwen3-0.6B")
    t2p_sft.add_argument("--adapter", default="lora")
    t2p_sft.add_argument(
        "--target-modules",
        default="q_proj,v_proj",
        help="comma-separated hook sites: named linear submodules (e.g. q_proj,v_proj) "
        "for weight-space adapters, or 'block' for activation-space ones "
        "(hooks the whole decoder layer / residual stream)",
    )
    t2p_sft.add_argument("--condition-encoder", default="Alibaba-NLP/gte-large-en-v1.5")
    t2p_sft.add_argument("--max-descriptions", type=int, default=8)
    t2p_sft.add_argument("--limit", type=int, default=20, help="examples per task")
    t2p_sft.add_argument("--batch-size", type=int, default=4)
    t2p_sft.add_argument("--steps", type=int, default=200)
    t2p_sft.add_argument("--learning-rate", type=float, default=1e-3)
    t2p_sft.add_argument("--max-grad-norm", type=float, default=1.0)
    t2p_sft.add_argument("--l2-reg-generated-w", type=float, default=1e-3)
    t2p_sft.add_argument("--seed", type=int, default=777)
    t2p_sft.add_argument("--device", default="cuda:0")
    t2p_sft.add_argument("--output", default="results/t2p_sft/pilot.json")
    t2p_sft.set_defaults(func=_t2p_sft_command)

    t2p_sft_pilot = subparsers.add_parser(
        "t2p-sft-pilot",
        help="small multi-task live-SFT pilot: train 2-3 adapters to convergence on the shared 8-task "
        "training split, then score each via HypernetworkDownstreamEvaluator against real held-out benchmark "
        "examples (Phase 4's 'next' step)",
    )
    t2p_sft_pilot.add_argument("--tasks-dir", default=str(REPO_ROOT / "upstream" / "text-to-lora" / "tasks"))
    t2p_sft_pilot.add_argument("--tasks", default=_DEFAULT_SFT_TRAIN_TASKS)
    t2p_sft_pilot.add_argument(
        "--decontam-config",
        default=str(REPO_ROOT / "upstream" / "text-to-lora" / "configs" / "hyper_lora_decontam_lol_tasks.yaml"),
        help="see 't2p-sft --decontam-config'",
    )
    t2p_sft_pilot.add_argument("--interpreter", default="Qwen/Qwen3-0.6B")
    t2p_sft_pilot.add_argument(
        "--adapters",
        default="lora,ia3,activation_steering",
        help="comma-separated adapters to train and compare; each uses a fixed default hook site "
        "(lora/freeze_a_lora/lokr/fourierft -> q_proj,v_proj; ia3 -> k_proj,v_proj,down_proj; "
        "activation_steering -> block)",
    )
    t2p_sft_pilot.add_argument("--condition-encoder", default="Alibaba-NLP/gte-large-en-v1.5")
    t2p_sft_pilot.add_argument("--max-descriptions", type=int, default=8)
    t2p_sft_pilot.add_argument("--limit", type=int, default=20, help="training examples per task")
    t2p_sft_pilot.add_argument("--batch-size", type=int, default=4)
    t2p_sft_pilot.add_argument("--steps", type=int, default=400)
    t2p_sft_pilot.add_argument("--learning-rate", type=float, default=1e-3)
    t2p_sft_pilot.add_argument("--max-grad-norm", type=float, default=1.0)
    t2p_sft_pilot.add_argument("--l2-reg-generated-w", type=float, default=1e-3)
    t2p_sft_pilot.add_argument("--seed", type=int, default=777)
    t2p_sft_pilot.add_argument("--device", default="cuda:0")
    t2p_sft_pilot.add_argument(
        "--eval-descriptions",
        default=str(REPO_ROOT / "upstream" / "text-to-lora" / "trained_t2l" / "gemma_2b_t2l" / "args.yaml"),
        help="args.yaml (eval_ds_info) to source held-out benchmark task descriptions from; only the "
        "description text is used (embedded fresh via --condition-encoder), so any released checkpoint's "
        "args.yaml works regardless of --interpreter",
    )
    t2p_sft_pilot.add_argument("--eval-tasks", default="boolq,hellaswag")
    t2p_sft_pilot.add_argument("--eval-limit", type=int, default=20, help="eval examples per family")
    t2p_sft_pilot.add_argument("--eval-variant", type=int, default=0)
    t2p_sft_pilot.add_argument("--eval-split", default="test")
    t2p_sft_pilot.add_argument(
        "--use-icl",
        action="store_true",
        help="see 'run' subcommand's --use-icl; applies the same ICL-prompt/prefill protocol here",
    )
    t2p_sft_pilot.add_argument("--output", default="results/t2p_sft_pilot")
    t2p_sft_pilot.set_defaults(func=_t2p_sft_pilot_command)

    t2p_sft_sweep = subparsers.add_parser(
        "t2p-sft-sweep",
        help="checkpointed counterpart to t2p-sft-pilot: scores held-out accuracy at several step budgets per "
        "adapter under one persistent optimizer, to tell 'still converging' apart from 'already past the point "
        "where held-out generalization peaks' - see PROJECT_PLAN.md's Phase 4 section for why this exists",
    )
    t2p_sft_sweep.add_argument("--tasks-dir", default=str(REPO_ROOT / "upstream" / "text-to-lora" / "tasks"))
    t2p_sft_sweep.add_argument("--tasks", default=_DEFAULT_SFT_TRAIN_TASKS)
    t2p_sft_sweep.add_argument(
        "--decontam-config",
        default=str(REPO_ROOT / "upstream" / "text-to-lora" / "configs" / "hyper_lora_decontam_lol_tasks.yaml"),
        help="see 't2p-sft --decontam-config'",
    )
    t2p_sft_sweep.add_argument("--interpreter", default="Qwen/Qwen3-0.6B")
    t2p_sft_sweep.add_argument(
        "--adapters",
        default="lora,freeze_a_lora,ia3,lokr,fourierft,activation_steering",
        help="see 't2p-sft-pilot --adapters'",
    )
    t2p_sft_sweep.add_argument(
        "--checkpoint-steps",
        default="100,200,400,800,1200",
        help="comma-separated, strictly ascending cumulative step counts to evaluate held-out accuracy at "
        "(e.g. '100,200,400' trains 100 steps, evaluates, trains 100 more to reach 200, evaluates, ...) - one "
        "persistent optimizer across all checkpoints, so this isn't confounded by an Adam-restart discontinuity "
        "at each boundary the way calling t2p-sft-pilot once per budget would be",
    )
    t2p_sft_sweep.add_argument("--condition-encoder", default="Alibaba-NLP/gte-large-en-v1.5")
    t2p_sft_sweep.add_argument("--max-descriptions", type=int, default=8)
    t2p_sft_sweep.add_argument("--limit", type=int, default=20, help="training examples per task")
    t2p_sft_sweep.add_argument("--batch-size", type=int, default=4)
    t2p_sft_sweep.add_argument("--learning-rate", type=float, default=1e-3)
    t2p_sft_sweep.add_argument("--max-grad-norm", type=float, default=1.0)
    t2p_sft_sweep.add_argument("--l2-reg-generated-w", type=float, default=1e-3)
    t2p_sft_sweep.add_argument("--seed", type=int, default=777)
    t2p_sft_sweep.add_argument("--device", default="cuda:0")
    t2p_sft_sweep.add_argument(
        "--eval-descriptions",
        default=str(REPO_ROOT / "upstream" / "text-to-lora" / "trained_t2l" / "gemma_2b_t2l" / "args.yaml"),
        help="see 't2p-sft-pilot --eval-descriptions'",
    )
    t2p_sft_sweep.add_argument("--eval-tasks", default="boolq,hellaswag")
    t2p_sft_sweep.add_argument("--eval-limit", type=int, default=40, help="eval examples per family")
    t2p_sft_sweep.add_argument("--eval-variant", type=int, default=0)
    t2p_sft_sweep.add_argument("--eval-split", default="test")
    t2p_sft_sweep.add_argument(
        "--use-icl", action="store_true", help="see 'run' subcommand's --use-icl"
    )
    t2p_sft_sweep.add_argument("--output", default="results/t2p_sft_sweep")
    t2p_sft_sweep.set_defaults(func=_t2p_sft_sweep_command)

    t2p_synthetic_pilot = subparsers.add_parser(
        "t2p-synthetic-pilot",
        help="lightweight live-SFT setting: a tiny, freshly initialized transformers causal LM (no download) trained "
        "on synthetic algorithmic tasks with exactly known correct answers — validates codec expressiveness "
        "decoupled from real-world task noise, CPU-friendly and network-independent",
    )
    t2p_synthetic_pilot.add_argument(
        "--families", default="", help="comma-separated task families (default: all of copy,reverse,increment,sort,constant)"
    )
    t2p_synthetic_pilot.add_argument(
        "--adapters", default="lora,freeze_a_lora,ia3,lokr,fourierft,activation_steering"
    )
    t2p_synthetic_pilot.add_argument("--vocab-size", type=int, default=16)
    t2p_synthetic_pilot.add_argument("--hidden-size", type=int, default=32)
    t2p_synthetic_pilot.add_argument("--num-layers", type=int, default=2)
    t2p_synthetic_pilot.add_argument("--num-heads", type=int, default=4)
    t2p_synthetic_pilot.add_argument("--num-kv-heads", type=int, default=2)
    t2p_synthetic_pilot.add_argument("--intermediate-size", type=int, default=64)
    t2p_synthetic_pilot.add_argument("--max-position-embeddings", type=int, default=32)
    t2p_synthetic_pilot.add_argument("--seq-len", type=int, default=5, help="digits per synthetic example")
    t2p_synthetic_pilot.add_argument("--condition-encoder", default="Alibaba-NLP/gte-large-en-v1.5")
    t2p_synthetic_pilot.add_argument(
        "--train-descriptions", type=int, default=3, help="description variants per family used for training"
    )
    t2p_synthetic_pilot.add_argument("--examples-per-family", type=int, default=200, help="training examples per epoch")
    t2p_synthetic_pilot.add_argument("--rank", type=int, default=4, help="LoRA/FreezeALoRA/LoKr rank")
    t2p_synthetic_pilot.add_argument(
        "--n-frequency", type=int, default=32,
        help="FourierFT frequency count; must be <= the smallest target module's in_features*out_features "
        "(tiny model dimensions need a much smaller value than the real-model default of 1000)",
    )
    t2p_synthetic_pilot.add_argument("--batch-size", type=int, default=16)
    t2p_synthetic_pilot.add_argument("--steps", type=int, default=400)
    t2p_synthetic_pilot.add_argument("--learning-rate", type=float, default=1e-3)
    t2p_synthetic_pilot.add_argument("--max-grad-norm", type=float, default=1.0)
    t2p_synthetic_pilot.add_argument("--l2-reg-generated-w", type=float, default=1e-3)
    t2p_synthetic_pilot.add_argument("--eval-examples-per-family", type=int, default=50)
    t2p_synthetic_pilot.add_argument("--eval-seed", type=int, default=1234)
    t2p_synthetic_pilot.add_argument("--seed", type=int, default=777)
    t2p_synthetic_pilot.add_argument("--device", default="cpu")
    t2p_synthetic_pilot.add_argument("--output", default="results/t2p_synthetic_pilot")
    t2p_synthetic_pilot.set_defaults(func=_t2p_synthetic_pilot_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
