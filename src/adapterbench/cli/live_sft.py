"""Live end-to-end SFT commands: train a hypernetwork from scratch,
hooking its generated adapter into a frozen interpreter's forward pass, then score
held-out benchmarks.

- `t2p-sft`         single adapter, single run, writes a loss-curve JSON.
- `t2p-sft-pilot`   several adapters x seeds, task-description conditioning.
- `d2p-niah`        document-conditioning variant (synthetic NIAH; early-exit conditioner).
- `t2p-sft-sweep`   checkpointed step-budget sweep under one persistent optimizer.
"""

from __future__ import annotations

import dataclasses
import json
from functools import partial
from pathlib import Path

from ._shared import (
    D2P_LATENT_DIM,
    DEFAULT_CATALOG,
    DEFAULT_SFT_TRAIN_TASKS,
    PILOT_DEFAULT_TARGET_MODULES,
    REPO_ROOT,
    T2L_DECONTAM_CONFIG,
    T2L_EVAL_DESCRIPTIONS,
    T2L_TASKS_DIR,
    ResultRecorder,
    embed_training_conditions,
    load_frozen_interpreter,
    write_json,
)

# D2L-parity hook site for the NIAH document-conditioning command (`d2p-niah`): the
# validated recipe hooks LoRA at `down_proj` (see PROJECT_PLAN.md's D2P section). Kept a
# module constant here rather than in `_shared.py` since only `d2p-niah` uses it.
D2L_PARITY_TARGET_MODULES = {
    "lora": ["down_proj"],
    # Same validated D2L projection site; IA3 changes only the live generated shape.
    "ia3": ["down_proj"],
    # LoKr is another generated weight update, so it uses the same locked D2L hook
    # site. Its factor geometry is determined solely by this linear projection.
    "lokr": ["down_proj"],
    # LoHa is another additive weight-space update at the same locked D2L site.
    "loha": ["down_proj"],
    "fourierft": ["down_proj"],
}


def _t2p_sft_command(args) -> None:
    import torch

    from ..t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes
    from ..t2p.lol_data import LolSFTDataset, lol_collate_fn, validate_training_tasks
    from ..t2p.sft_trainer import train_downstream_hypernetwork

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
    tokenizer, interpreter, layers = load_frozen_interpreter(args.interpreter, args.device)
    module_shapes = infer_module_shapes(layers, target_modules, hidden_size=interpreter.config.hidden_size)

    print(f"[2/5] embedding task descriptions via {args.condition_encoder}...", flush=True)
    encoder_model, _, metadata_by_task, embeddings_by_task, condition_dim = embed_training_conditions(
        args.condition_encoder, args.device, tasks_dir, task_ids, args.max_descriptions
    )
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
    write_json(
        args.output,
        {
            "interpreter": args.interpreter,
            "adapter": args.adapter,
            "target_modules": target_modules,
            "tasks": task_ids,
            "steps": stats.steps,
            "initial_loss": stats.initial_loss,
            "final_loss": stats.final_loss,
            "losses": list(stats.losses),
        },
    )
    print(f"wrote {args.output}", flush=True)


def _t2p_sft_pilot_command(args) -> None:
    import torch

    from ..task_examples import build_all_task_examples, load_task_descriptions
    from ..t2p.condition_encoder import embed_task_descriptions
    from ..t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes
    from ..t2p.live_evaluator import HypernetworkDownstreamEvaluator
    from ..t2p.lol_data import (
        LolSFTDataset,
        load_decontaminated_train_task_ids,
        lol_collate_fn,
        validate_training_tasks,
    )
    from ..t2p.sft_trainer import train_downstream_hypernetwork, train_downstream_hypernetwork_restartable

    if args.all_decontam_tasks:
        task_ids = sorted(load_decontaminated_train_task_ids(args.decontam_config))
    else:
        task_ids = args.tasks.split(",")
    adapters = args.adapters.split(",")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    validate_training_tasks(task_ids, args.decontam_config)
    eval_task_ids = args.eval_tasks.split(",")
    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] loading interpreter {args.interpreter}...", flush=True)
    tokenizer, interpreter, layers = load_frozen_interpreter(args.interpreter, args.device)

    print(f"[2/6] embedding task descriptions via {args.condition_encoder}...", flush=True)
    encoder_model, encoder_tokenizer, metadata_by_task, train_embeddings_by_task, condition_dim = (
        embed_training_conditions(args.condition_encoder, args.device, tasks_dir, task_ids, args.max_descriptions)
    )
    eval_descriptions = load_task_descriptions(args.eval_descriptions)
    eval_condition_embeddings = {
        family: embed_task_descriptions(
            [eval_descriptions[family][args.eval_variant]], encoder_model, encoder_tokenizer
        )[0].to(args.device)
        for family in eval_task_ids
    }
    # Mismatched-description control (invariant #1). Two modes:
    #  - `adversarial` (--adversarial-control): score each family with an adapter generated from a
    #    maximally *dissimilar / meaningless* description (Text-to-LoRA's own `additional_eval_descs`
    #    - e.g. "dogs;cats;bananas;", random noise). The STRONG control: if a genuine task
    #    description and junk produce the same gain, the adapter is not using the description. This
    #    avoids the weak-swap confound of deranging among the (all similar QA) eval families.
    #  - else derange the eval families' own descriptions (weak swap; needs >=2 families).
    _families = list(eval_condition_embeddings)
    if args.adversarial_control:
        import yaml

        if args.adversarial_descs:
            adv_descs = [d for d in args.adversarial_descs.split("||") if d]
        else:
            adv_descs = yaml.safe_load(Path(args.decontam_config).read_text()).get("additional_eval_descs", [])
        if not adv_descs:
            raise ValueError("--adversarial-control set but no adversarial descriptions found "
                             "(decontam yaml has no additional_eval_descs; pass --adversarial-descs)")
        adv_emb = [embed_task_descriptions([d], encoder_model, encoder_tokenizer)[0].to(args.device) for d in adv_descs]
        # each family cycles through the adversarial descriptions (deterministic, order-stable)
        mismatched_condition_embeddings = {fam: adv_emb[i % len(adv_emb)] for i, fam in enumerate(_families)}
        print(f"[control] adversarial mismatched control from {len(adv_descs)} dissimilar descriptions", flush=True)
    elif len(_families) >= 2:
        mismatched_condition_embeddings = {
            fam: eval_condition_embeddings[other] for fam, other in zip(_families, _families[1:] + _families[:1])
        }
    else:
        mismatched_condition_embeddings = None
        print("[warn] one eval family and no --adversarial-control - mismatched control unavailable", flush=True)
    del encoder_model
    scales = [float(s) for s in args.scales.split(",")] if args.scales else [None]

    print(f"[3/6] loading + tokenizing {len(task_ids)} training task(s) from {args.tasks_dir}...", flush=True)
    datasets = [
        LolSFTDataset(tokenizer, metadata_by_task[task_id], train_embeddings_by_task[task_id], limit=args.limit)
        for task_id in task_ids
    ]
    dataset = torch.utils.data.ConcatDataset(datasets)
    print(f"[3/6] built {len(dataset)} example(s) across {len(datasets)} task(s)", flush=True)

    print(f"[4/6] building {args.eval_limit} eval example(s) per family for {eval_task_ids}...", flush=True)
    eval_examples_by_family = build_all_task_examples(
        eval_task_ids, eval_descriptions, args.eval_limit, variant=args.eval_variant, use_icl=args.use_icl
    )
    all_eval_examples = [example for group in eval_examples_by_family.values() for example in group]

    recorder = ResultRecorder(output_dir, lambda r: f"  {r.adapter:20} {r.task_id:16} {r.metrics}")
    loss_curves: dict = {}

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
        recorder.record(dataclasses.replace(result, metadata={**result.metadata, "seed": None}))

    warmup_steps = int(args.warmup_frac * args.steps)
    for seed in seeds:
        # Batch order depends on the seed too (not just weight init/dropout - see
        # _t2p_sft_command's comment) - rebuilt per seed, reusing the already-tokenized
        # `dataset` so the (often expensive, network-bound) per-task data loading above
        # only ever happens once regardless of how many seeds are requested.
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            collate_fn=partial(lol_collate_fn, pad_token_id=tokenizer.pad_token_id),
        )
        batches = [batch.to(args.device) for batch in dataloader]
        print(f"[6/6] seed={seed}: {len(batches)} batch(es)/epoch", flush=True)

        for adapter in adapters:
          for scale in scales:
            # Re-seeded per (seed, adapter, scale) - not just once at the top of this function -
            # so each adapter's weight init/dropout trajectory is independent of which
            # adapters were trained before it in this same process/loop position, and each
            # requested seed actually produces an independent run.
            torch.manual_seed(seed)
            target_modules = PILOT_DEFAULT_TARGET_MODULES[adapter]
            scale_tag = "default" if scale is None else f"{scale:g}"
            print(f"[6/6] seed={seed} adapter={adapter} scale={scale_tag} target_modules={target_modules}: training...", flush=True)
            module_shapes = infer_module_shapes(layers, target_modules, hidden_size=interpreter.config.hidden_size)
            hypernetwork = TextToPeftHypernetwork(
                condition_dim=condition_dim,
                module_shapes=module_shapes,
                num_layers=len(layers),
                adapter=adapter,
                seed=seed,
            ).to(args.device)
            # Per-codec LoRA-scale sweep (invariant #2): apply the swept scale directly to each
            # codec before training (as d2p-niah does), so "shape matters" means "even at its own
            # best scale" - not an artifact of a fixed default. --scales empty => codec default.
            if scale is not None:
                from ..t2p.codecs import LoRACodec

                for codec in hypernetwork.codecs.values():
                    if isinstance(codec, LoRACodec):
                        codec.scaling = scale
            # For long runs, --checkpoint-every > 0 uses the restart-safe trainer (atomic
            # model+optimizer+step checkpoint, resumes if the run is killed/restarted).
            if args.checkpoint_every > 0:
                stats = train_downstream_hypernetwork_restartable(
                    hypernetwork, interpreter, layers, batches,
                    steps=args.steps, learning_rate=args.learning_rate,
                    checkpoint_path=output_dir / f"ckpt_{adapter}_seed{seed}_scale{scale_tag}.pt",
                    max_grad_norm=args.max_grad_norm, l2_reg_generated_w=args.l2_reg_generated_w,
                    grad_accum_steps=args.grad_accum_steps, warmup_steps=warmup_steps,
                    checkpoint_every=args.checkpoint_every,
                )
            else:
                stats = train_downstream_hypernetwork(
                    hypernetwork,
                    interpreter,
                    layers,
                    batches,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    max_grad_norm=args.max_grad_norm,
                    l2_reg_generated_w=args.l2_reg_generated_w,
                    grad_accum_steps=args.grad_accum_steps,
                    warmup_steps=warmup_steps,
                )
            print(f"  seed={seed} {adapter} scale={scale_tag}: initial_loss={stats.initial_loss:.4f} final_loss={stats.final_loss:.4f}", flush=True)
            loss_curves.setdefault(str(seed), {})[f"{adapter}::scale{scale_tag}"] = {
                "target_modules": target_modules,
                "lora_scale": scale_tag,
                "initial_loss": stats.initial_loss,
                "final_loss": stats.final_loss,
                "steps": stats.steps,
                "losses": list(stats.losses),
            }
            (output_dir / "loss_curves.json").write_text(json.dumps(loss_curves, indent=2) + "\n")

            print(f"[6/6] seed={seed} adapter={adapter} scale={scale_tag}: evaluating (matched + mismatched control)...", flush=True)
            hypernetwork.eval()  # disable dropout for deterministic held-out scoring
            evaluator = HypernetworkDownstreamEvaluator(
                interpreter,
                layers,
                hypernetwork,
                tokenizer,
                trial_id=f"t2p_sft_pilot::{adapter}::seed{seed}::scale{scale_tag}",
                device=args.device,
                use_icl=args.use_icl,
            )
            for result in evaluator.iter_evaluate(
                eval_condition_embeddings, all_eval_examples, split=args.eval_split,
                mismatched_embeddings=mismatched_condition_embeddings,
            ):
                recorder.record(dataclasses.replace(
                    result, metadata={**result.metadata, "seed": seed, "lora_scale": scale_tag}))

            del hypernetwork, evaluator
            torch.cuda.empty_cache()

    print(f"wrote {output_dir}/results.jsonl, {output_dir}/results.csv, {output_dir}/loss_curves.json", flush=True)


def _d2p_niah_command(args) -> None:
    """Doc-to-LoRA-parity NIAH training through the shared codec seam: the validated
    document-conditioned config for held-out needle retrieval (see PROJECT_PLAN.md's D2L
    section). Trains one hypernetwork per `--adapters` entry with the
    early-exit context encoder + Perceiver-IO generation path (`EarlyExitPerceiverConditioner`)
    on generic-needle chat-tokenized NIAH documents, checkpointing model+optimizer every eval
    (restart-safe) and logging exact-digit `accuracy` AND `accuracy_ctxswap` per eval - never
    gating on loss (gotcha #9). Additional committed codecs plug into the same recipe
    unchanged so the document-conditioning comparison varies the generated representation.
    """
    import torch

    from ..t2p.document_conditioning import EarlyExitPerceiverConditioner
    from ..t2p.document_sft_trainer import train_doc_niah_checkpointed
    from ..t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes
    from ..t2p.live_evaluator import DocumentHypernetworkDownstreamEvaluator
    from ..t2p.niah_data import (
        DocSFTDataset,
        assert_context_fits_in_one_pass,
        build_niah_eval_examples,
        doc_collate_fn,
    )

    torch.manual_seed(args.seed)
    context_lengths = [int(length) for length in args.context_lengths.split(",")]
    # Decouple train vs eval lengths: --eval-context-lengths turns a single-length "does
    # LoRA hit 1.0" run into a length-generalization curve (train short, eval a sweep out
    # to lengths never seen in training). Empty (default) => eval at the training lengths,
    # so every pre-existing invocation is byte-for-byte unchanged.
    eval_context_lengths = (
        [int(length) for length in args.eval_context_lengths.split(",")]
        if args.eval_context_lengths
        else context_lengths
    )
    adapters = args.adapters.split(",")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading interpreter {args.interpreter}...", flush=True)
    tokenizer, interpreter, layers = load_frozen_interpreter(args.interpreter, args.device)
    hidden_size = interpreter.config.hidden_size
    num_layers = len(layers)
    exit_layer = args.exit_layer if args.exit_layer > 0 else max(1, num_layers // 4)
    lora_scaling = args.lora_scaling if args.lora_scaling > 0 else 2 * args.rank**1.5
    # Both train and eval documents are packed into one context window (gotcha #5), so
    # every eval length must fit too - length generalization pushes eval far past training.
    for context_length in sorted(set(context_lengths) | set(eval_context_lengths)):
        assert_context_fits_in_one_pass(context_length, interpreter.config.max_position_embeddings)
    print(
        f"[1/5] num_layers={num_layers} exit_layer={exit_layer} lora_scaling={lora_scaling:.3f} "
        f"ia3_scaling={args.ia3_scaling:.3f} "
        f"lokr_scaling={args.lokr_scaling:.3f} "
        f"loha_scaling={args.loha_scaling:.3f} "
        f"fourierft_scaling={args.fourierft_scaling:.3f} "
        f"needle_style={args.needle_style} numeric_decoy_count={args.numeric_decoy_count} "
        f"train_lengths={context_lengths} eval_lengths={eval_context_lengths}",
        flush=True,
    )

    train_items = None
    collate = None
    if args.evaluate_checkpoint is None:
        print(f"[2/5] building {args.num_train_documents} {args.needle_style} NIAH training documents...", flush=True)
        dataset = DocSFTDataset(
            tokenizer, num_examples=args.num_train_documents, context_lengths=context_lengths,
            seed=args.seed, needle_style=args.needle_style, numeric_decoy_count=args.numeric_decoy_count,
        )
        # Pass the raw per-doc items (not pre-collated batches) so train_doc_niah_checkpointed
        # can reform batches from a document-level reshuffle every epoch (see its docstring - a
        # frozen single-shuffle materialisation slows NIAH's phase transition).
        train_items = [dataset[i] for i in range(len(dataset))]
        collate = partial(doc_collate_fn, pad_token_id=tokenizer.pad_token_id)
    else:
        if len(adapters) != 1:
            raise ValueError("--evaluate-checkpoint requires exactly one adapter")
        if not args.evaluate_checkpoint.exists():
            raise FileNotFoundError(f"missing evaluation checkpoint: {args.evaluate_checkpoint}")
        print(f"[2/5] evaluation-only: loading {args.evaluate_checkpoint} after held-out data setup...", flush=True)

    print(f"[3/5] building {args.eval_limit} held-out NIAH eval documents per bin {eval_context_lengths}...", flush=True)
    eval_examples_by_family = build_niah_eval_examples(
        tokenizer, eval_context_lengths, examples_per_bin=args.eval_limit, needle_style=args.needle_style,
        numeric_decoy_count=args.numeric_decoy_count, seed=args.eval_seed,
    )

    recorder = ResultRecorder(output_dir, lambda r: f"  {r.adapter:20} {r.task_id:16} {r.metrics}")
    loss_curves: dict = {}
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(args.warmup_frac * args.steps)

    print("[4/5] scoring frozen interpreter baseline (no document access)...", flush=True)
    frozen_evaluator = DocumentHypernetworkDownstreamEvaluator(
        interpreter, layers, None, tokenizer, trial_id="d2p_niah::frozen_interpreter", device=args.device,
    )
    for result in frozen_evaluator.iter_evaluate_frozen(eval_examples_by_family, split=args.eval_split):
        recorder.record(dataclasses.replace(result, metadata={**result.metadata, "adapter_family": "frozen"}))

    for adapter in adapters:
        torch.manual_seed(args.seed)
        target_modules = D2L_PARITY_TARGET_MODULES[adapter]
        print(f"[5/5] adapter={adapter} target_modules={target_modules}: training...", flush=True)
        module_shapes = infer_module_shapes(layers, target_modules, hidden_size=hidden_size)
        conditioner = EarlyExitPerceiverConditioner(
            hidden_size=hidden_size, task_dim=D2P_LATENT_DIM // 2, num_layers=num_layers,
            exit_layer=exit_layer, n_latents=args.n_latents, num_blocks=args.num_blocks, seed=args.seed,
        )
        hypernetwork = TextToPeftHypernetwork(
            module_shapes=module_shapes, num_layers=num_layers, adapter=adapter,
            latent_dim=D2P_LATENT_DIM, rank=args.rank, seed=args.seed, conditioner=conditioner,
            ia3_scaling=args.ia3_scaling, lokr_scaling=args.lokr_scaling, loha_scaling=args.loha_scaling,
            fourierft_scaling=args.fourierft_scaling,
        ).to(args.device)
        # LoRA retains its established parity scale. Other codecs each receive only
        # their own explicit scale argument; their locators start from the respective
        # identity parameterization rather than inheriting another codec's result.
        from ..t2p.codecs import FourierFTCodec, IA3Codec, LoHaCodec, LoKrCodec, LoRACodec

        scaled_types = (LoRACodec,)
        for codec in hypernetwork.codecs.values():
            if isinstance(codec, scaled_types):
                codec.scaling = lora_scaling
            elif isinstance(codec, IA3Codec):
                codec.scaling = args.ia3_scaling
            elif isinstance(codec, LoKrCodec):
                codec.scaling = args.lokr_scaling
            elif isinstance(codec, LoHaCodec):
                codec.scaling = args.loha_scaling
            elif isinstance(codec, FourierFTCodec):
                codec.scaling = args.fourierft_scaling

        evaluator = DocumentHypernetworkDownstreamEvaluator(
            interpreter, layers, hypernetwork, tokenizer,
            trial_id=f"d2p_niah::{adapter}", device=args.device,
        )

        if args.evaluate_checkpoint is not None:
            checkpoint = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
            hypernetwork.load_state_dict(checkpoint["model"])
            hypernetwork.eval()
            for result in evaluator.iter_evaluate(eval_examples_by_family, split=args.eval_split):
                recorder.record(dataclasses.replace(
                    result,
                    metadata={
                        **result.metadata, "adapter_family": adapter, "steps": checkpoint["step"],
                        "lokr_scaling": args.lokr_scaling, "loha_scaling": args.loha_scaling,
                        "fourierft_scaling": args.fourierft_scaling, "evaluation_only": True,
                    },
                ))
            del hypernetwork, evaluator
            torch.cuda.empty_cache()
            continue

        def _evaluate(net, step, _adapter=adapter, _evaluator=evaluator):
            metrics = {}
            for result in _evaluator.iter_evaluate(eval_examples_by_family, split=args.eval_split):
                metrics[result.task_id] = {
                    "accuracy": result.metrics.get("accuracy"),
                    "accuracy_ctxswap": result.metrics.get("accuracy_ctxswap"),
                }
            return metrics

        history = train_doc_niah_checkpointed(
            hypernetwork, interpreter, layers, train_items,
            collate=collate, device=args.device, batch_size=args.batch_size,
            steps=args.steps, eval_every=args.eval_every, learning_rate=args.learning_rate,
            evaluate=_evaluate, checkpoint_path=output_dir / f"ckpt_{adapter}.pt",
            l2_reg_generated_w=args.l2_reg_generated_w, grad_accum_steps=args.grad_accum_steps,
            warmup_steps=warmup_steps,
        )
        loss_curves[adapter] = history
        (output_dir / "history.json").write_text(json.dumps(loss_curves, indent=2) + "\n")

        # Record the final eval as EvaluationResults (matched + ctxswap per family).
        hypernetwork.eval()
        for result in evaluator.iter_evaluate(eval_examples_by_family, split=args.eval_split):
            recorder.record(dataclasses.replace(
                result,
                metadata={
                    **result.metadata, "adapter_family": adapter, "steps": args.steps,
                    "lokr_scaling": args.lokr_scaling, "loha_scaling": args.loha_scaling,
                    "fourierft_scaling": args.fourierft_scaling,
                },
            ))
        del hypernetwork, evaluator
        torch.cuda.empty_cache()

    print(f"wrote {output_dir}/results.jsonl, {output_dir}/history.json", flush=True)


def _t2p_sft_sweep_command(args) -> None:
    """Checkpointed counterpart to `t2p-sft-pilot`: scores held-out accuracy at several
    step budgets per adapter under one persistent optimizer (`t2p.sft_trainer.
    train_with_checkpoints`), instead of only ever reporting a single fixed-step endpoint.

    Originally motivated by an apparent finding that longer training made held-out
    accuracy worse for every adapter - but running
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
    import torch

    from ..task_examples import build_all_task_examples, load_task_descriptions
    from ..t2p.condition_encoder import embed_task_descriptions
    from ..t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes
    from ..t2p.live_evaluator import HypernetworkDownstreamEvaluator
    from ..t2p.lol_data import LolSFTDataset, lol_collate_fn, validate_training_tasks
    from ..t2p.sft_trainer import train_with_checkpoints

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
    tokenizer, interpreter, layers = load_frozen_interpreter(args.interpreter, args.device)

    print(f"[2/6] embedding task descriptions via {args.condition_encoder}...", flush=True)
    encoder_model, encoder_tokenizer, metadata_by_task, train_embeddings_by_task, condition_dim = (
        embed_training_conditions(args.condition_encoder, args.device, tasks_dir, task_ids, args.max_descriptions)
    )
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

    recorder = ResultRecorder(
        output_dir,
        lambda r: f"  step={r.metadata['checkpoint_step']:<5} {r.adapter:20} {r.task_id:16} {r.metrics}",
    )
    loss_curves: dict = {}

    print("[5/6] scoring frozen interpreter baseline (checkpoint_step=0)...", flush=True)
    frozen_evaluator = HypernetworkDownstreamEvaluator(
        interpreter, layers, None, tokenizer, trial_id="t2p_sft_sweep::frozen_interpreter", device=args.device,
        use_icl=args.use_icl,
    )
    for result in frozen_evaluator.iter_evaluate_frozen(all_eval_examples, split=args.eval_split):
        recorder.record(dataclasses.replace(result, metadata={**result.metadata, "checkpoint_step": 0}))

    for adapter in adapters:
        # Re-seeded per adapter, not just once at the top - see t2p-sft-pilot's identical
        # comment for why (each adapter's trajectory must be independent of loop position).
        torch.manual_seed(args.seed)
        target_modules = PILOT_DEFAULT_TARGET_MODULES[adapter]
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

        hypernetwork.eval()  # disable dropout for deterministic held-out scoring
        for step in checkpoint_steps:
            print(f"  {adapter} @ step {step}: final_loss={stats_by_checkpoint[step].final_loss:.4f} - evaluating...", flush=True)
            evaluator = HypernetworkDownstreamEvaluator(
                interpreter, layers, hypernetwork, tokenizer, trial_id=f"t2p_sft_sweep::{adapter}::step{step}",
                device=args.device, use_icl=args.use_icl,
            )
            for result in evaluator.iter_evaluate(eval_condition_embeddings, all_eval_examples, split=args.eval_split):
                recorder.record(dataclasses.replace(result, metadata={**result.metadata, "checkpoint_step": step}))
            del evaluator

        del hypernetwork
        torch.cuda.empty_cache()

    print(f"wrote {output_dir}/results.jsonl, {output_dir}/results.csv, {output_dir}/loss_curves.json", flush=True)


def register(subparsers) -> None:
    _register_t2p_sft(subparsers)
    _register_t2p_sft_pilot(subparsers)
    _register_d2p_niah(subparsers)
    _register_t2p_sft_sweep(subparsers)


def _register_d2p_niah(subparsers) -> None:
    p = subparsers.add_parser(
        "d2p-niah",
        help="Doc-to-LoRA-parity NIAH training through the shared codec seam: early-exit context encoder + "
        "Perceiver-IO generation path on generic-needle chat-tokenized documents, restart-safe "
        "(checkpoints model+optimizer every eval), logging exact-digit accuracy AND accuracy_ctxswap per "
        "eval. This is the validated document-conditioned recipe for held-out NIAH retrieval; pass "
        "several --adapters as additional codecs are committed.",
    )
    p.add_argument("--interpreter", default="Qwen/Qwen3-0.6B")
    p.add_argument("--adapters", default="lora", help="comma-separated codecs (see 't2p-sft-pilot --adapters')")
    p.add_argument("--needle-style", default="generic", choices=["generic", "realistic", "realistic_numeric_decoys"],
                   help="generic = parity format; realistic = real prose; realistic_numeric_decoys = one target plus irrelevant numeric decoys")
    p.add_argument("--numeric-decoy-count", type=int, default=0,
                   help="distinct irrelevant four-digit values for realistic_numeric_decoys; must be 0 for other styles")
    p.add_argument("--evaluate-checkpoint", type=Path,
                   help="skip training and score one saved D2L checkpoint on freshly generated held-out documents")
    p.add_argument("--context-lengths", default="384", help="comma-separated NIAH document token lengths (training)")
    p.add_argument("--eval-context-lengths", default="",
                   help="comma-separated NIAH document token lengths for held-out eval; empty (default) => eval at "
                        "the training --context-lengths. Set to a longer sweep (e.g. 256,512,1024,2048,4096,8192) to "
                        "measure length generalization: train short, eval far beyond the training length.")
    p.add_argument("--num-train-documents", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--learning-rate", type=float, default=4e-5, help="D2L NIAH parity lr")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora-scaling", type=float, default=-1.0,
                   help="LoRA scale applied directly; <=0 (default) computes D2L parity 2*r^1.5 (=45.25 at r=8)")
    p.add_argument("--ia3-scaling", type=float, default=1.0,
                   help="IA3 multiplier scale in W -> diag(1 + scale*v) W; sweep this per codec.")
    p.add_argument("--lokr-scaling", type=float, default=1.0,
                   help="LoKr scale in ΔW = scale * (L ⊗ R); sweep from its identity convention per codec.")
    p.add_argument("--loha-scaling", type=float, default=1.0,
                   help="LoHa scale in ΔW = scale * (B1@A1) ⊙ (B2@A2); sweep from its identity convention per codec.")
    p.add_argument("--fourierft-scaling", type=float, default=1.0,
                   help="FourierFT coefficient scale; sweep from its identity convention per codec.")
    p.add_argument("--scale-weight-codecs", action="store_true",
                   help="also apply --lora-scaling to any other LoRA-family weight codecs added later "
                        "(none in the LoRA-only baseline, so currently a no-op) so a multi-codec "
                        "comparison isn't confounded by only LoRA getting the load-bearing scale")
    p.add_argument("--l2-reg-generated-w", type=float, default=0.0, help="D2L NIAH parity uses ~0 (see gotcha)")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--warmup-steps", type=int,
                   help="override --warmup-frac with an absolute count; use a fixed final-budget value for checkpoint promotion")
    p.add_argument("--exit-layer", type=int, default=-1, help="early-exit ctx encoder depth; <=0 => num_layers//4")
    p.add_argument("--n-latents", type=int, default=208, help="Perceiver-IO latent queries (D2L parity: 208)")
    p.add_argument("--num-blocks", type=int, default=8, help="Perceiver-IO cross-attention blocks (D2L parity: 8)")
    p.add_argument("--eval-limit", type=int, default=32, help="held-out eval documents per context-length bin")
    p.add_argument("--eval-seed", type=int, default=778,
                   help="deterministic held-out document seed; keep development and final evaluation disjoint")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="results/d2p_niah")
    p.set_defaults(func=_d2p_niah_command)


def _register_t2p_sft(subparsers) -> None:
    t2p_sft = subparsers.add_parser(
        "t2p-sft",
        help="live end-to-end SFT: hook the hypernetwork's generated output into a frozen interpreter and train on real next-token loss",
    )
    t2p_sft.add_argument("--tasks-dir", default=str(T2L_TASKS_DIR))
    t2p_sft.add_argument("--tasks", default=DEFAULT_SFT_TRAIN_TASKS)
    t2p_sft.add_argument(
        "--decontam-config",
        default=str(T2L_DECONTAM_CONFIG),
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


def _register_t2p_sft_pilot(subparsers) -> None:
    t2p_sft_pilot = subparsers.add_parser(
        "t2p-sft-pilot",
        help="multi-task live-SFT pilot: train the selected codecs on a shared task corpus, "
        "then score each via HypernetworkDownstreamEvaluator against real held-out benchmark examples",
    )
    t2p_sft_pilot.add_argument("--tasks-dir", default=str(T2L_TASKS_DIR))
    t2p_sft_pilot.add_argument("--tasks", default=DEFAULT_SFT_TRAIN_TASKS)
    t2p_sft_pilot.add_argument(
        "--all-decontam-tasks",
        action="store_true",
        help="train on T2L's full 479-task decontaminated split (--decontam-config's train_ds_names) instead "
        "of --tasks - matches Text-to-LoRA's own training scale rather than this project's small-pilot default",
    )
    t2p_sft_pilot.add_argument(
        "--decontam-config",
        default=str(T2L_DECONTAM_CONFIG),
        help="see 't2p-sft --decontam-config'",
    )
    t2p_sft_pilot.add_argument("--interpreter", default="Qwen/Qwen3-0.6B")
    t2p_sft_pilot.add_argument(
        "--adapters",
        default="lora",
        help="comma-separated adapters to train and compare; each uses a fixed default hook site "
        "(lora -> q_proj,v_proj). LoRA is the only baseline codec; more are added one at a time, "
        "each with its own leaderboard entry (see PROJECT_PLAN.md).",
    )
    t2p_sft_pilot.add_argument("--condition-encoder", default="Alibaba-NLP/gte-large-en-v1.5")
    t2p_sft_pilot.add_argument("--max-descriptions", type=int, default=8)
    t2p_sft_pilot.add_argument("--limit", type=int, default=20, help="training examples per task")
    t2p_sft_pilot.add_argument("--batch-size", type=int, default=4)
    t2p_sft_pilot.add_argument("--steps", type=int, default=400)
    t2p_sft_pilot.add_argument("--learning-rate", type=float, default=1e-3)
    t2p_sft_pilot.add_argument("--max-grad-norm", type=float, default=1.0)
    t2p_sft_pilot.add_argument("--l2-reg-generated-w", type=float, default=1e-3)
    t2p_sft_pilot.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="micro-batches accumulated per optimizer step (matches upstream's grad_accum_steps - see "
        "hyper_lora_decontam_lol_tasks.yaml's batch_size=4/grad_accum_steps=64 for an effective batch of 256)",
    )
    t2p_sft_pilot.add_argument(
        "--warmup-frac", type=float, default=0.0,
        help="fraction of --steps spent on linear LR warmup before holding constant (upstream's own recipe "
        "uses 0.1); 0 (default) disables warmup entirely, matching prior behavior",
    )
    t2p_sft_pilot.add_argument("--seed", type=int, default=777)
    t2p_sft_pilot.add_argument(
        "--seeds", default="",
        help="comma-separated seeds to run every adapter under (overrides --seed if set); the expensive "
        "per-task data loading happens once and is shared across all requested seeds",
    )
    t2p_sft_pilot.add_argument("--device", default="cuda:0")
    t2p_sft_pilot.add_argument(
        "--eval-descriptions",
        default=str(T2L_EVAL_DESCRIPTIONS),
        help="args.yaml (eval_ds_info) to source held-out benchmark task descriptions from; only the "
        "description text is used (embedded fresh via --condition-encoder), so any released checkpoint's "
        "args.yaml works regardless of --interpreter",
    )
    t2p_sft_pilot.add_argument(
        "--scales", default="",
        help="comma-separated LoRA scales to sweep (invariant #2): trains one hypernetwork per scale "
        "with the scale applied directly to each codec, and reports per-scale best-of. Empty (default) "
        "uses the codec's own default scale (single run).",
    )
    t2p_sft_pilot.add_argument(
        "--adversarial-control", action="store_true",
        help="strong mismatched control: score each family with an adapter generated from a maximally "
        "dissimilar/meaningless description (the decontam yaml's additional_eval_descs) instead of "
        "deranging the (similar) eval-family descriptions - avoids the weak-swap confound (see PROJECT_PLAN "
        "T2L rigor).",
    )
    t2p_sft_pilot.add_argument(
        "--adversarial-descs", default="",
        help="'||'-separated adversarial description strings to override the decontam yaml's "
        "additional_eval_descs for --adversarial-control.",
    )
    t2p_sft_pilot.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="if >0, train with the restart-safe trainer, checkpointing hypernetwork+optimizer "
        "every N steps to results/<output>/ckpt_*.pt and resuming from it on re-run - use for "
        "long (multi-hour/day) runs so a crash or restart does not lose progress.",
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


def _register_t2p_sft_sweep(subparsers) -> None:
    t2p_sft_sweep = subparsers.add_parser(
        "t2p-sft-sweep",
        help="checkpointed counterpart to t2p-sft-pilot: scores held-out accuracy at several step budgets per "
        "adapter under one persistent optimizer, to tell 'still converging' apart from 'already past the point "
        "where held-out generalization peaks'",
    )
    t2p_sft_sweep.add_argument("--tasks-dir", default=str(T2L_TASKS_DIR))
    t2p_sft_sweep.add_argument("--tasks", default=DEFAULT_SFT_TRAIN_TASKS)
    t2p_sft_sweep.add_argument(
        "--decontam-config",
        default=str(T2L_DECONTAM_CONFIG),
        help="see 't2p-sft --decontam-config'",
    )
    t2p_sft_sweep.add_argument("--interpreter", default="Qwen/Qwen3-0.6B")
    t2p_sft_sweep.add_argument(
        "--adapters",
        default="lora",
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
        default=str(T2L_EVAL_DESCRIPTIONS),
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
