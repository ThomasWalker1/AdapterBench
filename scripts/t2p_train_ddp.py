"""4-GPU DDP training for the T2L (task-description) live-SFT benchmark.

Why this exists: the single-GPU `t2p-sft-pilot` at batch 8 runs ~0.2 s/step, so a
paper-scale run is ~2 days. This entrypoint keeps the *data budget* fixed (same
example-visits as an N-step batch-8 run) but raises the effective batch via
data-parallelism across GPUs, so far fewer optimizer steps are needed and each step is
GPU-efficient. Combined with `torch.compile` + persistent codec hooks it brings a
benchmark trajectory down to a few hours. Trajectory shape differs from the batch-8
emergence curve *by design* (see PROJECT_PLAN §T2L) - the benchmark ships at this scale.

It deliberately reuses the pilot's data pipeline (`LolSFTDataset`/`lol_collate_fn`),
its held-out eval (`HypernetworkDownstreamEvaluator` + adversarial control), and its
on-disk format (`ResultRecorder` -> results.jsonl/.csv), so the output is a drop-in for
`scripts/t2p_rigor_aggregate.py` and comparable to the existing 150K/1M runs.

Launch (uses physical GPUs 1-4, leaves cuda:0 free):
    CUDA_VISIBLE_DEVICES=1,2,3,4 .venv/bin/torchrun --standalone --nproc_per_node=4 \
        scripts/t2p_train_ddp.py --all-decontam-tasks --max-descriptions 128 \
        --per-gpu-batch 32 --steps 62500 --learning-rate 1e-4 --warmup-frac 0.1 \
        --eval-tasks arc_easy,arc_challenge,hellaswag,boolq --eval-limit 80 \
        --adversarial-control --seed 777 --checkpoint-every 5000 \
        --output results/t2p_cond_ddp/s777

Restart-safe: re-run the identical command; rank 0's checkpoint (model+optimizer+
scheduler+step) is read by every rank and training resumes from the last checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch._dynamo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

# repo src is importable when launched from the repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapterbench.cli._shared import (  # noqa: E402
    PILOT_DEFAULT_TARGET_MODULES,
    ResultRecorder,
    T2L_DECONTAM_CONFIG,
    T2L_EVAL_DESCRIPTIONS,
    T2L_TASKS_DIR,
    embed_training_conditions,
    load_frozen_interpreter,
)
from adapterbench.task_examples import build_all_task_examples, load_task_descriptions  # noqa: E402
from adapterbench.t2p.condition_encoder import embed_task_descriptions  # noqa: E402
from adapterbench.t2p.hypernetwork import (  # noqa: E402
    StaticAdapter,
    TextToPeftHypernetwork,
    _resolve_target,
    infer_module_shapes,
)
from adapterbench.t2p.live_evaluator import HypernetworkDownstreamEvaluator  # noqa: E402
from adapterbench.t2p.lol_data import (  # noqa: E402
    LolSFTDataset,
    load_decontaminated_train_task_ids,
    lol_collate_fn,
    validate_training_tasks,
)
from adapterbench.t2p.sft_trainer import SFTBatch, masked_cross_entropy  # noqa: E402


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def register_persistent_hooks(codecs, layers, current: dict) -> list:
    """Register the codec forward-hooks ONCE (unlike `hypernetwork.apply`'s per-step
    register/remove) so `torch.compile` does not re-trace every step. Each hook reads the
    generated parameters for its layer from the mutable ``current`` dict, refilled each
    step from the (DDP-wrapped) hypernetwork's forward. Codecs are stateless w.r.t.
    trainable params (LoRACodec.apply uses only its scalar ``scaling`` + the generated
    tensor), so referencing them directly here is safe under DDP."""
    handles = []
    for layer_index, layer in enumerate(layers):
        for name, codec in codecs.items():
            module = _resolve_target(layer, name)

            def hook(module, args, output, *, codec=codec, name=name, layer_index=layer_index):
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output
                updated = codec.apply(args[0], hidden, current[name][layer_index], layer_index)
                return (updated, *output[1:]) if is_tuple else updated

            handles.append(module.register_forward_hook(hook))
    return handles


def _linear_warmup_then_constant(optimizer, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interpreter", default="Qwen/Qwen3-0.6B")
    p.add_argument("--condition-encoder", default="Alibaba-NLP/gte-large-en-v1.5")
    p.add_argument("--tasks-dir", default=str(T2L_TASKS_DIR))
    p.add_argument("--tasks", default="")
    p.add_argument("--all-decontam-tasks", action="store_true")
    p.add_argument("--decontam-config", default=str(T2L_DECONTAM_CONFIG))
    p.add_argument("--max-descriptions", type=int, default=128)
    p.add_argument("--limit", type=int, default=40, help="training examples per task")
    p.add_argument("--strip-task-def", action="store_true",
                   help="drop the task definition from the input (`{problem}` only), so the task is "
                        "specified ONLY via the description->hypernetwork. Makes conditioning necessary "
                        "(the frozen input no longer reveals the task); the pipeline fix for T2L.")
    p.add_argument("--per-gpu-batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=62500, help="optimizer steps (each = world_size*per_gpu_batch examples)")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--l2-reg-generated-w", type=float, default=0.0)
    p.add_argument("--contrastive-lambda", type=float, default=0.0,
                   help="if >0, add a mismatched-negative conditioning loss: per step also generate "
                        "an adapter from a WRONG (in-batch rolled) description and penalize "
                        "max(0, margin + CE_matched - CE_mismatched). Forces the description to matter "
                        "(the thing the junk-control measures). Costs a 2nd interpreter forward/step.")
    p.add_argument("--contrastive-margin", type=float, default=0.5,
                   help="target CE gap (mismatched - matched) the contrastive hinge drives toward")
    p.add_argument("--neutral-junk-lambda", type=float, default=0.0,
                   help="NON-GAMEABLE conditioning objective: loss = CE_matched + lambda*(CE_mismatched "
                        "- CE_frozen)^2. Rewards the matched adapter for HELPING while forcing the "
                        "wrong-description adapter to stay NEUTRAL (~frozen). Unlike the contrastive "
                        "hinge, sabotage (junk << frozen) is PENALIZED, so the only way to earn "
                        "matched-junk is for matched to genuinely help. Costs a 3rd (no-grad) forward.")
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--adapter", default="lora")
    p.add_argument("--static", action="store_true",
                   help="train a single directly-optimized adapter of the codec's shape (the "
                        "multi-task-LoRA reference for matched-static) instead of the hypernetwork; "
                        "ignores the condition. Incompatible with the conditioning objectives.")
    p.add_argument("--no-compile", action="store_true", help="disable torch.compile (debugging)")
    p.add_argument("--lora-scaling", type=float, default=-1.0,
                   help="LoRA output scale for the invariant-#2 best-of-scale sweep; <=0 uses the "
                        "codec default. Applied to each LoRA codec before training (as the pilot's "
                        "--scales does), so a scale sweep at the shipped DDP recipe is a per-value run.")
    p.add_argument("--ia3-scaling", type=float, default=1.0,
                   help="IA3 multiplier scale in W -> diag(1 + scale*v) W; select it with a per-codec sweep.")
    p.add_argument("--lokr-scaling", type=float, default=1.0,
                   help="LoKr Kronecker-update scale; select it with a codec-specific geometric sweep.")
    p.add_argument("--loha-scaling", type=float, default=1.0,
                   help="LoHa Hadamard-update scale; select it with a codec-specific geometric sweep.")
    p.add_argument(
        "--fixed-seq-len", type=int, default=0,
        help="pad every batch to this fixed length instead of the per-batch max. Numerically "
        "identical (pad tokens are masked in attention and loss) but gives static shapes: "
        "torch.compile traces ONE graph (no dynamic-shape thrash) and all DDP ranks do equal "
        "work each step (no variable-length straggler). Recommended = the tokenizer truncation "
        "cap (512). 0 = dynamic per-batch padding.",
    )
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="if >0, save model-only weights to snapshots/step{n}.pt at this cadence "
                        "(+final) for post-hoc training-curve eval via scripts/t2p_eval_checkpoint.py")
    p.add_argument("--loss-log-every", type=int, default=500)
    # eval (rank 0 only, after training) - mirrors t2p-sft-pilot
    p.add_argument("--eval-tasks", default="arc_easy,arc_challenge,hellaswag,boolq")
    p.add_argument("--eval-descriptions", default=str(T2L_EVAL_DESCRIPTIONS))
    p.add_argument("--eval-limit", type=int, default=80)
    p.add_argument("--eval-variant", type=int, default=0)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--use-icl", action="store_true")
    p.add_argument("--skip-inline-eval", action="store_true",
                   help="save the completed checkpoint without the legacy self-describing eval; "
                        "use when a caller will run the held-out paired evaluator instead")
    p.add_argument("--adversarial-control", action="store_true")
    p.add_argument("--adversarial-descs", default="")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = build_args()
    torch.set_float32_matmul_precision("high")  # TF32 for the fp32 hypernetwork matmuls

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    eff_batch = world * args.per_gpu_batch
    log(f"[ddp] world={world} per_gpu_batch={args.per_gpu_batch} effective_batch={eff_batch} "
        f"steps={args.steps} -> ~{eff_batch * args.steps / 1e6:.1f}M example-visits")

    torch.manual_seed(args.seed)

    if args.all_decontam_tasks:
        task_ids = sorted(load_decontaminated_train_task_ids(args.decontam_config))
    else:
        task_ids = args.tasks.split(",")
    validate_training_tasks(task_ids, args.decontam_config)
    eval_task_ids = args.eval_tasks.split(",")
    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output)
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[1/6] loading interpreter {args.interpreter} on {device}...")
    tokenizer, interpreter, layers = load_frozen_interpreter(args.interpreter, device)

    log(f"[2/6] embedding task descriptions via {args.condition_encoder}...")
    encoder_model, encoder_tokenizer, metadata_by_task, train_embeddings_by_task, condition_dim = (
        embed_training_conditions(args.condition_encoder, device, tasks_dir, task_ids, args.max_descriptions)
    )
    # eval-side embeddings (kept; encoder is deleted afterwards)
    eval_descriptions = load_task_descriptions(args.eval_descriptions)
    eval_condition_embeddings = {
        family: embed_task_descriptions(
            [eval_descriptions[family][args.eval_variant]], encoder_model, encoder_tokenizer
        )[0].to(device)
        for family in eval_task_ids
    }
    _families = list(eval_condition_embeddings)
    mismatched_condition_embeddings = None
    if args.adversarial_control:
        import yaml

        if args.adversarial_descs:
            adv_descs = [d for d in args.adversarial_descs.split("||") if d]
        else:
            adv_descs = yaml.safe_load(Path(args.decontam_config).read_text()).get("additional_eval_descs", [])
        if not adv_descs:
            raise ValueError("--adversarial-control set but no adversarial descriptions found")
        adv_emb = [embed_task_descriptions([d], encoder_model, encoder_tokenizer)[0].to(device) for d in adv_descs]
        mismatched_condition_embeddings = {fam: adv_emb[i % len(adv_emb)] for i, fam in enumerate(_families)}
    elif len(_families) >= 2:
        mismatched_condition_embeddings = {
            fam: eval_condition_embeddings[other] for fam, other in zip(_families, _families[1:] + _families[:1])
        }
    del encoder_model

    # --limit <= 0 means "use every example in each task" (paper trains on the full datasets; the
    # default 40-example cap, repeated ~hundreds of times, invites a memorized description-agnostic
    # solution — see the T2L conditioning diagnostic).
    per_task_limit = args.limit if args.limit and args.limit > 0 else None
    log(f"[3/6] loading + tokenizing {len(task_ids)} training task(s) (limit={per_task_limit})...")
    datasets = [
        LolSFTDataset(tokenizer, metadata_by_task[task_id], train_embeddings_by_task[task_id],
                     limit=per_task_limit, strip_task_def=args.strip_task_def)
        for task_id in task_ids
    ]
    if args.strip_task_def:
        log("[pipeline] task definition STRIPPED from input — task specified only via description")
    dataset = ConcatDataset(datasets)
    log(f"[3/6] built {len(dataset)} example(s) across {len(datasets)} task(s)")

    # --- model + DDP ---
    torch.manual_seed(args.seed)
    target_modules = PILOT_DEFAULT_TARGET_MODULES[args.adapter]
    module_shapes = infer_module_shapes(layers, target_modules, hidden_size=interpreter.config.hidden_size)
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=condition_dim,
        module_shapes=module_shapes,
        num_layers=len(layers),
        adapter=args.adapter,
        ia3_scaling=args.ia3_scaling,
        lokr_scaling=args.lokr_scaling,
        loha_scaling=args.loha_scaling,
        seed=args.seed,
    ).to(device)
    if args.lora_scaling > 0:
        # invariant-#2 scale sweep: apply the swept output scale directly to each LoRA codec
        # (same as the pilot's --scales), so "best-of-scale" is a per-value run.
        from adapterbench.t2p.codecs import LoRACodec

        for codec in hypernetwork.codecs.values():
            if isinstance(codec, LoRACodec):
                codec.scaling = args.lora_scaling
        log(f"[scale] LoRA codec scaling set to {args.lora_scaling}")
    if args.adapter == "ia3":
        log(f"[scale] IA3 codec scaling set to {args.ia3_scaling}")
    if args.adapter == "lokr":
        log(f"[scale] LoKr codec scaling set to {args.lokr_scaling}")
    if args.adapter == "loha":
        log(f"[scale] LoHa codec scaling set to {args.loha_scaling}")
    if args.static:
        if args.contrastive_lambda > 0 or args.neutral_junk_lambda > 0:
            raise ValueError("--static trains a single unconditioned adapter; it is incompatible "
                             "with the conditioning objectives (--contrastive-lambda/--neutral-junk-lambda)")
        trainable = StaticAdapter(hypernetwork.codecs, len(layers)).to(device)
        log(f"[static] training a single {args.adapter} adapter (multi-task reference), no conditioning")
    else:
        trainable = hypernetwork
    ddp_hyper = DDP(trainable, device_ids=[local_rank], find_unused_parameters=False)

    current: dict = {}
    hook_handles = register_persistent_hooks(hypernetwork.codecs, layers, current)
    if args.no_compile:
        interp_fwd = interpreter
    elif args.fixed_seq_len > 0:
        # Static shapes (every batch padded to --fixed-seq-len): Dynamo traces ONE graph
        # and reuses it forever. This is the recommended, robust compile path.
        interp_fwd = torch.compile(interpreter)
    else:
        # Dynamic per-batch length: a soft dynamic hint (below) plus a raised cache backstop.
        # NOTE: hard mark_dynamic raises ConstraintViolationError here (transformers
        # specializes the seq dim internally), so this path can still recompile; prefer
        # --fixed-seq-len for compile.
        torch._dynamo.config.cache_size_limit = 64
        interp_fwd = torch.compile(interpreter, dynamic=True)

    optimizer = torch.optim.AdamW(ddp_hyper.parameters(), lr=args.learning_rate)
    warmup_steps = int(args.warmup_frac * args.steps)
    scheduler = _linear_warmup_then_constant(optimizer, warmup_steps)

    ckpt = output_dir / f"ckpt_{args.adapter}_seed{args.seed}_scaledefault.pt"
    start, losses = 0, []
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        trainable.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, losses = state["step"], list(state["losses"])
        log(f"[resume] from step {start}/{args.steps}")
    # keep all ranks' weights identical after a resume
    for p in trainable.parameters():
        dist.broadcast(p.data, src=0)

    def save(done: int) -> None:
        if not is_main():
            return
        tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
        torch.save(
            {"model": trainable.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "step": done, "losses": losses},
            tmp,
        )
        tmp.replace(ckpt)

    def save_snapshot(done: int) -> None:
        """Model-only, step-tagged weights for a post-hoc training curve. Decoupled from the
        restart checkpoint (which also carries optimizer/scheduler and is rolling)."""
        if not is_main() or args.snapshot_every <= 0:
            return
        snap_dir = output_dir / "snapshots"
        snap_dir.mkdir(exist_ok=True)
        torch.save({"model": trainable.state_dict(), "step": done, "static": args.static},
                   snap_dir / f"step{done}.pt")

    base_collate = partial(lol_collate_fn, pad_token_id=tokenizer.pad_token_id)
    if args.fixed_seq_len > 0:
        pad_id, fixed_len = tokenizer.pad_token_id, args.fixed_seq_len

        def collate(items):
            b = base_collate(items)
            seq = b.input_ids.shape[1]
            if seq > fixed_len:
                raise ValueError(f"batch seq {seq} exceeds --fixed-seq-len {fixed_len}")
            if seq == fixed_len:
                return b
            n = fixed_len - seq
            return SFTBatch(
                input_ids=torch.nn.functional.pad(b.input_ids, (0, n), value=pad_id),
                attention_mask=torch.nn.functional.pad(b.attention_mask, (0, n), value=0),
                labels=torch.nn.functional.pad(b.labels, (0, n), value=-100),
                condition_embeddings=b.condition_embeddings,
            )
    else:
        collate = base_collate

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=args.per_gpu_batch,
        sampler=sampler,
        collate_fn=collate,
        drop_last=True,
    )

    def epoch_batches():
        epoch = 0
        while True:
            sampler.set_epoch(epoch)
            for batch in loader:
                yield batch
            epoch += 1

    batch_iter = epoch_batches()
    ddp_hyper.train()
    log(f"[4/6] training: {len(dataset) // eff_batch} step(s)/epoch, warmup={warmup_steps}")

    import time
    timer_start = None
    timer_from_step = start + 20  # exclude compile-warmup steps from the throughput estimate
    for step in range(start, args.steps):
        if step == timer_from_step:
            torch.cuda.synchronize(device)
            timer_start = time.perf_counter()
        batch = next(batch_iter).to(device)
        if not args.no_compile and args.fixed_seq_len == 0:
            # soft dynamic hint (won't error if the seq dim must specialize); only for the
            # dynamic-padding path — fixed-seq-len is already static so no hint is needed.
            torch._dynamo.maybe_mark_dynamic(batch.input_ids, 1)
            torch._dynamo.maybe_mark_dynamic(batch.attention_mask, 1)
        optimizer.zero_grad(set_to_none=True)
        _z = torch.zeros((), device=device)
        ce_m_t = ce_mm_t = ce_fz_t = aux_t = _z
        if args.neutral_junk_lambda > 0:
            # Non-gameable objective: matched HELPS + mismatched stays NEUTRAL (~frozen).
            cond = batch.condition_embeddings
            b = cond.shape[0]
            cond_mm = torch.roll(cond, shifts=1, dims=0)
            gen_both = ddp_hyper(torch.cat([cond, cond_mm], dim=0))  # single DDP forward
            gen_m = {k: v[:, :b] for k, v in gen_both.items()}
            gen_mm = {k: v[:, b:] for k, v in gen_both.items()}
            # frozen (no-adapter) CE target: zero the generated params so the codec update is a no-op.
            with torch.no_grad():
                current.clear()
                current.update({k: torch.zeros_like(v) for k, v in gen_m.items()})
                out_fz = interp_fwd(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                ce_frozen = masked_cross_entropy(out_fz.logits, batch.labels)
            current.clear()
            current.update(gen_m)
            out_m = interp_fwd(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            ce_m = masked_cross_entropy(out_m.logits, batch.labels)
            current.clear()
            current.update(gen_mm)
            out_mm = interp_fwd(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            ce_mm = masked_cross_entropy(out_mm.logits, batch.labels)
            neutral_pen = (ce_mm - ce_frozen).pow(2)  # ce_frozen is detached (no_grad); push ce_mm -> frozen
            loss = ce_m + args.neutral_junk_lambda * neutral_pen
            ce_m_t, ce_mm_t, ce_fz_t, aux_t = ce_m.detach(), ce_mm.detach(), ce_frozen.detach(), neutral_pen.detach()
            if args.l2_reg_generated_w:
                reg = torch.stack([v.float().pow(2).mean() for v in gen_m.values()]).mean()
                loss = loss + args.l2_reg_generated_w * reg
        elif args.contrastive_lambda > 0:
            # Mismatched-negative conditioning loss. ONE ddp_hyper forward on [matched ; mismatched]
            # condition embeddings (keeps DDP's single-forward-per-backward contract), split along the
            # batch dim, then TWO interpreter forwards (interpreter is not DDP-wrapped). The hinge
            # rewards the matched adapter for fitting the batch better than an adapter built from
            # another in-batch task's description.
            cond = batch.condition_embeddings
            b = cond.shape[0]
            cond_mm = torch.roll(cond, shifts=1, dims=0)
            gen_both = ddp_hyper(torch.cat([cond, cond_mm], dim=0))  # each: (num_layers, 2b, D)
            gen_m = {k: v[:, :b] for k, v in gen_both.items()}
            gen_mm = {k: v[:, b:] for k, v in gen_both.items()}
            current.clear()
            current.update(gen_m)
            out_m = interp_fwd(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            ce_m = masked_cross_entropy(out_m.logits, batch.labels)
            current.clear()
            current.update(gen_mm)
            out_mm = interp_fwd(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            ce_mm = masked_cross_entropy(out_mm.logits, batch.labels)
            contrast = torch.relu(args.contrastive_margin + ce_m - ce_mm)
            loss = ce_m + args.contrastive_lambda * contrast
            ce_m_t, ce_mm_t, aux_t = ce_m.detach(), ce_mm.detach(), contrast.detach()
            if args.l2_reg_generated_w:
                reg = torch.stack([v.float().pow(2).mean() for v in gen_m.values()]).mean()
                loss = loss + args.l2_reg_generated_w * reg
        else:
            generated = ddp_hyper(batch.input_ids.shape[0]) if args.static else ddp_hyper(batch.condition_embeddings)
            current.clear()
            current.update(generated)
            outputs = interp_fwd(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            loss = masked_cross_entropy(outputs.logits, batch.labels)
            if args.l2_reg_generated_w:
                reg = torch.stack([v.float().pow(2).mean() for v in generated.values()]).mean()
                loss = loss + args.l2_reg_generated_w * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp_hyper.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        done = step + 1
        if done % args.loss_log_every == 0 or done == args.steps:
            stats = torch.stack([loss.detach(), ce_m_t, ce_mm_t, ce_fz_t, aux_t]).float()
            dist.all_reduce(stats, op=dist.ReduceOp.AVG)  # cross-rank means
            if is_main():
                losses.append(float(stats[0]))
                if args.neutral_junk_lambda > 0:
                    extra = (f" ce_m={float(stats[1]):.4f} ce_mm={float(stats[2]):.4f} "
                             f"ce_froz={float(stats[3]):.4f} neutral_pen={float(stats[4]):.4f}")
                elif args.contrastive_lambda > 0:
                    extra = f" ce_m={float(stats[1]):.4f} ce_mm={float(stats[2]):.4f} contrast={float(stats[4]):.4f}"
                else:
                    extra = ""
                log(f"  step {done}/{args.steps} loss={float(stats[0]):.4f}{extra} lr={scheduler.get_last_lr()[0]:.2e}")
        if done % args.checkpoint_every == 0 or done == args.steps:
            dist.barrier()
            save(done)
            dist.barrier()
        if args.snapshot_every > 0 and (done % args.snapshot_every == 0 or done == args.steps):
            dist.barrier()
            save_snapshot(done)
            dist.barrier()

    if timer_start is not None:
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - timer_start
        n = args.steps - timer_from_step
        if is_main() and n > 0:
            per_step = elapsed / n
            log(f"[timing] {per_step*1000:.1f} ms/step over {n} steps (excl. warmup) -> "
                f"full {args.steps} steps ~= {per_step * args.steps / 3600:.2f} h")

    for h in hook_handles:
        h.remove()
    dist.barrier()

    if not is_main():
        dist.destroy_process_group()
        return

    if args.static:
        # No matched/junk to score: the static adapter is unconditioned. Its per-family accuracy is
        # computed post-hoc (t2p_eval_checkpoint.py --static-snapshot) as the `matched - static`
        # reference. Just persist metadata; the final snapshot already holds the trained weights.
        (output_dir / "train_meta.json").write_text(json.dumps({
            "static": True, "adapter": args.adapter, "interpreter": args.interpreter,
            "effective_batch": eff_batch, "steps": args.steps, "learning_rate": args.learning_rate,
            "per_task_limit": per_task_limit, "lora_scaling": args.lora_scaling,
            "ia3_scaling": args.ia3_scaling, "lokr_scaling": args.lokr_scaling, "loha_scaling": args.loha_scaling, "losses": losses,
        }, indent=2) + "\n")
        log(f"[6/6] static adapter trained -> {output_dir} "
            f"(eval post-hoc: t2p_eval_checkpoint.py --static-snapshot)")
        dist.destroy_process_group()
        return

    if args.skip_inline_eval:
        (output_dir / "train_meta.json").write_text(json.dumps({
            "effective_batch": eff_batch, "per_gpu_batch": args.per_gpu_batch, "world_size": world,
            "steps": args.steps, "learning_rate": args.learning_rate, "warmup_steps": warmup_steps,
            "example_visits": eff_batch * args.steps, "losses": losses,
            "per_task_limit": per_task_limit, "interpreter": args.interpreter,
            "contrastive_lambda": args.contrastive_lambda, "contrastive_margin": args.contrastive_margin,
            "neutral_junk_lambda": args.neutral_junk_lambda, "strip_task_def": args.strip_task_def,
            "adapter": args.adapter, "lora_scaling": args.lora_scaling, "ia3_scaling": args.ia3_scaling,
            "lokr_scaling": args.lokr_scaling, "loha_scaling": args.loha_scaling,
            "inline_eval_skipped": True,
        }, indent=2) + "\n")
        log(f"[5/6] legacy inline evaluation skipped; paired held-out evaluation is run by the caller -> {output_dir}")
        dist.destroy_process_group()
        return

    # ---- rank 0: held-out eval (same protocol/format as t2p-sft-pilot) ----
    log(f"[5/6] scoring frozen interpreter baseline + matched/adversarial ({eval_task_ids})...")
    eval_examples_by_family = build_all_task_examples(
        eval_task_ids, eval_descriptions, args.eval_limit, variant=args.eval_variant, use_icl=args.use_icl
    )
    all_eval_examples = [ex for group in eval_examples_by_family.values() for ex in group]
    recorder = ResultRecorder(output_dir, lambda r: f"  {r.adapter:20} {r.task_id:16} {r.metrics}")

    frozen_evaluator = HypernetworkDownstreamEvaluator(
        interpreter, layers, None, tokenizer,
        trial_id="t2p_train_ddp::frozen_interpreter", device=device, use_icl=args.use_icl,
    )
    for result in frozen_evaluator.iter_evaluate_frozen(all_eval_examples, split=args.eval_split):
        recorder.record(dataclasses.replace(result, metadata={**result.metadata, "seed": None}))

    hypernetwork.eval()
    evaluator = HypernetworkDownstreamEvaluator(
        interpreter, layers, hypernetwork, tokenizer,
        trial_id=f"t2p_train_ddp::{args.adapter}::seed{args.seed}::scaledefault",
        device=device, use_icl=args.use_icl,
    )
    for result in evaluator.iter_evaluate(
        eval_condition_embeddings, all_eval_examples, split=args.eval_split,
        mismatched_embeddings=mismatched_condition_embeddings,
    ):
        scale_metadata = {"seed": args.seed}
        scale_metadata[{"ia3": "ia3_scale", "lokr": "lokr_scale", "loha": "loha_scale"}.get(args.adapter, "lora_scale")] = (
            args.ia3_scaling if args.adapter == "ia3" else args.lokr_scaling if args.adapter == "lokr" else args.loha_scaling if args.adapter == "loha" else args.lora_scaling
        )
        recorder.record(dataclasses.replace(
            result, metadata={**result.metadata, **scale_metadata}))

    (output_dir / "train_meta.json").write_text(json.dumps({
        "effective_batch": eff_batch, "per_gpu_batch": args.per_gpu_batch, "world_size": world,
        "steps": args.steps, "learning_rate": args.learning_rate, "warmup_steps": warmup_steps,
        "example_visits": eff_batch * args.steps, "losses": losses,
        "per_task_limit": per_task_limit, "interpreter": args.interpreter,
        "contrastive_lambda": args.contrastive_lambda, "contrastive_margin": args.contrastive_margin,
        "neutral_junk_lambda": args.neutral_junk_lambda, "strip_task_def": args.strip_task_def,
        "adapter": args.adapter, "lora_scaling": args.lora_scaling, "ia3_scaling": args.ia3_scaling,
        "lokr_scaling": args.lokr_scaling, "loha_scaling": args.loha_scaling,
    }, indent=2) + "\n")
    log(f"[6/6] done -> {output_dir}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
