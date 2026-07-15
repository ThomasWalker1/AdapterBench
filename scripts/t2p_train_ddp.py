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
    p.add_argument("--per-gpu-batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=62500, help="optimizer steps (each = world_size*per_gpu_batch examples)")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--l2-reg-generated-w", type=float, default=0.0)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--adapter", default="lora")
    p.add_argument("--no-compile", action="store_true", help="disable torch.compile (debugging)")
    p.add_argument(
        "--fixed-seq-len", type=int, default=0,
        help="pad every batch to this fixed length instead of the per-batch max. Numerically "
        "identical (pad tokens are masked in attention and loss) but gives static shapes: "
        "torch.compile traces ONE graph (no dynamic-shape thrash) and all DDP ranks do equal "
        "work each step (no variable-length straggler). Recommended = the tokenizer truncation "
        "cap (512). 0 = dynamic per-batch padding.",
    )
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--loss-log-every", type=int, default=500)
    # eval (rank 0 only, after training) - mirrors t2p-sft-pilot
    p.add_argument("--eval-tasks", default="arc_easy,arc_challenge,hellaswag,boolq")
    p.add_argument("--eval-descriptions", default=str(T2L_EVAL_DESCRIPTIONS))
    p.add_argument("--eval-limit", type=int, default=80)
    p.add_argument("--eval-variant", type=int, default=0)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--use-icl", action="store_true")
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

    log(f"[3/6] loading + tokenizing {len(task_ids)} training task(s)...")
    datasets = [
        LolSFTDataset(tokenizer, metadata_by_task[task_id], train_embeddings_by_task[task_id], limit=args.limit)
        for task_id in task_ids
    ]
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
        seed=args.seed,
    ).to(device)
    ddp_hyper = DDP(hypernetwork, device_ids=[local_rank], find_unused_parameters=False)

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
        hypernetwork.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, losses = state["step"], list(state["losses"])
        log(f"[resume] from step {start}/{args.steps}")
    # keep all ranks' weights identical after a resume
    for p in hypernetwork.parameters():
        dist.broadcast(p.data, src=0)

    def save(done: int) -> None:
        if not is_main():
            return
        tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
        torch.save(
            {"model": hypernetwork.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "step": done, "losses": losses},
            tmp,
        )
        tmp.replace(ckpt)

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
        generated = ddp_hyper(batch.condition_embeddings)
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
            reduced = loss.detach().clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
            if is_main():
                losses.append(float(reduced))
                log(f"  step {done}/{args.steps} loss={float(reduced):.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        if done % args.checkpoint_every == 0 or done == args.steps:
            dist.barrier()
            save(done)
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
        recorder.record(dataclasses.replace(
            result, metadata={**result.metadata, "seed": args.seed, "lora_scale": "default"}))

    (output_dir / "train_meta.json").write_text(json.dumps({
        "effective_batch": eff_batch, "per_gpu_batch": args.per_gpu_batch, "world_size": world,
        "steps": args.steps, "learning_rate": args.learning_rate, "warmup_steps": warmup_steps,
        "example_visits": eff_batch * args.steps, "losses": losses,
    }, indent=2) + "\n")
    log(f"[6/6] done -> {output_dir}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
