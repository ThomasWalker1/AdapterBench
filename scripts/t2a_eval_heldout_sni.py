"""Teacher-forced-CE eval on T2A's held-out SNI validation tasks under strip-task-def.

This is the decisive T2A conditioning test. On tasks the hypernetwork NEVER trained on, whose
input has the task definition STRIPPED (``strip_task_def``), the only signal for "what task is
this" is the description -> gte -> hypernetwork -> adapter. We measure response cross-entropy under:

  * matched  - adapter generated from the task's own held-out description (averaged over the
               3 held-out description variants T2A publishes for each task);
  * static   - a task-agnostic multi-task LoRA of the SAME shape (the ``--static`` reference run);
  * frozen   - the base model with a zeroed adapter (no-op), i.e. no adaptation at all.

Lower CE = better. Genuine conditioning is a matched < static ordering: the description-conditioned
adapter fits held-out tasks better than the best task-agnostic adapter of equal capacity. Because
the input carries no task statement, that gap cannot come from input redundancy. ``matched - static``
is the headline number (negative = conditioning helps); ``matched - frozen`` is the helpfulness floor.

Teacher-forced CE is a single forward (no autoregression), so this runs in minutes and is the EXACT
quantity the trainer optimizes -- train and eval are directly comparable. Single-GPU, no DDP.

Usage:
    .venv/bin/python scripts/t2a_eval_heldout_sni.py \
        --interpreter google/gemma-2-2b-it \
        --snapshot        results/.../gemma2b_stripdef_hyper/s777/snapshots/step20000.pt \
        --static-snapshot results/.../gemma2b_stripdef_static/s777/snapshots/step20000.pt \
        --limit 64 --device cuda:0 \
        --out results/.../gemma2b_stripdef_hyper/s777/heldout_sni_ce.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for register_persistent_hooks

from adapterbench.cli._shared import (  # noqa: E402
    PILOT_DEFAULT_TARGET_MODULES,
    T2A_DECONTAM_CONFIG,
    load_frozen_interpreter,
)
from adapterbench.t2a.condition_encoder import (  # noqa: E402
    DEFAULT_CONDITION_ENCODER,
    embed_task_descriptions,
    load_condition_encoder,
)
from adapterbench.t2a.hypernetwork import (  # noqa: E402
    StaticAdapter,
    TextToPeftHypernetwork,
    infer_module_shapes,
)
from adapterbench.t2a.lol_data import LolSFTDataset, load_task_metadata, lol_collate_fn  # noqa: E402
from adapterbench.t2a.sft_trainer import masked_cross_entropy  # noqa: E402
from t2a_train_ddp import register_persistent_hooks  # noqa: E402


def build_task_examples(tokenizer, tasks_root, task_id: str, limit: int, max_len: int,
                        offset: int = 0) -> list[dict]:
    """Definition-stripped, response-only-supervised examples for one task.

    `strip_task_def=True` => the frozen model sees ONLY the problem, so the task is specifiable
    only via the description-conditioned adapter. `training=False` => deterministic (the attached
    condition embedding is ignored). Returns `[]` when the task has no vendored metadata.

    `offset` skips the first N examples so an evaluation can be example-disjoint from training on
    a task that is in `train_ds_names` (training selects a LEADING range of the same split). It is
    applied by over-fetching and slicing rather than by teaching `LolSFTDataset` about offsets,
    because that class defines the training examples and must keep its current semantics exactly.
    """
    if not (Path(tasks_root) / task_id / "metadata.yaml").exists():
        return []
    metadata = load_task_metadata(Path(tasks_root), task_id)
    dataset = LolSFTDataset(
        tokenizer, metadata, torch.zeros(1, 1), max_len=max_len,
        limit=offset + limit, training=False, strip_task_def=True,
    )
    return [dataset[i] for i in range(min(offset, len(dataset)), len(dataset))]


def make_batched_ce(interpreter, current: dict, *, pad_id: int, batch_size: int, device: str):
    """Example-weighted mean of `masked_cross_entropy`, with the codec hooks reading generated
    params from `current` exactly as in training -- so this is the quantity the trainer optimized."""

    def batched_ce(examples: list[dict], gen_fn) -> float:
        total_ce, total_n = 0.0, 0
        for i in range(0, len(examples), batch_size):
            batch = lol_collate_fn(examples[i : i + batch_size], pad_id).to(device)
            b = batch.input_ids.shape[0]
            current.clear()
            current.update(gen_fn(b))
            with torch.no_grad():
                out = interpreter(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                ce = masked_cross_entropy(out.logits, batch.labels)
            total_ce += float(ce) * b
            total_n += b
        return total_ce / max(total_n, 1)

    return batched_ce


def ce_for_task(batched_ce, examples: list[dict], *, hypernetwork, static,
                desc_embs: list[torch.Tensor], roles=("matched", "static", "frozen")) -> dict[str, float]:
    """CE per role for one task. `matched` averages over the description variants."""
    out: dict[str, float] = {}
    if "matched" in roles:
        out["matched"] = sum(
            batched_ce(examples, lambda b, e=emb: hypernetwork(e.unsqueeze(0).expand(b, -1)))
            for emb in desc_embs
        ) / len(desc_embs)
    if "static" in roles:
        out["static"] = batched_ce(examples, lambda b: static(b))
    if "frozen" in roles:
        out["frozen"] = batched_ce(examples, lambda b: {k: torch.zeros_like(v) for k, v in static(b).items()})
    return out


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interpreter", required=True)
    p.add_argument("--snapshot", required=True, help="hypernetwork snapshot (from a --strip-task-def run)")
    p.add_argument("--static-snapshot", required=True, help="StaticAdapter snapshot (from a --static --strip-task-def run)")
    p.add_argument("--step", type=int, default=-1, help="override the step (else parsed from --snapshot filename)")
    p.add_argument("--condition-encoder", default=DEFAULT_CONDITION_ENCODER)
    p.add_argument("--decontam-config", default=str(T2A_DECONTAM_CONFIG))
    p.add_argument("--tasks-root", default="data/t2a/tasks")
    p.add_argument("--tasks", default="", help="comma-separated held-out task ids; default = all lol_* in eval_ds_info")
    p.add_argument("--n-desc", type=int, default=3, help="held-out description variants to average matched CE over")
    p.add_argument("--limit", type=int, default=64, help="examples per held-out task")
    p.add_argument("--example-offset", type=int, default=0,
                   help="skip the first N examples of each task. Training consumes a LEADING range "
                        "of the same split, so set this to the training --limit (40 for every "
                        "recorded T2A run) to score a trained-on task on examples it never saw.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--adapter", default="lora")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora-scaling", type=float, default=-1.0)
    p.add_argument("--ia3-scaling", type=float, default=1.0)
    p.add_argument("--lokr-scaling", type=float, default=1.0)
    p.add_argument("--fourierft-scaling", type=float, default=1.0)
    p.add_argument("--codec-scaling", type=float, default=None,
                   help="generic output-scale override for the selected --adapter's codec; "
                        "must match the value the snapshot was trained with")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="", help="JSONL to append one row per task + one aggregate row")
    return p.parse_args()


def main() -> None:
    args = build_args()
    device = args.device
    step = args.step if args.step >= 0 else int(re.search(r"step(\d+)", Path(args.snapshot).name).group(1))

    decontam = yaml.safe_load(Path(args.decontam_config).read_text())
    eval_ds_info = decontam["eval_ds_info"]
    if args.tasks:
        task_ids = [t for t in args.tasks.split(",") if t]
    else:
        task_ids = [k for k in eval_ds_info if str(k).startswith("lol_")]

    print(f"[1/4] loading interpreter {args.interpreter} on {device}...", flush=True)
    tokenizer, interpreter, layers = load_frozen_interpreter(args.interpreter, device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    print("[2/4] rebuilding hypernetwork + static, loading snapshots...", flush=True)
    torch.manual_seed(0)
    target_modules = PILOT_DEFAULT_TARGET_MODULES[args.adapter]
    module_shapes = infer_module_shapes(layers, target_modules, hidden_size=interpreter.config.hidden_size)
    encoder_model, encoder_tokenizer = load_condition_encoder(args.condition_encoder, device)
    with torch.no_grad():
        condition_dim = embed_task_descriptions(["probe"], encoder_model, encoder_tokenizer).shape[-1]
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=condition_dim, module_shapes=module_shapes, num_layers=len(layers),
        adapter=args.adapter, rank=args.rank, ia3_scaling=args.ia3_scaling, lokr_scaling=args.lokr_scaling,
        fourierft_scaling=args.fourierft_scaling, seed=0,
    ).to(device)
    if args.lora_scaling > 0:
        from adapterbench.t2a.codecs import LoRACodec
        for codec in hypernetwork.codecs.values():
            if isinstance(codec, LoRACodec):
                codec.scaling = args.lora_scaling
    if args.codec_scaling is not None:
        from adapterbench.t2a.codecs import set_codec_scaling
        set_codec_scaling(hypernetwork.codecs, args.codec_scaling)
    hypernetwork.load_state_dict(torch.load(args.snapshot, map_location=device, weights_only=False)["model"])
    hypernetwork.eval()

    static = StaticAdapter(hypernetwork.codecs, len(layers)).to(device)
    static.load_state_dict(torch.load(args.static_snapshot, map_location=device, weights_only=False)["model"])
    static.eval()

    # Persistent codec hooks read the generated params from `current` on every interpreter forward,
    # exactly as in training -- so the CE we compute here is the same quantity the trainer optimized.
    current: dict = {}
    register_persistent_hooks(hypernetwork.codecs, layers, current)
    batched_ce = make_batched_ce(interpreter, current, pad_id=pad_id, batch_size=args.batch_size, device=device)

    print(f"[3/4] embedding held-out descriptions for {len(task_ids)} task(s)...", flush=True)
    rows = []
    for ti, task_id in enumerate(task_ids):
        examples = build_task_examples(tokenizer, args.tasks_root, task_id, args.limit, args.max_len,
                                       offset=args.example_offset)
        if not examples:
            print(f"  [skip] {task_id}: no vendored metadata.yaml or no examples", flush=True)
            continue
        descs = eval_ds_info[task_id]["descriptions"][: args.n_desc]
        with torch.no_grad():
            desc_embs = [embed_task_descriptions([d], encoder_model, encoder_tokenizer)[0].to(device) for d in descs]

        ce = ce_for_task(batched_ce, examples, hypernetwork=hypernetwork, static=static, desc_embs=desc_embs)
        ce_matched, ce_static, ce_frozen = ce["matched"], ce["static"], ce["frozen"]

        row = {
            "step": step, "task_id": task_id, "n_examples": len(examples), "n_desc": len(desc_embs),
            "example_offset": args.example_offset,
            "ce_matched": ce_matched, "ce_static": ce_static, "ce_frozen": ce_frozen,
            "matched_minus_static": ce_matched - ce_static,
            "matched_minus_frozen": ce_matched - ce_frozen,
        }
        rows.append(row)
        print(f"  [{ti + 1}/{len(task_ids)}] {task_id:10} n={len(examples):3}  "
              f"matched={ce_matched:.3f} static={ce_static:.3f} frozen={ce_frozen:.3f}  "
              f"m-static={row['matched_minus_static']:+.3f}  m-frozen={row['matched_minus_frozen']:+.3f}", flush=True)

    print("[4/4] aggregate over held-out SNI tasks:", flush=True)
    if rows:
        n = len(rows)
        agg = {
            "step": step, "task_id": "__aggregate__", "n_tasks": n,
            "ce_matched": sum(r["ce_matched"] for r in rows) / n,
            "ce_static": sum(r["ce_static"] for r in rows) / n,
            "ce_frozen": sum(r["ce_frozen"] for r in rows) / n,
            "matched_minus_static": sum(r["matched_minus_static"] for r in rows) / n,
            "matched_minus_frozen": sum(r["matched_minus_frozen"] for r in rows) / n,
            # fraction of tasks where conditioning strictly helps vs the static reference
            "frac_tasks_matched_lt_static": sum(r["matched_minus_static"] < 0 for r in rows) / n,
        }
        print(f"  MEAN over {n} tasks: matched={agg['ce_matched']:.3f} static={agg['ce_static']:.3f} "
              f"frozen={agg['ce_frozen']:.3f}", flush=True)
        print(f"  matched - static = {agg['matched_minus_static']:+.4f}   (negative = conditioning helps)", flush=True)
        print(f"  matched - frozen = {agg['matched_minus_frozen']:+.4f}   (helpfulness floor)", flush=True)
        print(f"  tasks where matched < static: {agg['frac_tasks_matched_lt_static']*100:.0f}%", flush=True)
        rows.append(agg)
    else:
        print("  no tasks evaluated", flush=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"appended {len(rows)} row(s) to {out}", flush=True)


if __name__ == "__main__":
    main()
