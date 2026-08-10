"""Score T2A checkpoints on the selection or report split under ROUGE-L, exact match and CE.

The single scoring path for the setting. Roles: a `hyper` checkpoint is scored as `matched` (averaged
over the 3 held-out description variants); a `static` checkpoint is scored as `static`
(description-independent). They are separate rows because the protocol selects the static on its own
score rather than at the hypernetwork's point.

**Splits.** `eval_ds_info` holds 21 `lol_` tasks but only 11 are absent from `train_ds_names`:

  * `--split selection` (default) -- the 10 in-distribution tasks. Every hyperparameter is chosen
    here. Requires `--example-offset 40`: the trainer consumes the leading 40 examples of the same
    `train[:10000]` split, so offset 0 replays training examples, which inflates matched, inflates
    the static more, and biases the scale argmax high.
  * `--split report` -- the 11 genuinely held-out tasks, spent exactly once by the confirmation seeds
    of an already-selected configuration. Guarded: it additionally requires
    `--confirm-spend-report-split`, a `--checkpoint-filter` restricting it to those confirmation
    checkpoints, and `--example-offset 0`. `--expect-checkpoints` fails before any GPU work if the
    filter selects an unintended number.

**Metrics.** `--metric generation` decodes greedily once and reads ROUGE-L and exact match off the
same pass, because generation is the expensive step. `--metric ce` computes teacher-forced
cross-entropy at the same fidelity, so the appendix CE column is comparable to the ROUGE-L column.

Each row is one JSON line carrying the full instrument, so the pass is auditable and resumable: a
checkpoint already scored under the *same* instrument is skipped, and a fidelity change re-scores
rather than silently mixing two instruments.

Usage (8-way data parallel):
    for i in $(seq 0 7); do
      HF_HUB_OFFLINE=1 .venv/bin/python scripts/t2a_score_checkpoints.py \
        --shard "$i" --num-shards 8 --device "cuda:$i" &
    done; wait
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from adapterbench.cli._shared import T2A_DECONTAM_CONFIG, load_frozen_interpreter  # noqa: E402
from adapterbench.t2a.condition_encoder import DEFAULT_CONDITION_ENCODER, embed_task_descriptions, load_condition_encoder  # noqa: E402
from t2a_eval_heldout_sni import build_task_examples, ce_for_task, make_batched_ce  # noqa: E402
from t2a_eval_heldout_sni_acc import (  # noqa: E402
    apply_codec_scaling,
    build_modules,
    build_task_prompts,
    load_snapshot,
    make_generate_batch,
    score_task,
)
from t2a_train_ddp import register_persistent_hooks  # noqa: E402

# The 10 in-distribution `lol_` eval tasks. AUTORESEARCH.md "T2A: metric, data split, and the
# independently selected control" fixes this list; it is hard-coded rather than derived so a
# config edit cannot silently pull a report-split task into a selection sweep.
SELECTION_SPLIT = ("lol_084", "lol_140", "lol_275", "lol_636", "lol_705",
                   "lol_717", "lol_742", "lol_1198", "lol_1448", "lol_1711")
# The 11 genuinely held-out tasks: absent from `train_ds_names`, so no example of them was ever
# trained on. They are spent EXACTLY ONCE, by the confirmation seeds of an already-selected
# configuration (AUTORESEARCH.md 4). Scoring them requires `--split report` AND the explicit
# `--confirm-spend-report-split` acknowledgement, so it can never happen as a default or a typo.
REPORT_SPLIT = ("lol_035", "lol_039", "lol_202", "lol_304", "lol_362", "lol_614",
                "lol_701", "lol_706", "lol_710", "lol_726", "lol_1557")


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="results/autoresearch/t2a/evaluation/checkpoint_index.jsonl")
    p.add_argument("--out-dir", default="results/autoresearch/t2a/evaluation/scores")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--interpreter", default="google/gemma-2-2b-it")
    p.add_argument("--condition-encoder", default=DEFAULT_CONDITION_ENCODER)
    p.add_argument("--decontam-config", default=str(T2A_DECONTAM_CONFIG))
    p.add_argument("--tasks-root", default="data/t2a/tasks")
    p.add_argument("--limit", type=int, default=32, help="examples per task (reduced fidelity)")
    p.add_argument("--example-offset", type=int, default=40,
                   help="skip the first N examples of each task. Defaults to 40, the training "
                        "--limit every recorded T2A run used: the 10 selection tasks are all in "
                        "`train_ds_names` and training consumes a LEADING range of the same "
                        "`train[:10000]` split, so offset 0 replays training examples verbatim. "
                        "Pass 0 only to reproduce the contaminated first pass.")
    p.add_argument("--n-desc", type=int, default=3, help="description variants averaged for matched")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--codecs", default="lora,ia3,lokr,fourierft")
    p.add_argument("--split", default="selection", choices=("selection", "report"),
                   help="`selection` = the 10 in-distribution lol_ tasks used for every "
                        "hyperparameter decision. `report` = the 11 genuinely held-out tasks, which "
                        "are spent once by the confirmation seeds of an already-selected "
                        "configuration and must never be used for a sweep.")
    p.add_argument("--confirm-spend-report-split", action="store_true",
                   help="required alongside `--split report`: an explicit acknowledgement that this "
                        "run consumes the one-shot held-out split.")
    p.add_argument("--checkpoint-filter", default="",
                   help="substring a checkpoint path must contain to be scored. REQUIRED with "
                        "`--split report`: the report split may only be measured on the confirmation "
                        "checkpoints of an already-selected configuration, and this filter is what "
                        "enforces that. Without it a report-split run would score the whole grid, "
                        "which is how a future re-selection ends up tuned against held-out data.")
    p.add_argument("--expect-checkpoints", type=int, default=-1,
                   help="fail unless exactly this many checkpoints pass the filter. Turns a silent "
                        "over-broad run into an error before any GPU work happens.")
    p.add_argument("--metric", default="generation", choices=("generation", "ce"),
                   help="generation = greedy decode -> ROUGE-L + EM (the selector). "
                        "ce = teacher-forced cross-entropy on the SAME split and fidelity, so the "
                        "appendix CE column is comparable to the ROUGE-L column instead of being "
                        "read off the old 21-task sweep at a different examples-per-task.")
    p.add_argument("--dry-run", action="store_true", help="print the shard's job list and exit")
    return p.parse_args()


def tasks_for(args) -> tuple[str, ...]:
    return REPORT_SPLIT if args.split == "report" else SELECTION_SPLIT


def instrument_of(args) -> dict:
    """Everything that could move a number. A resumed run only skips rows whose recorded
    instrument matches this exactly, so a fidelity change re-scores instead of silently mixing."""
    common = {
        "split": args.split, "tasks": list(tasks_for(args)),
        "limit": args.limit, "example_offset": args.example_offset,
        "n_desc": args.n_desc, "batch_size": args.batch_size,
        "max_len": args.max_len, "interpreter": args.interpreter,
        "condition_encoder": args.condition_encoder, "rank": args.rank,
        "metric_mode": args.metric,
    }
    if args.metric == "ce":
        return {**common, "metrics": ["ce"]}
    return {**common, "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
            "metrics": ["rouge_l", "rouge_l_untruncated", "em"]}


def load_jobs(args) -> list[dict]:
    codecs = {c for c in args.codecs.split(",") if c}
    rows = []
    for line in (REPO / args.index).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("in_scope") or r.get("codec") not in codecs:
            continue
        if args.checkpoint_filter and args.checkpoint_filter not in r["checkpoint"]:
            continue
        if r.get("scale") is None:
            raise SystemExit(f"refusing to score {r['checkpoint']}: unresolved codec scale")
        rows.append(r)
    rows.sort(key=lambda r: (r["codec"], r["role"], float(r["scale"]), float(r["lr"] or 0),
                             r["steps"], r["seed"], r["checkpoint"]))
    return [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]


def load_jobs_unsharded(args) -> list[dict]:
    """The full filtered set across all shards, for the --expect-checkpoints assertion."""
    import copy
    a = copy.copy(args)
    a.num_shards, a.shard = 1, 0
    return load_jobs(a)


def _normalized(instrument: dict) -> dict:
    """`metric_mode` was added when the CE mode landed, after the first generation pass had
    already been written. Defaulting it keeps those rows resumable instead of silently
    re-scoring 146 checkpoints because a key appeared."""
    return {**instrument, "metric_mode": instrument.get("metric_mode", "generation"),
            "example_offset": instrument.get("example_offset", 0)}


def already_done(out_dir: Path, instrument: dict) -> set[str]:
    """Union over every shard's file, so a re-shard still resumes correctly."""
    instrument = _normalized(instrument)
    done = set()
    for f in sorted(out_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a killed run; it will simply be re-scored
            if row.get("status") == "ok" and _normalized(row.get("instrument") or {}) == instrument:
                done.add(row["checkpoint"])
    return done


def main() -> None:
    args = build_args()
    instrument = instrument_of(args)
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap = set(SELECTION_SPLIT) & set(REPORT_SPLIT)
    if overlap:
        raise SystemExit(f"selection/report split overlap: {sorted(overlap)}")
    if args.split == "report":
        if not args.confirm_spend_report_split:
            raise SystemExit(
                "--split report consumes the one-shot held-out split (11 tasks reserved for the "
                "confirmation seeds of an already-selected configuration). Pass "
                "--confirm-spend-report-split to acknowledge, or use --split selection.")
        if args.example_offset != 0:
            raise SystemExit(
                f"--split report requires --example-offset 0 (got {args.example_offset}): none of "
                "the 11 report tasks is in train_ds_names, so no example of them was ever trained "
                "on and an offset would discard clean data for no reason.")
        if not args.checkpoint_filter:
            raise SystemExit(
                "--split report requires --checkpoint-filter (e.g. 'confirmation_v2'). The report "
                "split may only be measured on the confirmation checkpoints of an already-selected "
                "configuration; without a filter this would score the entire grid on held-out data "
                "and compromise any future re-selection.")
        print(f"[report split] scoring {len(REPORT_SPLIT)} genuinely held-out tasks, restricted to "
              f"checkpoints matching {args.checkpoint_filter!r}. This is the one-shot measurement.",
              flush=True)

    jobs = load_jobs(args)
    if args.expect_checkpoints >= 0:
        total = len(load_jobs_unsharded(args))
        if total != args.expect_checkpoints:
            raise SystemExit(f"--expect-checkpoints {args.expect_checkpoints} but the filter selects "
                             f"{total}; refusing to run rather than score an unintended set.")
    done = already_done(out_dir, instrument)
    pending = [j for j in jobs if j["checkpoint"] not in done]
    print(f"[shard {args.shard}/{args.num_shards}] {len(jobs)} jobs, {len(jobs) - len(pending)} already scored, "
          f"{len(pending)} pending", flush=True)
    if args.dry_run:
        for j in pending:
            print(f"  {j['codec']:10} {j['role']:7} scale={j['scale']:<12} lr={j['lr']:<8} "
                  f"steps={j['steps']:<6} s{j['seed']}  {j['checkpoint']}")
        return
    if not pending:
        return

    decontam = yaml.safe_load((REPO / args.decontam_config).read_text())
    eval_ds_info = decontam["eval_ds_info"]

    print(f"loading interpreter {args.interpreter} on {args.device}...", flush=True)
    tokenizer, interpreter, layers = load_frozen_interpreter(args.interpreter, args.device)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder_model, encoder_tokenizer = load_condition_encoder(args.condition_encoder, args.device)
    with torch.no_grad():
        condition_dim = embed_task_descriptions(["probe"], encoder_model, encoder_tokenizer).shape[-1]

    # Prompts, golds and description embeddings are instrument-level constants: build once and
    # reuse for every checkpoint, so each rung is scored on byte-identical inputs.
    split_tasks = tasks_for(args)
    print(f"caching {len(split_tasks)} {args.split}-split tasks at limit={args.limit} "
          f"offset={args.example_offset} ({args.metric} mode)...", flush=True)
    tasks: list[dict] = []
    for task_id in split_tasks:
        if args.metric == "ce":
            payload = {"examples": build_task_examples(tokenizer, REPO / args.tasks_root, task_id,
                                                       args.limit, args.max_len,
                                                       offset=args.example_offset)}
            n = len(payload["examples"])
        else:
            prompts, golds = build_task_prompts(tokenizer, REPO / args.tasks_root, task_id, args.limit,
                                                offset=args.example_offset)
            payload = {"prompts": prompts, "golds": golds}
            n = len(prompts)
        if not n:
            print(f"  [WARN] {task_id}: no examples or no vendored metadata; excluded", flush=True)
            continue
        descs = eval_ds_info[task_id]["descriptions"][: args.n_desc]
        with torch.no_grad():
            embs = [embed_task_descriptions([d], encoder_model, encoder_tokenizer)[0].to(args.device) for d in descs]
        if n < args.limit:
            print(f"  [WARN] {task_id}: only {n}/{args.limit} examples available at offset "
                  f"{args.example_offset}; this task is scored on fewer examples than the rest",
                  flush=True)
        tasks.append({"task_id": task_id, **payload, "desc_embs": embs, "n_desc": len(embs), "n": n})
        print(f"  {task_id:10} n={n} n_desc={len(embs)} offset={args.example_offset}", flush=True)
    if len(tasks) != len(split_tasks):
        print(f"[WARN] scoring {len(tasks)}/{len(split_tasks)} {args.split} tasks", flush=True)

    out_path = out_dir / f"shard{args.shard:02d}.jsonl"
    current: dict = {}
    built: dict | None = None          # {codec, hypernetwork, static, handles, generate_batch, frozen}

    def ensure_codec(codec: str):
        """Build the modules and hooks for `codec`, reusing them across its rungs.

        Construction cost is codec-level (FourierFT's frozen DCT bases dominate) and the swept
        scale is just a scalar on each codec object the hooks already reference, so only a codec
        change requires a rebuild -- and the old hooks must come off first or they stack.
        """
        nonlocal built
        if built is not None and built["codec"] == codec:
            return built
        if built is not None:
            for h in built["handles"]:
                h.remove()
            del built["hypernetwork"], built["static"]
            built = None
            torch.cuda.empty_cache()
        t0 = time.time()
        hypernetwork, static = build_modules(
            interpreter=interpreter, layers=layers, device=args.device, condition_dim=condition_dim,
            adapter=codec, rank=args.rank,
        )
        handles = register_persistent_hooks(hypernetwork.codecs, layers, current)
        if args.metric == "ce":
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            scorer = make_batched_ce(interpreter, current, pad_id=pad_id,
                                     batch_size=args.batch_size, device=args.device)
        else:
            scorer = make_generate_batch(tokenizer, interpreter, current, batch_size=args.batch_size,
                                         max_new_tokens=args.max_new_tokens, max_len=args.max_len,
                                         device=args.device)
        built = {"codec": codec, "hypernetwork": hypernetwork, "static": static,
                 "handles": handles, "scorer": scorer, "frozen": None}
        print(f"  built {codec} modules in {time.time() - t0:.1f}s", flush=True)
        return built

    def score_one_task(ctx, task: dict, role: str) -> dict:
        """One task, one role, in whichever metric mode is active. `ce` mode returns
        `{"ce": float}` so both modes produce a uniform per-task dict of scalars."""
        if args.metric == "ce":
            ce = ce_for_task(ctx["scorer"], task["examples"], hypernetwork=ctx["hypernetwork"],
                             static=ctx["static"], desc_embs=task["desc_embs"], roles=(role,))
            return {"ce": ce[role], "n": task["n"]}
        scored = score_task(ctx["scorer"], task["prompts"], task["golds"],
                            hypernetwork=ctx["hypernetwork"], static=ctx["static"],
                            desc_embs=task["desc_embs"], roles=(role,))
        return scored[role]

    def frozen_scores(ctx) -> dict:
        """The zeroed-adapter baseline. Every codec preserves identity at the all-zero generated
        initialization, so this should agree across codecs; it is computed (and recorded) per
        codec anyway so the analysis can verify that rather than assume it."""
        if ctx["frozen"] is None:
            ctx["frozen"] = {t["task_id"]: score_one_task(ctx, {**t, "desc_embs": []}, "frozen")
                             for t in tasks}
        return ctx["frozen"]

    for n, job in enumerate(pending, start=1):
        t0 = time.time()
        label = (f"{job['codec']}/{job['role']} scale={job['scale']} lr={job['lr']} "
                 f"steps={job['steps']} s{job['seed']}")
        print(f"[{n}/{len(pending)}] {label}", flush=True)
        row = {**{k: job[k] for k in ("checkpoint", "codec", "role", "seed", "steps", "scale", "lr",
                                      "warmup_frac", "phase", "rung_dir", "role_dir", "ledger",
                                      "ledger_status", "converged", "ce_selection_metric",
                                      "ce_control_metric", "ce_matched", "ce_helpfulness_metric",
                                      "has_ledger_entry") if k in job},
               "instrument": instrument, "device": args.device, "shard": args.shard}
        try:
            ctx = ensure_codec(job["codec"])
            apply_codec_scaling(ctx["hypernetwork"], float(job["scale"]))
            role = "matched" if job["role"] == "hyper" else "static"
            module = ctx["hypernetwork"] if role == "matched" else ctx["static"]
            load_snapshot(module, str(REPO / job["checkpoint"]), args.device)

            frozen = frozen_scores(ctx)
            per_task = {t["task_id"]: score_one_task(ctx, t, role) for t in tasks}
            keys = ("ce",) if args.metric == "ce" else ("rouge_l", "rouge_l_untruncated", "em")
            row.update({
                "status": "ok", "scored_role": role, "n_tasks": len(per_task),
                "per_task": per_task,
                "aggregate": {k: sum(v[k] for v in per_task.values()) / len(per_task) for k in keys},
                "frozen_per_task": frozen,
                "frozen_aggregate": {k: sum(v[k] for v in frozen.values()) / len(frozen) for k in keys},
                "elapsed_s": round(time.time() - t0, 1),
            })
            agg, fro = row["aggregate"], row["frozen_aggregate"]
            if args.metric == "ce":
                print(f"      CE={agg['ce']:.4f}  frozen CE={fro['ce']:.4f}  [{row['elapsed_s']}s]", flush=True)
            else:
                print(f"      rougeL={agg['rouge_l']:.4f} (untrunc {agg['rouge_l_untruncated']:.4f}) "
                      f"EM={agg['em']:.4f}  frozen rougeL={fro['rouge_l']:.4f}  "
                      f"[{row['elapsed_s']}s]", flush=True)
        except Exception as exc:  # a bad snapshot must not take the whole shard down
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=6),
                        "elapsed_s": round(time.time() - t0, 1)})
            print(f"      ERROR {row['error']}", flush=True)
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())  # so a killed shard resumes from a complete file

    print(f"[shard {args.shard}] done; wrote to {out_path}", flush=True)


if __name__ == "__main__":
    main()
