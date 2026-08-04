"""Post-hoc eval of a T2L hypernetwork snapshot (from scripts/t2p_train_ddp.py --snapshot-every).

Loads one model-only snapshot, rebuilds the hypernetwork against the given interpreter, and scores
the held-out families with the SAME protocol as the training script's final eval: frozen baseline,
matched description, and the junk/adversarial mismatched control. Emits one compact curve row per
family to --curve-out (JSONL, appended) plus a readable summary. Single-GPU, no DDP.

Because eval is decoupled from training here, the SAME snapshot can be re-scored with a different
--eval-tasks set later (e.g. the paper's full 10-task suite) without retraining.

Usage:
    .venv/bin/python scripts/t2p_eval_checkpoint.py \
        --interpreter mistralai/Mistral-7B-Instruct-v0.2 \
        --snapshot results/.../snapshots/step10000.pt \
        --eval-tasks arc_easy,arc_challenge,hellaswag,boolq --eval-limit 80 \
        --adversarial-control --device cuda:0 \
        --curve-out results/.../curve.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapterbench.cli._shared import (  # noqa: E402
    PILOT_DEFAULT_TARGET_MODULES,
    T2L_DECONTAM_CONFIG,
    T2L_EVAL_DESCRIPTIONS,
    load_frozen_interpreter,
)
from adapterbench.task_examples import build_all_task_examples, load_task_descriptions  # noqa: E402
from adapterbench.t2p.condition_encoder import (  # noqa: E402
    DEFAULT_CONDITION_ENCODER,
    embed_task_descriptions,
    load_condition_encoder,
)
from adapterbench.t2p.hypernetwork import TextToPeftHypernetwork, infer_module_shapes  # noqa: E402
from adapterbench.t2p.live_evaluator import HypernetworkDownstreamEvaluator  # noqa: E402


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interpreter", required=True)
    p.add_argument("--snapshot", required=True, help="path to snapshots/step{n}.pt (model-only)")
    p.add_argument("--step", type=int, default=-1, help="override the step (else parsed from filename)")
    p.add_argument("--condition-encoder", default=DEFAULT_CONDITION_ENCODER)
    p.add_argument("--eval-tasks", default="arc_easy,arc_challenge,hellaswag,boolq")
    p.add_argument("--eval-descriptions", default=str(T2L_EVAL_DESCRIPTIONS))
    p.add_argument("--eval-limit", type=int, default=80)
    p.add_argument("--eval-variant", type=int, default=0)
    p.add_argument("--avg-descriptions", type=int, default=1,
                   help="average MATCHED accuracy over the first N description variants per family "
                        "(paper uses 3); junk/frozen are computed once")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--adapter", default="lora")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--adversarial-control", action="store_true")
    p.add_argument("--adversarial-descs", default="")
    p.add_argument("--decontam-config", default=str(T2L_DECONTAM_CONFIG))
    p.add_argument("--lora-scaling", type=float, default=-1.0)
    p.add_argument("--codec-scaling", type=float, default=None,
                   help="generic output-scale override for the selected --adapter's codec; "
                        "must match the value the snapshot was trained with")
    p.add_argument("--static-snapshot", default="", help="path to a --static run's snapshot; if given, "
                   "also score the static (multi-task) adapter per family and report matched - static")
    p.add_argument("--skip-frozen", action="store_true", help="skip the frozen baseline (constant across steps)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--curve-out", default="", help="JSONL to append one row per family (step,family,frozen,matched,junk)")
    return p.parse_args()


def main() -> None:
    args = build_args()
    step = args.step if args.step >= 0 else int(re.search(r"step(\d+)", Path(args.snapshot).name).group(1))
    device = args.device
    eval_task_ids = args.eval_tasks.split(",")

    print(f"[1/4] loading interpreter {args.interpreter} on {device}...", flush=True)
    tokenizer, interpreter, layers = load_frozen_interpreter(args.interpreter, device)

    print("[2/4] rebuilding hypernetwork + loading snapshot...", flush=True)
    torch.manual_seed(0)
    target_modules = PILOT_DEFAULT_TARGET_MODULES[args.adapter]
    module_shapes = infer_module_shapes(layers, target_modules, hidden_size=interpreter.config.hidden_size)
    encoder_model, encoder_tokenizer = load_condition_encoder(args.condition_encoder, device)
    with torch.no_grad():
        condition_dim = embed_task_descriptions(["probe"], encoder_model, encoder_tokenizer).shape[-1]
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=condition_dim, module_shapes=module_shapes, num_layers=len(layers),
        adapter=args.adapter, rank=args.rank, seed=0,
    ).to(device)
    if args.lora_scaling > 0:
        from adapterbench.t2p.codecs import LoRACodec
        for codec in hypernetwork.codecs.values():
            if isinstance(codec, LoRACodec):
                codec.scaling = args.lora_scaling
    if args.codec_scaling is not None:
        from adapterbench.t2p.codecs import set_codec_scaling
        set_codec_scaling(hypernetwork.codecs, args.codec_scaling)
    state = torch.load(args.snapshot, map_location=device, weights_only=False)
    hypernetwork.load_state_dict(state["model"])
    hypernetwork.eval()

    print("[3/4] embedding eval descriptions + adversarial control...", flush=True)
    eval_descriptions = load_task_descriptions(args.eval_descriptions)
    families = list(eval_task_ids)
    # matched embeddings for each of the first N description variants (clamped to what's available)
    n_avg = max(1, args.avg_descriptions)
    variant_embeddings = []
    for v in range(n_avg):
        emb = {}
        for fam in families:
            descs = eval_descriptions[fam]
            emb[fam] = embed_task_descriptions([descs[min(v, len(descs) - 1)]], encoder_model, encoder_tokenizer)[0].to(device)
        variant_embeddings.append(emb)
    eval_condition_embeddings = variant_embeddings[0]  # variant used for the junk-control pairing
    if args.adversarial_control:
        import yaml
        if args.adversarial_descs:
            adv_descs = [d for d in args.adversarial_descs.split("||") if d]
        else:
            adv_descs = yaml.safe_load(Path(args.decontam_config).read_text()).get("additional_eval_descs", [])
        adv_emb = [embed_task_descriptions([d], encoder_model, encoder_tokenizer)[0].to(device) for d in adv_descs]
        mismatched = {fam: adv_emb[i % len(adv_emb)] for i, fam in enumerate(families)}
    else:
        mismatched = {fam: eval_condition_embeddings[other] for fam, other in zip(families, families[1:] + families[:1])}
    del encoder_model

    examples_by_family = build_all_task_examples(
        eval_task_ids, eval_descriptions, args.eval_limit, variant=args.eval_variant
    )
    all_examples = [ex for group in examples_by_family.values() for ex in group]

    print(f"[4/4] scoring step {step}...", flush=True)
    frozen_acc: dict[str, float] = {}
    if not args.skip_frozen:
        fev = HypernetworkDownstreamEvaluator(
            interpreter, layers, None, tokenizer, trial_id="eval_ckpt::frozen", device=device
        )
        for r in fev.iter_evaluate_frozen(all_examples, split=args.eval_split):
            frozen_acc[r.task_id] = r.metrics.get("accuracy", r.metrics.get("exact_match"))

    # static (multi-task) adapter reference, if provided: same shape, no conditioning.
    static_acc: dict[str, float] = {}
    if args.static_snapshot:
        from adapterbench.t2p.hypernetwork import StaticAdapter
        static = StaticAdapter(hypernetwork.codecs, len(layers)).to(device)
        static.load_state_dict(torch.load(args.static_snapshot, map_location=device, weights_only=False)["model"])
        static.eval()
        sev = HypernetworkDownstreamEvaluator(
            interpreter, layers, static, tokenizer, trial_id="eval_ckpt::static", device=device
        )
        for r in sev.iter_evaluate(eval_condition_embeddings, all_examples, split=args.eval_split):
            static_acc[r.task_id] = r.metrics.get("accuracy", r.metrics.get("exact_match"))

    ev = HypernetworkDownstreamEvaluator(
        interpreter, layers, hypernetwork, tokenizer, trial_id=f"eval_ckpt::step{step}", device=device
    )
    # matched accuracy averaged over n_avg description variants; junk computed once (variant 0's pairing)
    matched_sum: dict[str, float] = {fam: 0.0 for fam in families}
    junk_acc: dict[str, float] = {}
    for v, emb in enumerate(variant_embeddings):
        mm = mismatched if v == 0 else None
        for r in ev.iter_evaluate(emb, all_examples, split=args.eval_split, mismatched_embeddings=mm):
            m = r.metrics
            matched_sum[r.task_id] += m.get("accuracy", m.get("exact_match"))
            if v == 0:
                junk_acc[r.task_id] = m.get("accuracy_mismatched", m.get("exact_match_mismatched"))

    rows = []
    for fam in families:
        matched = matched_sum[fam] / len(variant_embeddings)
        junk = junk_acc.get(fam)
        static = static_acc.get(fam)
        row = {"step": step, "family": fam, "frozen": frozen_acc.get(fam),
               "matched": matched, "junk": junk, "static": static, "n_desc_avg": len(variant_embeddings),
               "matched_minus_junk": (matched - junk) if junk is not None else None,
               "matched_minus_frozen": (matched - frozen_acc[fam]) if fam in frozen_acc else None,
               "matched_minus_static": (matched - static) if static is not None else None}
        rows.append(row)
        mj = f"{row['matched_minus_junk']:+.3f}" if row['matched_minus_junk'] is not None else "n/a"
        ms = f" m-static={row['matched_minus_static']:+.3f}" if static is not None else ""
        print(f"  step {step:>6}  {fam:16} frozen={row['frozen']}  matched={matched:.3f}  "
              f"junk={junk:.3f}  m-j={mj}{ms}  (avg {len(variant_embeddings)} desc)", flush=True)

    if args.curve_out:
        out = Path(args.curve_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"appended {len(rows)} row(s) to {out}", flush=True)


if __name__ == "__main__":
    main()
