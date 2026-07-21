"""Generation accuracy on the held-out SNI tasks under strip-task-def (companion to the CE eval).

The CE eval (`t2p_eval_heldout_sni.py`) measures the probability the adapter assigns to the gold
answer. This measures the interpretable outcome: does the conditioned adapter make the model
*generate* the correct answer more often than the static reference or the frozen model? Same
definition-stripped prompts, greedy decoding, normalized exact-match. The scorer is identical across
matched / static / frozen, so the accuracy *delta* is a clean read of conditioning's effect even
though absolute exact-match is a strict proxy for the official SNI Rouge-L.

Exact-match is meaningful for the classification / extraction tasks (the majority: MNLI, disease NER,
MMMLU, word-overlap, ...) and near-meaningless for the open-ended ones (poem generation, grammar
correction); per-task accuracy is reported so those can be read separately.

Usage:
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/t2p_eval_heldout_sni_acc.py \
        --interpreter google/gemma-2-2b-it \
        --snapshot        .../gemma2b_stripdef_hyper/s777/snapshots/step20000.pt \
        --static-snapshot .../gemma2b_stripdef_static/s777/snapshots/step20000.pt \
        --limit 48 --batch-size 16 --device cuda:0 --out .../s777/heldout_sni_acc.jsonl
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapterbench.cli._shared import PILOT_DEFAULT_TARGET_MODULES, T2L_DECONTAM_CONFIG, load_frozen_interpreter  # noqa: E402
from adapterbench.t2p.condition_encoder import DEFAULT_CONDITION_ENCODER, embed_task_descriptions, load_condition_encoder  # noqa: E402
from adapterbench.t2p.hypernetwork import StaticAdapter, TextToPeftHypernetwork, infer_module_shapes  # noqa: E402
from adapterbench.t2p.lol_data import format_prompt_response, load_task_metadata, preprocess_lol_example  # noqa: E402
from datasets import load_dataset  # noqa: E402
from t2p_train_ddp import register_persistent_hooks  # noqa: E402


def normalize(text: str) -> str:
    text = text.strip().split("\n")[0].strip().lower()
    return re.sub(r"[\s]+", " ", text).strip(" .,:;!?\"'`)")


def is_correct(generated: str, gold: str) -> bool:
    g, a = normalize(generated), normalize(gold)
    if not a:
        return False
    return g == a or a in g  # exact, or gold appears in the generation


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interpreter", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--static-snapshot", required=True)
    p.add_argument("--step", type=int, default=-1)
    p.add_argument("--condition-encoder", default=DEFAULT_CONDITION_ENCODER)
    p.add_argument("--decontam-config", default=str(T2L_DECONTAM_CONFIG))
    p.add_argument("--tasks-root", default="data/t2l/tasks")
    p.add_argument("--tasks", default="")
    p.add_argument("--limit", type=int, default=48, help="examples per task")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--adapter", default="lora")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--ia3-scaling", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="")
    return p.parse_args()


def main() -> None:
    args = build_args()
    device = args.device
    step = args.step if args.step >= 0 else int(re.search(r"step(\d+)", Path(args.snapshot).name).group(1))

    decontam = yaml.safe_load(Path(args.decontam_config).read_text())
    eval_ds_info = decontam["eval_ds_info"]
    task_ids = [t for t in args.tasks.split(",") if t] if args.tasks else [k for k in eval_ds_info if str(k).startswith("lol_")]

    print(f"[1/3] loading interpreter {args.interpreter} on {device}...", flush=True)
    tokenizer, interpreter, layers = load_frozen_interpreter(args.interpreter, device)
    tokenizer.padding_side = "left"  # left-pad for batched generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/3] rebuilding hypernetwork + static, loading snapshots...", flush=True)
    torch.manual_seed(0)
    module_shapes = infer_module_shapes(layers, PILOT_DEFAULT_TARGET_MODULES[args.adapter], hidden_size=interpreter.config.hidden_size)
    encoder_model, encoder_tokenizer = load_condition_encoder(args.condition_encoder, device)
    with torch.no_grad():
        condition_dim = embed_task_descriptions(["probe"], encoder_model, encoder_tokenizer).shape[-1]
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=condition_dim, module_shapes=module_shapes, num_layers=len(layers),
        adapter=args.adapter, rank=args.rank, ia3_scaling=args.ia3_scaling, seed=0,
    ).to(device)
    hypernetwork.load_state_dict(torch.load(args.snapshot, map_location=device, weights_only=False)["model"])
    hypernetwork.eval()
    static = StaticAdapter(hypernetwork.codecs, len(layers)).to(device)
    static.load_state_dict(torch.load(args.static_snapshot, map_location=device, weights_only=False)["model"])
    static.eval()

    current: dict = {}
    register_persistent_hooks(hypernetwork.codecs, layers, current)

    def generate_batch(prompts, gen_fn):
        outs = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i : i + args.batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to(device)
            b = enc["input_ids"].shape[0]
            current.clear()
            current.update(gen_fn(b))
            with torch.no_grad():
                out_ids = interpreter.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for j in range(b):
                cont = out_ids[j, enc["input_ids"].shape[1]:]
                outs.append(tokenizer.decode(cont, skip_special_tokens=True))
        return outs

    rows = []
    print(f"[3/3] scoring {len(task_ids)} task(s) by generation accuracy...", flush=True)
    for ti, task_id in enumerate(task_ids):
        meta_path = Path(args.tasks_root) / task_id / "metadata.yaml"
        if not meta_path.exists():
            continue
        metadata = load_task_metadata(Path(args.tasks_root), task_id)
        raw = load_dataset(path=metadata.dataset_id, split=metadata.split, name=metadata.config_name)
        raw = raw.select(range(min(args.limit, len(raw))))
        prompts, golds = [], []
        for r in raw:
            proc = preprocess_lol_example(r)
            prompt, _ = format_prompt_response(tokenizer, proc["task_def"], proc["problem"], proc["answer"], "{problem}")
            prompts.append(prompt)
            golds.append(proc["answer"])
        if not prompts:
            continue

        desc = eval_ds_info[task_id]["descriptions"][0]
        with torch.no_grad():
            demb = embed_task_descriptions([desc], encoder_model, encoder_tokenizer)[0].to(device)

        gen_matched = generate_batch(prompts, lambda b: hypernetwork(demb.unsqueeze(0).expand(b, -1)))
        gen_static = generate_batch(prompts, lambda b: static(b))
        gen_frozen = generate_batch(prompts, lambda b: {k: torch.zeros_like(v) for k, v in static(b).items()})

        n = len(golds)
        acc_m = sum(is_correct(g, a) for g, a in zip(gen_matched, golds)) / n
        acc_s = sum(is_correct(g, a) for g, a in zip(gen_static, golds)) / n
        acc_f = sum(is_correct(g, a) for g, a in zip(gen_frozen, golds)) / n
        row = {"step": step, "task_id": task_id, "n": n, "acc_matched": acc_m, "acc_static": acc_s,
               "acc_frozen": acc_f, "matched_minus_static": acc_m - acc_s, "matched_minus_frozen": acc_m - acc_f}
        rows.append(row)
        print(f"  [{ti+1}/{len(task_ids)}] {task_id:10} n={n:3} matched={acc_m:.3f} static={acc_s:.3f} "
              f"frozen={acc_f:.3f}  m-static={acc_m-acc_s:+.3f}  m-frozen={acc_m-acc_f:+.3f}", flush=True)

    if rows:
        k = len(rows)
        agg = {"step": step, "task_id": "__aggregate__", "n_tasks": k,
               "acc_matched": sum(r["acc_matched"] for r in rows) / k,
               "acc_static": sum(r["acc_static"] for r in rows) / k,
               "acc_frozen": sum(r["acc_frozen"] for r in rows) / k,
               "matched_minus_static": sum(r["matched_minus_static"] for r in rows) / k,
               "matched_minus_frozen": sum(r["matched_minus_frozen"] for r in rows) / k,
               "frac_tasks_matched_gt_static": sum(r["matched_minus_static"] > 0 for r in rows) / k,
               "frac_tasks_matched_ge_static": sum(r["matched_minus_static"] >= 0 for r in rows) / k}
        print(f"\n  MEAN over {k} tasks: matched={agg['acc_matched']:.3f} static={agg['acc_static']:.3f} frozen={agg['acc_frozen']:.3f}")
        print(f"  matched - static = {agg['matched_minus_static']:+.4f}   (positive = conditioning helps accuracy)")
        print(f"  matched - frozen = {agg['matched_minus_frozen']:+.4f}")
        print(f"  tasks matched > static: {agg['frac_tasks_matched_gt_static']*100:.0f}%  (>= : {agg['frac_tasks_matched_ge_static']*100:.0f}%)")
        rows.append(agg)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"appended {len(rows)} row(s) to {out}", flush=True)


if __name__ == "__main__":
    main()
