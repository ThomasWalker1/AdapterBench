"""Generation-based ROUGE-L + exact-match on the held-out SNI tasks under strip-task-def.

The CE eval (`t2a_eval_heldout_sni.py`) measures the probability the adapter assigns to the gold
answer -- the quantity the trainer optimizes. This measures behavior: does the conditioned adapter
make the model *generate* the right answer more often than the static reference or the frozen
model? Same definition-stripped prompts, greedy decoding.

Two metrics come off **one** decode pass, because generation is the expensive step:

  * **ROUGE-L** -- Super-NaturalInstructions' own aggregate metric (LCS F-measure, see
    `adapterbench.scoring.rouge_l`). This is T2A's selection and headline metric.
  * **normalized exact match** -- stricter and unambiguous, so it catches ROUGE-L partial-credit
    inflation, but near-meaningless on the open-ended tasks (poem generation, grammar correction).
    Reported as a cross-check, never as the selector.

Exact match is meaningful for the classification / extraction tasks (the majority: MNLI, disease
NER, MMMLU, word-overlap, ...); per-task rows are emitted so those can be read separately.

Both metrics are computed on the generation truncated at its first newline, matching this repo's
long-standing exact-match convention. ROUGE-L is *also* reported untruncated
(`rouge_l_*_untruncated`) because its precision term penalizes an over-long decode: with a finite
`--max-new-tokens`, truncation choice can move the number, and the pair makes that visible instead
of hiding it behind a convention.

Usage:
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/t2a_eval_heldout_sni_acc.py \
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

from adapterbench.cli._shared import PILOT_DEFAULT_TARGET_MODULES, T2A_DECONTAM_CONFIG, load_frozen_interpreter  # noqa: E402
from adapterbench.scoring import rouge_l  # noqa: E402
from adapterbench.t2a.condition_encoder import DEFAULT_CONDITION_ENCODER, embed_task_descriptions, load_condition_encoder  # noqa: E402
from adapterbench.t2a.hypernetwork import StaticAdapter, TextToPeftHypernetwork, infer_module_shapes  # noqa: E402
from adapterbench.t2a.lol_data import format_prompt_response, load_task_metadata, preprocess_lol_example  # noqa: E402
from datasets import load_dataset  # noqa: E402
from t2a_train_ddp import register_persistent_hooks  # noqa: E402

# Roles scored per snapshot pair, in the order they are generated.
ROLES = ("matched", "static", "frozen")


def normalize(text: str) -> str:
    text = text.strip().split("\n")[0].strip().lower()
    return re.sub(r"[\s]+", " ", text).strip(" .,:;!?\"'`)")


def is_correct(generated: str, gold: str) -> bool:
    g, a = normalize(generated), normalize(gold)
    if not a:
        return False
    return g == a or a in g  # exact, or gold appears in the generation


def first_line(text: str) -> str:
    return str(text).strip().split("\n")[0].strip()


def score_generations(generations: list[str], golds: list[str]) -> dict:
    """Read every metric off one set of generations.

    `em` is this repo's normalized exact match. `rouge_l` is SNI's aggregate metric on the first
    generated line;
    `rouge_l_untruncated` is the same metric on the full decode, which differs only when the
    model runs past its answer.
    """
    n = len(golds)
    if n == 0:
        return {"em": 0.0, "rouge_l": 0.0, "rouge_l_untruncated": 0.0, "n": 0}
    return {
        "em": sum(is_correct(g, a) for g, a in zip(generations, golds)) / n,
        "rouge_l": sum(rouge_l(first_line(g), a) for g, a in zip(generations, golds)) / n,
        "rouge_l_untruncated": sum(rouge_l(g, a) for g, a in zip(generations, golds)) / n,
        "n": n,
    }


def mean_of_scores(scores: list[dict]) -> dict:
    """Average per-description score dicts (matched is averaged over description variants,
    exactly as the CE eval averages matched CE over them)."""
    keys = [k for k in scores[0] if k != "n"]
    out = {k: sum(s[k] for s in scores) / len(scores) for k in keys}
    out["n"] = scores[0]["n"]
    return out


def build_task_prompts(tokenizer, tasks_root: Path, task_id: str, limit: int,
                       offset: int = 0) -> tuple[list[str], list[str]]:
    """Definition-stripped prompts (`"{problem}"` template) and gold answers for one task.

    `offset` skips the first N examples. Training takes the FIRST `--limit` examples of the same
    `train[:10000]` split (`LolSFTDataset` also selects a leading range), so for a task that is in
    `train_ds_names` an evaluation at offset 0 replays training examples verbatim. Setting
    `offset` to the training limit makes the evaluation example-disjoint from training on those
    tasks -- see AUTORESEARCH.md "T2A: metric, data split, and the independently selected control".

    Returns `([], [])` when the task has no vendored metadata or the offset exhausts it, so a
    caller can skip it.
    """
    if not (Path(tasks_root) / task_id / "metadata.yaml").exists():
        return [], []
    metadata = load_task_metadata(Path(tasks_root), task_id)
    raw = load_dataset(path=metadata.dataset_id, split=metadata.split, name=metadata.config_name)
    raw = raw.select(range(min(offset, len(raw)), min(offset + limit, len(raw))))
    prompts, golds = [], []
    for r in raw:
        proc = preprocess_lol_example(r)
        prompt, _ = format_prompt_response(tokenizer, proc["task_def"], proc["problem"], proc["answer"], "{problem}")
        prompts.append(prompt)
        golds.append(proc["answer"])
    return prompts, golds


def make_generate_batch(tokenizer, interpreter, current: dict, *, batch_size: int, max_new_tokens: int,
                        max_len: int, device: str):
    """Greedy batched decode with the codec hooks reading generated params from `current`."""

    def generate_batch(prompts: list[str], gen_fn) -> list[str]:
        outs = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            b = enc["input_ids"].shape[0]
            current.clear()
            current.update(gen_fn(b))
            with torch.no_grad():
                out_ids = interpreter.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for j in range(b):
                cont = out_ids[j, enc["input_ids"].shape[1]:]
                outs.append(tokenizer.decode(cont, skip_special_tokens=True))
        return outs

    return generate_batch


def score_task(generate_batch, prompts: list[str], golds: list[str], *, hypernetwork, static,
               desc_embs: list[torch.Tensor], roles=ROLES) -> dict[str, dict]:
    """Generate and score one task for each requested role.

    `matched` decodes once per description embedding and averages the resulting scores; `static`
    and `frozen` are description-independent, so they decode once. Returns `{role: score_dict}`.
    """
    out: dict[str, dict] = {}
    if "matched" in roles:
        out["matched"] = mean_of_scores([
            score_generations(generate_batch(prompts, lambda b, e=emb: hypernetwork(e.unsqueeze(0).expand(b, -1))), golds)
            for emb in desc_embs
        ])
    if "static" in roles:
        out["static"] = score_generations(generate_batch(prompts, lambda b: static(b)), golds)
    if "frozen" in roles:
        out["frozen"] = score_generations(
            generate_batch(prompts, lambda b: {k: torch.zeros_like(v) for k, v in static(b).items()}), golds
        )
    return out


def build_modules(*, interpreter, layers, device, condition_dim, adapter, rank=8,
                  codec_scaling=None, lora_scaling=-1.0, ia3_scaling=1.0, lokr_scaling=1.0,
                  loha_scaling=1.0, fourierft_scaling=1.0):
    """Construct the hypernetwork and its same-shape StaticAdapter, without loading weights.

    Construction is the expensive part for codecs with large frozen buffers (FourierFT's fixed
    DCT bases), and it depends only on the codec identity -- `codec_scaling` is a plain scalar on
    each codec, and `load_snapshot` swaps weights in place. So a caller sweeping many rungs of one
    codec can build once and then vary scale and weights per rung.
    """
    torch.manual_seed(0)
    module_shapes = infer_module_shapes(layers, PILOT_DEFAULT_TARGET_MODULES[adapter],
                                       hidden_size=interpreter.config.hidden_size)
    hypernetwork = TextToPeftHypernetwork(
        condition_dim=condition_dim, module_shapes=module_shapes, num_layers=len(layers),
        adapter=adapter, rank=rank, ia3_scaling=ia3_scaling, lokr_scaling=lokr_scaling,
        loha_scaling=loha_scaling, fourierft_scaling=fourierft_scaling, seed=0,
    ).to(device)
    if lora_scaling > 0:
        from adapterbench.t2a.codecs import LoRACodec
        for codec in hypernetwork.codecs.values():
            if isinstance(codec, LoRACodec):
                codec.scaling = lora_scaling
    if codec_scaling is not None:
        apply_codec_scaling(hypernetwork, codec_scaling)
    static = StaticAdapter(hypernetwork.codecs, len(layers)).to(device)
    hypernetwork.eval()
    static.eval()
    return hypernetwork, static


def apply_codec_scaling(hypernetwork, codec_scaling: float) -> None:
    """Set the swept output scale. `StaticAdapter` shares the same codec objects, so this
    reaches both modules."""
    from adapterbench.t2a.codecs import set_codec_scaling
    set_codec_scaling(hypernetwork.codecs, codec_scaling)


def load_snapshot(module, path: str, device: str) -> None:
    module.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    module.eval()


def build_hypernetwork_and_static(args, interpreter, layers, device, condition_dim):
    """CLI path: build both modules at the snapshot's codec/scale and load their weights."""
    hypernetwork, static = build_modules(
        interpreter=interpreter, layers=layers, device=device, condition_dim=condition_dim,
        adapter=args.adapter, rank=args.rank, codec_scaling=args.codec_scaling,
        lora_scaling=args.lora_scaling, ia3_scaling=args.ia3_scaling, lokr_scaling=args.lokr_scaling,
        loha_scaling=args.loha_scaling, fourierft_scaling=args.fourierft_scaling,
    )
    load_snapshot(hypernetwork, args.snapshot, device)
    load_snapshot(static, args.static_snapshot, device)
    return hypernetwork, static


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interpreter", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--static-snapshot", required=True)
    p.add_argument("--step", type=int, default=-1)
    p.add_argument("--condition-encoder", default=DEFAULT_CONDITION_ENCODER)
    p.add_argument("--decontam-config", default=str(T2A_DECONTAM_CONFIG))
    p.add_argument("--tasks-root", default="data/t2a/tasks")
    p.add_argument("--tasks", default="", help="comma-separated task ids; default = all lol_* in eval_ds_info")
    p.add_argument("--n-desc", type=int, default=1,
                   help="description variants to average matched over. Pass 3 to average over all three "
                        "held-out variants, which is what the T2A protocol scores.")
    p.add_argument("--limit", type=int, default=48, help="examples per task")
    p.add_argument("--example-offset", type=int, default=0,
                   help="skip the first N examples of each task. Training consumes a LEADING range "
                        "of the same split, so set this to the training --limit (40 for every "
                        "recorded T2A run) to score a trained-on task on examples it never saw.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--adapter", default="lora")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora-scaling", type=float, default=-1.0)
    p.add_argument("--ia3-scaling", type=float, default=1.0)
    p.add_argument("--lokr-scaling", type=float, default=1.0)
    p.add_argument("--loha-scaling", type=float, default=1.0)
    p.add_argument("--fourierft-scaling", type=float, default=1.0)
    p.add_argument("--codec-scaling", type=float, default=None,
                   help="generic output-scale override for the selected --adapter's codec; "
                        "must match the value the snapshot was trained with")
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
    encoder_model, encoder_tokenizer = load_condition_encoder(args.condition_encoder, device)
    with torch.no_grad():
        condition_dim = embed_task_descriptions(["probe"], encoder_model, encoder_tokenizer).shape[-1]
    hypernetwork, static = build_hypernetwork_and_static(args, interpreter, layers, device, condition_dim)

    current: dict = {}
    register_persistent_hooks(hypernetwork.codecs, layers, current)
    generate_batch = make_generate_batch(tokenizer, interpreter, current, batch_size=args.batch_size,
                                         max_new_tokens=args.max_new_tokens, max_len=args.max_len, device=device)

    rows = []
    print(f"[3/3] scoring {len(task_ids)} task(s) by generation ROUGE-L + exact match...", flush=True)
    for ti, task_id in enumerate(task_ids):
        prompts, golds = build_task_prompts(tokenizer, args.tasks_root, task_id, args.limit,
                                            offset=args.example_offset)
        if not prompts:
            continue
        descs = eval_ds_info[task_id]["descriptions"][: args.n_desc]
        with torch.no_grad():
            desc_embs = [embed_task_descriptions([d], encoder_model, encoder_tokenizer)[0].to(device) for d in descs]

        scored = score_task(generate_batch, prompts, golds, hypernetwork=hypernetwork, static=static,
                            desc_embs=desc_embs)
        row = {"step": step, "task_id": task_id, "n": len(golds), "n_desc": len(desc_embs),
               "example_offset": args.example_offset}
        for role in ROLES:
            row[f"acc_{role}"] = scored[role]["em"]
            row[f"rouge_{role}"] = scored[role]["rouge_l"]
            row[f"rouge_{role}_untruncated"] = scored[role]["rouge_l_untruncated"]
        row["matched_minus_static"] = row["acc_matched"] - row["acc_static"]
        row["matched_minus_frozen"] = row["acc_matched"] - row["acc_frozen"]
        row["rouge_matched_minus_static"] = row["rouge_matched"] - row["rouge_static"]
        row["rouge_matched_minus_frozen"] = row["rouge_matched"] - row["rouge_frozen"]
        rows.append(row)
        print(f"  [{ti+1}/{len(task_ids)}] {task_id:10} n={len(golds):3} "
              f"rougeL m={row['rouge_matched']:.3f} s={row['rouge_static']:.3f} f={row['rouge_frozen']:.3f} "
              f"(m-s={row['rouge_matched_minus_static']:+.3f}) | "
              f"EM m={row['acc_matched']:.3f} s={row['acc_static']:.3f} f={row['acc_frozen']:.3f} "
              f"(m-s={row['matched_minus_static']:+.3f})", flush=True)

    if rows:
        k = len(rows)
        mean = lambda key: sum(r[key] for r in rows) / k  # noqa: E731
        agg = {"step": step, "task_id": "__aggregate__", "n_tasks": k,
               **{key: mean(key) for key in rows[0] if key not in ("step", "task_id", "n", "n_desc")},
               "frac_tasks_matched_gt_static": sum(r["matched_minus_static"] > 0 for r in rows) / k,
               "frac_tasks_matched_ge_static": sum(r["matched_minus_static"] >= 0 for r in rows) / k,
               "frac_tasks_rouge_matched_gt_static": sum(r["rouge_matched_minus_static"] > 0 for r in rows) / k}
        print(f"\n  MEAN over {k} tasks:")
        print(f"    ROUGE-L: matched={agg['rouge_matched']:.4f} static={agg['rouge_static']:.4f} "
              f"frozen={agg['rouge_frozen']:.4f}")
        print(f"    ROUGE-L matched - static = {agg['rouge_matched_minus_static']:+.4f}   "
              f"(positive = conditioning helps; this is the selection metric)")
        print(f"    ROUGE-L matched - frozen = {agg['rouge_matched_minus_frozen']:+.4f}   (helpfulness floor)")
        print(f"    EM:      matched={agg['acc_matched']:.4f} static={agg['acc_static']:.4f} "
              f"frozen={agg['acc_frozen']:.4f}")
        print(f"    EM matched - static = {agg['matched_minus_static']:+.4f}")
        print(f"    tasks matched > static: ROUGE-L {agg['frac_tasks_rouge_matched_gt_static']*100:.0f}%  "
              f"EM {agg['frac_tasks_matched_gt_static']*100:.0f}%")
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
