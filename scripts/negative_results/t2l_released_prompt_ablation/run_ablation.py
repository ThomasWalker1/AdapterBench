"""Prompt-conditioning ablation for a RELEASED Text-to-LoRA checkpoint.

This is a negative-results verification, NOT a benchmark setting. It answers one
question about the *standard, input-visible* Text-to-LoRA setting (NEGATIVE_RESULTS.md
section 1): given one of Sakana's own published T2L hypernetworks, does conditioning it
on the *correct* per-task description actually beat conditioning it on a wrong task's
description ("shuffled") or on meaningless text ("junk")?

For each held-out family the frozen interpreter sees an identical, self-describing
benchmark prompt (ARC / BoolQ / HellaSwag question). Only the text handed to the
hypernetwork changes:

  matched   family's own task description        (args.yaml: eval_ds_info[family])
  shuffled  a DIFFERENT family's real description (rotate families by one)
  junk      meaningless strings                  (args.yaml: additional_eval_descs)

If matched is not reliably above shuffled/junk, the generated adapter is generically
helpful but the per-prompt condition is not load-bearing -- the negative result.

Two environments: adapter generation runs under the upstream text-to-lora venv
(``--gen-python``, which has ``hyper_llm_modulator``); downstream evaluation runs under
this interpreter (plain transformers+peft). The two never share a process; they only
exchange saved PEFT adapters + a JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so ``t2l_released_prompt_ablation`` is importable

from t2l_released_prompt_ablation._contracts import AdapterArtifact  # noqa: E402
from t2l_released_prompt_ablation.hf_downstream_evaluator import HFDownstreamEvaluator  # noqa: E402
from t2l_released_prompt_ablation.task_examples import (  # noqa: E402
    build_all_task_examples,
    load_task_descriptions,
)

import yaml  # noqa: E402

KINDS = ("matched", "shuffled", "junk")


def build_condition_sets(args_yaml: Path, families: list[str], variant: int) -> dict[str, dict[str, str]]:
    """Return {kind: {family: condition_text}} for matched / shuffled / junk."""
    descriptions = load_task_descriptions(args_yaml)
    payload = yaml.safe_load(args_yaml.read_text())
    junk_pool = payload.get("additional_eval_descs") or []
    if not junk_pool:
        raise SystemExit(f"no additional_eval_descs (junk) in {args_yaml}")

    matched = {f: descriptions[f][variant] for f in families}
    # shuffled: each family gets the NEXT family's real description (rotate by one).
    rotated = families[1:] + families[:1]
    shuffled = {f: descriptions[other][variant] for f, other in zip(families, rotated)}
    # junk: cycle through the meaningless strings.
    junk = {f: junk_pool[i % len(junk_pool)] for i, f in enumerate(families)}
    return {"matched": matched, "shuffled": shuffled, "junk": junk}


def generate_all_adapters(
    gen_python: Path,
    gen_cwd: Path,
    checkpoint: Path,
    condition_sets: dict[str, dict[str, str]],
    output_dir: Path,
    device: str,
) -> dict:
    """One subprocess, one model load: generate every (family, kind) adapter."""
    conditions: dict[str, str] = {}
    for kind, mapping in condition_sets.items():
        for family, text in mapping.items():
            conditions[f"{family}__{kind}"] = text

    output_dir.mkdir(parents=True, exist_ok=True)
    conditions_path = output_dir / "conditions.json"
    conditions_path.write_text(json.dumps(conditions, indent=2))

    cmd = [
        str(gen_python),
        str(HERE / "generate_t2l_adapter.py"),
        "--checkpoint", str(checkpoint.resolve()),
        "--conditions-json", str(conditions_path.resolve()),
        "--output-dir", str(output_dir.resolve()),
        "--device", device,
    ]
    print(f"[gen] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(gen_cwd))
    return json.loads((output_dir / "manifest.json").read_text())


def artifacts_for_kind(manifest: dict, families: list[str], kind: str) -> dict[str, AdapterArtifact]:
    """Map family -> AdapterArtifact for one condition kind. ``adapter`` is made unique
    per kind so the evaluator caches the three adapters under distinct peft names."""
    out: dict[str, AdapterArtifact] = {}
    for family in families:
        entry = manifest["adapters"][f"{family}__{kind}"]
        out[family] = AdapterArtifact(
            task_id=family,
            adapter=f"lora_{kind}",
            path=Path(entry["path"]),
            format="peft",
            generated_parameter_count=entry["generated_parameter_count"],
            generation_seconds=entry["generation_seconds"],
            metadata={"condition": entry["condition"], "kind": kind},
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to a released hypermod.pt")
    p.add_argument("--interpreter", required=True, help="HF id of the frozen interpreter")
    p.add_argument("--chat-template", required=True, help="path to chat_template.jinja")
    p.add_argument("--families", default="arc_easy,arc_challenge,boolq,hellaswag")
    p.add_argument("--limit", type=int, default=200, help="examples per family")
    p.add_argument("--variant", type=int, default=0, help="which description variant is 'matched'")
    p.add_argument("--gen-python", required=True, help="upstream text-to-lora venv python")
    p.add_argument("--gen-cwd", required=True, help="upstream text-to-lora repo root (for chat-template lookups)")
    p.add_argument("--gen-device", default="cuda:0")
    p.add_argument("--eval-device", default="cuda:0")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    args_yaml = checkpoint.parent / "args.yaml"
    families = args.families.split(",")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] building matched/shuffled/junk conditions from {args_yaml}", flush=True)
    condition_sets = build_condition_sets(args_yaml, families, args.variant)
    (output / "conditions_by_kind.json").write_text(json.dumps(condition_sets, indent=2))

    print("[2/4] generating adapters (upstream venv, one model load)...", flush=True)
    manifest = generate_all_adapters(
        Path(args.gen_python), Path(args.gen_cwd), checkpoint,
        condition_sets, output / "adapters", args.gen_device,
    )

    print(f"[3/4] building up to {args.limit} example(s) per family...", flush=True)
    descriptions = load_task_descriptions(args_yaml)
    examples_by_family = build_all_task_examples(families, descriptions, args.limit, variant=args.variant)
    all_examples = [ex for group in examples_by_family.values() for ex in group]
    n_by_family = {f: len(g) for f, g in examples_by_family.items()}
    print(f"      examples: {n_by_family}", flush=True)

    print(f"[4/4] loading interpreter {args.interpreter} and scoring...", flush=True)
    evaluator = HFDownstreamEvaluator(
        model_id=args.interpreter,
        chat_template_path=args.chat_template,
        trial_id=f"t2l_released_ablation::{checkpoint.parent.name}",
        device=args.eval_device,
        use_icl=False,
    )

    acc: dict[str, dict[str, float]] = {f: {} for f in families}
    for kind in KINDS:
        arts = artifacts_for_kind(manifest, families, kind)
        for r in evaluator.iter_evaluate(arts, all_examples, split="test"):
            acc[r.task_id][kind] = r.metrics.get("accuracy", r.metrics.get("exact_match"))
            print(f"  {kind:8} {r.task_id:16} acc={acc[r.task_id][kind]:.4f}", flush=True)
    for r in evaluator.iter_evaluate_frozen(all_examples, split="test"):
        acc[r.task_id]["frozen"] = r.metrics.get("accuracy", r.metrics.get("exact_match"))
        print(f"  {'frozen':8} {r.task_id:16} acc={acc[r.task_id]['frozen']:.4f}", flush=True)

    rows = []
    for f in families:
        a = acc[f]
        rows.append({
            "family": f, "n_examples": n_by_family[f],
            "frozen": a.get("frozen"), "matched": a.get("matched"),
            "shuffled": a.get("shuffled"), "junk": a.get("junk"),
            "matched_minus_frozen": a["matched"] - a["frozen"],
            "matched_minus_shuffled": a["matched"] - a["shuffled"],
            "matched_minus_junk": a["matched"] - a["junk"],
        })

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    summary = {
        "checkpoint": str(checkpoint),
        "interpreter": args.interpreter,
        "families": families,
        "limit": args.limit,
        "variant": args.variant,
        "attn_implementation": "eager",
        "per_family": rows,
        "mean_matched_minus_frozen": mean("matched_minus_frozen"),
        "mean_matched_minus_shuffled": mean("matched_minus_shuffled"),
        "mean_matched_minus_junk": mean("matched_minus_junk"),
    }
    (output / "ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n==== SUMMARY ====", flush=True)
    print(f"{'family':16} {'frozen':>8} {'matched':>8} {'shuffled':>8} {'junk':>8} "
          f"{'m-froz':>8} {'m-shuf':>8} {'m-junk':>8}", flush=True)
    for r in rows:
        print(f"{r['family']:16} {r['frozen']:8.4f} {r['matched']:8.4f} {r['shuffled']:8.4f} "
              f"{r['junk']:8.4f} {r['matched_minus_frozen']:+8.4f} "
              f"{r['matched_minus_shuffled']:+8.4f} {r['matched_minus_junk']:+8.4f}", flush=True)
    print(f"{'MEAN':16} {'':>8} {'':>8} {'':>8} {'':>8} "
          f"{summary['mean_matched_minus_frozen']:+8.4f} "
          f"{summary['mean_matched_minus_shuffled']:+8.4f} "
          f"{summary['mean_matched_minus_junk']:+8.4f}", flush=True)
    print(f"\nwrote {output / 'ablation_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
