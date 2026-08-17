"""Write the T2A canonical records from the confirmation result.

Two jobs, both of which must be exact rather than transcribed:

1. **Per-seed source artifacts.** `results.validate_record` requires at least one artifact per seed,
   repository-relative, with a real SHA-256. Scoring writes shard files, so this emits one JSONL per
   (codec, seed) carrying that seed's matched and static rows -- per-task and aggregate -- and
   digests it.

2. **The canonical records.** Generated from `confirmation_result.json` and passed through
   `validate_record`, so `headline.value == matched - control`, `headline.value == mean(seed deltas)`
   and `headline.variation == sample SD(seed deltas)` hold to the validator's tolerance. The
   variation is the SD of the *paired* per-seed deltas: the two roles share a seed index, so the
   paired statistic is both correct and what the schema checks.

The headline metric is `held_out_rouge_l` with `direction: higher_is_better`. Exact match and CE are
recorded under `summary` as appendix figures, which is the role AUTORESEARCH.md assigns CE: a
divergence detector and an eligibility gate that never selects. `free_hyperparameters.static_control`
records the control's own scale and LR, which need not equal the hypernetwork's.

Usage:
    .venv/bin/python scripts/t2a_write_canonical.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from adapterbench.results import validate_record  # noqa: E402

RESCORE = REPO / "results/autoresearch/t2a/evaluation"

# The canonical filename per codec. `lora` keeps its `_r8` record name and `lora_r8` codec id, which
# the renderers special-case.
RECORD = {"lora": "t2a_lora_r8.json", "ia3": "t2a_ia3.json", "lokr": "t2a_lokr.json",
          "fourierft": "t2a_fourierft.json", "steering": "t2a_steering.json",
          "dora": "t2a_dora.json"}
CODEC_ID = {"lora": "lora_r8", "ia3": "ia3", "lokr": "lokr", "fourierft": "fourierft",
            "steering": "steering", "dora": "dora"}
DISPLAY = {"lora": "LoRA", "ia3": "(IA)³", "lokr": "LoKr", "fourierft": "FourierFT",
           "steering": "Steering", "dora": "DoRA"}

MODELS = [
    {"model_id": "google/gemma-2-2b-it", "revision": "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"},
    {"model_id": "Alibaba-NLP/gte-large-en-v1.5", "revision": "104333d6af6f97649377c2afbde10a7704870c7b"},
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def load_rows() -> tuple[dict, dict]:
    def read(sub):
        out = {}
        for f in sorted((RESCORE / sub).glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        out[r["checkpoint"]] = r
        return out
    return read("report_scores"), read("report_scores_ce")


def emit_seed_artifacts(gen: dict, ce: dict, dry: bool) -> dict:
    """One JSONL per (codec, seed): both roles, per-task and aggregate, plus the CE appendix."""
    by: dict[tuple, dict] = {}
    for ckpt, r in gen.items():
        by.setdefault((r["codec"], r["seed"]), {})[r["scored_role"]] = r
    artifacts: dict[str, list[dict]] = {}
    for (codec, seed), roles in sorted(by.items()):
        if set(roles) != {"matched", "static"}:
            raise SystemExit(f"{codec} seed {seed}: expected both roles, got {sorted(roles)}")
        rel = f"results/autoresearch/t2a/{codec}/confirmation_v2/report_split_s{seed}.jsonl"
        lines = []
        for role in ("matched", "static"):
            r = roles[role]
            for task, v in sorted(r["per_task"].items()):
                lines.append({"role": role, "seed": seed, "task_id": task, "n": v["n"],
                              "rouge_l": v["rouge_l"], "em": v["em"],
                              "rouge_l_untruncated": v["rouge_l_untruncated"]})
            lines.append({"role": role, "seed": seed, "task_id": "__aggregate__",
                          "n_tasks": r["n_tasks"], **r["aggregate"],
                          "ce": ce.get(r["checkpoint"], {}).get("aggregate", {}).get("ce"),
                          "checkpoint": r["checkpoint"], "instrument": r["instrument"]})
        lines.append({"role": "frozen", "seed": seed, "task_id": "__aggregate__",
                      **roles["matched"]["frozen_aggregate"]})
        payload = "".join(json.dumps(x) + "\n" for x in lines).encode()
        if not dry:
            p = REPO / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(payload)
        artifacts.setdefault(codec, []).append(
            {"seed": seed, "path": rel, "sha256": sha256_bytes(payload)})
    return artifacts


def ledger_trail(codec: str) -> list[dict]:
    p = REPO / f"results/autoresearch/t2a/{codec}/state.jsonl"
    return [{"path": str(p.relative_to(REPO)), "sha256": sha256_file(p)}] if p.exists() else []


def ranking_note(codec: str, separable: list[dict]) -> str:
    """What the confirmation seeds actually support about this shape versus the others.

    Adjacent pairs all fail to separate, but the extremes do, so "order nothing" would understate
    the evidence. Only pairs that clear 2x the SE of their difference are stated.
    """
    if not separable:
        return ("no pair of shapes separates at the measured confirmation-seed spread; report each "
                "shape against its own control and do not order the shapes")
    better = [p["worse"] for p in separable if p["better"] == CODEC_ID[codec] or p["better"] == codec]
    worse = [p["better"] for p in separable if p["worse"] == CODEC_ID[codec] or p["worse"] == codec]
    parts = []
    if better:
        parts.append("separably better than " + ", ".join(sorted(better)))
    if worse:
        parts.append("separably worse than " + ", ".join(sorted(worse)))
    if not parts:
        parts.append("not separable from any other shape")
    return (
        "PARTIAL ORDER ONLY. " + "; ".join(parts)
        + ". Every other comparison, including every adjacent one, falls below 2x the SE of the "
          "difference and must not be reported as an ordering. The supported relations across all "
          "shapes are: " + "; ".join(f"{p['better']} > {p['worse']} ({p['ratio']:.1f}x SE)"
                                          for p in separable) + "."
    )


def build_record(codec: str, entry: dict, frozen: dict, artifacts: list[dict],
                 instrument: dict, separable: list[dict]) -> dict:
    hyper, static = entry["hyper_config"], entry["static_config"]
    ms = entry["matched"]["per_seed"]
    ss = entry["static_star"]["per_seed"]
    seeds = sorted(int(s) for s in ms)
    seed_results = [{"seed": s, "matched": ms[str(s)], "control": ss[str(s)],
                     "delta": ms[str(s)] - ss[str(s)]} for s in seeds]
    deltas = [r["delta"] for r in seed_results]
    value = sum(deltas) / len(deltas)
    var = math.sqrt(sum((d - value) ** 2 for d in deltas) / (len(deltas) - 1))
    matched_mean = sum(ms.values()) / len(ms)
    control_mean = sum(ss.values()) / len(ss)
    wins = sum(1 for d in deltas if d > 0)

    return {
        "schema_version": 1,
        "setting": "T2A",
        "codec": CODEC_ID[codec],
        "display_name": DISPLAY[codec],
        "metric": "held_out_rouge_l",
        "headline": {
            "comparison": "matched_minus_control",
            "control_name": "independently_selected_same_shape_static_adapter",
            "matched": matched_mean,
            "control": control_mean,
            "value": value,
            "variation": var,
            "variation_kind": "sample_standard_deviation",
            "unit": "rouge_l_f_measure",
            "direction": "higher_is_better",
        },
        "seed_results": seed_results,
        "difficulty_curve": [{
            "axis": "held_out_sni_task_seed_pairs",
            "matched": matched_mean, "control": control_mean, "delta": value,
            "note": f"{wins}/{len(deltas)} confirmation seeds with a positive difference; "
                    f"{len(instrument['tasks'])} genuinely held-out tasks x "
                    f"{instrument['limit']} examples x {instrument['n_desc']} descriptions",
        }],
        "summary": {
            "wins": f"{wins}/{len(deltas)} confirmation seeds",
            "matched_minus_frozen": {"value": matched_mean - frozen["rouge_l"],
                                     "frozen_rouge_l": frozen["rouge_l"],
                                     "note": "helpfulness floor, ROUGE-L on the same split"},
            "exact_match_matched_minus_static": {
                "value": entry["em"]["delta"], "matched": entry["em"]["matched"],
                "control": entry["em"]["static"], "frozen": frozen["em"],
                "note": "appendix cross-check off the same decode pass; stricter than ROUGE-L and "
                        "near-meaningless on the open-ended tasks, so never the selector",
            },
            "cross_entropy_matched_minus_static": {
                "value": entry["ce"]["delta"], "matched": entry["ce"]["matched"],
                "control": entry["ce"]["static"], "unit": "nats",
                "direction": "lower_is_better",
                "note": "appendix only. CE is the training objective; AUTORESEARCH.md keeps it as a "
                        "divergence detector and eligibility gate and forbids it as a selector. On "
                        "this split CE and ROUGE-L rank the shapes with Spearman -0.50, so CE would "
                        "have selected differently and worse.",
            },
            "tied_operating_points": entry["tied_set_size"],
            "shape_ranking": ranking_note(codec, separable),
        },
        "free_hyperparameters": {
            "learning_rate": hyper["lr"],
            "warmup_fraction": 0.1,
            "steps": hyper["steps"],
            "scale": hyper["scale"],
            "static_control": {"scale": static["scale"], "learning_rate": static["lr"],
                               "steps": static["steps"],
                               "note": "selected independently on the static's own selection-split "
                                       "ROUGE-L, not yoked to the hypernetwork's point"},
        },
        "fixed_shape_parameters": FIXED_SHAPE[codec],
        "selection_trail": {
            "protocol": "AUTORESEARCH.md",
            "state_artifacts": ledger_trail(codec),
            "summary": SELECTION_SUMMARY[codec],
        },
        "provenance": {
            "models": MODELS,
            "data": [
                {"path": "data/t2a/hyper_lora_decontam_lol_tasks.yaml",
                 "sha256": sha256_file(REPO / "data/t2a/hyper_lora_decontam_lol_tasks.yaml")},
                {"path": "data/t2a/eval_ds_info.yaml",
                 "sha256": sha256_file(REPO / "data/t2a/eval_ds_info.yaml")},
            ],
            "source_artifacts": artifacts,
            "evaluation": {
                "split": "report", "tasks": instrument["tasks"],
                "examples_per_task": instrument["limit"],
                "descriptions_per_task": instrument["n_desc"],
                "decoding": "greedy", "max_new_tokens": instrument["max_new_tokens"],
                "example_offset": instrument["example_offset"],
                "note": "the 11 lol_ tasks absent from train_ds_names, scored exactly once",
            },
            "environment": {"python": "3.11.15",
                            "lockfile_sha256": sha256_file(REPO / "uv.lock")},
        },
        "reproduction": {
            # `script` is the single entry point a reader would run to reproduce this row's
            # confirmation seeds; the stage keys below are the full pipeline that selected the point.
            # The committed reproduce wrapper, not the raw driver: re-running this writer must not
            # silently downgrade an already-published record's entry point (it did once).
            "script": f"scripts/reproduce/t2a_reproduce_all.sh {codec}",
            "scout": "scripts/t2a_sweep_axes.py",
            "selection": "scripts/t2a_selection_seeds.py",
            "confirmation": "scripts/t2a_confirmation_seeds.py",
            "score": f"scripts/reproduce/t2a_score_codec.sh {codec}",
            "aggregate": "scripts/t2a_confirmation_result.py",
        },
    }


FIXED_SHAPE = {
    "lora": {"target_modules": ["q_proj", "v_proj"], "rank": 8, "use_rslora": True},
    "ia3": {"target_modules": ["q_proj", "v_proj"], "generated_scalars_per_layer": 3072},
    "lokr": {"target_modules": ["q_proj", "v_proj"],
             "factorization": "deterministic balanced Kronecker factorization",
             "generated_scalars_per_layer": 5120},
    "fourierft": {"target_modules": ["q_proj", "v_proj"],
                  "factorization": "fixed seeded 2D orthonormal DCT-II frequency-pair basis",
                  "generated_scalars_per_layer": 61440,
                  "n_freqs_rule": "8 * (in_features + out_features)"},
    "steering": {"target_modules": ["block"],
                 "note": "activation-space codec: hooks the whole decoder layer and adds one "
                         "generated residual-stream vector per (layer, example); performs no "
                         "weight update at all"},
    # DoRA: rank-8 directional factors over q_proj (2304->2048) and v_proj (2304->1024) plus one
    # magnitude scalar per output channel, i.e. 8*(2304+2048)+2048 = 36,864 and
    # 8*(2304+1024)+1024 = 27,648 per layer.
    "dora": {"target_modules": ["q_proj", "v_proj"], "rank": 8,
             "generated_scalars_per_layer": 64512,
             "budget_note": "the locked rank-8 LoRA budget plus d_out per projection for the "
                            "magnitude vector (+5.9% on q_proj, +3.8% on v_proj); the magnitude "
                            "vector is DoRA's shape identity, not a tunable extra",
             "note": "weight-decomposed codec: the only registered shape whose update reads the "
                     "frozen weight it edits, applied as "
                     "W -> m * (W0 + scale*B@A)/||W0 + scale*B@A||_row with a detached "
                     "denominator and generated magnitudes as a delta on the frozen row norms"},
}

_COMMON = (" Re-derived under the corrected T2A rules (AUTORESEARCH.md): ROUGE-L rather than the "
           "training objective, a static control selected independently on its own score, and "
           "selection data disjoint from the report split on BOTH task and example axes -- the "
           "selection split is the 10 in-distribution lol_ tasks at example offset 40, since "
           "training consumes the leading 40 examples of the same split. Every swept axis closed "
           "with an interior optimum; the operating point passed the §3b stability gate 3/3 on "
           "selection seeds disjoint from these confirmation seeds.")
SELECTION_SUMMARY = {
    "lora": "Scale moved from 22.627417 to 1.414214 and LR from 1e-4 to 5e-5; the static's own "
            "optimum stayed at scale 22.627417 but moved to lr 5e-5, with lr 2.5e-5 worse in both "
            "roles." + _COMMON,
    "ia3": "Scale stayed at 16 but the hypernetwork's LR moved from 4e-4 to 5e-5 (lr 2.5e-5 worse). "
           "The static's own optimum is lr 8e-4 -- the learning rate at which the hypernetwork "
           "catastrophically fails the helpfulness floor -- so a yoked control could never have "
           "been trained there; lr 1.6e-3 was worse, closing the axis." + _COMMON,
    "lokr": "Scale moved from 16 to 0.25 and LR from 1e-4 to 2e-4, stable across both example "
            "offsets. The static's own optimum is also scale 0.25 lr 2e-4: lr 4e-4 tied within "
            "0.0001 (seed SD 0.021) and lr 8e-4 was worse, so the axis closed at the interior "
            "point under §2's materiality threshold." + _COMMON,
    "fourierft": "Scale moved from 16 to 0.25 with LR held at 1e-4 (5e-5 and 2e-4 both worse), "
                 "stable across both example offsets. The static's own optimum is scale 4 lr 1e-4, "
                 "interior on every swept axis." + _COMMON,
    "steering": "First confirmed T2A row for this codec: its earlier search stopped at a "
                "selection_promotion_declaration for scale 64 lr 1e-4 that was never run. Under "
                "the corrected rules the hypernetwork's optimum is scale 1 lr 1e-4 and the static's "
                "own optimum is scale 64 lr 2e-4 (lr 4e-4 worse). The codec's own ledger had "
                "already documented that a yoked control can be inflated by handicapping its "
                "optimization; under the independent control its degenerate scale-0.0625 point "
                "falls from rank 1 of 9 to rank 5." + _COMMON,
    "dora": "Both roles closed with interior optima on BOTH free axes, and they landed far apart: the "
            "hypernetwork at scale 0.25 lr 1e-4 (0.7609 on the scout; scale 0.0625 gives 0.7143 and 1.0 gives "
            "0.7252; lr 5e-5 gives 0.7357 and 2e-4 gives 0.7093) and the static at scale 16 lr 1e-4 (0.6074; its "
            "mandatory upward extension to 64 scored 0.5537, below 16, so the second extension step was not "
            "licensed; lr 5e-5 gives 0.5683 and 2e-4 gives 0.5641). The 64x scale gap between the two roles is "
            "direct evidence for the independent-control rule: a control yoked to the hypernetwork's scale 0.25 "
            "would have scored 0.4867 rather than 0.6074, inflating this row by about 0.12. The LR axis was swept "
            "rather than assumed because DoRA's D2A search established that its scale and LR axes interact; in T2A "
            "they did not, and the setting default survived. Both roles passed the section 3b stability gate 3/3 on "
            "selection seeds 6711-6713 (hyper mean 0.7272, SD 0.0140; static mean 0.5895, SD 0.0169; divergence "
            "0/3 for both), and the measured SD is tighter than the 0.021 materiality threshold used to close the "
            "ladders. Note the hyper multi-seed mean 0.7272 sits below the 0.7609 scout peak that selected the "
            "point, which is the winner's-curse gap the gate exists to expose." + _COMMON,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    result = json.loads((RESCORE / "confirmation_result.json").read_text())
    gen, ce = load_rows()
    artifacts = emit_seed_artifacts(gen, ce, args.dry_run)
    instrument = result["instrument"]

    for codec, entry in result["codecs"].items():
        record = build_record(codec, entry, result["frozen"], artifacts[codec], instrument,
                              result.get("separable_pairs", []))
        validate_record(record)          # the arbiter: refuses inconsistent arithmetic
        path = REPO / "canonical_results" / RECORD[codec]
        h = record["headline"]
        print(f"  {RECORD[codec]:24} matched {h['matched']:.4f}  control {h['control']:.4f}  "
              f"value {h['value']:+.4f} ± {h['variation']:.4f}  [validated]")
        if not args.dry_run:
            path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\n{'dry run: nothing written' if args.dry_run else 'wrote 5 canonical records'}")


if __name__ == "__main__":
    main()
