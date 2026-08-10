"""Select each codec's operating point from the selection-split grid.

For every codec, reads the scored grid and reports:

  1. the hypernetwork configuration maximising ROUGE-L `matched - static*`;
  2. the static's own optimum over the same grid, which is what `static*` means -- the control is
     selected on its own score, never yoked to the hypernetwork's point;
  3. whether either optimum sits at an end of a swept ladder, per (role, scale) slice, which is the
     axis actually explored and therefore the axis that needs extending;
  4. how the behavioural and CE orderings compare, since CE is an appendix figure here and a
     disagreement is expected rather than alarming.

Because `static*` is one number per codec, subtracting it cannot reorder the hypernetwork
configurations: the argmax under the corrected rule is the argmax of `matched`. That is the point --
a per-rung control lets a rung win by having a weak control.

Configurations within one measured seed SD of the argmax are reported as a tied set: a scout seed
cannot separate points inside its own noise, so the size of that set is part of the answer.

Usage:
    .venv/bin/python scripts/t2a_select_operating_point.py --codecs lora,ia3,lokr,fourierft,steering
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODECS = ("lora", "ia3", "lokr", "fourierft")
CANONICAL = {"lora": "canonical_results/t2a_lora_r8.json", "ia3": "canonical_results/t2a_ia3.json",
             "lokr": "canonical_results/t2a_lokr.json", "fourierft": "canonical_results/t2a_fourierft.json"}

# A codec with no canonical row still has a CE-selected point to compare an argmax against -- the one
# its ledger DECLARED before running selection seeds. `steering`'s T2A search stopped there, so this
# is the point the old rules were about to promote, not a published result.
DECLARED_ONLY = {
    "steering": {
        "free_hyperparameters": {"scale": "64", "learning_rate": "1e-4", "warmup_fraction": 0.1,
                                 "steps": 8000},
        "metric": "held_out_teacher_forced_cross_entropy",
        "headline": {"value": -0.4633861558989834, "matched": None, "control": 2.1493635605725028,
                     "note": "scout seed 4702 only; never confirmed"},
        "summary": {"accuracy_matched_minus_static": None},
        "provenance": "results/autoresearch/t2a/steering/state.jsonl "
                      "selection_promotion_declaration (status=declared, selection seeds never run)",
    },
}


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for f in sorted(path.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") == "ok":
                rows[r["checkpoint"]] = r  # a re-scored checkpoint's latest row wins
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    """Sample standard deviation, the statistic the leaderboards report. None below 2 samples --
    a single scout seed has no spread, and printing 0.0 would imply it did."""
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def config_key(r) -> tuple:
    return (float(r["scale"]), float(r["lr"]), int(r["steps"]))


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def fmt_scale(s: float) -> str:
    return f"{s:g}"


def fmt_cfg(k: tuple) -> str:
    return f"scale={fmt_scale(k[0])} lr={k[1]:g} steps={k[2]}"


def build_grid(gen: dict, ce: dict) -> dict:
    """{codec: {config: {role: {metric: {mean, sd, n, seeds, values}}}}}"""
    grid: dict = {}
    for ckpt, r in gen.items():
        cell = (grid.setdefault(r["codec"], {}).setdefault(config_key(r), {})
                    .setdefault(r["scored_role"], {"seeds": [], "rouge_l": [], "em": [],
                                                   "rouge_l_untruncated": [], "ce": [],
                                                   "phases": set(), "converged": [], "checkpoints": []}))
        cell["seeds"].append(r["seed"])
        cell["checkpoints"].append(ckpt)
        cell["phases"].add(r.get("phase"))
        cell["converged"].append(r.get("converged"))
        for k in ("rouge_l", "em", "rouge_l_untruncated"):
            cell[k].append(r["aggregate"][k])
        if ckpt in ce:
            cell["ce"].append(ce[ckpt]["aggregate"]["ce"])
    return grid


def summarize(cell: dict, frozen_rouge: float | None = None) -> dict:
    out = {"n_seeds": len(cell["seeds"]), "seeds": sorted(cell["seeds"]),
           "phases": sorted(p for p in cell["phases"] if p),
           "converged_false": sum(1 for c in cell["converged"] if c is False)}
    # A run at or below the frozen model has no behavioral signal at all: for `matched` that is
    # AUTORESEARCH.md's helpfulness-floor failure, and in practice it is what a diverged run looks
    # like from the outside. Counted per seed so a config's divergence RATE is visible -- a config
    # whose seeds split is exactly what the §3b stability gate keys on.
    if frozen_rouge is not None:
        below = [v for v in cell["rouge_l"] if v <= frozen_rouge]
        out["n_at_or_below_frozen"] = len(below)
        out["divergence_rate"] = len(below) / len(cell["rouge_l"]) if cell["rouge_l"] else None
        out["all_seeds_clear_frozen"] = len(below) == 0
    for k in ("rouge_l", "em", "rouge_l_untruncated", "ce"):
        out[k] = mean(cell[k])
        out[f"{k}_sd"] = sd(cell[k])
        out[f"{k}_n"] = len(cell[k])
    out["_values_rouge_l"] = list(cell["rouge_l"])
    out["_values_ce"] = list(cell["ce"])
    return out


def frozen_baseline(gen: dict) -> dict:
    """Per-codec frozen (zeroed-adapter) score, plus the max spread across codecs. Every codec is
    identity at the all-zero generated init, so a nonzero spread would mean a codec is not."""
    per_codec: dict = {}
    for r in gen.values():
        per_codec.setdefault(r["codec"], {}).setdefault("rouge_l", set()).add(
            round(r["frozen_aggregate"]["rouge_l"], 10))
        per_codec[r["codec"]].setdefault("em", set()).add(round(r["frozen_aggregate"]["em"], 10))
    out = {c: {k: sorted(v)[0] for k, v in d.items()} for c, d in per_codec.items()}
    vals = [d["rouge_l"] for d in out.values()]
    out["_max_spread_across_codecs"] = (max(vals) - min(vals)) if vals else None
    out["_within_codec_disagreements"] = {
        c: sorted(d["rouge_l"]) for c, d in per_codec.items() if len(d["rouge_l"]) > 1
    }
    return out


def seed_noise(table: dict, converged_only: bool = False) -> dict:
    """Pooled within-configuration seed SD, from the configurations that have >= 2 seeds.

    The grid is deliberately heterogeneous: most ladder rungs are a single scout seed, while the
    gate-selection and confirmation rungs carry three. Comparing a one-seed rung's peak against a
    three-seed rung's mean is the winner's curse AUTORESEARCH.md §3b exists to prevent, so any
    claim that an argmax moved has to be read against this number. Pooling uses the standard
    within-group estimator (sum of squared deviations over sum of degrees of freedom).

    `converged_only` drops configurations where some seed fell to the frozen model. Those
    configurations have a huge spread, but it is a divergence rate, not measurement noise -- pooling
    it in would inflate the yardstick and make every argmax move look unsupported.
    """
    ss, dof, groups = 0.0, 0, []
    for cfg, v in table.items():
        vals = v.get("_values_rouge_l") or []
        if len(vals) < 2:
            continue
        if converged_only and not v.get("all_seeds_clear_frozen", True):
            continue
        m = mean(vals)
        ss += sum((x - m) ** 2 for x in vals)
        dof += len(vals) - 1
        groups.append({"config": fmt_cfg(cfg), "n": len(vals), "sd": sd(vals)})
    return {"pooled_sd": (math.sqrt(ss / dof) if dof else None), "dof": dof, "groups": groups}


def top_k(table: dict, key="rouge_l", k=4, restrict_steps=None) -> list[dict]:
    items = [(cfg, v) for cfg, v in table.items() if v.get(key) is not None
             and (restrict_steps is None or cfg[2] == restrict_steps)]
    items.sort(key=lambda kv: -kv[1][key])
    return [{"config": fmt_cfg(cfg), "value": v[key], "n_seeds": v["n_seeds"],
             "sd": v.get(f"{key}_sd"), "phases": v["phases"]} for cfg, v in items[:k]]


def analyze_codec(codec: str, configs: dict, frozen_rouge: float | None = None) -> dict:
    canonical = (json.loads((REPO / CANONICAL[codec]).read_text()) if codec in CANONICAL
                 else DECLARED_ONLY[codec])
    fh = canonical["free_hyperparameters"]
    ce_cfg = (float(fh["scale"]), float(fh["learning_rate"]), int(fh["steps"]))

    matched = {k: summarize(v["matched"], frozen_rouge) for k, v in configs.items() if "matched" in v}
    static = {k: summarize(v["static"], frozen_rouge) for k, v in configs.items() if "static" in v}

    scales = sorted({k[0] for k in configs})
    lrs = sorted({k[1] for k in configs})
    steps = sorted({k[2] for k in configs})

    def argmax(table: dict, key="rouge_l", restrict_steps=None):
        items = [(k, v) for k, v in table.items() if v.get(key) is not None
                 and (restrict_steps is None or k[2] == restrict_steps)]
        return max(items, key=lambda kv: kv[1][key])[0] if items else None

    def argmin_ce_delta(restrict_steps=None):
        """The OLD rule, recomputed on the selection split: minimize `matched CE - static CE` with
        the static taken at the SAME rung. Shows whether the old selector, moved onto the corrected
        split and fidelity, still points where CE pointed on the 21-task set."""
        items = []
        for k in configs:
            m, s = matched.get(k), static.get(k)
            if m and s and m.get("ce") is not None and s.get("ce") is not None:
                items.append((k, m["ce"] - s["ce"]))
        items = [i for i in items if restrict_steps is None or i[0][2] == restrict_steps]
        return min(items, key=lambda kv: kv[1])[0] if items else None

    static_star = argmax(static)
    hyper_star = argmax(matched)
    budget = ce_cfg[2]
    static_star_eq = argmax(static, restrict_steps=budget)
    hyper_star_eq = argmax(matched, restrict_steps=budget)

    def cell(table, k):
        return table.get(k) if k is not None else None

    def boundary_of(cfg):
        """Which swept axes, if any, this configuration sits at an end of. An optimum at an
        endpoint means the ladder was never closed on that axis, and closing it is training work."""
        if cfg is None:
            return None

        def end(value, swept):
            # A one-value axis is not a closed ladder with the optimum at its end -- it was never
            # swept at all. Calling that "at the boundary" would understate what Stage 2 must do.
            if len(swept) < 2:
                return "not_swept"
            return "min" if value == swept[0] else "max" if value == swept[-1] else None

        return {
            "config": fmt_cfg(cfg),
            "scale_end": end(cfg[0], scales),
            "lr_end": end(cfg[1], lrs),
            "steps_end": end(cfg[2], steps),
        }

    boundary = None
    if static_star is not None:
        boundary = {
            "static_star_any_budget": boundary_of(static_star),
            "static_star_at_canonical_budget": boundary_of(static_star_eq),
            # The hypernetwork's own argmax can sit at an endpoint too, which is equally unclosed.
            "hyper_argmax_any_budget": boundary_of(hyper_star),
            "hyper_argmax_at_canonical_budget": boundary_of(hyper_star_eq),
            "swept_scales": scales, "swept_lrs": lrs, "swept_steps": steps,
            # How much of the grid a new argmax's own axes were actually explored at.
            "lrs_swept_at_hyper_argmax_scale": sorted({k[1] for k in configs if k[0] == hyper_star[0]}),
            "scales_swept_at_hyper_argmax_lr": sorted({k[0] for k in configs if k[1] == hyper_star[1]}),
        }

    return {
        "codec": codec,
        "canonical": {
            "ce_selected_config": list(ce_cfg),
            "ce_selected_config_str": fmt_cfg(ce_cfg),
            "recorded_metric": canonical["metric"],
            "recorded_headline": canonical["headline"],
            "recorded_accuracy_matched_minus_static": canonical["summary"].get(
                "accuracy_matched_minus_static"),
            "is_published_row": codec in CANONICAL,
            "declared_only_provenance": canonical.get("provenance"),
        },
        "grid_size": {"configs": len(configs), "matched_configs": len(matched),
                      "static_configs": len(static)},
        "ladder": {"scales": scales, "lrs": lrs, "steps": steps},
        "matched_by_config": {fmt_cfg(k): v for k, v in sorted(matched.items())},
        "static_by_config": {fmt_cfg(k): v for k, v in sorted(static.items())},
        "q1_hyper_argmax": {
            "rouge_argmax_any_budget": list(hyper_star) if hyper_star else None,
            "rouge_argmax_any_budget_str": fmt_cfg(hyper_star) if hyper_star else None,
            "rouge_argmax_at_canonical_budget": list(hyper_star_eq) if hyper_star_eq else None,
            "rouge_argmax_at_canonical_budget_str": fmt_cfg(hyper_star_eq) if hyper_star_eq else None,
            "moved_any_budget": (hyper_star != ce_cfg) if hyper_star else None,
            "moved_at_canonical_budget": (hyper_star_eq != ce_cfg) if hyper_star_eq else None,
            "at_argmax": cell(matched, hyper_star),
            "at_ce_selected": cell(matched, ce_cfg),
            "selection_split_ce_argmin_same_rung_control": (
                list(argmin_ce_delta()) if argmin_ce_delta() else None),
            "selection_split_ce_argmin_at_canonical_budget": (
                list(argmin_ce_delta(restrict_steps=budget)) if argmin_ce_delta(restrict_steps=budget) else None),
        },
        "q2_static_optimum": {
            "static_star_any_budget": list(static_star) if static_star else None,
            "static_star_any_budget_str": fmt_cfg(static_star) if static_star else None,
            "static_star_at_canonical_budget": list(static_star_eq) if static_star_eq else None,
            "static_star_at_canonical_budget_str": fmt_cfg(static_star_eq) if static_star_eq else None,
            "at_static_star": cell(static, static_star),
            "at_hypernetwork_chosen_hparams": cell(static, ce_cfg),
            "at_static_star_canonical_budget": cell(static, static_star_eq),
            "static_star_is_at_hypernetwork_point": static_star == ce_cfg,
        },
        "q3_boundary": boundary,
        "seed_noise": {
            "matched": seed_noise(matched), "static": seed_noise(static),
            "matched_converged_only": seed_noise(matched, converged_only=True),
            "static_converged_only": seed_noise(static, converged_only=True),
        },
        "frozen_rouge_l": frozen_rouge,
        "helpfulness_floor": {
            "frozen_rouge_l": frozen_rouge,
            "at_rouge_argmax": (matched.get(hyper_star) or {}).get("rouge_l"),
            "argmax_clears_floor": ((matched.get(hyper_star) or {}).get("rouge_l") or 0) > (frozen_rouge or 0),
            "configs_failing_floor": [fmt_cfg(k) for k, v in sorted(matched.items())
                                      if v.get("n_at_or_below_frozen")],
        },
        "top_matched": top_k(matched),
        "top_static": top_k(static),
        "top_matched_at_canonical_budget": top_k(matched, restrict_steps=budget),
        "top_static_at_canonical_budget": top_k(static, restrict_steps=budget),
    }


def headline(codec_analysis: dict, budget_matched: bool) -> dict:
    """`matched - static*`, i.e. the corrected headline. `budget_matched=True` restricts both
    argmaxes to the canonical step budget so no rung wins on a longer budget."""
    q1, q2 = codec_analysis["q1_hyper_argmax"], codec_analysis["q2_static_optimum"]
    m = q1["at_argmax"] if not budget_matched else codec_analysis["matched_by_config"].get(
        q1["rouge_argmax_at_canonical_budget_str"])
    s = q2["at_static_star"] if not budget_matched else codec_analysis["static_by_config"].get(
        q2["static_star_at_canonical_budget_str"])
    yoked = codec_analysis["static_by_config"].get(codec_analysis["canonical"]["ce_selected_config_str"])
    out = {"budget_matched": budget_matched}
    for label, cellv in (("matched", m), ("static_star", s), ("static_yoked", yoked)):
        for metric in ("rouge_l", "em", "ce"):
            out[f"{label}_{metric}"] = (cellv or {}).get(metric)
            out[f"{label}_{metric}_sd"] = (cellv or {}).get(f"{metric}_sd")
        out[f"{label}_n_seeds"] = (cellv or {}).get("n_seeds")
    for metric in ("rouge_l", "em"):
        a, b, c = out[f"matched_{metric}"], out[f"static_star_{metric}"], out[f"static_yoked_{metric}"]
        out[f"delta_{metric}_vs_static_star"] = (a - b) if (a is not None and b is not None) else None
        out[f"delta_{metric}_vs_static_yoked"] = (a - c) if (a is not None and c is not None) else None
    # CE is lower-is-better, so its controlled quantity keeps the matched - control sign convention
    a, b, c = out["matched_ce"], out["static_star_ce"], out["static_yoked_ce"]
    out["delta_ce_vs_static_star"] = (a - b) if (a is not None and b is not None) else None
    out["delta_ce_vs_static_yoked"] = (a - c) if (a is not None and c is not None) else None
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/autoresearch/t2a/evaluation")
    p.add_argument("--scores", default="scores", help="generation-pass directory, relative to --out")
    p.add_argument("--scores-ce", default="scores_ce", help="CE-pass directory, relative to --out")
    p.add_argument("--analysis-name", default="analysis.json")
    p.add_argument("--codecs", default=",".join(CODECS))
    args = p.parse_args()
    out = REPO / args.out

    gen = load_rows(out / args.scores)
    ce = load_rows(out / args.scores_ce)
    print(f"loaded {len(gen)} generation rows, {len(ce)} CE rows")

    index = [json.loads(l) for l in (out / "checkpoint_index.jsonl").read_text().splitlines() if l.strip()]
    codecs = tuple(c for c in args.codecs.split(",") if c)
    expected = {r["checkpoint"] for r in index if r.get("in_scope") and r.get("codec") in codecs}
    coverage = {"expected": len(expected), "generation_scored": len(set(gen) & expected),
                "ce_scored": len(set(ce) & expected),
                "missing_generation": sorted(expected - set(gen)),
                "missing_ce": sorted(expected - set(ce))}

    grid = build_grid(gen, ce)
    analysis = {
        "coverage": coverage,
        "instrument": next(iter(gen.values()))["instrument"] if gen else None,
        "ce_instrument": next(iter(ce.values()))["instrument"] if ce else None,
        "frozen_baseline": frozen_baseline(gen),
        "rouge_truncation_sensitivity": {
            "max_abs_diff_first_line_vs_full": max(
                (abs(r["aggregate"]["rouge_l"] - r["aggregate"]["rouge_l_untruncated"]) for r in gen.values()),
                default=None),
            "n_rows_differing": sum(1 for r in gen.values()
                                    if abs(r["aggregate"]["rouge_l"] - r["aggregate"]["rouge_l_untruncated"]) > 1e-12),
        },
        "codecs": {},
    }
    fb = analysis["frozen_baseline"]
    for codec in codecs:
        if codec in grid:
            a = analyze_codec(codec, grid[codec], frozen_rouge=fb.get(codec, {}).get("rouge_l"))
            a["q4_headline_any_budget"] = headline(a, budget_matched=False)
            a["q4_headline_canonical_budget"] = headline(a, budget_matched=True)
            analysis["codecs"][codec] = a

    (out / args.analysis_name).write_text(json.dumps(analysis, indent=2, default=str) + "\n")
    print_report(analysis)
    print(f"\nwrote {out / args.analysis_name}")


def print_report(a: dict) -> None:
    print("\n" + "=" * 100)
    print(f"coverage: {a['coverage']['generation_scored']}/{a['coverage']['expected']} generation, "
          f"{a['coverage']['ce_scored']}/{a['coverage']['expected']} CE")
    if a["coverage"]["missing_generation"]:
        print(f"  MISSING generation: {a['coverage']['missing_generation']}")
    if a["coverage"]["missing_ce"]:
        print(f"  MISSING ce: {len(a['coverage']['missing_ce'])} rows")
    fb = a["frozen_baseline"]
    print(f"frozen ROUGE-L per codec: "
          f"{ {c: round(d['rouge_l'], 4) for c, d in fb.items() if not c.startswith('_')} }")
    print(f"  max spread across codecs: {fb['_max_spread_across_codecs']:.2e}"
          if fb["_max_spread_across_codecs"] is not None else "")
    rts = a["rouge_truncation_sensitivity"]
    print(f"ROUGE-L first-line vs untruncated: max abs diff {rts['max_abs_diff_first_line_vs_full']}, "
          f"{rts['n_rows_differing']} rows differ")

    for codec, c in a["codecs"].items():
        print("\n" + "=" * 100)
        print(f"### {codec}   (CE-selected: {c['canonical']['ce_selected_config_str']}; "
              f"grid: {c['grid_size']['matched_configs']} matched / {c['grid_size']['static_configs']} static configs)")
        print(f"  ladder: scales {[fmt_scale(s) for s in c['ladder']['scales']]}  "
              f"lrs {[f'{x:g}' for x in c['ladder']['lrs']]}  steps {c['ladder']['steps']}")
        print(f"\n  {'config':40} {'n':>2} {'matched RL':>11} {'static RL':>10} {'m-s(rung)':>10} "
              f"{'matched EM':>10} {'static EM':>9} {'matched CE':>10} {'static CE':>9}")
        for key in sorted(set(c["matched_by_config"]) | set(c["static_by_config"])):
            m = c["matched_by_config"].get(key, {})
            s = c["static_by_config"].get(key, {})
            rung = ((m.get("rouge_l") - s.get("rouge_l"))
                    if m.get("rouge_l") is not None and s.get("rouge_l") is not None else None)
            marks = []
            if key == c["canonical"]["ce_selected_config_str"]:
                marks.append("CE-sel")
            if key == c["q1_hyper_argmax"]["rouge_argmax_any_budget_str"]:
                marks.append("RL-hyper*")
            if key == c["q2_static_optimum"]["static_star_any_budget_str"]:
                marks.append("RL-static*")
            print(f"  {key:40} {m.get('n_seeds') or s.get('n_seeds'):>2} "
                  f"{fmt(m.get('rouge_l')):>11} {fmt(s.get('rouge_l')):>10} {fmt(rung):>10} "
                  f"{fmt(m.get('em')):>10} {fmt(s.get('em')):>9} "
                  f"{fmt(m.get('ce'), 3):>10} {fmt(s.get('ce'), 3):>9}  {' '.join(marks)}")

        sn = c["seed_noise"]
        print(f"\n  seed noise (pooled within-config SD of ROUGE-L): matched "
              f"{fmt(sn['matched']['pooled_sd'])} (dof {sn['matched']['dof']})  static "
              f"{fmt(sn['static']['pooled_sd'])} (dof {sn['static']['dof']})")
        print(f"    converged-only: matched {fmt(sn['matched_converged_only']['pooled_sd'])} "
              f"(dof {sn['matched_converged_only']['dof']})  static "
              f"{fmt(sn['static_converged_only']['pooled_sd'])} (dof {sn['static_converged_only']['dof']})")
        hf = c["helpfulness_floor"]
        print(f"  helpfulness floor (frozen ROUGE-L {fmt(hf['frozen_rouge_l'])}): argmax clears = "
              f"{hf['argmax_clears_floor']}; configs with a seed at/below frozen: {hf['configs_failing_floor']}")
        for label in ("top_matched_at_canonical_budget", "top_static_at_canonical_budget"):
            print(f"  {label}:")
            for t in c[label]:
                print(f"      {t['value']:.4f}  n={t['n_seeds']}  sd={fmt(t['sd'])}  {t['config']}  {t['phases']}")

        q1, q2, q3 = c["q1_hyper_argmax"], c["q2_static_optimum"], c["q3_boundary"]
        print(f"\n  Q1 hyper argmax (ROUGE-L): {q1['rouge_argmax_any_budget_str']}  "
              f"MOVED={q1['moved_any_budget']}   (at canonical budget: "
              f"{q1['rouge_argmax_at_canonical_budget_str']}, MOVED={q1['moved_at_canonical_budget']})")
        print(f"     old-rule CE argmin on this split: {q1['selection_split_ce_argmin_same_rung_control']}")
        print(f"  Q2 static* : {q2['static_star_any_budget_str']}  "
              f"score {fmt((q2['at_static_star'] or {}).get('rouge_l'))} vs "
              f"{fmt((q2['at_hypernetwork_chosen_hparams'] or {}).get('rouge_l'))} at the hypernetwork point"
              f"   (at hyper point = static*: {q2['static_star_is_at_hypernetwork_point']})")
        print(f"  Q3 boundary: {json.dumps({k: v for k, v in (q3 or {}).items() if not k.startswith('swept')})}")
        for label in ("q4_headline_any_budget", "q4_headline_canonical_budget"):
            h = c[label]
            print(f"  {label}: matched RL {fmt(h['matched_rouge_l'])} (n={h['matched_n_seeds']}, "
                  f"sd {fmt(h['matched_rouge_l_sd'])})  static* RL {fmt(h['static_star_rouge_l'])}  "
                  f"delta {fmt(h['delta_rouge_l_vs_static_star'])}  |  "
                  f"vs yoked static {fmt(h['static_yoked_rouge_l'])} -> {fmt(h['delta_rouge_l_vs_static_yoked'])}")

    print("\n" + "=" * 100)
    print("Q4 cross-codec ranking (corrected headline = matched - static*, canonical budget):")
    print(f"  {'codec':10} {'RL m':>8} {'RL s*':>8} {'RL delta':>9} {'EM delta':>9} {'CE delta':>9} "
          f"{'CE(recorded 21-task)':>21}")
    for codec, c in a["codecs"].items():
        h = c["q4_headline_canonical_budget"]
        rec = c["canonical"]["recorded_headline"]["value"]
        print(f"  {codec:10} {fmt(h['matched_rouge_l']):>8} {fmt(h['static_star_rouge_l']):>8} "
              f"{fmt(h['delta_rouge_l_vs_static_star']):>9} {fmt(h['delta_em_vs_static_star']):>9} "
              f"{fmt(h['delta_ce_vs_static_star'], 3):>9} "
              f"{rec:>21.4f}")


if __name__ == "__main__":
    main()
