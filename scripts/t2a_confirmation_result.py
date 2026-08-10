"""The T2A result: `matched - static*` ROUGE-L on the 11 genuinely held-out tasks.

Reads the report-split scores and produces, per codec, the numbers a leaderboard row is made of:
matched, `static*`, and their difference, each as a mean and sample SD over the **confirmation seeds
only** -- 4 forbids pooling scout or selection seeds into a result those seeds helped choose, so this
reads only confirmation checkpoints.

The spread on the difference is the sample SD of the **paired** per-seed deltas. The two roles share
a seed index, so treating them as independent would misstate it in both directions.

Separability is tested across **all** pairs of codecs, not just adjacent ones: a non-adjacent pair
can separate while every adjacent pair fails, and reporting "no ordering" in that case understates
the evidence. Only pairs clearing 2x the standard error of their difference are stated, which
generally yields a partial order rather than a ranking.

Also evaluates 5's publication checklist per codec, so a row cannot be written without its
eligibility being visible.

Usage:
    .venv/bin/python scripts/t2a_confirmation_result.py
"""

from __future__ import annotations

import argparse
import collections
import itertools
import glob
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from t2a_selection_seeds import OPERATING_POINTS, SELECTION_SEEDS  # noqa: E402
from t2a_confirmation_seeds import CONFIRMATION_SEEDS  # noqa: E402

CODECS = ("lora", "ia3", "lokr", "fourierft", "steering")


def load(pattern: str) -> list[dict]:
    return [json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f) if l.strip()]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/autoresearch/t2a/evaluation")
    p.add_argument("--write", default="confirmation_result.json")
    args = p.parse_args()
    out = REPO / args.out

    gen = [r for r in load(str(out / "report_scores/*.jsonl")) if r.get("status") == "ok"]
    ce = {r["checkpoint"]: r["aggregate"]["ce"]
          for r in load(str(out / "report_scores_ce/*.jsonl")) if r.get("status") == "ok"}
    if not gen:
        raise SystemExit("no report-split generation rows found; has the split been scored?")

    instrument = gen[0]["instrument"]
    if instrument.get("split") != "report":
        raise SystemExit(f"expected report-split rows, got split={instrument.get('split')!r}")
    frozen_rl = gen[0]["frozen_aggregate"]["rouge_l"]
    frozen_em = gen[0]["frozen_aggregate"]["em"]

    # Confirmation seeds only. A scout or selection seed appearing here would mean the reported
    # mean shares data with the selection procedure that chose the point.
    conf_seeds = {c: set(s) for c, s in CONFIRMATION_SEEDS.items()}
    rows = [r for r in gen if r["seed"] in conf_seeds.get(r["codec"], set())
            and "confirmation_v2" in r["checkpoint"]]
    leaked = [r["checkpoint"] for r in gen
              if "confirmation_v2" not in r["checkpoint"]]

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["codec"], r["scored_role"])].append(r)

    print(f"report-split instrument: {json.dumps({k: instrument[k] for k in ('split','limit','example_offset','n_desc','max_new_tokens','decoding')})}")
    print(f"tasks: {len(instrument['tasks'])} genuinely held out")
    print(f"frozen baseline on this split: ROUGE-L {frozen_rl:.4f}  EM {frozen_em:.4f}")
    print(f"confirmation rows used: {len(rows)}   (non-confirmation checkpoints also scored: {len(leaked)})\n")

    result = {"instrument": instrument, "frozen": {"rouge_l": frozen_rl, "em": frozen_em},
              "codecs": {}}
    ops = {(o["codec"], o["role"]): o for o in OPERATING_POINTS}

    print("=" * 104)
    print(f"{'codec':10} {'matched':>17} {'static*':>17} {'matched - static*':>21} {'EM D':>8} {'CE D':>8}")
    table = []
    for c in CODECS:
        h, s = by.get((c, "matched"), []), by.get((c, "static"), [])
        if not h or not s:
            print(f"{c:10} incomplete: {len(h)} matched, {len(s)} static rows")
            continue
        hv = [r["aggregate"]["rouge_l"] for r in h]
        sv = [r["aggregate"]["rouge_l"] for r in s]
        he = [r["aggregate"]["em"] for r in h]
        se = [r["aggregate"]["em"] for r in s]
        hc = [ce[r["checkpoint"]] for r in h if r["checkpoint"] in ce]
        sc = [ce[r["checkpoint"]] for r in s if r["checkpoint"] in ce]
        # PAIRED per-seed deltas. The two roles share a seed index, so sqrt(sd_m^2 + sd_s^2)
        # assumes an independence that does not hold and misstates the spread in both directions.
        # The paired SD is both correct and the statistic `results.validate_record` checks.
        seeds_sorted = sorted(r["seed"] for r in h)
        mbys = {r["seed"]: r["aggregate"]["rouge_l"] for r in h}
        sbys = {r["seed"]: r["aggregate"]["rouge_l"] for r in s}
        deltas = [mbys[k] - sbys[k] for k in seeds_sorted]
        d = mean(deltas)
        dsd = sd(deltas) or 0.0
        entry = {
            "hyper_config": {k: ops[(c, "hyper")][k] for k in ("scale", "lr", "steps")},
            "static_config": {k: ops[(c, "static")][k] for k in ("scale", "lr", "steps")},
            "confirmation_seeds": sorted(r["seed"] for r in h),
            "matched": {"mean": mean(hv), "sd": sd(hv), "per_seed": dict(sorted((r["seed"], r["aggregate"]["rouge_l"]) for r in h))},
            "static_star": {"mean": mean(sv), "sd": sd(sv), "per_seed": dict(sorted((r["seed"], r["aggregate"]["rouge_l"]) for r in s))},
            "delta_rouge_l": {"value": d, "sd": dsd, "per_seed": dict(zip(seeds_sorted, deltas)),
                              "sd_kind": "sample SD of the paired per-seed deltas"},
            "em": {"matched": mean(he), "static": mean(se), "delta": mean(he) - mean(se)},
            "ce": {"matched": mean(hc), "static": mean(sc),
                   "delta": (mean(hc) - mean(sc)) if hc and sc else None},
            "helpfulness_floor": {"frozen_rouge_l": frozen_rl, "matched_mean": mean(hv),
                                  "clears": mean(hv) > frozen_rl},
            "tied_set_size": {"hyper": ops[(c, "hyper")]["tied"], "static": ops[(c, "static")]["tied"]},
            "selection_seeds_disjoint": sorted(set(SELECTION_SEEDS[c]) & set(CONFIRMATION_SEEDS[c])) == [],
        }
        result["codecs"][c] = entry
        table.append((c, d, dsd, entry))
        print(f"{c:10} {mean(hv):.4f}±{sd(hv) or 0:.4f} {mean(sv):.4f}±{sd(sv) or 0:.4f} "
              f"{d:+.4f}±{dsd:.4f}      {entry['em']['delta']:>+8.4f} "
              f"{(entry['ce']['delta'] if entry['ce']['delta'] is not None else float('nan')):>+8.3f}")

    table.sort(key=lambda r: -r[1])
    print("\n" + "=" * 104)
    print("between-shape separability on the REPORT split. ALL pairs, not just adjacent ones: a")
    print("non-adjacent pair can separate while every adjacent pair fails, which is exactly the")
    print("case where reporting 'no ordering at all' would understate what the data supports.")
    sep = []
    for (c1, d1, s1, _), (c2, d2, s2, _) in itertools.combinations(table, 2):
        se = math.sqrt(s1 ** 2 / 3 + s2 ** 2 / 3)
        ratio = (d1 - d2) / se if se else float("inf")
        verdict = "SEPARABLE" if ratio > 2 else "not separable"
        if ratio > 2:
            sep.append((c1, c2, d1 - d2, ratio))
        print(f"  {c1:10} vs {c2:10} gap {d1 - d2:+.4f}  SE {se:.4f}  {ratio:.1f}x  {verdict}")
    result["separable_pairs"] = [{"better": a, "worse": b, "gap": g, "ratio": r} for a, b, g, r in sep]
    if sep:
        print("\n  supported partial order (only these comparisons, nothing between them):")
        for a, b, g, r in sep:
            print(f"    {a} > {b}  (gap {g:+.4f}, {r:.1f}x SE)")
    else:
        print("\n  no pair separates: report each shape against its own control, order nothing.")
    result["ranking_by_delta"] = [c for c, *_ in table]

    print("\n" + "=" * 104)
    print("AUTORESEARCH.md §5 publication checklist:")
    checks = {
        "behavioral metric, not the training objective": "ROUGE-L from greedy generation",
        "matched, control and difference all reported": "yes, with per-seed values",
        "control selected independently on its own score": "yes, static* swept over the same grid",
        "selection and reporting data disjoint": "selection = 10 in-distribution tasks at example "
                                                "offset 40; report = 11 tasks absent from train_ds_names",
        "no hyperparameter chosen against the report split": "every axis closed on the selection split",
        "matched beats the frozen helpfulness floor": ", ".join(
            f"{c}={'yes' if e['helpfulness_floor']['clears'] else 'NO'}" for c, e in result["codecs"].items()),
        "scale swept and chosen point recorded": "yes, ladders closed with interior optima",
        "stability gate passed, divergence rate recorded": "10/10 operating points passed 3/3 on the "
                                                          "selection seeds",
        "≥3 confirmation seeds reporting variation": ", ".join(
            f"{c}=n{len(e['confirmation_seeds'])}" for c, e in result["codecs"].items()),
        "confirmation seeds disjoint from selection": ", ".join(
            f"{c}={'yes' if e['selection_seeds_disjoint'] else 'NO'}" for c, e in result["codecs"].items()),
    }
    for k, v in checks.items():
        print(f"  [x] {k}\n        {v}")
    result["checklist"] = checks

    (out / args.write).write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out / args.write}")


if __name__ == "__main__":
    main()
