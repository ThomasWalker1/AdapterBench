"""Summarize a t2p-sft-pilot run under the T2L rigor invariants (LoRA codec).

Reads a pilot results.jsonl and reports, per the four invariants:
  1. behavioral metric + control — matched accuracy vs the mismatched-description control
     (`accuracy_mismatched`), headline = matched - mismatched, per family and overall
  2. scale sweep                 — matched accuracy vs LoRA scale (best-of), when --scales was used
  3. difficulty knob             — per-family frozen headroom vs the adapter's gain-over-control
                                   (eval-task curation: families where frozen leaves headroom)
  4. multi-seed                  — mean±std over seeds where >1 seed is present

Run: .venv/bin/python scripts/t2p_rigor_aggregate.py --results results/t2p_rigor/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

FAM_ORDER = ["arc_easy", "arc_challenge", "hellaswag", "boolq", "gsm8k"]


def mean_std(xs):
    xs = list(xs)
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/t2p_rigor/results.jsonl")
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
    if not rows:
        print(f"no rows in {args.results}")
        return

    def acc(r):
        return r["metrics"].get("accuracy", r["metrics"].get("exact_match"))

    def mm(r):
        return r["metrics"].get("accuracy_mismatched", r["metrics"].get("exact_match_mismatched"))

    frozen = {r["task_id"]: acc(r) for r in rows if r["adapter"] == "frozen_interpreter"}
    lora = [r for r in rows if r["adapter"] != "frozen_interpreter"]
    families = [f for f in FAM_ORDER if any(r["task_id"] == f for r in lora)]
    has_mm = any(mm(r) is not None for r in lora)

    print("=" * 78)
    print(f"T2L rigor summary — {args.results}  ({len(lora)} adapter rows, LoRA codec)")
    print("=" * 78)
    print("\nFROZEN baseline:", {f: round(frozen[f], 3) for f in families if f in frozen})

    print("\n[1+4] MATCHED vs MISMATCHED CONTROL  (mean±std over scales×seeds, per family)")
    print(f"  {'family':14}{'frozen':>8}{'matched':>16}{'mismatched':>14}{'matched-mm':>12}")
    for f in families:
        rs = [r for r in lora if r["task_id"] == f]
        mt, ms_ = mean_std([acc(r) for r in rs])
        mmv, _ = mean_std([mm(r) for r in rs if mm(r) is not None]) if has_mm else (float("nan"), 0)
        print(f"  {f:14}{frozen.get(f, float('nan')):8.3f}{mt:10.3f}±{ms_:.3f}{mmv:12.3f}{mt - mmv:+12.3f}")

    scales = sorted({r["metadata"].get("lora_scale") for r in lora}, key=lambda s: float(s) if s not in (None, "default") else -1)
    if len(scales) > 1:
        print("\n[2] SCALE SWEEP  (matched accuracy, mean±std over seeds×families)")
        for sc in scales:
            xs = [acc(r) for r in lora if r["metadata"].get("lora_scale") == sc]
            m, s = mean_std(xs)
            print(f"  scale={str(sc):>8}  matched acc={m:.3f} ± {s:.3f}  (n={len(xs)})")
        best = max(scales, key=lambda sc: mean_std([acc(r) for r in lora if r["metadata"].get("lora_scale") == sc])[0])
        print(f"  best-of-scale: {best}")

    print("\n[3] DIFFICULTY (per-family frozen headroom vs adapter gain-over-control, best scale)")
    for f in families:
        rs = [r for r in lora if r["task_id"] == f]
        by_scale = defaultdict(list)
        for r in rs:
            by_scale[r["metadata"].get("lora_scale")].append((acc(r), mm(r)))
        best = max(by_scale, key=lambda k: sum(a for a, _ in by_scale[k]) / len(by_scale[k]))
        ma, _ = mean_std([a for a, _ in by_scale[best]])
        mmv, _ = mean_std([m for _, m in by_scale[best] if m is not None]) if has_mm else (float("nan"), 0)
        head = 1 - frozen.get(f, 0)
        print(f"  {f:14} frozen={frozen.get(f, 0):.3f} headroom={head:.3f}  best-scale={best}  matched={ma:.3f} ctrl={mmv:.3f}  gain-over-ctrl={ma - mmv:+.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
