"""Derive the four-invariant summary for the i2p-hypernoise pipeline (LoRA codec).

Reads every results.jsonl under the pipeline output dir and prints:
  1. reward-swap control  — ImageReward gain of the imagereward-adapter (matched) vs the
                            red-adapter (control), mean±std over seeds; headline = matched-control
  2. scale sweep          — ImageReward gain vs LoRA scale (best-of is invariant #2's report)
  3. difficulty knob      — ImageReward gain + CLIP-T drop vs reg_weight (the fidelity<->reward curve)
  4. multi-seed           — the per-seed ImageReward gains behind (1)

Run: .venv/bin/python scripts/i2p_hypernoise_aggregate.py --out results/i2p_hypernoise
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_cells(out_root: Path) -> list[dict]:
    cells = []
    for path in sorted(out_root.glob("*/results.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                cells.append(json.loads(line))
    return cells


def mean_std(xs):
    xs = list(xs)
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return m, sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/i2p_hypernoise")
    args = ap.parse_args()
    cells = load_cells(Path(args.out))
    if not cells:
        print(f"no results under {args.out}")
        return

    def meta(c, k):
        return c["metadata"].get(k)

    def irgain(c):
        return c["metrics"]["reward_imagereward_gain"]

    center = [c for c in cells if meta(c, "lora_scale") == "default" and float(meta(c, "reg_weight")) == 0.5]
    matched = [irgain(c) for c in center if meta(c, "reward_target") == "imagereward"]
    control = [irgain(c) for c in center if meta(c, "reward_target") == "red"]

    print("=" * 74)
    print("i2p-hypernoise — four-invariant summary (LoRA codec, ImageReward headline)")
    print("=" * 74)

    print("\n[1+4] REWARD-SWAP CONTROL + MULTI-SEED  (center: scale=default, reg=0.5)")
    mm, ms = mean_std(matched)
    cm, cs = mean_std(control)
    print(f"  matched  (imagereward-adapter, IR gain): {mm:+.4f} ± {ms:.4f}   n={len(matched)}  per-seed={[round(x,4) for x in matched]}")
    print(f"  control  (red-adapter,         IR gain): {cm:+.4f} ± {cs:.4f}   n={len(control)}  per-seed={[round(x,4) for x in control]}")
    print(f"  HEADLINE  matched - control            : {mm - cm:+.4f}")
    if center:
        cl = [c for c in center if meta(c, "reward_target") == "imagereward"]
        if cl:
            cd = mean_std([c["metrics"]["clipt_drop"] for c in cl])[0]
            print(f"  fidelity  (CLIP-T drop, matched)       : {cd:+.4f}  (frozen CLIP-T ~ {mean_std([c['metrics']['clipt_frozen'] for c in cl])[0]:.3f})")

    print("\n[2] SCALE SWEEP  (imagereward, reg=0.5, seed=777)")
    scale_cells = [c for c in cells if meta(c, "reward_target") == "imagereward" and float(meta(c, "reg_weight")) == 0.5 and meta(c, "seed") == 777]
    def scale_key(c):
        s = meta(c, "lora_scale")
        return -1.0 if s == "default" else float(s)
    for c in sorted(scale_cells, key=scale_key):
        print(f"  scale={str(meta(c,'lora_scale')):>7}  IR gain={irgain(c):+.4f}  CLIP-T drop={c['metrics']['clipt_drop']:+.4f}")
    if scale_cells:
        best = max(scale_cells, key=irgain)
        print(f"  best-of-scale: scale={meta(best,'lora_scale')}  IR gain={irgain(best):+.4f}")

    print("\n[3] DIFFICULTY KNOB — reg_weight  (imagereward, scale=default, seed=777)")
    reg_cells = [c for c in cells if meta(c, "reward_target") == "imagereward" and meta(c, "lora_scale") == "default" and meta(c, "seed") == 777]
    for c in sorted(reg_cells, key=lambda c: float(meta(c, "reg_weight"))):
        m = c["metrics"]
        print(f"  reg={float(meta(c,'reg_weight')):>5g}  IR gain={irgain(c):+.4f}  CLIP-T adapter={m['clipt_adapter']:.3f} (drop {m['clipt_drop']:+.4f})  redness gain={m['reward_red_gain']:+.4f}")

    print("=" * 74)


if __name__ == "__main__":
    main()
