"""Derive the four-invariant summary for an i2p-hypernoise pipeline dir (LoRA codec).

Reads every results.jsonl under the pipeline output dir and auto-detects the sweeps present
(it does not assume a fixed center config), printing:
  1. reward-swap control  — at each (scale,reg) where both `imagereward` and `red` adapters
                            exist: ImageReward gain of the imagereward-adapter (matched) vs the
                            red-adapter (control), mean±std over seeds; headline = matched-control
  2. scale sweep          — for each reg with >=2 distinct scales: ImageReward gain vs LoRA scale
  3. difficulty knob      — for each scale with >=2 distinct regs: ImageReward gain + CLIP-T drop
                            vs reg_weight, mean±std over seeds (the fidelity<->reward curve)
  4. multi-seed           — the per-seed spread is shown inline via ±std / n

Run: .venv/bin/python scripts/i2p_hypernoise_aggregate.py --out results/i2p_hypernoise_v2
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
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


def scale_val(s):
    return -1.0 if s == "default" else float(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/i2p_hypernoise_v2")
    args = ap.parse_args()
    cells = load_cells(Path(args.out))
    if not cells:
        print(f"no results under {args.out}")
        return

    def m(c, k):
        return c["metadata"].get(k)

    def irgain(c):
        return c["metrics"]["reward_imagereward_gain"]

    # index by (reward, scale, reg) -> list of cells (one per seed)
    by_key = defaultdict(list)
    for c in cells:
        by_key[(m(c, "reward_target"), m(c, "lora_scale"), float(m(c, "reg_weight")))].append(c)

    print("=" * 78)
    print(f"i2p-hypernoise four-invariant summary — {args.out}  ({len(cells)} cells, LoRA codec)")
    print("=" * 78)

    # [1] reward-swap control: any (scale,reg) with both imagereward and red present
    print("\n[1+4] REWARD-SWAP CONTROL + MULTI-SEED")
    found_swap = False
    scales_regs = sorted({(s, r) for (rw, s, r) in by_key}, key=lambda x: (scale_val(x[0]), x[1]))
    for s, r in scales_regs:
        matched = [irgain(c) for c in by_key.get(("imagereward", s, r), [])]
        control = [irgain(c) for c in by_key.get(("red", s, r), [])]
        if matched and control:
            found_swap = True
            mm, ms = mean_std(matched); cm, cs = mean_std(control)
            cd = mean_std([c["metrics"]["clipt_drop"] for c in by_key[("imagereward", s, r)]])[0]
            print(f"  scale={s} reg={r:g}:")
            print(f"    matched (imagereward)  IR gain = {mm:+.4f} ± {ms:.4f}  n={len(matched)}  {[round(x,3) for x in matched]}")
            print(f"    control (red)          IR gain = {cm:+.4f} ± {cs:.4f}  n={len(control)}")
            print(f"    HEADLINE matched-control = {mm - cm:+.4f}   | matched CLIP-T drop = {cd:+.4f}")
    if not found_swap:
        print("  (no (scale,reg) has both imagereward and red adapters)")

    # [2] scale sweeps: reg values that have >=2 distinct scales (imagereward, per seed)
    print("\n[2] SCALE SWEEP (imagereward)")
    ir_cells = [c for c in cells if m(c, "reward_target") == "imagereward"]
    by_reg_seed = defaultdict(dict)  # (reg,seed) -> {scale: gain}
    for c in ir_cells:
        by_reg_seed[(float(m(c, "reg_weight")), m(c, "seed"))][m(c, "lora_scale")] = irgain(c)
    shown = False
    for (reg, seed), scales in sorted(by_reg_seed.items()):
        if len(scales) >= 2:
            shown = True
            print(f"  reg={reg:g} seed={seed}:")
            for sc in sorted(scales, key=scale_val):
                print(f"    scale={str(sc):>7}  IR gain={scales[sc]:+.4f}")
            best = max(scales, key=lambda k: scales[k])
            print(f"    best-of-scale: {best} ({scales[best]:+.4f})")
    if not shown:
        print("  (no reg has >=2 scales)")

    # [3] difficulty knob: scales that have >=2 distinct regs (imagereward), aggregate seeds
    print("\n[3] DIFFICULTY KNOB — reg_weight (imagereward, mean±std over seeds)")
    by_scale = defaultdict(lambda: defaultdict(list))  # scale -> reg -> [cells]
    for c in ir_cells:
        by_scale[m(c, "lora_scale")][float(m(c, "reg_weight"))].append(c)
    shown = False
    for scale, regs in sorted(by_scale.items(), key=lambda kv: scale_val(kv[0])):
        if len(regs) >= 2:
            shown = True
            print(f"  scale={scale}:")
            for reg in sorted(regs):
                cs = regs[reg]
                gm, gs = mean_std([irgain(c) for c in cs])
                dd = mean_std([c["metrics"]["clipt_drop"] for c in cs])[0]
                rr = mean_std([c["metrics"]["reward_red_gain"] for c in cs])[0]
                print(f"    reg={reg:>5g}  IR gain={gm:+.4f} ± {gs:.4f} (n={len(cs)})  CLIP-T drop={dd:+.4f}  redness gain={rr:+.4f}")
    if not shown:
        print("  (no scale has >=2 regs)")
    print("=" * 78)


if __name__ == "__main__":
    main()
