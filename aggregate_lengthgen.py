#!/usr/bin/env python3
"""Aggregate the length-generalization grid into the per-codec benchmark vector.

Reads every results/d2p_lengthgen_<adapter>_s<seed>.log, parses all per-eval records, and
reports per codec (across seeds):

  - Final length curve: held-out accuracy at each eval bin, per seed + mean/best.
  - Length-crossover: the largest eval length at which final accuracy >= 0.5 (the
    length-extrapolation scalar; training length is 256, so this is the extrapolation reach).
  - Transition step: first step where niah_256 accuracy >= 0.5 (sample efficiency /
    trainability); None if the run never transitioned within its budget.
  - Control: max context-swap accuracy at the final eval (must be ~0 for a finding to count).

Works on in-progress or completed runs (log-based). Headline per PROJECT_PLAN invariant #1
is matched - control, and only counts when the control is near chance.
"""
import ast
import glob
import math
import os
import re

BINS = [256, 512, 1024, 2048, 4096, 8192]
TRAIN_LEN = 256  # longest training context (train on 128,256)
STEP_RE = re.compile(r"\[step (\d+)\] loss=([0-9.]+) (\{.*\})")


def parse_log(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if m:
                recs.append((int(m.group(1)), float(m.group(2)), ast.literal_eval(m.group(3))))
    done = "wrote results/" in open(path).read()
    return recs, done


def acc(metrics, b):
    return metrics.get(f"niah_{b}", {}).get("accuracy")


def swap(metrics, b):
    return metrics.get(f"niah_{b}", {}).get("accuracy_ctxswap") or 0.0


def crossover_len(final_metrics):
    hits = [b for b in BINS if (acc(final_metrics, b) or 0.0) >= 0.5]
    return max(hits) if hits else 0


def transition_step(recs):
    for step, _loss, m in recs:
        if (acc(m, 256) or 0.0) >= 0.5:
            return step
    return None


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (None, None)
    mu = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return (mu, sd)


def main():
    runs = {}  # adapter -> {seed -> (recs, done)}
    for log in sorted(glob.glob("results/d2p_lengthgen_*.log")):
        name = os.path.basename(log)[len("d2p_lengthgen_"):-len(".log")]
        adapter, _, seed = name.rpartition("_s")
        recs, done = parse_log(log)
        if recs:
            runs.setdefault(adapter, {})[seed] = (recs, done)

    for adapter in sorted(runs):
        seeds = runs[adapter]
        print(f"\n### {adapter}  ({len(seeds)} seed(s))")
        header = "seed   " + "  ".join(f"{b:>5}" for b in BINS) + "   xover  transit  ctxswap  status"
        print(header)
        finals_by_bin = {b: [] for b in BINS}
        for seed in sorted(seeds):
            recs, done = seeds[seed]
            step, _loss, fm = recs[-1]
            row = [f"{(acc(fm,b) if acc(fm,b) is not None else float('nan')):5.2f}" for b in BINS]
            for b in BINS:
                finals_by_bin[b].append(acc(fm, b))
            xo = crossover_len(fm)
            tr = transition_step(recs)
            sw = max(swap(fm, b) for b in BINS)
            status = "done" if done else f"@{step}"
            print(f"s{seed:<5}" + "  ".join(row) + f"   {xo:>5}  {str(tr):>7}  {sw:6.2f}   {status}")
        # aggregate row
        agg = []
        for b in BINS:
            mu, sd = mean_std(finals_by_bin[b])
            agg.append(f"{mu:4.2f}±{sd:4.2f}" if mu is not None else "   -   ")
        best_xo = max((crossover_len(seeds[s][0][-1][2]) for s in seeds), default=0)
        print("mean   " + "  ".join(f"{a:>5}" for a in agg))
        print(f"       best length-crossover across seeds: {best_xo} (train len {TRAIN_LEN}, "
              f"extrapolation x{best_xo // TRAIN_LEN if best_xo else 0})")


if __name__ == "__main__":
    main()
