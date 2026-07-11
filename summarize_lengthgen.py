#!/usr/bin/env python3
"""Compact status/results summary for the length-generalization grid.

Scans results/d2p_lengthgen_<adapter>_s<seed>.log, pulls the last per-eval line, and
prints one row per run: latest step, held-out accuracy at each eval length bin, and the
max context-swap control seen at the final eval (must stay ~0 for a finding to count).
Used both for live oversight (via a Monitor poll loop) and final reporting.
"""
import ast
import glob
import os
import re
import sys

BINS = [256, 512, 1024, 2048, 4096, 8192]
STEP_RE = re.compile(r"\[step (\d+)\] loss=([0-9.]+) (\{.*\})")


def last_eval(path):
    step = loss = metrics = None
    with open(path) as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if m:
                step, loss, metrics = int(m.group(1)), float(m.group(2)), ast.literal_eval(m.group(3))
    return step, loss, metrics


def alive(out_dir):
    # crude: a run is "done" if it wrote the final results.jsonl marker line in its log
    return None


def main():
    rows = []
    for log in sorted(glob.glob("results/d2p_lengthgen_*.log")):
        name = os.path.basename(log)[len("d2p_lengthgen_"):-len(".log")]
        try:
            step, loss, metrics = last_eval(log)
        except Exception as e:  # noqa: BLE001
            rows.append((name, f"<parse error: {e}>"))
            continue
        if metrics is None:
            rows.append((name, "<no eval yet>"))
            continue
        accs = [metrics.get(f"niah_{b}", {}).get("accuracy") for b in BINS]
        swaps = [metrics.get(f"niah_{b}", {}).get("accuracy_ctxswap") or 0.0 for b in BINS]
        acc_str = " ".join(f"{a:.2f}" if a is not None else "  - " for a in accs)
        rows.append((name, f"step={step:<5} loss={loss:.3f} ctxswap_max={max(swaps):.2f}  acc[{'/'.join(map(str,BINS))}]= {acc_str}"))
    width = max((len(n) for n, _ in rows), default=10)
    for name, info in rows:
        print(f"{name:<{width}}  {info}")


if __name__ == "__main__":
    sys.exit(main())
