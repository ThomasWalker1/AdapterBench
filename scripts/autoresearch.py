"""Per-codec autoresearch driver — the free-HP search that produces a codec's leaderboard numbers.

Fix a codec's shape, search its FREE optimization HPs over ≥3 seeds, select best-of on
`matched − control` (never loss), and report the codec at its own best config with the full tuned-HP
table and search budget. Wraps the settings' existing restart-safe runners; see
`adapterbench.autoresearch` for the driver and `adapterbench.autoresearch_settings` for the specs.

Read-only (default) — select over already-computed cells (validate the loop with no new compute):
  .venv/bin/python scripts/autoresearch.py --setting i2p_hypernoise \
      --results results/i2p_hypernoise_v2 --search "scale=2,3,4,6,8,16,32" \
      --fixed "reg_weight=0.25" --n-seeds 3

Launch — produce the missing (config × seed) cells first, then select:
  .venv/bin/python scripts/autoresearch.py --setting i2p_hypernoise --launch \
      --results results/autoresearch_i2p_lora --search "scale=2,4,8" --fixed "reg_weight=0.25" \
      --n-seeds 3 --gpus 1,2,3 --steps 3000
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

from adapterbench.autoresearch import load_cells, search_over_cells
from adapterbench.autoresearch_settings import REGISTRY


def _parse_assignments(spec: str | None, numeric_lists: bool) -> dict:
    """Parse 'a=1,2,3;b=0.25' into {'a':[1.0,2.0,3.0],'b':[0.25]} (numeric_lists) or {'a':1.0,...}."""
    out: dict = {}
    if not spec:
        return out
    for part in spec.replace(";", " ").split():
        key, _, raw = part.partition("=")
        vals = [float(v) if _is_num(v) else v for v in raw.split(",")]
        out[key] = vals if numeric_lists else vals[0]
    return out


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _launch(spec, out_root: Path, search_dims: dict, fixed_free: dict, seeds, codec, gpus, profile_overrides):
    """Concurrent launch: one cell per (config × seed), fanned across GPUs, restart-safe (skip a
    cell whose results.jsonl exists). Mirrors scripts/i2p_hypernoise_pipeline.py's proven pool.
    `profile_overrides` (the selected run profile's fixed launch params — steps/eval cadence/breadth)
    are merged into every cell so the whole sweep runs at one profile."""
    from adapterbench.autoresearch import _cell_dirname

    jobs = []
    for combo in itertools.product(*search_dims.values()):
        config = dict(zip(search_dims, combo))
        config.update(fixed_free)
        for k, v in (profile_overrides or {}).items():
            config.setdefault(k, v)
        for seed in seeds:
            jobs.append((config, int(seed)))

    pending, running, free = list(jobs), [], list(gpus)
    print(f"[autoresearch/launch] {len(jobs)} cells over {len(gpus)} GPUs -> {out_root}", flush=True)
    while pending or running:
        while pending and free:
            config, seed = pending.pop(0)
            out_dir = out_root / _cell_dirname(spec, codec, config, seed)
            if (out_dir / "results.jsonl").exists():
                print(f"[skip] {out_dir.name}", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            gpu = free.pop(0)
            argv = spec.launch_argv(codec, config, seed, out_dir) + ["--device", f"cuda:{gpu}"]
            logf = open(out_dir / "run.log", "w")
            proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT)
            print(f"[launch gpu{gpu}] {out_dir.name} (pid {proc.pid})", flush=True)
            running.append((proc, gpu, logf, out_dir))
        time.sleep(10)
        still = []
        for proc, gpu, logf, out_dir in running:
            if proc.poll() is None:
                still.append((proc, gpu, logf, out_dir))
            else:
                logf.close()
                print(f"[done gpu{gpu}] {out_dir.name} rc={proc.returncode}", flush=True)
                free.append(gpu)
        running = still


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--results", required=True, help="dir of result cells (read) / to write into (launch)")
    ap.add_argument("--codec", default="lora")
    ap.add_argument("--search", required=True, help="free HPs to sweep, e.g. 'scale=2,3,4;reg_weight=0.25'")
    ap.add_argument("--fixed", default="", help="free HPs pinned (not searched), e.g. 'reg_weight=0.25'")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--min-seeds", type=int, default=None,
                    help="guardrail #2: a config needs >= this many seeds to be eligible as best "
                         "(default = --n-seeds)")
    ap.add_argument("--launch", action="store_true", help="produce missing cells via the setting's CLI first")
    ap.add_argument("--gpus", default="0", help="launch mode: comma-separated GPU ids")
    ap.add_argument("--seeds", default="777,778,779", help="launch mode: seeds to run")
    ap.add_argument("--profile", default="proxy", help="launch mode: run profile (proxy|full) from the SettingSpec")
    ap.add_argument("--steps", type=int, default=None, help="launch mode: override steps per cell (else the profile's)")
    ap.add_argument("--out-json", default=None, help="write the SearchReport dict here")
    args = ap.parse_args()

    spec = REGISTRY[args.setting]
    search_dims = _parse_assignments(args.search, numeric_lists=True)
    fixed_free = _parse_assignments(args.fixed, numeric_lists=False)
    out_root = Path(args.results)

    # Fail fast on a partition violation BEFORE any compute.
    spec.validate_search_space(search_dims, fixed_free)

    if args.launch:
        seeds = [int(s) for s in args.seeds.split(",")][: args.n_seeds]
        overrides = dict((spec.profiles or {}).get(args.profile, {}))
        if args.steps is not None:
            overrides["steps"] = args.steps
        print(f"[autoresearch] profile={args.profile} overrides={overrides}")
        _launch(spec, out_root, search_dims, fixed_free, seeds, args.codec,
                [g.strip() for g in args.gpus.split(",")], overrides)

    cells = load_cells(out_root)
    report = search_over_cells(spec, cells, search_dims=search_dims, fixed_free=fixed_free,
                               n_seeds=args.n_seeds, min_seeds=args.min_seeds, codec=args.codec)
    print(report.render())
    out_json = Path(args.out_json) if args.out_json else out_root / f"autoresearch_{spec.name}_{args.codec}.json"
    out_json.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
