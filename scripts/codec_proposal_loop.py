"""The codec proposal loop — propose → implement → gate → merge → evaluate → report — scaffolded
and dry-run on the LoRA baseline (PROJECT_PLAN § "The proposal (autoresearch) loop").

The loop needs NO new codec to be real: LoRA (optionally a rank/scale variant as a stand-in second
entry) exercises every path. This script wires the pieces built for #3 (autoresearch evaluation)
and #4 (merge gate + leaderboard) into the end-to-end flow, so the mechanism is proven before any
genuinely new shape arrives. Steps 1–2 (propose a shape + hypothesis, implement the subclass +
manifest) are the human/agent-authored inputs; steps 3–6 are automated and run here:

  3. GATE     — correctness only (adapterbench.merge_gate); red → iterate, never merge on results.
  4. MERGE    — on green, the codec joins `main` (winner or loser). Dry-run: reported, not executed.
  5. EVALUATE — the per-codec autoresearch loop (free-HP search, best-of on matched−control),
                reused read-only here over an existing results dir so the dry-run needs no new GPU.
  6. REPORT   — append a provenance-stamped leaderboard record; regenerate the board (losers kept).

Dry-run on LoRA over the existing I2P sweep:
  .venv/bin/python scripts/codec_proposal_loop.py --adapter lora_r8_t2l --setting i2p_hypernoise \
      --results results/i2p_hypernoise_v2 --search "scale=2,3,4,6,8,16,32" --fixed "reg_weight=0.25" \
      --hypothesis "LoRA baseline: should set the reference bar on ImageReward reach" --no-gpu-smoke
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from adapterbench.autoresearch import load_cells, search_over_cells
from adapterbench.autoresearch_settings import REGISTRY
from adapterbench.catalog import load_catalog
from adapterbench.leaderboard import append_record, record_from_autoresearch, render_leaderboard
from adapterbench.merge_gate import generated_output_size, run_gate

LEADERBOARD_STORE = "results/leaderboard"


def _runner(argv):
    p = subprocess.run(argv, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _current_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="manifest name of the (guinea-pig) codec")
    ap.add_argument("--setting", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--results", required=True, help="results dir to evaluate over (read-only)")
    ap.add_argument("--search", required=True, help="free-HP search space, e.g. 'scale=2,3,4'")
    ap.add_argument("--fixed", default="", help="pinned free HPs, e.g. 'reg_weight=0.25'")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--hypothesis", default="(none stated)", help="step-1 hypothesis: which axis should move")
    ap.add_argument("--changed-files", default="src/adapterbench/t2p/codecs.py,configs/adapters/lora_t2l.yaml",
                    help="the PR's changed files (dry-run defaults to a codec-only diff)")
    ap.add_argument("--no-gpu-smoke", action="store_true")
    ap.add_argument("--catalog-root", default="configs", help="catalog root (contains adapters/)")
    args = ap.parse_args()

    spec = REGISTRY[args.setting]
    _, adapters = load_catalog(args.catalog_root)
    manifest = adapters[args.adapter]

    print("#" * 74)
    print(f"# CODEC PROPOSAL LOOP (dry-run) — {args.adapter} on {args.setting}")
    print(f"# [1] PROPOSE  hypothesis: {args.hypothesis}")
    print(f"# [2] IMPLEMENT (assumed authored: subclass + make_codec entry + manifest)")
    print("#" * 74)

    # [3] GATE — correctness only
    print("\n[3] GATE")
    report = run_gate(manifest, args.changed_files.split(","), runner=_runner,
                      run_gpu_smoke=not args.no_gpu_smoke)
    print(report.render())
    if not report.passed:
        print("\n[3] GATE FAILED → iterate on the branch; NOT merging (never merges on results).")
        sys.exit(1)

    # [4] MERGE (dry-run: reported, not executed)
    print("\n[4] MERGE — gate green → codec would merge to `main` (winner or loser). [dry-run: not executed]")

    # [5] EVALUATE — per-codec autoresearch (read-only over existing cells)
    print("\n[5] EVALUATE — per-codec autoresearch (best-of on matched−control, multi-seed)")

    def _parse(spec_str, lists):
        out = {}
        for part in (spec_str or "").replace(";", " ").split():
            k, _, raw = part.partition("=")
            vals = [float(v) if _isnum(v) else v for v in raw.split(",")]
            out[k] = vals if lists else vals[0]
        return out

    def _isnum(s):
        try:
            float(s); return True
        except ValueError:
            return False

    cells = load_cells(args.results)
    ar = search_over_cells(spec, cells, search_dims=_parse(args.search, True),
                           fixed_free=_parse(args.fixed, False), n_seeds=args.n_seeds,
                           codec=manifest.family)
    print(ar.render())

    # [6] REPORT — provenance-stamped leaderboard record + regenerated board
    print("\n[6] REPORT — append provenance-stamped record; regenerate the board")
    if ar.best is None:
        print("  no eligible best config (under-seeded / empty) — nothing to record.")
        sys.exit(0)
    budget = manifest.parameter_budget
    gen_size = (generated_output_size(manifest.family, manifest.hyperparameters, budget.reference_dim)
                if budget else None)
    record = record_from_autoresearch(ar.to_dict(), main_git_sha=_current_sha(),
                                      trial_id=f"{args.setting}--{manifest.name}",
                                      data_split="eval", generated_output_size=gen_size)
    append_record(record, LEADERBOARD_STORE)
    outcome = "confirmed" if ar.best.objective_mean > 0 else "refuted"
    print(f"  hypothesis: {args.hypothesis}\n  outcome vs hypothesis: {outcome} "
          f"(best matched−control = {ar.best.objective_mean:+.4f})")
    print(f"\n{render_leaderboard(LEADERBOARD_STORE)}")


if __name__ == "__main__":
    main()
