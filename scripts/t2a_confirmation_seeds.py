"""Run the confirmation seeds that produce the reported T2A numbers (AUTORESEARCH.md 4).

Three fresh seeds per operating point, at exactly the configurations the selection seeds verified --
imported from `t2a_selection_seeds.OPERATING_POINTS` rather than restated, so they cannot drift.

The seeds are disjoint from the scout and selection seeds, and the script asserts that rather than
trusting it. The reason is concrete: the separability analysis uses seed SDs measured on the
selection seeds, so reporting means from those same seeds would make the test statistic and the
estimate share data.

**These checkpoints are then scored on the 11 genuinely held-out report tasks -- the one and only
time that split is measured.** Scoring is deliberately not chained here: training completing is the
moment to confirm every seed converged before consuming a one-shot resource. Run
`scripts/t2a_score_report_split.sh`, which verifies the set and then scores it.

Usage:
    .venv/bin/python scripts/t2a_confirmation_seeds.py --dry-run
    .venv/bin/python scripts/t2a_confirmation_seeds.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from t2a_sweep_axes import ROOT, run_queue, sha256_of  # noqa: E402
from t2a_selection_seeds import OPERATING_POINTS, SELECTION_SEEDS  # noqa: E402

# Confirmation seeds. Disjoint from every seed used anywhere: the scouts (1701/1702/2702/4702/5001),
# the historical selection and confirmation seeds (2801-2803, 5101-5103, 1801-1803, 1901-1903,
# 2811-2813, 5111-5113, 2704-2707), and Phase 2's selection seeds (1721-1723, 1731-1733, 2721-2723,
# 5021-5023, 4721-4723). Asserted below rather than trusted.
CONFIRMATION_SEEDS = {
    "lora": (1741, 1742, 1743),
    "ia3": (1751, 1752, 1753),
    "lokr": (2741, 2742, 2743),
    "fourierft": (5041, 5042, 5043),
    "steering": (4741, 4742, 4743),
}

# Every seed consumed by an earlier phase, for the disjointness assertion.
HISTORICAL_SEEDS = {
    1701, 1702, 2702, 4702, 5001,                    # scouts
    2801, 2802, 2803, 5101, 5102, 5103,              # historical selection seeds
    1801, 1802, 1803, 1901, 1902, 1903,              # historical confirmations
    2811, 2812, 2813, 5111, 5112, 5113,
    2704, 2705, 2706, 2707,                          # lokr naive-point runs
}


def assert_disjoint() -> None:
    used = set(HISTORICAL_SEEDS)
    for seeds in SELECTION_SEEDS.values():
        used |= set(seeds)
    mine = {s for seeds in CONFIRMATION_SEEDS.values() for s in seeds}
    clash = mine & used
    if clash:
        raise SystemExit(f"confirmation seeds collide with seeds already used: {sorted(clash)}. "
                         "A confirmation set that reuses a selection seed is not held out.")


def jobs() -> list[dict]:
    out = []
    for op in OPERATING_POINTS:
        tag = f"{op['codec']}_scale{op['scale']}_lr{op['lr']}_{op['role']}"
        for seed in CONFIRMATION_SEEDS[op["codec"]]:
            j = {**op, "seed": seed, "tag": f"{tag}_s{seed}",
                 "root": f"{ROOT}/{op['codec']}/confirmation_v2/{tag}"}
            j["snapshot"] = f"{j['root']}/{op['role']}/s{seed}/snapshots/step{op['steps']}.pt"
            out.append(j)
    return out


def command_for(job: dict, gpus: str) -> tuple[str, list[str]]:
    env = (f"STEPS={job['steps']} LR={job['lr']} ROLES={job['role']} LIMIT=40 PGB=16 "
           f"SKIP_EVAL=1 SKIP_PREFLIGHT=1 NO_COMPILE=1")
    cmd = (f"{env} bash scripts/t2a_codec_trial.sh {job['codec']} {job['scale']} {job['seed']} "
           f"{gpus} {gpus} {job['root']}")
    return cmd, ["bash", "-lc", cmd]


def append_ledger(job: dict, command: str, ok: bool, elapsed: float) -> None:
    ledger = REPO / ROOT / job["codec"] / "state.jsonl"
    snap = REPO / job["snapshot"]
    record = {
        "phase": "confirmation", "setting": "t2a", "codec": job["codec"], "role": job["role"],
        "seed": job["seed"],
        "free_hparams": {"scale": float(job["scale"]), "learning_rate": float(job["lr"]),
                         "warmup_frac": 0.1, "steps": job["steps"]},
        "command": command,
        "artifact_root": f"{job['root']}/{job['role']}/s{job['seed']}",
        "artifact_sha256": sha256_of(snap) if ok and snap.exists() else None,
        "status": "complete" if ok else "failed",
        "selection_metric": None, "control_metric": None, "helpfulness_metric": None,
        "scoring": {
            "instrument": "scripts/t2a_score_checkpoints.py --split report "
                          "--confirm-spend-report-split --example-offset 0 --limit 64 --n-desc 3 "
                          "(the 11 genuinely held-out lol_ tasks)",
            "status": "pending",
        },
        "confirmation_context": {
            "selection_seeds": list(SELECTION_SEEDS[job["codec"]]),
            "tied_set_size": job["tied"],
            "gate": "operating point passed the AUTORESEARCH.md 3b stability gate 3/3 on the "
                    "selection seeds; this set is disjoint from those",
            "note": "Seeds are disjoint from the scout and from the selection seeds because the "
                    "separability claim uses SDs measured on the selection seeds; reusing them "
                    "would make the test statistic and the reported mean share data.",
        },
        "elapsed_seconds": round(elapsed, 1),
        "notes": "AUTORESEARCH.md 4 confirmation seed at the selected operating point. Reported "
                 "mean and SD come from this set only and are never pooled with the scout or "
                 "selection seeds. Scored ONCE on the 11-task report split.",
    }
    with ledger.open("a") as f:
        f.write(json.dumps(record) + "\n")


def run_one(job: dict, gpus: str, log_dir: Path) -> dict:
    command, argv = command_for(job, gpus)
    t0 = time.time()
    with (log_dir / f"{job['tag']}.log").open("w") as fh:
        fh.write(f"# {command}\n\n")
        fh.flush()
        rc = subprocess.call(argv, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    ok = rc == 0 and (REPO / job["snapshot"]).exists()
    append_ledger(job, command, ok, elapsed)
    print(f"[{time.strftime('%H:%M:%S')}] {job['tag']:52} {'ok' if ok else f'FAILED rc={rc}'}  "
          f"({elapsed/60:.0f} min)", flush=True)
    return {**job, "ok": ok, "rc": rc, "elapsed_s": elapsed, "command": command}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gpu-sets", default="0,1,2,3;4,5,6,7")
    p.add_argument("--log-dir", default="results/autoresearch/t2a/stage3_logs")
    args = p.parse_args()

    assert_disjoint()
    todo = jobs()
    pending = [j for j in todo if not (REPO / j["snapshot"]).exists()]
    print(f"{len(todo)} jobs ({len(OPERATING_POINTS)} operating points x 3 confirmation seeds), "
          f"{len(todo) - len(pending)} already trained, {len(pending)} pending")
    if args.dry_run:
        for j in pending:
            cmd, _ = command_for(j, args.gpu_sets.split(";")[0])
            print(f"\n  {j['tag']}\n    {cmd}")
        return
    if not pending:
        return

    sets = [s for s in args.gpu_sets.split(";") if s]
    log_dir = REPO / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nrunning {len(pending)} jobs, {len(sets)} at a time on {sets}", flush=True)
    t0 = time.time()
    results = run_queue(pending, sets, lambda j, g: run_one(j, g, log_dir))
    failed = [r for r in results if not r["ok"]]
    print(f"\ndone in {(time.time()-t0)/3600:.1f}h: {len(results)-len(failed)} ok, {len(failed)} failed")
    for r in failed:
        print(f"  FAILED {r['tag']} rc={r['rc']}")
    (log_dir / "stage3_summary.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print("\nTraining complete. The report split is NOT yet spent -- confirm every seed converged, "
          "then score once with:\n"
          "  --split report --confirm-spend-report-split --example-offset 0 --limit 64 --n-desc 3")


if __name__ == "__main__":
    main()
