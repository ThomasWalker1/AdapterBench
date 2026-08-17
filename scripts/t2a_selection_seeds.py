"""Run the selection seeds that verify each operating point (AUTORESEARCH.md 3b).

An operating point read off the scout grid is a single-seed peak, and 3b forbids promoting one: the
configuration taken to confirmation must be verified on multiple seeds using the statistic the
leaderboard reports. This trains three selection seeds per operating point, on seeds disjoint from
every scout and confirmation seed.

It runs only the declared operating point, not the other members of its tied set. Under 2's
materiality threshold those configurations are statistically indistinguishable from it, so choosing
between them cannot move the reported effect by more than the spread being reported; the tied-set
size is recorded instead, so a row reads "chosen from N indistinguishable points" rather than
implying a resolved optimum. A tied runner-up is promoted only if the declared point fails the gate.

The gate itself is applied afterwards from the scored results: a candidate is eligible only if it
converges on at least ceil(2/3) of its seeds, where converged means finite, clearing the frozen
helpfulness floor, and showing no training divergence.

Usage:
    .venv/bin/python scripts/t2a_selection_seeds.py --dry-run
    .venv/bin/python scripts/t2a_selection_seeds.py
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

# Selection seeds, disjoint from every seed already used: scouts 1701/1702/2702/4702/5001;
# earlier selection seeds 2801-2803/5101-5103; confirmations 1801-1803/1901-1903/2811-2813/
# 5111-5113 and 2704-2707. Stage 3's confirmation seeds will be a further disjoint block.
SELECTION_SEEDS = {
    "lora": (1721, 1722, 1723),
    "ia3": (1731, 1732, 1733),
    "lokr": (2721, 2722, 2723),
    "fourierft": (5021, 5022, 5023),
    "steering": (4721, 4722, 4723),
    "dora": (6711, 6712, 6713),
}

# The operating point per (codec, role), read off the decontaminated selection-split grid after
# Phase 1 closed every axis. `tied` is the number of configurations within one measured seed SD of
# this point, i.e. how many the scout seed could not separate it from.
#
# lokr's static is lr 2e-4, NOT the bare argmax lr 4e-4: those two scored 0.5226 and 0.5227, a gap
# 200x smaller than the seed SD, so §2's materiality rule closes that axis at the interior point.
OPERATING_POINTS = [
    dict(codec="lora",      role="hyper",  scale="1.414214",  lr="5e-5", steps=8000, rl=0.7422, tied=2),
    dict(codec="lora",      role="static", scale="22.627417", lr="5e-5", steps=8000, rl=0.5945, tied=4),
    dict(codec="ia3",       role="hyper",  scale="16",        lr="5e-5", steps=8000, rl=0.6732, tied=4),
    dict(codec="ia3",       role="static", scale="16",        lr="8e-4", steps=8000, rl=0.5111, tied=2),
    dict(codec="lokr",      role="hyper",  scale="0.25",      lr="2e-4", steps=6000, rl=0.7042, tied=1),
    dict(codec="lokr",      role="static", scale="0.25",      lr="2e-4", steps=6000, rl=0.5226, tied=3),
    dict(codec="fourierft", role="hyper",  scale="0.25",      lr="1e-4", steps=6000, rl=0.7261, tied=3),
    dict(codec="fourierft", role="static", scale="4",         lr="1e-4", steps=6000, rl=0.5342, tied=1),
    dict(codec="steering",  role="hyper",  scale="1",         lr="1e-4", steps=8000, rl=0.7271, tied=5),
    dict(codec="steering",  role="static", scale="64",        lr="2e-4", steps=8000, rl=0.5140, tied=2),
    # DoRA: both roles closed on interior optima of BOTH axes (scale and LR), and the two roles
    # land 64x apart in scale -- a yoked control at the hypernetwork's 0.25 would have scored
    # 0.4867 instead of 0.6074, inflating matched-minus-static by ~0.12.
    dict(codec="dora",      role="hyper",  scale="0.25",      lr="1e-4", steps=8000, rl=0.7609, tied=1),
    dict(codec="dora",      role="static", scale="16",        lr="1e-4", steps=8000, rl=0.6074, tied=1),
]


def jobs() -> list[dict]:
    out = []
    for op in OPERATING_POINTS:
        tag = f"{op['codec']}_scale{op['scale']}_lr{op['lr']}_{op['role']}"
        for seed in SELECTION_SEEDS[op["codec"]]:
            j = {**op, "seed": seed, "tag": f"{tag}_s{seed}",
                 "root": f"{ROOT}/{op['codec']}/selection_v2/{tag}"}
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
        "phase": "gate_selection", "setting": "t2a", "codec": job["codec"], "role": job["role"],
        "seed": job["seed"],
        "free_hparams": {"scale": float(job["scale"]), "learning_rate": float(job["lr"]),
                         "warmup_frac": 0.1, "steps": job["steps"]},
        "command": command,
        "artifact_root": f"{job['root']}/{job['role']}/s{job['seed']}",
        "artifact_sha256": sha256_of(snap) if ok and snap.exists() else None,
        "status": "complete" if ok else "failed",
        "selection_metric": None, "control_metric": None, "helpfulness_metric": None,
        "scoring": {"instrument": "scripts/t2a_score_checkpoints.py --example-offset 40 "
                                  "--limit 32 --n-desc 3, selection split (10 in-distribution "
                                  "lol_ tasks, examples 40+)",
                    "status": "pending"},
        "selection_context": {
            "scout_rouge_l": job["rl"],
            "tied_set_size": job["tied"],
            "operating_point_chosen_within_noise": job["tied"] > 1,
            "note": "Declared operating point after Phase 1 closed every swept axis. It is tied "
                    f"with {job['tied'] - 1} other configuration(s) under AUTORESEARCH.md 2's "
                    "materiality threshold; those are not run because choosing among tied points "
                    "cannot move the reported effect beyond the reported spread. A tied runner-up "
                    "is promoted only if this point fails the 3b stability gate.",
        },
        "elapsed_seconds": round(elapsed, 1),
        "notes": "AUTORESEARCH.md 3b selection seed. Disjoint from the scout seed and from the "
                 "later confirmation seeds. Scored on the decontaminated selection split under "
                 "ROUGE-L; the stability gate is applied from the scored results, not here.",
    }
    with ledger.open("a") as f:
        f.write(json.dumps(record) + "\n")


def run_one(job: dict, gpus: str, log_dir: Path) -> dict:
    command, argv = command_for(job, gpus)
    log = log_dir / f"{job['tag']}.log"
    t0 = time.time()
    with log.open("w") as fh:
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
    p.add_argument("--log-dir", default="results/autoresearch/t2a/stage2_selection_logs")
    args = p.parse_args()

    todo = jobs()
    pending = [j for j in todo if not (REPO / j["snapshot"]).exists()]
    print(f"{len(todo)} jobs ({len(OPERATING_POINTS)} operating points x 3 seeds), "
          f"{len(todo) - len(pending)} already trained, {len(pending)} pending")
    if args.dry_run:
        for j in pending:
            cmd, _ = command_for(j, args.gpu_sets.split(";")[0])
            print(f"\n  {j['tag']}  (tied set {j['tied']}, scout RL {j['rl']:.4f})\n    {cmd}")
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
    (log_dir / "phase2_summary.json").write_text(json.dumps(results, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
