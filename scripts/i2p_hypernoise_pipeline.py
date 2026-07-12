"""Launch the full four-invariant i2p-hypernoise pipeline across GPUs (LoRA codec).

Runs a set of `adapterbench i2p-hypernoise` cells, one per GPU at a time, that together
exercise all four benchmark invariants on the LoRA baseline codec with ImageReward as the
headline reward:

  1. reward-swap control  — {imagereward, red} targets × 3 seeds at the center config
                            (matched = ImageReward-adapter's IR gain; control = red-adapter's IR gain)
  2. scale sweep          — imagereward × LoRA scale {1,2,4,8}  (invariant #2, best-of)
  3. difficulty knob      — imagereward × reg_weight {0.5,2,8,32}  (invariant #3, fidelity<->reward curve)
  4. multi-seed           — the 3 seeds in (1)

Restart-safe: a cell whose results.jsonl already exists is skipped, so re-running resumes.
Each cell is itself checkpointed. Summarize with scripts/i2p_hypernoise_aggregate.py.

Run (background): nohup .venv/bin/python scripts/i2p_hypernoise_pipeline.py --out results/i2p_hypernoise &
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CENTER_SCALE = 0.0   # 0 => codec default (alpha/rank = 2.0)
CENTER_REG = 0.5
SEEDS = [777, 778, 779]


def build_jobs() -> list[dict]:
    jobs: list[dict] = []
    # (1) reward-swap control across seeds (also supplies invariant #4 multi-seed for imagereward)
    for reward in ("imagereward", "red"):
        for seed in SEEDS:
            jobs.append(dict(reward=reward, scale=CENTER_SCALE, reg=CENTER_REG, seed=seed))
    # (2) LoRA-scale sweep (invariant #2), headline reward, center seed/reg
    for scale in (1.0, 2.0, 4.0, 8.0):
        jobs.append(dict(reward="imagereward", scale=scale, reg=CENTER_REG, seed=777))
    # (3) reg-weight difficulty knob (invariant #3); reg 0.5 already covered at center
    for reg in (2.0, 8.0, 32.0):
        jobs.append(dict(reward="imagereward", scale=CENTER_SCALE, reg=reg, seed=777))
    return jobs


def cell_name(job: dict) -> str:
    scale = "default" if job["scale"] <= 0 else f"{job['scale']:g}"
    return f"{job['reward']}__scale{scale}__reg{job['reg']:g}__s{job['seed']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/i2p_hypernoise")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--n-seeds", type=int, default=2)
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",")]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    print(f"[pipeline] {len(jobs)} cells over {len(gpus)} GPUs -> {out_root}", flush=True)

    running: list[tuple[subprocess.Popen, dict, object]] = []
    free_gpus = list(gpus)
    pending = list(jobs)

    def launch(job, gpu):
        out_dir = out_root / cell_name(job)
        if (out_dir / "results.jsonl").exists():
            print(f"[skip] {cell_name(job)} (results.jsonl exists)", flush=True)
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            ".venv/bin/adapterbench", "i2p-hypernoise", "--device", f"cuda:{gpu}",
            "--reward", job["reward"], "--scale", str(job["scale"]), "--reg-weight", str(job["reg"]),
            "--seed", str(job["seed"]), "--steps", str(args.steps), "--eval-every", str(args.eval_every),
            "--n-seeds", str(args.n_seeds), "--output", str(out_dir),
        ]
        logf = open(out_dir / "run.log", "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        print(f"[launch gpu{gpu}] {cell_name(job)} (pid {proc.pid})", flush=True)
        return (proc, job, logf)

    while pending or running:
        while pending and free_gpus:
            job = pending.pop(0)
            gpu = free_gpus.pop(0)
            handle = launch(job, gpu)
            if handle is None:  # skipped; free the gpu immediately
                free_gpus.append(gpu)
                continue
            running.append((handle[0], handle[1], handle[2], gpu))
        time.sleep(10)
        still = []
        for proc, job, logf, gpu in running:
            if proc.poll() is None:
                still.append((proc, job, logf, gpu))
            else:
                logf.close()
                status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
                print(f"[done gpu{gpu}] {cell_name(job)} -> {status}", flush=True)
                free_gpus.append(gpu)
        running = still

    print("[pipeline] all cells complete", flush=True)


if __name__ == "__main__":
    main()
