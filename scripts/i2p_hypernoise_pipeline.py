"""Launch the full four-invariant i2p-hypernoise pipeline across GPUs (LoRA codec).

Runs a set of `adapterbench i2p-hypernoise` cells, one per GPU at a time, that together
exercise all four benchmark invariants on the LoRA baseline codec with ImageReward as the
headline reward. The first full run showed the default operating point (scale 2.0, reg 0.5)
is under-powered for ImageReward (matched gain within noise), so this grid moves to a STRONG
operating point (higher LoRA scale, lower reg) where the matched signal is expected to clear
noise, and multi-seeds the reg curve:

  1. reward-swap control  — {imagereward, red} × 3 seeds at (STRONG_SCALE, LOW_REG)
                            (matched = ImageReward-adapter's IR gain; control = red-adapter's IR gain)
  2. scale sweep          — imagereward × LoRA scale {4,8,16,32} at LOW_REG, seed 777 (invariant #2)
  3. difficulty knob      — imagereward × reg_weight {0.1,0.25,0.5,2,8} at STRONG_SCALE × 3 seeds (invariant #3+4)
  4. multi-seed           — the 3 seeds in (1) and (3)

Restart-safe: a cell whose results.jsonl already exists is skipped, so re-running resumes.
Each cell is itself checkpointed. Summarize with scripts/i2p_hypernoise_aggregate.py.

Run (background): nohup .venv/bin/python scripts/i2p_hypernoise_pipeline.py --out results/i2p_hypernoise_v2 &
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

# Operating point pinned empirically: the scale sweep (at reg 0.25) peaks at scale 4 (IR gain
# +0.17, CLIP-T preserved) and DEGRADES above it (8: -0.11, 16: -1.20 - the noise edit gets big
# enough to wreck the image). So the headline point is (scale 4, reg 0.25), not a "bigger is
# better" extrapolation.
PEAK_SCALE = 4.0
PEAK_REG = 0.25
SEEDS = [777, 778, 779]


def _dedup(jobs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for j in jobs:
        key = (j["reward"], j["scale"], j["reg"], j["seed"])
        if key not in seen:
            seen.add(key); out.append(j)
    return out


def build_jobs() -> list[dict]:
    jobs: list[dict] = []
    # (2) LoRA-scale sweep at reg=PEAK_REG, seed 777 - fine near the peak (4,8,16,32 from the
    # first pass are reused via restart-safe skip; 2,3,6 pin the peak shape). Invariant #2.
    for scale in (2.0, 3.0, 4.0, 6.0, 8.0, 16.0, 32.0):
        jobs.append(dict(reward="imagereward", scale=scale, reg=PEAK_REG, seed=777))
    # (3) reg-weight difficulty knob at PEAK_SCALE, seed 777 (invariant #3, fidelity<->reward curve)
    for reg in (0.1, 0.25, 0.5, 1.0, 2.0):
        jobs.append(dict(reward="imagereward", scale=PEAK_SCALE, reg=reg, seed=777))
    # (1+4) reward-swap control + multi-seed AT the headline point (scale 4, reg 0.25):
    # matched (imagereward) and control (red) each over 3 seeds.
    for seed in SEEDS:
        jobs.append(dict(reward="imagereward", scale=PEAK_SCALE, reg=PEAK_REG, seed=seed))
        jobs.append(dict(reward="red", scale=PEAK_SCALE, reg=PEAK_REG, seed=seed))
    return _dedup(jobs)


def cell_name(job: dict) -> str:
    scale = "default" if job["scale"] <= 0 else f"{job['scale']:g}"
    return f"{job['reward']}__scale{scale}__reg{job['reg']:g}__s{job['seed']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/i2p_hypernoise_v2")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4, help="lower (e.g. 2) to shrink the per-cell GPU footprint on a shared box")
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
            "--n-seeds", str(args.n_seeds), "--batch-size", str(args.batch_size), "--output", str(out_dir),
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
