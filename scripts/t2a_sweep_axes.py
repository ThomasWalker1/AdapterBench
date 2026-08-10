"""Sweep and close a codec's free hyperparameter axes on the selection split.

Trains the scout-seed rungs needed to close an axis: an extension one geometric step outward from a
ladder end where the optimum currently sits, or a slice of an axis that was never swept at the
operating point. Both are single-seed work on each codec's own scout seed, so a new rung is
comparable to the ladder it extends rather than confounding hyperparameter with seed. Multi-seed
selection and confirmation are separate scripts.

Each job trains ONE role, so every output directory describes exactly one (role, scale, lr) and the
checkpoint index resolves hyperparameters from the path. Two jobs run concurrently on disjoint 4-GPU
sets, which is the same footprint as one paired trial (PGB 16 x 4 GPUs = the released effective
batch of 64).

Extension is capped at two steps per (codec, axis, direction) per AUTORESEARCH.md 2, enforced here
rather than left to the document: a third step is refused with an error saying to declare a fresh
ladder instead. "Extend while the endpoint wins" is unbounded exactly when the endpoint wins for the
wrong reason, so the cap and the materiality threshold bound it from both sides.

Scoring is not done here; the inline CE eval is not the selector. Score with
`scripts/t2a_score_checkpoints.py`.

Usage:
    .venv/bin/python scripts/t2a_sweep_axes.py --dry-run
    .venv/bin/python scripts/t2a_sweep_axes.py            # resumable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from threading import Lock

REPO = Path(__file__).resolve().parents[1]
ROOT = "results/autoresearch/t2a"

# Each codec's own scout seed, so an extended rung is comparable to the ladder it extends rather
# than confounding a new hyperparameter with a new seed.
SCOUT_SEED = {"lora": 1701, "ia3": 1702, "lokr": 2702, "fourierft": 5001, "steering": 4702}
# The common step budget each codec's ladder was swept at.
BUDGET = {"lora": 8000, "ia3": 8000, "lokr": 6000, "fourierft": 6000, "steering": 8000}
# Scales at which a codec's hypernetwork optimum sits and whose LR axis needs sweeping there.
NEW_ARGMAX_SCALE = {"lokr": "0.25", "fourierft": "0.25", "lora": "1.414214", "steering": "1"}
# x2 geometric neighbours of the 1e-4 setting default, matching the grid ia3 itself explored.
NEW_ARGMAX_LRS = ["5e-5", "2e-4"]

# AUTORESEARCH.md 2's two-extension cap: at most two steps outward from a declared ladder, counted
# per (codec, axis, direction). Enforced here rather than left to the doc, because "extend while the
# endpoint wins" is unbounded precisely when the endpoint wins for the wrong reason -- `ia3` chased
# its scale ladder four steps down into a region where its static control had degenerated to a no-op.
EXTENSION_CAP = 2


def jobs() -> list[dict]:
    out: list[dict] = []

    # --- A. boundary extensions -------------------------------------------------------
    # One geometric step outward from the endpoint the optimum sits on. AUTORESEARCH.md 2's rule:
    # keep extending while the endpoint wins, stop when the next point is worse or training is
    # numerically invalid. Only the FIRST step is queued here; a further step is a follow-up
    # decision once this one is scored, not a pre-authorised sweep.
    out += [
        dict(codec="ia3", role="static", scale="16", lr="1.6e-3", phase="boundary_extension",
             axis="lr", direction="up", extension_step=1,
             tag="ia3_static_lr1.6e-3",
             why="static* sat at lr 8e-4, the top of the swept LR ladder, so static* is a lower "
                 "bound. Static only: the hypernetwork diverges at 8e-4, so the axis cannot be co-swept."),
        dict(codec="ia3", role="hyper", scale="16", lr="2.5e-5", phase="boundary_extension",
             axis="lr", direction="down", extension_step=1,
             tag="ia3_hyper_lr2.5e-5",
             why="hypernetwork argmax sat at lr 5e-5, the bottom of the swept LR ladder."),
        dict(codec="lora", role="static", scale="22.627417", lr="2.5e-5", phase="boundary_extension",
             axis="lr", direction="down", extension_step=1,
             tag="lora_static_lr2.5e-5",
             why="static* sat at lr 5e-5, the bottom of the swept LR ladder."),
        dict(codec="steering", role="static", scale="64", lr="4e-4", phase="boundary_extension",
             axis="lr", direction="up", extension_step=1,
             tag="steering_static_lr4e-4",
             why="static* sat at lr 2e-4, the top of the swept LR ladder."),
    ]

    # --- B. LR sweep at the new argmax scale ------------------------------------------
    # Both roles, because the static is selected over the same grid and a scale it was never run at
    # cannot be ruled out as its optimum. Fairness ground (steering's own ladder declaration made
    # the same argument): ia3 was reported at a swept LR, so reporting the others at an unswept
    # default would under-tune them relative to it.
    for codec, scale in NEW_ARGMAX_SCALE.items():
        for lr in NEW_ARGMAX_LRS:
            for role in ("hyper", "static"):
                out += [dict(codec=codec, role=role, scale=scale, lr=lr,
                             phase="learning_rate_locator_at_new_argmax",
                             tag=f"{codec}_scale{scale}_lr{lr}_{role}",
                             why=f"The {codec} hypernetwork optimum is at scale {scale}; the LR axis is "
                                 f"swept there so the point is selected on a closed axis.")]

    # --- C. Phase 1b: close the axes Phase 1's own results opened ----------------------
    # Scored on the decontaminated gate, Phase 1 closed every boundary extension (each step-1 point
    # was WORSE than the incumbent) and closed the LR axis at fourierft's and steering's new argmax
    # scales with interior peaks. Two slices are still open, and they are open in opposite senses:
    #
    #   * lokr at scale 0.25 peaks at lr 2e-4 for BOTH roles, and 2e-4 is the top of lokr's swept
    #     LR set -- a genuine ladder extension, so it is step 1 of 2 for (lokr, lr, up);
    #   * lora's hypernetwork at scale 1.414214 peaks at lr 5e-5, the bottom of the LRs run in that
    #     slice. But lr 2.5e-5 is ALREADY in lora's swept set (its static's boundary extension put it
    #     there), so this is not extending the ladder, it is filling a slice of an existing one --
    #     no extension_step, and it does not consume lora's (lr, down) budget.
    out += [
        dict(codec="lokr", role="hyper", scale="0.25", lr="4e-4", phase="boundary_extension",
             axis="lr", direction="up", extension_step=1, tag="lokr_scale0.25_lr4e-4_hyper",
             why="Phase 1 put the lokr hypernetwork optimum at scale 0.25 lr 2e-4, the top of the "
                 "swept LR set, so the optimum is unbounded above in that slice."),
        dict(codec="lokr", role="static", scale="0.25", lr="4e-4", phase="boundary_extension",
             axis="lr", direction="up", extension_step=1, tag="lokr_scale0.25_lr4e-4_static",
             why="Phase 1 also moved the lokr static optimum to scale 0.25 lr 2e-4, the same "
                 "endpoint, so static* is a lower bound until this is run."),
        dict(codec="lora", role="hyper", scale="1.414214", lr="2.5e-5",
             phase="learning_rate_locator_at_new_argmax", tag="lora_scale1.414214_lr2.5e-5_hyper",
             why="Phase 1 moved the lora hypernetwork optimum to scale 1.414214 lr 5e-5, the lowest "
                 "LR run in that slice. lr 2.5e-5 is already in lora's swept set, so this fills the "
                 "slice rather than extending the ladder."),
    ]

    # --- D. Phase 1c: the one axis still nominally open ---------------------------------
    # lokr's hypernetwork closed at lr 2e-4 (lr 4e-4 was worse). Its STATIC came back 0.5227 at
    # lr 4e-4 against 0.5226 at 2e-4 -- a 0.0001 difference against a seed SD of ~0.021, i.e. a flat
    # plateau, not an improvement. The literal "extend while the endpoint wins" rule still points
    # outward, so step 2 of 2 is taken to settle it by measurement rather than by argument. This
    # exhausts lokr's (lr, up) budget: whatever this returns, the axis is reported closed.
    out += [
        dict(codec="lokr", role="static", scale="0.25", lr="8e-4", phase="boundary_extension",
             axis="lr", direction="up", extension_step=2, tag="lokr_scale0.25_lr8e-4_static",
             why="lokr static at scale 0.25 was nominally still at its LR endpoint (0.5227 at 4e-4 "
                 "vs 0.5226 at 2e-4, a difference 200x smaller than the seed SD). Step 2 of 2 under "
                 "the AUTORESEARCH.md 2 cap; the axis is reported closed after this regardless. Note "
                 "8e-4 is where ia3 catastrophically failed its helpfulness floor, so divergence here "
                 "would itself be the closing evidence."),
    ]

    # Refuse to queue past the cap. A third step is not a bigger sweep, it is a signal that the
    # declared ladder was centred wrong -- which needs a fresh preregistered ladder and a human
    # decision, not another rung appended to a runaway chain.
    over = [j for j in out if j.get("extension_step", 0) > EXTENSION_CAP]
    if over:
        raise SystemExit(
            "AUTORESEARCH.md 2 two-extension cap exceeded for: "
            + ", ".join(f"{j['tag']} ({j['codec']} {j.get('axis')} {j.get('direction')} "
                        f"step {j['extension_step']})" for j in over)
            + ". Declare a fresh ladder centred on the new region instead of extending again, and "
              "record the previous cap as a limitation.")

    for j in out:
        j["seed"] = SCOUT_SEED[j["codec"]]
        j["steps"] = BUDGET[j["codec"]]
        j["root"] = f"{ROOT}/{j['codec']}/stage2/{j['tag']}"
        j["snapshot"] = f"{j['root']}/{j['role']}/s{j['seed']}/snapshots/step{j['steps']}.pt"
    return out


def command_for(job: dict, gpus: str) -> tuple[str, list[str]]:
    """The exact shell command, recorded verbatim in the ledger. `GH`/`GS` are both set to the
    job's GPU set; the trial script only launches the role named by ROLES."""
    env = (f"STEPS={job['steps']} LR={job['lr']} ROLES={job['role']} LIMIT=40 PGB=16 "
           f"SKIP_EVAL=1 SKIP_PREFLIGHT=1 NO_COMPILE=1")
    cmd = (f"{env} bash scripts/t2a_codec_trial.sh {job['codec']} {job['scale']} {job['seed']} "
           f"{gpus} {gpus} {job['root']}")
    return cmd, ["bash", "-lc", cmd]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def append_ledger(job: dict, command: str, ok: bool, elapsed: float) -> None:
    """One append-only record per role-run, per AUTORESEARCH.md "Search state".

    The schema gains a `role` field because a record now describes ONE role: the corrected protocol
    selects the static independently, so a rung is no longer a hyper/static pair at one
    hyperparameter. `selection_metric` is null on purpose -- the inline 21-task CE eval is no longer
    the selector, and the behavioural metric is computed afterwards by the re-scoring driver on the
    selection split at example offset 40. A second record is appended once that score exists;
    this one is never overwritten.
    """
    ledger = REPO / ROOT / job["codec"] / "state.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    snap = REPO / job["snapshot"]
    record = {
        "phase": job["phase"], "setting": "t2a", "codec": job["codec"], "role": job["role"],
        "seed": job["seed"],
        "free_hparams": {"scale": float(job["scale"]), "learning_rate": float(job["lr"]),
                         "warmup_frac": 0.1, "steps": job["steps"]},
        "command": command,
        "artifact_root": f"{job['root']}/{job['role']}/s{job['seed']}",
        "artifact_sha256": sha256_of(snap) if ok and snap.exists() else None,
        "status": "complete" if ok else "failed",
        # Extension bookkeeping for AUTORESEARCH.md 2's two-step cap, recorded so the depth of any
        # extension chain is auditable from the ledger alone rather than reconstructed from paths.
        "extension": ({"axis": job["axis"], "direction": job["direction"],
                       "step": job["extension_step"], "cap": EXTENSION_CAP}
                      if job.get("extension_step") else None),
        "selection_metric": None, "control_metric": None, "helpfulness_metric": None,
        "scoring": {
            "instrument": "scripts/t2a_score_checkpoints.py --example-offset 40 --limit 32 "
                          "--n-desc 3, selection split (10 in-distribution lol_ tasks)",
            "status": "pending",
        },
        "elapsed_seconds": round(elapsed, 1),
        "notes": job["why"] + " Scored separately under the corrected protocol: ROUGE-L "
                 "matched - static* on the decontaminated selection split, never the inline "
                 "21-task CE. Stage 2 Phase 1 (scout seed, single role); multi-seed selection and "
                 "confirmation are later phases.",
    }
    with ledger.open("a") as f:
        f.write(json.dumps(record) + "\n")


def run_queue(pending: list[dict], gpu_sets: list[str], run) -> list[dict]:
    """Run `pending` over `gpu_sets`, one worker thread per set, each using ONLY its own set for its
    whole lifetime.

    Assigning sets round-robin by job index instead lets a fast slot start a job whose set is still
    busy -- two trainings on the same GPUs. Shared by the scout and selection phases so that fix
    lives in exactly one place.
    """
    results: list[dict] = []
    queue: "Queue[dict]" = Queue()
    for job in pending:
        queue.put(job)
    lock = Lock()

    def worker(gpus: str) -> None:
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                return
            try:
                result = run(job, gpus)
            finally:
                queue.task_done()
            with lock:
                results.append(result)

    with ThreadPoolExecutor(max_workers=len(gpu_sets)) as pool:
        for fut in [pool.submit(worker, s) for s in gpu_sets]:
            fut.result()
    return results


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
    status = "ok" if ok else f"FAILED rc={rc}"
    print(f"[{time.strftime('%H:%M:%S')}] {job['tag']:44} {status}  ({elapsed/60:.0f} min)", flush=True)
    return {**job, "ok": ok, "rc": rc, "elapsed_s": elapsed, "command": command}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gpu-sets", default="0,1,2,3;4,5,6,7",
                   help="semicolon-separated GPU sets; one concurrent job per set")
    p.add_argument("--log-dir", default="results/autoresearch/t2a/stage2_logs")
    args = p.parse_args()

    todo = jobs()
    pending = [j for j in todo if not (REPO / j["snapshot"]).exists()]
    print(f"{len(todo)} jobs, {len(todo) - len(pending)} already trained, {len(pending)} pending")
    by_phase: dict[str, int] = {}
    for j in pending:
        by_phase[j["phase"]] = by_phase.get(j["phase"], 0) + 1
    for k, v in by_phase.items():
        print(f"  {k}: {v}")
    if args.dry_run:
        for j in pending:
            cmd, _ = command_for(j, args.gpu_sets.split(";")[0])
            print(f"\n  {j['tag']}  ({j['codec']} {j['role']} scale={j['scale']} lr={j['lr']} "
                  f"steps={j['steps']} s{j['seed']})\n    {cmd}")
        return
    if not pending:
        return

    sets = [s for s in args.gpu_sets.split(";") if s]
    log_dir = REPO / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nrunning {len(pending)} jobs, {len(sets)} at a time on GPU sets {sets}", flush=True)
    t0 = time.time()
    results = run_queue(pending, sets, lambda j, g: run_one(j, g, log_dir))

    failed = [r for r in results if not r["ok"]]
    print(f"\ndone in {(time.time()-t0)/3600:.1f}h: {len(results)-len(failed)} ok, {len(failed)} failed")
    for r in failed:
        print(f"  FAILED {r['tag']} rc={r['rc']}")
    (log_dir / "phase1_summary.json").write_text(json.dumps(results, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
