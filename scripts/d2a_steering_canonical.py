#!/usr/bin/env python3
"""Build the compact canonical D2A record for steering from its confirmation artifacts.

Reads the five held-out confirmation runs, computes the schema-v1 fields
(`canonical_results/schema-v1.json`) exactly as `src/adapterbench/results.py`
validates them, and writes `canonical_results/d2a_steering.json`.

The headline is taken at the TRAINING length (512), matching every committed D2A
record; the difficulty curve is the seed mean at each declared eval length. Nothing
here selects anything: the free hyperparameters are passed in, frozen from the
autoresearch ledger, and the script fails rather than guessing if a seed is missing.

Usage:
  d2a_steering_canonical.py --root results/repro/d2a_numeric_decoy/steering \
      --scale 32 --learning-rate 2e-5 --steps 32000 --warmup-steps 960 [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

TRAIN_LENGTH = 512
LENGTHS = (512, 1024, 2048, 4096, 8192, 16384, 32768)
CONFIRMATION_SEEDS = (2901, 2902, 2903, 2905, 2906)
REPO = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def population_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def log_auc(deltas: dict[int, float], lengths: tuple[int, ...]) -> float:
    xs = [math.log2(length) for length in lengths]
    ys = [deltas[length] for length in lengths]
    area = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1))
    return area / (xs[-1] - xs[0])


def load_seed(run_dir: Path) -> tuple[dict[int, float], dict[int, float]]:
    matched: dict[int, float] = {}
    control: dict[int, float] = {}
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["split"] != "test" or row["adapter"] != "steering":
            continue
        if not row["task_id"].startswith("niah_") or "accuracy_ctxswap" not in row["metrics"]:
            continue
        length = int(row["task_id"].split("_", 1)[1])
        matched[length] = float(row["metrics"]["accuracy"])
        control[length] = float(row["metrics"]["accuracy_ctxswap"])
    missing = [length for length in LENGTHS if length not in matched]
    if missing:
        raise SystemExit(f"{run_dir} is missing eval lengths {missing}")
    return matched, control


def model_revision(model_id: str) -> str:
    """Read the cached HF snapshot commit for the interpreter, for provenance."""
    cache = Path.home() / ".cache/huggingface/hub" / f"models--{model_id.replace('/', '--')}" / "refs/main"
    return cache.read_text().strip() if cache.exists() else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--learning-rate", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--ledger", type=Path, default=Path("results/autoresearch/d2a/steering/state.jsonl"))
    parser.add_argument("--summary", default="", help="selection-trail prose; required for a publishable record")
    parser.add_argument("--out", type=Path, default=Path("canonical_results/d2a_steering.json"))
    args = parser.parse_args()

    seeds, per_seed = [], {}
    for seed in CONFIRMATION_SEEDS:
        matched, control = load_seed(args.root / f"s{seed}")
        per_seed[seed] = (matched, control)
        seeds.append({
            "seed": seed,
            "matched": matched[TRAIN_LENGTH],
            "control": control[TRAIN_LENGTH],
            "delta": matched[TRAIN_LENGTH] - control[TRAIN_LENGTH],
        })

    deltas = [entry["delta"] for entry in seeds]
    value = sum(deltas) / len(deltas)
    curve = []
    for length in LENGTHS:
        matched_mean = sum(per_seed[s][0][length] for s in CONFIRMATION_SEEDS) / len(CONFIRMATION_SEEDS)
        control_mean = sum(per_seed[s][1][length] for s in CONFIRMATION_SEEDS) / len(CONFIRMATION_SEEDS)
        curve.append({
            "axis": length, "matched": matched_mean, "control": control_mean,
            "delta": matched_mean - control_mean,
        })

    # crossover: largest eval length whose seed-mean matched retrieval is still >= 0.5
    crossover = max((p["axis"] for p in curve if p["matched"] >= 0.5), default=None)
    tail_aucs = [
        log_auc({l: per_seed[s][0][l] - per_seed[s][1][l] for l in LENGTHS}, LENGTHS[1:])
        for s in CONFIRMATION_SEEDS
    ]

    record = {
        "schema_version": 1,
        "setting": "D2A",
        "codec": "steering",
        "display_name": "steering",
        "metric": "numeric_decoy_needle_retrieval_exact_match_accuracy",
        "headline": {
            "comparison": "matched_minus_control",
            "control_name": "wrong_document_context_swap",
            "matched": sum(entry["matched"] for entry in seeds) / len(seeds),
            "control": sum(entry["control"] for entry in seeds) / len(seeds),
            "value": value,
            "variation": population_std(deltas),
            "variation_kind": "population_standard_deviation",
            "unit": "exact_match_accuracy",
            "direction": "higher_is_better",
        },
        "seed_results": seeds,
        "difficulty_curve": curve,
        "summary": {
            "crossover_length": crossover,
            "training_length": TRAIN_LENGTH,
            "numeric_decoy_count": 4,
            "tail_log_auc": sum(tail_aucs) / len(tail_aucs),
            "tail_log_auc_variation": population_std(tail_aucs),
        },
        "free_hyperparameters": {
            "scale": args.scale,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "steps": args.steps,
        },
        "fixed_shape_parameters": {
            "target_modules": ["block"],
            "update": "h -> h + scale * v on the residual stream (activation-space; no weight edit)",
            "generated_scalars_per_layer": 1024,
            "generated_scalars_all_layers": 1024 * 28,
            "n_latents": 208,
            "num_blocks": 8,
        },
        "selection_trail": {
            "protocol": "AUTORESEARCH.md",
            "state_artifacts": [{
                "path": str(args.ledger),
                "sha256": sha256(REPO / args.ledger),
            }],
            "summary": args.summary,
        },
        "provenance": {
            "models": [{"model_id": "Qwen/Qwen3-0.6B", "revision": model_revision("Qwen/Qwen3-0.6B")}],
            "data": [{
                "path": "src/adapterbench/t2a/niah_data.py",
                "sha256": sha256(REPO / "src/adapterbench/t2a/niah_data.py"),
            }],
            "source_artifacts": [
                {
                    "seed": seed,
                    "path": str(args.root / f"s{seed}" / "results.jsonl"),
                    "sha256": sha256(args.root / f"s{seed}" / "results.jsonl"),
                }
                for seed in CONFIRMATION_SEEDS
            ],
            "environment": {
                "python": subprocess.run(
                    [str(REPO / ".venv/bin/python"), "-c", "import platform;print(platform.python_version())"],
                    capture_output=True, text=True, check=True,
                ).stdout.strip(),
                "lockfile_sha256": sha256(REPO / "uv.lock"),
            },
        },
        "reproduction": {
            "script": "document_niah_numeric_decoy_steering_all.sh",
            "smoke": (
                "uv run adapterbench d2a-niah --adapters steering --codec-scaling 32 --needle-style "
                "realistic_numeric_decoys --numeric-decoy-count 4 --context-lengths 512 --eval-context-lengths 512 "
                "--num-train-documents 2 --steps 1 --eval-every 1 --learning-rate 2e-5 --warmup-steps 1 "
                "--n-latents 208 --num-blocks 8 --eval-limit 1 --eval-seed 2904 --seed 902 --device cuda:0 "
                "--output results/autoresearch/d2a/steering/smoke"
            ),
            "full": "bash scripts/reproduce/document_niah_numeric_decoy_steering.sh cuda:0 2901",
            "aggregate": (
                "uv run python scripts/d2a_niah_aggregate.py --root results/repro/d2a_numeric_decoy/steering "
                "--min-seeds 5 --train-length 512"
            ),
        },
    }

    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"  headline: matched-control = {value:+.3f} +/- {population_std(deltas):.3f} over {len(seeds)} seeds")
    print(f"  crossover: {crossover} ({crossover // TRAIN_LENGTH}x training length)" if crossover else "  crossover: none")
    print(f"  tail_log_auc: {sum(tail_aucs)/len(tail_aucs):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
