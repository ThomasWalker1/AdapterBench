"""Run the correctness merge gate on a codec PR (PROJECT_PLAN § "Git-native benchmark").

A codec merges iff it is a valid, deterministic, fairly-comparable panel member — never because it
won. This wraps `adapterbench.merge_gate.run_gate`: it resolves the PR's changed files from git,
loads the codec manifest, and runs the correctness checks (path guard, shape-identity lint,
validate, catalog, pytest, and the generate→hook→backprop GPU smoke), exiting non-zero on any red.
Intended as the CI check on a `codec/<name>` branch.

  .venv/bin/python scripts/codec_merge_gate.py --adapter lora_r8_t2l --base main
  .venv/bin/python scripts/codec_merge_gate.py --adapter lora_r8_t2l --changed-files codecs.py,...  # explicit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from adapterbench.catalog import load_catalog
from adapterbench.merge_gate import run_gate


def _git_changed_files(base: str) -> list[str]:
    out = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"],
                         capture_output=True, text=True)
    if out.returncode != 0:  # e.g. no such ref / not a repo — fall back to working-tree changes
        out = subprocess.run(["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True)
    return [f for f in out.stdout.splitlines() if f.strip()]


def _runner(argv: list[str]) -> tuple[int, str]:
    proc = subprocess.run(argv, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="manifest name of the codec under review")
    ap.add_argument("--root", default="configs", help="catalog root (contains adapters/)")
    ap.add_argument("--base", default="main", help="git ref to diff the PR against")
    ap.add_argument("--changed-files", default=None, help="comma-separated override for the diff")
    ap.add_argument("--no-gpu-smoke", action="store_true", help="skip the GPU generate/hook/backprop smoke")
    args = ap.parse_args()

    _, adapters = load_catalog(args.root)
    if args.adapter not in adapters:
        print(f"unknown adapter '{args.adapter}'; catalog has {sorted(adapters)}")
        sys.exit(2)
    manifest = adapters[args.adapter]

    changed = (args.changed_files.split(",") if args.changed_files else _git_changed_files(args.base))
    print(f"[gate] adapter={args.adapter}  changed files ({len(changed)}): {changed}\n")

    report = run_gate(manifest, changed, runner=_runner, run_gpu_smoke=not args.no_gpu_smoke)
    print(report.render())
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
