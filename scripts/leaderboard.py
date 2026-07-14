"""Manage the derived, provenance-stamped leaderboard (PROJECT_PLAN § "Evaluation and the leaderboard").

The board is a *view* regenerated from append-only records — never hand-edited, never a merge gate.

Record a codec's result from an autoresearch report (stamped with the current main sha):
  .venv/bin/python scripts/leaderboard.py record \
      --report results/i2p_hypernoise_v2/autoresearch_i2p_hypernoise_lora.json \
      --trial-id "i2p--lora--r8" --data-split eval_prompts --gen-size 32768

Regenerate the board view (optionally at a specific sha for a reproducible snapshot):
  .venv/bin/python scripts/leaderboard.py render [--at-sha <sha>]

`results/` is gitignored scratch; leaderboard records under results/leaderboard/ are the intended
force-add exception (`git add -f`) or a side store — that decision is yours, this only reads/writes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from adapterbench.leaderboard import append_record, record_from_autoresearch, render_leaderboard

DEFAULT_STORE = "results/leaderboard"


def _current_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="append a provenance-stamped record from an autoresearch report")
    rec.add_argument("--report", required=True, help="an autoresearch SearchReport json")
    rec.add_argument("--trial-id", required=True)
    rec.add_argument("--data-split", required=True)
    rec.add_argument("--gen-size", type=int, default=None, help="generated output size (parameter-efficiency axis)")
    rec.add_argument("--sha", default=None, help="main git sha (default: current HEAD)")
    rec.add_argument("--store", default=DEFAULT_STORE)

    ren = sub.add_parser("render", help="regenerate the board view from the record set")
    ren.add_argument("--store", default=DEFAULT_STORE)
    ren.add_argument("--at-sha", default=None)

    args = ap.parse_args()
    if args.cmd == "record":
        report = json.loads(Path(args.report).read_text())
        record = record_from_autoresearch(
            report, main_git_sha=args.sha or _current_sha(), trial_id=args.trial_id,
            data_split=args.data_split, generated_output_size=args.gen_size)
        append_record(record, args.store)
        print(f"recorded {record.codec}@{record.setting} = {record.objective_mean:+.4f} "
              f"(sha {record.main_git_sha[:12]}) -> {args.store}")
    elif args.cmd == "render":
        print(render_leaderboard(args.store, at_sha=args.at_sha))


if __name__ == "__main__":
    main()
