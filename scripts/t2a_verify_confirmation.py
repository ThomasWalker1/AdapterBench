"""Verify the confirmation set is complete before the report split is measured.

The 11-task report split can be scored once, so this stands between an incomplete confirmation set
and an irreversibly consumed resource. It fails closed: any missing snapshot, any ledger record not
marked complete, or any non-finite final training loss exits nonzero and the caller must not score.

It looks at no model output and never touches the report split -- only whether every expected
checkpoint exists, whether the ledger agrees, and whether every run finished with a finite loss.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from t2a_confirmation_seeds import CONFIRMATION_SEEDS, jobs  # noqa: E402


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codecs",
        default=",".join(CONFIRMATION_SEEDS),
        help="comma-separated codec subset to verify (default: all)",
    )
    args = parser.parse_args()
    selected = {c.strip() for c in args.codecs.split(",") if c.strip()}
    unknown = sorted(selected - set(CONFIRMATION_SEEDS))
    if unknown:
        print(f"unknown codec(s): {unknown}", file=sys.stderr)
        return 2

    expected = [j for j in jobs() if j["codec"] in selected]
    problems: list[str] = []

    # 1. every snapshot present
    missing = [j["tag"] for j in expected if not (REPO / j["snapshot"]).exists()]
    if missing:
        problems.append(f"{len(missing)} missing snapshot(s): {missing[:6]}")

    # 2. ledger agrees: one `confirmation` record per (codec, role, seed), status complete
    ledger_ok: set[tuple] = set()
    for codec in selected:
        path = REPO / "results/autoresearch/t2a" / codec / "state.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("phase") == "confirmation" and r.get("status") == "complete"
                    and r.get("seed") in CONFIRMATION_SEEDS.get(codec, ())):
                ledger_ok.add((codec, r.get("role"), r.get("seed")))
    want = {(j["codec"], j["role"], j["seed"]) for j in expected}
    unledgered = sorted(want - ledger_ok)
    if unledgered:
        problems.append(f"{len(unledgered)} run(s) with no complete ledger record: {unledgered[:6]}")

    # 3. no non-finite final training loss
    nonfinite = []
    for j in expected:
        log = REPO / j["root"] / f"{j['role']}_s{j['seed']}_train.log"
        if not log.exists():
            nonfinite.append(f"{j['tag']} (no train log)")
            continue
        vals = re.findall(r"loss=([0-9.]+|nan|inf|-inf)", log.read_text(errors="ignore"))
        if not vals:
            nonfinite.append(f"{j['tag']} (no loss lines)")
        elif vals[-1] in ("nan", "inf", "-inf"):
            nonfinite.append(f"{j['tag']} (final loss {vals[-1]})")
    if nonfinite:
        problems.append(f"{len(nonfinite)} run(s) with a non-finite or unreadable loss: {nonfinite[:6]}")

    print(f"expected confirmation runs: {len(expected)}")
    print(f"  snapshots present:        {len(expected) - len(missing)}/{len(expected)}")
    print(f"  complete ledger records:  {len(want & ledger_ok)}/{len(expected)}")
    print(f"  finite final losses:      {len(expected) - len(nonfinite)}/{len(expected)}")
    if problems:
        print("\nVERIFICATION FAILED -- do not score the report split:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nverification passed: the confirmation set is complete and every run converged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
