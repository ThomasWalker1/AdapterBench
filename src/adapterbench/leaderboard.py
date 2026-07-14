"""The derived, provenance-stamped leaderboard (PROJECT_PLAN § "Evaluation and the leaderboard").

The board is **not** hand-authored and is **never a merge gate**: it is a *view* regenerated from
append-only `EvaluationResult`-derived records that the post-merge autoresearch loop produces. Each
record is stamped with `(main_git_sha, trial_id, seed, data_split, search_budget)` so the board is
reproducible and re-runnable — without `main_git_sha`, "inspect the research through git history"
breaks the first time `main` moves. Records are append-only (a new sha adds rows, never overwrites),
so losers stay on the board: the "does shape matter?" question needs the negative results.

`results/` is gitignored scratch by default, so leaderboard records live under
`results/leaderboard/` and are the intended force-add exception (or a side store) — that decision is
the caller's; this module just reads/writes the JSONL store.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

RECORDS_FILE = "records.jsonl"


@dataclass(frozen=True)
class LeaderboardRecord:
    """One codec×setting result at its own best config, fully provenance-stamped."""

    setting: str
    codec: str
    main_git_sha: str
    trial_id: str
    objective_name: str
    objective_mean: float
    objective_std: float
    best_config: dict
    seeds: list[int]
    data_split: str
    search_budget: dict
    generated_output_size: int | None = None  # for the parameter-efficiency axis of the board
    extra: dict = field(default_factory=dict)

    def key(self) -> tuple:
        """Provenance identity — a record is unique in (sha, trial, setting, codec, split, budget).
        Re-running the same sha reproduces the row; a new sha appends a new one."""
        return (self.main_git_sha, self.trial_id, self.setting, self.codec, self.data_split,
                json.dumps(self.search_budget, sort_keys=True))


def append_record(record: LeaderboardRecord, store_dir: str | Path) -> None:
    """Append one record (append-only; never overwrites). Idempotent on `key()` — re-appending an
    identical-provenance record is a no-op so re-runs don't duplicate rows."""
    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    path = store / RECORDS_FILE
    if any(r.key() == record.key() for r in load_records(store)):
        return
    with path.open("a") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def load_records(store_dir: str | Path) -> list[LeaderboardRecord]:
    path = Path(store_dir) / RECORDS_FILE
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(LeaderboardRecord(**json.loads(line)))
    return out


def record_from_autoresearch(report: dict, *, main_git_sha: str, trial_id: str, data_split: str,
                             generated_output_size: int | None = None) -> LeaderboardRecord:
    """Build a provenance-stamped record from an autoresearch `SearchReport.to_dict()` (the codec at
    its own best config). Raises if the report has no eligible best config."""
    if report.get("best_config") is None:
        raise ValueError("autoresearch report has no best config; nothing to record")
    return LeaderboardRecord(
        setting=report["setting"], codec=report["codec"], main_git_sha=main_git_sha,
        trial_id=trial_id, objective_name=report["objective"],
        objective_mean=report["best_objective_mean"], objective_std=report["best_objective_std"],
        best_config=report["best_config"], seeds=[], data_split=data_split,
        search_budget=report["search_budget"], generated_output_size=generated_output_size,
        extra={"best_under_seeded": report.get("best_under_seeded", False)},
    )


def render_leaderboard(store_dir: str | Path, *, at_sha: str | None = None) -> str:
    """Regenerate the board view from the record set. Groups by setting; within a setting shows one
    row per codec at its best config (best objective wins, ties broken by lower generated size —
    parameter efficiency). Optionally filter to a single `main_git_sha` for a reproducible snapshot.
    Losers are shown, not curated away."""
    records = load_records(store_dir)
    if at_sha is not None:
        records = [r for r in records if r.main_git_sha == at_sha]
    if not records:
        return "(no leaderboard records)"

    by_setting: dict[str, list[LeaderboardRecord]] = {}
    for r in records:
        by_setting.setdefault(r.setting, []).append(r)

    lines = ["=" * 84, "AdapterBench leaderboard (derived view — regenerated, never hand-edited)", "=" * 84]
    for setting in sorted(by_setting):
        rows = by_setting[setting]
        # one row per codec: its best-objective record (parameter efficiency breaks ties)
        best_per_codec: dict[str, LeaderboardRecord] = {}
        for r in rows:
            cur = best_per_codec.get(r.codec)
            if cur is None or (r.objective_mean, -(r.generated_output_size or 0)) > (
                    cur.objective_mean, -(cur.generated_output_size or 0)):
                best_per_codec[r.codec] = r
        ranked = sorted(best_per_codec.values(), key=lambda r: r.objective_mean, reverse=True)
        obj = ranked[0].objective_name
        lines += [f"\n[{setting}]  objective: {obj}",
                  f"  {'codec':<16}{'objective':>16}{'gen.size':>10}   best_config           provenance(sha)"]
        for r in ranked:
            cfg = ", ".join(f"{k}={v}" for k, v in r.best_config.items())
            gsize = str(r.generated_output_size) if r.generated_output_size is not None else "-"
            flag = " *under-seeded" if r.extra.get("best_under_seeded") else ""
            lines.append(f"  {r.codec:<16}{r.objective_mean:>+11.4f}±{r.objective_std:.3f}{gsize:>10}   "
                         f"{cfg:<20} {r.main_git_sha[:12]}{flag}")
    lines.append("=" * 84)
    return "\n".join(lines)
