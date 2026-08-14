"""Index every T2A checkpoint on disk and reconcile it against the append-only ledgers.

Emits, under `--out`:

  * `checkpoint_index.jsonl` -- one row per snapshot: codec, role, scale, lr, steps, seed, path, the
    ledger record it matches, and whether it is in scope for scoring;
  * `index_reconciliation.json` -- ledger records whose checkpoint is missing, checkpoints with no
    ledger record, duplicate records, and rungs missing a role their ledger declares.

Everything downstream keys off this index, so it is the one place hyperparameters are resolved: a
checkpoint whose codec scale cannot be established is refused rather than scored at a default.
Nothing here loads a model or touches a GPU.

A snapshot is IN SCOPE when it belongs to an audited codec and is a real sweep rung. Excluded, with
the reason recorded rather than dropped silently: smoke runs, and probes at step budgets or scales
that appear on no declared ladder (AUTORESEARCH.md compares only equal final budgets).

Usage:
    .venv/bin/python scripts/t2a_checkpoint_index.py --codecs lora,ia3,lokr,fourierft,steering,dora
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDITED_CODECS = ("lora", "ia3", "lokr", "fourierft")
# Extra roots that ledgers point at but which do not live under results/autoresearch/t2a.
EXTRA_ROOTS = ("results/repro",)

# Rungs whose checkpoints exist but carry no state.jsonl record. Both are the ia3 and lora
# three-seed confirmation sets: fourierft's and lokr's confirmations WERE ledgered (as
# `results/repro/...` roots), these two were not. The hyperparameters are recovered from two
# committed sources that agree with each other, not inferred from the directory name -- the
# directory name carries no scale. Recorded here explicitly so the re-scoring pass never
# silently guesses a scale, which would make its numbers meaningless.
LEDGERLESS_RUNGS = {
    "results/autoresearch/t2a/ia3/confirmation": {
        "codec": "ia3", "scale": 16.0, "lr": 4e-4, "warmup_frac": 0.1, "steps": 8000,
        "phase": "confirmation",
        "hparam_provenance": "canonical_results/t2a_ia3.json free_hyperparameters + "
                             "scripts/reproduce/task_t2a_ia3.sh (agree: scale 16, lr 4e-4, 8000 steps)",
    },
    "results/autoresearch/t2a/lora/confirmation": {
        "codec": "lora", "scale": 22.627417, "lr": 1e-4, "warmup_frac": 0.1, "steps": 8000,
        "phase": "confirmation",
        "hparam_provenance": "canonical_results/t2a_lora_r8.json free_hyperparameters + "
                             "scripts/reproduce/task_t2a_lora.sh (agree: scale 22.627417, lr 1e-4, 8000 steps)",
    },
}

SNAPSHOT_RE = re.compile(r"snapshots/step(\d+)\.pt$")
SEED_DIR_RE = re.compile(r"^s(\d+)$")
SCALE_DIR_RE = re.compile(r"scale([0-9.]+(?:e-?\d+)?)")
LR_DIR_RE = re.compile(r"lr([0-9.]+e-?\d+|[0-9.]+)")


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO))


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# --- ledger side ---------------------------------------------------------------------


def normalize_ledger_root(root: str, seed) -> tuple[str, int | None]:
    """Ledger `artifact_root` is written three ways across the sweep's history: the rung
    directory, the rung's `hyper/s<seed>` run directory, or a bare `results/repro/...` run
    directory. Collapse all three to `(rung_dir, seed)` so a rung can be matched to its
    `hyper/` and `static/` snapshots."""
    parts = Path(root).parts
    if len(parts) >= 2 and SEED_DIR_RE.match(parts[-1]) and parts[-2] in ("hyper", "static", "static_recovered"):
        return str(Path(*parts[:-2])), int(SEED_DIR_RE.match(parts[-1]).group(1))
    return root, (int(seed) if seed is not None else None)


def load_ledgers() -> list[dict]:
    entries = []
    for ledger in sorted((REPO / "results/autoresearch/t2a").rglob("state.jsonl")):
        for lineno, line in enumerate(ledger.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            fh = record.get("free_hparams") or {}
            root, seed = normalize_ledger_root(record.get("artifact_root") or "", record.get("seed"))
            entries.append({
                "ledger": rel(ledger), "lineno": lineno,
                "codec": record.get("codec"), "phase": record.get("phase"),
                "status": record.get("status"), "seed": seed,
                "scale": fh.get("scale"), "lr": fh.get("learning_rate"),
                "warmup_frac": fh.get("warmup_frac"), "steps": fh.get("steps"),
                "rung_dir": root,
                "selection_metric": record.get("selection_metric"),
                "control_metric": record.get("control_metric"),
                "helpfulness_metric": record.get("helpfulness_metric"),
                "role": record.get("role"),
                "ce_matched": record.get("ce_matched"),
                "converged": record.get("converged"),
                "notes": record.get("notes"),
            })
    return entries


# --- disk side -----------------------------------------------------------------------


def scope_of(snapshot_rel: str, codec: str | None, steps: int, audited=AUDITED_CODECS) -> tuple[bool, str]:
    parts = Path(snapshot_rel).parts
    if codec not in audited:
        return False, (f"codec {codec!r} is not in the audited set {sorted(audited)}"
                       + (" (steering has no canonical_results/t2a_*.json row: its T2A search stopped at "
                          "a selection_promotion_declaration and was never confirmed)"
                          if codec == "steering" else ""))
    if any(p.startswith("smoke") for p in parts):
        return False, "smoke run, never selection evidence"
    if "t2a_base_diag" in parts:
        return False, "pre-autoresearch base diagnostic, not a ladder rung"
    if snapshot_rel.startswith("results/repro/t2a_ia3/"):
        return False, "historical scale-32 probe at a 5k budget that appears on no declared ladder"
    if steps <= 100:
        return False, f"step budget {steps} is a smoke, not a rung"
    return True, ""


def infer_codec(snapshot_rel: str) -> str | None:
    parts = Path(snapshot_rel).parts
    if parts[:3] == ("results", "autoresearch", "t2a") and len(parts) > 3:
        return parts[3]
    for codec in (*AUDITED_CODECS, "steering", "dora"):
        if re.search(rf"(?:^|[/_]){codec}(?:[/_]|$)", snapshot_rel):
            return codec
    return None


def walk_snapshots(audited=AUDITED_CODECS) -> list[dict]:
    roots = [REPO / "results/autoresearch/t2a"] + [REPO / r for r in EXTRA_ROOTS]
    found = []
    for root in roots:
        if not root.exists():
            continue
        for snapshot in sorted(root.rglob("snapshots/step*.pt")):
            snapshot_rel = rel(snapshot)
            m = SNAPSHOT_RE.search(snapshot_rel)
            if not m:
                continue
            steps = int(m.group(1))
            run_dir = snapshot.parent.parent          # .../<role>/s<seed>
            seed_m = SEED_DIR_RE.match(run_dir.name)
            role_dir = run_dir.parent.name
            role = {"hyper": "hyper", "static": "static", "static_recovered": "static"}.get(role_dir)
            if role is None or seed_m is None:
                found.append({"path": snapshot_rel, "steps": steps, "role": None, "seed": None,
                              "in_scope": False, "scope_reason": f"unrecognised layout under {role_dir!r}"})
                continue
            codec = infer_codec(snapshot_rel)
            in_scope, reason = scope_of(snapshot_rel, codec, steps, audited)
            found.append({
                "path": snapshot_rel, "codec": codec, "role": role,
                "role_dir": role_dir, "seed": int(seed_m.group(1)), "steps": steps,
                "rung_dir": rel(run_dir.parent.parent),
                "in_scope": in_scope, "scope_reason": reason,
            })
    return found


# --- join ----------------------------------------------------------------------------


def scale_from_dir(rung_dir: str) -> float | None:
    m = SCALE_DIR_RE.search(Path(rung_dir).name)
    return float(m.group(1)) if m else None


def lr_from_dir(rung_dir: str) -> float | None:
    m = LR_DIR_RE.search(Path(rung_dir).name)
    return float(m.group(1)) if m else None


def build(with_hashes: bool, audited=AUDITED_CODECS) -> tuple[list[dict], dict]:
    ledger_entries = load_ledgers()
    snapshots = walk_snapshots(audited)

    # (rung_dir, seed, steps) -> ledger entries. A rung can carry several records: an
    # append-only rerun, or a rejected attempt followed by a completed one.
    by_key: dict[tuple, list[dict]] = {}
    for e in ledger_entries:
        if e["steps"] is None:
            continue
        by_key.setdefault((e["rung_dir"], e["seed"], int(e["steps"])), []).append(e)

    index, unledgered, duplicates = [], [], []
    for snap in snapshots:
        if snap.get("role") is None:
            index.append(snap)
            continue
        key = (snap["rung_dir"], snap["seed"], snap["steps"])
        matches = by_key.get(key, [])
        matches = [m for m in matches
                   if m.get("role") in (None, snap["role"])]
        complete = [m for m in matches if m["status"] == "complete"]
        chosen = complete[0] if complete else (matches[0] if matches else None)
        if len(complete) > 1 and key not in {tuple(d["key"]) for d in duplicates}:
            metrics = {round(m["selection_metric"], 12) if isinstance(m["selection_metric"], float) else m["selection_metric"]
                       for m in complete}
            duplicates.append({
                "key": [*key], "n_complete_records": len(complete),
                "metrics_agree": len(metrics) == 1,
                "records": [{"ledger": m["ledger"], "lineno": m["lineno"], "phase": m["phase"],
                             "selection_metric": m["selection_metric"]} for m in complete]})
        row = {
            "checkpoint": snap["path"], "codec": snap["codec"], "role": snap["role"],
            "seed": snap["seed"], "steps": snap["steps"], "rung_dir": snap["rung_dir"],
            "role_dir": snap["role_dir"],
            "scale": (chosen or {}).get("scale", scale_from_dir(snap["rung_dir"])),
            "lr": (chosen or {}).get("lr", lr_from_dir(snap["rung_dir"])),
            "warmup_frac": (chosen or {}).get("warmup_frac"),
            "phase": (chosen or {}).get("phase"),
            "ledger": (chosen or {}).get("ledger"),
            "ledger_lineno": (chosen or {}).get("lineno"),
            "ledger_status": (chosen or {}).get("status"),
            "ledger_role": (chosen or {}).get("role"),
            "ce_selection_metric": (chosen or {}).get("selection_metric"),
            "ce_control_metric": (chosen or {}).get("control_metric"),
            "ce_matched": (chosen or {}).get("ce_matched"),
            "ce_helpfulness_metric": (chosen or {}).get("helpfulness_metric"),
            "converged": (chosen or {}).get("converged"),
            "in_scope": snap["in_scope"], "scope_reason": snap["scope_reason"],
            "has_ledger_entry": chosen is not None,
        }
        recovered = LEDGERLESS_RUNGS.get(snap["rung_dir"])
        if chosen is None and recovered is not None:
            row.update({k: v for k, v in recovered.items() if k != "codec"})
            row["hparams_recovered"] = True
        if row["scale"] is None:
            row["scale"] = scale_from_dir(snap["rung_dir"])
        if row["lr"] is None:
            row["lr"] = lr_from_dir(snap["rung_dir"])
        if with_hashes:
            row["sha256"] = sha256_of(REPO / snap["path"])
        if chosen is None and snap["in_scope"]:
            unledgered.append({"checkpoint": snap["path"], "rung_dir": snap["rung_dir"],
                               "seed": snap["seed"], "steps": snap["steps"],
                               "hparams_recovered_from": (recovered or {}).get("hparam_provenance")})
        index.append(row)

    on_disk = {(r["rung_dir"], r["seed"], r["steps"], r["role"]) for r in index if r.get("role")}
    missing = []
    for e in ledger_entries:
        if e["steps"] is None or e["seed"] is None or e["codec"] not in audited:
            continue
        # `declared` / `observation` records are bookkeeping (a pre-registered ladder, a confound
        # note); `amendment` records annotate an existing rung. None of them promises a checkpoint.
        if e["status"] in ("declared", "observation", "amendment"):
            continue
        # A record naming a single role only promises that role's checkpoint. Checking both would
        # report the deliberately-absent sibling of every Stage 2 single-role rung as missing.
        for role in ([e["role"]] if e.get("role") else ["hyper", "static"]):
            key = (e["rung_dir"], e["seed"], int(e["steps"]), role)
            if key not in on_disk:
                missing.append({"ledger": e["ledger"], "lineno": e["lineno"], "codec": e["codec"],
                                "phase": e["phase"], "status": e["status"], "role": role,
                                "rung_dir": e["rung_dir"], "seed": e["seed"], "steps": e["steps"],
                                "expected": f"{e['rung_dir']}/{role}/s{e['seed']}/snapshots/step{e['steps']}.pt"})

    scoped = [r for r in index if r.get("in_scope")]
    reconciliation = {
        "totals": {
            "snapshots_on_disk": len([r for r in index if r.get("role")]),
            "in_scope": len(scoped),
            "in_scope_hyper": len([r for r in scoped if r["role"] == "hyper"]),
            "in_scope_static": len([r for r in scoped if r["role"] == "static"]),
            "ledger_records": len(ledger_entries),
        },
        "in_scope_by_codec": {
            c: {"hyper": len([r for r in scoped if r["codec"] == c and r["role"] == "hyper"]),
                "static": len([r for r in scoped if r["codec"] == c and r["role"] == "static"])}
            for c in audited
        },
        "excluded_by_reason": _count_reasons(index),
        "ledger_entries_with_missing_checkpoint": missing,
        "in_scope_checkpoints_with_no_ledger_entry": unledgered,
        "duplicate_complete_ledger_records": duplicates,
        "renamed_role_directories": sorted({r["role_dir"] for r in scoped if r["role_dir"] != r["role"]}),
        # A scoped checkpoint with no resolvable scale cannot be re-instantiated correctly, so the
        # re-scoring pass must refuse it rather than fall back to a codec default.
        "in_scope_checkpoints_with_unresolved_scale": [
            r["checkpoint"] for r in scoped if r.get("scale") is None
        ],
        "rungs_missing_a_declared_role": [u for u in _unpaired(scoped) if u["roles_missing"]],
        "single_role_rungs_by_design": len([u for u in _unpaired(scoped) if u["single_role_by_design"]]),
    }
    return index, reconciliation


def _unpaired(scoped: list[dict]) -> list[dict]:
    """Rungs missing a role, split into "by design" and "actually missing".

    Under the corrected protocol the static is selected independently, so `t2a_codec_trial.sh`'s
    `ROLES` mode trains ONE role per rung and a single-role directory is normal — the two roles of a
    reported pair no longer share a rung. That made the original "every rung needs both roles" check
    fire on all 20 Stage 2 rungs at once, which is noise, not signal.

    The real defect is a rung whose LEDGER declares a role that has no checkpoint. A rung whose
    ledger declares exactly the role present is complete, whatever its sibling directory contains.
    """
    roles: dict[tuple, set] = {}
    declared: dict[tuple, set] = {}
    for r in scoped:
        key = (r["codec"], r["rung_dir"], r["seed"], r["steps"])
        roles.setdefault(key, set()).add(r["role"])
        # `ledger_role` is set when the matched ledger record names a single role (Stage 2 onward).
        if r.get("ledger_role"):
            declared.setdefault(key, set()).add(r["ledger_role"])

    out = []
    for k, present in sorted(roles.items()):
        if len(present) == 2:
            continue
        want = declared.get(k) or {"hyper", "static"}
        missing = sorted(want - present)
        out.append({"codec": k[0], "rung_dir": k[1], "seed": k[2], "steps": k[3],
                    "roles_present": sorted(present), "roles_declared": sorted(want),
                    "roles_missing": missing,
                    "single_role_by_design": not missing})
    return out


def _count_reasons(index: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in index:
        if not r.get("in_scope") and r.get("scope_reason"):
            counts[r["scope_reason"]] = counts.get(r["scope_reason"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/autoresearch/t2a/evaluation")
    p.add_argument("--hashes", action="store_true", help="sha256 every snapshot (slow: ~GBs of reads)")
    p.add_argument("--codecs", default=",".join(AUDITED_CODECS),
                   help="codecs to bring in scope. Add `steering` to include the fifth codec, whose "
                        "T2A search stopped at a declaration and has no canonical row to compare against.")
    args = p.parse_args()

    index, reconciliation = build(args.hashes, tuple(c for c in args.codecs.split(",") if c))
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "checkpoint_index.jsonl").open("w") as f:
        for row in index:
            f.write(json.dumps(row) + "\n")
    (out / "index_reconciliation.json").write_text(json.dumps(reconciliation, indent=2) + "\n")

    print(json.dumps(reconciliation, indent=2))
    print(f"\nwrote {len(index)} rows to {out / 'checkpoint_index.jsonl'}")


if __name__ == "__main__":
    main()
