"""Canonical benchmark-result records, validation, and deterministic table rendering."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "canonical_results"
SCHEMA_VERSION = 1


class ResultValidationError(ValueError):
    pass


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ResultValidationError(f"{name} must be a finite number")
    return float(value)


def _variation(values: list[float], kind: str) -> float:
    if kind not in {"sample_standard_deviation", "population_standard_deviation"}:
        raise ResultValidationError(f"unsupported variation kind: {kind}")
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    divisor = len(values) - 1 if kind == "sample_standard_deviation" else len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / divisor)


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version", "setting", "codec", "metric", "headline", "seed_results",
        "difficulty_curve", "free_hyperparameters", "fixed_shape_parameters",
        "provenance", "reproduction",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ResultValidationError(f"missing fields: {', '.join(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ResultValidationError(f"unsupported schema_version: {record['schema_version']}")
    headline = record["headline"]
    needed_headline = {"comparison", "matched", "control", "value", "variation", "variation_kind", "unit", "direction"}
    if not isinstance(headline, dict) or needed_headline - set(headline):
        raise ResultValidationError("headline is incomplete")
    matched, control, value = (_number(headline[field], f"headline.{field}") for field in ("matched", "control", "value"))
    if headline["comparison"] != "matched_minus_control" or not math.isclose(value, matched - control, abs_tol=1e-12):
        raise ResultValidationError("headline must be matched minus control")
    seeds = record["seed_results"]
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ResultValidationError("seed_results must contain at least three seeds")
    deltas = []
    for seed in seeds:
        if not isinstance(seed, dict) or {"seed", "matched", "control", "delta"} - set(seed):
            raise ResultValidationError("every seed result needs seed, matched, control, and delta")
        sm, sc, sd = (_number(seed[field], f"seed.{field}") for field in ("matched", "control", "delta"))
        if not math.isclose(sd, sm - sc, abs_tol=1e-12):
            raise ResultValidationError(f"seed {seed['seed']} is not matched minus control")
        deltas.append(sd)
    if not math.isclose(value, sum(deltas) / len(deltas), abs_tol=1e-12):
        raise ResultValidationError("headline is not the mean seed delta")
    variation = _variation(deltas, headline["variation_kind"])
    if not math.isclose(_number(headline["variation"], "headline.variation"), variation, abs_tol=1e-12):
        raise ResultValidationError("headline variation does not match the seed deltas")
    curve = record["difficulty_curve"]
    if not isinstance(curve, list) or not curve:
        raise ResultValidationError("difficulty_curve must be non-empty")
    for point in curve:
        if not isinstance(point, dict) or {"axis", "matched", "control", "delta"} - set(point):
            raise ResultValidationError("difficulty point is incomplete")
        if not math.isclose(_number(point["delta"], "curve.delta"), _number(point["matched"], "curve.matched") - _number(point["control"], "curve.control"), abs_tol=1e-12):
            raise ResultValidationError("difficulty delta is not matched minus control")
    provenance = record["provenance"]
    if not isinstance(provenance, dict) or not provenance.get("models") or not provenance.get("data"):
        raise ResultValidationError("provenance needs models and data")
    artifacts = provenance.get("source_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) < len(seeds):
        raise ResultValidationError("provenance needs at least one source artifact per seed")
    for artifact in artifacts:
        path, digest = artifact.get("path"), artifact.get("sha256")
        if not isinstance(path, str) or path.startswith("/") or ".." in Path(path).parts:
            raise ResultValidationError("artifact paths must be repository-relative")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ResultValidationError("artifact sha256 must be a SHA-256 digest")
    trail = record.get("selection_trail")
    if trail is not None:
        if not isinstance(trail, dict) or {"protocol", "state_artifacts", "summary"} - set(trail):
            raise ResultValidationError("selection_trail is incomplete")
        if not isinstance(trail["protocol"], str) or not isinstance(trail["summary"], str):
            raise ResultValidationError("selection_trail protocol and summary must be strings")
        state_artifacts = trail["state_artifacts"]
        if not isinstance(state_artifacts, list) or not state_artifacts:
            raise ResultValidationError("selection_trail needs state artifacts")
        for artifact in state_artifacts:
            if not isinstance(artifact, dict):
                raise ResultValidationError("selection_trail artifacts must be objects")
            path, digest = artifact.get("path"), artifact.get("sha256")
            if not isinstance(path, str) or path.startswith("/") or ".." in Path(path).parts:
                raise ResultValidationError("selection_trail paths must be repository-relative")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ResultValidationError("selection_trail sha256 must be a SHA-256 digest")


def load_records(root: Path = DEFAULT_RESULTS_ROOT) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("schema-"):
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ResultValidationError(f"invalid JSON in {path}: {error}") from error
        if not isinstance(record, dict):
            raise ResultValidationError(f"record must be an object: {path}")
        validate_record(record)
        records.append(record)
    if not records:
        raise ResultValidationError(f"no records under {root}")
    identities = {(record["setting"], record["codec"]) for record in records}
    if len(identities) != len(records):
        raise ResultValidationError("canonical (setting, codec) pairs must be unique")
    return records


def _by_setting(records: list[dict[str, Any]], setting: str) -> list[dict[str, Any]]:
    return [record for record in records if record["setting"] == setting]


def _t2a_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the validated LoRA reference first, then sort explored codecs by name."""
    selected = _by_setting(records, "T2A")
    if not selected:
        raise ResultValidationError("no canonical T2A records")
    return sorted(selected, key=lambda record: (record["codec"] != "lora_r8", record["codec"]))


def _d2a_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the validated LoRA reference first, then sort explored codecs by name."""
    return sorted(_by_setting(records, "D2A"), key=lambda record: (record["codec"] != "lora_r8", record["codec"]))


def _display_name(record: dict[str, Any]) -> str:
    return str(record.get("display_name", record["codec"]))


def _t2a_display_name(record: dict[str, Any]) -> str:
    return "LoRA" if record["codec"] == "lora_r8" else _display_name(record)


def _signed(value: float, decimals: int) -> str:
    return f"{value:+.{decimals}f}".replace("-", "−")


def render_fragment(records: list[dict[str, Any]], fragment: str) -> str:
    t2a_records, d2a_records = _t2a_records(records), _d2a_records(records)
    if not d2a_records:
        raise ResultValidationError("no canonical D2A records")
    if fragment == "t2a-markdown":
        lines = [
            "| Shape | rank | scale | lr | steps | static\\* scale / lr | seeds | matched | static\\* | "
            "**matched − static\\* (ROUGE-L)** | m − frozen | EM Δ | CE Δ (appendix) |",
            "|---|:---:|:---:|:---:|---:|:---:|:---:|---:|---:|:---:|---:|---:|---:|",
        ]
        for t2a in t2a_records:
            th, hp = t2a["headline"], t2a["free_hyperparameters"]
            rank = t2a["fixed_shape_parameters"].get("rank", "—")
            steps = f"{int(hp['steps']):,}".replace(",", " ")
            summary = t2a["summary"]
            static = hp.get("static_control", {})
            em = summary["exact_match_matched_minus_static"]
            ce = summary["cross_entropy_matched_minus_static"]
            lines.append(
                f"| {_t2a_display_name(t2a)} | {rank} | {hp['scale']} | {hp['learning_rate']} | {steps} | "
                f"{static.get('scale', '—')} / {static.get('learning_rate', '—')} | "
                f"{len(t2a['seed_results'])} | {th['matched']:.3f} | {th['control']:.3f} | "
                f"**{_signed(th['value'], 3)} ± {th['variation']:.3f}** | "
                f"{_signed(summary['matched_minus_frozen']['value'], 3)} | "
                f"{_signed(em['value'], 3)} | {_signed(ce['value'], 3)} |"
            )
        return "\n".join(lines)
    if fragment == "t2a-selection-markdown":
        lines = [
            "| Shape | compact audit trail | selected final configuration |",
            "|---|---|---|",
        ]
        for t2a in t2a_records:
            trail = t2a.get("selection_trail")
            if trail is None:
                lines.append(f"| {_t2a_display_name(t2a)} | not yet backfilled | — |")
                continue
            hp = t2a["free_hyperparameters"]
            ledgers = ", ".join(f"`{Path(item['path']).parent.name}/{Path(item['path']).name}`" for item in trail["state_artifacts"])
            static = hp.get("static_control", {})
            selected = (f"hyper: scale {hp['scale']}, lr {hp['learning_rate']}, {hp['steps']:,} steps; "
                        f"static\\*: scale {static.get('scale', '—')}, lr {static.get('learning_rate', '—')}")
            lines.append(f"| {_t2a_display_name(t2a)} | {trail['summary']} State ledgers: {ledgers}. | {selected} |")
        return "\n".join(lines)
    if fragment == "d2a-markdown":
        lines = [
            "| Shape | scale | lr | steps | seeds | accuracy | ctxswap | **matched − control** |",
            "|---|:---:|---:|---:|:---:|---:|---:|:---:|",
        ]
        for d2a in d2a_records:
            dh, hp = d2a["headline"], d2a["free_hyperparameters"]
            lines.append(
                f"| {_display_name(d2a)} | {hp['scale']} | {hp['learning_rate']} | "
                f"{hp['steps']:,} | {len(d2a['seed_results'])} | {dh['matched']:.3f} ± {dh['variation']:.3f} | "
                f"{dh['control']:.3f} | **{_signed(dh['value'], 3)} ± {dh['variation']:.3f}** |"
            )
        axes = " | ".join(str(point["axis"]) for point in d2a_records[0]["difficulty_curve"])
        lines.extend(["", f"| eval len | {axes} |", "|---|" + "---|" * len(d2a_records[0]["difficulty_curve"])])
        for d2a in d2a_records:
            values = " | ".join(f"{point['delta']:.3f}" for point in d2a["difficulty_curve"])
            lines.append(f"| matched − control ({_display_name(d2a)}) | {values} |")
        trails = [record for record in d2a_records if record.get("selection_trail")]
        if trails:
            lines.extend(["", "**Selection trail.**"])
            for d2a in trails:
                trail = d2a["selection_trail"]
                ledgers = ", ".join(f"`{item['path']}`" for item in trail["state_artifacts"])
                lines.append(f"- **{_display_name(d2a)}** — {trail['summary']} State ledger: {ledgers}.")
        return "\n".join(lines)
    if fragment == "release-summary-markdown":
        t2a_summary = "; ".join(
            f"{_t2a_display_name(t2a)} `matched − static* = {_signed(t2a['headline']['value'], 3)} ± {t2a['headline']['variation']:.3f}`"
            for t2a in t2a_records
        )
        t2a_control = "; ".join(
            f"{_t2a_display_name(t2a)} wins {t2a['summary']['wins']}" for t2a in t2a_records
        )
        d2a_summary = "; ".join(
            f"{_display_name(d2a)}: `matched − context-swap = {_signed(d2a['headline']['value'], 3)} ± {d2a['headline']['variation']:.3f}`"
            for d2a in d2a_records
        )
        d2a_control = "; ".join(
            f"{_display_name(d2a)} control `{d2a['headline']['control']:.3f}`" for d2a in d2a_records
        )
        return "\n".join([
            "| setting | frozen interpreter | primary result | condition control |",
            "|---|---|---|---|",
            f"| T2A | gemma-2-2b | {t2a_summary} ROUGE-L on 11 held-out SNI tasks (3 confirmation seeds each) | "
            f"independently selected same-shape static control: {t2a_control} |",
            f"| D2A | Qwen3-0.6B | {d2a_summary} exact-match | {d2a_control} |",
        ])
    if fragment == "repro-summary-markdown":
        lines = [
            "| Setting | Script | Leaderboard | Headline |",
            "|---|---|---|---|",
        ]
        for t2a in t2a_records:
            th = t2a["headline"]
            lines.append(
                f"| Task (T2A) — {_t2a_display_name(t2a)} | `{t2a['reproduction']['script']}` | "
                f"`task_conditioned_t2a.md` | matched − static\\* = **{_signed(th['value'], 3)} ± {th['variation']:.3f}** "
                f"ROUGE-L ({t2a['summary']['wins']}, {len(t2a['seed_results'])} confirmation seeds) |"
            )
        for d2a in d2a_records:
            dh = d2a["headline"]
            decoys = d2a.get("summary", {}).get("numeric_decoy_count")
            setting_label = (
                f"realistic-prose, {decoys} numeric decoys" if decoys is not None else "realistic haystack"
            )
            lines.append(
                f"| Document (NIAH) — {_display_name(d2a)} | `{d2a['reproduction']['script']} [DEVICE]` | "
                f"`document_niah_d2a.md` | matched − ctxswap = **{_signed(dh['value'], 3)} ± {dh['variation']:.3f}** "
                f"({len(d2a['seed_results'])} seeds, {setting_label}; crossover {d2a['summary']['crossover_length'] / d2a['summary']['training_length']:.0f}×) |"
            )
        return "\n".join(lines)
    if fragment == "t2a-html":
        rows = []
        for t2a in t2a_records:
            th, hp = t2a["headline"], t2a["free_hyperparameters"]
            summary = t2a["summary"]
            static = hp.get("static_control", {})
            em = summary["exact_match_matched_minus_static"]
            ce = summary["cross_entropy_matched_minus_static"]
            label = _t2a_display_name(t2a)
            if t2a["codec"] == "lora_r8":
                label += ' <span class="baseline-badge">baseline</span>'
            rank = t2a["fixed_shape_parameters"].get("rank", "—")
            steps = f"{int(hp['steps']):,}".replace(",", " ")
            rows.append(
                f'<tr><td>{label}</td><td>{rank}</td><td>{hp["scale"]}</td><td>{hp["learning_rate"]}</td>'
                f'<td>{steps}</td><td>{static.get("scale", "—")} / {static.get("learning_rate", "—")}</td>'
                f'<td>{len(t2a["seed_results"])}</td><td>{th["matched"]:.3f}</td><td>{th["control"]:.3f}</td>'
                f'<td class="headline">{_signed(th["value"], 3)} ± {th["variation"]:.3f}</td>'
                f'<td>{_signed(summary["matched_minus_frozen"]["value"], 3)}</td>'
                f'<td>{_signed(em["value"], 3)}</td><td class="muted">{_signed(ce["value"], 3)}</td></tr>'
            )
        return "\n".join(rows)
    if fragment == "d2a-html":
        rows = []
        for d2a in d2a_records:
            dh, hp = d2a["headline"], d2a["free_hyperparameters"]
            label = _display_name(d2a)
            if d2a["codec"] == "lora_r8":
                label += ' <span class="baseline-badge">baseline</span>'
            rows.append(
                f'<tr><td>{label}</td><td>{hp["scale"]}</td><td>{hp["learning_rate"]}</td><td>{hp["steps"]:,}</td>'
                f'<td>{len(d2a["seed_results"])}</td><td>{dh["matched"]:.3f} ± {dh["variation"]:.3f}</td>'
                f'<td>{dh["control"]:.3f}</td><td class="headline">{_signed(dh["value"], 3)} ± {dh["variation"]:.3f}</td></tr>'
            )
        return "\n".join(rows)
    raise ResultValidationError(f"unknown fragment: {fragment}")


RENDER_TARGETS = [
    ("leaderboards/task_conditioned_t2a.md", "t2a-markdown"),
    ("leaderboards/task_conditioned_t2a.md", "t2a-selection-markdown"),
    ("leaderboards/document_niah_d2a.md", "d2a-markdown"),
    ("PROJECT_PLAN.md", "release-summary-markdown"),
    ("scripts/reproduce/README.md", "repro-summary-markdown"),
    ("docs/index.html", "t2a-html"),
    ("docs/index.html", "d2a-html"),
]


def write_rendered_documents(records: list[dict[str, Any]], root: Path = REPO_ROOT) -> list[str]:
    """Rewrite every `canonical-results:<fragment>` marker block in place from the records.

    The counterpart to `check_rendered_documents`: that function reports drift, this one removes it.
    Only the text BETWEEN the markers is replaced, so the surrounding prose -- which does not
    regenerate and has to be maintained by hand -- is never touched.
    """
    written = []
    for rel, fragment in RENDER_TARGETS:
        path = root / rel
        text = path.read_text()
        start, end = f"<!-- canonical-results:{fragment}:start -->", f"<!-- canonical-results:{fragment}:end -->"
        if start not in text or end not in text:
            raise ResultValidationError(f"{path}: missing canonical-results markers for {fragment}")
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        body = render_fragment(records, fragment)
        new = f"{head}{start}\n{body}\n{end}{tail}"
        if new != text:
            path.write_text(new)
            written.append(f"{rel}: {fragment}")
    return written


def check_rendered_documents(records: list[dict[str, Any]], root: Path = REPO_ROOT) -> list[str]:
    targets = [
        (root / "leaderboards/task_conditioned_t2a.md", "t2a-markdown"),
        (root / "leaderboards/task_conditioned_t2a.md", "t2a-selection-markdown"),
        (root / "leaderboards/document_niah_d2a.md", "d2a-markdown"),
        (root / "PROJECT_PLAN.md", "release-summary-markdown"),
        (root / "scripts/reproduce/README.md", "repro-summary-markdown"),
        (root / "docs/index.html", "t2a-html"),
        (root / "docs/index.html", "d2a-html"),
    ]
    errors = []
    for path, fragment in targets:
        text = path.read_text()
        start, end = f"<!-- canonical-results:{fragment}:start -->", f"<!-- canonical-results:{fragment}:end -->"
        if start not in text or end not in text:
            errors.append(f"{path}: missing canonical-results markers for {fragment}")
            continue
        actual = text.split(start, 1)[1].split(end, 1)[0].strip()
        if actual != render_fragment(records, fragment).strip():
            errors.append(f"{path}: {fragment} drifted from canonical results")
    return errors
