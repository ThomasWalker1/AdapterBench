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
    if len({record["setting"] for record in records}) != len(records):
        raise ResultValidationError("canonical settings must be unique")
    return records


def _by_setting(records: list[dict[str, Any]], setting: str) -> dict[str, Any]:
    return next(record for record in records if record["setting"] == setting)


def _signed(value: float, decimals: int) -> str:
    return f"{value:+.{decimals}f}".replace("-", "−")


def render_fragment(records: list[dict[str, Any]], fragment: str) -> str:
    t2l, d2l = _by_setting(records, "T2L"), _by_setting(records, "D2L")
    th, dh = t2l["headline"], d2l["headline"]
    if fragment == "t2l-markdown":
        return "\n".join([
            "| Shape | rank | lr | steps | seeds | **matched − static (CE, nats)** | matched − frozen | accuracy m−static / m−frozen |",
            "|---|:---:|:---:|---:|:---:|:---:|:---:|:---:|",
            f"| LoRA | 8 | 1e-4 | 20 000 | 3 | **{_signed(th['value'], 3)} ± {th['variation']:.3f}** ({t2l['summary']['wins']}) | {_signed(t2l['summary']['matched_minus_frozen']['value'], 2)} ± {t2l['summary']['matched_minus_frozen']['variation']:.2f} | {_signed(t2l['summary']['accuracy_matched_minus_static']['value'], 4)} ± {t2l['summary']['accuracy_matched_minus_static']['variation']:.4f} / {_signed(t2l['summary']['accuracy_matched_minus_frozen'], 3)} |",
        ])
    if fragment == "d2l-markdown":
        axes = " | ".join(str(point["axis"]) for point in d2l["difficulty_curve"])
        values = " | ".join(f"{point['delta']:.3f}" for point in d2l["difficulty_curve"])
        return "\n".join([
            "| Shape | scale | lr | steps | seeds | accuracy | ctxswap | **matched − control** |",
            "|---|:---:|---:|---:|:---:|---:|---:|:---:|",
            f"| LoRA (r=8) | {d2l['free_hyperparameters']['scale']} | 4e-5 | 12 000 | 5 | {dh['matched']:.3f} ± {dh['variation']:.3f} | {dh['control']:.3f} | **{_signed(dh['value'], 3)} ± {dh['variation']:.3f}** |",
            "",
            f"| eval len | {axes} |",
            "|---|" + "---|" * len(d2l["difficulty_curve"]),
            f"| matched − control | {values} |",
        ])
    if fragment == "release-summary-markdown":
        return "\n".join([
            "| setting | frozen interpreter | primary result | condition control |",
            "|---|---|---|---|",
            f"| T2L | gemma-2-2b | `matched − static = {_signed(th['value'], 3)} ± {th['variation']:.3f}` nats CE over 3 seeds | matched beats a same-shape static multi-task LoRA on {t2l['summary']['wins']} |",
            f"| D2L | Qwen3-0.6B | `matched − context-swap = {_signed(dh['value'], 3)} ± {dh['variation']:.3f}` exact-match over 5 seeds | wrong-document adapters score `{dh['control']:.3f}` |",
        ])
    if fragment == "repro-summary-markdown":
        return "\n".join([
            "| Setting | Script | Leaderboard | Headline |",
            "|---|---|---|---|",
            f"| Task (T2L) | `task_t2l_lora_all.sh [GPUS_HYPER] [GPUS_STATIC]` | `task_conditioned_t2l.md` | matched − static = **{_signed(th['value'], 3)} ± {th['variation']:.3f}** nats CE ({t2l['summary']['wins']}, 3 seeds) |",
            f"| Document (NIAH) | `document_niah_lora.sh [DEVICE]` | `document_niah_d2l.md` | matched − ctxswap = **{_signed(dh['value'], 3)} ± {dh['variation']:.3f}** (5 seeds, realistic haystack; crossover 16×) |",
        ])
    if fragment == "t2l-html":
        frozen = t2l["summary"]["matched_minus_frozen"]
        accuracy = t2l["summary"]["accuracy_matched_minus_static"]
        return (
            '<tr><td>LoRA <span class="baseline-badge">baseline</span></td><td>8</td>'
            f'<td>1e-4</td><td>20 000</td><td>3</td><td class="headline">{_signed(th["value"], 3)} ± {th["variation"]:.3f} '
            f'<span class="muted">({t2l["summary"]["wins"]})</span></td><td>{_signed(frozen["value"], 2)} ± {frozen["variation"]:.2f}</td>'
            f'<td>{_signed(accuracy["value"], 4)} ± {accuracy["variation"]:.4f} / {_signed(t2l["summary"]["accuracy_matched_minus_frozen"], 3)}</td></tr>'
        )
    if fragment == "d2l-html":
        return f'<tr><td>LoRA <span class="baseline-badge">baseline</span></td><td>{d2l["free_hyperparameters"]["scale"]}</td><td>4e-5</td><td>12 000</td><td>5</td><td>{dh["matched"]:.3f} ± {dh["variation"]:.3f}</td><td>{dh["control"]:.3f}</td><td class="headline">{_signed(dh["value"], 3)} ± {dh["variation"]:.3f}</td></tr>'
    if fragment == "paper-results":
        return "\n".join([
            f"T2L & rank 8, lr $10^{{-4}}$, 20k steps, 3 seeds & held-out answer CE & ${th['value']:.3f}\\pm{th['variation']:.3f}$ nats matched $-$ static; {t2l['summary']['wins']} " + r"\\",
            f"D2L & rank 8, scale {d2l['free_hyperparameters']['scale']}, lr $4{{\\times}}10^{{-5}}$, 12k steps, 5 seeds & {dh['matched']:.3f} / {dh['control']:.3f} accuracy & ${dh['value']:+.3f}\\pm{dh['variation']:.3f}$ matched $-$ context-swap " + r"\\",
        ])
    raise ResultValidationError(f"unknown fragment: {fragment}")


def check_rendered_documents(records: list[dict[str, Any]], root: Path = REPO_ROOT) -> list[str]:
    targets = [
        (root / "leaderboards/task_conditioned_t2l.md", "t2l-markdown"),
        (root / "leaderboards/document_niah_d2l.md", "d2l-markdown"),
        (root / "PROJECT_PLAN.md", "release-summary-markdown"),
        (root / "scripts/reproduce/README.md", "repro-summary-markdown"),
        (root / "docs/index.html", "t2l-html"),
        (root / "docs/index.html", "d2l-html"),
        (root.parent / "adapterbench-paper.tex", "paper-results"),
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
