#!/usr/bin/env python3
"""Generate the AdapterBench static website from canonical results and manifests.

Usage:
    .venv/bin/python scripts/build_website.py
    uv run adapterbench website build

Output lands in ``website/`` (``index.html`` plus one page per codec under
``website/codecs/``). The styling matches the Thomas Walker personal-site /
PAARBench benchmark pages (IBM Plex, dark header, sortable tables).

After updating ``canonical_results/``, run this script and commit the generated
HTML so the site stays in sync with leaderboard numbers.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"
CONFIGS = ROOT / "configs" / "adapters"
CANONICAL = ROOT / "canonical_results"

sys.path.insert(0, str(ROOT / "src"))
from adapterbench.results import (  # noqa: E402
    _d2a_records,
    _display_name,
    _t2a_display_name,
    _t2a_records,
    load_records,
    render_fragment,
)

GITHUB = "https://github.com/ThomasWalker1/AdapterBench"
GITHUB_BLOB = f"{GITHUB}/blob/main"

CODEC_SLUG = {
    "lora_r8": "lora",
    "ia3": "ia3",
    "lokr": "lokr",
    "fourierft": "fourierft",
    "steering": "steering",
}

CODEC_FAMILY_BLURB = {
    "lora": "Low-rank A/B factorization — the validated baseline representation.",
    "ia3": "Elementwise scaling W ↦ diag(1+v) W — symmetry-free, tiny budget.",
    "lokr": "Kronecker factorization ΔW = B ⊗ A — full-rank reach from few scalars.",
    "fourierft": "Sparse Fourier coefficients in a fixed global basis.",
    "steering": "Activation-space residual-stream vector h ↦ h + s·v — no weight edit.",
}


def esc(text: str | None) -> str:
    return html.escape(text or "", quote=True)


def codec_slug(record: dict) -> str:
    return CODEC_SLUG.get(record["codec"], record["codec"].replace("_r8", ""))


def load_manifest(slug: str) -> dict:
    for path in CONFIGS.glob("*.yaml"):
        if path.stem.replace("_r8", "") == slug or path.stem == slug:
            text = path.read_text()
            data: dict = {}
            for line in text.splitlines():
                if not line.strip() or line.strip().startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if value.startswith("[") and value.endswith("]"):
                    inner = value[1:-1].strip()
                    data[key] = [v.strip() for v in inner.split(",") if v.strip()] if inner else []
                else:
                    data[key] = value.strip('"')
            return data
    return {}


def link_codec_rows(table_html: str, records: list[dict], display_fn) -> str:
    for record in records:
        name = display_fn(record)
        slug = codec_slug(record)
        badge = ' <span class="baseline-badge">baseline</span>'
        if badge in table_html and f"<td>{name}{badge}" in table_html:
            table_html = table_html.replace(
                f"<td>{name}{badge}",
                f'<td><a class="codec-link" href="codecs/{slug}.html">{esc(name)}</a>{badge}',
                1,
            )
        elif f"<td>{name}</td>" in table_html:
            table_html = table_html.replace(
                f"<td>{name}</td>",
                f'<td><a class="codec-link" href="codecs/{slug}.html">{esc(name)}</a></td>',
                1,
            )
    return table_html


def page_shell(title: str, body: str, *, depth: int = 0) -> str:
    prefix = "../" * depth
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)} — AdapterBench</title>
  <meta name="description" content="AdapterBench tests whether the shape of a hypernetwork-generated PEFT adapter matters under live end-to-end SFT.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="{prefix}css/style.css">
</head>
<body>
  <header class="site-header">
    <div class="site-header-inner">
      <h1><a href="{prefix}index.html" style="color:inherit;text-decoration:none;">AdapterBench</a></h1>
      <p class="tagline">Does the <em>shape</em> of a hypernetwork-generated adapter matter?</p>
      <nav class="site-nav">
        <a href="{prefix}index.html#about">About</a>
        <a href="{prefix}index.html#settings">Settings</a>
        <a href="{prefix}index.html#leaderboards">Leaderboards</a>
        <a href="{prefix}index.html#contribute">Contribute</a>
        <a href="{GITHUB}">GitHub</a>
      </nav>
    </div>
  </header>
  <main class="page-main">
    {body}
  </main>
  <footer class="site-footer">
    <div class="site-footer-inner">
      <p>AdapterBench — holding the hypernetwork, training loop, data, and evaluator fixed; varying only the generated representation.</p>
    </div>
  </footer>
  <script src="{prefix}js/sort-tables.js" defer></script>
</body>
</html>
"""


def seed_table(seed_results: list[dict], unit: str) -> str:
    rows = []
    for row in seed_results:
        rows.append(
            f"<tr><td>{row['seed']}</td><td class=\"num\">{row['matched']:.4f}</td>"
            f"<td class=\"num\">{row['control']:.4f}</td><td class=\"num headline\">{row['delta']:+.4f}</td></tr>"
        )
    return (
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Seed</th><th class=\"num\">Matched</th><th class=\"num\">Control</th>"
        f"<th class=\"num\">Δ ({esc(unit)})</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def difficulty_table(curve: list[dict], title: str) -> str:
    if not curve:
        return "<p class=\"muted\">No difficulty curve recorded.</p>"
    rows = []
    for point in curve:
        axis = point.get("axis", point.get("note", "—"))
        rows.append(
            f"<tr><td>{esc(str(axis))}</td><td class=\"num\">{point['matched']:.4f}</td>"
            f"<td class=\"num\">{point['control']:.4f}</td><td class=\"num\">{point['delta']:+.4f}</td></tr>"
        )
    return (
        f"<h3>{esc(title)}</h3>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Axis</th><th class=\"num\">Matched</th><th class=\"num\">Control</th><th class=\"num\">Δ</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_codec_page(slug: str, setting_records: list[dict]) -> str:
    manifest = load_manifest(slug)
    family = manifest.get("family", slug)
    blurb = CODEC_FAMILY_BLURB.get(slug, manifest.get("output_structure", ""))
    display = _t2a_display_name(setting_records[0]) if setting_records[0]["setting"] == "T2A" else _display_name(setting_records[0])
    for record in setting_records:
        if record["codec"] == "lora_r8":
            display = _t2a_display_name(record) if record["setting"] == "T2A" else _display_name(record)
            break
        display = _t2a_display_name(record) if record["setting"] == "T2A" else _display_name(record)

    is_baseline = any(r["codec"] == "lora_r8" for r in setting_records)
    badges = [f'<span class="badge">{esc(family)}</span>']
    for record in setting_records:
        badges.append(f'<span class="badge badge-accent">{esc(record["setting"])}</span>')
    if is_baseline:
        badges.append('<span class="badge badge-accent">baseline</span>')
    if manifest.get("implementation"):
        badges.append(f'<span class="badge">{esc(str(manifest["implementation"]))}</span>')

    setting_sections = []
    for record in setting_records:
        setting = record["setting"]
        hp = record["free_hyperparameters"]
        headline = record["headline"]
        repro = record.get("reproduction", {})
        unit = headline.get("unit", "score").replace("_", " ")
        h_sign = "+" if headline.get("direction") == "higher_is_better" else "−"

        params = []
        for key, value in hp.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    params.append(f"<li><span>{esc(subkey)}</span> = {esc(str(subval))}</li>")
            else:
                params.append(f"<li><span>{esc(key)}</span> = {esc(str(value))}</li>")
        fixed = record.get("fixed_shape_parameters", {})
        for key, value in fixed.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            params.append(f"<li><span>{esc(key)} (fixed)</span> = {esc(str(value))}</li>")

        repro_cmd = repro.get("script") or repro.get("full") or repro.get("confirmation", "")
        repro_block = f"<pre><code>{esc(repro_cmd)}</code></pre>" if repro_cmd else ""

        trail = record.get("selection_trail")
        trail_html = ""
        if trail:
            trail_html = f"<p class=\"muted\">{esc(trail.get('summary', ''))}</p>"

        setting_sections.append(f"""
<section>
  <h2>{esc(setting)} headline</h2>
  <p class="lead"><strong>{esc(headline.get('comparison', 'matched_minus_control').replace('_', ' '))}</strong>
  vs {esc(headline.get('control_name', 'control').replace('_', ' '))}:
  <span class="headline">{headline['value']:+.4f} {h_sign} {headline['variation']:.4f}</span>
  ({len(record['seed_results'])} seeds).</p>
  {seed_table(record['seed_results'], unit)}
  <h3>Hyperparameters</h3>
  <ul class="param-list">{''.join(params)}</ul>
  {difficulty_table(record.get('difficulty_curve', []), 'Difficulty curve')}
  {trail_html}
  <h3>Reproduce</h3>
  {repro_block}
</section>
""")

    body = f"""
<a class="back-link" href="../index.html#leaderboards">← Back to leaderboards</a>

<div class="codec-hero">
  <h1>{esc(display.split('(')[0].strip() if '(' in display else display)}</h1>
  <p class="lead">{esc(blurb)}</p>
  <div class="badge-row">{''.join(badges)}</div>
</div>

{''.join(setting_sections)}

<section>
  <h2>Source</h2>
  <div class="prose">
    <p>Implementation: <a href="{GITHUB_BLOB}/src/adapterbench/t2a/codecs.py">codecs.py</a>
    · manifest: <a href="{GITHUB_BLOB}/configs/adapters/{esc(manifest.get('name', slug))}.yaml">{esc(manifest.get('name', slug))}.yaml</a>
    · contribute: <a href="{GITHUB_BLOB}/CONTRIBUTING.md">CONTRIBUTING.md</a></p>
  </div>
</section>
"""
    title = display.split("(")[0].strip() if "(" in display else display
    return page_shell(title, body, depth=1)


def render_index(records: list[dict]) -> str:
    t2a = _t2a_records(records)
    d2a = _d2a_records(records)
    t2a_rows = link_codec_rows(render_fragment(records, "t2a-html"), t2a, _t2a_display_name)
    d2a_rows = link_codec_rows(render_fragment(records, "d2a-html"), d2a, _display_name)

    body = f"""
<section id="about">
  <h2>About the benchmark</h2>
  <div class="about-copy">
    <p class="lead">
      <strong>AdapterBench</strong> asks whether the <em>shape</em> of a hypernetwork-generated
      PEFT adapter matters. Text-to-LoRA and Doc-to-LoRA both emit a LoRA — but neither tests that
      representation against alternatives under a fixed generation-and-evaluation protocol.
    </p>
    <p>
      The benchmark holds the hypernetwork shell, training loop, data, and evaluator
      <strong>constant</strong> within each setting, and varies <strong>only</strong> the generated
      representation (the <em>codec</em>). Any difference in the scored result is therefore attributable
      to shape, not to a confound in the recipe.
    </p>
    <div class="premise">
      <div class="fixed-vary">
        <div class="fv"><h4>Fixed substrate</h4><ul>
          <li>hypernetwork trunk &amp; conditioner</li>
          <li>training data &amp; optimizer loop</li>
          <li>evaluator &amp; behavioral control</li>
        </ul></div>
        <div class="fv vary"><h4>Varied — the codec</h4><ul>
          <li>how many numbers the net emits</li>
          <li>how they become a weight or activation update</li>
          <li>LoRA, (IA)³, LoKr, FourierFT, steering, …</li>
        </ul></div>
      </div>
    </div>
    <p class="muted" style="margin-top:1rem">
      Every headline is <code>matched − control</code> on a behavioral metric, never raw training loss.
      Scale is swept per codec; ≥3 seeds are reported with spread. See
      <a href="{GITHUB_BLOB}/BENCHMARK_CONTRACT.md">BENCHMARK_CONTRACT.md</a>.
    </p>
  </div>
</section>

<section id="settings">
  <h2>Two active settings</h2>
  <p class="lead">Each fixes a frozen interpreter and differs only in what the hypernetwork is conditioned on and how the adapter is scored. Settings are never pooled.</p>
  <div class="card-grid">
    <div class="card">
      <span class="tag">T2A</span>
      <h3>Task-conditioned</h3>
      <div class="sub">Text-to-Adapter · gemma-2-2b</div>
      <dl>
        <dt>Conditioned on</dt><dd>a free-text task description (definition stripped from the input)</dd>
        <dt>Metric</dt><dd><strong>ROUGE-L</strong> on 11 held-out SNI tasks (primary); CE and exact match are appendix figures</dd>
        <dt>Control</dt><dd><code>matched − static*</code> — independently selected same-shape multi-task adapter</dd>
      </dl>
    </div>
    <div class="card">
      <span class="tag">D2A</span>
      <h3>Document-conditioned</h3>
      <div class="sub">Doc-to-Adapter · NIAH · Qwen3-0.6B</div>
      <dl>
        <dt>Conditioned on</dt><dd>cross-attention over interpreter activations for one document</dd>
        <dt>Metric</dt><dd>needle-in-a-haystack exact-match accuracy with numeric decoys</dd>
        <dt>Control</dt><dd><code>context-swap</code> — wrong-document adapter must fall to chance</dd>
      </dl>
    </div>
  </div>
</section>

<section id="leaderboards">
  <h2>Leaderboards</h2>
  <p class="lead">Click a codec name for hyperparameters, per-seed results, and reproduce commands. Column headers are sortable.</p>

  <div class="setting-block" id="t2a">
    <h3>T2A — task-conditioned</h3>
    <p class="setting-meta"><strong>matched − static*</strong> ROUGE-L on 11 genuinely held-out SNI tasks (3 confirmation seeds each). Higher is better.</p>
    <div class="table-wrap">
      <table class="sortable">
        <thead>
          <tr>
            <th data-type="text">Shape</th><th>rank</th><th>scale</th><th>lr</th><th>steps</th>
            <th>static* scale / lr</th><th class="num" data-type="number">seeds</th>
            <th class="num" data-type="number">matched</th><th class="num" data-type="number">static*</th>
            <th class="num" data-type="number">matched − static*</th>
            <th class="num" data-type="number">m − frozen</th>
            <th class="num" data-type="number">EM Δ</th><th class="num" data-type="number">CE Δ</th>
          </tr>
        </thead>
        <tbody>
{t2a_rows}
        </tbody>
      </table>
    </div>
    <div class="table-foot">Reproduce: <code>bash scripts/reproduce/t2a_reproduce_all.sh &lt;codec&gt;</code> — see <a href="{GITHUB_BLOB}/scripts/reproduce/README.md">reproduce README</a>.</div>
  </div>

  <div class="setting-block" id="d2a">
    <h3>D2A — document-conditioned (NIAH)</h3>
    <p class="setting-meta"><strong>matched − context-swap</strong> exact-match accuracy (5 seeds, 512-token training, numeric decoys). Higher is better.</p>
    <div class="table-wrap">
      <table class="sortable">
        <thead>
          <tr>
            <th data-type="text">Shape</th><th>scale</th><th>lr</th><th>steps</th>
            <th class="num" data-type="number">seeds</th>
            <th class="num" data-type="number">accuracy</th>
            <th class="num" data-type="number">ctx-swap</th>
            <th class="num" data-type="number">matched − control</th>
          </tr>
        </thead>
        <tbody>
{d2a_rows}
        </tbody>
      </table>
    </div>
    <div class="table-foot">Reproduce: <code>bash scripts/reproduce/document_niah_numeric_decoy_&lt;codec&gt;_all.sh cuda:0</code>.</div>
  </div>

  <p class="muted">LoRA is the validated reference, not the answer to the benchmark question. Losing shapes stay on the board — a codec that fails to condition is itself a result.</p>
</section>

<section id="contribute">
  <h2>Contribute a codec</h2>
  <p class="lead">A new adapter shape needs an output structure and a hook site. Everything else is shared — that is the whole point.</p>
  <ol class="contribute-steps">
    <li><strong>Subclass <code>GeneratedUpdateCodec</code></strong>
      <span class="muted">in <code>src/adapterbench/t2a/codecs.py</code>: <code>output_size</code>, <code>apply</code>, <code>dense_delta</code>, and <code>initial_bias</code> when bilinear.</span></li>
    <li><strong>Register and manifest</strong>
      <span class="muted">one <code>make_codec</code> entry plus <code>configs/adapters/&lt;name&gt;.yaml</code>.</span></li>
    <li><strong>Run both settings</strong>
      <span class="muted">≥3 seeds at the codec's own best swept scale; record <code>matched − control</code>.</span></li>
    <li><strong>Open a pull request</strong>
      <span class="muted">append a leaderboard row, a canonical record, a reproduce script, and regenerate this site with <code>uv run adapterbench website build</code>.</span></li>
  </ol>
  <p class="muted">Full checklist: <a href="{GITHUB_BLOB}/CONTRIBUTING.md">CONTRIBUTING.md</a> · user guide: <a href="{GITHUB_BLOB}/GUIDE.md">GUIDE.md</a>.</p>
</section>
"""
    return page_shell("Home", body)


def main() -> int:
    records = load_records(CANONICAL)
    (WEBSITE / "codecs").mkdir(parents=True, exist_ok=True)

    index_path = WEBSITE / "index.html"
    index_path.write_text(render_index(records))

    pages = 0
    by_slug: dict[str, list[dict]] = {}
    for record in records:
        by_slug.setdefault(codec_slug(record), []).append(record)

    for slug, setting_records in sorted(by_slug.items()):
        ordered = sorted(setting_records, key=lambda r: r["setting"])
        (WEBSITE / "codecs" / f"{slug}.html").write_text(render_codec_page(slug, ordered))
        pages += 1

    for slug in by_slug:
        path = WEBSITE / "codecs" / f"{slug}.html"
        if not path.exists():
            raise SystemExit(f"missing codec page for {slug}")

    print(f"Generated {index_path} and {pages} codec pages under {WEBSITE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
