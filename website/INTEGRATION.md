# AdapterBench website — personal site integration

The `website/` directory is a **static site** generated from committed canonical
results. It uses the same visual language as
[PAARBench](https://github.com/ThomasWalker1/PAARBench) and the Thomas Walker
personal site (IBM Plex Sans/Mono, dark header, sortable leaderboard tables).

## Build locally

```bash
uv pip install -e ".[dev]"
uv run adapterbench website build
# or: .venv/bin/python scripts/build_website.py
```

Output:

| Path | Purpose |
|------|---------|
| `website/index.html` | Benchmark landing page (about, settings, leaderboards, contribute) |
| `website/codecs/<name>.html` | Per-codec detail pages with HPs, seeds, reproduce commands |
| `website/css/style.css` | Shared stylesheet |
| `website/js/sort-tables.js` | Sortable table behaviour |

Regenerate after updating `canonical_results/` or adding a codec row, then commit
the HTML so GitHub Pages / a vendored copy stays current.

## Add to [personal_webpage](https://github.com/ThomasWalker1/personal_webpage) / thomasw.org

`benchmark.yaml` at the repository root declares metadata for site aggregators:

```yaml
website:
  source_dir: website
  entry: index.html
  deploy_path: benchmarks/adapterbench
```

### Option A — copy the built site (recommended)

After running `adapterbench website build` in AdapterBench:

```bash
# From your personal_webpage checkout:
rsync -av --delete /path/to/AdapterBench/website/ public/benchmarks/adapterbench/
```

Link from the main site navigation, e.g.:

```html
<a href="/benchmarks/adapterbench/index.html">AdapterBench</a>
```

### Option B — git submodule

```bash
cd public/benchmarks
git submodule add https://github.com/ThomasWalker1/AdapterBench.git adapterbench-src
# Build inside the submodule and symlink or copy website/ to adapterbench/
cd adapterbench-src && uv run adapterbench website build
rsync -av website/ ../adapterbench/
```

### Option C — standalone GitHub Pages

Serve `website/` directly from the AdapterBench repo (Settings → Pages → deploy
from `/website` on `main`), then link `https://<user>.github.io/AdapterBench/`
from the personal site.

## Drift checks

Leaderboard **numbers** in markdown tables are checked by:

```bash
uv run adapterbench results check
```

The **website HTML** is regenerated from the same `canonical_results/` JSON files
via `build_website.py`, so running both after a new codec row keeps markdown,
website, and canonical records aligned:

```bash
uv run adapterbench results write    # refresh leaderboards / PROJECT_PLAN markers
uv run adapterbench website build    # refresh website/
uv run adapterbench results check
```

## Adding a new codec to the site

1. Land the codec and append `canonical_results/<codec>.json` for each setting.
2. Run `uv run adapterbench results write` and `uv run adapterbench website build`.
3. Commit `website/codecs/<slug>.html` and the updated `website/index.html`.

No manual HTML editing is required — the builder reads canonical records and
manifests under `configs/adapters/`.
