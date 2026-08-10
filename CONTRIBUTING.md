# Contributing to AdapterBench

AdapterBench is a research benchmark: the shared substrate (conditioner, hypernetwork
trunk, training loop, evaluator, controls) stays fixed within each setting, and
contributions vary **only** the generated adapter representation (the *codec*).

This document is the path from a new codec idea to a merged pull request with a
reproducible leaderboard row in both active settings.

## Before you start

1. Read [GUIDE.md](GUIDE.md) for the two settings (T2A and D2A), the scoring
   invariants, and the codec interface.
2. Read [BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md) for the trial/result contract.
3. Set up the environment per [SETUP.md](SETUP.md) and confirm a clean install:

```bash
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
uv run adapterbench doctor --require-cuda
uv run pytest -q
uv run adapterbench validate
```

## Adding a new codec

A codec needs exactly two things:

1. **Output structure** — a differentiable mapping from hypernetwork scalars to an
   adapter update (`GeneratedUpdateCodec` in `src/adapterbench/t2a/codecs.py`).
2. **Hook site** — a named linear submodule for weight-space adapters, or the
   residual stream (`block`) for activation-space ones.

### Implementation checklist

1. **Subclass `GeneratedUpdateCodec`** in `src/adapterbench/t2a/codecs.py`:
   - `output_size` — scalars emitted per (layer, module).
   - `apply(inputs, base_output, generated, layer_index)` — fold generated values
     into the hooked module's output.
   - `dense_delta(generated, layer_index)` — weight-space ΔW when applicable.
   - `initial_bias()` — only if `apply()` is bilinear (LoRA, LoKr, LoHa); linear
     codecs keep the default `None`.

2. **Register** the codec in `make_codec()` and add default hook sites in
   `PILOT_DEFAULT_TARGET_MODULES` (`cli/_shared.py`) and
   `DOC_TO_LORA_PARITY_TARGET_MODULES` (`cli/live_sft.py`).

3. **Add a manifest** at `configs/adapters/<name>.yaml` and extend the
   `AdapterManifest.family` literal in `schema.py`.

4. **Add tests** covering geometry, initialization, and hook application (see
   `tests/test_live_evaluator.py` and `tests/test_document_live_evaluator.py` for
   patterns used by existing codecs).

5. **Validate locally**:

```bash
uv run adapterbench validate
uv run adapterbench catalog   # your codec should appear
uv run pytest -q -k <your_codec>
```

### Smoke runs (plumbing only)

T2A:

```bash
uv run adapterbench t2a-sft --device cuda:0 --tasks lol_022 \
  --adapter <your_codec> --target-modules q_proj,v_proj \
  --steps 60 --output results/t2a_sft/smoke_<your_codec>.json
```

D2A:

```bash
uv run adapterbench d2a-niah --adapters <your_codec> --steps 20 \
  --grad-accum-steps 1 --context-lengths 256 --num-train-documents 40 \
  --eval-limit 10 --device cuda:0 --output results/d2a_niah_smoke
```

These validate that training and evaluation hooks work; they do **not** produce a
leaderboard row.

## Producing a leaderboard row

Every published result satisfies four invariants (see [GUIDE.md](GUIDE.md) §2):

1. Headline is **matched − control** on a behavioral metric, never raw loss.
2. **Scale is swept**; report best-of at the codec's own optimum.
3. A **graded difficulty axis** is reported (T2A: held-out task spread; D2A:
   context-length generalization).
4. **≥3 seeds** with mean ± spread.

Hyperparameters partition:

- **Shared substrate** (never tuned per shape): data, trunk, evaluator, control protocol.
- **Free** (tuned per shape): scale, learning rate, warmup, step budget — separately
  for the hypernetwork and the static control in T2A.
- **Shape identity** (fixed): e.g. LoRA rank.

### T2A protocol

Follow [AUTORESEARCH.md](AUTORESEARCH.md) for the full selection pipeline. At a
high level:

1. **Scout** — sweep free axes on the selection split:
   `scripts/t2a_sweep_axes.py --codecs <your_codec>`
2. **Select operating point** — verify stability on 3 selection seeds:
   `scripts/t2a_selection_seeds.py --codecs <your_codec>`
3. **Confirm** — train 3 fresh confirmation seeds:
   `scripts/t2a_confirmation_seeds.py --codecs <your_codec>`
4. **Score once** on the 11-task report split:
   `bash scripts/reproduce/t2a_score_codec.sh <your_codec>`
5. **Aggregate**:
   `.venv/bin/python scripts/t2a_confirmation_result.py`

Or use the all-in-one wrapper after configuring operating points in
`scripts/t2a_selection_seeds.py`:

```bash
bash scripts/reproduce/t2a_reproduce_all.sh <your_codec> 0,1,2,3 4,5,6,7
```

### D2A protocol

1. Run a scale sweep (D2A useful scales differ widely from PEFT defaults):
   `bash scripts/d2a_scale_sweep.sh` or the codec-specific locator scripts.
2. Train ≥5 seeds at the locked numeric-decoy setting:
   model the existing `scripts/reproduce/document_niah_numeric_decoy_*.sh` scripts.
3. Aggregate:
   `.venv/bin/python scripts/d2a_niah_aggregate.py --root <run> --min-seeds 5 --train-length 512`

### Recording the result

1. Write a compact canonical record under `canonical_results/` (see existing JSON
   files for schema). Use `scripts/t2a_write_canonical.py` or
   `scripts/d2a_steering_canonical.py` as templates.
2. Append a row to the setting leaderboard (`leaderboards/task_conditioned_t2a.md`
   or `leaderboards/document_niah_d2a.md`) with exact free HPs, seeds, and the
   reproduce command.
3. Add a reproduce script under `scripts/reproduce/` (thin wrapper around the
   shared drivers is fine — see `task_t2a_lora.sh`).
4. Verify drift checks pass and regenerate the public site:

```bash
uv run adapterbench results validate
uv run adapterbench results check
uv run adapterbench website build
```

## Pull request guidelines

- **Scope:** one codec (or one focused fix) per PR. Do not change the shared
  substrate unless there is a documented bug affecting all codecs equally.
- **Tests:** new codec PRs must include unit tests; run `uv run pytest -q`.
- **Artifacts:** do not commit checkpoints, logs, or full `results/` trees.
  Canonical metrics live in `canonical_results/`; large training outputs stay local.
- **Negative results welcome:** a codec that fails to condition is itself a result.
  Record it on the leaderboard with the measured `matched − control`.
- **Description:** include the selected free HPs, seed count, headline
  `matched − control`, and the exact reproduce command.

## Repository layout (quick reference)

| Path | Purpose |
|------|---------|
| `src/adapterbench/t2a/codecs.py` | Codec implementations and registration |
| `configs/adapters/` | Per-codec manifests |
| `leaderboards/` | Published rows with reproduce commands |
| `canonical_results/` | Versioned aggregate records (source of truth) |
| `scripts/reproduce/` | One script per committed leaderboard row |
| `website/` | Public benchmark site (`adapterbench website build`); see `website/INTEGRATION.md` |
| `AUTORESEARCH.md` | T2A hyperparameter search protocol |
| `NEGATIVE_RESULTS.md` | Retired settings and failed controls |

## Questions

Open a GitHub issue for protocol questions before starting a large GPU run.
The benchmark intentionally has no automated hyperparameter search — every free HP
choice must be documented in the leaderboard row.
