# Changelog

## 0.1.0 — 2026-07-20

Initial release candidate with the two active language settings: definition-stripped T2A
and document-conditioned D2A. It ships the rank-8 LoRA reference result, versioned
canonical aggregate records, deterministic table consistency checks, provenance, and
restart-safe reproduction wrappers.

## Unreleased

- **Static website builder** — `website/` with PAARBench-style layout (IBM Plex, sortable
  leaderboards, per-codec detail pages). Build via `uv run adapterbench website build`;
  deploy to a personal site with `website/INTEGRATION.md`. Root `benchmark.yaml` declares
  metadata for site aggregators.
- **Repository cleanup for open-source contributors:** added [CONTRIBUTING.md](CONTRIBUTING.md);
  unified T2A reproduction under `scripts/reproduce/t2a_reproduce_*.sh` with operating points
  matching `canonical_results/`; removed legacy one-off scripts (`migrate_seed3_at_15k.sh`,
  `t2a_promote.sh`, `t2a_release_aggregate.py`); aligned leaderboard and setup docs with the
  AUTORESEARCH confirmation protocol (seeds 1741–4743, not the superseded 1801-era runs).
- **New codec: `dora`** — weight-decomposed low-rank adaptation (Liu et al., 2024) as a
  generated shape. The hypernetwork emits LoRA's rank-8 directional `A`/`B` factors **plus
  one magnitude scalar per output channel**, applied as
  `W' = m ⊙ (W0 + scale·B@A)/||W0 + scale·B@A||_row` at the same q_proj/v_proj (T2A) and
  down_proj (D2A) sites LoRA uses. Generated magnitudes are a delta on the frozen row
  norms, so the zero-init head is exactly the frozen projection; the renormalizing
  denominator is detached, as in the reference implementation (DoRA §4.3, PEFT's
  `DoraLinearLayer`), the magnitude is per output channel with the norm reduced over fan-in
  (following the implementations rather than the paper's transposed "column" wording), and a
  hook-site bias is excluded from the rescaling. Live row norms use an exact algebraic
  expansion, so no per-example dense `ΔW` is materialized.
  - **New codec-interface seam: `GeneratedUpdateCodec.apply_at(module, ...)`.** DoRA is the
    first registered shape whose update is defined *relative to the frozen weight it
    edits*, which `base_output = W0 @ x` cannot recover. The forward hooks now pass the
    resolved hook-site module; the default implementation delegates to `apply()`, so every
    existing codec is unchanged bit-for-bit. Weight-decomposed codecs therefore require a
    linear-projection hook site (never `"block"`), which the codec enforces.
  - Registered in `make_codec`, both hook-site maps, `configs/adapters/dora.yaml`, and the
    manifest schema; unit tests cover geometry, identity init, the bilinear saddle bias,
    equivalence with a materialized reference implementation, `dense_delta`/`apply`
    consistency, hook application, and the static-control path.
  - **Both benchmark rows landed** under the AUTORESEARCH.md protocol. D2A:
    `matched − context-swap = +0.956 ± 0.073` (5 held-out seeds, scale 64, lr 2e-5, 32k steps),
    perfect retrieval at every length to 32768 tokens (64× the training length), tail log-AUC
    0.969 — joint-top with LoKr and not separable from it. T2A:
    `matched − static* = +0.069 ± 0.031` ROUGE-L on the 11 held-out SNI tasks (3 confirmation
    seeds, hyper scale 0.25 / lr 1e-4, independently selected static* at scale 16 / lr 1e-4),
    mid-table and not separable from any other shape. DoRA is the first codec whose relative
    standing differs sharply between the two settings.
  - **DoRA's scale and learning rate interact**, so the D2A operating point was located by a joint
    scale x LR sweep rather than one axis at a time: scale 64 does not retrieve at the setting-default
    lr 4e-5 and reaches the gate ceiling at 2e-5. The sweep used the scout seed and dev eval
    instrument only; the point was then re-derived on protocol at the full instrument, where scales
    32/64/128 form a tied set at the ceiling and the LR axis closes interior.
- **New codec: `steering`** — the first activation-space shape. The hypernetwork emits
  one `d_model` steering vector per (layer, example), added to the residual stream at
  the `"block"` hook site in both settings (`h -> h + scale * v`); linear in the
  generated output, so zero-init is exact identity with a nonzero gradient. Registered
  in `make_codec`, both hook-site maps, `configs/adapters/steering.yaml`, and the
  manifest schema; unit tests cover geometry, identity init, per-example broadcast, and
  residual-stream hook backprop. Benchmark rows pending the AUTORESEARCH.md protocol.
- **Generic `--codec-scaling` flag** on `d2a-niah`, `t2a_train_ddp.py`, and the three
  snapshot evaluators: one uniform output-scale override for the selected codec
  (`set_codec_scaling`), replacing the per-codec `--<name>-scaling` pattern for new
  codecs (existing flags retained for recorded commands). `t2a_codec_trial.sh` and
  `d2a_niah_multifidelity.sh` now use it, so a new codec needs no scale-flag mapping in
  any driver script.
- `TextToPeftHypernetwork` forwards codec-specific kwargs to `make_codec` via
  `**codec_kwargs` instead of naming each codec's scale parameter (dead `n_frequency`/
  `steering_scale` parameters removed); adding a codec no longer touches the
  constructor.
- `t2a-sft-pilot --scales` now applies the swept scale to every codec as documented
  (previously a silent no-op for non-LoRA adapters); `d2a-niah` validates `--adapters`
  names up front.
- Docs refreshed to the multi-codec reality (README/GUIDE/PROJECT_PLAN/AGENTS: registered
  codec list, D2A numeric-decoy 512-token recipe, codec-exploration phase);
  `aggregate_lengthgen.py` moved under `scripts/`.
