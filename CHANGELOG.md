# Changelog

## 0.1.0 — 2026-07-20

Initial release candidate with the two active language settings: definition-stripped T2A
and document-conditioned D2A. It ships the rank-8 LoRA reference result, versioned
canonical aggregate records, deterministic table consistency checks, provenance, and
restart-safe reproduction wrappers.

## Unreleased

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
