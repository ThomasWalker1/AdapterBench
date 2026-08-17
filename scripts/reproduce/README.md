# Reproduce scripts

One script per committed leaderboard row. Each wraps the setting's CLI at the exact
free hyperparameters recorded in `leaderboards/<setting>.md`, runs ≥3 seeds, and aggregates to the
headline `matched − control` number. Re-running a script against the codec on `main` regenerates the
row up to seed variance — this is the benchmark's reproducibility contract.

<!-- canonical-results:repro-summary-markdown:start -->
| Setting | Script | Leaderboard | Headline |
|---|---|---|---|
| Task (T2A) — LoRA | `scripts/reproduce/t2a_reproduce_all.sh lora` | `task_conditioned_t2a.md` | matched − static\* = **+0.049 ± 0.027** ROUGE-L (3/3 confirmation seeds, 3 confirmation seeds) |
| Task (T2A) — DoRA | `scripts/reproduce/t2a_reproduce_all.sh dora` | `task_conditioned_t2a.md` | matched − static\* = **+0.069 ± 0.031** ROUGE-L (3/3 confirmation seeds, 3 confirmation seeds) |
| Task (T2A) — FourierFT | `scripts/reproduce/t2a_reproduce_all.sh fourierft` | `task_conditioned_t2a.md` | matched − static\* = **+0.068 ± 0.044** ROUGE-L (3/3 confirmation seeds, 3 confirmation seeds) |
| Task (T2A) — (IA)³ | `scripts/reproduce/t2a_reproduce_all.sh ia3` | `task_conditioned_t2a.md` | matched − static\* = **+0.128 ± 0.049** ROUGE-L (3/3 confirmation seeds, 3 confirmation seeds) |
| Task (T2A) — LoKr | `scripts/reproduce/t2a_reproduce_all.sh lokr` | `task_conditioned_t2a.md` | matched − static\* = **+0.077 ± 0.033** ROUGE-L (3/3 confirmation seeds, 3 confirmation seeds) |
| Task (T2A) — Steering | `scripts/reproduce/t2a_reproduce_all.sh steering` | `task_conditioned_t2a.md` | matched − static\* = **+0.119 ± 0.038** ROUGE-L (3/3 confirmation seeds, 3 confirmation seeds) |
| Document (NIAH) — LoRA (r=8) | `document_niah_numeric_decoy_lora.sh [DEVICE]` | `document_niah_d2a.md` | matched − ctxswap = **+0.556 ± 0.327** (5 seeds, realistic-prose, 4 numeric decoys; crossover 16×) |
| Document (NIAH) — DoRA | `document_niah_numeric_decoy_dora_all.sh [DEVICE]` | `document_niah_d2a.md` | matched − ctxswap = **+0.956 ± 0.073** (5 seeds, realistic-prose, 4 numeric decoys; crossover 64×) |
| Document (NIAH) — FourierFT | `document_niah_numeric_decoy_fourierft_all.sh [DEVICE]` | `document_niah_d2a.md` | matched − ctxswap = **+0.656 ± 0.352** (5 seeds, realistic-prose, 4 numeric decoys; crossover 64×) |
| Document (NIAH) — (IA)³ | `document_niah_numeric_decoy_ia3.sh [DEVICE]` | `document_niah_d2a.md` | matched − ctxswap = **+0.738 ± 0.327** (5 seeds, realistic-prose, 4 numeric decoys; crossover 64×) |
| Document (NIAH) — LoKr | `document_niah_numeric_decoy_lokr_all.sh [DEVICE]` | `document_niah_d2a.md` | matched − ctxswap = **+0.981 ± 0.037** (5 seeds, realistic-prose, 4 numeric decoys; crossover 64×) |
| Document (NIAH) — steering | `document_niah_numeric_decoy_steering_all.sh [DEVICE]` | `document_niah_d2a.md` | matched − ctxswap = **+0.881 ± 0.050** (5 seeds, realistic-prose, 4 numeric decoys; crossover 64×) |
<!-- canonical-results:repro-summary-markdown:end -->

## T2A drivers

Shared infrastructure lives in this directory:

| Script | Role |
|--------|------|
| `t2a_codec_config.sh` | Operating points and confirmation seeds per codec |
| `t2a_reproduce_seed.sh CODEC [SEED] [GPUS_HYPER] [GPUS_STATIC]` | Train one confirmation seed |
| `t2a_reproduce_all.sh CODEC [GPUS_HYPER] [GPUS_STATIC]` | Train all seeds, score report split, aggregate |
| `t2a_score_codec.sh CODEC[,...]` | Score the one-shot report split for one or more codecs |
| `task_t2a_<codec>.sh` | Thin wrapper around `t2a_reproduce_seed.sh` |

The full AUTORESEARCH selection pipeline (scout → selection → confirmation) is documented in
[AUTORESEARCH.md](../AUTORESEARCH.md). Confirmation training can also be launched for all codecs at
once via `scripts/t2a_confirmation_seeds.py`.

All scripts run `adapterbench preflight` before training (unless resuming) and are restart-safe.

Smoke paths (plumbing only, not the released metric):

```bash
uv run adapterbench t2a-sft --device cuda:0 --tasks lol_022 --adapter lora --steps 60 --output results/t2a_sft/smoke_lol022_lora.json
uv run adapterbench d2a-niah --adapters lora --steps 20 --grad-accum-steps 1 --context-lengths 256 --num-train-documents 40 --eval-limit 10 --device cuda:0 --output results/d2a_niah_smoke
```

## How a leaderboard row is produced (the pattern every future codec follows)

A row is a `(shape, free-HPs, matched − control, #seeds, reproduce command)` tuple. To add one:

1. **Land the codec** — a `GeneratedUpdateCodec` subclass + a `make_codec` entry + a manifest under
   `configs/adapters/<name>.yaml`. See [CONTRIBUTING.md](../../CONTRIBUTING.md).
2. **Choose the free HPs** — scale, learning rate, warmup, and step budget (separately for hyper
   and static* in T2A). Sweep scale and report best-of.
3. **Run ≥3 confirmation seeds** and score the report split exactly once.
4. **Append the row** to the setting's leaderboard and add a thin reproduce wrapper here.

## Notes

- **T2A** trains hypernetwork and static* on disjoint GPU sets at independently selected free HPs.
  Evaluations require `HF_HUB_OFFLINE=1` (set by the scripts).
- **GPU selection.** T2A splits GPUs across the two roles; D2A takes a single `DEVICE` like `cuda:0`.
