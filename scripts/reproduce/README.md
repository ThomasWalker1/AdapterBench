# Reproduce scripts

One script per committed leaderboard row. Each wraps the setting's CLI at the exact
free-hyperparameters recorded in `leaderboards/<setting>.md`, runs ≥3 seeds, and aggregates to the
headline `matched − control` number. Re-running a script against the codec on `main` regenerates the
row up to seed variance — this is the benchmark's reproducibility contract.

<!-- canonical-results:repro-summary-markdown:start -->
| Setting | Script | Leaderboard | Headline |
|---|---|---|---|
| Task (T2A) — LoRA | `scripts/t2a_confirmation_seeds.py` | `task_conditioned_t2a.md` | matched − static\* = **+0.049 ± 0.027** ROUGE-L (3/3 confirmation seeds, 3 confirmation seeds) |
| Task (T2A) — FourierFT | `scripts/t2a_confirmation_seeds.py` | `task_conditioned_t2a.md` | matched − static\* = **+0.068 ± 0.044** ROUGE-L (3/3 confirmation seeds, 3 confirmation seeds) |
| Task (T2A) — (IA)³ | `scripts/t2a_confirmation_seeds.py` | `task_conditioned_t2a.md` | matched − static\* = **+0.128 ± 0.049** ROUGE-L (3/3 confirmation seeds, 3 confirmation seeds) |
| Task (T2A) — LoKr | `scripts/t2a_confirmation_seeds.py` | `task_conditioned_t2a.md` | matched − static\* = **+0.077 ± 0.033** ROUGE-L (3/3 confirmation seeds, 3 confirmation seeds) |
| Task (T2A) — Steering | `scripts/t2a_confirmation_seeds.py` | `task_conditioned_t2a.md` | matched − static\* = **+0.119 ± 0.038** ROUGE-L (3/3 confirmation seeds, 3 confirmation seeds) |
| Document (NIAH) — LoRA (r=8) | `document_niah_numeric_decoy_lora.sh [DEVICE]` | `document_niah_d2a.md` | matched − ctxswap = **+0.556 ± 0.327** (5 seeds, realistic-prose, 4 numeric decoys; crossover 16×) |
| Document (NIAH) — FourierFT | `document_niah_numeric_decoy_fourierft_all.sh [DEVICE]` | `document_niah_d2a.md` | matched − ctxswap = **+0.656 ± 0.352** (5 seeds, realistic-prose, 4 numeric decoys; crossover 64×) |
| Document (NIAH) — (IA)³ | `document_niah_numeric_decoy_ia3.sh [DEVICE]` | `document_niah_d2a.md` | matched − ctxswap = **+0.738 ± 0.327** (5 seeds, realistic-prose, 4 numeric decoys; crossover 64×) |
| Document (NIAH) — LoKr | `document_niah_numeric_decoy_lokr_all.sh [DEVICE]` | `document_niah_d2a.md` | matched − ctxswap = **+0.981 ± 0.037** (5 seeds, realistic-prose, 4 numeric decoys; crossover 64×) |
| Document (NIAH) — steering | `document_niah_numeric_decoy_steering_all.sh [DEVICE]` | `document_niah_d2a.md` | matched − ctxswap = **+0.881 ± 0.050** (5 seeds, realistic-prose, 4 numeric decoys; crossover 64×) |
<!-- canonical-results:repro-summary-markdown:end -->

All scripts run `adapterbench preflight` before training and are restart-safe: re-run the
identical command to resume from the last checkpoint. Use the matching one-seed T2A codec
script when only one seed is needed; each `*_all.sh` wrapper launches all three sequentially
and aggregates their existing files without retraining completed work.

Smoke paths (they validate plumbing, not the released metric):

```bash
uv run adapterbench t2a-sft --device cuda:0 --tasks lol_022 --adapter lora --steps 60 --output results/t2a_sft/smoke_lol022_lora.json
uv run adapterbench d2a-niah --adapters lora --steps 20 --grad-accum-steps 1 --context-lengths 256 --num-train-documents 40 --eval-limit 10 --device cuda:0 --output results/d2a_niah_smoke
```

## How a leaderboard row is produced (the pattern every future codec follows)

A row is a `(shape, free-HPs, matched − control, #seeds, reproduce command)` tuple. To add one:

1. **Land the codec** — a `GeneratedUpdateCodec` subclass + a `make_codec` entry + a manifest under
   `configs/adapters/<name>.yaml`. (The baseline registers only LoRA; this is how a new shape arrives.)
2. **Choose the free HPs** — scale, learning rate, warmup, and step budget.
   Sweep scale and report best-of (invariant #2); the shared substrate and the shape's rank
   are fixed, never tuned.
3. **Run ≥3 seeds** through the setting's CLI (invariant #4) and aggregate `matched − control`
   (invariant #1 — never a *raw* loss; a controlled difference like T2A's matched − static is fair).
   Report the difficulty-knob curve too (invariant #3).
4. **Append the row** to the setting's leaderboard with the number and the exact command, and add a
   reproduce script here mirroring the ones above.

The current LoRA rows exist to demonstrate this end to end with a single, well-understood shape;
the "does shape matter?" comparison is populated by repeating steps 1–4 per codec.

## Notes

- **T2A** trains two data-parallel runs — the strip-def hypernetwork and the same-shape static
  reference at its OWN selected hyperparameters — then scores `matched − static*` ROUGE-L on the 11
  genuinely held-out SNI tasks, with exact match and CE as appendix figures. It needs
  `HF_HUB_OFFLINE=1` (set by the script; the model and datasets are cached). Args are `SEED`,
  `GPUS_HYPER`, `GPUS_STATIC` (e.g. `1801 0,1,2,3 4,5,6,7`).
- **GPU selection.** The T2A script splits GPUs across the two runs; the document script
  takes a single `DEVICE` like `cuda:0`.
