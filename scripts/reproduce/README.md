# Reproduce scripts

One script per committed leaderboard row. Each wraps the setting's CLI at the exact
free-hyperparameters recorded in `leaderboards/<setting>.md`, runs ≥3 seeds, and aggregates to the
headline `matched − control` number. Re-running a script against the codec on `main` regenerates the
row up to seed variance — this is the benchmark's reproducibility contract.

| Setting | Script | Leaderboard | Headline |
|---|---|---|---|
| Task (T2L) | `task_t2l_lora.sh [SEED] [GPUS_HYPER] [GPUS_STATIC]` | `task_conditioned_t2l.md` | matched − static = **−0.72 ± 0.16** nats CE (59/63 task-seed pairs, 3 seeds) |
| Document (NIAH) | `document_niah_lora.sh [DEVICE]` | `document_niah_d2l.md` | matched − ctxswap = **+0.887 ± 0.143** (5 seeds, realistic haystack; crossover 16×) |

All scripts are restart-safe: re-run the identical command to resume from the last checkpoint.

## How a leaderboard row is produced (the pattern every future codec follows)

A row is a `(shape, free-HPs, matched − control, #seeds, reproduce command)` tuple. To add one:

1. **Land the codec** — a `GeneratedUpdateCodec` subclass + a `make_codec` entry + a manifest under
   `configs/adapters/<name>.yaml`. (The baseline registers only LoRA; this is how a new shape arrives.)
2. **Choose the free HPs** — scale, learning rate, warmup, and step budget.
   Sweep scale and report best-of (invariant #2); the shared substrate and the shape's rank
   are fixed, never tuned.
3. **Run ≥3 seeds** through the setting's CLI (invariant #4) and aggregate `matched − control`
   (invariant #1 — never a *raw* loss; a controlled difference like T2L's matched − static is fair).
   Report the difficulty-knob curve too (invariant #3).
4. **Append the row** to the setting's leaderboard with the number and the exact command, and add a
   reproduce script here mirroring the ones above.

The current LoRA rows exist to demonstrate this end to end with a single, well-understood shape;
the "does shape matter?" comparison is populated by repeating steps 1–4 per codec.

## Notes

- **T2L** trains two data-parallel runs — the strip-def hypernetwork and the same-shape static
  reference — then scores `matched − static` on the 21 held-out SNI tasks (CE + accuracy). It needs
  `HF_HUB_OFFLINE=1` (set by the script; the model and datasets are cached). Args are `SEED`,
  `GPUS_HYPER`, `GPUS_STATIC` (e.g. `777 0,1,2,3 4,5,6,7`).
- **GPU selection.** The T2L script splits GPUs across the two runs; the document script
  takes a single `DEVICE` like `cuda:0`.
