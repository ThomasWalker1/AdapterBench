# Reproduce scripts

One script per committed leaderboard row. Each wraps the setting's CLI at the exact
free-hyperparameters recorded in `leaderboards/<setting>.md`, runs ≥3 seeds, and aggregates to the
headline `matched − control` number. Re-running a script against the codec on `main` regenerates the
row up to seed variance — this is the benchmark's reproducibility contract.

| Setting | Script | Leaderboard | Headline |
|---|---|---|---|
| Image (reward tilting) | `image_lora.sh [DEVICE]` | `image_reward_tilting.md` | matched − reward-swap = **+3.50** |
| Task (T2L), shipped full-scale | `task_t2l_lora_ddp.sh [GPUS]` | `task_conditioned_t2l.md` | matched − adversarial (DDP run in progress) |
| Task (T2L), batch-8 reference | `task_t2l_lora.sh [DEVICE] [STEPS]` | `task_conditioned_t2l.md` | **+0.033** (150K × 3, emergence-curve reference) |
| Document (NIAH) | `document_niah_lora.sh [DEVICE]` | `document_niah_d2l.md` | matched − ctxswap = **+1.00** |

All scripts are restart-safe: re-run the identical command to resume from the last checkpoint.

## How a leaderboard row is produced (the pattern every future codec follows)

A row is a `(shape, free-HPs, matched − control, #seeds, reproduce command)` tuple. To add one:

1. **Land the codec** — a `GeneratedUpdateCodec` subclass + a `make_codec` entry + a manifest under
   `configs/adapters/<name>.yaml`. (The baseline registers only LoRA; this is how a new shape arrives.)
2. **Choose the free HPs** — scale, learning rate, warmup, step budget (and `λ` for the image
   setting). Sweep scale and report best-of (invariant #2); the shared substrate and the shape's rank
   are fixed, never tuned.
3. **Run ≥3 seeds** through the setting's CLI (invariant #4) and aggregate `matched − control`
   (invariant #1 — never a loss). Report the difficulty-knob curve too (invariant #3).
4. **Append the row** to the setting's leaderboard with the number and the exact command, and add a
   reproduce script here mirroring the ones above.

The current LoRA rows exist to demonstrate this end to end with a single, well-understood shape;
the "does shape matter?" comparison is populated by repeating steps 1–4 per codec.

## Notes

- **T2L shipped vs reference.** `task_t2l_lora_ddp.sh` is the fast data-parallel path
  (`scripts/t2p_train_ddp.py`, effective batch 128, ~8h); `task_t2l_lora.sh` is the single-GPU
  batch-8 recipe that produced the +0.033 emergence-curve reference. See the T2L leaderboard.
- **GPU selection.** The DDP script takes a comma-separated `GPUS` list (default `1,2,3,4`); its
  length sets `--nproc_per_node`. The single-GPU scripts take a `DEVICE` like `cuda:0`.
