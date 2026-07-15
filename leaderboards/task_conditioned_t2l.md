# Language domain — task-conditioned generation (Text-to-LoRA)

**Task.** The hypernetwork receives a pooled embedding of a free-text task description and emits an
adapter specializing a frozen `Qwen3-0.6B` to that task. Trained from scratch on the
Lots-of-LoRAs / Super-Natural-Instructions 479-task decontaminated split with the live SFT
objective; evaluated on held-out families (arc-easy, arc-challenge, boolq, hellaswag) by generation
and answer extraction.

**Metric — `matched − control`.** Matched = accuracy of the adapter generated from the *correct*
description. Control = **mismatched-description** (adversarial): the adapter is generated from an
unrelated/meaningless description; conditioning counts only when the correct description wins.
Headline = mean over the four families of (matched − adversarial).

**Free HPs:** scale, learning rate, warmup, steps. **Fixed and load-bearing (substrate):** 128
descriptions per task, eval-limit 80, the four eval families, the adversarial control — reducing any
of these reproduces the over-claim in which a task-*independent* adapter looks like conditioning.
**Fixed:** rank (shape identity).

Baseline commit: `1ec5951`.

| Shape | scale | lr | warmup | steps | seeds | **matched − adversarial** |
|-------|:-----:|-------:|:------:|-------:|:-----:|--------------------------:|
| LoRA (r=8) | default | 2.5e-5 | 0.1 | 150 000 | 3 | **+0.033** (all 4 families) |

Conditioning **emerges with training**: the effect strengthens along the budget trajectory
(5K → 20K → 60K → 150K). This is the setting where the control both *caught* an under-powered
over-claim and then *tracked* the real effect once the recipe was scaled.

**Shipped full-scale config (headline, run in progress).** The committed row above is the confirmed
150K × 3-seed batch-8 result. The benchmark's *shipped* full-scale training is the faster
data-parallel path (`scripts/t2p_train_ddp.py`): N-GPU DDP at a data-budget-matched effective batch
of 128 (62,500 steps = 8M example-visits = same data/epochs as a single-GPU 1M × batch-8 run), ~8h
instead of ~2 days. A seed-777 run is training now; when it and seeds 778/779 land, this row moves to
the DDP config (the +0.033 batch-8 number stays as the emergence-curve reference). **Still open at
that point:** invariant #2 (report best-of-scale) — a short `--scales` sweep at the shipped recipe,
currently "scale default".

**Reproduce (shipped config):** `scripts/reproduce/task_t2l_lora_ddp.sh [GPUS]` (3 seeds; default
GPUs `1,2,3,4`). The per-seed command it wraps:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/t2p_train_ddp.py --all-decontam-tasks --max-descriptions 128 --limit 40 \
  --per-gpu-batch 32 --steps 62500 --learning-rate 1e-4 --warmup-frac 0.1 \
  --max-grad-norm 1.0 --fixed-seq-len 512 \
  --eval-tasks arc_easy,arc_challenge,hellaswag,boolq --eval-limit 80 \
  --adversarial-control --seed 777 --checkpoint-every 5000 \
  --output results/repro/task_t2l_lora_ddp/s777
```

**Reproduce (batch-8 emergence-curve reference):** `scripts/reproduce/task_t2l_lora.sh [DEVICE] [STEPS]`
(3 seeds; `STEPS=1000000` for the single-GPU paper-scale point). The per-seed command it wraps:

```bash
.venv/bin/adapterbench t2p-sft-pilot --all-decontam-tasks --adapters lora \
  --max-descriptions 128 --batch-size 8 --grad-accum-steps 1 \
  --learning-rate 2.5e-5 --warmup-frac 0.1 \
  --eval-tasks arc_easy,arc_challenge,hellaswag,boolq --eval-limit 80 \
  --adversarial-control --seeds 777 --steps 150000 --checkpoint-every 10000 \
  --output results/repro/task_t2l_lora/s777
```
