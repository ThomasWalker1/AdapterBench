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
(5K → 20K → 60K → 150K). A paper-scale single-trajectory run (1M steps) is in progress to extend
the curve. This is the setting where the control both *caught* an under-powered over-claim and then
*tracked* the real effect once the recipe was scaled.

**Reproduce:** `scripts/reproduce/task_t2l_lora.sh [DEVICE] [STEPS]` (3 seeds; `STEPS=1000000` for
paper scale). The per-seed command it wraps:

```bash
.venv/bin/adapterbench t2p-sft-pilot --all-decontam-tasks --adapters lora \
  --max-descriptions 128 --batch-size 8 --grad-accum-steps 1 \
  --learning-rate 2.5e-5 --warmup-frac 0.1 \
  --eval-tasks arc_easy,arc_challenge,hellaswag,boolq --eval-limit 80 \
  --adversarial-control --seeds 777 --steps 150000 --checkpoint-every 10000 \
  --output results/repro/task_t2l_lora/s777
```
