# Language domain — task-conditioned generation (Text-to-LoRA)

**Task.** The hypernetwork receives a pooled embedding of a free-text task description and emits an
adapter specializing a frozen `gemma-2-2b` to that task. Trained from scratch on the
Lots-of-LoRAs / Super-Natural-Instructions 479-task decontaminated split with the live SFT objective
(plain cross-entropy). The task **definition is stripped from the input** (`--strip-task-def`), so the
description is the *only* route to the task — otherwise the frozen model reads the task straight from
its prompt and the description-conditioned adapter is redundant (every codec then scores ≈0).

**Metric — `matched − static`.** Matched = the adapter generated from the task's held-out description.
Static = a single adapter of the *same shape*, directly optimized on the same SFT data (a multi-task
LoRA, not hypernetwork-emitted), which absorbs all task-*generic* help. Scored on the **21 held-out
SNI validation tasks** (`lol_###` in `eval_ds_info`):

- **CE (primary):** teacher-forced cross-entropy over the reference answer — the quantity the trainer
  optimizes, non-saturating, defined even for open-ended tasks. `matched − static < 0` = conditioning
  helps.
- **accuracy (corroborating):** greedy-generation normalized exact-match. A smaller signal than CE
  (it saturates where both adapters already succeed; multiple-choice tasks leak answer content).

`matched − static` is **non-gameable** (no wrong-condition case to sabotage), **subsumes the
helpfulness floor** (static ≥ frozen), and needs **no junk descriptions**; `matched − frozen` is
reported alongside.

**Free HPs:** scale, learning rate, warmup, steps. **Fixed substrate:** 128 descriptions per task, the
strip-def input template, the 21 held-out SNI tasks, the static-reference control, the CE metric.
**Fixed:** rank (shape identity).

| Shape | rank | lr | steps | seeds | **matched − static (CE, nats)** | matched − frozen | accuracy m−static / m−frozen |
|-------|:----:|:----:|------:|:-----:|:-------------------------------:|:----------------:|:----------------------------:|
| LoRA | 8 | 1e-4 | 20 000 | 3 | **−0.72 ± 0.16** (59/63 task-seed pairs) | −10.80 ± 0.11 | +0.032 / +0.235 |

Seeds 777, 2, and 3. More-negative CE is better; positive accuracy is better. The huge
`matched − frozen` (−10.80 nats CE, +0.235 accuracy) confirms the stripped-definition task genuinely
*requires* the adapter; `matched − static` isolates the description's contribution beyond generic help.

**Reproduce:** `scripts/reproduce/task_t2l_lora.sh [SEED] [GPUS_HYPER] [GPUS_STATIC]` — trains the
strip-def hypernetwork + the static reference, then runs both evals (sets `HF_HUB_OFFLINE=1` for you).

```bash
scripts/reproduce/task_t2l_lora.sh 777 0,1,2,3 4,5,6,7
# results (per-task + __aggregate__ rows):
#   results/repro/t2l_base_diag/gemma2b_stripdef_hyper/s777/heldout_sni_ce_full21.jsonl
#   results/repro/t2l_base_diag/gemma2b_stripdef_hyper/s777/heldout_sni_acc.jsonl
```
