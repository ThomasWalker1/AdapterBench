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

<!-- canonical-results:t2l-markdown:start -->
| Shape | rank | scale | lr | steps | seeds | **matched − static (CE, nats)** | matched − frozen | accuracy m−static / m−frozen |
|---|:---:|:---:|:---:|---:|:---:|:---:|:---:|:---:|
| LoRA | 8 | 22.627417 | 1e-4 | 8 000 | 3 | **−0.571 ± 0.045** (54/63 task-seed pairs) | −11.21 ± 0.07 | −0.0050 ± 0.0278 / +0.158 |
| (IA)³ | — | 16 | 4e-4 | 8 000 | 3 | **−0.381 ± 0.023** (48/63 task-seed pairs) | −11.31 ± 0.06 | +0.0562 ± 0.0087 / +0.150 |
| LoKr | — | 16 | 2e-4 | 6 000 | 3 | **+4.249 ± 8.098** (37/63 task-seed pairs) | −6.69 ± 8.16 | −0.0241 ± 0.1032 / +0.048 |
<!-- canonical-results:t2l-markdown:end -->

More-negative CE is better; accuracy is corroborating and can diverge from CE. Both codecs beat
the frozen helpfulness floor; `matched − static` isolates the description's contribution beyond
generic help.

### Selection trails

Each compact trail below points to append-only scratch ledgers containing the exact scout commands,
candidate metrics, artifacts, and rejected points. Only fresh confirmation seeds enter the headline.

<!-- canonical-results:t2l-selection-markdown:start -->
| Shape | compact audit trail | selected final configuration |
|---|---|---|
| LoRA | Scale scout 0.353553→90.509668; 90.509668 was numerically invalid. At the common 8k rung, 22.627417 beat 1.414214 and 5.656854; a 20k continuation regressed. Fresh LR checks at 5e-5 and 2e-4 were worse than 1e-4. State ledgers: `scale_locator/state.jsonl`, `lr_locator/state.jsonl`. | scale 22.627417; lr 1e-4; 8,000 steps |
| (IA)³ | Tiny scales were retained as static-control failures, not selection evidence. In the viable high-scale ladder, 16 beat 4 and 64. At 8k, LR 4e-4 beat 5e-5, 1e-4, and 2e-4; 8e-4 catastrophically failed the helpfulness floor. State ledgers: `scale_locator/state.jsonl`, `lr_locator/state.jsonl`, `lr_locator_scale16/state.jsonl`. | scale 16; lr 4e-4; 8,000 steps |
| LoKr | The LoKr-owned scale ladder 0.0625→16 improved through 16; the geometric boundary at 64 catastrophically failed both the static-control comparison and frozen helpfulness floor. At scale 16, LR 2e-4 beat 1e-4 and 5e-5. Fresh confirmations were unstable: seeds 2704/2705 helped, while seed 2706 failed both controlled CE and the frozen helpfulness floor; the losing aggregate is retained. State ledgers: `lokr/state.jsonl`. | scale 16; lr 2e-4; 6,000 steps |
<!-- canonical-results:t2l-selection-markdown:end -->

**Reproduce:** `scripts/reproduce/task_t2l_lora.sh` and `scripts/reproduce/task_t2l_ia3.sh`
accept `[SEED] [GPUS_HYPER] [GPUS_STATIC]`, run preflight, train the strip-def hypernetwork + static
reference, and run both evaluations (sets `HF_HUB_OFFLINE=1`).

```bash
scripts/reproduce/task_t2l_lora.sh 1801 0,1,2,3 4,5,6,7
# results (per-task + __aggregate__ rows):
#   results/repro/t2l_lora_scale22.627417_lr1e-4/hyper/s1801/heldout_sni_ce_full21.jsonl
#   results/repro/t2l_lora_scale22.627417_lr1e-4/hyper/s1801/heldout_sni_acc.jsonl

scripts/reproduce/task_t2l_ia3.sh 1901 0,1,2,3 4,5,6,7
#   results/repro/t2l_ia3_scale16_lr4e-4/hyper/s1901/heldout_sni_ce_full21.jsonl
#   results/repro/t2l_ia3_scale16_lr4e-4/hyper/s1901/heldout_sni_acc.jsonl
```
