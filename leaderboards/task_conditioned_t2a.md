# Language domain — task-conditioned generation (Text-to-Adapter)

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

<!-- canonical-results:t2a-markdown:start -->
| Shape | rank | scale | lr | steps | seeds | **matched − static (CE, nats)** | matched − frozen | accuracy m−static / m−frozen |
|---|:---:|:---:|:---:|---:|:---:|:---:|:---:|:---:|
| LoRA | 8 | 22.627417 | 1e-4 | 8 000 | 3 | **−0.571 ± 0.045** (54/63 task-seed pairs) | −11.21 ± 0.07 | −0.0050 ± 0.0278 / +0.158 |
| FourierFT | — | 16 | 1e-4 | 6 000 | 3 | **−0.525 ± 0.064** (55/63 task-seed pairs) | −11.01 ± 0.08 | −0.0129 ± 0.0156 / +0.105 |
| (IA)³ | — | 16 | 4e-4 | 8 000 | 3 | **−0.381 ± 0.023** (48/63 task-seed pairs) | −11.31 ± 0.06 | +0.0562 ± 0.0087 / +0.150 |
| LoKr | — | 16 | 1e-4 | 6 000 | 3 | **−0.403 ± 0.172** (49/63 task-seed pairs) | −11.36 ± 0.09 | +0.0443 ± 0.0208 / +0.127 |
<!-- canonical-results:t2a-markdown:end -->

More-negative CE is better; accuracy is corroborating and can diverge from CE. All four codecs beat
the frozen helpfulness floor; `matched − static` isolates the description's contribution beyond
generic help.

### Selection trails

Each compact trail below points to append-only scratch ledgers containing the exact scout commands,
candidate metrics, artifacts, and rejected points. Only fresh confirmation seeds enter the headline.

<!-- canonical-results:t2a-selection-markdown:start -->
| Shape | compact audit trail | selected final configuration |
|---|---|---|
| LoRA | Scale scout 0.353553→90.509668; 90.509668 was numerically invalid. At the common 8k rung, 22.627417 beat 1.414214 and 5.656854; a 20k continuation regressed. Fresh LR checks at 5e-5 and 2e-4 were worse than 1e-4. State ledgers: `scale_locator/state.jsonl`, `lr_locator/state.jsonl`. | scale 22.627417; lr 1e-4; 8,000 steps |
| FourierFT | The scale ladder at seed 5001 improved monotonically through scale 16; scale 64 diverged, closing the upper boundary. At scale 16, LR 5e-5 and 1e-4 were each run on three fresh selection seeds (5101–5103); both converged 3/3, and LR 1e-4 had the better mean (−0.734 vs −0.644) and lower sample SD (0.089 vs 0.115), so it was selected. Three disjoint confirmation seeds (5111–5113) converged 3/3. State ledgers: `fourierft/state.jsonl`. | scale 16; lr 1e-4; 6,000 steps |
| (IA)³ | Tiny scales were retained as static-control failures, not selection evidence. In the viable high-scale ladder, 16 beat 4 and 64. At 8k, LR 4e-4 beat 5e-5, 1e-4, and 2e-4; 8e-4 catastrophically failed the helpfulness floor. State ledgers: `scale_locator/state.jsonl`, `lr_locator/state.jsonl`, `lr_locator_scale16/state.jsonl`. | scale 16; lr 4e-4; 8,000 steps |
| LoKr | The originally-selected scale-16 point at LR 2e-4 was unstable: 2 of 4 seeds diverged to a dead ~16-nat plateau, so its 3-seed aggregate was outlier-dominated (+4.249 ± 8.098). Re-selected under the AUTORESEARCH §3b stability gate: LR {1e-4, 5e-5} at scale 16 were each run on three fresh selection seeds (2801–2803); both converged 3/3, and LR 1e-4 had the better mean (−0.424 vs −0.377), so it was selected. Three disjoint confirmation seeds (2811–2813) converged 3/3. State ledgers: `reselect/state.jsonl`. | scale 16; lr 1e-4; 6,000 steps |
<!-- canonical-results:t2a-selection-markdown:end -->

**Reproduce:** `scripts/reproduce/task_t2a_lora.sh`, `scripts/reproduce/task_t2a_ia3.sh`, and
`scripts/reproduce/task_t2a_fourierft.sh`
accept `[SEED] [GPUS_HYPER] [GPUS_STATIC]`, run preflight, train the strip-def hypernetwork + static
reference, and run both evaluations (sets `HF_HUB_OFFLINE=1`).

```bash
scripts/reproduce/task_t2a_lora.sh 1801 0,1,2,3 4,5,6,7
# results (per-task + __aggregate__ rows):
#   results/repro/t2a_lora_scale22.627417_lr1e-4/hyper/s1801/heldout_sni_ce_full21.jsonl
#   results/repro/t2a_lora_scale22.627417_lr1e-4/hyper/s1801/heldout_sni_acc.jsonl

scripts/reproduce/task_t2a_ia3.sh 1901 0,1,2,3 4,5,6,7
#   results/repro/t2a_ia3_scale16_lr4e-4/hyper/s1901/heldout_sni_ce_full21.jsonl
#   results/repro/t2a_ia3_scale16_lr4e-4/hyper/s1901/heldout_sni_acc.jsonl

scripts/reproduce/task_t2a_fourierft.sh 5111 0,1,2,3 4,5,6,7
#   results/repro/t2a_fourierft_scale16_lr1e-4_confirm/hyper/s5111/heldout_sni_ce_full21.jsonl
#   results/repro/t2a_fourierft_scale16_lr1e-4_confirm/hyper/s5111/heldout_sni_acc.jsonl
```
