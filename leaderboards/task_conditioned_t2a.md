# Language domain — task-conditioned generation (Text-to-Adapter)

**Task.** The hypernetwork receives a pooled embedding of a free-text task description and emits an
adapter specializing a frozen `gemma-2-2b` to that task. Trained from scratch on the
Lots-of-LoRAs / Super-Natural-Instructions 479-task decontaminated split with the live SFT objective
(plain cross-entropy). The task **definition is stripped from the input** (`--strip-task-def`), so the
description is the *only* route to the task — otherwise the frozen model reads the task straight from
its prompt and the description-conditioned adapter is redundant (every codec then scores ≈0).

**Metric — `matched − static*` ROUGE-L.** Matched = the adapter generated from the task's held-out
description, averaged over the 3 held-out description variants. `static*` = a single adapter of the
*same shape*, directly optimized on the same SFT data (a multi-task adapter, not hypernetwork-emitted),
which absorbs all task-*generic* help — and **selected independently, on its own score**, not yoked to
the hypernetwork's hyperparameters.

- **ROUGE-L (primary):** Super-NaturalInstructions' own aggregate metric — LCS F-measure over greedy
  generations. Higher is better; `matched − static* > 0` = conditioning helps.
- **exact match (appendix):** normalized exact match off the *same* decode pass. Stricter and
  unambiguous, so it catches ROUGE-L partial-credit inflation, but near-meaningless on the open-ended
  tasks, so never the selector.
- **cross-entropy (appendix):** teacher-forced CE, the quantity the trainer optimizes. It keeps two
  jobs it is good at — divergence detector and eligibility gate — and **loses selection**. On this
  split CE and ROUGE-L rank the shapes at Spearman −0.50, so a CE-selected point is not the
  behaviourally best point.

`matched − static*` is **not** non-gameable: it cannot be gamed by sabotaging a wrong-condition case,
but it *can* be inflated by handicapping the control's optimization — which is exactly why the control
is now selected on its own score. See `AUTORESEARCH.md` §"T2A: metric, data split, and the
independently selected control" rule 3 for the measured example.

**Data splits.** `eval_ds_info` holds 21 `lol_` tasks, but only **11 are absent from
`train_ds_names`**. Selection and reporting are disjoint on **two** axes:

- **selection split** — the 10 in-distribution tasks, **examples 40 and up**. Every free
  hyperparameter, for the hypernetwork *and* the static, is chosen here. The example offset is
  mandatory: the trainer consumes the leading 40 examples of the same `train[:10000]` split, so
  offset 0 replays training examples verbatim.
- **report split** — the 11 genuinely held-out tasks, all examples, 64 per task. Touched **exactly
  once**, by the confirmation seeds of an already-selected configuration. Every number in the table
  below comes from that single measurement.

**Free HPs:** scale, learning rate, warmup, steps — chosen separately for the hypernetwork and the
static. **Fixed substrate:** 128 training descriptions per task, the 3 held-out evaluation
descriptions, the strip-def input template, the task splits, the static-reference control.
**Fixed:** rank (shape identity).

<!-- canonical-results:t2a-markdown:start -->
| Shape | rank | scale | lr | steps | static\* scale / lr | seeds | matched | static\* | **matched − static\* (ROUGE-L)** | m − frozen | EM Δ | CE Δ (appendix) |
|---|:---:|:---:|:---:|---:|:---:|:---:|---:|---:|:---:|---:|---:|---:|
| LoRA | 8 | 1.414214 | 5e-5 | 8 000 | 22.627417 / 5e-5 | 3 | 0.321 | 0.272 | **+0.049 ± 0.027** | +0.267 | +0.025 | −0.917 |
| FourierFT | — | 0.25 | 1e-4 | 6 000 | 4 / 1e-4 | 3 | 0.295 | 0.226 | **+0.068 ± 0.044** | +0.241 | +0.043 | −0.973 |
| (IA)³ | — | 16 | 5e-5 | 8 000 | 16 / 8e-4 | 3 | 0.323 | 0.195 | **+0.128 ± 0.049** | +0.269 | +0.039 | −0.452 |
| LoKr | — | 0.25 | 2e-4 | 6 000 | 0.25 / 2e-4 | 3 | 0.315 | 0.238 | **+0.077 ± 0.033** | +0.261 | +0.012 | +0.498 |
| Steering | — | 1 | 1e-4 | 8 000 | 64 / 2e-4 | 3 | 0.322 | 0.204 | **+0.119 ± 0.038** | +0.268 | +0.065 | −0.158 |
<!-- canonical-results:t2a-markdown:end -->

Higher ROUGE-L is better. All five codecs clear the frozen helpfulness floor (frozen ROUGE-L 0.054)
by roughly 6×, and every one shows a positive `matched − static*`, so conditioning is real for every
shape tested.

**The shapes are not ranked, and mostly cannot be.** At the measured confirmation-seed spread only two
pairs separate at 2× the standard error of their difference: `(IA)³ > LoRA` (+0.079, 2.4×) and
`Steering > LoRA` (+0.069, 2.6×). **Every adjacent pair fails to separate**, and `LoKr` vs `(IA)³`
would need on the order of 90 seeds per codec. Read this table as five independent measurements of
conditioning against five independent controls, not as a league table. Ordering the middle of it would
be reporting noise.

### Selection trails

Each compact trail below points to append-only scratch ledgers containing the exact scout commands,
candidate metrics, artifacts, and rejected points. Only fresh confirmation seeds enter the headline.

<!-- canonical-results:t2a-selection-markdown:start -->
| Shape | compact audit trail | selected final configuration |
|---|---|---|
| LoRA | Scale moved from 22.627417 to 1.414214 and LR from 1e-4 to 5e-5; the static's own optimum stayed at scale 22.627417 but moved to lr 5e-5, with lr 2.5e-5 worse in both roles. Re-derived under the corrected T2A rules (AUTORESEARCH.md): ROUGE-L rather than the training objective, a static control selected independently on its own score, and selection data disjoint from the report split on BOTH task and example axes -- the selection split is the 10 in-distribution lol_ tasks at example offset 40, since training consumes the leading 40 examples of the same split. Every swept axis closed with an interior optimum; the operating point passed the §3b stability gate 3/3 on selection seeds disjoint from these confirmation seeds. State ledgers: `lora/state.jsonl`. | hyper: scale 1.414214, lr 5e-5, 8,000 steps; static\*: scale 22.627417, lr 5e-5 |
| FourierFT | Scale moved from 16 to 0.25 with LR held at 1e-4 (5e-5 and 2e-4 both worse), stable across both example offsets. The static's own optimum is scale 4 lr 1e-4, interior on every swept axis. Re-derived under the corrected T2A rules (AUTORESEARCH.md): ROUGE-L rather than the training objective, a static control selected independently on its own score, and selection data disjoint from the report split on BOTH task and example axes -- the selection split is the 10 in-distribution lol_ tasks at example offset 40, since training consumes the leading 40 examples of the same split. Every swept axis closed with an interior optimum; the operating point passed the §3b stability gate 3/3 on selection seeds disjoint from these confirmation seeds. State ledgers: `fourierft/state.jsonl`. | hyper: scale 0.25, lr 1e-4, 6,000 steps; static\*: scale 4, lr 1e-4 |
| (IA)³ | Scale stayed at 16 but the hypernetwork's LR moved from 4e-4 to 5e-5 (lr 2.5e-5 worse). The static's own optimum is lr 8e-4 -- the learning rate at which the hypernetwork catastrophically fails the helpfulness floor -- so a yoked control could never have been trained there; lr 1.6e-3 was worse, closing the axis. Re-derived under the corrected T2A rules (AUTORESEARCH.md): ROUGE-L rather than the training objective, a static control selected independently on its own score, and selection data disjoint from the report split on BOTH task and example axes -- the selection split is the 10 in-distribution lol_ tasks at example offset 40, since training consumes the leading 40 examples of the same split. Every swept axis closed with an interior optimum; the operating point passed the §3b stability gate 3/3 on selection seeds disjoint from these confirmation seeds. State ledgers: `ia3/state.jsonl`. | hyper: scale 16, lr 5e-5, 8,000 steps; static\*: scale 16, lr 8e-4 |
| LoKr | Scale moved from 16 to 0.25 and LR from 1e-4 to 2e-4, stable across both example offsets. The static's own optimum is also scale 0.25 lr 2e-4: lr 4e-4 tied within 0.0001 (seed SD 0.021) and lr 8e-4 was worse, so the axis closed at the interior point under §2's materiality threshold. Re-derived under the corrected T2A rules (AUTORESEARCH.md): ROUGE-L rather than the training objective, a static control selected independently on its own score, and selection data disjoint from the report split on BOTH task and example axes -- the selection split is the 10 in-distribution lol_ tasks at example offset 40, since training consumes the leading 40 examples of the same split. Every swept axis closed with an interior optimum; the operating point passed the §3b stability gate 3/3 on selection seeds disjoint from these confirmation seeds. State ledgers: `lokr/state.jsonl`. | hyper: scale 0.25, lr 2e-4, 6,000 steps; static\*: scale 0.25, lr 2e-4 |
| Steering | First confirmed T2A row for this codec: its earlier search stopped at a selection_promotion_declaration for scale 64 lr 1e-4 that was never run. Under the corrected rules the hypernetwork's optimum is scale 1 lr 1e-4 and the static's own optimum is scale 64 lr 2e-4 (lr 4e-4 worse). The codec's own ledger had already documented that a yoked control can be inflated by handicapping its optimization; under the independent control its degenerate scale-0.0625 point falls from rank 1 of 9 to rank 5. Re-derived under the corrected T2A rules (AUTORESEARCH.md): ROUGE-L rather than the training objective, a static control selected independently on its own score, and selection data disjoint from the report split on BOTH task and example axes -- the selection split is the 10 in-distribution lol_ tasks at example offset 40, since training consumes the leading 40 examples of the same split. Every swept axis closed with an interior optimum; the operating point passed the §3b stability gate 3/3 on selection seeds disjoint from these confirmation seeds. State ledgers: `steering/state.jsonl`. | hyper: scale 1, lr 1e-4, 8,000 steps; static\*: scale 64, lr 2e-4 |
<!-- canonical-results:t2a-selection-markdown:end -->

**Reproduce:** use the shared drivers in `scripts/reproduce/`:

```bash
# one confirmation seed (LoRA):
scripts/reproduce/task_t2a_lora.sh 1741 0,1,2,3 4,5,6,7

# full three-seed headline for one codec:
bash scripts/reproduce/t2a_reproduce_all.sh lora 0,1,2,3 4,5,6,7

# all five codecs (train, then score report split once):
.venv/bin/python scripts/t2a_confirmation_seeds.py
bash scripts/t2a_score_report_split.sh
```

Report-split scores land under `results/autoresearch/t2a/evaluation/report_scores/`.
The headline aggregate is `results/autoresearch/t2a/evaluation/confirmation_result.json`,
produced by `scripts/t2a_confirmation_result.py`. Per-codec wrappers (`task_t2a_ia3.sh`, etc.)
delegate to the same drivers; operating points and seeds are defined in
`scripts/reproduce/t2a_codec_config.sh`.
