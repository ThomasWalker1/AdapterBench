# Codec autoresearch protocol

This protocol searches for a **codec's best measured free-hyperparameter operating
point** in AdapterBench without changing the scientific comparison. It is a runbook for
an agent or human experiment driver; it is not permission to alter the benchmark
substrate or to auto-commit results.

The question remains: *does a generated adapter's shape matter when the conditioning
path is held fixed?* Autoresearch may tune how a shape is optimized, but never the
information, model, data, or control that make that question meaningful.

## Immutable substrate

Before a search, record the setup name, codec name and source revision, model revision,
and exact reproduction command. Do not tune any of the following per codec:

- frozen interpreter or any interpreter parameter;
- conditioner, early-exit depth, Perceiver configuration, or hypernetwork trunk;
- training documents/tasks, templates, tokenization, or data limits;
- hook sites where codecs are semantically compatible;
- evaluator, decoding, held-out examples, or condition control;
- generated-scalar budget or another shape-defining parameter such as an internal
  shape dimension.

Every generated tensor remains live through the hooked forward; hooks are removed in
`finally`. Checkpoints, logs, and full artifacts remain under `results/` and are never
committed.

## Searchable parameters

The only standard free hyperparameters are:

| Parameter | Search rule |
|---|---|
| codec output scale | Required sweep; derive a codec-specific geometric ladder from its own identity-scale convention and observed stability, never import a value or range from another codec. Sweep via the generic `--codec-scaling` flag (uniform across codecs); the per-codec flags exist only for recorded historical commands. |
| learning rate | Optional geometric sweep after scale has a viable region. |
| warmup fraction | Optional small discrete sweep after scale/LR are viable. |
| training steps | Optional continuation from restart-safe checkpoints; compare equal final budgets across candidate configurations. |

Shape identity is not a free parameter. Adding an internal shape dimension, targets,
conditioner capacity,
training data, or a new regularizer creates a different comparison and requires a
separate preregistered budget point rather than optimization by this loop.

## Search state

Maintain a small append-only state file under
`results/autoresearch/<setting>/<codec>/state.jsonl` (scratch) with one object per
trial. It must include:

```json
{
  "phase": "scale_locator",
  "setting": "d2a",
  "codec": "ia3",
  "seed": 777,
  "free_hparams": {"scale": 32, "learning_rate": 0.00004, "warmup_frac": 0.03, "steps": 12000},
  "command": "exact shell command",
  "artifact_root": "results/...",
  "status": "complete",
  "selection_metric": 1.0,
  "control_metric": 0.0,
  "helpfulness_metric": 1.0,
  "notes": "why this trial was continued or rejected"
}
```

Never overwrite a trial record. A rerun receives a new record that points to the
resumed checkpoint and preserves the original command.

## The loop

### 1. Preflight

1. Run `uv run pytest -q`, `uv run adapterbench validate`, and the codec's short GPU
smoke on the active setting.
2. Confirm the frozen interpreter has no trainable parameters, generated tensors are not
detached, and the control is emitted by the evaluator.
3. Run `adapterbench preflight` and verify the requested GPUs are idle. Use a detached
`tmux` session for jobs longer than an interactive window.

Do not start a long sweep if the smoke fails. Repair the execution issue first and rerun
the smoke; changing compilation or launcher settings is allowed only when it leaves the
model forward and all benchmark semantics unchanged.

### 2. Scale locator

Start with one fixed scout seed and all other free parameters at the setting's validated
defaults. Sweep a finite geometric scale ladder in parallel when hardware permits. The
ladder must be chosen without using another codec's results.

- Initial ladder: declare at least five log-spaced points around the codec's own
  identity-scale convention. Its numerical limits are practical safety limits only, not
  evidence transferred from another codec.
- If the best point is an endpoint, extend outward by a constant log-space factor until
  the next point is worse, training becomes numerically invalid, the declared safety
  limit is reached, or **two extension steps have been taken in that direction** —
  whichever comes first. If an endpoint is reached because of the safety limit or the
  two-step cap, report that limitation rather than calling the result optimal.
- **The two-extension cap.** At most two steps outward from the declared ladder, counted
  independently per `(codec, axis, direction)`: extending a ladder downward and extending
  it upward are separate searches, and an LR ladder is separate from a scale ladder, so
  each gets its own budget of two. After the second step, stop and record the boundary as
  a declared limitation even if the endpoint is still winning.

  The cap exists because "extend while the endpoint wins" is unbounded in exactly the case
  where the endpoint is winning for the wrong reason. A `matched − control` metric can be
  inflated without limit by driving the control toward a no-op (see §"T2A: metric, data
  split, and the independently selected control", rule 3), so a mechanical extension rule
  will chase that artifact as far as the safety bound allows. The diagnostic is whether the
  control's score is approaching the frozen model's: a control that has degenerated toward a
  no-op turns `matched − control` into `matched − frozen`, and rungs in that region are control
  failures rather than selection evidence.

  A cap reached is a finding, not a failure: record which axis and direction were still
  improving, so a reviewer can decide whether a fresh preregistered ladder centred
  elsewhere is warranted. Do not quietly resume the same chain in a later phase.
- **An endpoint only "wins" if it wins by more than the noise.** "Extend while the endpoint
  wins" has no meaning until *wins* is defined, and a bare argmax comparison makes an
  arbitrarily small gain mandate another run. So: an endpoint counts as winning only if it
  improves on its inward neighbour by **more than the measured seed standard deviation for
  that codec and role**. Within that, the two points are a **tie**: declare the axis closed at
  the interior point, record the plateau, and do not extend.

  Use the pooled within-configuration seed SD measured for that codec and role, excluding
  non-converged configurations (their spread is a divergence rate, not measurement noise). If
  no configuration of that codec and role has two or more seeds yet, borrow the largest SD
  measured across codecs as a conservative stand-in and record that it was borrowed.

  This closes the other end of the runaway the two-extension cap bounds: a curve that has
  asymptoted keeps producing nominal argmax improvements indefinitely, and without a threshold
  each one mandates another rung.

  The same threshold applies wherever one configuration is said to beat another, not only at
  ladder ends — including which point is promoted to §3b's selection seeds. Two configurations
  within one seed SD are tied, so both are candidates rather than one being "the" argmax; see
  §3b.
- Use the setting's behavioral controlled metric, not training loss, to select the
  provisional winner. A nonzero raw score with a failed control is rejected.

The scout is for locating a region, not for the final evidence. Retain every scale's
metrics and logs, including failures.

### 3. Optional LR/warmup/step search

Only enter this phase if the scale locator produces a viable controlled signal.

1. Test learning rates around the setting default, e.g. one lower and one higher
   geometric neighbor, at the provisional scale.
2. Test warmup only if LR is unstable or if the setting's learning curve shows a delayed
   onset; use a small declared set.
3. Continue candidates to a common final step budget only when their checkpointed
   trajectories are still improving. Do not compare a 36k endpoint against another
   candidate's 12k endpoint.

Use one scout seed only to **prune** an axis coarsely. Stop an axis when the best point is
interior and its neighbors are worse, or when all points fail the helpfulness floor.
This makes the search bounded and auditable rather than an endless quest for a lucky run.
The configuration finally promoted to confirmation is never chosen on the scout seed
alone — that decision is made on multiple seeds under the stability gate in §3b.

### 3a. Multi-fidelity promotion

When a setting is expensive, use checkpointed common-budget rungs rather than giving every
candidate the full budget. A candidate's optimizer, scheduler, and generated-adapter state
must resume from the same checkpoint at each promotion; do not restart it at a later rung.

For D2A, the standard locator rungs are 8k, 16k, and 36k steps. Evaluate at rung boundaries,
apply the shortest in-distribution helpfulness/control gate, and promote by the declared controlled
hard-length AUC. Keep at least three candidates after the first rung and one after the second;
then run fresh three-seed confirmation only for the selected final configuration. Use a fixed
absolute warmup-step count derived from the final budget so promotion does not redefine the
optimizer schedule.

If an early rung gives every candidate the same non-helpful score, it contains no selection
signal: retain all candidates for the next declared rung rather than applying an arbitrary
tie-break. The helpfulness/control gate remains required as soon as any candidate has a signal,
and remains mandatory for any reported result.

### 3b. Stability gate and multi-seed final selection

A single scout seed rewards a configuration's *best-case* run, so it may only prune
obviously-weak regions of an axis. The configuration actually **promoted to confirmation
must be chosen on multiple seeds**, using the same statistic the leaderboard reports (a
mean over seeds), never a one-seed peak. Concretely:

1. Narrow each axis coarsely on the scout seed as in §2–§3.
2. **Form the tied set by the same materiality threshold §2 uses**: every configuration within
   one measured seed SD of the scout-seed argmax is *tied* with it. A scout seed cannot
   separate points inside its own noise, so treating its argmax as a decision over those
   points is a one-seed peak dressed up as a choice.

   Then **declare the argmax the operating point and run only it** on at least three selection
   seeds, disjoint from the scout and from the later confirmation seeds. Do *not* run every
   tied configuration: they are tied, so choosing among them cannot change the reported effect
   by more than the spread already being reported, and paying 3 seeds each to discover that is
   spend without an inference attached. Record the size of the tied set and the fact that the
   operating point was chosen within noise, so the report says "chosen from N statistically
   indistinguishable points" rather than implying a resolved optimum.

   Run a second tied candidate **only** when there is a reason to prefer knowing which wins:
   the declared operating point **fails the stability gate** below (then promote the next tied
   candidate and re-gate — adaptive fallback, not a shortcut), or the report intends to *claim*
   one configuration beats another, which requires the seeds to support that claim.

   This keeps the threshold's benefit — no false precision about which point is best — without
   letting a wide tie multiply the training budget. Measured example: on the T2A grid the tied
   sets ran 1–5 configurations per (codec, role), 27 in total; running all of them at 3 seeds
   would have cost 81 trials to choose between points no seed count in reach can separate.
3. Apply a **stability gate**: a candidate is eligible only if it *converges* on at least
   ⌈2/3⌉ of its selection seeds, where a converged run is finite, clears the setting's
   helpfulness floor, and shows no training divergence (e.g. a loss that climbs to and
   stays on a degenerate plateau). Record each candidate's divergence rate as a
   first-class selection metric alongside its controlled score.
4. Among gate-passing candidates, select the best mean controlled metric. If no candidate
   passes the gate, the codec has no stable operating point in the searched space — a
   legitimate negative result, reported as such rather than by selecting an unstable point
   on a lucky seed.

The gate exists because selection and reporting must optimize the same quantity: a
configuration selected on a one-seed peak but reported as a multi-seed mean is a
winner's-curse estimate.

### 4. Multi-seed confirmation

Run the selected configuration at **at least three independent confirmation seeds**.
These seeds must be disjoint from the scout and from the §3b selection seeds. All
confirmation runs use the same final free hyperparameters and final step budget. They may
run in parallel on distinct GPUs.

Compute mean and standard deviation only from the confirmation set. The scout and §3b
selection seeds may be shown in a scale-selection appendix or scratch record, but are not
silently pooled into a confirmation result selected from them. If instability observed in
a confirmation set triggers a re-tune, the re-selected configuration earns a **fresh**
confirmation set: the seeds that revealed the instability may not double as its
confirmation, or the confirmation is no longer held out.

### 5. Decide and publish

The result is eligible for a leaderboard row only if all are true:

- controlled behavioral metric is reported as `matched − control`, and the metric is
  behavioral rather than the training objective;
- `matched`, `control`, and their difference are each reported, not the difference alone;
- the control was selected independently, on its own score, under this same protocol;
- selection and reporting used disjoint data, and no hyperparameter — including step
  budget — was chosen against the reporting split;
- matched also beats the frozen helpfulness floor;
- scale was swept and the chosen point is recorded;
- the selected configuration passes the §3b stability gate, and its selection-seed
  divergence rate is recorded;
- at least three confirmation seeds report variation, and the spread is small relative to
  the between-shape differences being claimed;
- difficulty-axis output is present;
- exact commands, seeds, hyperparameters, provenance, and compact aggregate artifacts
  are recorded.

Append both setting rows only after both T2A and D2A pass this protocol. A losing codec
is still published with its measured negative or null result. The human reviews the
scratch artifacts, updates canonical compact metrics and leaderboards, runs drift checks,
and creates the commit.

#### Canonical selection trail

Finalization also copies a **compact selection trail** into the codec's canonical result
record: the protocol name, hashes and repository-relative paths of the append-only state
ledgers, and a plain-language account of the selected point and rejected boundaries. The
leaderboard renders this trail beneath the headline row. Keep the full commands and every
candidate metric in `results/autoresearch/.../state.jsonl`; never commit checkpoints or
verbose logs. Historical probes from a different locked setting may be retained as history,
but must not be described as selecting the current leaderboard configuration.

## Retroactive audit of already-published codecs

Tightening this protocol does not by itself invalidate a codec selected under an earlier
version. A stricter selection changes an outcome only for a codec whose earlier
single-seed choice hid an instability; for a stable codec the single-seed peak and the
multi-seed robust point coincide. Audit each already-published codec against the §3b
stability gate using its **existing** confirmation seeds — no re-run. A codec that passes
the gate on its committed seeds keeps its row, and the audit is noted in its selection
trail. Only a codec that *fails* the gate is re-selected under §2–§3b and re-confirmed.
The comparison stays fair because the substrate is unchanged; the gate keys on training
non-convergence, not on a setting's inherent metric variance (D2A retrieval's
phase-transition spread across seeds is not a divergence).

## Setting-specific selection rules

| Setting | Select on | Helpfulness floor | Required final evidence |
|---|---|---|---|
| T2A | Highest `matched − static*` **ROUGE-L on the selection split** (the 10 in-distribution `lol_` eval tasks, **examples 40+ only**) | matched ROUGE-L above frozen, on the same split | 11 genuinely held-out SNI tasks, all examples; ROUGE-L primary against an **independently selected** static; CE and exact match reported as appendix data points. |
| D2A | Highest controlled hard-length score after passing the shortest in-distribution gate: normalized log-length AUC over every declared doubled evaluation length after that gate | matched accuracy at the shortest in-distribution length higher than frozen | realistic-prose numeric-decoy NIAH length curve through every declared hard bin; context-swap near zero at every reported length. |

For D2A, never select on language-model loss: retrieval can remain at chance after loss is
nearly zero. The shortest in-distribution score establishes that the codec is helpful and
condition-dependent; when it reaches a ceiling, it must not decide between otherwise
viable configurations.

### T2A: metric, data split, and the independently selected control

Three rules, all introduced together because each is unsound without the others.

**1. Behavioral metric, not the training objective.** The selector and the headline are
**ROUGE-L** — Super-NaturalInstructions' own aggregate metric, and the metric upstream's
evaluation protocol uses. Teacher-forced cross-entropy is the quantity the trainer
optimizes, so selecting on it measures optimization quality and systematically favors
shapes whose inductive bias reduces token-level likelihood whether or not behavior
changes. CE keeps two jobs it is genuinely good at and loses the third: it remains a cheap
**divergence detector** (non-finite loss, a dead plateau) and a cheap **eligibility gate**,
but it never selects. Normalized exact match is computed from the same generations at no
extra cost and reported alongside; it is a stricter, unambiguous cross-check that catches
ROUGE-L partial-credit inflation, but it is near-meaningless on the open-ended tasks and so
is never the selector either.

**2. Selection and reporting are disjoint on two axes: task *and* example.** `eval_ds_info`
contains 21 `lol_` tasks, but only **11 are absent from `train_ds_names`**. The other 10 are
trained on, so they are in-distribution monitors, not held-out generalization:

- **selection split (10 tasks, in-distribution), examples 40 and up:** `lol_084, lol_140,
  lol_275, lol_636, lol_705, lol_717, lol_742, lol_1198, lol_1448, lol_1711`
- **report split (11 tasks, genuinely held out), all examples:** `lol_035, lol_039, lol_202,
  lol_304, lol_362, lol_614, lol_701, lol_706, lol_710, lol_726, lol_1557`

**The example offset is mandatory, not a refinement.** Every vendored `lol_` task draws from
one `train[:10000]` HuggingFace split, and both the trainer (`LolSFTDataset`) and the
evaluators select a **leading range** of it. Training uses `--limit 40`, so evaluating a
selection-split task at offset 0 does not merely test in-distribution behavior — it **replays
the exact training examples**, and every evaluation at 24, 32, or 40 examples/task is a strict
subset of what the model was fit on. So:

- score the selection split with `--example-offset 40`, i.e. examples 40 onward, which no run
  has ever trained on. All ten tasks have at least 42 unused examples (17,686 in total; the
  binding task is `lol_742` with 42), so a uniform 32-examples/task instrument fits, and 42 is
  the ceiling for a uniform one. Above that, allow uneven per-task counts rather than dropping
  tasks: the aggregate is an unweighted mean over per-task scores, so uneven `n` costs
  precision on the small tasks but does not bias the mean.
- if the training `--limit` ever changes, the offset changes with it. The offset is recorded in
  every evaluation row so a mismatch is auditable rather than invisible.
- the **report split needs no offset**: those 11 tasks are absent from `train_ds_names`, so
  none of their examples was ever trained on.
- the three **descriptions** used at evaluation are already disjoint from the 128 the trainer
  consumes (verified: 0/3 overlap on all ten selection tasks). Only *task identity* is shared
  between training and the selection split.

Every free hyperparameter — for the hypernetwork *and* for the static — is chosen on the
selection split only. The report split is touched once, by the confirmation seeds of the
already-selected configuration, and never during a sweep. Nothing may be tuned against it,
including step budget and the decision to extend a ladder. Because hyperparameters are still
chosen on in-distribution *tasks*, record that as a known limitation rather than claiming a
clean dev set; the alternative — carving a dev set out of the 479 training tasks — would change
the immutable substrate and is out of scope for this loop. The example offset removes the
replay term of that limitation, which is the largest one, but not the task-identity term.

**Why the offset is mandatory rather than optional.** Replay does not merely add noise. A larger
output scale has more capacity to fit the specific training examples, so scoring them again inflates
high-scale rungs more than low-scale ones — and it inflates the task-agnostic *control* more than the
conditioned adapter, because the control has nothing else to fit. The combined effect biases the
scale argmax upward and biases the reported difference toward zero. The residual task-identity term
remains unmeasurable without the report split; record it as a declared limitation and do not test it,
since testing it spends the split on a sweep.

**Difficulty control for the offset.** When introducing or changing an example offset, check the
frozen (zeroed-adapter) score on both ranges. If frozen is unchanged, the two ranges are equally
hard and any drop is specific to the adapted model, i.e. memorisation rather than a harder sample.

**3. The static control is selected independently, on its own score.** The static reference
must be swept over the same axes and under the same stability gate as the hypernetwork, and
selected to maximize **its own** selection-split ROUGE-L — never to maximize the gap.

`matched − control` is **not** non-gameable, and this is the failure mode. It cannot be gamed by
sabotaging a wrong-condition case, but it **can** be inflated by handicapping the control's
optimization — so a rung where the control merely under-trains will outrank a rung where the
adapter is genuinely better. The clearest measured case is `steering`, whose static reference
trains the steering vector `v` directly, making the achieved intervention `scale·||v||`
rate-limited by scale: at scale 0.0625 its static is indistinguishable from no adapter at all,
and the yoked metric ranked that rung **first of nine** by a wide margin. Under this rule the
same rung ranks fifth. Earlier revisions of the protocol had to exclude such points by hand, via
a convention recorded in one codec's selection trail; selecting the control independently makes
the exclusion **structural**, because the quantity subtracted is the shape's best static
regardless of rung. A degenerate control can no longer win.

Yoking the static to the hypernetwork's chosen scale/LR/steps gives the search a structural
incentive to settle where the control is weak, because the reported quantity is a difference the
search is maximizing. Under this rule the headline is `best hypernetwork of shape S − best static
of shape S`, which is a deliberately conservative estimate of conditioning.

Report `matched`, `static*`, and their difference as separate columns, never the difference
alone. A large gap over a mediocre matched score means the control moved, not the adapter,
and the reader must be able to see that without recomputing it.

**Variance.** The report split is scored with enough examples per task and enough
confirmation seeds that the seed spread is small relative to the between-shape differences
being claimed. A spread that overlaps the gap between two shapes does not support ranking
them; increase examples per task (which reduces within-seed sampling noise) and seeds (which
tighten the estimate of the mean) rather than reporting the ranking anyway.

## Stop conditions

Stop and preserve the negative result when:

- every bounded scale point fails the control or helpfulness floor;
- the selected configuration has been confirmed across three seeds and common step
  budgets no longer improve its controlled metric materially;
- a proposed improvement requires changing an immutable substrate item.

At that point, move to the next codec rather than hiding the result or expanding the
search without a new, reviewed protocol.
