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
  "setting": "d2l",
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
  the next point is worse, training becomes numerically invalid, or the declared safety
  limit is reached. If an endpoint is reached because of the safety limit, report that
  limitation rather than calling the result optimal.
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

For D2L, the standard locator rungs are 8k, 16k, and 36k steps. Evaluate at rung boundaries,
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
2. Re-run the surviving candidates (typically the top two) on **at least three selection
   seeds**, disjoint from both the scout and the later confirmation seeds.
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

- controlled behavioral metric is reported as `matched − control`;
- matched also beats the frozen helpfulness floor;
- scale was swept and the chosen point is recorded;
- the selected configuration passes the §3b stability gate, and its selection-seed
  divergence rate is recorded;
- at least three confirmation seeds report variation;
- difficulty-axis output is present;
- exact commands, seeds, hyperparameters, provenance, and compact aggregate artifacts
  are recorded.

Append both setting rows only after both T2L and D2L pass this protocol. A losing codec
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
non-convergence, not on a setting's inherent metric variance (D2L retrieval's
phase-transition spread across seeds is not a divergence).

## Setting-specific selection rules

| Setting | Select on | Helpfulness floor | Required final evidence |
|---|---|---|---|
| T2L | Lowest `matched − static` held-out CE (negative is better) | matched CE lower than frozen CE | 21 held-out SNI tasks; CE primary and generation accuracy corroborating; same-shape static adapter for every candidate. |
| D2L | Highest controlled hard-length score after passing the shortest in-distribution gate: normalized log-length AUC over every declared doubled evaluation length after that gate | matched accuracy at the shortest in-distribution length higher than frozen | realistic-prose numeric-decoy NIAH length curve through every declared hard bin; context-swap near zero at every reported length. |

For T2L, train the conditioned hypernetwork and the same-shape static reference together
for every candidate. For D2L, never select on language-model loss: retrieval can remain
at chance after loss is nearly zero. The shortest in-distribution score establishes that the codec is
helpful and condition-dependent; when it reaches a ceiling, it must not decide between
otherwise viable configurations.

## Stop conditions

Stop and preserve the negative result when:

- every bounded scale point fails the control or helpfulness floor;
- the selected configuration has been confirmed across three seeds and common step
  budgets no longer improve its controlled metric materially;
- a proposed improvement requires changing an immutable substrate item.

At that point, move to the next codec rather than hiding the result or expanding the
search without a new, reviewed protocol.
