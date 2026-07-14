# Benchmark contract

The unit of comparison is a **trial within a setup**. A setup fixes the
conditioning information, hypernetwork shell, frozen interpreter, tasks,
training budget, evaluator, and decoding. A trial changes the generated PEFT
adapter and, where exact payload matching is impossible, one declared
budget point.

Different setups are not pooled: their absolute scores use different conditioning,
objectives, interpreters, task distributions, and evaluators. The useful cross-setup
question is whether an adapter's relative behavior repeats across protocols.

## Interfaces

1. A task loader emits `TaskExample(task_id, condition, input, target, family)`.
2. A paper-specific `HypernetworkBackend` converts one condition per task into
   an `AdapterArtifact`. Artifacts record adapter, exact generated
   parameter count, generation latency, format, and provenance.
3. A `DownstreamEvaluator` attaches the artifact to the frozen interpreter and
   scores that task's held-out examples.
4. `EvaluationResult` records downstream metrics plus resource measurements.

This artifact boundary accommodates the live-hook SFT mechanism's in-process generation
and future document/context hypernetworks without pretending their generators are
identical.

## Required evaluation invariants

Every leaderboard entry satisfies the same four rules:

1. **Use a behavioral metric with a built-in control.** The headline is always
   `matched - control`, never training or reconstruction loss. T2L uses a mismatched
   task description, D2L uses a context-swapped document, and I2P uses a reward-swapped
   adapter.
2. **Sweep adapter scale.** Report the shape at its best measured scale so a comparison
   does not merely rank incompatible defaults.
3. **Include a graded difficulty axis.** D2L reports length generalization; I2P reports
   the reward/fidelity tradeoff; T2L reports across task families with different frozen
   headroom. A setting where every shape saturates is not discriminative.
4. **Run at least three seeds.** Report mean and spread, including failures and
   non-finite runs.

The fixed substrate (data, conditioner/hypernetwork shell, frozen model, hook protocol,
evaluator, and control) remains identical within a setting. Per-entry free optimization
hyperparameters are recorded alongside the result; shape-defining hyperparameters such
as rank are fixed. See `leaderboards/README.md` for the exact row format.

## Adapter catalog

The baseline catalog registers a single codec — **LoRA** — as the reference
representation validated across all settings. Additional shapes (other low-rank
factorizations, spectral/Fourier coefficients, activation-space vectors, …) are added
one at a time; previously explored shapes live in git history.

A new shape requires a `GeneratedUpdateCodec` subclass, a `make_codec` registration,
an adapter manifest in `configs/adapters`, exact generated-parameter accounting, and
focused tests. It is evaluated through each applicable setting CLI at a manually chosen,
recorded free-HP configuration, then appended to the corresponding committed leaderboard.
There is no automated search, proxy score, or merge gate.
