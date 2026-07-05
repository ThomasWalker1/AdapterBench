# Benchmark contract

The unit of comparison is a **trial within a setup**. A setup fixes the
conditioning information, hypernetwork shell, frozen interpreter, tasks,
training budget, evaluator, and decoding. A trial changes the generated PEFT
representation and, where exact payload matching is impossible, one declared
budget point.

PAW and Text-to-LoRA are separate setups. Their absolute scores are not pooled:
they use different conditioning, objectives, interpreters, task distributions,
and evaluators. The useful cross-setup question is whether a representation's
relative behavior repeats across protocols.

## Interfaces

1. A task loader emits `TaskExample(task_id, condition, input, target, family)`.
2. A paper-specific `HypernetworkBackend` converts one condition per task into
   an `AdapterArtifact`. Artifacts record representation, exact generated
   parameter count, generation latency, format, and provenance.
3. A `DownstreamEvaluator` attaches the artifact to the frozen interpreter and
   scores that task's held-out examples.
4. `EvaluationResult` records downstream metrics plus resource measurements.

This artifact boundary accommodates PAW's end-to-end compiler, Text-to-LoRA's
description encoder and reconstruction training, and future document/context
hypernetworks without pretending their generators are identical.

## Required controls and baselines

Every completed representation sweep reports:

- identical train/validation/test task identities and complete-family holdouts;
- the frozen interpreter and exact revision;
- target modules, tensor shapes, generated parameter count, compiler parameter
  count, optimizer updates, tokens/examples, and GPU-hours;
- downstream metric per task and macro average, not adapter reconstruction alone;
- generation latency, interpreter throughput, and peak memory;
- frozen-interpreter, prompted/ICL, task-specific oracle-adapter, and released
  paper baseline where available;
- at least three seeds and failures/non-finite steps;
- both a payload-matched sweep and each method's recommended configuration.

Reconstruction loss is a diagnostic. The primary result is always frozen
interpreter performance on held-out downstream examples.

## Representation catalog

- LoRA and freeze-A/generate-B LoRA
- FourierFT coefficients at fixed frequency locations
- Kronecker factors (maintained LoKr implementation)
- IA3 activation-scaling vectors
- prefix-tuning per-layer KV tensors

The manifests live in `configs/adapters`. Adding a proposal requires an adapter
manifest, a generator output codec, exact parameter accounting, and a test that
materializes the result in the chosen interpreter.

