# Plan for a genuinely inference-time adaptive image setting

**Status: research plan; there is currently no active image-domain AdapterBench setting.**

## 1. Admission criterion

An image setting belongs in AdapterBench only if all of the following are true:

1. A hypernetwork receives a per-instance condition `c` and emits the tested adapter in
   one forward pass. There is no per-instance gradient descent at evaluation time.
2. The frozen generator cannot access `c` through its ordinary inputs. The adapter is the
   only information path from `c` to the generated image.
3. For the same generator input and latent, two conditions require incompatible outputs.
   A prompt-generic or static adapter therefore cannot solve both.
4. A matched adapter beats both a same-shape static adapter and a condition-shuffled
   adapter. Raw quality or reward improvement is insufficient.
5. Training data, hypernetwork trunk, optimizer, generator, hook sites, evaluator, and
   generated-parameter budget remain fixed when codec shape changes.

This definition separates an adapter that is merely recomputed at inference time from one
whose behavior causally depends on the inference-time condition.

## 2. Recommended setting: reference-image-conditioned appearance transfer

The best first candidate is **reference image → hypernetwork → generator adapter**.

- **Condition available only to the hypernetwork:** one or more reference images showing
  a subject or appearance.
- **Ordinary generator input:** a text prompt describing a new scene, pose, or composition.
  It contains a neutral placeholder such as “the subject,” not the subject identity or a
  caption derived from the reference.
- **Required behavior:** preserve the reference identity/appearance while following the
  new text instruction.
- **Why conditioning is necessary:** hold prompt and latent fixed, then pair two different
  reference identities. The requested images are mutually incompatible, and the frozen
  generator has no other route to learn which identity is requested.
- **Adapter:** a hypernetwork emits weight-space updates for a fixed set of UNet/DiT
  attention projections. LoRA is the feasibility baseline; alternative codecs later use
  the same targets and generated-scalar budget.

This is preferable to prompt-conditioned reward tilting because the generator does not
already see the conditioning signal. It is preferable to reward-ID conditioning because
the condition is a natural per-instance input rather than a small artificial menu of
objectives.

The main risk is capability: a small generated adapter may not inject an unseen identity
into a frozen generator. The plan therefore starts with a cheap capacity oracle and stops
early if that upper bound is weak.

## 3. Alternatives, in priority order

1. **Reference style/material transfer.** Easier than exact identity and still
   structurally non-redundant. Use paired references with the same prompt/latent and score
   style similarity separately from content preservation. The danger is that broad style
   labels may collapse to a small static mixture; references must be held out and
   sufficiently fine-grained.
2. **Layout-conditioned generation.** Give only the hypernetwork a segmentation map,
   bounding-box layout, or pose skeleton while the generator receives a generic text
   prompt. This has a very clean condition-swap control, but global weight-space LoRA may
   be a poor carrier of spatial information; do not change the hook/output interface just
   to manufacture a positive result.
3. **Reward-conditioned tilting.** A reward embedding visible only to the hypernetwork and
   mutually incompatible reward targets should produce a positive result. Keep this as a
   diagnostic or fallback, not the preferred core setting: a small fixed reward menu is
   task-conditioned adaptation, not rich per-image adaptation.

Do not revive same-prompt-to-hypernetwork-and-generator tilting or deliberately degraded
prompts as the core task. The former is redundant; the latter creates necessity by
withholding information from a generator input that naturally ought to contain it.

## 4. Staged go/no-go protocol

### Stage 0 — freeze the claim and controls

Before training, fix:

- train/validation/test identity or style splits;
- generator prompts and latents, paired across conditions;
- adapter targets, rank/output budget, and scale sweep;
- the identity/style metric and independent content/fidelity metric;
- static, shuffled-condition, frozen-generator, and reference-retrieval controls;
- a smallest-effect threshold derived from evaluator repeatability, not chosen after the
  run.

No benchmark integration occurs at this stage.

### Stage 1 — direct-optimization capacity oracle

For 16–32 reference conditions, directly optimize one LoRA per condition while keeping
the eventual hook sites, rank, generator, and objective fixed. This is not a benchmark
result; it asks whether the frozen generator plus adapter has enough capacity at all.

**Go:** on held-out prompts/latents, the per-condition oracle beats frozen and a shared
static LoRA on reference identity/style while staying inside the preregistered
content-fidelity bound.

**No-go:** if individually optimized adapters cannot carry the condition, a hypernetwork
will not rescue the setting. Stop or move from identity to the easier style candidate.

### Stage 2 — small hypernetwork causality probe

Train LoRA generation on roughly 200–500 conditions, then evaluate held-out conditions
with the same prompt and latent rendered under matched and shuffled references.

Report, with paired bootstrap confidence intervals:

- `matched − static`;
- `matched − shuffled-condition`;
- `matched − frozen`;
- content/fidelity change;
- top-1 reference retrieval or verification success.

**Go:** both primary differences are positive with 95% confidence, exceed twice the
measured evaluator/resampling noise floor, and improve reference identification by at
least 10 percentage points without violating the fidelity bound.

**No-go:** a positive `matched − static` with null `matched − shuffled` is a generic
hypernetwork advantage and is rejected, exactly as in the retired prompt-conditioning
probe.

### Stage 3 — robustness and difficulty

Only after Stage 2 passes:

- run at least three seeds;
- sweep adapter scale per codec;
- evaluate a graded axis such as reference/query pose distance, number of reference
  shots, occlusion, or identity similarity;
- test unseen identities/styles and unseen prompt templates;
- confirm the condition signal is not recoverable from the text prompt or dataset order;
- measure generation latency and exact emitted parameter count.

### Stage 4 — integrate one LoRA baseline, then compare shapes

Create an active image package, CLI, reproduce script, tests, and leaderboard only after
the causal gate passes. Lock that substrate before adding a second codec. A codec row must
use identical condition features, hypernetwork trunk, generator, training examples,
evaluator, controls, hook sites where semantically compatible, and emitted-scalar budget.

## 5. Decision tree

1. Try reference-identity capacity oracle.
2. If identity capacity is weak, try reference-style transfer.
3. If the direct oracle works but the hypernetwork fails the shuffle control, report a
   hypernetwork-generalization boundary result and stop.
4. If reference conditioning passes, build the full image setting.
5. Use reward conditioning only if the project explicitly wants a guaranteed-positive
   diagnostic; label it honestly and keep it out of the natural image-adaptation claim.

Negative results at Stages 1 or 2 are valid outcomes. The benchmark should not engineer a
positive image row at the cost of making its conditioning claim artificial.
