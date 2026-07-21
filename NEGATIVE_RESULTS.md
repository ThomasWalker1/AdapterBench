# Negative results: settings that did not demonstrate genuine adaptivity

AdapterBench counts a setting as genuinely adaptive only when changing the condition
available to the adapter generator changes held-out behavior while the frozen
interpreter/generator input is held fixed. The decisive quantity is therefore
`matched − control`, where the control uses a shuffled condition, a wrong adapter, or a
same-shape static adapter as appropriate. A raw gain over the frozen model is not enough.

The investigations below were scientifically useful, but they do not qualify as active
benchmark settings.

## Summary

| investigated setting | control result | conclusion |
|---|---|---|
| Standard Text-to-LoRA, with the task definition still in the interpreter prompt | mean matched − junk accuracy: Qwen3-0.6B `+0.003`, Gemma-2-2B `+0.022`, Mistral-7B `−0.019`; all within roughly `±0.05` evaluation noise. Confirmed on Sakana's released checkpoints: matched − junk is `+0.029` (Gemma) / `−0.001` (Mistral) even though matched − frozen is `+0.011` / `+0.130` | the generated adapter can be generically helpful, but the task description is redundant and the matched adapter is not reliably better than a junk-description adapter |
| Prompt-conditioned image reward tilting | matched − shuffled ImageReward at LoRA scales 0.5/1/2/4: `−0.001`, `+0.011`, `≈0.000`, `+0.023` | the apparent matched − static gain was a generic hypernetwork/optimization advantage, not prompt-specific adaptation |
| Conditioned selective object erasure | the final UnHype-style objective showed no significant matched advantage over static or swapped controls; target and retained objects were suppressed almost identically | the condition induced mild proxy-level steering, but not reliable conditional deletion |
| Reference-image-conditioned exact chair identity | matched top-1 retrieval was `6.25%` at every tested scale, equal to frozen/static chance; wrong-adapter retrieval was `5.47%–6.25%` | even directly optimized per-identity LoRAs could not transmit identity, so a generated-adapter benchmark would compare codecs at a shared floor |

These are different failure modes. The first two have a redundant condition and fail a
condition-control audit. Selective erasure has a valid non-redundant condition but fails
the intended behavioral task. Reference identity fails an earlier adapter-capacity gate,
before training a hypernetwork would be informative.

## 1. Standard Text-to-LoRA: the interpreter can already read the task

The standard Text-to-LoRA SFT template gives the frozen interpreter
`{task_definition}\n\n{problem}` while also giving a description of that task to the
hypernetwork. The condition is therefore available through two paths:

```text
task description ──> hypernetwork ──> generated LoRA ──> interpreter
task definition  ──────────────────────────────────────> interpreter
```

A base-model diagnostic held the shipped recipe fixed and changed only the interpreter.
The mean matched-minus-junk accuracy gaps were:

| interpreter | matched − junk |
|---|---:|
| Qwen3-0.6B | `+0.003` |
| Gemma-2-2B | `+0.022` |
| Mistral-7B | `−0.019` |

All three are within the approximately `±0.05` evaluation noise. An earlier
under-powered three-seed run made the same issue especially clear: matched minus
mismatched accuracy was `−0.017`, `−0.021`, `−0.021`, and `−0.029` on ARC-Easy,
ARC-Challenge, HellaSwag, and BoolQ respectively. The adapters could improve raw
performance, but a wrong or meaningless description improved it just as much.

### Verification on Sakana's released checkpoints

The gaps above came from our own diagnostic training runs. To confirm the effect is a
property of the standard setting and not of our reimplementation, we loaded Sakana's
**released** Text-to-LoRA hypernetworks (`SakanaAI/text-to-lora`, revision
`6e571eda2188b216f027263cd28c99c0fdcf2fa3`), generated one LoRA per condition, and scored
each on 200 held-out examples per family under upstream's own prompt/answer-extraction
protocol. Only the text handed to the hypernetwork changed: `matched` (the family's own
task description), `shuffled` (a different family's real description), or `junk`
(`args.yaml`'s `additional_eval_descs`, e.g. `dogs;cats;bananas;`). The frozen interpreter
saw an identical prompt in every case.

| released checkpoint | matched − frozen | matched − shuffled | matched − junk |
|---|---:|---:|---:|
| `gemma_2b_t2l` (google/gemma-2-2b-it) | `+0.011` | `+0.022` | `+0.029` |
| `mistral_7b_t2l` (mistralai/Mistral-7B-Instruct-v0.2) | `+0.130` | `−0.009` | `−0.001` |

Means over ARC-Easy, ARC-Challenge, BoolQ, and HellaSwag. On Mistral the generated
adapter is genuinely and generically helpful (`matched − frozen = +0.130`), yet giving it
the *correct* per-task description buys nothing over a wrong task's description or
meaningless text (`matched − shuffled = −0.009`, `matched − junk = −0.001`). On Gemma every
gap is within the same `±0.05` noise band. The published checkpoints reproduce the
diagnostic: the adapter helps, the per-prompt condition does not.

This verification lives entirely outside the benchmark code, as a standalone record, in
`scripts/negative_results/t2l_released_prompt_ablation/` (per-family numbers in
`results/negative_results/t2l_released_prompt_ablation/*/ablation_summary.json`). It is not
a registered setting and does not import the active `adapterbench` package.

### The active setting, by contrast

This is a negative result about the **standard input-visible setting**, not the active
AdapterBench T2L setting. AdapterBench strips the task definition from the interpreter
input and uses the harder-to-game `matched − static` control. That redesigned setting
does pass: `matched − static = −0.723 ± 0.162` nats of held-out cross-entropy over three
seeds.

## 2. Prompt-conditioned image reward tilting: matched beats static, not shuffled

The image hypernetwork read a text prompt and emitted a per-prompt LoRA, but SD-Turbo
also received the same prompt. It appeared adaptive if evaluated only against a
separately optimized static LoRA:

| LoRA scale | matched − static ImageReward | matched − shuffled |
|---:|---:|---:|
| 0.5 | `+0.063 ± 0.026` | `−0.001 ± 0.006` |
| 1 | `+0.153 ± 0.036` | `+0.011 ± 0.008` |
| 2 | `+0.152 ± 0.036` | `≈0.000 ± 0.010` |
| 4 | `+0.200 ± 0.042` | `+0.023 ± 0.012` |

The shuffled-condition comparison changes only which prompt the hypernetwork sees.
Its near-zero result at every scale shows that the generated adapter did not need to
match the instance prompt. `Matched − static` alone would have incorrectly promoted a
generic parameterization or optimization advantage as adaptivity.

## 3. Selective object erasure: proxy steering did not become selective deletion

This was a stronger causal construction. SD-Turbo received the same two-object prompt
and latent, while only the hypernetwork received either `remove A` or `remove B`. The
two conditions demanded incompatible outputs, so redundancy was not the problem.

Early relative-score proxies did move in the requested direction. For example, at scale
2 the independent detector margin gave `matched − static = +0.0256 ± 0.0079` and
`matched − swapped = +0.0511 ± 0.0159`. But only `0.8%` of valid examples crossed the
actual selective-erasure threshold.

More task-aligned objectives exposed the failure:

- Absolute suppression reached `6.7%` selective success, while target confidence fell
  `0.113`, retained-object confidence also fell `0.071`, and CLIP-T fell `0.039`.
- The UnHype-style denoising target reached `11.7%–17.5%` apparent target deletion, but
  target and retained objects were suppressed almost identically
  (`0.363/0.361` and `0.407/0.403`), fidelity collapsed, and matched did not
  significantly beat static or swapped controls on the final task-valid comparison.

The setting demonstrated small condition-routed semantic steering, not dependable
per-request object removal. A proxy-only matched/swapped gap is not sufficient when the
claimed behavior fails an independent evaluator.

## 4. Reference-image identity: the direct LoRA capacity oracle failed

The proposed condition was a reference image of one of 16 held-out chair identities.
The generator received a neutral prompt, so identity was available only through the
adapter path. Before training a hypernetwork, one rank-4 LoRA was directly optimized for
each identity as an upper-bound capacity test.

The identity evaluator itself was viable: it retrieved the reference images at `98.44%`.
Nevertheless, across LoRA scales 0.5, 1, 2, and 4:

- matched generated-image retrieval stayed at `6.25%`, exactly 1-of-16 chance;
- frozen and shared-static controls also scored `6.25%`;
- wrong-adapter retrieval ranged from `5.47%` to `6.25%`;
- fidelity remained acceptable, ruling out wholesale prompt collapse as the explanation.

This was not a failed hypernetwork audit: Stage 1 deliberately used direct per-identity
optimization. It showed that the tested SD-Turbo attention-LoRA substrate could not
express held-out exact identity even under that stronger oracle. Training a condition
encoder or comparing codec shapes on top of this floor would not be informative.

## Structurally non-adaptive result: the shared HyperNoise image adapter

The original image reward-tilting result reported a real ImageReward gain of
`+0.160 ± 0.015`, but it used one directly optimized LoRA shared by every prompt. There
was no per-instance condition and therefore no meaningful shuffled-condition counterpart.
It is evidence for generic reward tilting, not for inference-time adaptivity.

## What is deliberately not counted as a negative

- **ByteDance HyperLoRA:** its released fidelity checkpoint passed the matched-vs-shuffled
  audit (`76.04%` matched retrieval versus `0%` shuffled). It was removed from the active
  codebase because the full image training stack was outside the desired benchmark scope,
  not because adaptivity failed.
- **Program-as-Weights/planning:** this was considered and later dropped, but no
  AdapterBench condition-control experiment was completed. It would be misleading to
  turn an untested idea into a negative empirical result.

## Provenance

The standard Text-to-LoRA diagnostic artifacts remain under
`results/repro/t2l_base_diag/`, `results/t2p_cond_ddp/`, and
`results/_archive/t2p_cond_long/`. The released-checkpoint verification (§1) is a
standalone, self-contained record under
`scripts/negative_results/t2l_released_prompt_ablation/` (see its `README.md` for exact
setup and reproduction commands), with per-family numbers in
`results/negative_results/t2l_released_prompt_ablation/{gemma_2b,mistral_7b}/ablation_summary.json`.
The disk-artifact "setting 1" code it was rebuilt from was removed from the active tree
in git commits `a8867de`/`64de45e` and can be recovered from `a8867de^`.

The retired image implementation and its detailed
write-up are recoverable from git commit `65b1593`:

```bash
git show 65b1593:archive/retired_i2p/README.md
git show 65b1593:PROJECT_PLAN.md
```

The reference-identity numbers are the completed Stage-1 report recorded before the
user-requested removal of all image research code and artifacts. This document preserves
the measured conclusion without restoring any retired benchmark infrastructure.
