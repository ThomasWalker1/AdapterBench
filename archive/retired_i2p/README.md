# Retired image-domain experiments: HyperNoise and UnHype-style conditioning

**Status: archived 2026-07-20. This is not an active AdapterBench setting.**

This directory preserves the former I2P implementation and the prompt-conditioned
go/no-go experiments that led to its retirement. The files are a historical snapshot,
not an installed package or supported CLI. The corresponding scratch outputs were moved
to `results/_archive/retired_i2p/` and remain ignored by git because they include large
checkpoints.

## Why I2P was removed

AdapterBench is about the shape of an adapter that a hypernetwork generates from a
per-instance condition. The shipped HyperNoise-style I2P row did not satisfy that premise:
`DirectCodecAdapter` was one directly optimized parameter shared by every prompt. It was
an identity-hypernetwork/unconditional optimization baseline, not an inference-time
adaptive system. Its positive ImageReward result therefore does not belong beside the
conditioned T2L and D2L rows.

The prompt-conditioned extension did compute a different LoRA from each prompt, but the
same prompt was also supplied to SD-Turbo. Under a condition-shuffle test, the emitted
adapter did not need the matched prompt. In AdapterBench's behavioral sense it was
**architecturally dynamic but not causally inference-time adaptive**: changing the
hypernetwork-only condition did not reliably change task success.

This distinction also applies to the UnHype-inspired selective-erasure probe archived
here. The probe had a legitimate extra erase condition unavailable to the generator, so
it could test adaptation causally. It produced small semantic routing effects, but none
of the tested objectives achieved reliable selective deletion while preserving the other
object. Stronger deletion objectives increasingly suppressed both objects. That is a
boundary result, not a failed positive result.

## Evidence retained

### Unconditional HyperNoise baseline (historical, not benchmark-valid)

- Directly optimized LoRA, shared across prompts.
- ImageReward gain: `+0.160 ± 0.015` over three seeds.
- Reward-swap score: `−3.34 ± 0.01`; historical headline difference `+3.50`.
- CLIP-T drop: `+0.004`.

These numbers demonstrate generic reward tilting, not condition-dependent adaptation.
The former leaderboard text is retained as
[`leaderboard_image_reward_tilting.md`](leaderboard_image_reward_tilting.md).

### Mechanism-free prompt conditioning

The hypernetwork read the same prompt that SD-Turbo already read and emitted weight-space
LoRA updates for UNet attention projections. At LoRA scales 0.5, 1, 2, and 4,
`matched − static` ImageReward was respectively `+0.063`, `+0.153`, `+0.152`, and
`+0.200`, but `matched − shuffled-condition` was `−0.001`, `+0.011`, approximately
`0.000`, and `+0.023`. The apparent gain over a separately optimized static LoRA was a
generic hypernetwork/optimization advantage; it was not prompt specialization.

### UnHype-style selective erasure

The generator received the same two-object prompt and latent while only the hypernetwork
received `remove A` versus `remove B`. This made the condition non-redundant, but the
behavioral gate still failed:

- Relative CLIP objective, scale 2: detector-margin `matched − static =
  +0.0256 ± 0.0079`, but selective success was only `0.8%`.
- Absolute suppression objective, strongest cell: target confidence fell `0.113`, retained
  confidence also fell `0.071`, CLIP-T fell `0.039`, and selective success was `6.7%`.
- UnHype-style denoising target, scale 2: apparent deletion reached `11.7%–17.5%`, but
  target and retained objects were suppressed almost identically (`0.363/0.361` and
  `0.407/0.403`), CLIP-T collapsed, and matched did not significantly beat static or
  swapped-condition controls.

The honest conclusion is that the tested system learned broad suppression or mild
semantic steering, not dependable per-request selective erasure.

## Archive layout

- `src/adapterbench/i2p/` — former generator, reward, HyperNoise, prompt-hypernetwork,
  scoring, and prompt-tilting modules.
- `src/adapterbench/cli/image_sft.py` — former `i2p-hypernoise` command.
- `scripts/` — former baseline, prompt-conditioning, detector-gate, and UnHype-style
  experiment drivers.
- `tests/test_image_hypernoise.py` — former unit tests.
- `results/_archive/retired_i2p/` — local metrics, images, logs, and checkpoints.

The active replacement research plan is
[`../../IMAGE_DOMAIN_PLAN.md`](../../IMAGE_DOMAIN_PLAN.md).
