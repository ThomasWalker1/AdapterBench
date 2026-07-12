"""Unit tests for the reward-tilting image setting that need no model download / GPU.

Covers the two seam-critical, model-free invariants: (1) `DirectCodecAdapter` is exactly
zero at init (the LoRA zero-init contract that makes Δx₀=0), yet all its parameters are
trainable; (2) the reward registry returns near-orthogonal channel rewards. The SD-Turbo /
ImageReward end-to-end path is exercised by `scripts/i2p_hypernoise_smoke.py` (needs a GPU).
"""

from __future__ import annotations

import torch
from torch import nn

from adapterbench.i2p.hypernoise import DirectCodecAdapter
from adapterbench.i2p.rewards import build_reward, redness_reward


def _fake_targets(n=3, in_f=8, out_f=16):
    return [(f"blk{i}.attn2.to_k", nn.Linear(in_f, out_f)) for i in range(n)]


def test_direct_codec_adapter_zero_init_but_trainable():
    torch.manual_seed(0)
    targets = _fake_targets()
    adapter = DirectCodecAdapter(targets, codec_name="lora", rank=4)
    # LoRA zero-init: every generated delta contributes exactly zero at init (B=0), so the
    # noise-transform difference is exactly zero — the Δx₀=0 initialization the setting needs.
    for name in adapter.target_names:
        codec = adapter.codecs[name.replace(".", "__")]
        delta = adapter.deltas[name.replace(".", "__")]
        dense = codec.dense_delta(delta.unsqueeze(0), 0)
        assert torch.allclose(dense, torch.zeros_like(dense), atol=1e-6)
    # ...but the parameters are real and trainable (gradient must be able to reach them).
    assert adapter.generated_parameter_count() > 0
    assert all(p.requires_grad for p in adapter.parameters())


def test_scaling_override_propagates_to_codecs():
    targets = _fake_targets()
    adapter = DirectCodecAdapter(targets, codec_name="lora", rank=4, scaling=7.5)
    for codec in adapter.codecs.values():
        assert codec.scaling == 7.5


def test_reward_registry_channels_orthogonal():
    red = build_reward("red")
    green = build_reward("green")
    assert red is not None and green is not None
    # a pure-red image: high red reward, negative green reward (near-orthogonal → sharp swap control)
    img = torch.zeros(1, 3, 8, 8)
    img[:, 0] = 1.0
    assert redness_reward(img).item() > 0.9
    assert red(img, ["x"]).item() > 0.9
    assert green(img, ["x"]).item() < 0.0
