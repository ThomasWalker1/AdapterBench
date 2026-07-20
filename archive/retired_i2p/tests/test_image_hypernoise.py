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
from adapterbench.i2p.prompt_data import (
    build_prompt_conditioning_probe_prompts,
    build_selective_erasure_probe_scenes,
)
from adapterbench.i2p.prompt_hypernetwork import PromptConditionedUNetAdapter
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


def test_prompt_hypernetwork_emits_zero_init_same_shape_adapters():
    torch.manual_seed(0)
    targets = _fake_targets(n=4)
    hyper = PromptConditionedUNetAdapter(
        targets, condition_dim=12, latent_dim=32, head_dim=16, rank=4
    )
    static = DirectCodecAdapter(targets, rank=4, scaling=1.0)
    condition = torch.randn(2, 12)
    generated = hyper(condition)

    assert hyper.generated_parameter_count() == static.generated_parameter_count()
    assert len(hyper.heads) == 1  # all fake targets share one shape head
    for name in hyper.target_names:
        key = name.replace(".", "__")
        assert generated[name].shape == (2, hyper.codecs[key].output_size)
        dense = hyper.codecs[key].dense_delta(generated[name], 0)
        assert torch.allclose(dense, torch.zeros_like(dense), atol=1e-6)


def test_prompt_hypernetwork_can_depend_on_condition():
    torch.manual_seed(0)
    hyper = PromptConditionedUNetAdapter(
        _fake_targets(n=2), condition_dim=12, latent_dim=32, head_dim=16, rank=2
    )
    # Heads are intentionally zero at initialization. Once training moves them,
    # different pooled prompts must be able to produce different adapters.
    with torch.no_grad():
        for head in hyper.heads.values():
            head.weight.normal_(std=0.1)
    first, second = torch.randn(1, 12), torch.randn(1, 12)
    name = hyper.target_names[0]
    assert not torch.allclose(hyper(first)[name], hyper(second)[name])


def test_prompt_probe_data_is_deterministic_and_disjoint():
    train, eval_ = build_prompt_conditioning_probe_prompts(train_size=256, eval_size=64)
    train_again, eval_again = build_prompt_conditioning_probe_prompts(train_size=256, eval_size=64)
    assert (train, eval_) == (train_again, eval_again)
    assert len(train) == len(set(train)) == 256
    assert len(eval_) == len(set(eval_)) == 64
    assert set(train).isdisjoint(eval_)


def test_selective_erasure_scenes_are_paired_deterministic_and_disjoint():
    train, eval_ = build_selective_erasure_probe_scenes(train_size=128, eval_size=32)
    train_again, eval_again = build_selective_erasure_probe_scenes(
        train_size=128, eval_size=32
    )
    assert (train, eval_) == (train_again, eval_again)
    assert {scene.prompt for scene in train}.isdisjoint(
        {scene.prompt for scene in eval_}
    )
    for scene in train + eval_:
        assert scene.tasks() == (
            (scene.first_object, scene.second_object),
            (scene.second_object, scene.first_object),
        )
