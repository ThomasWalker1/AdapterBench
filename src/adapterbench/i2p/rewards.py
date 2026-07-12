"""Differentiable image rewards for the reward-tilting (HyperNoise) image setting.

Every reward is a callable `reward(images01, prompts) -> (B,)` returning a per-image scalar
where **higher is better** (the trainer maximizes it; the loss negates it). Two are provided:

- `ImageRewardReward` — the headline reward: ImageReward-v1.0 (a BLIP-based human-preference
  model). Backpropagates image gradients through the frozen reward model to the generator's
  noise, so a small adapter learns to nudge initial noise toward higher-preference images.
  Prompt-aware (the score is image↔prompt alignment), so it supports a within-run prompt-swap
  control (score against a mismatched prompt must not rise).
- `channel_reward` / `redness_reward` — a zero-dependency, near-orthogonal reward used as the
  reward-swap control partner (invariant #1): an adapter optimized for ImageReward must not
  raise redness and vice-versa. Trivially hackable by design (the NIAH-needle analogue).

`build_reward(name, device)` is the registry the CLI/trainer dispatch through.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# BLIP/CLIP image normalization (ImageReward's visual encoder expects 224px, these stats).
_IMG_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IMG_STD = (0.26862954, 0.26130258, 0.27577711)


def channel_reward(channel: int):
    """r = c - ½(other two channels), mean over pixels. `images01` is (B,3,H,W) in [0,1]."""
    others = [c for c in (0, 1, 2) if c != channel]

    def reward(images01: Tensor, prompts: list[str] | None = None) -> Tensor:
        return (images01[:, channel] - 0.5 * (images01[:, others[0]] + images01[:, others[1]])).mean(dim=(1, 2))

    return reward


redness_reward = channel_reward(0)


class ImageRewardReward:
    """Differentiable ImageReward-v1.0 wrapper: `reward(images01, prompts) -> (B,)` = the
    normalized human-preference score, gradient-capable in the pixels.

    Loads the frozen model once (freezes it). Preprocessing (resize→224 + BLIP normalize) is
    differentiable so the reward's gradient reaches the generator. Mirrors the reference
    `rewards/imagereward.py::score_diff` but returns the raw score to *maximize* (the reference
    returns `2 - score` as a loss)."""

    name = "imagereward"

    def __init__(self, device: str = "cuda:0", model_id: str = "ImageReward-v1.0"):
        # ImageReward's bundled BLIP imports three symbols from transformers.modeling_utils
        # that current transformers moved to transformers.pytorch_utils; shim before import.
        import transformers.modeling_utils as _mu
        import transformers.pytorch_utils as _pu

        for _s in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
            if not hasattr(_mu, _s):
                setattr(_mu, _s, getattr(_pu, _s))
        import ImageReward as RM

        self.device = device
        self.model = RM.load(model_id).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._mean = torch.tensor(_IMG_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMG_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, images01: Tensor, prompts: list[str]) -> Tensor:
        x = F.interpolate(images01, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x.clamp(0, 1) - self._mean) / self._std
        blip = self.model.blip
        tok = blip.tokenizer(prompts, padding="max_length", truncation=True, max_length=35, return_tensors="pt").to(self.device)
        image_embeds = blip.visual_encoder(x)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
        text_output = blip.text_encoder(
            tok.input_ids, attention_mask=tok.attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=True,
        )
        feats = text_output.last_hidden_state[:, 0, :].float()
        rewards = self.model.mlp(feats)
        rewards = (rewards - self.model.mean) / self.model.std
        return rewards.squeeze(-1)


def build_reward(name: str, device: str = "cuda:0"):
    """Reward registry. `imagereward` is the headline; red/green/blue are the orthogonal
    swap-control partners. Returns a callable `reward(images01, prompts) -> (B,)`."""
    if name == "imagereward":
        return ImageRewardReward(device=device)
    if name in ("red", "green", "blue"):
        return channel_reward({"red": 0, "green": 1, "blue": 2}[name])
    raise ValueError(f"unknown reward {name!r} (expected imagereward|red|green|blue)")
