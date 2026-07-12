"""Frozen CLIP-T scorer: the prompt-fidelity control for the image-domain setting.

CLIP-T (cosine between a generated image and its prompt) is the anti-reward-hacking
fidelity control (invariant #1): a noise-adapter that inflates its reward by collapsing
image content shows up as a CLIP-T drop below the frozen baseline. Computed with a frozen
`openai/clip-vit-base-patch32`, deliberately a *different* encoder from any reward model so
the control is not circular. Inputs are plain [0, 1] image tensors `(n, 3, H, W)`.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_SIZE = 224
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipScorer:
    """Holds a frozen CLIP model + tokenizer and computes CLIP-T. Load once, reuse."""

    def __init__(self, device: str = "cuda:0", model_id: str = CLIP_ID):
        from transformers import AutoTokenizer, CLIPModel

        self.device = device
        # use_safetensors=True: the cached CLIP is a .bin, which transformers refuses to
        # torch.load under torch<2.6 (CVE-2025-32434); the hub also ships model.safetensors.
        self.model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(_CLIP_STD, device=device).view(1, 3, 1, 1)

    def _preprocess(self, images01: Tensor) -> Tensor:
        images01 = images01.to(self.device, torch.float32)
        if images01.shape[-1] != CLIP_SIZE or images01.shape[-2] != CLIP_SIZE:
            images01 = F.interpolate(images01, size=CLIP_SIZE, mode="bicubic", align_corners=False, antialias=True)
        return (images01.clamp(0, 1) - self._mean) / self._std

    @torch.no_grad()
    def image_features(self, images01: Tensor) -> Tensor:
        features = self.model.get_image_features(pixel_values=self._preprocess(images01))
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def text_features(self, prompts: list[str]) -> Tensor:
        tokens = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        features = self.model.get_text_features(**tokens)
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def clip_t(self, generated01: Tensor, prompt: str) -> float:
        """CLIP-T: mean cosine between generated images and the prompt text (prompt fidelity)."""
        gen = self.image_features(generated01)
        text = self.text_features([prompt])
        return float((gen @ text.T).mean())
