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
GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-tiny"
CLIP_SIZE = 224
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipScorer:
    """Holds a frozen CLIP model + tokenizer and computes CLIP-T. Load once, reuse."""

    def __init__(self, device: str = "cuda:0", model_id: str = CLIP_ID):
        from transformers import AutoTokenizer, CLIPConfig, CLIPModel

        self.device = device
        # Transformers refuses to torch.load the cached .bin under torch<2.6
        # (CVE-2025-32434). Prefer the normal safetensors path, but older Hub
        # caches keep the converted weights under refs/pr/66 rather than the
        # main snapshot; in offline mode auto-conversion cannot discover that
        # ref. Rebuild from the locally cached config + converted state in that
        # case, without ever deserializing pickle.
        try:
            model = CLIPModel.from_pretrained(
                model_id, use_safetensors=True, local_files_only=True
            )
        except Exception as error:
            from huggingface_hub import try_to_load_from_cache
            from safetensors.torch import load_file

            converted = try_to_load_from_cache(
                model_id, "model.safetensors", revision="refs/pr/66"
            )
            if not isinstance(converted, str):
                raise RuntimeError(
                    f"no locally cached safetensors checkpoint for {model_id}"
                ) from error
            config = CLIPConfig.from_pretrained(model_id, local_files_only=True)
            model = CLIPModel(config)
            incompatible = model.load_state_dict(load_file(converted), strict=False)
            allowed_unexpected = {
                "text_model.embeddings.position_ids",
                "vision_model.embeddings.position_ids",
            }
            if incompatible.missing_keys or set(incompatible.unexpected_keys) - allowed_unexpected:
                raise RuntimeError(
                    "cached CLIP safetensors state is incompatible: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
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

    def image_features_with_grad(self, images01: Tensor) -> Tensor:
        """CLIP image features with gradients only to ``images01``.

        The scorer parameters are frozen in ``__init__``.  Unlike
        :meth:`image_features`, this method deliberately does not enter
        ``no_grad`` so a CLIP-margin objective can train an upstream image
        adapter without updating CLIP itself.
        """
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


class GroundingDinoScorer:
    """Independent open-vocabulary object-presence confidence.

    This scorer is evaluation-only. Each image is paired with one text label,
    and the returned value is the maximum Grounding DINO box confidence for
    that label. A low score means the requested object was not detected.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        model_id: str = GROUNDING_DINO_ID,
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_id, local_files_only=True
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, local_files_only=True
        ).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def confidences(self, images01: Tensor, labels: list[str]) -> Tensor:
        if len(images01) != len(labels):
            raise ValueError("one Grounding DINO label is required per image")
        # The image processor normally expects uint8 inputs and rescales by
        # 1/255. Our generator already returns float [0, 1], so disable that
        # extra rescaling.
        images = [image.detach().float().cpu() for image in images01]
        text_labels = [[label] for label in labels]
        inputs = self.processor(
            images=images,
            text=text_labels,
            do_rescale=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=0.0,
            text_threshold=0.0,
            target_sizes=[tuple(image.shape[-2:]) for image in images],
            text_labels=text_labels,
        )
        confidences = [
            result["scores"].max() if result["scores"].numel() else torch.zeros((), device=self.device)
            for result in results
        ]
        return torch.stack(confidences)
