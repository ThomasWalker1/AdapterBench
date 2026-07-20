"""Frozen image generator (SD-Turbo) plumbing for the image-domain setting.

Image-domain counterpart to `cli/_shared.py::load_frozen_interpreter`: load a frozen
distilled text-to-image generator, freeze every parameter, and expose the pieces the
reward-tilting HyperNoise setting needs (see PROJECT_PLAN.md's "Image domain" section).
The hook site is the UNet's attention linears (`i2p/hypernoise.py::find_attention_linears`),
resolved directly off `unet`, so nothing here is adapter- or hook-specific.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

GENERATOR_ID = "stabilityai/sd-turbo"
CROSS_ATTN_KV_DIM = 1024  # SD-Turbo's uniform cross_attention_dim (K/V input width)


@dataclass
class FrozenGenerator:
    """Everything the image setting needs from the frozen generator, bundled so call
    sites (the reward-tilting trainer/evaluator, the smoke) never re-derive it."""

    pipe: object
    unet: nn.Module
    vae: nn.Module
    text_encoder: nn.Module
    tokenizer: object
    scheduler: object  # the pipe's inference scheduler (EulerDiscrete), used by generation
    device: str
    dtype: torch.dtype


def load_frozen_generator(device: str = "cuda:0", dtype: torch.dtype = torch.float32) -> FrozenGenerator:
    """Load SD-Turbo in eval mode on `device` and freeze every parameter.

    Mirrors `load_frozen_interpreter`'s contract (eval + freeze). dtype defaults to
    float32: the from-scratch adapter and the differentiable reward backward are fp32.
    """
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(GENERATOR_ID, torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(device)
    unet, vae, text_encoder = pipe.unet, pipe.vae, pipe.text_encoder
    for module in (unet, vae, text_encoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    return FrozenGenerator(
        pipe=pipe, unet=unet, vae=vae, text_encoder=text_encoder, tokenizer=pipe.tokenizer,
        scheduler=pipe.scheduler, device=device, dtype=dtype,
    )


@torch.no_grad()
def embed_prompt_with_pool(gen: FrozenGenerator, prompts: list[str]) -> tuple[Tensor, Tensor]:
    """Return generator token states and the pooled prompt condition.

    The token states ``(batch, 77, 1024)`` remain the frozen UNet's normal
    cross-attention input.  The CLIP text encoder's pooled output ``(batch,
    1024)`` is separately fed to the prompt hypernetwork; computing both in one
    frozen encoder pass avoids accidentally changing what the generator sees.
    """
    tokens = gen.tokenizer(
        prompts, padding="max_length", max_length=gen.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(gen.device)
    encoded = gen.text_encoder(tokens.input_ids)
    return encoded.last_hidden_state, encoded.pooler_output


@torch.no_grad()
def embed_prompt(gen: FrozenGenerator, prompts: list[str]) -> Tensor:
    """Frozen text-encoder hidden states for existing unconditional I2P."""
    return embed_prompt_with_pool(gen, prompts)[0]
