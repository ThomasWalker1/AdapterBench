"""Training primitives for prompt-conditioned weight-space reward tilting."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor

from .hypernoise import DirectCodecAdapter, generate_from_latents
from .image_generator import FrozenGenerator
from .prompt_hypernetwork import PromptConditionedUNetAdapter


def generate_with_weight_adapter(
    gen: FrozenGenerator,
    adapter: PromptConditionedUNetAdapter | DirectCodecAdapter | None,
    latents: Tensor,
    encoder_hidden_states: Tensor,
    condition_embeddings: Tensor | None = None,
    *,
    num_steps: int = 1,
) -> Tensor:
    """Generate with a LoRA applied directly to every denoising UNet call."""
    if adapter is None:
        return generate_from_latents(gen, latents, encoder_hidden_states, num_steps=num_steps)
    if isinstance(adapter, PromptConditionedUNetAdapter):
        if condition_embeddings is None:
            raise ValueError("condition_embeddings are required for the prompt hypernetwork")
        context = adapter.apply(adapter(condition_embeddings))
    else:
        context = adapter.apply(batch=latents.shape[0])

    scheduler = gen.scheduler
    scheduler.set_timesteps(num_steps, device=gen.device)
    current = latents * scheduler.init_noise_sigma
    with context:
        for timestep in scheduler.timesteps:
            model_input = scheduler.scale_model_input(current, timestep)
            predicted = gen.unet(
                model_input, timestep, encoder_hidden_states=encoder_hidden_states
            ).sample
            current = scheduler.step(predicted, timestep, current).prev_sample
    images = gen.vae.decode(current / gen.vae.config.scaling_factor).sample
    return (images / 2 + 0.5).clamp(0, 1)


def weight_space_reward_loss(
    gen: FrozenGenerator,
    adapter: PromptConditionedUNetAdapter | DirectCodecAdapter,
    latents: Tensor,
    encoder_hidden_states: Tensor,
    reward_fn,
    prompts: list[str],
    *,
    condition_embeddings: Tensor | None = None,
    reference_images: Tensor | None = None,
    fidelity_weight: float = 0.25,
    num_steps: int = 1,
) -> tuple[Tensor, dict[str, float]]:
    """Maximize reward while penalizing same-noise pixel drift from the frozen image."""
    images = generate_with_weight_adapter(
        gen,
        adapter,
        latents,
        encoder_hidden_states,
        condition_embeddings,
        num_steps=num_steps,
    )
    reward = reward_fn(images, prompts).mean()
    if reference_images is None:
        with torch.no_grad():
            reference_images = generate_with_weight_adapter(
                gen, None, latents, encoder_hidden_states, num_steps=num_steps
            )
    fidelity = (images - reference_images).square().mean()
    loss = fidelity_weight * fidelity - reward
    return loss, {
        "loss": float(loss.detach()),
        "reward": float(reward.detach()),
        "pixel_mse": float(fidelity.detach()),
    }


def selective_erasure_clip_loss(
    gen: FrozenGenerator,
    adapter: PromptConditionedUNetAdapter | DirectCodecAdapter,
    latents: Tensor,
    encoder_hidden_states: Tensor,
    clip_scorer,
    erase_objects: list[str],
    retain_objects: list[str],
    *,
    condition_embeddings: Tensor | None = None,
    reference_images: Tensor | None = None,
    fidelity_weight: float = 0.25,
    reward_scale: float = 10.0,
    erase_weight: float = 1.0,
    retain_weight: float = 1.0,
    num_steps: int = 1,
) -> tuple[Tensor, dict[str, float]]:
    """Favor the retained object over the requested erase object.

    The two CLIP similarities are evaluated on the same generated image.  In
    the probe, each scene is presented in both erase directions with identical
    latents, making the requirements contradictory for a static adapter but
    selectable for the condition-generated adapter.
    """
    images = generate_with_weight_adapter(
        gen,
        adapter,
        latents,
        encoder_hidden_states,
        condition_embeddings,
        num_steps=num_steps,
    )
    image_features = clip_scorer.image_features_with_grad(images)
    erase_features = clip_scorer.text_features(
        [f"a photo of a {name}" for name in erase_objects]
    )
    retain_features = clip_scorer.text_features(
        [f"a photo of a {name}" for name in retain_objects]
    )
    erase_similarity = (image_features * erase_features).sum(dim=-1)
    retain_similarity = (image_features * retain_features).sum(dim=-1)
    margin = retain_similarity - erase_similarity
    objective = retain_weight * retain_similarity - erase_weight * erase_similarity

    if reference_images is None:
        with torch.no_grad():
            reference_images = generate_with_weight_adapter(
                gen, None, latents, encoder_hidden_states, num_steps=num_steps
            )
    fidelity = (images - reference_images).square().mean()
    reward = reward_scale * objective.mean()
    loss = fidelity_weight * fidelity - reward
    return loss, {
        "loss": float(loss.detach()),
        "margin": float(margin.detach().mean()),
        "objective": float(objective.detach().mean()),
        "erase_similarity": float(erase_similarity.detach().mean()),
        "retain_similarity": float(retain_similarity.detach().mean()),
        "pixel_mse": float(fidelity.detach()),
    }


def selective_erasure_denoising_loss(
    gen: FrozenGenerator,
    adapter: PromptConditionedUNetAdapter | DirectCodecAdapter,
    latents: Tensor,
    full_encoder_hidden_states: Tensor,
    retain_encoder_hidden_states: Tensor,
    *,
    condition_embeddings: Tensor | None = None,
    negative_guidance: float = 1.0,
    num_steps: int = 1,
) -> tuple[Tensor, dict[str, float]]:
    """UnHype/ESD-style semantic removal in denoising-prediction space.

    The frozen full-prompt prediction is repelled toward and then beyond the
    prediction for a prompt containing only the object that should remain.
    Grounding DINO is deliberately not involved in this training loss; it
    remains an independent object-presence evaluator.
    """
    scheduler = gen.scheduler
    scheduler.set_timesteps(num_steps, device=gen.device)
    timestep = scheduler.timesteps[0]
    current = latents * scheduler.init_noise_sigma
    model_input = scheduler.scale_model_input(current, timestep)

    with torch.no_grad():
        prediction_full = gen.unet(
            model_input,
            timestep,
            encoder_hidden_states=full_encoder_hidden_states,
        ).sample
        prediction_retain = gen.unet(
            model_input,
            timestep,
            encoder_hidden_states=retain_encoder_hidden_states,
        ).sample
        target = prediction_retain - negative_guidance * (
            prediction_full - prediction_retain
        )

    if isinstance(adapter, PromptConditionedUNetAdapter):
        if condition_embeddings is None:
            raise ValueError("condition_embeddings are required for the prompt hypernetwork")
        context = adapter.apply(adapter(condition_embeddings))
    elif isinstance(adapter, DirectCodecAdapter):
        context = adapter.apply(batch=latents.shape[0])
    else:
        context = nullcontext()
    with context:
        prediction_adapted = gen.unet(
            model_input,
            timestep,
            encoder_hidden_states=full_encoder_hidden_states,
        ).sample

    loss = (prediction_adapted - target).square().mean()
    shift = (prediction_adapted - prediction_full).square().mean().sqrt()
    teacher_shift = (target - prediction_full).square().mean().sqrt()
    return loss, {
        "loss": float(loss.detach()),
        "prediction_shift_rms": float(shift.detach()),
        "teacher_shift_rms": float(teacher_shift.detach()),
    }
