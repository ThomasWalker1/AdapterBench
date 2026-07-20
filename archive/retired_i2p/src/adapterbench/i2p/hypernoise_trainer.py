"""Training + evaluation for the reward-tilting (HyperNoise) image setting.

`train_hypernoise_checkpointed` fits one `DirectCodecAdapter` on a frozen generator to
maximize one reward, restart-safe (atomic model+optimizer+step+history checkpoint every
eval), never gating on loss. `evaluate_rewards` produces the four-invariant metric vector for
an adapter: every reward's adapter-vs-frozen value (for the cross-run reward-swap control), a
within-run prompt-swap control for ImageReward, and CLIP-T prompt fidelity (anti reward
hacking). The CLI turns this dict into `EvaluationResult` records.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .hypernoise import DirectCodecAdapter, generate_from_latents, hypernoise_loss, noise_transform
from .image_generator import FrozenGenerator, embed_prompt


@torch.no_grad()
def evaluate_rewards(
    gen: FrozenGenerator, adapter: DirectCodecAdapter, rewards: dict, clip_scorer, eval_prompts: list[str],
    *, n_seeds: int = 2, seed_base: int = 1000, num_steps: int = 1,
) -> dict:
    """Score the adapter (and the frozen generator) on every reward in `rewards`, plus CLIP-T.

    Returns floats: `reward_{name}_adapter`/`reward_{name}_frozen` for each reward name;
    `reward_imagereward_promptswap_adapter`/`_frozen` (ImageReward scored against a *mismatched*
    prompt — the within-run control: a genuine alignment gain must not transfer to a wrong
    prompt) when `imagereward` is present; and `clipt_adapter`/`clipt_frozen`.
    """
    adapter.eval()
    acc: dict[str, list[float]] = {}

    def add(key, value):
        acc.setdefault(key, []).append(float(value))

    n = len(eval_prompts)
    in_ch = gen.unet.config.in_channels
    for i, prompt in enumerate(eval_prompts):
        mismatched = eval_prompts[(i + 1) % n]  # for the prompt-swap control
        ehs = embed_prompt(gen, [prompt])
        for s in range(n_seeds):
            g = torch.Generator(device=gen.device).manual_seed(seed_base + s)
            x0 = torch.randn(1, in_ch, 64, 64, device=gen.device, generator=g)
            x_hat = x0 + noise_transform(gen, adapter, x0, ehs)
            img_a = generate_from_latents(gen, x_hat, ehs, num_steps=num_steps)
            img_f = generate_from_latents(gen, x0, ehs, num_steps=num_steps)
            for name, fn in rewards.items():
                add(f"reward_{name}_adapter", fn(img_a, [prompt]).mean())
                add(f"reward_{name}_frozen", fn(img_f, [prompt]).mean())
                if name == "imagereward":
                    add("reward_imagereward_promptswap_adapter", fn(img_a, [mismatched]).mean())
                    add("reward_imagereward_promptswap_frozen", fn(img_f, [mismatched]).mean())
            add("clipt_adapter", clip_scorer.clip_t(img_a, prompt))
            add("clipt_frozen", clip_scorer.clip_t(img_f, prompt))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def train_hypernoise_checkpointed(
    gen: FrozenGenerator, adapter: DirectCodecAdapter, reward_fn, train_prompts: list[str],
    *, device: str, steps: int, eval_every: int, batch_size: int = 4, learning_rate: float = 1e-3,
    momentum: float = 0.9, reg_weight: float = 0.5, grad_clip: float = 10.0, num_steps: int = 1,
    seed: int = 777, evaluate=None, checkpoint_path: str | Path | None = None,
) -> list[dict]:
    """Restart-safe reward-tilting training. `evaluate(adapter, step) -> dict` is called every
    `eval_every` steps (and at the end); its dict is merged into the logged history record.
    Checkpoints model+optimizer+step+history atomically so re-running resumes exactly."""
    opt = torch.optim.SGD(adapter.parameters(), lr=learning_rate, momentum=momentum)
    start, history = 0, []
    ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
    if ckpt is not None and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        adapter.load_state_dict(state["adapter"]); opt.load_state_dict(state["opt"])
        start, history = state["step"], state["history"]
        print(f"[resume] step {start}", flush=True)

    gen_rng = torch.Generator(device=device).manual_seed(seed)
    in_ch = gen.unet.config.in_channels
    for step in range(start, steps):
        adapter.train()
        prompts = [train_prompts[(step * batch_size + i) % len(train_prompts)] for i in range(batch_size)]
        ehs = embed_prompt(gen, prompts)
        x0 = torch.randn(batch_size, in_ch, 64, 64, device=device, generator=gen_rng)
        opt.zero_grad(set_to_none=True)
        loss, metrics = hypernoise_loss(gen, adapter, x0, ehs, reward_fn, prompts, reg_weight=reg_weight, num_steps=num_steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), grad_clip)
        opt.step()
        done = step + 1
        if done % eval_every == 0 or done == steps:
            rec = {"step": done, **metrics}
            if evaluate is not None:
                rec.update(evaluate(adapter, done))
            history.append(rec)
            if ckpt is not None:
                tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
                torch.save({"adapter": adapter.state_dict(), "opt": opt.state_dict(), "step": done, "history": history}, tmp)
                tmp.replace(ckpt)
    return history
