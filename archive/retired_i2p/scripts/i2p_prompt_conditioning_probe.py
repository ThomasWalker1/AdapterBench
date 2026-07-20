"""Short go/no-go probe for mechanism-free prompt-conditioned I2P.

Trains, on identical prompts/noise/objective:

1. a real prompt -> hypernetwork -> per-UNet-Linear LoRA, and
2. one directly optimized static LoRA with the exact same emitted shape.

The held-out headline is ImageReward ``matched - static``.  ``matched -
shuffled`` is reported alongside it to distinguish genuine use of the prompt
from a merely helpful hypernetwork parameterization.

This is deliberately a boundary probe, not the full four-invariant I2P build.
It uses ordinary prompts and plain ImageReward: no hard-requirement,
reward-conditioning, or degraded-generator necessity mechanism is present.

Example:
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/i2p_prompt_conditioning_probe.py \
      --device cuda:0 --steps 300 --output results/i2p_prompt_probe
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapterbench.i2p.hypernoise import DirectCodecAdapter, find_attention_linears  # noqa: E402
from adapterbench.i2p.image_generator import (  # noqa: E402
    embed_prompt_with_pool,
    load_frozen_generator,
)
from adapterbench.i2p.image_scoring import ClipScorer  # noqa: E402
from adapterbench.i2p.prompt_data import build_prompt_conditioning_probe_prompts  # noqa: E402
from adapterbench.i2p.prompt_hypernetwork import PromptConditionedUNetAdapter  # noqa: E402
from adapterbench.i2p.prompt_tilting import (  # noqa: E402
    generate_with_weight_adapter,
    weight_space_reward_loss,
)
from adapterbench.i2p.rewards import ImageRewardReward  # noqa: E402


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--train-prompts", type=int, default=256)
    p.add_argument("--eval-prompts", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-seeds", type=int, default=1)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--fidelity-weight", type=float, default=0.25)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--output", required=True)
    return p.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _paired_summary(left: list[float], right: list[float]) -> tuple[float, float]:
    delta = torch.tensor(left, dtype=torch.float64) - torch.tensor(right, dtype=torch.float64)
    mean = float(delta.mean())
    se = float(delta.std(unbiased=True) / math.sqrt(len(delta))) if len(delta) > 1 else float("nan")
    return mean, se


@torch.no_grad()
def evaluate(
    gen,
    hyper: PromptConditionedUNetAdapter,
    static: DirectCodecAdapter,
    reward_fn,
    clip_scorer,
    prompts: list[str],
    *,
    batch_size: int,
    n_seeds: int,
    seed_base: int = 10000,
) -> tuple[dict[str, float], list[dict]]:
    hyper.eval()
    static.eval()
    reward_values = {name: [] for name in ("matched", "shuffled", "static", "frozen")}
    clipt_values = {name: [] for name in ("matched", "shuffled", "static", "frozen")}
    rows: list[dict] = []
    in_channels = gen.unet.config.in_channels

    for seed_index in range(n_seeds):
        rng = torch.Generator(device=gen.device).manual_seed(seed_base + seed_index)
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            swapped_prompts = [
                prompts[(start + offset + 1) % len(prompts)]
                for offset in range(len(batch_prompts))
            ]
            ehs, condition = embed_prompt_with_pool(gen, batch_prompts)
            _, shuffled_condition = embed_prompt_with_pool(gen, swapped_prompts)
            latents = torch.randn(
                len(batch_prompts), in_channels, 64, 64, device=gen.device, generator=rng
            )
            images = {
                "frozen": generate_with_weight_adapter(gen, None, latents, ehs),
                "matched": generate_with_weight_adapter(gen, hyper, latents, ehs, condition),
                "shuffled": generate_with_weight_adapter(
                    gen, hyper, latents, ehs, shuffled_condition
                ),
                "static": generate_with_weight_adapter(gen, static, latents, ehs),
            }
            batch_rewards = {
                name: reward_fn(image, batch_prompts).detach().float().cpu()
                for name, image in images.items()
            }
            batch_clipt = {}
            text_features = clip_scorer.text_features(batch_prompts)
            for name, image in images.items():
                image_features = clip_scorer.image_features(image)
                batch_clipt[name] = (image_features * text_features).sum(dim=-1).cpu()

            for offset, prompt in enumerate(batch_prompts):
                row = {"prompt": prompt, "seed": seed_base + seed_index}
                for name in images:
                    reward = float(batch_rewards[name][offset])
                    clipt = float(batch_clipt[name][offset])
                    reward_values[name].append(reward)
                    clipt_values[name].append(clipt)
                    row[f"reward_{name}"] = reward
                    row[f"clipt_{name}"] = clipt
                rows.append(row)

    metrics: dict[str, float] = {}
    for name, values in reward_values.items():
        metrics[f"reward_{name}"] = _mean(values)
    for name, values in clipt_values.items():
        metrics[f"clipt_{name}"] = _mean(values)
    for comparator in ("static", "shuffled", "frozen"):
        delta, se = _paired_summary(reward_values["matched"], reward_values[comparator])
        metrics[f"matched_minus_{comparator}"] = delta
        metrics[f"matched_minus_{comparator}_se"] = se
    metrics["static_minus_frozen"], metrics["static_minus_frozen_se"] = _paired_summary(
        reward_values["static"], reward_values["frozen"]
    )
    metrics["clipt_matched_minus_static"], _ = _paired_summary(
        clipt_values["matched"], clipt_values["static"]
    )
    return metrics, rows


def _save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main() -> None:
    args = build_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    train_prompts, eval_prompts = build_prompt_conditioning_probe_prompts(
        train_size=args.train_prompts, eval_size=args.eval_prompts
    )

    torch.manual_seed(args.seed)
    print(f"[1/5] loading frozen SD-Turbo + ImageReward + CLIP-T on {args.device}", flush=True)
    gen = load_frozen_generator(args.device, torch.float32)
    reward_fn = ImageRewardReward(args.device)
    clip_scorer = ClipScorer(args.device)
    targets = find_attention_linears(gen.unet)

    print("[2/5] building prompt hypernetwork and same-shape static reference", flush=True)
    torch.manual_seed(args.seed)
    hyper = PromptConditionedUNetAdapter(
        targets,
        rank=args.rank,
        scaling=args.scale,
        latent_dim=args.latent_dim,
        head_dim=args.head_dim,
        seed=args.seed,
    ).to(args.device)
    static = DirectCodecAdapter(
        targets, codec_name="lora", rank=args.rank, scaling=args.scale, seed=args.seed
    ).to(args.device)
    if hyper.generated_parameter_count() != static.generated_parameter_count():
        raise AssertionError("hypernetwork and static reference do not emit the same adapter shape")
    print(
        f"      targets={len(targets)} emitted={hyper.generated_parameter_count():,} "
        f"hyper-trainable={hyper.trainable_parameter_count():,} "
        f"static-trainable={static.generated_parameter_count():,}",
        flush=True,
    )

    def build_optimizer(parameters):
        if args.optimizer == "sgd":
            return torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum)
        return torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)

    hyper_opt = build_optimizer(hyper.parameters())
    static_opt = build_optimizer(static.parameters())
    start = 0
    history: list[dict] = []
    rng = torch.Generator(device=args.device).manual_seed(args.seed + 1)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=args.device, weights_only=False)
        hyper.load_state_dict(state["hyper"])
        static.load_state_dict(state["static"])
        hyper_opt.load_state_dict(state["hyper_opt"])
        static_opt.load_state_dict(state["static_opt"])
        rng.set_state(state["rng_state"].cpu())
        start = state["step"]
        history = state["history"]
        print(f"      resumed at step {start}", flush=True)

    print(
        f"[3/5] training both adapters for {args.steps} steps on "
        f"{len(train_prompts)} ordinary prompts",
        flush=True,
    )
    t0 = time.time()
    in_channels = gen.unet.config.in_channels
    for step in range(start, args.steps):
        batch_prompts = [
            train_prompts[(step * args.batch_size + offset) % len(train_prompts)]
            for offset in range(args.batch_size)
        ]
        ehs, condition = embed_prompt_with_pool(gen, batch_prompts)
        latents = torch.randn(
            args.batch_size, in_channels, 64, 64, device=args.device, generator=rng
        )
        with torch.no_grad():
            reference = generate_with_weight_adapter(gen, None, latents, ehs)

        step_metrics = {}
        for name, adapter, optimizer, adapter_condition in (
            ("matched", hyper, hyper_opt, condition),
            ("static", static, static_opt, None),
        ):
            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = weight_space_reward_loss(
                gen,
                adapter,
                latents,
                ehs,
                reward_fn,
                batch_prompts,
                condition_embeddings=adapter_condition,
                reference_images=reference,
                fidelity_weight=args.fidelity_weight,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()
            step_metrics.update({f"{name}_{key}": value for key, value in metrics.items()})
            step_metrics[f"{name}_grad_norm"] = float(grad_norm)

        done = step + 1
        if done % args.log_every == 0 or done == args.steps:
            record = {"step": done, **step_metrics}
            history.append(record)
            print(
                f"      step {done:4d} matched={record['matched_reward']:+.3f} "
                f"static={record['static_reward']:+.3f} "
                f"mse={record['matched_pixel_mse']:.4f}/{record['static_pixel_mse']:.4f}",
                flush=True,
            )
        if done % args.checkpoint_every == 0 or done == args.steps:
            _save_checkpoint(
                checkpoint,
                {
                    "hyper": hyper.state_dict(),
                    "static": static.state_dict(),
                    "hyper_opt": hyper_opt.state_dict(),
                    "static_opt": static_opt.state_dict(),
                    "rng_state": rng.get_state(),
                    "step": done,
                    "history": history,
                    "config": vars(args),
                },
            )
    train_seconds = time.time() - t0

    print(
        f"[4/5] held-out paired eval: {len(eval_prompts)} prompts x {args.eval_seeds} seed(s)",
        flush=True,
    )
    metrics, rows = evaluate(
        gen,
        hyper,
        static,
        reward_fn,
        clip_scorer,
        eval_prompts,
        batch_size=args.eval_batch_size,
        n_seeds=args.eval_seeds,
    )
    result = {
        "config": vars(args),
        "train_seconds_this_invocation": train_seconds,
        "emitted_parameter_count": hyper.generated_parameter_count(),
        "hypernetwork_trainable_parameter_count": hyper.trainable_parameter_count(),
        "metrics": metrics,
        "history": history,
        "per_example": rows,
    }
    temporary = output / "probe_results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output / "probe_results.json")

    print("[5/5] go/no-go result", flush=True)
    print(
        f"      ImageReward matched-static = {metrics['matched_minus_static']:+.4f} "
        f"± {metrics['matched_minus_static_se']:.4f} SE",
        flush=True,
    )
    print(
        f"      ImageReward matched-shuffled = {metrics['matched_minus_shuffled']:+.4f} "
        f"± {metrics['matched_minus_shuffled_se']:.4f} SE",
        flush=True,
    )
    print(
        f"      static-frozen = {metrics['static_minus_frozen']:+.4f}; "
        f"CLIP-T matched-static = {metrics['clipt_matched_minus_static']:+.4f}",
        flush=True,
    )
    print(f"      wrote {output / 'probe_results.json'}", flush=True)


if __name__ == "__main__":
    main()
