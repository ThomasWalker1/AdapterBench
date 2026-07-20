"""Image-domain live-SFT CLI: the reward-tilting (HyperNoise) setting.

`i2p-hypernoise` is the image counterpart of `d2p-niah`: it trains one from-scratch
`DirectCodecAdapter` (LoRA baseline codec) on a frozen SD-Turbo to maximize one reward, then
records the four-invariant metric vector as `EvaluationResult`s. One invocation is one atomic
cell of the pipeline grid `(reward target × LoRA scale × reg weight × seed)`; the launcher
`scripts/i2p_hypernoise_pipeline.sh` fans the grid across GPUs and
`scripts/i2p_hypernoise_aggregate.py` derives the four-invariant summary. Restart-safe:
re-running the same invocation resumes from its checkpoint. Never gates on loss (gotcha #9).
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

from ..contracts import EvaluationResult
from ._shared import ResultRecorder, write_json


def _i2p_hypernoise_command(args) -> None:
    import torch

    from ..i2p.hypernoise import DirectCodecAdapter, find_attention_linears
    from ..i2p.hypernoise_trainer import evaluate_rewards, train_hypernoise_checkpointed
    from ..i2p.image_generator import load_frozen_generator
    from ..i2p.image_scoring import ClipScorer
    from ..i2p.prompt_data import EVAL_PROMPTS, TRAIN_PROMPTS
    from ..i2p.rewards import ImageRewardReward, build_reward

    torch.manual_seed(args.seed)
    scale = args.scale if args.scale > 0 else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading frozen SD-Turbo + CLIP-T + ImageReward on {args.device}...", flush=True)
    gen = load_frozen_generator(device=args.device, dtype=torch.float32)
    clip_scorer = ClipScorer(device=args.device)
    # ImageReward is the headline reward and is scored on EVERY run so the reward-swap matrix
    # (invariant #1) is computable across runs; redness is the near-orthogonal swap partner.
    imagereward = ImageRewardReward(device=args.device)
    eval_rewards = {"imagereward": imagereward, "red": build_reward("red", args.device)}
    reward_fn = imagereward if args.reward == "imagereward" else build_reward(args.reward, args.device)
    if args.reward not in eval_rewards:
        eval_rewards[args.reward] = reward_fn

    targets = find_attention_linears(gen.unet)
    torch.manual_seed(args.seed)
    adapter = DirectCodecAdapter(targets, codec_name="lora", rank=args.rank, scaling=scale, seed=args.seed).to(args.device)
    n_params = adapter.generated_parameter_count()
    print(f"[2/4] hooked {len(targets)} attn linears; adapter params={n_params:,}; "
          f"reward={args.reward} scale={scale} reg={args.reg_weight} seed={args.seed}", flush=True)

    def _evaluate(net, step):
        return evaluate_rewards(gen, net, eval_rewards, clip_scorer, EVAL_PROMPTS, n_seeds=args.n_seeds)

    print(f"[3/4] training {args.steps} steps (eval every {args.eval_every})...", flush=True)
    t0 = time.time()
    history = train_hypernoise_checkpointed(
        gen, adapter, reward_fn, TRAIN_PROMPTS, device=args.device, steps=args.steps,
        eval_every=args.eval_every, batch_size=args.batch_size, learning_rate=args.lr,
        reg_weight=args.reg_weight, seed=args.seed, evaluate=_evaluate,
        checkpoint_path=output_dir / "checkpoint.pt",
    )
    train_seconds = time.time() - t0
    config = {k: v for k, v in vars(args).items() if k != "func"}
    write_json(output_dir / "history.json", {"config": config, "history": history})

    print("[4/4] recording final four-invariant metrics...", flush=True)
    adapter.eval()
    t1 = time.time()
    ev = evaluate_rewards(gen, adapter, eval_rewards, clip_scorer, EVAL_PROMPTS, n_seeds=args.n_seeds)
    inference_seconds = time.time() - t1

    # Derived headline scalars: gain = adapter - frozen (per reward), fidelity drop, prompt-swap gain.
    metrics = dict(ev)
    for name in eval_rewards:
        metrics[f"reward_{name}_gain"] = ev[f"reward_{name}_adapter"] - ev[f"reward_{name}_frozen"]
    metrics["clipt_drop"] = ev["clipt_frozen"] - ev["clipt_adapter"]
    if "reward_imagereward_promptswap_adapter" in ev:
        metrics["reward_imagereward_promptswap_gain"] = (
            ev["reward_imagereward_promptswap_adapter"] - ev["reward_imagereward_promptswap_frozen"]
        )

    scale_tag = f"{scale:g}" if scale is not None else "default"
    trial_id = f"i2p_hypernoise::lora::{args.reward}::s{args.seed}::reg{args.reg_weight:g}::scale{scale_tag}"
    result = EvaluationResult(
        trial_id=trial_id, task_id=f"hypernoise_{args.reward}", split="eval_prompts", adapter="lora",
        metrics=metrics, generated_parameter_count=n_params, generation_seconds=train_seconds,
        inference_seconds=inference_seconds,
        metadata={
            "setting": "i2p_hypernoise", "reward_target": args.reward, "reg_weight": args.reg_weight,
            "lora_scale": scale_tag, "rank": args.rank, "seed": args.seed, "steps": args.steps,
            "n_eval_prompts": len(EVAL_PROMPTS), "n_eval_seeds": args.n_seeds,
        },
    )
    recorder = ResultRecorder(output_dir, lambda r: f"  {r.adapter:6} {r.task_id:22} {r.metrics}")
    recorder.record(result)
    print(f"wrote {output_dir}/results.jsonl, {output_dir}/history.json", flush=True)


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "i2p-hypernoise",
        help="Image-domain reward-tilting: train a LoRA noise-adapter on frozen SD-Turbo to "
             "maximize a reward (ImageReward headline), recording the four-invariant metrics.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reward", choices=["imagereward", "red", "green", "blue"], default="imagereward",
                   help="reward target to maximize; the run always also scores ImageReward + redness "
                        "for the cross-run reward-swap control")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg-weight", type=float, default=0.5, help="the fidelity<->reward difficulty knob (invariant #3)")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--scale", type=float, default=0.0, help="LoRA scale (invariant #2 sweep axis); <=0 uses the codec default")
    p.add_argument("--n-seeds", type=int, default=2, help="generator seeds averaged per eval prompt")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--output", required=True)
    p.set_defaults(func=_i2p_hypernoise_command)
