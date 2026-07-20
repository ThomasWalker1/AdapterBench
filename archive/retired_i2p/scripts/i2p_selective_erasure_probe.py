"""Go/no-go probe for target-conditioned selective object erasure.

For every two-object scene, the frozen SD-Turbo generator receives the same
prompt and latent twice.  The separate hypernetwork condition requests either
``erase A`` or ``erase B``.  A directly optimized, same-shape static LoRA is
trained on the identical contradictory examples.

The primary score is a frozen CLIP margin:

    similarity(image, retained object) - similarity(image, erased object)

The go criteria require both ``matched - static`` and ``matched - swapped`` to
be positive while retaining prompt fidelity.  This is a disposable de-risk
probe, not an integrated AdapterBench setting.
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
from adapterbench.i2p.image_generator import embed_prompt_with_pool, load_frozen_generator  # noqa: E402
from adapterbench.i2p.image_scoring import ClipScorer, GroundingDinoScorer  # noqa: E402
from adapterbench.i2p.prompt_data import (  # noqa: E402
    SelectiveEraseScene,
    build_selective_erasure_probe_scenes,
)
from adapterbench.i2p.prompt_hypernetwork import PromptConditionedUNetAdapter  # noqa: E402
from adapterbench.i2p.prompt_tilting import (  # noqa: E402
    generate_with_weight_adapter,
    selective_erasure_clip_loss,
    selective_erasure_denoising_loss,
)
from adapterbench.i2p.rewards import ImageRewardReward  # noqa: E402


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--scene-batch-size", type=int, default=1)
    parser.add_argument("--train-scenes", type=int, default=128)
    parser.add_argument("--eval-scenes", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-seeds", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--fidelity-weight", type=float, default=0.25)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--erase-weight", type=float, default=1.0)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument(
        "--train-objective", choices=["clip", "diffusion"], default="clip"
    )
    parser.add_argument("--negative-guidance", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--independent-eval", action="store_true")
    parser.add_argument("--detector-eval", action="store_true")
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    parser.add_argument("--save-grid", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _paired_tasks(scenes: list[SelectiveEraseScene]):
    prompts, erase, retain = [], [], []
    for scene in scenes:
        for erase_object, retain_object in scene.tasks():
            prompts.append(scene.prompt)
            erase.append(erase_object)
            retain.append(retain_object)
    return prompts, erase, retain


def _paired_summary(left: list[float], right: list[float]) -> tuple[float, float]:
    delta = torch.tensor(left, dtype=torch.float64) - torch.tensor(right, dtype=torch.float64)
    mean = float(delta.mean())
    se = float(delta.std(unbiased=True) / math.sqrt(len(delta))) if len(delta) > 1 else float("nan")
    return mean, se


def _scene_pair_averages(values: list[float]) -> list[float]:
    """Collapse the two opposite requests for each identical scene/latent.

    The scene, not an individual direction, is the independent experimental
    unit.  Pairing also removes the frozen generator's arbitrary preference
    for one of the two objects.
    """
    if len(values) % 2:
        raise ValueError("selective-erasure values must contain both directions per scene")
    return [(values[index] + values[index + 1]) / 2 for index in range(0, len(values), 2)]


def _save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(
    gen,
    hyper: PromptConditionedUNetAdapter,
    static: DirectCodecAdapter,
    scorer: ClipScorer,
    scenes: list[SelectiveEraseScene],
    *,
    batch_size: int,
    n_seeds: int,
    independent_reward=None,
    detector=None,
    detector_threshold: float = 0.25,
    grid_path: Path | None = None,
    seed_base: int = 20000,
) -> tuple[dict[str, float], list[dict]]:
    hyper.eval()
    static.eval()
    metric_names = ["margin", "erase_similarity", "retain_similarity", "clipt"]
    if independent_reward is not None:
        metric_names.append("imagereward_margin")
    if detector is not None:
        metric_names.extend(
            ["detector_margin", "detector_erase", "detector_retain"]
        )
    values = {
        name: {metric_name: [] for metric_name in metric_names}
        for name in ("matched", "swapped", "static", "frozen")
    }
    rows: list[dict] = []
    grid_rows: list[tuple[str, str, str, dict[str, torch.Tensor]]] = []
    prompts, erase_objects, retain_objects = _paired_tasks(scenes)
    in_channels = gen.unet.config.in_channels

    for seed_index in range(n_seeds):
        rng = torch.Generator(device=gen.device).manual_seed(seed_base + seed_index)
        scene_latents = torch.randn(
            len(scenes), in_channels, 64, 64, device=gen.device, generator=rng
        )
        task_latents = scene_latents.repeat_interleave(2, dim=0)
        for start in range(0, len(prompts), batch_size):
            stop = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:stop]
            batch_erase = erase_objects[start:stop]
            batch_retain = retain_objects[start:stop]
            latents = task_latents[start:stop]
            encoder_states, _ = embed_prompt_with_pool(gen, batch_prompts)
            _, matched_condition = embed_prompt_with_pool(
                gen, [f"remove the {name}" for name in batch_erase]
            )
            _, swapped_condition = embed_prompt_with_pool(
                gen, [f"remove the {name}" for name in batch_retain]
            )
            images = {
                "frozen": generate_with_weight_adapter(gen, None, latents, encoder_states),
                "matched": generate_with_weight_adapter(
                    gen, hyper, latents, encoder_states, matched_condition
                ),
                "swapped": generate_with_weight_adapter(
                    gen, hyper, latents, encoder_states, swapped_condition
                ),
                "static": generate_with_weight_adapter(gen, static, latents, encoder_states),
            }
            erase_text = scorer.text_features(
                [f"a photo of a {name}" for name in batch_erase]
            )
            retain_text = scorer.text_features(
                [f"a photo of a {name}" for name in batch_retain]
            )
            prompt_text = scorer.text_features(batch_prompts)
            batch_metrics = {}
            for name, image in images.items():
                image_features = scorer.image_features(image)
                erase_sim = (image_features * erase_text).sum(dim=-1)
                retain_sim = (image_features * retain_text).sum(dim=-1)
                batch_metrics[name] = {
                    "margin": retain_sim - erase_sim,
                    "erase_similarity": erase_sim,
                    "retain_similarity": retain_sim,
                    "clipt": (image_features * prompt_text).sum(dim=-1),
                }
                if independent_reward is not None:
                    reward_erase = independent_reward(
                        image, [f"a photo of a {item}" for item in batch_erase]
                    )
                    reward_retain = independent_reward(
                        image, [f"a photo of a {item}" for item in batch_retain]
                    )
                    batch_metrics[name]["imagereward_margin"] = (
                        reward_retain - reward_erase
                    )
                if detector is not None:
                    detector_erase = detector.confidences(image, batch_erase)
                    detector_retain = detector.confidences(image, batch_retain)
                    batch_metrics[name].update(
                        {
                            "detector_margin": detector_retain - detector_erase,
                            "detector_erase": detector_erase,
                            "detector_retain": detector_retain,
                        }
                    )
                for metric_name, tensor in batch_metrics[name].items():
                    values[name][metric_name].extend(tensor.float().cpu().tolist())

            for offset, (prompt, erase, retain) in enumerate(
                zip(batch_prompts, batch_erase, batch_retain)
            ):
                row = {
                    "prompt": prompt,
                    "erase": erase,
                    "retain": retain,
                    "seed": seed_base + seed_index,
                }
                for name in images:
                    for metric_name, tensor in batch_metrics[name].items():
                        row[f"{metric_name}_{name}"] = float(tensor[offset])
                rows.append(row)
                if grid_path is not None and seed_index == 0 and len(grid_rows) < 8:
                    grid_rows.append(
                        (
                            erase,
                            retain,
                            prompt,
                            {name: image[offset].detach().float().cpu() for name, image in images.items()},
                        )
                    )

    metrics = {}
    for name, metric_values in values.items():
        for metric_name, samples in metric_values.items():
            metrics[f"{metric_name}_{name}"] = sum(samples) / len(samples)
    for comparator in ("static", "swapped", "frozen"):
        per_scene_matched = _scene_pair_averages(values["matched"]["margin"])
        per_scene_comparator = _scene_pair_averages(values[comparator]["margin"])
        delta, se = _paired_summary(
            per_scene_matched, per_scene_comparator
        )
        metrics[f"matched_minus_{comparator}"] = delta
        metrics[f"matched_minus_{comparator}_se"] = se
    metrics["erase_suppression_vs_frozen"], _ = _paired_summary(
        values["frozen"]["erase_similarity"], values["matched"]["erase_similarity"]
    )
    metrics["retain_change_vs_frozen"], _ = _paired_summary(
        values["matched"]["retain_similarity"], values["frozen"]["retain_similarity"]
    )
    metrics["clipt_change_vs_frozen"], _ = _paired_summary(
        values["matched"]["clipt"], values["frozen"]["clipt"]
    )
    if independent_reward is not None:
        for comparator in ("static", "swapped", "frozen"):
            delta, se = _paired_summary(
                _scene_pair_averages(values["matched"]["imagereward_margin"]),
                _scene_pair_averages(values[comparator]["imagereward_margin"]),
            )
            metrics[f"imagereward_matched_minus_{comparator}"] = delta
            metrics[f"imagereward_matched_minus_{comparator}_se"] = se
        metrics["independent_go"] = bool(
            metrics["imagereward_matched_minus_static"]
            > 2 * metrics["imagereward_matched_minus_static_se"]
            and metrics["imagereward_matched_minus_swapped"]
            > 2 * metrics["imagereward_matched_minus_swapped_se"]
        )
    if detector is not None:
        valid = [
            erase >= detector_threshold and retain >= detector_threshold
            for erase, retain in zip(
                values["frozen"]["detector_erase"],
                values["frozen"]["detector_retain"],
            )
        ]
        valid_indices = [index for index, keep in enumerate(valid) if keep]
        metrics["detector_valid_tasks"] = len(valid_indices)
        metrics["detector_valid_scene_seeds"] = len(valid_indices) // 2
        metrics["detector_valid_fraction"] = len(valid_indices) / len(valid)

        def valid_values(method: str, metric_name: str) -> list[float]:
            return [values[method][metric_name][index] for index in valid_indices]

        if valid_indices:
            for comparator in ("static", "swapped", "frozen"):
                delta, se = _paired_summary(
                    _scene_pair_averages(valid_values("matched", "detector_margin")),
                    _scene_pair_averages(valid_values(comparator, "detector_margin")),
                )
                metrics[f"detector_matched_minus_{comparator}"] = delta
                metrics[f"detector_matched_minus_{comparator}_se"] = se
            metrics["detector_erase_suppression_vs_frozen"], _ = _paired_summary(
                valid_values("frozen", "detector_erase"),
                valid_values("matched", "detector_erase"),
            )
            metrics["detector_retain_change_vs_frozen"], _ = _paired_summary(
                valid_values("matched", "detector_retain"),
                valid_values("frozen", "detector_retain"),
            )
            for method in ("matched", "swapped", "static", "frozen"):
                success = [
                    erase < detector_threshold and retain >= detector_threshold
                    for erase, retain in zip(
                        valid_values(method, "detector_erase"),
                        valid_values(method, "detector_retain"),
                    )
                ]
                metrics[f"detector_selective_success_{method}"] = (
                    sum(success) / len(success)
                )
            metrics["detector_go"] = bool(
                metrics["detector_valid_scene_seeds"] >= 10
                and metrics["detector_matched_minus_static"]
                > 2 * metrics["detector_matched_minus_static_se"]
                and metrics["detector_matched_minus_swapped"]
                > 2 * metrics["detector_matched_minus_swapped_se"]
                and metrics["detector_erase_suppression_vs_frozen"] > 0
                and metrics["detector_retain_change_vs_frozen"] >= -0.02
                and metrics["detector_selective_success_matched"] >= 0.10
                and (
                    metrics["detector_selective_success_matched"]
                    - metrics["detector_selective_success_static"]
                )
                >= 0.05
            )
        else:
            metrics["detector_go"] = False
    metrics["go"] = bool(
        metrics["matched_minus_static"] > 2 * metrics["matched_minus_static_se"]
        and metrics["matched_minus_swapped"] > 2 * metrics["matched_minus_swapped_se"]
        and metrics["retain_change_vs_frozen"] > -0.02
    )
    if grid_path is not None:
        _write_grid(grid_path, grid_rows)
    return metrics, rows


def _write_grid(
    path: Path,
    rows: list[tuple[str, str, str, dict[str, torch.Tensor]]],
) -> None:
    """Write a compact labeled qualitative grid for manual inspection."""
    from PIL import Image, ImageDraw
    from torchvision.transforms.functional import to_pil_image

    methods = ("frozen", "matched", "swapped", "static")
    thumb = 192
    label_height = 42
    canvas = Image.new(
        "RGB", (thumb * len(methods), (thumb + label_height) * len(rows)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (erase, retain, _prompt, images) in enumerate(rows):
        top = row_index * (thumb + label_height)
        for column, method in enumerate(methods):
            image = to_pil_image(images[method].clamp(0, 1)).resize((thumb, thumb))
            canvas.paste(image, (column * thumb, top + label_height))
            draw.text(
                (column * thumb + 4, top + 3),
                f"{method}\nerase {erase}; keep {retain}",
                fill="black",
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = build_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    train_scenes, eval_scenes = build_selective_erasure_probe_scenes(
        train_size=args.train_scenes, eval_size=args.eval_scenes
    )

    torch.manual_seed(args.seed)
    print(f"[1/5] loading frozen SD-Turbo + CLIP on {args.device}", flush=True)
    gen = load_frozen_generator(args.device, torch.float32)
    scorer = ClipScorer(args.device)
    independent_reward = (
        ImageRewardReward(args.device) if args.independent_eval else None
    )
    detector = GroundingDinoScorer(args.device) if args.detector_eval else None
    targets = find_attention_linears(gen.unet)

    print("[2/5] building condition-generated and same-shape static LoRAs", flush=True)
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
        raise AssertionError("hypernetwork and static reference emit different adapter shapes")
    print(
        f"      targets={len(targets)} emitted={hyper.generated_parameter_count():,} "
        f"hyper-trainable={hyper.trainable_parameter_count():,}",
        flush=True,
    )

    def build_optimizer(parameters):
        if args.optimizer == "sgd":
            return torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum)
        return torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)

    hyper_opt = build_optimizer(hyper.parameters())
    static_opt = build_optimizer(static.parameters())
    rng = torch.Generator(device=args.device).manual_seed(args.seed + 1)
    history: list[dict] = []
    start = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=args.device, weights_only=False)
        hyper.load_state_dict(state["hyper"])
        static.load_state_dict(state["static"])
        hyper_opt.load_state_dict(state["hyper_opt"])
        static_opt.load_state_dict(state["static_opt"])
        rng.set_state(state["rng_state"].cpu())
        history = state["history"]
        start = state["step"]
        print(f"      resumed at step {start}", flush=True)
    elif args.eval_only:
        raise FileNotFoundError(f"--eval-only requires {checkpoint}")
    if args.eval_only and start != args.steps:
        raise ValueError(
            f"--eval-only requested step {args.steps}, checkpoint is at step {start}"
        )

    print(
        f"[3/5] training for {args.steps} steps on paired erase requests "
        f"from {len(train_scenes)} scenes",
        flush=True,
    )
    in_channels = gen.unet.config.in_channels
    t0 = time.time()
    for step in range(start, args.steps):
        chosen = [
            train_scenes[(step * args.scene_batch_size + offset) % len(train_scenes)]
            for offset in range(args.scene_batch_size)
        ]
        prompts, erase_objects, retain_objects = _paired_tasks(chosen)
        encoder_states, _ = embed_prompt_with_pool(gen, prompts)
        _, conditions = embed_prompt_with_pool(
            gen, [f"remove the {name}" for name in erase_objects]
        )
        retain_encoder_states = None
        if args.train_objective == "diffusion":
            retain_encoder_states, _ = embed_prompt_with_pool(
                gen, [f"a photo of a {name}" for name in retain_objects]
            )
        scene_latents = torch.randn(
            len(chosen), in_channels, 64, 64, device=args.device, generator=rng
        )
        latents = scene_latents.repeat_interleave(2, dim=0)
        reference = None
        if args.train_objective == "clip":
            with torch.no_grad():
                reference = generate_with_weight_adapter(
                    gen, None, latents, encoder_states
                )

        step_metrics = {}
        for name, adapter, optimizer, adapter_condition in (
            ("matched", hyper, hyper_opt, conditions),
            ("static", static, static_opt, None),
        ):
            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            if args.train_objective == "clip":
                loss, current = selective_erasure_clip_loss(
                    gen,
                    adapter,
                    latents,
                    encoder_states,
                    scorer,
                    erase_objects,
                    retain_objects,
                    condition_embeddings=adapter_condition,
                    reference_images=reference,
                    fidelity_weight=args.fidelity_weight,
                    reward_scale=args.reward_scale,
                    erase_weight=args.erase_weight,
                    retain_weight=args.retain_weight,
                )
            else:
                if retain_encoder_states is None:
                    raise AssertionError("diffusion objective requires retain embeddings")
                loss, current = selective_erasure_denoising_loss(
                    gen,
                    adapter,
                    latents,
                    encoder_states,
                    retain_encoder_states,
                    condition_embeddings=adapter_condition,
                    negative_guidance=args.negative_guidance,
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()
            step_metrics.update({f"{name}_{key}": value for key, value in current.items()})
            step_metrics[f"{name}_grad_norm"] = float(grad_norm)

        done = step + 1
        if done % args.log_every == 0 or done == args.steps:
            record = {"step": done, **step_metrics}
            history.append(record)
            if args.train_objective == "clip":
                detail = (
                    f"margin matched/static={record['matched_margin']:+.4f}/"
                    f"{record['static_margin']:+.4f} "
                    f"mse={record['matched_pixel_mse']:.4f}/"
                    f"{record['static_pixel_mse']:.4f}"
                )
            else:
                detail = (
                    f"loss matched/static={record['matched_loss']:.4f}/"
                    f"{record['static_loss']:.4f} "
                    f"shift={record['matched_prediction_shift_rms']:.4f}/"
                    f"{record['static_prediction_shift_rms']:.4f}"
                )
            print(f"      step {done:4d} {detail}", flush=True)
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
        f"[4/5] held-out eval: {len(eval_scenes)} scenes x two targets x "
        f"{args.eval_seeds} seed(s)",
        flush=True,
    )
    metrics, rows = evaluate(
        gen,
        hyper,
        static,
        scorer,
        eval_scenes,
        batch_size=args.eval_batch_size,
        n_seeds=args.eval_seeds,
        independent_reward=independent_reward,
        detector=detector,
        detector_threshold=args.detector_threshold,
        grid_path=output / "qualitative_grid.png" if args.save_grid else None,
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
        f"      CLIP margin matched-static = {metrics['matched_minus_static']:+.4f} "
        f"± {metrics['matched_minus_static_se']:.4f} SE",
        flush=True,
    )
    print(
        f"      CLIP margin matched-swapped = {metrics['matched_minus_swapped']:+.4f} "
        f"± {metrics['matched_minus_swapped_se']:.4f} SE",
        flush=True,
    )
    print(
        f"      erase suppression={metrics['erase_suppression_vs_frozen']:+.4f}; "
        f"retain change={metrics['retain_change_vs_frozen']:+.4f}; "
        f"CLIP-T change={metrics['clipt_change_vs_frozen']:+.4f}",
        flush=True,
    )
    if args.independent_eval:
        print(
            "      ImageReward margin matched-static = "
            f"{metrics['imagereward_matched_minus_static']:+.4f} ± "
            f"{metrics['imagereward_matched_minus_static_se']:.4f} SE; "
            "matched-swapped = "
            f"{metrics['imagereward_matched_minus_swapped']:+.4f} ± "
            f"{metrics['imagereward_matched_minus_swapped_se']:.4f} SE",
            flush=True,
        )
        print(
            f"      independent decision="
            f"{'GO' if metrics['independent_go'] else 'NO-GO'}",
            flush=True,
        )
    if args.detector_eval:
        print(
            f"      detector valid scene-seeds="
            f"{metrics['detector_valid_scene_seeds']} "
            f"({metrics['detector_valid_fraction']:.1%}); "
            f"matched-static={metrics.get('detector_matched_minus_static', float('nan')):+.4f} "
            f"± {metrics.get('detector_matched_minus_static_se', float('nan')):.4f}; "
            f"matched-swapped={metrics.get('detector_matched_minus_swapped', float('nan')):+.4f} "
            f"± {metrics.get('detector_matched_minus_swapped_se', float('nan')):.4f}",
            flush=True,
        )
        print(
            f"      detector target suppression="
            f"{metrics.get('detector_erase_suppression_vs_frozen', float('nan')):+.4f}; "
            f"retain change="
            f"{metrics.get('detector_retain_change_vs_frozen', float('nan')):+.4f}; "
            f"selective success matched/static="
            f"{metrics.get('detector_selective_success_matched', float('nan')):.1%}/"
            f"{metrics.get('detector_selective_success_static', float('nan')):.1%}; "
            f"decision={'GO' if metrics['detector_go'] else 'NO-GO'}",
            flush=True,
        )
    print(f"      decision={'GO' if metrics['go'] else 'NO-GO'}", flush=True)
    print(f"      wrote {output / 'probe_results.json'}", flush=True)


if __name__ == "__main__":
    main()
