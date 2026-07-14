"""Declarative `SettingSpec`s for the three benchmark settings the autoresearch driver wraps.

Each spec encodes (a) the **HP partition** — which HPs are the frozen shared substrate, which are
free to search, and which are shape-identity (fixed by definition) — and (b) how to read a
setting's result cells: the free-HP getters, the seed getter, and the **matched − control**
objective (never loss). Validated end-to-end on I2P (its cells carry the full free-HP metadata; see
`tests/test_autoresearch.py` and the read-only validation against `results/i2p_hypernoise_v2`). The
D2L and T2L specs use the objective extractors confirmed from their record formats
(`accuracy − accuracy_ctxswap`); searching a given free HP there requires that HP to appear in the
runner's cell metadata.
"""

from __future__ import annotations

from pathlib import Path

from .autoresearch import FREE, SHAPE_IDENTITY, SUBSTRATE, SettingSpec


# ── I2P: reward-tilting on frozen SD-Turbo (i2p-hypernoise) ───────────────────────────────────
def _i2p_objective(cells: list[dict]) -> float:
    """matched − frozen: the ImageReward gain of the imagereward-target adapter (frozen is already
    subtracted inside `reward_imagereward_gain`). Red (reward-swap) cells sharing the (scale,reg)
    are ignored here — that control is a separate validity gate, not the selection axis."""
    gains = [c["metrics"]["reward_imagereward_gain"]
             for c in cells if c["metadata"].get("reward_target") == "imagereward"]
    return sum(gains) / len(gains) if gains else float("nan")


def _i2p_launch(codec: str, config: dict, seed: int, out_dir: Path) -> list[str]:
    argv = [".venv/bin/adapterbench", "i2p-hypernoise", "--reward", "imagereward",
            "--seed", str(seed), "--output", str(out_dir)]
    if "scale" in config:
        argv += ["--scale", str(config["scale"])]
    if "reg_weight" in config:
        argv += ["--reg-weight", str(config["reg_weight"])]
    if "steps" in config:
        argv += ["--steps", str(int(config["steps"]))]
    if "eval_every" in config:
        argv += ["--eval-every", str(int(config["eval_every"]))]
    if "n_seeds" in config:
        argv += ["--n-seeds", str(int(config["n_seeds"]))]
    if "batch_size" in config:
        argv += ["--batch-size", str(int(config["batch_size"]))]
    return argv


# Run profiles. `proxy` is the lightweight-but-faithful profile: for I2P the scale ranking is
# single-seed noise until ~step 1500 (verified from the full run's per-step eval history), so the
# proxy halves steps (3000→1500) and eval breadth rather than slashing training, and stays
# multi-seed. Validate a proxy recovers the full profile's best-of before trusting it.
_I2P_PROFILES = {
    "full":  {"steps": 3000, "eval_every": 500, "n_seeds": 2, "batch_size": 4},
    "proxy": {"steps": 1500, "eval_every": 750, "n_seeds": 2, "batch_size": 2},
}


I2P_SPEC = SettingSpec(
    name="i2p_hypernoise",
    hp_classes={
        "scale": FREE, "reg_weight": FREE, "lr": FREE, "warmup": FREE, "steps": FREE,
        "rank": SHAPE_IDENTITY,           # a shape-identity HP: fixed, never maximized
        "reward": SUBSTRATE,              # the task target
        "batch_size": SUBSTRATE, "n_seeds": SUBSTRATE,  # eval protocol
    },
    hp_getters={
        "scale": lambda c: c["metadata"]["lora_scale"],
        "reg_weight": lambda c: c["metadata"]["reg_weight"],
        "steps": lambda c: c["metadata"]["steps"],
    },
    seed_getter=lambda c: c["metadata"]["seed"],
    objective=_i2p_objective,
    objective_name="ImageReward gain (adapter − frozen)",
    launch_argv=_i2p_launch,
    profiles=_I2P_PROFILES,
)


# ── D2L-NIAH: document-conditioned retrieval (d2p-niah) ───────────────────────────────────────
def _d2l_objective(cells: list[dict]) -> float:
    """matched − control = accuracy − accuracy_ctxswap for the *generated* (non-frozen) family
    record — the gold-standard context-swap control (adapter from the wrong document must sit near
    chance). Frozen-family records (no adapter) are skipped."""
    vals = []
    for c in cells:
        if c["metadata"].get("adapter_family") in (None, "frozen"):
            continue
        acc = c["metrics"].get("accuracy")
        swap = c["metrics"].get("accuracy_ctxswap")
        if acc is not None and swap is not None:
            vals.append(acc - swap)
    return sum(vals) / len(vals) if vals else float("nan")


D2L_NIAH_SPEC = SettingSpec(
    name="d2p_niah",
    hp_classes={
        "scale": FREE, "lr": FREE, "warmup": FREE, "steps": FREE,
        "rank": SHAPE_IDENTITY,
        "exit_layer": SUBSTRATE, "n_latents": SUBSTRATE, "num_blocks": SUBSTRATE,  # conditioner/trunk
    },
    hp_getters={
        "scale": lambda c: c["metadata"]["scale"],
        "steps": lambda c: c["metadata"]["steps"],
    },
    seed_getter=lambda c: c["metadata"]["seed"],
    objective=_d2l_objective,
    objective_name="NIAH retrieval (accuracy − context-swap)",
    launch_argv=None,  # wire to d2p-niah argv when running launch mode on this setting
)


# ── T2L: task-description conditioning (t2p-sft-pilot --adversarial-control) ───────────────────
def _t2l_objective(cells: list[dict]) -> float:
    """matched − adversarial, averaged over eval families: accuracy − accuracy_ctxswap on each
    generated (non-frozen) family record."""
    vals = []
    for c in cells:
        if c["metadata"].get("adapter") in (None, "frozen_interpreter"):
            continue
        acc = c["metrics"].get("accuracy")
        swap = c["metrics"].get("accuracy_ctxswap")
        if acc is not None and swap is not None:
            vals.append(acc - swap)
    return sum(vals) / len(vals) if vals else float("nan")


T2L_SPEC = SettingSpec(
    name="t2p_sft_pilot",
    hp_classes={
        "scale": FREE, "learning_rate": FREE, "warmup_frac": FREE, "steps": FREE,
        "rank": SHAPE_IDENTITY,
        "max_descriptions": SUBSTRATE, "eval_tasks": SUBSTRATE, "eval_limit": SUBSTRATE,
    },
    hp_getters={
        "scale": lambda c: c["metadata"].get("scale", "default"),
        "steps": lambda c: c["metadata"]["steps"],
    },
    seed_getter=lambda c: c["metadata"]["seed"],
    objective=_t2l_objective,
    objective_name="T2L conditioning (accuracy − adversarial-mismatch)",
    launch_argv=None,
)


REGISTRY = {s.name: s for s in (I2P_SPEC, D2L_NIAH_SPEC, T2L_SPEC)}
