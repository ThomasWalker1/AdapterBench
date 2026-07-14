"""Per-codec autoresearch: the free-HP search that produces a codec's leaderboard numbers.

This is the post-merge *Evaluate* step of the git-native pipeline (PROJECT_PLAN § "Per-codec
autoresearch" and § "Git-native benchmark"): fix a codec's *shape*, search its *free* optimization
HPs to best-of on the frozen shared substrate, multi-seed, scored on **matched − control (never
loss)**, and report the codec at its own best config. Built and validated on the LoRA baseline —
new shapes are the payload, not a prerequisite (LoRA exercises every path here).

The module is deliberately split into (a) a declarative **HP partition** per setting — the trap
that makes or breaks a *shape* benchmark, encoded so a future codec cannot cheat by tuning the
substrate — and (b) a setting-agnostic **search driver** (partition enforcement, multi-seed
aggregation, best-of selection, budget reporting). The driver reads the *same* restart-safe result
cells the settings already emit (`results/<cell>/results.jsonl`), so it wraps the existing runners
(`i2p-hypernoise`, `d2p-niah`, `t2p-sft-pilot`) rather than reinventing training.

Three guardrails, each grounded in a bug this project hit (PROJECT_PLAN §§ Optimization):
  1. optimize `matched − control`, never loss (loss 0.02 co-existed with 0.2 retrieval);
  2. every config is multi-seed (the NIAH transition is stochastic, ~1500–4500 steps);
  3. equal search budget + space per codec, both reported (publish the tuned-HP table, not a scalar).
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# ── The HP partition (invariant #2, generalized) ─────────────────────────────────────────────
# Every HP a setting exposes is exactly one of these. The driver refuses to *search* anything that
# is not FREE, so the comparison stays about the adapter's shape and not about who tuned the
# substrate hardest.
SUBSTRATE = "substrate"            # identical across codecs, NEVER tuned (task data, conditioner/
#                                    trunk, evaluator, control). Tuning it stops the comparison
#                                    being about the adapter.
FREE = "free"                      # the loop MAY search these (lr, warmup, steps, scale).
SHAPE_IDENTITY = "shape_identity"  # fixed by definition, or moved ONLY along the parameter-
#                                    efficiency axis — never *maximized* (e.g. rank). A loop that
#                                    freely maximizes these drives every shape "as dense as the
#                                    budget allows" and "shape" dissolves.


@dataclass(frozen=True)
class SettingSpec:
    """Declarative description of one benchmark setting for the search driver.

    `objective` receives the cells that share a single (config, seed) and returns that trial's
    **matched − control** scalar (never a loss). `hp_getters` reads each FREE HP's value out of a
    cell's metadata so cells can be grouped into configs; `seed_getter` reads the seed.
    `launch_argv`, if given, builds the argv to *produce* a missing cell via the setting's existing
    restart-safe CLI (launch mode); leave it None to run the driver read-only over existing cells.
    """

    name: str
    hp_classes: Mapping[str, str]                  # every exposed HP -> SUBSTRATE|FREE|SHAPE_IDENTITY
    hp_getters: Mapping[str, Callable[[dict], float]]  # FREE hp -> (cell) -> normalized value
    seed_getter: Callable[[dict], int]
    objective: Callable[[list[dict]], float]       # (cells for one config+seed) -> matched-control
    objective_name: str
    launch_argv: Callable[[str, dict, int, Path], list[str]] | None = None

    def hps_of_class(self, cls: str) -> list[str]:
        return [h for h, c in self.hp_classes.items() if c == cls]

    def validate_search_space(self, search_dims: Mapping[str, Sequence], fixed_free: Mapping[str, object]) -> None:
        """Enforce the HP partition: only FREE HPs may be searched or pinned by the loop. Raises
        ValueError naming any SUBSTRATE/SHAPE_IDENTITY (or unknown) HP the caller tried to touch —
        the guard that keeps this a *shape* benchmark rather than a substrate-tuning contest."""
        touched = list(search_dims) + list(fixed_free)
        offenders = {}
        for hp in touched:
            cls = self.hp_classes.get(hp)
            if cls != FREE:
                offenders[hp] = cls or "unknown"
        if offenders:
            raise ValueError(
                f"[{self.name}] these HPs are not FREE and may not be searched/pinned by the "
                f"autoresearch loop: {offenders}. Only FREE HPs "
                f"({self.hps_of_class(FREE)}) are searchable; SUBSTRATE is fixed across codecs and "
                f"SHAPE_IDENTITY is fixed by definition (never maximized)."
            )


# ── Aggregation + selection (pure, unit-testable) ────────────────────────────────────────────
def _mean_std(xs: Sequence[float]) -> tuple[float, float, int]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / n) if n > 1 else 0.0
    return m, sd, n


@dataclass(frozen=True)
class ConfigResult:
    config: tuple                # (free-hp-name, value) pairs, sorted — the tuned point
    objective_mean: float        # mean matched-control over seeds
    objective_std: float
    n_seeds: int
    per_seed: dict               # seed -> objective value (transparency; multi-seed guardrail)


@dataclass
class SearchReport:
    setting: str
    codec: str
    objective_name: str
    partition: dict              # class -> [hp names]
    search_dims: dict            # searched free HP -> candidate values
    fixed_free: dict             # pinned free HP -> value
    n_seeds_requested: int
    configs: list                # list[ConfigResult], sorted best-first
    best: ConfigResult | None
    min_seeds: int = 1           # guardrail #2: a config must meet this to be eligible as best
    best_under_seeded: bool = False  # True if no config met min_seeds and best is a fallback

    @property
    def search_budget(self) -> dict:
        """Guardrail #3: N and the space are part of the result. Report both."""
        return {
            "n_configs": len(self.search_dims and list(itertools.product(*self.search_dims.values())) or [()]),
            "n_seeds": self.n_seeds_requested,
            "n_trials": len(self.search_dims and list(itertools.product(*self.search_dims.values())) or [()]) * self.n_seeds_requested,
            "search_space": {k: list(v) for k, v in self.search_dims.items()},
        }

    def to_dict(self) -> dict:
        return {
            "setting": self.setting,
            "codec": self.codec,
            "objective": self.objective_name,
            "partition": self.partition,
            "search_budget": self.search_budget,
            "fixed_free": self.fixed_free,
            "min_seeds": self.min_seeds,
            "best_under_seeded": self.best_under_seeded,
            "best_config": dict(self.best.config) if self.best else None,
            "best_objective_mean": self.best.objective_mean if self.best else None,
            "best_objective_std": self.best.objective_std if self.best else None,
            "tuned_hp_table": [
                {"config": dict(c.config), "objective_mean": c.objective_mean,
                 "objective_std": c.objective_std, "n_seeds": c.n_seeds, "per_seed": c.per_seed}
                for c in self.configs
            ],
        }

    def render(self) -> str:
        lines = [
            "=" * 78,
            f"autoresearch: {self.setting} / codec={self.codec}",
            f"objective (maximize, best-of): {self.objective_name}   [never loss]",
            "=" * 78,
            f"HP partition:  FREE={self.partition[FREE]}",
            f"               SUBSTRATE (fixed)={self.partition[SUBSTRATE]}",
            f"               SHAPE_IDENTITY (fixed)={self.partition[SHAPE_IDENTITY]}",
            f"search space:  {self.search_budget['search_space']}   fixed_free={self.fixed_free}",
            f"budget:        {self.search_budget['n_configs']} configs × {self.n_seeds_requested} seeds "
            f"= {self.search_budget['n_trials']} trials",
            "-" * 78,
            "tuned-HP table (best-first):",
        ]
        for c in self.configs:
            star = "  <-- best-of" if self.best and c.config == self.best.config else ""
            under = "  (under-seeded, ineligible)" if c.n_seeds < self.min_seeds else ""
            cfg = ", ".join(f"{k}={v:g}" for k, v in c.config)
            lines.append(f"  {cfg:<28} {c.objective_mean:+.4f} ± {c.objective_std:.4f}  (n={c.n_seeds})"
                         f"  {[round(v, 3) for v in c.per_seed.values()]}{star}{under}")
        if self.best:
            best_cfg = ", ".join(f"{k}={v:g}" for k, v in self.best.config)
            note = "  [WARNING: no config met min_seeds; this is an under-seeded fallback]" if self.best_under_seeded else ""
            lines += ["-" * 78,
                      f"REPORTED AT ITS OWN BEST CONFIG (min_seeds={self.min_seeds}): {best_cfg}  ->  "
                      f"{self.best.objective_mean:+.4f} ± {self.best.objective_std:.4f}{note}"]
        lines.append("=" * 78)
        return "\n".join(lines)


def _norm(v) -> float:
    """Normalize an HP value to a float for matching (I2P stores scale as the string '4' or
    'default'; 'default' -> a sentinel that only equals another 'default')."""
    if isinstance(v, str):
        return -1.0 if v == "default" else float(v)
    return float(v)


def load_cells(out_root: Path) -> list[dict]:
    """Every EvaluationResult record under `out_root/*/results.jsonl` (the layout the pipeline and
    the settings' CLIs already write)."""
    cells = []
    for path in sorted(Path(out_root).glob("*/results.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                cells.append(json.loads(line))
    return cells


def search_over_cells(
    spec: SettingSpec, cells: list[dict], *, search_dims: Mapping[str, Sequence],
    fixed_free: Mapping[str, object] | None = None, n_seeds: int, codec: str = "lora",
    min_seeds: int | None = None,
) -> SearchReport:
    """Run the per-codec free-HP search over already-computed `cells` (read-only): enforce the
    partition, group cells into (config, seed), aggregate the matched-control objective over seeds,
    and select the config with the best mean. This is the selection logic the leaderboard depends
    on; launch mode (below) only adds "produce the missing cells first, then call this".

    `min_seeds` (default = `n_seeds`) is guardrail #2: a config must have at least this many seeds
    to be eligible as the reported best — a single-seed point estimate must not out-rank a
    multi-seed one on noise. Under-seeded configs still appear in the table (transparency); if none
    meet `min_seeds`, the best available is returned with `best_under_seeded=True`.
    """
    fixed_free = dict(fixed_free or {})
    min_seeds = n_seeds if min_seeds is None else min_seeds
    spec.validate_search_space(search_dims, fixed_free)

    dim_names = list(search_dims)
    candidate_sets = {k: {_norm(v) for v in vs} for k, vs in search_dims.items()}
    fixed_norm = {k: _norm(v) for k, v in fixed_free.items()}

    # config -> seed -> [cells]
    grouped: dict[tuple, dict[int, list[dict]]] = {}
    for cell in cells:
        try:
            vals = {hp: _norm(spec.hp_getters[hp](cell)) for hp in dim_names}
            fixed_vals = {hp: _norm(spec.hp_getters[hp](cell)) for hp in fixed_free}
        except (KeyError, TypeError, ValueError):
            continue  # a cell missing a needed HP (e.g. a control-only record) is not a search point
        if any(fixed_vals[hp] != fixed_norm[hp] for hp in fixed_free):
            continue
        if any(vals[hp] not in candidate_sets[hp] for hp in dim_names):
            continue
        config = tuple(sorted((hp, vals[hp]) for hp in dim_names))
        seed = spec.seed_getter(cell)
        grouped.setdefault(config, {}).setdefault(seed, []).append(cell)

    results: list[ConfigResult] = []
    for config, by_seed in grouped.items():
        per_seed = {seed: spec.objective(cs) for seed, cs in by_seed.items()}
        m, sd, n = _mean_std(list(per_seed.values()))
        results.append(ConfigResult(config=config, objective_mean=m, objective_std=sd,
                                    n_seeds=n, per_seed=dict(sorted(per_seed.items()))))
    # best-of on the mean objective (NaN configs sink to the bottom)
    results.sort(key=lambda c: (c.objective_mean if not math.isnan(c.objective_mean) else -math.inf),
                 reverse=True)

    # Guardrail #2: prefer configs meeting min_seeds; only fall back to under-seeded if none do.
    def _valid(c):
        return not math.isnan(c.objective_mean)
    eligible = [c for c in results if _valid(c) and c.n_seeds >= min_seeds]
    under_seeded = False
    if eligible:
        best = eligible[0]
    else:
        fallback = [c for c in results if _valid(c)]
        best = fallback[0] if fallback else None
        under_seeded = best is not None

    return SearchReport(
        setting=spec.name, codec=codec, objective_name=spec.objective_name,
        partition={FREE: spec.hps_of_class(FREE), SUBSTRATE: spec.hps_of_class(SUBSTRATE),
                   SHAPE_IDENTITY: spec.hps_of_class(SHAPE_IDENTITY)},
        search_dims={k: list(v) for k, v in search_dims.items()}, fixed_free=fixed_free,
        n_seeds_requested=n_seeds, configs=results, best=best,
        min_seeds=min_seeds, best_under_seeded=under_seeded,
    )


def launch_missing_cells(
    spec: SettingSpec, out_root: Path, *, search_dims: Mapping[str, Sequence],
    fixed_free: Mapping[str, object], seeds: Sequence[int], codec: str,
    gpus: Sequence[str], run: Callable[[list[str], Path], object],
) -> None:
    """Launch mode: produce every (config × seed) cell that does not yet exist by invoking the
    setting's restart-safe CLI (`spec.launch_argv`), fanning across `gpus`. `run(argv, out_dir)`
    executes one cell (injected so tests can stub it); a cell whose `results.jsonl` exists is
    skipped, so the whole driver is restart-safe. Selection is then `search_over_cells`."""
    if spec.launch_argv is None:
        raise ValueError(f"[{spec.name}] has no launch_argv; run the driver read-only over existing cells")
    spec.validate_search_space(search_dims, fixed_free)
    jobs = []
    for combo in itertools.product(*search_dims.values()):
        config = dict(zip(search_dims, combo))
        config.update(fixed_free)
        for seed in seeds:
            jobs.append((config, seed))
    for i, (config, seed) in enumerate(jobs):
        out_dir = Path(out_root) / _cell_dirname(spec, codec, config, seed)
        if (out_dir / "results.jsonl").exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = spec.launch_argv(codec, config, seed, out_dir)
        run(argv, out_dir)  # caller handles GPU assignment / concurrency via `run`


def _cell_dirname(spec: SettingSpec, codec: str, config: dict, seed: int) -> str:
    parts = [spec.name, codec] + [f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}{v}"
                                  for k, v in sorted(config.items())] + [f"s{seed}"]
    return "__".join(parts)
