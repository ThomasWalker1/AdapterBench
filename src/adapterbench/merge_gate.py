"""The correctness merge gate for a codec PR (PROJECT_PLAN § "Git-native benchmark").

A codec merges iff it is a *valid, deterministic, fairly-comparable* panel member — **never because
it won**. Quality (did it beat LoRA) is answered *after* merge by the autoresearch evaluation, as
records. This module holds the gate's pure, unit-testable logic:

- `check_path_guard` — a codec PR may touch only the codec registry region, `configs/adapters/`,
  `schema.py`'s family list, and `results/`. Anything else is a *substrate* change and must go down
  a separate, more-scrutinized path (it invalidates cross-sha leaderboard comparability).
- `check_shape_identity` — the codec's per-target generated-output size (at the manifest's declared
  reference dimension) must fall within the panel's declared budget band, so shapes compete at
  matched capacity rather than "as dense as the budget allows".

`run_gate` sequences these plus the existing correctness commands (`validate`, `catalog`,
`pytest -q`, and the `peft-smoke` / generate→hook→backprop check) via an injected `runner`, so the
orchestration is testable without a GPU. See `scripts/codec_merge_gate.py` for the CLI.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from .schema import AdapterManifest

# A codec PR may touch ONLY these paths (the merge unit). Anything else is a substrate change.
ALLOWED_CODEC_PR_FILES = (
    "src/adapterbench/t2p/codecs.py",     # the GeneratedUpdateCodec subclass + make_codec entry
    "src/adapterbench/schema.py",         # the family Literal (one line)
)
ALLOWED_CODEC_PR_DIRS = (
    "configs/adapters/",                  # the manifest
    "results/",                           # derived records (leaderboard); never gates
)
ALLOWED_CODEC_PR_GLOBS = ALLOWED_CODEC_PR_FILES + tuple(d + "*" for d in ALLOWED_CODEC_PR_DIRS)


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class GateReport:
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, check: GateCheck) -> GateCheck:
        self.checks.append(check)
        return check

    def render(self) -> str:
        lines = ["=" * 70, "codec merge gate — CORRECTNESS ONLY (never quality)", "=" * 70]
        for c in self.checks:
            lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name:<22} {c.detail}")
        lines += ["-" * 70, f"GATE: {'PASS — mergeable panel member' if self.passed else 'FAIL — not mergeable'}",
                  "=" * 70]
        return "\n".join(lines)


def _matches_allowed(path: str) -> bool:
    path = path.strip().lstrip("./")
    if path in ALLOWED_CODEC_PR_FILES:
        return True
    return any(path.startswith(d) for d in ALLOWED_CODEC_PR_DIRS)


def check_path_guard(changed_files: Iterable[str]) -> GateCheck:
    """A codec PR may touch only the allowed paths (codec registry + manifest + family Literal +
    results). A change anywhere else is a substrate change and fails the gate — forcing it down the
    separate, more-scrutinized substrate path that a shape proposal must not take."""
    changed = [f for f in changed_files if f.strip()]
    offenders = [f for f in changed if not _matches_allowed(f)]
    if offenders:
        return GateCheck("path-guard", False,
                         f"touches non-codec (substrate) paths: {offenders}. Allowed: {list(ALLOWED_CODEC_PR_GLOBS)}")
    return GateCheck("path-guard", True, f"{len(changed)} file(s), all within the codec merge unit")


def generated_output_size(family: str, hyperparameters: dict, reference_dim: int) -> int:
    """The codec's per-target generated-output size on a square `reference_dim` linear — the
    capacity number the shape-identity lint compares. Uses the real `make_codec`, so it stays
    correct as new families are added."""
    from .t2p.codecs import make_codec

    codec = make_codec(
        family, reference_dim, reference_dim, num_layers=1,
        rank=hyperparameters.get("r", hyperparameters.get("rank", 8)),
        alpha=hyperparameters.get("lora_alpha", hyperparameters.get("alpha", 16.0)),
    )
    return codec.output_size


def check_shape_identity(manifest: AdapterManifest) -> GateCheck:
    """The codec's generated-output size at the declared reference dim must not exceed the budget
    band — shapes compete at matched capacity. A manifest with no `parameter_budget` is not
    eligible for a matched-capacity panel and fails (declaring the budget is part of joining)."""
    budget = manifest.parameter_budget
    if budget is None:
        return GateCheck("shape-identity", False,
                         "no parameter_budget declared; a matched-capacity panel member must declare one")
    size = generated_output_size(manifest.family, manifest.hyperparameters, budget.reference_dim)
    if size > budget.max_output_size:
        return GateCheck("shape-identity", False,
                         f"generated output {size} > budget {budget.max_output_size} "
                         f"@ dim {budget.reference_dim} — exceeds the panel's capacity band")
    return GateCheck("shape-identity", True,
                     f"generated output {size} <= budget {budget.max_output_size} @ dim {budget.reference_dim}")


def run_gate(
    manifest: AdapterManifest, changed_files: Sequence[str], *,
    runner: Callable[[list[str]], tuple[int, str]], run_gpu_smoke: bool = True,
) -> GateReport:
    """Sequence the full correctness gate. `runner(argv) -> (returncode, output)` executes the
    existing CLI commands (injected so tests can stub them); a non-zero return fails that check.
    Correctness only — this never inspects whether the codec beat LoRA."""
    report = GateReport()
    report.add(check_path_guard(changed_files))
    report.add(check_shape_identity(manifest))

    steps = [
        ("validate", [".venv/bin/adapterbench", "validate"]),
        ("catalog", [".venv/bin/adapterbench", "catalog"]),
        ("pytest", [".venv/bin/python", "-m", "pytest", "-q"]),
    ]
    if run_gpu_smoke:
        # proves generate -> hook -> backprop through hypernetwork.apply(...) (not peft.load_adapter)
        steps.append(("gpu-smoke", [".venv/bin/adapterbench", "peft-smoke", "--adapters", manifest.name]))
    for name, argv in steps:
        rc, out = runner(argv)
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        report.add(GateCheck(name, rc == 0, (f"rc={rc}" + (f": {tail}" if tail else ""))))
    return report
