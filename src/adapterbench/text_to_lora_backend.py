"""HypernetworkBackend that generates LoRA adapters from a released Text-to-LoRA checkpoint.

Generation runs out-of-process under the upstream Text-to-LoRA environment
(``upstream/text-to-lora/.venv``): ``hyper_llm_modulator`` pins a torch/transformers/peft
stack incompatible with ``adapterbench``'s own dependencies (see SETUP.md). This backend never
imports ``hyper_llm_modulator`` directly - it shells out to ``scripts/generate_t2l_adapter.py``
and reads back the adapter directories + JSON manifest it writes.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Mapping

from .contracts import AdapterArtifact, HypernetworkBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_T2L_DIR = REPO_ROOT / "upstream" / "text-to-lora"
UPSTREAM_T2L_PYTHON = UPSTREAM_T2L_DIR / ".venv" / "bin" / "python"
GENERATE_SCRIPT = REPO_ROOT / "scripts" / "generate_t2l_adapter.py"


class ReleasedTextToLoRABackend(HypernetworkBackend):
    """Wraps a released ``hypermod.pt`` checkpoint; generation happens in a subprocess."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda:0",
        python: str | Path = UPSTREAM_T2L_PYTHON,
    ):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.python = Path(python)
        if not self.python.exists():
            raise FileNotFoundError(
                f"upstream text-to-lora interpreter not found at {self.python}; "
                "see SETUP.md to provision the upstream/text-to-lora uv venv"
            )
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint}")

    def generate(self, conditions: Mapping[str, str], output_dir: Path) -> Mapping[str, AdapterArtifact]:
        # Resolve every path to absolute before invoking the subprocess: it runs with
        # cwd=UPSTREAM_T2L_DIR (required so hyper_llm_modulator's own relative chat
        # template lookups resolve), so relative paths here would resolve against the
        # wrong directory.
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        conditions_path = output_dir / "conditions.json"
        conditions_path.write_text(json.dumps(dict(conditions)))

        started = time.perf_counter()
        subprocess.run(
            [
                # Do NOT .resolve() the interpreter path: it is a symlink into the
                # upstream venv, and fully resolving it collapses to the real system
                # interpreter, which skips venv site-package activation.
                str(self.python),
                str(GENERATE_SCRIPT.resolve()),
                "--checkpoint",
                str(self.checkpoint.resolve()),
                "--conditions-json",
                str(conditions_path),
                "--output-dir",
                str(output_dir),
                "--device",
                self.device,
            ],
            check=True,
            cwd=UPSTREAM_T2L_DIR,
        )
        subprocess_wall_seconds = time.perf_counter() - started

        manifest = json.loads((output_dir / "manifest.json").read_text())
        artifacts: dict[str, AdapterArtifact] = {}
        for task_id, entry in manifest["adapters"].items():
            artifacts[task_id] = AdapterArtifact(
                task_id=task_id,
                adapter=manifest["adapter"],
                path=Path(entry["path"]),
                format="peft",
                generated_parameter_count=entry["generated_parameter_count"],
                generation_seconds=entry["generation_seconds"],
                metadata={
                    "condition": entry["condition"],
                    "checkpoint": manifest["checkpoint"],
                    "base_model": manifest["base_model"],
                    "generated_bytes": entry["generated_bytes"],
                    "peak_gpu_memory_bytes_generation": entry["peak_gpu_memory_bytes"],
                    "model_load_seconds": manifest["model_load_seconds"],
                    "subprocess_wall_seconds": subprocess_wall_seconds,
                },
            )
        return artifacts
