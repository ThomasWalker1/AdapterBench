import json
from pathlib import Path

import pytest

from peft_hnet.text_to_lora_backend import ReleasedTextToLoRABackend


def _fake_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "hypermod.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    return checkpoint


def _fake_python(tmp_path: Path) -> Path:
    python = tmp_path / "fake_python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    return python


def test_generate_wraps_subprocess_manifest_into_adapter_artifacts(tmp_path, monkeypatch):
    checkpoint = _fake_checkpoint(tmp_path)
    python = _fake_python(tmp_path)
    output_dir = tmp_path / "out"

    def fake_run(cmd, check, cwd):
        # Emulate what scripts/generate_t2l_adapter.py would have written.
        out_index = cmd.index("--output-dir") + 1
        out = Path(cmd[out_index])
        out.mkdir(parents=True, exist_ok=True)
        (out / "arc_easy").mkdir(parents=True, exist_ok=True)
        manifest = {
            "checkpoint": str(checkpoint),
            "base_model": "google/gemma-2-2b-it",
            "representation": "lora",
            "model_load_seconds": 5.0,
            "torch": "2.4.0",
            "adapters": {
                "arc_easy": {
                    "condition": "Answer science questions.",
                    "path": str(out / "arc_easy"),
                    "generation_seconds": 0.1,
                    "generated_parameter_count": 1234,
                    "generated_bytes": 4936,
                    "peak_gpu_memory_bytes": 1024,
                }
            },
        }
        (out / "manifest.json").write_text(json.dumps(manifest))

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr("peft_hnet.text_to_lora_backend.subprocess.run", fake_run)

    backend = ReleasedTextToLoRABackend(checkpoint=checkpoint, python=python)
    artifacts = backend.generate({"arc_easy": "Answer science questions."}, output_dir)

    assert set(artifacts) == {"arc_easy"}
    artifact = artifacts["arc_easy"]
    assert artifact.task_id == "arc_easy"
    assert artifact.representation == "lora"
    assert artifact.format == "peft"
    assert artifact.generated_parameter_count == 1234
    assert artifact.generation_seconds == 0.1
    assert artifact.path == output_dir / "arc_easy"
    assert artifact.metadata["condition"] == "Answer science questions."
    assert artifact.metadata["model_load_seconds"] == 5.0


def test_missing_upstream_python_raises_before_any_subprocess_call(tmp_path):
    checkpoint = _fake_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="upstream text-to-lora interpreter"):
        ReleasedTextToLoRABackend(checkpoint=checkpoint, python=tmp_path / "does_not_exist")


def test_missing_checkpoint_raises(tmp_path):
    python = _fake_python(tmp_path)
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        ReleasedTextToLoRABackend(checkpoint=tmp_path / "missing.pt", python=python)
