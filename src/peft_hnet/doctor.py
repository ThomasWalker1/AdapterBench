from __future__ import annotations

from importlib import import_module
import platform


def environment_report() -> dict:
    report: dict = {"python": platform.python_version(), "packages": {}}
    for name in ("torch", "transformers", "datasets", "peft", "accelerate", "pydantic", "yaml"):
        try:
            module = import_module(name)
            report["packages"][name] = getattr(module, "__version__", "installed")
        except Exception as error:
            report["packages"][name] = f"missing ({type(error).__name__})"
    try:
        import torch

        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as error:
        report["cuda"] = {"available": False, "error": repr(error)}
    return report

