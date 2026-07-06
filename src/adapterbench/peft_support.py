"""Map benchmark adapter manifests onto the upstream Hugging Face PEFT API."""

from __future__ import annotations

from .schema import AdapterManifest


def make_peft_config(adapter: AdapterManifest, task_type: str = "CAUSAL_LM"):
    if adapter.implementation != "peft":
        raise ValueError(f"{adapter.name} requires its custom implementation")
    from peft import FourierFTConfig, IA3Config, LoKrConfig, LoraConfig, PrefixTuningConfig

    values = dict(adapter.hyperparameters)
    targets = adapter.target_modules or None
    constructors = {
        "lora": lambda: LoraConfig(task_type=task_type, target_modules=targets, **values),
        "fourierft": lambda: FourierFTConfig(task_type=task_type, target_modules=targets, **values),
        "lokr": lambda: LoKrConfig(target_modules=targets, **values),
        "ia3": lambda: IA3Config(task_type=task_type, target_modules=targets, **values),
        "prefix_tuning": lambda: PrefixTuningConfig(task_type=task_type, **values),
    }
    return constructors[adapter.family]()


def count_adapter_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

