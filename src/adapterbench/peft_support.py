"""Map benchmark adapter manifests onto the upstream Hugging Face PEFT API."""

from __future__ import annotations

from .schema import AdapterManifest


def make_peft_config(adapter: AdapterManifest, task_type: str = "CAUSAL_LM"):
    if adapter.implementation != "peft":
        raise ValueError(f"{adapter.name} requires its custom implementation")
    from peft import LoraConfig

    values = dict(adapter.hyperparameters)
    targets = adapter.target_modules or None
    # LoRA is the only baseline family; new families register their PEFT config here as
    # they are added one at a time with their leaderboard entries (see PROJECT_PLAN.md).
    constructors = {
        "lora": lambda: LoraConfig(task_type=task_type, target_modules=targets, **values),
    }
    return constructors[adapter.family]()


def count_adapter_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
