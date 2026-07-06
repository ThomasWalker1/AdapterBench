"""Portable flatten/unflatten contract for arbitrary generated adapter states."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class TensorLayout:
    name: str
    shape: tuple[int, ...]
    offset: int
    count: int
    dtype: str


class AdapterStateLayout:
    def __init__(self, tensors: tuple[TensorLayout, ...]):
        self.tensors = tensors
        expected = 0
        for tensor in tensors:
            if tensor.offset != expected:
                raise ValueError("tensor layout must be contiguous")
            expected += tensor.count
        self.parameter_count = expected

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Tensor]) -> "AdapterStateLayout":
        offset = 0
        layouts = []
        for name, value in sorted(state.items()):
            layouts.append(TensorLayout(name, tuple(value.shape), offset, value.numel(), str(value.dtype)))
            offset += value.numel()
        return cls(tuple(layouts))

    def flatten(self, state: Mapping[str, Tensor]) -> Tensor:
        missing = {item.name for item in self.tensors} - set(state)
        if missing:
            raise KeyError(f"adapter state is missing tensors: {sorted(missing)}")
        values = [state[item.name].reshape(-1) for item in self.tensors]
        return torch.cat(values) if values else torch.empty(0)

    def unflatten(self, vector: Tensor) -> dict[str, Tensor]:
        if vector.shape[-1] != self.parameter_count:
            raise ValueError(f"expected last dimension {self.parameter_count}, got {vector.shape[-1]}")
        return {
            item.name: vector[..., item.offset : item.offset + item.count].reshape(*vector.shape[:-1], *item.shape)
            for item in self.tensors
        }

    def write_metadata(self, path: str | Path) -> None:
        payload = {"parameter_count": self.parameter_count, "tensors": [asdict(item) for item in self.tensors]}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def trainable_state_dict(model) -> dict[str, Tensor]:
    return {name: parameter.detach() for name, parameter in model.named_parameters() if parameter.requires_grad}

