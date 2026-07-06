"""Small helpers for resolving a live HF causal LM's structure."""

from __future__ import annotations

from torch import nn


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return a live model's decoder-layer list, port of upstream's ``get_layers``.

    Handles the common HF wrapping convention (``AutoModelForCausalLM`` ->
    ``model.model.layers``) by recursing through any ``.model`` attribute until a
    ``.layers`` attribute is found.
    """
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model"):
        return get_decoder_layers(model.model)
    raise AttributeError(f"could not find a decoder-layer list on {type(model).__name__}")
