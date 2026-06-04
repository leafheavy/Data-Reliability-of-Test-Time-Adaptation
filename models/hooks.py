from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Layer '{name}' not found in model modules")
    return modules[name]


def register_activation_hooks(model: nn.Module, layer_names: List[str]) -> Dict[str, List[torch.Tensor]]:
    """Register forward hooks and return a mutable activation store.

    The returned dictionary is populated as ``activations[layer_name].append(tensor)``
    after each forward pass. Clear lists before a new measurement. Hook handles
    are attached to ``activations['_handles']`` for optional cleanup.
    """

    activations: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
    handles = []

    def make_hook(layer_name: str):
        def hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            activations[layer_name].append(tensor.detach())
        return hook

    for name in layer_names:
        handles.append(_get_module_by_name(model, name).register_forward_hook(make_hook(name)))
    activations["_handles"] = handles  # type: ignore[assignment]
    return activations


def clear_activations(activations: Dict[str, List[torch.Tensor]]) -> None:
    for key, value in activations.items():
        if key != "_handles":
            value.clear()


def remove_activation_hooks(activations: Dict[str, List[torch.Tensor]]) -> None:
    for handle in activations.get("_handles", []):  # type: ignore[union-attr]
        handle.remove()
