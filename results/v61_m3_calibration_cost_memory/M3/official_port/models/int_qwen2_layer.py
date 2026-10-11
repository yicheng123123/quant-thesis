"""Qwen2 architecture adapter: preserve native decoder forward, replace seven Linear modules only."""

from __future__ import annotations

import copy

import torch.nn as nn

from quantize.int_linear import QuantLinear


TARGETS = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}


def _set_submodule(root: nn.Module, dotted_name: str, module: nn.Module) -> None:
    path = dotted_name.split(".")
    parent = root
    for part in path[:-1]:
        parent = getattr(parent, part)
    setattr(parent, path[-1], module)


class QuantQwen2DecoderLayer(nn.Module):
    """Composition adapter retaining the installed Transformers Qwen2 layer implementation."""

    def __init__(self, config, ori_layer: nn.Module, args):
        super().__init__()
        del config  # All architecture behavior stays on the copied native layer.
        self.native_layer = copy.deepcopy(ori_layer)
        found = set()
        for name, module in list(self.native_layer.named_modules()):
            if name in TARGETS:
                if not isinstance(module, nn.Linear):
                    raise TypeError(f"expected native nn.Linear at {name}, got {type(module).__name__}")
                _set_submodule(
                    self.native_layer,
                    name,
                    QuantLinear(module, args.weight_quant_params, args.act_quant_params),
                )
                found.add(name)
        if found != TARGETS:
            raise RuntimeError(f"Qwen2 target discovery mismatch: missing={sorted(TARGETS-found)}, extra={sorted(found-TARGETS)}")

    def forward(self, *args, **kwargs):
        return self.native_layer(*args, **kwargs)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for module in self.modules():
            if isinstance(module, QuantLinear):
                module.set_quant_state(weight_quant, act_quant)
