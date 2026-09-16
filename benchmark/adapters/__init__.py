"""Plugin-style model loading adapters for quantized benchmark checkpoints."""

from benchmark.adapters.base import AdapterCapabilities, LoadedModel, QuantizationAdapter
from benchmark.adapters.registry import available_adapters, get_adapter, register_adapter

__all__ = [
    "AdapterCapabilities",
    "LoadedModel",
    "QuantizationAdapter",
    "available_adapters",
    "get_adapter",
    "register_adapter",
]
