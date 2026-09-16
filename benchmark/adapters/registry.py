"""Lazy adapter registry; importing it never imports Torch or Transformers."""

from __future__ import annotations

import importlib
from typing import TypeVar

from benchmark.adapters.base import QuantizationAdapter


AdapterType = TypeVar("AdapterType", bound=type[QuantizationAdapter])
_ADAPTERS: dict[str, type[QuantizationAdapter]] = {}
_BUILTIN_MODULES = {
    "fp16": "benchmark.adapters.fp16",
    "gptq": "benchmark.adapters.gptq",
    "awq": "benchmark.adapters.awq",
    "hf_auto": "benchmark.adapters.hf_auto",
}


def register_adapter(name: str):
    """Register an adapter class under a stable config-facing name."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Adapter name must be non-empty.")

    def decorator(adapter_cls: AdapterType) -> AdapterType:
        if not issubclass(adapter_cls, QuantizationAdapter):
            raise TypeError("Registered adapters must extend QuantizationAdapter.")
        existing = _ADAPTERS.get(normalized)
        if existing is not None and existing is not adapter_cls:
            raise ValueError(f"Adapter {normalized!r} is already registered.")
        adapter_cls.name = normalized
        _ADAPTERS[normalized] = adapter_cls
        return adapter_cls

    return decorator


def get_adapter(name: str, *, module: str | None = None) -> type[QuantizationAdapter]:
    """Resolve an adapter, optionally importing one plugin module first."""
    normalized = name.strip().lower()
    module_name = module or _BUILTIN_MODULES.get(normalized)
    if normalized not in _ADAPTERS and module_name:
        importlib.import_module(module_name)
    try:
        return _ADAPTERS[normalized]
    except KeyError as exc:
        available = ", ".join(available_adapters()) or "<none>"
        raise ValueError(
            f"Unknown loader adapter {name!r}. Registered adapters: {available}. "
            "Set loader.module for an external adapter plugin."
        ) from exc


def available_adapters() -> tuple[str, ...]:
    return tuple(sorted(set(_ADAPTERS) | set(_BUILTIN_MODULES)))
