"""Algorithm-neutral adapter dispatch for benchmark model loading."""

from __future__ import annotations

from benchmark.adapters.base import LoadedModel
from benchmark.adapters.registry import get_adapter
from benchmark.config import ExperimentConfig


def load_model(config: ExperimentConfig) -> LoadedModel:
    """Load one configured model through its explicitly selected adapter."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark workers require CUDA.")
    adapter_cls = get_adapter(config.loader_adapter, module=config.loader_module)
    return adapter_cls().load(config)
