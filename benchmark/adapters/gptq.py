"""GPTQModel-compatible Transformers loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark.adapters.base import LoadedModel, QuantizationAdapter
from benchmark.adapters.metadata import inspect_hf_checkpoint
from benchmark.adapters.registry import register_adapter

if TYPE_CHECKING:
    from benchmark.config import ExperimentConfig


def _configure_windows_eager() -> None:
    if os.name != "nt":
        return
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

    if getattr(TorchQuantLinear, "_quant_thesis_eager_patch", False):
        return

    def eager_optimize(self, backend="inductor", mode=None, fullgraph=False):
        del backend, mode, fullgraph
        self.optimized = True

    TorchQuantLinear.optimize = eager_optimize
    TorchQuantLinear._quant_thesis_eager_patch = True


@register_adapter("gptq")
class GPTQAdapter(QuantizationAdapter):
    @classmethod
    def supports_method(cls, method: str) -> bool:
        return method.lower() == "gptq"

    @classmethod
    def inspect_checkpoint(cls, checkpoint: Path | None) -> dict[str, Any]:
        return inspect_hf_checkpoint(checkpoint)

    def load(self, config: "ExperimentConfig") -> LoadedModel:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint = config.require_checkpoint()
        _configure_windows_eager()
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model, tokenizer = self.finalize(model, tokenizer)
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            detected_metadata=self.inspect_checkpoint(checkpoint),
            capabilities=self.capabilities,
        )
