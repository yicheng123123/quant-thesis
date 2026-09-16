"""Current compressed-tensors AWQ checkpoint loader."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark.adapters.base import LoadedModel, QuantizationAdapter
from benchmark.adapters.metadata import inspect_hf_checkpoint
from benchmark.adapters.registry import register_adapter

if TYPE_CHECKING:
    from benchmark.config import ExperimentConfig


@register_adapter("awq")
class AWQAdapter(QuantizationAdapter):
    @classmethod
    def supports_method(cls, method: str) -> bool:
        return method.lower() == "awq"

    @classmethod
    def inspect_checkpoint(cls, checkpoint: Path | None) -> dict[str, Any]:
        return inspect_hf_checkpoint(checkpoint)

    def load(self, config: "ExperimentConfig") -> LoadedModel:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint = config.require_checkpoint()
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch.float16,
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
