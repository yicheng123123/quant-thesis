"""FP16 Transformers loader."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark.adapters.base import AdapterCapabilities, LoadedModel, QuantizationAdapter
from benchmark.adapters.registry import register_adapter

if TYPE_CHECKING:
    from benchmark.config import ExperimentConfig


@register_adapter("fp16")
class FP16Adapter(QuantizationAdapter):
    capabilities = AdapterCapabilities(checkpoint_metadata=False)

    @classmethod
    def supports_method(cls, method: str) -> bool:
        return method.lower() == "fp16"

    @classmethod
    def inspect_checkpoint(cls, checkpoint: Path | None) -> dict[str, Any]:
        del checkpoint
        return {"status": "not_detectable", "files": []}

    def load(self, config: "ExperimentConfig") -> LoadedModel:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        model, tokenizer = self.finalize(model, tokenizer)
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            detected_metadata=self.inspect_checkpoint(None),
            capabilities=self.capabilities,
        )
