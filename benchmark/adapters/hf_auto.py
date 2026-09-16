"""Generic Hugging Face AutoModel adapter for arbitrary algorithm identities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark.adapters.base import LoadedModel, QuantizationAdapter
from benchmark.adapters.metadata import inspect_hf_checkpoint
from benchmark.adapters.registry import register_adapter

if TYPE_CHECKING:
    from benchmark.config import ExperimentConfig


def _model_kwargs(config: "ExperimentConfig") -> dict[str, Any]:
    import torch

    kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        **config.loader_model_kwargs,
    }
    dtype = kwargs.get("dtype")
    if isinstance(dtype, str):
        aliases = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "auto": "auto",
        }
        try:
            kwargs["dtype"] = aliases[dtype.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported loader.model_kwargs.dtype: {dtype!r}") from exc
    return kwargs


@register_adapter("hf_auto")
class HFAutoAdapter(QuantizationAdapter):
    @classmethod
    def inspect_checkpoint(cls, checkpoint: Path | None) -> dict[str, Any]:
        return inspect_hf_checkpoint(checkpoint)

    def load(self, config: "ExperimentConfig") -> LoadedModel:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint = config.require_checkpoint()
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **_model_kwargs(config))
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            **config.loader_tokenizer_kwargs,
        )
        model, tokenizer = self.finalize(model, tokenizer)
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            detected_metadata=self.inspect_checkpoint(checkpoint),
            capabilities=self.capabilities,
        )
