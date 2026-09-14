"""Method-aware model loading for existing FP16, GPTQ, and AWQ artifacts."""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.config import ExperimentConfig


def configure_windows_eager_gptq() -> None:
    """Apply the validated eager GPTQModel patch when Triton is unavailable."""
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


def load_model_and_tokenizer(config: ExperimentConfig) -> tuple[Any, Any]:
    """Load exactly one configured model and its matching tokenizer."""
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark workers require CUDA.")

    if config.method == "fp16":
        source: str | Any = config.model_id
        model = AutoModelForCausalLM.from_pretrained(
            source,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        checkpoint = config.checkpoint
        if checkpoint is None or not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        source = checkpoint
        if config.method == "gptq":
            configure_windows_eager_gptq()
            model = AutoModelForCausalLM.from_pretrained(
                source,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        elif config.method == "awq":
            model = AutoModelForCausalLM.from_pretrained(
                source,
                dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            raise ValueError(f"Unsupported method: {config.method}")

    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

