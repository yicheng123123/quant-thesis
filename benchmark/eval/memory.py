"""CUDA-resident and peak-memory measurement around repeated generation."""

from __future__ import annotations

from typing import Any

import torch

from benchmark.eval.generation import (
    prepare_generation,
    run_measurements,
    run_warmups,
)


def mib(value: int) -> float:
    """Convert bytes to mebibytes."""
    return value / 1024**2


def benchmark_system(
    model,
    tokenizer,
    generation_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure resident/peak VRAM and repeated deterministic generation speed."""
    encoded, kwargs, input_tokens = prepare_generation(
        model,
        tokenizer,
        generation_config,
    )
    warmup_runs = int(generation_config["warmup_runs"])
    benchmark_runs = int(generation_config["benchmark_runs"])

    torch.cuda.synchronize()
    resident_after_load = torch.cuda.memory_allocated()
    reserved_after_load = torch.cuda.memory_reserved()

    run_warmups(model, encoded, kwargs, warmup_runs)
    resident_after_warmup = torch.cuda.memory_allocated()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    generation_baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    performance = run_measurements(
        model,
        tokenizer,
        encoded,
        kwargs,
        input_tokens,
        benchmark_runs,
    )
    performance["warmup_runs"] = warmup_runs
    performance["input_tokens"] = input_tokens
    performance["prompt"] = generation_config["prompt"]
    performance["max_new_tokens"] = int(generation_config["max_new_tokens"])
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    memory = {
        "measurement": "torch.cuda memory_allocated/memory_reserved",
        "resident_vram_mib": mib(resident_after_load),
        "reserved_after_load_mib": mib(reserved_after_load),
        "resident_after_warmup_mib": mib(resident_after_warmup),
        "generation_baseline_mib": mib(generation_baseline),
        "peak_vram_mib": mib(peak_allocated),
        "peak_reserved_vram_mib": mib(peak_reserved),
        "generation_extra_mib": mib(peak_allocated - generation_baseline),
    }
    return memory, performance
