"""Deterministic repeated-generation performance measurements."""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch


def prepare_generation(
    model,
    tokenizer,
    generation_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], int]:
    """Tokenize the shared prompt and construct deterministic generation args."""
    device = next(model.parameters()).device
    encoded = tokenizer(generation_config["prompt"], return_tensors="pt")
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    input_tokens = int(encoded["input_ids"].shape[-1])
    kwargs = {
        "max_new_tokens": int(generation_config["max_new_tokens"]),
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    return encoded, kwargs, input_tokens


@torch.inference_mode()
def run_warmups(
    model,
    encoded: dict[str, torch.Tensor],
    kwargs: dict[str, Any],
    warmup_runs: int,
) -> None:
    """Run unmeasured generation passes before resetting peak memory."""
    for _ in range(warmup_runs):
        model.generate(**encoded, **kwargs)
    torch.cuda.synchronize()


@torch.inference_mode()
def run_measurements(
    model,
    tokenizer,
    encoded: dict[str, torch.Tensor],
    kwargs: dict[str, Any],
    input_tokens: int,
    benchmark_runs: int,
) -> dict[str, Any]:
    """Measure per-run latency and throughput, then report mean and std."""
    latencies: list[float] = []
    throughputs: list[float] = []
    generated_counts: list[int] = []
    output_ids = None

    for run_index in range(benchmark_runs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output_ids = model.generate(**encoded, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = int(output_ids.shape[-1] - input_tokens)
        if generated <= 0:
            raise RuntimeError("Generation produced no new tokens.")
        latencies.append(elapsed)
        generated_counts.append(generated)
        throughputs.append(generated / elapsed)
        print(
            f"benchmark_run={run_index + 1}/{benchmark_runs} "
            f"tokens={generated} latency={elapsed:.6f}s "
            f"throughput={throughputs[-1]:.4f}tok/s",
            flush=True,
        )

    assert output_ids is not None
    return {
        "warmup_runs": None,
        "benchmark_runs": benchmark_runs,
        "max_new_tokens": kwargs["max_new_tokens"],
        "generated_tokens_per_run": generated_counts,
        "latency_seconds_runs": latencies,
        "latency_seconds_mean": statistics.mean(latencies),
        "latency_seconds_std": statistics.stdev(latencies),
        "tokens_per_second_runs": throughputs,
        "tokens_per_second_mean": statistics.mean(throughputs),
        "tokens_per_second_std": statistics.stdev(throughputs),
        "output_text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
    }

