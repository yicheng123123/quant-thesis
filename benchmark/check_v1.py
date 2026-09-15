"""Fast offline checks for config schema, PPL coverage, and v1.2 scope."""

from __future__ import annotations

import math
from pathlib import Path

from benchmark.config import BENCHMARK_ROOT, load_config


def select_scope_length(sequence_length: int, scope: str, smoke_tokens: int) -> int:
    """Return the token count selected by the shared PPL scope contract."""
    if sequence_length < 2:
        raise ValueError("At least two input tokens are required.")
    if scope == "full":
        return sequence_length
    if scope == "smoke":
        if sequence_length < smoke_tokens:
            raise ValueError("The corpus is shorter than the requested smoke prefix.")
        return smoke_tokens
    raise ValueError(f"Unsupported scope: {scope}")


def sliding_window_target_counts(
    sequence_length: int,
    max_length: int,
    stride: int,
) -> list[int]:
    """Model the evaluator's non-overlapping target-token accounting offline."""
    if not 0 < stride <= max_length:
        raise ValueError("PPL requires 0 < stride <= max_length.")
    counts: list[int] = []
    previous_end = 0
    for begin in range(0, sequence_length, stride):
        end = min(begin + max_length, sequence_length)
        target_length = end - previous_end
        counts.append(target_length - 1 if begin == 0 else target_length)
        previous_end = end
        if end == sequence_length:
            break
    return counts


def main() -> None:
    config_paths = sorted((BENCHMARK_ROOT / "configs").glob("*.yaml"))
    if len(config_paths) != 7:
        raise AssertionError(f"Expected seven configs, found {len(config_paths)}")
    configs = [load_config(path, check_paths=True) for path in config_paths]
    if {config.experiment_id for config in configs} != {
        "fp16",
        "gptq_g32",
        "gptq_g64",
        "gptq_g128",
        "awq_g64",
        "awq_g64_duo_false",
        "awq_g64_duo_true",
    }:
        raise AssertionError("The config experiment IDs are incomplete.")

    downstream_ids = {
        "fp16",
        "gptq_g64",
        "awq_g64_duo_false",
        "awq_g64_duo_true",
    }
    for config in configs:
        if config.experiment_id in downstream_ids:
            downstream = config.data.get("downstream")
            if not isinstance(downstream, dict) or downstream.get("enabled") is not True:
                raise AssertionError(f"Downstream config missing for {config.experiment_id}.")
            if downstream.get("harness") != "lm_eval":
                raise AssertionError(f"Unexpected downstream harness for {config.experiment_id}.")

    full_length = select_scope_length(10, "full", 6)
    smoke_length = select_scope_length(10, "smoke", 6)
    if full_length != 10 or smoke_length != 6:
        raise AssertionError("PPL scope selection failed.")

    target_counts = sliding_window_target_counts(10, max_length=6, stride=3)
    tokens = sum(target_counts)
    average_nll = sum(2.0 * count for count in target_counts) / tokens
    ppl = math.exp(average_nll)
    if tokens != 9 or abs(average_nll - 2.0) > 1e-9 or abs(ppl - math.exp(2.0)) > 1e-6:
        raise AssertionError("Sliding-window token weighting failed.")
    print(
        "PASS: seven configs, local paths, downstream scope, and PPL token weighting "
        "(stdlib-only offline check)"
    )


if __name__ == "__main__":
    main()
