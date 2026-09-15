"""Run one VRAM and repeated-generation benchmark in a fresh process."""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
from pathlib import Path

from benchmark.config import (
    PROJECT_ROOT,
    apply_runtime_environment,
    directory_size_mib,
    load_config,
    print_manifest,
    utc_now,
    write_result,
)


def run_worker(config_path: Path) -> None:
    """Load exactly one model, benchmark it, persist results, and release CUDA."""
    config = load_config(config_path, check_paths=True)
    manifest = print_manifest(
        config,
        task="Generation",
        scope="full",
        dataset="configured generation prompt",
        split="n/a",
    )
    apply_runtime_environment()

    import torch

    from benchmark.eval.memory import benchmark_system
    from benchmark.eval.model import load_model_and_tokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("System evaluation requires CUDA.")

    model, tokenizer = load_model_and_tokenizer(config)
    memory, performance = benchmark_system(
        model,
        tokenizer,
        config.data["generation"],
    )
    comparable = bool(config.data["runtime"]["system_metrics_comparable"])
    had_legacy_result = config.raw_result_path.is_file()
    result_path = write_result(
        config,
        {
            "storage": {
                "checkpoint_mib": directory_size_mib(config.checkpoint),
            },
            "memory": memory,
            "performance": performance,
            "system_validation": {
                "isolated_process": True,
                "system_metrics_comparable": comparable,
                "backend": config.data["runtime"]["backend"],
                "environment": str(config.python_executable),
                "comparability_note": config.data["runtime"].get(
                    "comparability_note"
                ),
            },
            "provenance": {
                "status": "v1_rerun",
                "legacy_frozen": had_legacy_result,
                "system": "benchmark_v1_isolated_worker",
                "system_run_at_utc": utc_now(),
                "backend": config.data["runtime"]["backend"],
                "environment": str(config.python_executable),
                "checkpoint": str(config.checkpoint) if config.checkpoint else None,
                "input_tokens": performance["input_tokens"],
                "prompt": performance["prompt"],
                "max_new_tokens": performance["max_new_tokens"],
                "warmup_runs": performance["warmup_runs"],
                "measure_runs": performance["benchmark_runs"],
                "comparable": comparable,
                "system_metrics_comparable": comparable,
            },
        },
        manifest=manifest,
    )
    print(f"system_result={result_path}", flush=True)
    print(
        "throughput="
        f"{performance['tokens_per_second_mean']:.4f}±"
        f"{performance['tokens_per_second_std']:.4f} tok/s",
        flush=True,
    )
    print(
        f"resident={memory['resident_vram_mib']:.2f} MiB "
        f"peak={memory['peak_vram_mib']:.2f} MiB",
        flush=True,
    )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def launch_isolated(config_path: Path) -> None:
    """Spawn one worker and wait for complete process termination."""
    config = load_config(config_path, check_paths=True)
    print_manifest(
        config,
        task="Generation",
        scope="full",
        dataset="configured generation prompt",
        split="n/a",
    )
    command = [
        str(config.python_executable),
        "-X",
        "utf8",
        "-m",
        "benchmark.runners.eval_system",
        "--config",
        str(config.path),
        "--worker",
    ]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def dry_run(config_path: Path) -> None:
    """Validate and print the system manifest without starting a worker."""
    config = load_config(config_path, check_paths=True)
    print_manifest(
        config,
        task="Generation",
        scope="full",
        dataset="configured generation prompt",
        split="n/a",
    )
    print("DRY RUN: model, CUDA, and evaluation were not initialized.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified isolated system runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args.config)
        return
    if args.worker:
        run_worker(args.config)
    else:
        launch_isolated(args.config)


if __name__ == "__main__":
    main()
