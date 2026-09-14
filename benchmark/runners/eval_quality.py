"""Run one PPL evaluation in a fresh config-selected Python process."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from benchmark.config import (
    PROJECT_ROOT,
    apply_runtime_environment,
    directory_size_mib,
    load_baseline_quality,
    load_config,
    utc_now,
    write_result,
)


def run_worker(config_path: Path) -> None:
    """Load one model, evaluate one token stream, write one unified result."""
    config = load_config(config_path, check_paths=True)
    apply_runtime_environment()

    import torch

    from benchmark.eval.model import load_model_and_tokenizer
    from benchmark.eval.ppl import (
        evaluate_ppl,
        prepare_wikitext_input_ids,
        select_scope,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Quality evaluation requires CUDA.")

    ppl_config = config.data["ppl"]
    input_ids = prepare_wikitext_input_ids(
        config.model_id,
        ppl_config,
        str(PROJECT_ROOT / ".hf_cache" / "datasets"),
    )
    eval_ids = select_scope(input_ids, ppl_config)
    expected_targets = eval_ids.size(1) - 1
    model, tokenizer = load_model_and_tokenizer(config)
    del tokenizer

    ppl, average_nll, evaluated_tokens = evaluate_ppl(
        model,
        eval_ids,
        max_length=int(ppl_config["max_length"]),
        stride=int(ppl_config["stride"]),
    )
    if evaluated_tokens != expected_targets:
        raise AssertionError(
            f"Token coverage mismatch: expected {expected_targets}, got {evaluated_tokens}."
        )

    baseline = load_baseline_quality(config)
    if config.method == "fp16":
        ppl_delta = 0.0
        relative_increase = 0.0
    elif baseline is None:
        ppl_delta = None
        relative_increase = None
    else:
        ppl_delta = ppl - float(baseline["ppl"])
        relative_increase = ppl_delta / float(baseline["ppl"]) * 100.0

    validation = config.data.get("validation", {})
    expected_ppl = validation.get("expected_ppl")
    tolerance = float(validation.get("ppl_tolerance", 0.0))
    absolute_difference = (
        abs(ppl - float(expected_ppl)) if expected_ppl is not None else None
    )
    within_tolerance = (
        absolute_difference <= tolerance if absolute_difference is not None else None
    )
    if within_tolerance is False:
        raise AssertionError(
            f"PPL drifted from the frozen result: expected={expected_ppl}, "
            f"actual={ppl}, tolerance={tolerance}."
        )

    had_legacy_result = config.raw_result_path.is_file()
    result_path = write_result(
        config,
        {
            "quality": {
                "dataset": (
                    f"{ppl_config['dataset_id']}:"
                    f"{ppl_config['dataset_config']}:"
                    f"{ppl_config['split']}"
                ),
                "scope": ppl_config["scope"],
                "num_input_tokens": int(eval_ids.size(1)),
                "evaluated_tokens": evaluated_tokens,
                "max_length": int(ppl_config["max_length"]),
                "stride": int(ppl_config["stride"]),
                "avg_nll": average_nll,
                "ppl": ppl,
                "ppl_delta": ppl_delta,
                "ppl_relative_increase_percent": relative_increase,
            },
            "storage": {
                "checkpoint_mib": directory_size_mib(config.checkpoint),
            },
            "quality_validation": {
                "reference": "legacy_frozen",
                "expected_ppl": expected_ppl,
                "absolute_difference": absolute_difference,
                "tolerance": tolerance,
                "within_tolerance": within_tolerance,
            },
            "provenance": {
                "status": "v1_rerun",
                "legacy_frozen": had_legacy_result,
                "quality": "benchmark_v1_worker",
                "quality_run_at_utc": utc_now(),
                "backend": config.data["runtime"]["backend"],
                "environment": str(config.python_executable),
                "checkpoint": str(config.checkpoint) if config.checkpoint else None,
                "input_tokens": int(eval_ids.size(1)),
                "max_new_tokens": int(config.data["generation"]["max_new_tokens"]),
                "warmup_runs": int(config.data["generation"]["warmup_runs"]),
                "measure_runs": int(config.data["generation"]["benchmark_runs"]),
                "comparable": True,
                "system_metrics_comparable": bool(
                    config.data["runtime"]["system_metrics_comparable"]
                ),
            },
        },
    )
    print(f"quality_result={result_path}", flush=True)
    print(f"ppl={ppl:.12f} avg_nll={average_nll:.12f}", flush=True)


def launch_isolated(config_path: Path) -> None:
    """Spawn the worker under the Python interpreter declared by the config."""
    config = load_config(config_path, check_paths=True)
    command = [
        str(config.python_executable),
        "-X",
        "utf8",
        "-m",
        "benchmark.runners.eval_quality",
        "--config",
        str(config.path),
        "--worker",
    ]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified isolated PPL runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args.config)
    else:
        launch_isolated(args.config)


if __name__ == "__main__":
    main()
