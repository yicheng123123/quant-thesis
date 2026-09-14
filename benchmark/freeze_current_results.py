"""Import the existing validated experiment outputs into benchmark v1 JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.config import (
    BENCHMARK_ROOT,
    PROJECT_ROOT,
    directory_size_mib,
    load_config,
    utc_now,
    write_result,
)


LEGACY_ROOT = PROJECT_ROOT / "practice" / "full_w4a16" / "results"
AWQ_RESULTS_ROOT = PROJECT_ROOT / "practice" / "awq" / "results"
CONFIG_ROOT = BENCHMARK_ROOT / "configs"


def read_json(path: Path) -> dict[str, Any]:
    """Read one required legacy JSON object."""
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def quality_section(result: dict[str, Any], baseline_ppl: float) -> dict[str, Any]:
    """Map a legacy full-PPL result to the v1 quality section."""
    ppl = float(result["ppl"])
    delta = ppl - baseline_ppl
    return {
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
        "scope": "full",
        "num_input_tokens": int(result.get("num_input_tokens", 299078)),
        "evaluated_tokens": int(result["evaluated_tokens"]),
        "max_length": int(result["max_length"]),
        "stride": int(result["stride"]),
        "avg_nll": float(result["avg_nll"]),
        "ppl": ppl,
        "ppl_delta": delta,
        "ppl_relative_increase_percent": delta / baseline_ppl * 100.0,
    }


def freeze_fp16(baseline_ppl: float, *, force: bool) -> Path:
    """Freeze the existing FP16 quality and single-run system baseline."""
    config = load_config(CONFIG_ROOT / "fp16.yaml", check_paths=True)
    if config.raw_result_path.exists() and not force:
        raise FileExistsError(config.raw_result_path)
    quality_source = read_json(LEGACY_ROOT / "fp16_full_wikitext2_ppl.json")
    system_source = read_json(LEGACY_ROOT / "fp16_baseline.json")
    quality = quality_section(quality_source, baseline_ppl)
    quality["ppl_delta"] = 0.0
    quality["ppl_relative_increase_percent"] = 0.0
    return write_result(
        config,
        {
            "quality": quality,
            "storage": {"checkpoint_mib": system_source["checkpoint_mib"]},
            "memory": {
                "measurement": "legacy_peak_only",
                "resident_vram_mib": None,
                "peak_vram_mib": system_source["peak_vram_mib"],
            },
            "performance": {
                "warmup_runs": 1,
                "benchmark_runs": 1,
                "max_new_tokens": system_source["generated_tokens"],
                "tokens_per_second_mean": system_source["tokens_per_second"],
                "tokens_per_second_std": None,
                "aggregation": "legacy_single_measurement",
            },
            "provenance": {
                "status": "legacy_frozen",
                "frozen_at_utc": utc_now(),
                "source_files": [
                    str(LEGACY_ROOT / "fp16_full_wikitext2_ppl.json"),
                    str(LEGACY_ROOT / "fp16_baseline.json"),
                ],
            },
        },
        replace=True,
    )


def freeze_gptq(group_size: int, baseline_ppl: float, *, force: bool) -> Path:
    """Freeze one GPTQ quality result and its isolated legacy system result."""
    config = load_config(CONFIG_ROOT / f"gptq_g{group_size}.yaml", check_paths=True)
    if config.raw_result_path.exists() and not force:
        raise FileExistsError(config.raw_result_path)
    quality_path = LEGACY_ROOT / f"group_size_tradeoff_g{group_size}.json"
    system_path = LEGACY_ROOT / f"group_size_vram_isolated_g{group_size}.json"
    quality_source = read_json(quality_path)
    system_source = read_json(system_path)
    return write_result(
        config,
        {
            "quality": quality_section(quality_source, baseline_ppl),
            "storage": {"checkpoint_mib": quality_source["checkpoint_mib"]},
            "memory": {
                "measurement": "legacy_torch_cuda_memory_allocated",
                "resident_vram_mib": system_source["resident_after_load_mib"],
                "resident_after_warmup_mib": system_source[
                    "resident_after_warmup_mib"
                ],
                "generation_baseline_mib": system_source[
                    "generation_baseline_mib"
                ],
                "peak_vram_mib": system_source["peak_generation_mib"],
                "generation_extra_mib": system_source["generation_extra_mib"],
            },
            "performance": {
                "warmup_runs": system_source["warmup_runs"],
                "benchmark_runs": system_source["measure_runs"],
                "max_new_tokens": system_source["max_new_tokens"],
                "tokens_per_second_mean": system_source["tokens_per_second"],
                "tokens_per_second_std": None,
                "latency_seconds_median": system_source["latency_seconds"],
                "aggregation": "legacy_median_latency_derived",
            },
            "quantization": {
                "quantization_seconds": quality_source["quantization_seconds"],
                "save_seconds": quality_source["save_seconds"],
                "total_seconds": quality_source["total_quantization_seconds"],
            },
            "provenance": {
                "status": "legacy_frozen",
                "frozen_at_utc": utc_now(),
                "source_files": [str(quality_path), str(system_path)],
            },
        },
        replace=True,
    )


def freeze_awq(baseline_ppl: float, *, force: bool) -> Path:
    """Freeze AWQ quality/storage while leaving system metrics unavailable."""
    config = load_config(CONFIG_ROOT / "awq_g64.yaml", check_paths=True)
    if config.raw_result_path.exists() and not force:
        raise FileExistsError(config.raw_result_path)
    quality_path = AWQ_RESULTS_ROOT / "awq_sym_g64_full_ppl.json"
    metadata_path = config.checkpoint / "awq_metadata.json"  # type: ignore[operator]
    quality_source = read_json(quality_path)
    metadata = read_json(metadata_path)
    return write_result(
        config,
        {
            "quality": quality_section(quality_source, baseline_ppl),
            "storage": {"checkpoint_mib": directory_size_mib(config.checkpoint)},
            "memory": None,
            "performance": None,
            "quantization": {
                "quantization_seconds": metadata["elapsed_seconds"],
                "timing_scope": "llmcompressor_oneshot",
            },
            "provenance": {
                "status": "legacy_frozen",
                "frozen_at_utc": utc_now(),
                "source_files": [str(quality_path), str(metadata_path)],
            },
        },
        replace=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze current results into v1 JSON")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing unified raw results.",
    )
    args = parser.parse_args()
    baseline = read_json(LEGACY_ROOT / "fp16_full_wikitext2_ppl.json")
    baseline_ppl = float(baseline["ppl"])
    outputs = [
        freeze_fp16(baseline_ppl, force=args.force),
        freeze_gptq(32, baseline_ppl, force=args.force),
        freeze_gptq(64, baseline_ppl, force=args.force),
        freeze_gptq(128, baseline_ppl, force=args.force),
        freeze_awq(baseline_ppl, force=args.force),
    ]
    for output in outputs:
        print(f"frozen={output}")


if __name__ == "__main__":
    main()

