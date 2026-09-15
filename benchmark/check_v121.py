"""Lightweight regression checks for the v1.2.1 transparency patch."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from benchmark.config import (
    BENCHMARK_ROOT,
    RAW_RESULTS_DIR,
    base_result,
    inspect_checkpoint_metadata,
    load_config,
    manifest_context,
    validate_checkpoint_identity,
)
from benchmark.summarize import build_row, experiment_label


PROJECT_ROOT = BENCHMARK_ROOT.parent
CONFIG_ROOT = BENCHMARK_ROOT / "configs"


def check_dry_runs() -> None:
    """Each dry-run entry point must avoid importing Torch and model code."""
    commands = [
        [
            "benchmark.runners.eval_quality",
            "--config",
            str(CONFIG_ROOT / "fp16.yaml"),
            "--dry-run",
        ],
        [
            "benchmark.runners.eval_system",
            "--config",
            str(CONFIG_ROOT / "gptq_g64.yaml"),
            "--dry-run",
        ],
        [
            "benchmark.runners.eval_downstream",
            "--config",
            str(CONFIG_ROOT / "gptq_g64.yaml"),
            "--task",
            "hellaswag",
            "--scope",
            "smoke",
            "--dry-run",
        ],
    ]
    for module, *args in commands:
        probe = (
            "import runpy, sys; "
            "assert 'torch' not in sys.modules; "
            f"sys.argv = ['{module}', "
            + ", ".join(repr(value) for value in args)
            + "]; "
            f"runpy.run_module('{module}', run_name='__main__'); "
            "assert 'torch' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"dry-run failed for {module}:\n{completed.stdout}\n{completed.stderr}"
            )
        if "DRY RUN:" not in completed.stdout or "BENCHMARK EXPERIMENT" not in completed.stdout:
            raise AssertionError(f"dry-run manifest missing for {module}")


def check_manifest_fields() -> None:
    expected_ids = {
        "fp16",
        "gptq_g32",
        "gptq_g64",
        "gptq_g128",
        "awq_g64",
        "awq_g64_duo_false",
        "awq_g64_duo_true",
    }
    configs = [load_config(path, check_paths=True) for path in sorted(CONFIG_ROOT.glob("*.yaml"))]
    if {config.experiment_id for config in configs} != expected_ids:
        raise AssertionError("unexpected benchmark config IDs")
    for config in configs:
        metadata = config.experiment_metadata
        manifest = manifest_context(config)
        for field in (
            "experiment_id",
            "base_model",
            "checkpoint",
            "method",
            "weight_bits",
            "activation_bits",
            "activation_precision",
            "precision",
            "group_size",
            "symmetric",
            "backend",
            "duo_scaling",
        ):
            if manifest[field] != metadata[field]:
                raise AssertionError(f"manifest/config mismatch for {config.experiment_id}: {field}")
        if manifest["output_paths"]["result"] != str(config.raw_result_path):
            raise AssertionError(f"result path mismatch for {config.experiment_id}")


def check_raw_identity_and_numbers() -> None:
    """New result construction has identity metadata; frozen values stay unchanged."""
    expected_ppl = {
        "fp16": 12.463618645962649,
        "gptq_g32": 13.306268241552718,
        "gptq_g64": 13.492837547663278,
        "gptq_g128": 13.846270150694822,
        "awq_g64": 14.489811246370849,
        "awq_g64_duo_false": 14.489811246370849,
        "awq_g64_duo_true": 14.549754088368804,
    }
    for experiment_id, frozen_ppl in expected_ppl.items():
        path = RAW_RESULTS_DIR / f"{experiment_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual_ppl = payload.get("quality", {}).get("ppl")
        if actual_ppl is not None and abs(float(actual_ppl) - frozen_ppl) > 1e-12:
            raise AssertionError(f"frozen PPL changed for {experiment_id}")

        config = load_config(CONFIG_ROOT / f"{experiment_id}.yaml", check_paths=True)
        metadata = manifest_context(config)
        if not metadata.get("identity") or not metadata.get("quantization"):
            raise AssertionError(f"identity metadata is incomplete for {experiment_id}")
        synthetic_result = base_result(config)
        if synthetic_result["experiment_metadata"]["experiment_id"] != experiment_id:
            raise AssertionError(f"raw identity metadata mismatch for {experiment_id}")


def check_checkpoint_mismatch() -> None:
    """A detectable method mismatch must fail before model execution."""
    with tempfile.TemporaryDirectory(prefix="benchmark-v121-") as directory:
        checkpoint = Path(directory)
        (checkpoint / "awq_metadata.json").write_text(
            json.dumps({"method": "AWQ", "bits": 4, "group_size": 64, "symmetric": True}),
            encoding="utf-8",
        )
        if inspect_checkpoint_metadata(checkpoint).get("method") != "AWQ":
            raise AssertionError("AWQ metadata was not detected")
        fake = SimpleNamespace(
            experiment_id="mismatch",
            checkpoint=checkpoint,
            experiment_metadata={
                "method": "GPTQ",
                "weight_bits": 4,
                "group_size": 64,
                "symmetric": True,
                "duo_scaling": False,
            },
        )
        try:
            validate_checkpoint_identity(fake)
        except ValueError as exc:
            if "method" not in str(exc):
                raise AssertionError(f"mismatch error lacks field name: {exc}")
        else:
            raise AssertionError("checkpoint/config mismatch did not fail")


def check_summary_labels() -> None:
    labels = []
    for experiment_id in ("fp16", "gptq_g64", "awq_g64", "awq_g64_duo_false", "awq_g64_duo_true"):
        payload = json.loads((RAW_RESULTS_DIR / f"{experiment_id}.json").read_text(encoding="utf-8"))
        labels.append(experiment_label(build_row(payload)))
    if labels[0] != "FP16" or labels[1] != "GPTQ G64":
        raise AssertionError(f"unexpected FP16/GPTQ labels: {labels[:2]}")
    if "AWQ G64 duo=False" not in labels[2] or "AWQ G64 duo=False" not in labels[3]:
        raise AssertionError(f"AWQ false labels are ambiguous: {labels[2:4]}")
    if labels[4] != "AWQ G64 duo=True":
        raise AssertionError(f"AWQ duo=True label is ambiguous: {labels[4]}")
    if labels[2] == labels[3]:
        raise AssertionError("legacy AWQ and explicit AWQ configs collide in the summary")


def main() -> None:
    check_dry_runs()
    check_manifest_fields()
    check_raw_identity_and_numbers()
    check_checkpoint_mismatch()
    check_summary_labels()
    print("PASS: v1.2.1 dry-run, manifest, provenance, checkpoint mismatch, numerical, and label checks")


if __name__ == "__main__":
    main()
