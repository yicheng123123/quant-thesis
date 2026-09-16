"""Lightweight regression checks for the v1.2.2 adapter architecture."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.adapters.awq import AWQAdapter
from benchmark.adapters.fp16 import FP16Adapter
from benchmark.adapters.gptq import GPTQAdapter
from benchmark.config import BENCHMARK_ROOT, load_config, manifest_context
from benchmark.figures import label as figure_label
from benchmark.summarize import experiment_label


PROJECT_ROOT = BENCHMARK_ROOT.parent
CONFIG_ROOT = BENCHMARK_ROOT / "configs"


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"


class _FakeModel:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True


class _AutoFactory:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[tuple[object, dict[str, object]]] = []

    def from_pretrained(self, source: object, **kwargs: object) -> object:
        self.calls.append((source, kwargs))
        return self.value


def _generic_payload(checkpoint: Path) -> dict[str, object]:
    payload = copy.deepcopy(load_config(CONFIG_ROOT / "gptq_g64.yaml").data)
    payload.update(
        {
            "experiment_id": "hqq_generic_test",
            "method": "hqq",
            "bits": 3,
            "group_size": 128,
            "symmetric": False,
            "checkpoint": str(checkpoint),
            "calibration": None,
        }
    )
    payload["experiment"].update(
        {"id": "hqq_generic_test", "checkpoint": str(checkpoint), "scope": "smoke"}
    )
    payload["model"]["checkpoint"] = str(checkpoint)
    payload["quantization"].update(
        {
            "method": "hqq",
            "weight_bits": 3,
            "group_size": 128,
            "symmetric": False,
            "backend": "transformers_hqq",
            "duo_scaling": None,
        }
    )
    payload["loader"] = {"adapter": "hf_auto", "backend": "transformers_hqq"}
    payload["runtime"].update(
        {
            "python": sys.executable,
            "backend": "transformers_hqq",
            "packed_execution": True,
            "system_metrics_comparable": False,
            "comparability_note": "Synthetic config-only adapter regression.",
        }
    )
    return payload


def check_generic_hf_config_and_dry_run() -> None:
    """An arbitrary algorithm identity can use hf_auto without Torch imports."""
    with tempfile.TemporaryDirectory(prefix="benchmark-v122-") as directory:
        root = Path(directory)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        config_path = root / "hqq.json"
        config_path.write_text(json.dumps(_generic_payload(checkpoint)), encoding="utf-8")

        probe = (
            "import runpy, sys; assert 'torch' not in sys.modules; "
            f"sys.argv=['benchmark.runners.eval_quality','--config',{str(config_path)!r},'--dry-run']; "
            "runpy.run_module('benchmark.runners.eval_quality', run_name='__main__'); "
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
            raise AssertionError(f"generic hf_auto dry-run failed:\n{completed.stdout}\n{completed.stderr}")
        for expected in ("Quantization  : HQQ", "Loader        : hf_auto", "Checkpoint ID : NOT_DETECTABLE"):
            if expected not in completed.stdout:
                raise AssertionError(f"generic dry-run omitted {expected!r}")

        config = load_config(config_path, check_paths=True)
        manifest = manifest_context(config)
        if config.method != "hqq" or config.loader_adapter != "hf_auto":
            raise AssertionError("quantization.method and loader.adapter were coupled")
        if manifest["checkpoint_metadata"]["identity_validation"] != "NOT_DETECTABLE":
            raise AssertionError("missing checkpoint identity was treated as verified")


def check_detected_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="benchmark-v122-mismatch-") as directory:
        root = Path(directory)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(
            json.dumps(
                {
                    "quantization_config": {
                        "quant_method": "gptq",
                        "bits": 3,
                        "group_size": 128,
                        "sym": False,
                    }
                }
            ),
            encoding="utf-8",
        )
        config_path = root / "mismatch.json"
        config_path.write_text(json.dumps(_generic_payload(checkpoint)), encoding="utf-8")
        try:
            load_config(config_path, check_paths=True)
        except ValueError as exc:
            if "MISMATCH" not in str(exc) or "method" not in str(exc):
                raise AssertionError(f"mismatch error is not explicit: {exc}")
        else:
            raise AssertionError("detected checkpoint contradiction did not fail")


def check_unknown_adapter_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="benchmark-v122-unknown-") as directory:
        root = Path(directory)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        payload = _generic_payload(checkpoint)
        payload["loader"]["adapter"] = "does_not_exist"
        config_path = root / "unknown.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_config(config_path, check_paths=True)
        except ValueError as exc:
            if "Unknown loader adapter" not in str(exc):
                raise AssertionError(f"unknown adapter error is unclear: {exc}")
        else:
            raise AssertionError("unknown adapter was accepted")


def check_existing_loader_arguments() -> None:
    """Refactoring must preserve the exact validated from_pretrained arguments."""
    fake_torch = SimpleNamespace(float16=object())
    cases = (
        (
            FP16Adapter(),
            SimpleNamespace(model_id="base/model"),
            "base/model",
            {"dtype": fake_torch.float16, "device_map": "auto", "low_cpu_mem_usage": True},
        ),
        (
            GPTQAdapter(),
            SimpleNamespace(require_checkpoint=lambda: Path("checkpoint")),
            Path("checkpoint"),
            {"device_map": "auto", "low_cpu_mem_usage": True},
        ),
        (
            AWQAdapter(),
            SimpleNamespace(require_checkpoint=lambda: Path("checkpoint")),
            Path("checkpoint"),
            {"dtype": fake_torch.float16, "device_map": "auto", "low_cpu_mem_usage": True},
        ),
    )
    for adapter, config, source, expected_kwargs in cases:
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        model_factory = _AutoFactory(model)
        tokenizer_factory = _AutoFactory(tokenizer)
        fake_transformers = SimpleNamespace(
            AutoModelForCausalLM=model_factory,
            AutoTokenizer=tokenizer_factory,
        )
        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            if isinstance(adapter, GPTQAdapter):
                with patch("benchmark.adapters.gptq._configure_windows_eager"):
                    loaded = adapter.load(config)
            else:
                loaded = adapter.load(config)
        if model_factory.calls != [(source, expected_kwargs)]:
            raise AssertionError(f"{adapter.name} model args changed: {model_factory.calls}")
        if tokenizer_factory.calls != [(source, {})]:
            raise AssertionError(f"{adapter.name} tokenizer args changed: {tokenizer_factory.calls}")
        if loaded.model is not model or loaded.tokenizer is not tokenizer or not model.eval_called:
            raise AssertionError(f"{adapter.name} did not return a finalized LoadedModel")


def check_algorithm_agnostic_evaluators() -> None:
    paths = [BENCHMARK_ROOT / "eval" / "model.py", BENCHMARK_ROOT / "eval" / "downstream.py"]
    paths.extend((BENCHMARK_ROOT / "runners").glob("eval_*.py"))
    for path in paths:
        source = path.read_text(encoding="utf-8")
        if "config.method" in source or "load_model_and_tokenizer" in source:
            raise AssertionError(f"algorithm-specific dispatch remains in {path}")


def check_generic_reporting() -> None:
    row = {"method": "HQQ", "group_size": 64, "experiment_id": "hqq_g64"}
    if experiment_label(row) != "HQQ G64" or figure_label(row) != "HQQ G64":
        raise AssertionError("generic methods require a reporting-specific code branch")


def check_runtime_python_is_not_adapter_dispatch() -> None:
    config = load_config(CONFIG_ROOT / "gptq_g64.yaml", check_paths=True)
    if Path(sys.executable).resolve() != config.python_executable.resolve():
        try:
            config.validate_active_python()
        except RuntimeError as exc:
            if "Active Python does not match" not in str(exc):
                raise
        else:
            raise AssertionError("runtime.python mismatch was silently dispatched")
    for path in (BENCHMARK_ROOT / "runners").glob("eval_*.py"):
        source = path.read_text(encoding="utf-8")
        if "command = [\n        str(config.python_executable)" in source:
            raise AssertionError(f"runner still dispatches via runtime.python: {path}")


def check_frozen_raw_results_unchanged() -> None:
    completed = subprocess.run(
        ["git", "diff", "--quiet", "--", "benchmark/results/raw"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError("tracked v1.2 raw results changed")


def main() -> None:
    check_generic_hf_config_and_dry_run()
    check_detected_mismatch_fails()
    check_unknown_adapter_fails()
    check_existing_loader_arguments()
    check_algorithm_agnostic_evaluators()
    check_generic_reporting()
    check_runtime_python_is_not_adapter_dispatch()
    check_frozen_raw_results_unchanged()
    print(
        "PASS: v1.2.2 generic adapter, identity states, loader arguments, "
        "runtime isolation, evaluator neutrality, and frozen raw results"
    )


if __name__ == "__main__":
    main()
