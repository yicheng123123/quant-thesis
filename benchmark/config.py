"""Configuration schema, path handling, and unified result persistence."""

from __future__ import annotations

import json
import importlib.metadata as package_metadata
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - used by the stdlib-only freeze tools
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark"
RAW_RESULTS_DIR = BENCHMARK_ROOT / "results" / "raw"
RAW_RUNS_DIR = RAW_RESULTS_DIR / "runs"
SUMMARY_RESULTS_DIR = BENCHMARK_ROOT / "results" / "summary"
FIGURES_RESULTS_DIR = BENCHMARK_ROOT / "results" / "figures"
SCHEMA_VERSION = "1.1"
ALLOWED_METHODS = {"fp16", "gptq", "awq"}


def _parse_scalar(raw: str) -> Any:
    """Parse the scalar subset used by the checked-in benchmark configs."""
    value = raw.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _fallback_yaml_mapping(text: str) -> dict[str, Any]:
    """Parse the intentionally small mapping-only YAML config subset.

    This fallback keeps offline freeze and summary commands usable with the
    bundled stdlib-only Python runtime. Full YAML parsing still uses PyYAML
    whenever it is installed in the project virtual environment.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("-"):
            raise ValueError(f"Lists are not supported by fallback YAML parser at line {line_number}.")
        key, separator, raw_value = stripped.partition(":")
        if not separator or not key.strip():
            raise ValueError(f"Invalid YAML mapping at line {line_number}: {line!r}")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if not raw_value.strip():
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _parse_scalar(raw_value)
    return root


def utc_now() -> str:
    """Return a compact UTC timestamp for result provenance."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id(experiment_id: str) -> str:
    """Create a readable, collision-resistant immutable run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{experiment_id}_{uuid.uuid4().hex[:8]}"


def _package_version(distribution: str) -> str | None:
    """Read an installed distribution version without requiring the package."""
    try:
        return package_metadata.version(distribution)
    except package_metadata.PackageNotFoundError:
        return None


def _command_output(command: list[str]) -> str | None:
    """Return a short command result for environment provenance."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def collect_environment_fingerprint() -> dict[str, Any]:
    """Capture software, CUDA, GPU, OS, and Git provenance for a run."""
    try:
        import torch
    except Exception:  # pragma: no cover - migration can run without torch
        torch = None

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    gpu_name = None
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
    driver = _command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    git_commit = _command_output(
        [
            "git",
            "-c",
            f"safe.directory={PROJECT_ROOT}",
            "-C",
            str(PROJECT_ROOT),
            "rev-parse",
            "HEAD",
        ]
    )
    git_status = _command_output(
        [
            "git",
            "-c",
            f"safe.directory={PROJECT_ROOT}",
            "-C",
            str(PROJECT_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ]
    )
    return {
        "captured_at_utc": utc_now(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": getattr(torch, "__version__", None) if torch is not None else _package_version("torch"),
        "transformers": _package_version("transformers"),
        "gptqmodel": _package_version("gptqmodel"),
        "llmcompressor": _package_version("llmcompressor"),
        "compressed_tensors": _package_version("compressed-tensors"),
        "datasets": _package_version("datasets"),
        "cuda_runtime": (
            getattr(getattr(torch, "version", None), "cuda", None)
            if torch is not None
            else None
        ),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_driver": driver.splitlines()[0].strip() if driver else None,
        "os": platform.platform(),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
    }


def resolve_path(value: str | None) -> Path | None:
    """Resolve an absolute path or a path relative to the project root."""
    if value in (None, ""):
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated view over one benchmark YAML document."""

    path: Path
    data: dict[str, Any]

    @property
    def experiment_id(self) -> str:
        return str(self.data["experiment_id"])

    @property
    def method(self) -> str:
        return str(self.data["method"]).lower()

    @property
    def model_id(self) -> str:
        return str(self.data["model_id"])

    @property
    def checkpoint(self) -> Path | None:
        return resolve_path(self.data.get("checkpoint"))

    @property
    def python_executable(self) -> Path:
        value = self.data["runtime"]["python"]
        path = resolve_path(str(value))
        assert path is not None
        return path

    @property
    def raw_result_path(self) -> Path:
        return RAW_RESULTS_DIR / f"{self.experiment_id}.json"

    def validate(self, *, check_paths: bool = False) -> None:
        """Validate v1 invariants and optionally verify local paths."""
        required = ("schema_version", "experiment_id", "model_id", "method", "bits")
        missing = [key for key in required if key not in self.data]
        if missing:
            raise ValueError(f"Missing config fields: {missing}")
        if self.data["schema_version"] != 1:
            raise ValueError("Only benchmark schema_version=1 is supported.")
        if self.method not in ALLOWED_METHODS:
            raise ValueError(f"Unsupported method: {self.method}")

        ppl = self.data.get("ppl")
        generation = self.data.get("generation")
        runtime = self.data.get("runtime")
        if not isinstance(ppl, dict) or not isinstance(generation, dict):
            raise TypeError("ppl and generation must be mappings.")
        if not isinstance(runtime, dict):
            raise TypeError("runtime must be a mapping.")
        if not 0 < int(ppl["stride"]) <= int(ppl["max_length"]):
            raise ValueError("PPL requires 0 < stride <= max_length.")
        if ppl.get("scope") not in {"smoke", "full"}:
            raise ValueError("PPL scope must be smoke or full.")
        if int(generation["warmup_runs"]) < 1:
            raise ValueError("generation.warmup_runs must be positive.")
        if int(generation["benchmark_runs"]) < 2:
            raise ValueError("generation.benchmark_runs must be at least 2.")
        if int(generation["max_new_tokens"]) < 1:
            raise ValueError("generation.max_new_tokens must be positive.")

        if self.method == "fp16":
            if int(self.data["bits"]) != 16 or self.checkpoint is not None:
                raise ValueError("FP16 must use bits=16 and checkpoint=null.")
        else:
            if int(self.data["bits"]) != 4 or self.checkpoint is None:
                raise ValueError("Quantized configs require bits=4 and a checkpoint.")
            calibration = self.data.get("calibration")
            if not isinstance(calibration, dict):
                raise TypeError("Quantized configs require calibration metadata.")
            digest = str(calibration.get("pool_sha256", ""))
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("calibration.pool_sha256 must be a lowercase SHA256.")

        if check_paths:
            if not self.python_executable.is_file():
                raise FileNotFoundError(
                    f"Configured Python does not exist: {self.python_executable}"
                )
            if self.checkpoint is not None and not self.checkpoint.is_dir():
                raise FileNotFoundError(
                    f"Configured checkpoint does not exist: {self.checkpoint}"
                )


def load_config(path: str | Path, *, check_paths: bool = False) -> ExperimentConfig:
    """Load and validate one YAML experiment config."""
    config_path = Path(path).resolve()
    text = config_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if yaml is not None else _fallback_yaml_mapping(text)
    if not isinstance(payload, dict):
        raise TypeError(f"Config must contain a YAML mapping: {config_path}")
    config = ExperimentConfig(config_path, payload)
    config.validate(check_paths=check_paths)
    return config


def apply_runtime_environment() -> None:
    """Pin all caches and temporary files to the project drive."""
    cache_root = PROJECT_ROOT / ".hf_cache"
    temp_root = PROJECT_ROOT / ".tmp-benchmark"
    extension_root = PROJECT_ROOT / ".torch_extensions-benchmark"
    cuda_cache_root = PROJECT_ROOT / ".cuda_cache"
    for directory in (cache_root, cache_root / "hub", temp_root, extension_root, cuda_cache_root):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "hub")
    os.environ["TEMP"] = str(temp_root)
    os.environ["TMP"] = str(temp_root)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(extension_root)
    os.environ["GPTQMODEL_TORCH_EXTENSIONS_DIR"] = str(extension_root / "gptqmodel")
    os.environ["CUDA_CACHE_PATH"] = str(cuda_cache_root)


def directory_size_mib(path: Path | None) -> float | None:
    """Return the recursive file size of a checkpoint directory."""
    if path is None:
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1024**2


def base_result(config: ExperimentConfig) -> dict[str, Any]:
    """Create the stable top-level portion of a unified raw result."""
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "model_id": config.model_id,
        "method": config.method.upper(),
        "bits": config.data["bits"],
        "group_size": config.data.get("group_size"),
        "symmetric": config.data.get("symmetric"),
        "checkpoint": str(config.checkpoint) if config.checkpoint else None,
        "calibration": config.data.get("calibration"),
        "protocol": {
            "ppl": config.data["ppl"],
            "generation": config.data["generation"],
        },
        "runtime": {
            "backend": config.data["runtime"]["backend"],
            "system_metrics_comparable": bool(
                config.data["runtime"]["system_metrics_comparable"]
            ),
            "comparability_note": config.data["runtime"].get("comparability_note"),
        },
        "updated_at_utc": utc_now(),
    }


def write_result(
    config: ExperimentConfig,
    updates: dict[str, Any],
    *,
    replace: bool = False,
) -> Path:
    """Atomically replace or top-level merge one experiment result."""
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = base_result(config)
    if config.raw_result_path.exists() and not replace:
        current = json.loads(config.raw_result_path.read_text(encoding="utf-8"))
        if current.get("experiment_id") != config.experiment_id:
            raise ValueError("Existing result experiment_id does not match its file name.")
        result.update(current)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    run_id = make_run_id(config.experiment_id)
    result["schema_version"] = SCHEMA_VERSION
    result["run_id"] = run_id
    result["status"] = "passed"
    result["environment"] = collect_environment_fingerprint()
    result["updated_at_utc"] = utc_now()
    temp_path = config.raw_result_path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(config.raw_result_path)
    RAW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RAW_RUNS_DIR / f"{run_id}.json"
    run_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config.raw_result_path


def load_baseline_quality(config: ExperimentConfig) -> dict[str, Any] | None:
    """Load the configured baseline quality section when available."""
    baseline_id = config.data.get("baseline_experiment_id")
    if not baseline_id:
        return None
    path = RAW_RESULTS_DIR / f"{baseline_id}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    quality = payload.get("quality")
    return quality if isinstance(quality, dict) else None
