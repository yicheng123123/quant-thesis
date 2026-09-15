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
SCHEMA_VERSION = "1.2.1"
EXPERIMENT_METADATA_VERSION = "1.2.1"
LEGACY_SCHEMA_VERSIONS = {"1.1", "1.2"}
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
    lines = [
        (line_number, line)
        for line_number, line in enumerate(text.splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    for index, (line_number, line) in enumerate(lines):
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unexpected YAML list item at line {line_number}.")
            item_text = stripped[2:].strip()
            if not item_text:
                item: dict[str, Any] = {}
                parent.append(item)
                stack.append((indent, item))
                continue
            key, separator, raw_value = item_text.partition(":")
            if separator:
                item = {}
                parent.append(item)
                if raw_value.strip():
                    item[key.strip()] = _parse_scalar(raw_value)
                else:
                    item[key.strip()] = {}
                stack.append((indent, item))
            else:
                parent.append(_parse_scalar(item_text))
            continue

        key, separator, raw_value = stripped.partition(":")
        if not separator or not key.strip():
            raise ValueError(f"Invalid YAML mapping at line {line_number}: {line!r}")
        if not isinstance(parent, dict):
            raise ValueError(f"Expected a YAML mapping at line {line_number}.")
        if not raw_value.strip():
            next_is_list = False
            if index + 1 < len(lines):
                _, next_line = lines[index + 1]
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                next_is_list = next_indent > indent and next_line.strip().startswith("- ")
            child: dict[str, Any] | list[Any] = [] if next_is_list else {}
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
        "lm_eval": _package_version("lm_eval"),
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


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a checkpoint-side JSON mapping without importing model libraries."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _nested_mapping(payload: Any, *keys: str) -> dict[str, Any] | None:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, dict) else None


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_checkpoint_method(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"gptq", "gptqmodel", "gptq-v2"}:
        return "GPTQ"
    if normalized == "awq":
        return "AWQ"
    # compressed-tensors is a storage/format identifier, not proof of AWQ.
    return None


def inspect_checkpoint_metadata(checkpoint: Path | None) -> dict[str, Any]:
    """Inspect known checkpoint metadata files, returning only detectable fields."""
    if checkpoint is None:
        return {"status": "not_applicable", "files": []}
    if not checkpoint.is_dir():
        return {"status": "not_detectable", "files": []}

    metadata_files = (
        "awq_metadata.json",
        "group_size_ablation_metadata.json",
        "ablation_metadata.json",
        "config.json",
    )
    payloads: list[tuple[str, dict[str, Any]]] = []
    for filename in metadata_files:
        path = checkpoint / filename
        if path.is_file():
            payload = _read_json(path)
            if payload is not None:
                payloads.append((filename, payload))

    awq = next((payload for filename, payload in payloads if filename == "awq_metadata.json"), {})
    gptq = next(
        (
            payload
            for filename, payload in payloads
            if filename in {"group_size_ablation_metadata.json", "ablation_metadata.json"}
        ),
        {},
    )
    config = next((payload for filename, payload in payloads if filename == "config.json"), {})
    qconfig = config.get("quantization_config")
    if not isinstance(qconfig, dict):
        qconfig = {}
    awq_weights = _nested_mapping(qconfig, "config_groups", "group_0", "weights") or {}

    method = _normalize_checkpoint_method(
        _first_value(awq.get("method"), gptq.get("method"), qconfig.get("quant_method"))
    )
    bits = _first_value(
        awq.get("bits"),
        gptq.get("bits"),
        qconfig.get("bits"),
        awq_weights.get("num_bits"),
    )
    group_size = _first_value(
        awq.get("group_size"),
        gptq.get("group_size"),
        qconfig.get("group_size"),
        awq_weights.get("group_size"),
    )
    symmetric = _first_value(
        awq.get("symmetric"),
        gptq.get("symmetric"),
        gptq.get("sym"),
        qconfig.get("sym"),
        qconfig.get("symmetric"),
        awq_weights.get("symmetric"),
    )
    duo_scaling = _first_value(awq.get("duo_scaling"), gptq.get("duo_scaling"))
    detectable = {
        "method": method,
        "bits": bits,
        "group_size": group_size,
        "symmetric": symmetric,
        "duo_scaling": duo_scaling,
    }
    detected_fields = {key: value for key, value in detectable.items() if value is not None}
    return {
        "status": "detected" if detected_fields else "not_detectable",
        "files": [filename for filename, _ in payloads],
        **detected_fields,
    }


def validate_checkpoint_identity(config: "ExperimentConfig") -> dict[str, Any]:
    """Cross-check detectable checkpoint identity against the config declaration."""
    metadata = inspect_checkpoint_metadata(config.checkpoint)
    expected = config.experiment_metadata
    comparisons = {
        "method": (expected["method"], metadata.get("method")),
        "bits": (expected["weight_bits"], metadata.get("bits")),
        "group_size": (expected["group_size"], metadata.get("group_size")),
        "symmetric": (expected["symmetric"], metadata.get("symmetric")),
        "duo_scaling": (expected["duo_scaling"], metadata.get("duo_scaling")),
    }
    for field, (configured, detected) in comparisons.items():
        if detected is None or configured is None:
            continue
        if field == "method":
            matches = str(configured).upper() == str(detected).upper()
        elif field in {"bits", "group_size"}:
            try:
                matches = int(configured) == int(detected)
            except (TypeError, ValueError):
                matches = configured == detected
        elif field in {"symmetric", "duo_scaling"}:
            def _as_bool(value: Any) -> Any:
                if isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in {"true", "yes", "1"}:
                        return True
                    if lowered in {"false", "no", "0"}:
                        return False
                return value

            matches = _as_bool(configured) == _as_bool(detected)
        else:
            matches = configured == detected
        if not matches:
            raise ValueError(
                f"Checkpoint metadata mismatch for {config.experiment_id}: "
                f"{field} configured={configured!r}, detected={detected!r} "
                f"in {metadata.get('files', [])}."
            )
    return metadata


def manifest_context(
    config: "ExperimentConfig",
    *,
    task: str | None = None,
    scope: str | None = None,
    dataset: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Build the complete human-readable and machine-readable run manifest."""
    metadata = dict(config.experiment_metadata)
    checkpoint_metadata = validate_checkpoint_identity(config)
    metadata.update(
        {
            "task": task if task is not None else metadata.get("task"),
            "scope": scope if scope is not None else metadata.get("scope"),
            "dataset": dataset if dataset is not None else metadata.get("dataset"),
            "split": split if split is not None else metadata.get("split"),
            "checkpoint_metadata": checkpoint_metadata,
            "output_paths": {
                "result": str(config.raw_result_path),
                "run_snapshots_dir": str(RAW_RUNS_DIR),
                "run_snapshot_pattern": f"*_{config.experiment_id}_*.json",
            },
        }
    )
    metadata["identity"] = {
        "experiment_id": metadata["experiment_id"],
        "base_model": metadata["base_model"],
        "checkpoint": metadata["checkpoint"],
    }
    metadata["quantization"] = {
        "method": metadata["method"],
        "weight_bits": metadata["weight_bits"],
        "activation_bits": metadata["activation_bits"],
        "activation_precision": metadata["activation_precision"],
        "precision": metadata["precision"],
        "group_size": metadata["group_size"],
        "symmetric": metadata["symmetric"],
        "backend": metadata["backend"],
        "duo_scaling": metadata["duo_scaling"],
    }
    metadata["evaluation"] = {
        "task": metadata["task"],
        "scope": metadata["scope"],
        "dataset": metadata["dataset"],
        "split": metadata["split"],
    }
    return metadata


def print_manifest(
    config: "ExperimentConfig",
    *,
    task: str | None = None,
    scope: str | None = None,
    dataset: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Print a stable startup manifest and return its normalized mapping."""
    manifest = manifest_context(
        config,
        task=task,
        scope=scope,
        dataset=dataset,
        split=split,
    )
    checkpoint = manifest["checkpoint"] or "<none>"
    group_size = manifest["group_size"] if manifest["group_size"] is not None else "<none>"
    symmetric = manifest["symmetric"] if manifest["symmetric"] is not None else "<none>"
    duo_scaling = manifest["duo_scaling"] if manifest["duo_scaling"] is not None else "<none>"
    print("=" * 60)
    print("BENCHMARK EXPERIMENT")
    print("=" * 60)
    print(f"Experiment    : {manifest['experiment_id']}")
    print(f"Base model    : {manifest['base_model']}")
    print(f"Checkpoint    : {checkpoint}")
    print(f"Quantization  : {manifest['method']}")
    print(f"Precision     : {manifest['precision']}")
    print(f"Weight bits   : {manifest['weight_bits']}")
    print(f"Activation precision: {manifest['activation_precision']}")
    print(f"Group size    : {group_size}")
    print(f"Symmetric     : {symmetric}")
    print(f"Backend       : {manifest['backend']}")
    print(f"AWQ duo_scaling: {duo_scaling}")
    print(f"Task          : {manifest.get('task') or '<none>'}")
    print(f"Scope         : {manifest.get('scope') or '<none>'}")
    print(f"Dataset       : {manifest.get('dataset') or '<none>'}")
    print(f"Split         : {manifest.get('split') or '<none>'}")
    print(f"Result path   : {manifest['output_paths']['result']}")
    print(f"Run snapshots : {manifest['output_paths']['run_snapshots_dir']}")
    checkpoint_status = manifest["checkpoint_metadata"].get("status")
    print(f"Checkpoint ID : {checkpoint_status}")
    print("=" * 60)
    return manifest


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

    @property
    def experiment_metadata(self) -> dict[str, Any]:
        """Return the normalized identity and quantization declaration.

        v1.2 configs kept the original top-level fields for compatibility.  The
        v1.2.1 ``experiment`` and ``quantization`` mappings are authoritative,
        with the legacy fields accepted as aliases and checked for agreement.
        """
        identity = self.data.get("experiment")
        quantization = self.data.get("quantization")
        if not isinstance(identity, dict):
            identity = {}
        if not isinstance(quantization, dict):
            quantization = {}

        method = str(quantization.get("method", self.data.get("method", ""))).lower()
        weight_bits = quantization.get("weight_bits", self.data.get("bits"))
        activation_bits = quantization.get("activation_bits")
        activation_precision = quantization.get("activation_precision")
        if activation_bits is None and isinstance(activation_precision, str):
            if activation_precision.upper() == "FP16":
                activation_bits = 16
            elif activation_precision.upper().startswith("A"):
                try:
                    activation_bits = int(activation_precision[1:])
                except ValueError:
                    activation_bits = None
        if activation_precision is None:
            activation_precision = "FP16" if method == "fp16" else f"A{activation_bits or 16}"
        group_size = quantization.get("group_size", self.data.get("group_size"))
        symmetric = quantization.get("symmetric", self.data.get("symmetric"))
        duo_scaling = quantization.get("duo_scaling")
        if duo_scaling is None:
            calibration = self.data.get("calibration")
            if isinstance(calibration, dict):
                duo_scaling = calibration.get("duo_scaling")
        checkpoint_value = identity.get("checkpoint", self.data.get("checkpoint"))
        checkpoint = resolve_path(checkpoint_value)
        runtime = self.data.get("runtime")
        runtime_backend = runtime.get("backend") if isinstance(runtime, dict) else ""
        return {
            "metadata_version": EXPERIMENT_METADATA_VERSION,
            "experiment_id": str(identity.get("id", self.data.get("experiment_id", ""))),
            "base_model": str(identity.get("base_model", self.data.get("model_id", ""))),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "method": method.upper(),
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "activation_precision": str(activation_precision),
            "precision": (
                "FP16"
                if method == "fp16"
                else f"W{weight_bits}A{activation_bits or activation_precision}"
            ),
            "group_size": group_size,
            "symmetric": symmetric,
            "backend": str(quantization.get("backend", runtime_backend)),
            "duo_scaling": duo_scaling,
            "task": identity.get("task"),
            "scope": identity.get("scope"),
            "dataset": identity.get("dataset"),
            "split": identity.get("split"),
        }

    @property
    def weight_bits(self) -> int:
        return int(self.experiment_metadata["weight_bits"])

    @property
    def activation_bits(self) -> int:
        return int(self.experiment_metadata["activation_bits"])

    @property
    def activation_precision(self) -> str:
        return str(self.experiment_metadata["activation_precision"])

    @property
    def precision(self) -> str:
        return str(self.experiment_metadata["precision"])

    @property
    def backend(self) -> str:
        return str(self.experiment_metadata["backend"])

    @property
    def duo_scaling(self) -> bool | None:
        value = self.experiment_metadata["duo_scaling"]
        return bool(value) if value is not None else None

    @property
    def checkpoint_metadata(self) -> dict[str, Any]:
        return inspect_checkpoint_metadata(self.checkpoint)

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

        self._validate_experiment_metadata()

        ppl = self.data.get("ppl")
        generation = self.data.get("generation")
        runtime = self.data.get("runtime")
        if not isinstance(ppl, dict) or not isinstance(generation, dict):
            raise TypeError("ppl and generation must be mappings.")
        if not isinstance(runtime, dict):
            raise TypeError("runtime must be a mapping.")

        downstream = self.data.get("downstream")
        if downstream is not None:
            if not isinstance(downstream, dict):
                raise TypeError("downstream must be a mapping.")
            if downstream.get("enabled") is not True:
                raise ValueError("Configured downstream evaluation must set enabled=true.")
            if downstream.get("harness") != "lm_eval":
                raise ValueError("benchmark v1.2 supports downstream.harness=lm_eval only.")
            tasks = downstream.get("tasks")
            if not isinstance(tasks, list) or not tasks:
                raise TypeError("downstream.tasks must be a non-empty list.")
            for task in tasks:
                if not isinstance(task, dict) or not task.get("name"):
                    raise TypeError("Each downstream task must be a mapping with a name.")
                if int(task.get("num_fewshot", -1)) != 0:
                    raise ValueError("benchmark v1.2 downstream tasks must be zero-shot.")
                if task.get("primary_metric") not in {"acc", "acc_norm"}:
                    raise ValueError("Downstream primary_metric must be acc or acc_norm.")
            if int(downstream.get("batch_size", 0)) < 1:
                raise ValueError("downstream.batch_size must be positive.")
            if downstream.get("apply_chat_template") is not False:
                raise ValueError("benchmark v1.2 requires apply_chat_template=false.")
            scope = downstream.get("scope")
            if not isinstance(scope, dict) or int(scope.get("smoke_limit", 0)) < 1:
                raise ValueError("downstream.scope.smoke_limit must be positive.")
            if scope.get("full_limit") is not None:
                raise ValueError("downstream.scope.full_limit must be null for full evaluation.")
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

        if self.checkpoint is not None and self.checkpoint.is_dir():
            validate_checkpoint_identity(self)

    def _validate_experiment_metadata(self) -> None:
        """Validate the explicit v1.2.1 declaration and legacy aliases."""
        identity = self.data.get("experiment")
        quantization = self.data.get("quantization")
        if not isinstance(identity, dict):
            raise TypeError("experiment must be a mapping in benchmark v1.2.1 configs.")
        if not isinstance(quantization, dict):
            raise TypeError("quantization must be a mapping in benchmark v1.2.1 configs.")

        metadata = self.experiment_metadata
        required = (
            "id",
            "base_model",
            "checkpoint",
            "task",
            "scope",
            "dataset",
            "split",
        )
        missing_identity = [key for key in required if key not in identity]
        if missing_identity:
            raise ValueError(f"experiment is missing fields: {missing_identity}")
        for key in ("id", "base_model", "task", "scope", "dataset", "split"):
            if identity.get(key) in (None, ""):
                raise ValueError(f"experiment.{key} must be non-empty.")
        if identity.get("scope") not in {"smoke", "full"}:
            raise ValueError("experiment.scope must be smoke or full.")
        required_quantization = (
            "method",
            "weight_bits",
            "activation_bits",
            "activation_precision",
            "group_size",
            "symmetric",
            "backend",
            "duo_scaling",
        )
        missing_quantization = [key for key in required_quantization if key not in quantization]
        if missing_quantization:
            raise ValueError(f"quantization is missing fields: {missing_quantization}")

        aliases = {
            "experiment.id": (identity["id"], self.data.get("experiment_id")),
            "experiment.base_model": (identity["base_model"], self.data.get("model_id")),
            "experiment.checkpoint": (
                str(resolve_path(identity["checkpoint"])) if identity["checkpoint"] else None,
                str(self.checkpoint) if self.checkpoint else None,
            ),
            "quantization.method": (str(quantization["method"]).lower(), self.method),
            "quantization.weight_bits": (quantization["weight_bits"], self.data.get("bits")),
            "quantization.group_size": (quantization["group_size"], self.data.get("group_size")),
            "quantization.symmetric": (quantization["symmetric"], self.data.get("symmetric")),
            "quantization.backend": (
                quantization["backend"],
                self.data.get("runtime", {}).get("backend")
                if isinstance(self.data.get("runtime"), dict)
                else None,
            ),
        }
        for field, (declared, legacy) in aliases.items():
            if declared != legacy:
                raise ValueError(f"{field} disagrees with its legacy config field: {declared!r} != {legacy!r}")
        if self.method == "awq" and quantization["duo_scaling"] not in {False, True}:
            raise ValueError("AWQ quantization.duo_scaling must be true or false.")
        if self.method != "awq" and quantization["duo_scaling"] is not None:
            raise ValueError("Only AWQ configs may set quantization.duo_scaling.")
        if int(quantization["activation_bits"]) < 1:
            raise ValueError("quantization.activation_bits must be positive.")
        expected_activation_precision = (
            "FP16" if self.method == "fp16" else f"A{int(quantization['activation_bits'])}"
        )
        if str(quantization["activation_precision"]).upper() != expected_activation_precision:
            raise ValueError(
                "quantization.activation_precision must match activation_bits "
                f"({expected_activation_precision})."
            )
        if not str(quantization["backend"]).strip():
            raise ValueError("quantization.backend must be non-empty.")
        if metadata["precision"] != (
            "FP16" if self.method == "fp16" else f"W{metadata['weight_bits']}A{metadata['activation_bits']}"
        ):
            raise ValueError("quantization precision declaration is inconsistent.")


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


def base_result(
    config: ExperimentConfig,
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the stable top-level portion of a unified raw result."""
    metadata = manifest if manifest is not None else manifest_context(config)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "model_id": config.model_id,
        "method": config.method.upper(),
        "bits": config.data["bits"],
        "group_size": config.data.get("group_size"),
        "symmetric": config.data.get("symmetric"),
        "checkpoint": str(config.checkpoint) if config.checkpoint else None,
        "experiment_metadata": metadata,
        "calibration": config.data.get("calibration"),
        "protocol": {
            "ppl": config.data["ppl"],
            "generation": config.data["generation"],
            "downstream": config.data.get("downstream"),
        },
        "runtime": {
            "backend": config.data["runtime"]["backend"],
            "system_metrics_comparable": bool(
                config.data["runtime"]["system_metrics_comparable"]
            ),
            "comparability_note": config.data["runtime"].get("comparability_note"),
        },
        "provenance": {
            "experiment_metadata_version": EXPERIMENT_METADATA_VERSION,
            "result_path": str(config.raw_result_path),
            "run_snapshots_dir": str(RAW_RUNS_DIR),
        },
        "updated_at_utc": utc_now(),
    }


def write_result(
    config: ExperimentConfig,
    updates: dict[str, Any],
    *,
    replace: bool = False,
    manifest: dict[str, Any] | None = None,
) -> Path:
    """Atomically replace or top-level merge one experiment result."""
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = base_result(config, manifest=manifest)
    if config.raw_result_path.exists() and not replace:
        current = json.loads(config.raw_result_path.read_text(encoding="utf-8"))
        if current.get("experiment_id") != config.experiment_id:
            raise ValueError("Existing result experiment_id does not match its file name.")
        result.update(current)
    if manifest is not None:
        result["experiment_metadata"] = manifest
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    run_id = make_run_id(config.experiment_id)
    result["schema_version"] = SCHEMA_VERSION
    result["run_id"] = run_id
    result["status"] = "passed"
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance.update(
        {
            "experiment_metadata_version": EXPERIMENT_METADATA_VERSION,
            "result_path": str(config.raw_result_path),
            "run_snapshot_path": str(RAW_RUNS_DIR / f"{run_id}.json"),
        }
    )
    result["provenance"] = provenance
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
