"""Regression and protocol gates for benchmark v1.2 summary generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.config import (
    BENCHMARK_ROOT,
    LEGACY_SCHEMA_VERSIONS,
    SCHEMA_VERSION,
    load_config,
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate current raw results before summaries and figures are emitted."""
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    downstream_protocols: list[tuple[str, dict[str, Any]]] = []

    def check(name: str, passed: bool, detail: str, *, warning: bool = False) -> None:
        check_status = "passed" if passed else "warning" if warning else "failed"
        checks.append({"name": name, "status": check_status, "detail": detail})
        if not passed:
            (warnings if warning else errors).append(f"{name}: {detail}")

    for payload in payloads:
        experiment_id = str(payload.get("experiment_id", "<missing>"))
        config_path = BENCHMARK_ROOT / "configs" / f"{experiment_id}.yaml"
        try:
            config = load_config(config_path, check_paths=False)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{experiment_id}: cannot load matching config: {exc}")
            continue

        schema_version = payload.get("schema_version")
        downstream_config = config.data.get("downstream")
        requires_downstream = isinstance(downstream_config, dict) and downstream_config.get("enabled") is True
        schema_valid = schema_version == SCHEMA_VERSION or schema_version in LEGACY_SCHEMA_VERSIONS
        check(
            f"{experiment_id}.schema_version",
            schema_valid,
            f"expected {SCHEMA_VERSION!r} or a supported legacy version, got {schema_version!r}",
        )
        if schema_version in LEGACY_SCHEMA_VERSIONS:
            warnings.append(
                f"{experiment_id}.schema_version: retained legacy {schema_version} result"
            )

        run_id = payload.get("run_id")
        run_id_valid = isinstance(run_id, str) and bool(run_id.strip())
        if isinstance(run_id, str) and run_id in seen_run_ids:
            run_id_valid = False
            errors.append(f"{experiment_id}.run_id: duplicate run_id {run_id!r}")
        if isinstance(run_id, str):
            seen_run_ids.add(run_id)
        check(
            f"{experiment_id}.run_id",
            run_id_valid,
            "run_id must be a unique non-empty string",
        )

        check(
            f"{experiment_id}.status",
            payload.get("status") == "passed",
            f"expected 'passed', got {payload.get('status')!r}",
        )

        metadata = payload.get("experiment_metadata")
        if metadata is not None:
            metadata_ok = isinstance(metadata, dict)
            if metadata_ok:
                identity = metadata.get("identity")
                quantization = metadata.get("quantization")
                evaluation = metadata.get("evaluation")
                metadata_ok = all(
                    isinstance(section, dict)
                    for section in (identity, quantization, evaluation)
                ) and metadata.get("experiment_id") == experiment_id
            check(
                f"{experiment_id}.experiment_metadata",
                metadata_ok,
                "new raw results must carry identity, quantization, and evaluation metadata",
            )

        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        provenance_status = provenance.get("status")
        environment = payload.get("environment")
        if not isinstance(environment, dict):
            check(
                f"{experiment_id}.environment",
                False,
                "fingerprint is missing",
                warning=provenance_status == "legacy_frozen",
            )
        else:
            required_environment = ("python", "torch", "transformers", "os", "git_commit")
            if requires_downstream:
                required_environment += ("lm_eval",)
            missing_environment = [key for key in required_environment if not environment.get(key)]
            check(
                f"{experiment_id}.environment",
                not missing_environment,
                f"missing fields: {missing_environment}",
                warning=provenance_status == "legacy_frozen",
            )

        if requires_downstream:
            quality_section = payload.get("quality")
            downstream_results = (
                quality_section.get("downstream")
                if isinstance(quality_section, dict)
                else None
            )
            task_config = downstream_config["tasks"][0]
            task_name = str(task_config["name"])
            task_result = (
                downstream_results.get(task_name)
                if isinstance(downstream_results, dict)
                else None
            )
            if not isinstance(task_result, dict):
                check(
                    f"{experiment_id}.downstream.{task_name}",
                    False,
                    "full downstream result is missing",
                )
            else:
                for metric in ("acc", "acc_stderr", "acc_norm", "acc_norm_stderr"):
                    value = _number(task_result.get(metric))
                    check(
                        f"{experiment_id}.downstream.{task_name}.{metric}",
                        value is not None and 0.0 <= value <= 1.0,
                        f"expected a value in [0, 1], got {task_result.get(metric)!r}",
                    )
                check(
                    f"{experiment_id}.downstream.{task_name}.task_name",
                    task_result.get("task_name") == task_name == "hellaswag",
                    f"expected hellaswag, got {task_result.get('task_name')!r}",
                )
                check(
                    f"{experiment_id}.downstream.{task_name}.num_fewshot",
                    task_result.get("num_fewshot") == task_config["num_fewshot"] == 0,
                    f"expected zero-shot, got {task_result.get('num_fewshot')!r}",
                )
                check(
                    f"{experiment_id}.downstream.{task_name}.apply_chat_template",
                    task_result.get("apply_chat_template") is downstream_config["apply_chat_template"] is False,
                    f"expected false, got {task_result.get('apply_chat_template')!r}",
                )
                check(
                    f"{experiment_id}.downstream.{task_name}.scope",
                    task_result.get("scope") == "full" and task_result.get("limit") is None,
                    f"expected full scope with no limit, got scope={task_result.get('scope')!r}, limit={task_result.get('limit')!r}",
                )
                check(
                    f"{experiment_id}.downstream.{task_name}.samples",
                    isinstance(task_result.get("n_samples"), int)
                    and task_result.get("n_samples") == task_result.get("original_samples")
                    and task_result.get("n_samples") > 0,
                    f"effective={task_result.get('n_samples')!r}, original={task_result.get('original_samples')!r}",
                )
                check(
                    f"{experiment_id}.downstream.{task_name}.harness",
                    task_result.get("harness") == "lm-evaluation-harness"
                    and bool(task_result.get("harness_version")),
                    f"harness={task_result.get('harness')!r}, version={task_result.get('harness_version')!r}",
                )
                check(
                    f"{experiment_id}.downstream.{task_name}.environment_harness_version",
                    isinstance(environment, dict)
                    and environment.get("lm_eval") == task_result.get("harness_version"),
                    "environment fingerprint and downstream result must record the same lm-eval version",
                )
                check(
                    f"{experiment_id}.downstream.{task_name}.offline",
                    task_result.get("offline") is True,
                    f"expected cache-only execution, got {task_result.get('offline')!r}",
                )
                fingerprint_fields = (
                    "harness_version",
                    "task_version",
                    "task_config_sha256",
                    "dataset_name",
                    "dataset_split",
                    "dataset_fingerprint",
                    "n_samples",
                    "num_fewshot",
                    "apply_chat_template",
                    "tokenizer_fingerprint",
                    "tokenizer_fingerprint_algorithm",
                )
                missing = [field for field in fingerprint_fields if task_result.get(field) in (None, "")]
                check(
                    f"{experiment_id}.downstream.{task_name}.fingerprint",
                    not missing,
                    f"missing fields: {missing}",
                )
                downstream_protocols.append((experiment_id, task_result))

        protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
        generation = protocol.get("generation") if isinstance(protocol.get("generation"), dict) else {}
        expected_generation = config.data["generation"]
        for field in ("warmup_runs", "benchmark_runs", "max_new_tokens"):
            check(
                f"{experiment_id}.protocol.{field}",
                generation.get(field) == expected_generation[field],
                f"expected {expected_generation[field]!r}, got {generation.get(field)!r}",
            )

        quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
        ppl = _number(quality.get("ppl"))
        expected_ppl = _number(config.data.get("validation", {}).get("expected_ppl"))
        tolerance = float(config.data.get("validation", {}).get("ppl_tolerance", 0.0))
        if ppl is None or expected_ppl is None:
            check(
                f"{experiment_id}.quality.ppl",
                False,
                "PPL or expected PPL is missing",
            )
        else:
            check(
                f"{experiment_id}.quality.ppl",
                abs(ppl - expected_ppl) <= tolerance,
                f"actual={ppl:.12f}, expected={expected_ppl:.12f}, tolerance={tolerance}",
            )

        expected_tokens = int(config.data.get("validation", {}).get("expected_evaluated_tokens", 299077))
        evaluated_tokens = quality.get("evaluated_tokens")
        check(
            f"{experiment_id}.quality.evaluated_tokens",
            evaluated_tokens == expected_tokens,
            f"expected {expected_tokens}, got {evaluated_tokens!r}",
        )
        input_tokens = quality.get("num_input_tokens")
        check(
            f"{experiment_id}.quality.token_coverage",
            isinstance(input_tokens, int)
            and isinstance(evaluated_tokens, int)
            and evaluated_tokens == input_tokens - 1,
            f"expected evaluated_tokens == num_input_tokens - 1, got {input_tokens!r} and {evaluated_tokens!r}",
        )

        performance = payload.get("performance") if isinstance(payload.get("performance"), dict) else None
        if performance is not None:
            is_v1_system = provenance_status == "v1_rerun" or bool(
                (payload.get("system_validation") or {}).get("isolated_process")
            )
            if is_v1_system:
                check(
                    f"{experiment_id}.performance.warmup_runs",
                    performance.get("warmup_runs") == expected_generation["warmup_runs"],
                    f"expected {expected_generation['warmup_runs']}, got {performance.get('warmup_runs')!r}",
                )
                check(
                    f"{experiment_id}.performance.measure_runs",
                    performance.get("benchmark_runs") == expected_generation["benchmark_runs"],
                    f"expected {expected_generation['benchmark_runs']}, got {performance.get('benchmark_runs')!r}",
                )
            else:
                warnings.append(
                    f"{experiment_id}.performance: legacy frozen speed row is excluded from v1 repeated-run gates"
                )

        memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}
        resident = _number(memory.get("resident_vram_mib"))
        peak = _number(memory.get("peak_vram_mib"))
        if resident is not None and peak is not None:
            check(
                f"{experiment_id}.memory.order",
                peak >= resident,
                f"peak={peak:.4f} must be >= resident={resident:.4f}",
            )

        comparable = bool((payload.get("runtime") or {}).get("system_metrics_comparable"))
        if not comparable:
            warnings.append(
                f"{experiment_id}.system: system metrics are excluded because comparable=false"
            )

    if downstream_protocols:
        expected_downstream_count = 4
        check(
            "downstream.comparison.matrix",
            len(downstream_protocols) == expected_downstream_count,
            f"expected {expected_downstream_count} full HellaSwag rows, got {len(downstream_protocols)}",
        )
        comparison_fields = (
            "harness_version",
            "task_version",
            "task_config_sha256",
            "dataset_name",
            "dataset_split",
            "dataset_fingerprint",
            "n_samples",
            "num_fewshot",
            "apply_chat_template",
            "tokenizer_fingerprint",
            "tokenizer_fingerprint_algorithm",
        )
        for field in comparison_fields:
            values = {str(result.get(field)) for _, result in downstream_protocols}
            check(
                f"downstream.comparison.same_{field}",
                len(values) == 1,
                f"values differ across rows: {sorted(values)}",
            )
        for experiment_id, result in downstream_protocols:
            check(
                f"{experiment_id}.downstream.quality_comparable",
                result.get("quality_comparable") is True,
                "row is excluded from downstream comparison",
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def load_raw_payloads(raw_dir: Path) -> list[dict[str, Any]]:
    """Load only latest pointers, excluding immutable run snapshots."""
    payloads: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads
