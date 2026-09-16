"""Dependency-free helpers for inspecting Hugging Face checkpoint metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
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


def normalize_method(value: Any) -> str | None:
    """Normalize an explicit algorithm identifier without guessing from storage."""
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "gptqmodel": "GPTQ",
        "gptq-v2": "GPTQ",
        "gptq": "GPTQ",
        "awq": "AWQ",
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized in {"compressed-tensors", "compressed-tensors-format"}:
        return None
    return value.strip().upper()


def inspect_hf_checkpoint(checkpoint: Path | None) -> dict[str, Any]:
    """Inspect known JSON metadata while reporting only explicit fields."""
    if checkpoint is None or not checkpoint.is_dir():
        return {"status": "not_detectable", "files": []}

    filenames = (
        "awq_metadata.json",
        "group_size_ablation_metadata.json",
        "ablation_metadata.json",
        "quantization_config.json",
        "config.json",
    )
    payloads: list[tuple[str, dict[str, Any]]] = []
    for filename in filenames:
        payload = _read_json(checkpoint / filename)
        if payload is not None:
            payloads.append((filename, payload))

    named = {filename: payload for filename, payload in payloads}
    awq = named.get("awq_metadata.json", {})
    gptq = named.get("group_size_ablation_metadata.json", named.get("ablation_metadata.json", {}))
    standalone = named.get("quantization_config.json", {})
    config = named.get("config.json", {})
    qconfig = config.get("quantization_config")
    if not isinstance(qconfig, dict):
        qconfig = standalone
    if not isinstance(qconfig, dict):
        qconfig = {}
    weights = _nested_mapping(qconfig, "config_groups", "group_0", "weights") or {}

    method = normalize_method(
        _first_value(awq.get("method"), gptq.get("method"), qconfig.get("quant_method"))
    )
    bits = _first_value(
        awq.get("bits"), gptq.get("bits"), qconfig.get("bits"),
        qconfig.get("w_bit"), weights.get("num_bits"),
    )
    group_size = _first_value(
        awq.get("group_size"), gptq.get("group_size"), qconfig.get("group_size"),
        qconfig.get("q_group_size"), weights.get("group_size"),
    )
    symmetric = _first_value(
        awq.get("symmetric"), gptq.get("symmetric"), gptq.get("sym"),
        qconfig.get("sym"), qconfig.get("symmetric"), weights.get("symmetric"),
    )
    duo_scaling = _first_value(awq.get("duo_scaling"), gptq.get("duo_scaling"))
    detected = {
        key: value
        for key, value in {
            "method": method,
            "bits": bits,
            "group_size": group_size,
            "symmetric": symmetric,
            "duo_scaling": duo_scaling,
        }.items()
        if value is not None
    }
    return {
        "status": "detected" if detected else "not_detectable",
        "files": [filename for filename, _ in payloads],
        **detected,
    }
