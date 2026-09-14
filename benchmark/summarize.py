"""Generate JSON, CSV, and Markdown summaries from unified raw results."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.config import RAW_RESULTS_DIR, SUMMARY_RESULTS_DIR


ORDER = {"fp16": 0, "gptq_g32": 1, "gptq_g64": 2, "gptq_g128": 3, "awq_g64": 4}


def nested(payload: dict[str, Any], section: str, field: str) -> Any:
    """Read an optional nested result field."""
    value = payload.get(section)
    return value.get(field) if isinstance(value, dict) else None


def build_row(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten one raw result into a stable summary row."""
    throughput_mean = nested(payload, "performance", "tokens_per_second_mean")
    throughput_std = nested(payload, "performance", "tokens_per_second_std")
    comparable = bool(nested(payload, "runtime", "system_metrics_comparable"))
    return {
        "experiment_id": payload["experiment_id"],
        "method": payload["method"],
        "group_size": payload.get("group_size"),
        "bits": payload["bits"],
        "ppl": nested(payload, "quality", "ppl"),
        "ppl_delta": nested(payload, "quality", "ppl_delta"),
        "ppl_relative_increase_percent": nested(
            payload, "quality", "ppl_relative_increase_percent"
        ),
        "checkpoint_mib": nested(payload, "storage", "checkpoint_mib"),
        "resident_vram_mib": nested(payload, "memory", "resident_vram_mib"),
        "peak_vram_mib": nested(payload, "memory", "peak_vram_mib"),
        "tokens_per_second_mean": throughput_mean,
        "tokens_per_second_std": throughput_std,
        "system_metrics_comparable": comparable,
        "runtime_backend": nested(payload, "runtime", "backend"),
        "provenance_status": nested(payload, "provenance", "status"),
    }


def format_number(value: Any, digits: int = 2) -> str:
    """Format optional numeric values for the Markdown report."""
    return "—" if value is None else f"{float(value):.{digits}f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render the compact human-readable v1 benchmark table."""
    lines = [
        "# Unified quantization benchmark v1",
        "",
        "| Method | G | PPL | ΔPPL | CKPT MiB | Resident MiB | Peak MiB | System | tok/s mean ± std |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        group = "—" if row["group_size"] is None else str(row["group_size"])
        if row["tokens_per_second_mean"] is None:
            speed = "—"
        elif row["tokens_per_second_std"] is None:
            speed = f"{row['tokens_per_second_mean']:.2f} ± —"
        else:
            speed = (
                f"{row['tokens_per_second_mean']:.2f} ± "
                f"{row['tokens_per_second_std']:.2f}"
            )
        system_status = "comparable" if row["system_metrics_comparable"] else "N/C"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    group,
                    format_number(row["ppl"], 4),
                    format_number(row["ppl_delta"], 4),
                    format_number(row["checkpoint_mib"], 2),
                    format_number(row["resident_vram_mib"], 2),
                    format_number(row["peak_vram_mib"], 2),
                    system_status,
                    speed,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "`N/C` means the runtime representation is not comparable to the quantized GPTQ kernel.",
            "Legacy frozen speed rows may not have a standard deviation; rerun `eval_system` for v1 mean/std.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not RAW_RESULTS_DIR.is_dir():
        raise FileNotFoundError(f"Raw result directory not found: {RAW_RESULTS_DIR}")
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in RAW_RESULTS_DIR.glob("*.json")
    ]
    rows = sorted(
        (build_row(payload) for payload in payloads),
        key=lambda row: (ORDER.get(row["experiment_id"], 999), row["experiment_id"]),
    )
    if not rows:
        raise RuntimeError("No unified raw results were found.")

    SUMMARY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = SUMMARY_RESULTS_DIR / "benchmark_v1_summary.json"
    csv_path = SUMMARY_RESULTS_DIR / "benchmark_v1_summary.csv"
    markdown_path = SUMMARY_RESULTS_DIR / "benchmark_v1_summary.md"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(markdown_table(rows), encoding="utf-8")
    print(f"summary_json={json_path}")
    print(f"summary_csv={csv_path}")
    print(f"summary_markdown={markdown_path}")


if __name__ == "__main__":
    main()
