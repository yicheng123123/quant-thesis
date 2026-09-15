"""Generate JSON, CSV, Markdown, and visual HTML benchmark summaries."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from benchmark.config import FIGURES_RESULTS_DIR, RAW_RESULTS_DIR, SCHEMA_VERSION, SUMMARY_RESULTS_DIR
from benchmark.figures import generate_figures
from benchmark.validation import load_raw_payloads, validate_payloads


ORDER = {
    "fp16": 0,
    "gptq_g32": 1,
    "gptq_g64": 2,
    "gptq_g128": 3,
    "awq_g64": 4,
    "awq_g64_duo_false": 5,
    "awq_g64_duo_true": 6,
}


def nested(payload: dict[str, Any], section: str, field: str) -> Any:
    """Read an optional nested result field."""
    value = payload.get(section)
    return value.get(field) if isinstance(value, dict) else None


def build_row(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten one raw result into a stable summary row."""
    throughput_mean = nested(payload, "performance", "tokens_per_second_mean")
    throughput_std = nested(payload, "performance", "tokens_per_second_std")
    comparable = bool(nested(payload, "runtime", "system_metrics_comparable"))
    quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
    downstream = quality.get("downstream") if isinstance(quality.get("downstream"), dict) else {}
    hellaswag = downstream.get("hellaswag") if isinstance(downstream.get("hellaswag"), dict) else {}
    calibration = payload.get("calibration") if isinstance(payload.get("calibration"), dict) else {}
    return {
        "schema_version": payload.get("schema_version"),
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "experiment_id": payload["experiment_id"],
        "method": payload["method"],
        "group_size": payload.get("group_size"),
        "bits": payload["bits"],
        "duo_scaling": calibration.get("duo_scaling"),
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
        "hellaswag_acc": hellaswag.get("acc"),
        "hellaswag_acc_stderr": hellaswag.get("acc_stderr"),
        "hellaswag_acc_norm": hellaswag.get("acc_norm"),
        "hellaswag_acc_norm_stderr": hellaswag.get("acc_norm_stderr"),
        "hellaswag_n_samples": hellaswag.get("n_samples"),
        "hellaswag_harness_version": hellaswag.get("harness_version"),
        "hellaswag_task_version": hellaswag.get("task_version"),
        "hellaswag_dataset_fingerprint": hellaswag.get("dataset_fingerprint"),
        "hellaswag_tokenizer_fingerprint": hellaswag.get("tokenizer_fingerprint"),
        "downstream_quality_comparable": bool(hellaswag.get("quality_comparable")),
        "provenance_status": nested(payload, "provenance", "status"),
        "environment_python": nested(payload, "environment", "python"),
        "environment_torch": nested(payload, "environment", "torch"),
        "environment_cuda_runtime": nested(payload, "environment", "cuda_runtime"),
        "environment_gpu": nested(payload, "environment", "gpu_name"),
        "environment_git_commit": nested(payload, "environment", "git_commit"),
    }


def format_number(value: Any, digits: int = 2) -> str:
    """Format optional numeric values for the Markdown report."""
    return "—" if value is None else f"{float(value):.{digits}f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render the compact human-readable v1 benchmark table."""
    lines = [
        "# Unified quantization benchmark v1.2",
        "",
        "| Configuration | G | PPL | ΔPPL | CKPT MiB | Resident MiB | Peak MiB | System | tok/s mean ± std |",
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
                    experiment_label(row),
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
    downstream_rows = [row for row in rows if row["hellaswag_acc"] is not None]
    lines.extend(
        [
            "## HellaSwag downstream quality",
            "",
            "| Configuration | PPL | ΔPPL | HS acc ± stderr | HS acc_norm ± stderr | n |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in downstream_rows:
        acc = f"{float(row['hellaswag_acc']) * 100:.2f}% ± {float(row['hellaswag_acc_stderr']) * 100:.2f} pp"
        acc_norm = f"{float(row['hellaswag_acc_norm']) * 100:.2f}% ± {float(row['hellaswag_acc_norm_stderr']) * 100:.2f} pp"
        lines.append(
            f"| {experiment_label(row)} | {format_number(row['ppl'], 4)} | "
            f"{format_number(row['ppl_delta'], 4)} | {acc} | {acc_norm} | "
            f"{row['hellaswag_n_samples']} |"
        )
    conclusions = automatic_conclusions(rows)
    lines.extend(["", "## Automatic factual conclusions", ""])
    lines.extend(f"- {conclusion}" for conclusion in conclusions)
    lines.append("")
    return "\n".join(lines)


def experiment_label(row: dict[str, Any]) -> str:
    """Return a compact human-readable experiment label."""
    special = {
        "awq_g64_duo_false": "AWQ G64 duo=False",
        "awq_g64_duo_true": "AWQ G64 duo=True",
    }
    if row["experiment_id"] in special:
        return special[row["experiment_id"]]
    group_size = row["group_size"]
    return row["method"] if group_size is None else f"{row['method']} G{group_size}"


def automatic_conclusions(rows: list[dict[str, Any]]) -> list[str]:
    """Generate arithmetic-only quality statements without causal claims."""
    baseline = next(
        (
            row
            for row in rows
            if row["experiment_id"] == "fp16" and row["hellaswag_acc_norm"] is not None
        ),
        None,
    )
    if baseline is None:
        return []
    baseline_acc_norm = float(baseline["hellaswag_acc_norm"])
    conclusions: list[str] = []
    downstream_rows = [
        row
        for row in rows
        if row["experiment_id"] != "fp16"
        and row["hellaswag_acc_norm"] is not None
        and row["ppl_relative_increase_percent"] is not None
    ]
    for row in downstream_rows:
        delta_pp = (float(row["hellaswag_acc_norm"]) - baseline_acc_norm) * 100.0
        conclusions.append(
            f"{experiment_label(row)} increased PPL by "
            f"{float(row['ppl_relative_increase_percent']):.2f}% relative to FP16; "
            f"HellaSwag acc_norm changed by {delta_pp:+.2f} percentage points."
        )
    if downstream_rows and all(
        float(row["ppl_delta"]) > 0
        and float(row["hellaswag_acc_norm"]) < baseline_acc_norm
        for row in downstream_rows
    ):
        conclusions.append(
            "Higher PPL was accompanied by lower HellaSwag acc_norm in these evaluated configurations."
        )
    return conclusions


def provenance_label(row: dict[str, Any]) -> str:
    """Return a short provenance label for the visual report."""
    return "v1 rerun" if row["provenance_status"] == "v1_rerun" else "legacy frozen"


def chart_row(
    row: dict[str, Any],
    *,
    width_percent: float | None,
    value_text: str,
) -> str:
    """Render one responsive horizontal chart row."""
    method_class = str(row["method"]).lower()
    provenance = provenance_label(row)
    legacy_class = " legacy" if provenance != "v1 rerun" else ""
    if width_percent is None:
        track = '<div class="bar-track unavailable"><span>N/C</span></div>'
    else:
        bounded_width = min(max(width_percent, 0.0), 100.0)
        track = (
            '<div class="bar-track">'
            f'<div class="bar-fill {method_class}" style="width:{bounded_width:.3f}%"></div>'
            "</div>"
        )
    return (
        f'<div class="chart-row{legacy_class}">'
        '<div class="row-name">'
        f'<span>{escape(experiment_label(row))}</span>'
        f'<small>{escape(provenance)}</small>'
        "</div>"
        f"{track}"
        f'<div class="row-value">{escape(value_text)}</div>'
        "</div>"
    )


def quality_chart(rows: list[dict[str, Any]]) -> str:
    """Render PPL and delta-to-FP16 as a visual bar group."""
    deltas = [float(row["ppl_delta"]) for row in rows if row["ppl_delta"] is not None]
    maximum = max(deltas, default=1.0) or 1.0
    output: list[str] = []
    for row in rows:
        ppl = row["ppl"]
        delta = row["ppl_delta"]
        if ppl is None or delta is None:
            output.append(chart_row(row, width_percent=None, value_text="N/C"))
            continue
        percent = float(delta) / maximum * 100.0
        if float(delta) == 0.0:
            percent = 1.2
            value_text = f"PPL {float(ppl):.4f} · baseline"
        else:
            value_text = f"PPL {float(ppl):.4f} · Δ +{float(delta):.4f}"
        output.append(chart_row(row, width_percent=percent, value_text=value_text))
    return "\n".join(output)


def system_chart(
    rows: list[dict[str, Any]],
    *,
    field: str,
    unit: str,
    digits: int,
    include_std: bool = False,
) -> str:
    """Render one system metric while preserving comparability boundaries."""
    values = [
        float(row[field])
        for row in rows
        if row[field] is not None and row["system_metrics_comparable"]
    ]
    maximum = max(values, default=1.0) or 1.0
    output: list[str] = []
    for row in rows:
        value = row[field]
        if value is None or not row["system_metrics_comparable"]:
            output.append(chart_row(row, width_percent=None, value_text="N/C"))
            continue
        value_text = f"{float(value):.{digits}f} {unit}"
        if include_std and row["tokens_per_second_std"] is not None:
            value_text = (
                f"{float(value):.{digits}f} ± "
                f"{float(row['tokens_per_second_std']):.{digits}f} {unit}"
            )
        output.append(
            chart_row(
                row,
                width_percent=float(value) / maximum * 100.0,
                value_text=value_text,
            )
        )
    return "\n".join(output)


def html_table_rows(rows: list[dict[str, Any]]) -> str:
    """Render exact values below the visual comparisons."""
    output: list[str] = []
    for row in rows:
        if row["tokens_per_second_mean"] is None:
            throughput = "—"
        elif row["tokens_per_second_std"] is None:
            throughput = f"{float(row['tokens_per_second_mean']):.2f} ± —"
        else:
            throughput = (
                f"{float(row['tokens_per_second_mean']):.2f} ± "
                f"{float(row['tokens_per_second_std']):.2f}"
            )
        system = "comparable" if row["system_metrics_comparable"] else "N/C"
        hs_acc = (
            "—"
            if row["hellaswag_acc"] is None
            else f"{float(row['hellaswag_acc']) * 100:.2f}% ± {float(row['hellaswag_acc_stderr']) * 100:.2f} pp"
        )
        hs_acc_norm = (
            "—"
            if row["hellaswag_acc_norm"] is None
            else f"{float(row['hellaswag_acc_norm']) * 100:.2f}% ± {float(row['hellaswag_acc_norm_stderr']) * 100:.2f} pp"
        )
        output.append(
            "<tr>"
            f"<td>{escape(experiment_label(row))}</td>"
            f"<td>{escape(format_number(row['ppl'], 4))}</td>"
            f"<td>{escape(format_number(row['ppl_delta'], 4))}</td>"
            f"<td>{escape(hs_acc)}</td>"
            f"<td>{escape(hs_acc_norm)}</td>"
            f"<td>{escape(format_number(row['checkpoint_mib'], 2))}</td>"
            f"<td>{escape(format_number(row['peak_vram_mib'], 2))}</td>"
            f"<td>{escape(throughput)}</td>"
            f"<td>{escape(system)}</td>"
            f"<td>{escape(provenance_label(row))}</td>"
            "</tr>"
        )
    return "\n".join(output)


def html_dashboard(
    rows: list[dict[str, Any]],
    *,
    generated_at_utc: str,
    generation_protocol: dict[str, Any],
) -> str:
    """Build a self-contained responsive benchmark dashboard."""
    quantized = [row for row in rows if int(row["bits"]) < 16 and row["ppl"] is not None]
    best_quantized = min(quantized, key=lambda row: float(row["ppl"]))
    gptq_g64 = next((row for row in rows if row["experiment_id"] == "gptq_g64"), None)
    if gptq_g64 is None or gptq_g64["tokens_per_second_mean"] is None:
        gptq_speed = "—"
    else:
        gptq_speed = f"{float(gptq_g64['tokens_per_second_mean']):.2f} tok/s"

    warmup_runs = generation_protocol.get("warmup_runs", "—")
    benchmark_runs = generation_protocol.get("benchmark_runs", "—")
    max_new_tokens = generation_protocol.get("max_new_tokens", "—")
    conclusions = automatic_conclusions(rows)
    conclusions_html = "".join(f"<li>{escape(item)}</li>" for item in conclusions)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quantization benchmark v1.2</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f5f7fb;
      --surface: #ffffff;
      --surface-soft: #edf1f7;
      --text: #172033;
      --muted: #657089;
      --line: #d8deea;
      --fp16: #2878ff;
      --gptq: #7357d9;
      --awq: #e08a1e;
      --success: #16845b;
      --shadow: 0 12px 32px rgba(29, 43, 76, 0.08);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0f1420;
        --surface: #171e2c;
        --surface-soft: #20293a;
        --text: #edf1f8;
        --muted: #a8b2c7;
        --line: #303b50;
        --fp16: #6aa0ff;
        --gptq: #a48af4;
        --awq: #f1ab52;
        --success: #56c99a;
        --shadow: 0 12px 32px rgba(0, 0, 0, 0.24);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }}
    header {{ display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; margin-bottom: 22px; }}
    h1 {{ margin: 4px 0 6px; font-size: clamp(28px, 4vw, 44px); line-height: 1.08; }}
    h2 {{ margin: 0; font-size: 18px; }}
    p {{ margin: 0; color: var(--muted); }}
    .eyebrow {{ color: var(--muted); font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; }}
    .status {{ color: var(--success); border: 1px solid currentColor; border-radius: 999px; padding: 7px 12px; font-size: 12px; font-weight: 700; letter-spacing: 0.08em; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .stat, .panel {{ background: var(--surface); border: 1px solid var(--line); border-radius: 16px; box-shadow: var(--shadow); }}
    .stat {{ padding: 16px; }}
    .stat-label {{ color: var(--muted); font-size: 12px; margin-bottom: 7px; }}
    .stat-value {{ font-size: 22px; font-weight: 750; font-variant-numeric: tabular-nums; }}
    .stat-note {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .panel {{ padding: 18px; min-width: 0; }}
    .panel.wide {{ grid-column: 1 / -1; }}
    .panel-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 16px; }}
    .panel-head span {{ color: var(--muted); font-size: 12px; }}
    .chart {{ display: grid; gap: 11px; }}
    .chart-row {{ display: grid; grid-template-columns: 112px minmax(90px, 1fr) 174px; align-items: center; gap: 10px; }}
    .row-name {{ min-width: 0; }}
    .row-name span {{ display: block; font-weight: 650; white-space: nowrap; }}
    .row-name small {{ display: block; color: var(--muted); margin-top: 2px; }}
    .bar-track {{ height: 18px; background: var(--surface-soft); border-radius: 5px; overflow: hidden; position: relative; }}
    .bar-fill {{ height: 100%; min-width: 3px; border-radius: inherit; }}
    .bar-fill.fp16 {{ background: var(--fp16); }}
    .bar-fill.gptq {{ background: var(--gptq); }}
    .bar-fill.awq {{ background: var(--awq); }}
    .legacy .bar-fill {{ opacity: 0.52; }}
    .bar-track.unavailable {{ display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 11px; letter-spacing: 0.08em; }}
    .row-value {{ text-align: right; font-size: 13px; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; color: var(--muted); font-size: 12px; margin-top: 14px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
    .dot.fp16 {{ background: var(--fp16); }}
    .dot.gptq {{ background: var(--gptq); }}
    .dot.awq {{ background: var(--awq); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 9px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    footer {{ color: var(--muted); font-size: 12px; margin-top: 16px; text-align: center; }}
    @media (max-width: 880px) {{
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid {{ grid-template-columns: minmax(0, 1fr); }}
      .panel.wide {{ grid-column: auto; }}
    }}
    @media (max-width: 620px) {{
      main {{ padding: 22px 12px 36px; }}
      header {{ align-items: center; }}
      .stats {{ grid-template-columns: minmax(0, 1fr); }}
      .chart-row {{ grid-template-columns: 90px minmax(72px, 1fr); }}
      .row-value {{ grid-column: 2; text-align: left; white-space: normal; }}
      .panel {{ padding: 15px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="eyebrow">Unified quantization evaluation</div>
        <h1>Benchmark v1.2</h1>
        <p>PPL, HellaSwag downstream quality, throughput, memory, and comparability.</p>
      </div>
      <div class="status">VALIDATED</div>
    </header>

    <section class="stats" aria-label="Benchmark highlights">
      <article class="stat">
        <div class="stat-label">Configurations</div>
        <div class="stat-value">{len(rows)}</div>
        <div class="stat-note">FP16 · GPTQ · AWQ</div>
      </article>
      <article class="stat">
        <div class="stat-label">Best 4-bit PPL</div>
        <div class="stat-value">{float(best_quantized['ppl']):.4f}</div>
        <div class="stat-note">{escape(experiment_label(best_quantized))}</div>
      </article>
      <article class="stat">
        <div class="stat-label">GPTQ G64 throughput</div>
        <div class="stat-value">{escape(gptq_speed)}</div>
        <div class="stat-note">validated system rerun</div>
      </article>
      <article class="stat">
        <div class="stat-label">System protocol</div>
        <div class="stat-value">{escape(str(warmup_runs))} + {escape(str(benchmark_runs))}</div>
        <div class="stat-note">warmups + runs · {escape(str(max_new_tokens))} tokens</div>
      </article>
    </section>

    <section class="grid">
      <article class="panel wide">
        <div class="panel-head"><h2>Quality loss vs FP16</h2><span>Shorter is better</span></div>
        <div class="chart">{quality_chart(rows)}</div>
        <div class="legend">
          <span><i class="dot fp16"></i>FP16</span>
          <span><i class="dot gptq"></i>GPTQ</span>
          <span><i class="dot awq"></i>AWQ</span>
          <span>Muted bars = legacy frozen system rows</span>
        </div>
      </article>

      <article class="panel">
        <div class="panel-head"><h2>Generation throughput</h2><span>Higher is better</span></div>
        <div class="chart">{system_chart(rows, field='tokens_per_second_mean', unit='tok/s', digits=2, include_std=True)}</div>
      </article>

      <article class="panel">
        <div class="panel-head"><h2>Peak allocated VRAM</h2><span>Lower is better</span></div>
        <div class="chart">{system_chart(rows, field='peak_vram_mib', unit='MiB', digits=2)}</div>
      </article>

      <article class="panel wide">
        <div class="panel-head"><h2>Exact results</h2><span>N/C = system representation not comparable</span></div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>Configuration</th><th>PPL</th><th>ΔPPL</th><th>HS acc</th><th>HS acc_norm</th><th>Checkpoint MiB</th><th>Peak MiB</th><th>tok/s mean ± std</th><th>System</th><th>Provenance</th></tr>
            </thead>
            <tbody>{html_table_rows(rows)}</tbody>
          </table>
        </div>
      </article>
      <article class="panel wide">
        <div class="panel-head"><h2>Automatic factual conclusions</h2><span>Percentage-point changes; no causal claims</span></div>
        <ul>{conclusions_html}</ul>
      </article>
    </section>

    <footer>Generated {escape(generated_at_utc)} from benchmark/results/raw/*.json</footer>
  </main>
</body>
</html>
"""


def main() -> None:
    if not RAW_RESULTS_DIR.is_dir():
        raise FileNotFoundError(f"Raw result directory not found: {RAW_RESULTS_DIR}")
    payloads = load_raw_payloads(RAW_RESULTS_DIR)
    rows = sorted(
        (build_row(payload) for payload in payloads),
        key=lambda row: (ORDER.get(row["experiment_id"], 999), row["experiment_id"]),
    )
    if not rows:
        raise RuntimeError("No unified raw results were found.")

    validation = validate_payloads(payloads)
    for warning in validation["warnings"]:
        print(f"validation_warning={warning}")
    if validation["status"] != "passed":
        for error in validation["errors"]:
            print(f"validation_error={error}")
        raise RuntimeError("Benchmark validation failed; summaries and figures were not regenerated.")

    generated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    generation_protocol = payloads[0].get("protocol", {}).get("generation", {})
    if not isinstance(generation_protocol, dict):
        generation_protocol = {}

    SUMMARY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = SUMMARY_RESULTS_DIR / "benchmark_v1_summary.json"
    csv_path = SUMMARY_RESULTS_DIR / "benchmark_v1_summary.csv"
    markdown_path = SUMMARY_RESULTS_DIR / "benchmark_v1_summary.md"
    html_path = SUMMARY_RESULTS_DIR / "benchmark_v1_summary.html"
    figure_paths = generate_figures(rows, FIGURES_RESULTS_DIR)
    json_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "generated_at_utc": generated_at_utc,
                "validation": validation,
                "figures": [path.name for path in figure_paths],
                "conclusions": automatic_conclusions(rows),
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
    html_path.write_text(
        html_dashboard(
            rows,
            generated_at_utc=generated_at_utc,
            generation_protocol=generation_protocol,
        ),
        encoding="utf-8",
    )
    print(f"summary_json={json_path}")
    print(f"summary_csv={csv_path}")
    print(f"summary_markdown={markdown_path}")
    print(f"summary_html={html_path}")
    print(f"figures_dir={FIGURES_RESULTS_DIR}")


if __name__ == "__main__":
    main()
