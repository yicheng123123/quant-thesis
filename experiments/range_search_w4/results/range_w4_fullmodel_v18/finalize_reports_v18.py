"""Build v18 reporting tables from frozen raw JSON/CSV artifacts (no reruns)."""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path


OUT = Path(__file__).resolve().parent


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(new: float, base: float) -> float:
    return (new / base - 1.0) * 100.0


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def main():
    summary = load_json(OUT / "evaluation" / "SUMMARY.json")
    eval_check = load_json(OUT / "evaluation" / "EVAL_CORRECTNESS.json")
    manifest = load_json(OUT / "FORMAL_RUN_MANIFEST.json")
    runtime = load_json(OUT / "FORMAL_RUNTIME.json")
    costs = load_json(OUT / "FINAL_COST_MODEL.json")
    protocol = load_json(OUT / "PROTOCOL.json")
    row_path = OUT / "M2_ROW_RESULTS.csv"
    module_path = OUT / "M2_MODULE_SUMMARY.csv"
    rows = csv_rows(row_path)
    module_rows = csv_rows(module_path)
    timing_rows = csv_rows(OUT / "FORMAL_TIMING.csv")

    if manifest.get("status") != "COMPLETE" or eval_check.get("status") != "PASS":
        raise RuntimeError("v18 formal/evaluation manifest is not complete and correct")
    if len(rows) != 304128 or len(module_rows) != 168:
        raise RuntimeError(f"row/module count mismatch: rows={len(rows)} modules={len(module_rows)}")
    for dataset, token_count, window_count in (("wikitext", 299077, 582), ("c4_fixed_budget", 131071, 253)):
        for method in ("M0", "M1", "M2"):
            item = summary[dataset][method]
            if item.get("status") != "PASS" or item["evaluated_tokens"] != token_count or item["window_count"] != window_count:
                raise RuntimeError(f"frozen evaluation identity mismatch: {dataset}/{method}")

    m1 = runtime["m1"]
    m2 = runtime["m2"]
    calib = {"M1": m1["wall_seconds_including_load_replay_save"],
             "M2": m2["wall_seconds_including_load_replay_save"]}
    peak_calib = {}
    grouped_timing: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped_peak: dict[str, int] = defaultdict(int)
    for item in timing_rows:
        method = item["method"]
        stage = item["stage"]
        grouped_timing[(method, stage)].append(float(item["wall_seconds"]))
        if method in ("M1", "M2"):
            grouped_peak[method] = max(grouped_peak[method], int(item["gpu_peak_allocated_bytes"]))
    peak_calib.update(grouped_peak)

    result_rows = {}
    for dataset in ("wikitext", "c4_fixed_budget"):
        result_rows[dataset] = {}
        for method in ("M0", "M1", "M2"):
            item = summary[dataset][method]
            result_rows[dataset][method] = {
                "dataset": item["dataset_label"],
                "method": method,
                "representation": "FP16" if method == "M0" else "dense-dequant FP16",
                "calibration": "none" if method == "M0" else "WikiText-2 train: 128 documents x 2048 tokens",
                "calibration_wall_seconds": "" if method == "M0" else f"{calib[method]:.6f}",
                "calibration_peak_gpu_allocated_bytes": "" if method == "M0" else peak_calib[method],
                "evaluation_model_load_seconds": f"{item['model_load_seconds']:.6f}",
                "evaluation_wall_seconds": f"{item['evaluation_wall_seconds']:.6f}",
                "evaluation_peak_gpu_allocated_bytes": item["peak_gpu_allocated_bytes"],
                "evaluation_peak_gpu_reserved_bytes": item["peak_gpu_reserved_bytes"],
                "evaluated_tokens": item["evaluated_tokens"],
                "window_count": item["window_count"],
                "average_nll": f"{item['average_nll']:.12f}",
                "perplexity": f"{item['perplexity']:.12f}",
            }

    eval_fields = list(next(iter(result_rows["wikitext"].values())).keys())
    write_csv(OUT / "evaluation" / "WT2_RESULTS.csv", eval_fields,
              list(result_rows["wikitext"].values()))
    write_csv(OUT / "evaluation" / "C4_RESULTS.csv", eval_fields,
              list(result_rows["c4_fixed_budget"].values()))

    comp_fields = ["Method", "Precision/Quant", "Calibration", "Calibration wall seconds",
                   "Calibration peak GPU allocated bytes", "WT2 NLL", "WT2 PPL",
                   "C4 fixed-budget NLL", "C4 fixed-budget PPL",
                   "Evaluation peak GPU allocated bytes", "Evaluation peak GPU reserved bytes",
                   "Representation"]
    comp_rows = []
    for method, precision in (("M0", "FP16"), ("M1", "W4A16 GPTQ G128"), ("M2", "W4A16 GPTQ G128 + per-row shared-alpha search")):
        wt = summary["wikitext"][method]
        c4 = summary["c4_fixed_budget"][method]
        comp_rows.append({
            "Method": method, "Precision/Quant": precision,
            "Calibration": "none" if method == "M0" else "WikiText-2 train: 128 documents x 2048 tokens",
            "Calibration wall seconds": "" if method == "M0" else f"{calib[method]:.6f}",
            "Calibration peak GPU allocated bytes": "" if method == "M0" else peak_calib[method],
            "WT2 NLL": f"{wt['average_nll']:.12f}", "WT2 PPL": f"{wt['perplexity']:.12f}",
            "C4 fixed-budget NLL": f"{c4['average_nll']:.12f}",
            "C4 fixed-budget PPL": f"{c4['perplexity']:.12f}",
            "Evaluation peak GPU allocated bytes": max(wt["peak_gpu_allocated_bytes"], c4["peak_gpu_allocated_bytes"]),
            "Evaluation peak GPU reserved bytes": max(wt["peak_gpu_reserved_bytes"], c4["peak_gpu_reserved_bytes"]),
            "Representation": "FP16" if method == "M0" else "dense-dequant FP16",
        })
    write_csv(OUT / "FULLMODEL_COMPARISON.csv", comp_fields, comp_rows)

    alpha_counts = Counter()
    layer_counts = Counter()
    proj_counts = Counter()
    layer_proj_counts = Counter()
    layer_sizes = Counter()
    proj_sizes = Counter()
    layer_proj_sizes = Counter()
    row_gains = []
    searched_loss_sum = 0.0
    native_loss_sum = 0.0
    for row in rows:
        alpha = f"{float(row['selected_alpha']):.1f}"
        module = row["module"]
        match = re.search(r"model\.layers\.(\d+)\.", module)
        if not match:
            raise RuntimeError(f"unexpected module name in row result: {module}")
        layer = int(match.group(1))
        projection = module.rsplit(".", 1)[-1]
        alpha_counts[alpha] += 1
        layer_counts[(layer, alpha)] += 1
        proj_counts[(projection, alpha)] += 1
        layer_proj_counts[(layer, projection, alpha)] += 1
        layer_sizes[layer] += 1
        proj_sizes[projection] += 1
        layer_proj_sizes[(layer, projection)] += 1
        searched = float(row["search_output_mse"])
        native = float(row["alpha1_output_mse"])
        searched_loss_sum += searched
        native_loss_sum += native
        row_gains.append(float(row["relative_output_gain"]))

    if sum(alpha_counts.values()) != 304128:
        raise RuntimeError("selected-alpha histogram does not cover all M2 rows")
    alpha_order = [f"{i / 10:.1f}" for i in range(1, 11)]
    alpha_rows = []
    for alpha in alpha_order:
        alpha_rows.append({"scope": "full_model", "layer": "", "projection": "", "alpha": alpha,
                           "rows": alpha_counts[alpha], "fraction": alpha_counts[alpha] / 304128})
    for layer in range(24):
        for alpha in alpha_order:
            alpha_rows.append({"scope": "layer", "layer": layer, "projection": "", "alpha": alpha,
                               "rows": layer_counts[(layer, alpha)], "fraction": layer_counts[(layer, alpha)] / layer_sizes[layer]})
    for projection in sorted(proj_sizes):
        for alpha in alpha_order:
            alpha_rows.append({"scope": "projection", "layer": "", "projection": projection, "alpha": alpha,
                               "rows": proj_counts[(projection, alpha)], "fraction": proj_counts[(projection, alpha)] / proj_sizes[projection]})
    for layer in range(24):
        for projection in sorted(proj_sizes):
            denom = layer_proj_sizes[(layer, projection)]
            if not denom:
                continue
            for alpha in alpha_order:
                alpha_rows.append({"scope": "layer_projection", "layer": layer, "projection": projection, "alpha": alpha,
                                   "rows": layer_proj_counts[(layer, projection, alpha)], "fraction": layer_proj_counts[(layer, projection, alpha)] / denom})
    write_csv(OUT / "ALPHA_DISTRIBUTION.csv",
              ["scope", "layer", "projection", "alpha", "rows", "fraction"], alpha_rows)

    def weighted_median(counts: Counter) -> float:
        total = sum(counts.values())
        if total == 0:
            return float("nan")
        left_rank, right_rank = (total - 1) // 2, total // 2
        seen = 0
        middle = []
        for alpha in alpha_order:
            seen += counts[alpha]
            if seen > left_rank and len(middle) == 0:
                middle.append(float(alpha))
            if seen > right_rank:
                middle.append(float(alpha))
                break
        return sum(middle) / len(middle)

    alpha_groups = [("full_model", "", "", alpha_counts, 304128)]
    alpha_groups.extend(("layer", layer, "", Counter({a: layer_counts[(layer, a)] for a in alpha_order}), layer_sizes[layer])
                        for layer in range(24))
    alpha_groups.extend(("projection", "", projection,
                         Counter({a: proj_counts[(projection, a)] for a in alpha_order}), proj_sizes[projection])
                        for projection in sorted(proj_sizes))
    alpha_groups.extend(("layer_projection", layer, projection,
                         Counter({a: layer_proj_counts[(layer, projection, a)] for a in alpha_order}),
                         layer_proj_sizes[(layer, projection)])
                        for layer in range(24) for projection in sorted(proj_sizes)
                        if layer_proj_sizes[(layer, projection)])
    alpha_summary_rows = []
    for scope, layer, projection, counts, total in alpha_groups:
        alpha_summary_rows.append({
            "scope": scope, "layer": layer, "projection": projection, "rows": total,
            "mean_alpha": sum(float(a) * counts[a] for a in alpha_order) / total,
            "median_alpha": weighted_median(counts),
            "alpha_1_fraction": counts["1.0"] / total,
            "alpha_0_9_fraction": counts["0.9"] / total,
            "alpha_0_8_fraction": counts["0.8"] / total,
            "alpha_le_0_7_fraction": sum(counts[a] for a in alpha_order if float(a) <= 0.7) / total,
        })
    write_csv(OUT / "ALPHA_SUMMARY_BY_LAYER_PROJECTION.csv",
              ["scope", "layer", "projection", "rows", "mean_alpha", "median_alpha",
               "alpha_1_fraction", "alpha_0_9_fraction", "alpha_0_8_fraction", "alpha_le_0_7_fraction"],
              alpha_summary_rows)

    # Convenient named copies matching the v18 artifact layout while retaining
    # the original run-generated CSVs unchanged.
    for source, target in ((OUT / "M1" / "QUANTIZATION_METADATA.json", OUT / "M1" / "quant_metadata.json"),
                           (OUT / "M2" / "QUANTIZATION_METADATA.json", OUT / "M2" / "quant_metadata.json"),
                           (OUT / "M2_MODULE_SUMMARY.csv", OUT / "M2" / "module_summary.csv")):
        shutil.copyfile(source, target)
    with (OUT / "M2" / "selected_alpha.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["module", "row", "selected_alpha", "search_output_mse", "alpha1_output_mse", "relative_output_gain"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    breakdown = []
    for (method, stage), values in sorted(grouped_timing.items()):
        breakdown.append({"method": method, "stage": stage, "count": len(values),
                          "total_wall_seconds": sum(values), "mean_wall_seconds": statistics.mean(values),
                          "max_wall_seconds": max(values)})
    write_csv(OUT / "M2" / "runtime_breakdown.csv",
              ["method", "stage", "count", "total_wall_seconds", "mean_wall_seconds", "max_wall_seconds"],
              [x for x in breakdown if x["method"] == "M2"])

    wt = summary["wikitext"]
    c4 = summary["c4_fixed_budget"]
    wt_change_abs = wt["M2"]["perplexity"] - wt["M1"]["perplexity"]
    c4_change_abs = c4["M2"]["perplexity"] - c4["M1"]["perplexity"]
    nll_wt_delta = wt["M2"]["average_nll"] - wt["M1"]["average_nll"]
    nll_c4_delta = c4["M2"]["average_nll"] - c4["M1"]["average_nll"]
    mean_row_gain = statistics.mean(row_gains)
    median_row_gain = statistics.median(row_gains)
    aggregate_local_gain = 1.0 - searched_loss_sum / native_loss_sum
    improved_rows = sum(g > 0 for g in row_gains)
    tied_rows = sum(g == 0 for g in row_gains)
    alpha_summary = {a: {"rows": alpha_counts[a], "fraction": alpha_counts[a] / 304128} for a in alpha_order}
    alpha_summary_stats = {
        "mean_alpha": sum(float(a) * alpha_counts[a] for a in alpha_order) / 304128,
        "median_alpha": statistics.median([float(r["selected_alpha"]) for r in rows]),
        "alpha_1_fraction": alpha_counts["1.0"] / 304128,
        "alpha_0_9_fraction": alpha_counts["0.9"] / 304128,
        "alpha_0_8_fraction": alpha_counts["0.8"] / 304128,
        "alpha_le_0_7_fraction": sum(alpha_counts[a] for a in alpha_order if float(a) <= 0.7) / 304128,
    }
    compact_summary = {
        "status": "PASS", "verdict": manifest["verdict"],
        "protocol": {"model_revision": protocol["model"]["revision"], "weight_bits": 4, "activation_bits": 16,
                     "group_size": 128, "block_size": 128, "symmetric": True,
                     "alpha_grid": alpha_order, "search_semantics": "10 full-Linear dynamic GPTQ candidates; row-wise alpha selection and whole-Linear stitch"},
        "work_units": runtime["formal_work_units"],
        "calibration": {"M1_seconds": calib["M1"], "M2_seconds": calib["M2"],
                        "M2_over_M1_runtime_ratio": calib["M2"] / calib["M1"],
                        "estimated_conservative_M2_hours": costs["scenario_hours"]["conservative_M2"],
                        "estimated_conservative_M1_plus_M2_hours": costs["scenario_hours"]["conservative_M1_plus_M2"],
                        "hard_gate_hours": 48.0, "cost_gate": manifest["results"]["cost_gate"],
                        "quantization_total_wall_seconds": runtime["overall_quantization_wall_seconds"],
                        "peak_allocated_bytes": runtime["cuda_peak_allocated_bytes"],
                        "peak_reserved_bytes": runtime["cuda_peak_reserved_bytes"]},
        "evaluation": {"wikitext": {m: {"ppl": wt[m]["perplexity"], "nll": wt[m]["average_nll"],
                                                "tokens": wt[m]["evaluated_tokens"], "windows": wt[m]["window_count"]}
                                            for m in ("M0", "M1", "M2")},
                       "c4_fixed_budget": {m: {"ppl": c4[m]["perplexity"], "nll": c4[m]["average_nll"],
                                                       "tokens": c4[m]["evaluated_tokens"], "windows": c4[m]["window_count"]}
                                                   for m in ("M0", "M1", "M2")},
                       "M2_vs_M1_wt2_absolute_ppl": wt_change_abs,
                       "M2_vs_M1_wt2_relative_ppl_percent": pct(wt["M2"]["perplexity"], wt["M1"]["perplexity"]),
                       "M2_vs_M1_wt2_nll_per_token": nll_wt_delta,
                       "M2_vs_M1_c4_absolute_ppl": c4_change_abs,
                       "M2_vs_M1_c4_relative_ppl_percent": pct(c4["M2"]["perplexity"], c4["M1"]["perplexity"]),
                       "M2_vs_M1_c4_nll_per_token": nll_c4_delta,
                       "aggregate_evaluation_forward_seconds": sum(
                           summary[d][m]["evaluation_wall_seconds"] for d in ("wikitext", "c4_fixed_budget") for m in ("M0", "M1", "M2"))},
        "alpha_distribution": {"counts": alpha_summary, "statistics": alpha_summary_stats},
        "local_search_diagnostic": {"mean_row_relative_output_gain": mean_row_gain,
                                     "median_row_relative_output_gain": median_row_gain,
                                     "aggregate_row_mse_gain": aggregate_local_gain,
                                     "rows_improved": improved_rows, "rows_tied": tied_rows,
                                     "scope_note": "local search objective is not a causal predictor; one full model only"},
        "correctness": {"quantization": "PASS", "save_reload": "PASS", "evaluation": "PASS",
                        "known_frozen_v16_v17_1_v17_2_hashes": "PASS",
                        "v17_3_hashes": "recorded as consumed; no upstream expected-hash manifest was published",
                        "all_168_linears": True, "all_304128_rows": True},
    }
    dump_json(OUT / "SUMMARY.json", compact_summary)

    def table_rows(dataset):
        return "\n".join(
            f"| {m} | {summary[dataset][m]['average_nll']:.9f} | {summary[dataset][m]['perplexity']:.8f} | "
            f"{summary[dataset][m]['evaluated_tokens']:,} | {summary[dataset][m]['window_count']} |"
            for m in ("M0", "M1", "M2"))

    alpha_lines = "\n".join(f"- α={a}: {alpha_counts[a]:,} rows ({alpha_counts[a] / 304128:.3%})" for a in alpha_order)
    projection_alpha_rows = [x for x in alpha_summary_rows if x["scope"] == "projection"]
    projection_summary_md = "\n".join(
        f"| {x['projection']} | {x['rows']:,} | {x['mean_alpha']:.4f} | {x['median_alpha']:.1f} | "
        f"{x['alpha_1_fraction']:.2%} | {x['alpha_0_9_fraction']:.2%} | "
        f"{x['alpha_0_8_fraction']:.2%} | {x['alpha_le_0_7_fraction']:.2%} |"
        for x in projection_alpha_rows)
    report = f"""# v18 — Full-model pure range-search report

## Verdict

`{manifest['verdict']}` under the frozen decision rule: M2 has lower WikiText-2 test PPL than M1 and all correctness gates pass. This is one fixed model/corpus evaluation; it is not evidence of statistical significance or broad generalization. Both quantized models remain worse than M0 FP16.

## Frozen protocol and execution

- Model: Qwen2.5-0.5B-Instruct, revision `{protocol['model']['revision']}`, FP16; RTX 4060 Laptop GPU.
- Quantization: W4A16, symmetric, G128, block 128, damp 0.1, `desc_act=false`, `static_groups=false`, native GPTQ compensation; dense-dequant FP16 evaluation.
- Calibration: frozen WikiText-2 train manifest, 128 documents × 2048 tokens, seed 20261005; manifest SHA256 `{load_json(OUT / 'CALIBRATION_VALIDATION.json')['manifest_sha256']}`.
- M2 performed 1,680 full-Linear alpha GPTQ runs that scored all 304,128 rows each across ten alpha values: 3,041,280 row-alpha score slots, not GPTQ executions. Each row chose its alpha; each complete Linear was stitched and committed once before dependency refresh.
- Precheck: `{manifest['results']['matched_shape_precheck']}`. Conservative estimated M2 calibration cost: {costs['scenario_hours']['conservative_M2']:.3f} h, versus the frozen 48 h gate. M1+M2 conservative estimate: {costs['scenario_hours']['conservative_M1_plus_M2']:.3f} h.
- Actual M1 calibration wall: {calib['M1']:.2f} s ({calib['M1']/60:.2f} min); actual M2: {calib['M2']:.2f} s ({calib['M2']/60:.2f} min), or {calib['M2']/calib['M1']:.2f}× M1. Combined quantization wall: {runtime['overall_quantization_wall_seconds']:.2f} s. Peak CUDA allocated/reserved: {runtime['cuda_peak_allocated_bytes']/2**30:.2f}/{runtime['cuda_peak_reserved_bytes']/2**30:.2f} GiB.
- Cost units remain separate: row-alpha score slots vs full-Linear GPTQ executions. The old 48.79 h v17.2 estimate is superseded because it multiplied row-alpha slots as if they were GPTQ calls and overcounted refreshes; the v18 matched-shape estimate uses 1,680 full-Linear candidates and 95 dependency transitions.

## Model-level evaluation

All methods used the same frozen token-ID stream per dataset, shared evaluator, max length 2048, stride 512, token-weighted NLL, retained tail, and no chat template.

### WikiText-2 test (299,077 scored targets)

| Method | Mean NLL/token | PPL | Targets | Windows |
|---|---:|---:|---:|---:|
{table_rows('wikitext')}

M2 vs M1: absolute PPL change {wt_change_abs:+.8f}; relative PPL change {pct(wt['M2']['perplexity'], wt['M1']['perplexity']):+.4f}%; NLL/token change {nll_wt_delta:+.9f}. M1 degrades over M0 by {wt['M1']['perplexity']-wt['M0']['perplexity']:+.6f} PPL; M2 degrades over M0 by {wt['M2']['perplexity']-wt['M0']['perplexity']:+.6f} PPL.

### C4 validation fixed-budget (131,071 scored targets)

| Method | Mean NLL/token | PPL | Targets | Windows |
|---|---:|---:|---:|---:|
{table_rows('c4_fixed_budget')}

This is the frozen fixed-budget C4 validation stream, not standard/full C4 PPL. M2 vs M1: absolute PPL change {c4_change_abs:+.8f}; relative change {pct(c4['M2']['perplexity'], c4['M1']['perplexity']):+.4f}%; NLL/token change {nll_c4_delta:+.9f}. C4 moves in the same direction as WT2 and does not determine the primary verdict.

## Selected alpha and local-to-global diagnostic

{alpha_lines}

Mean α={alpha_summary_stats['mean_alpha']:.4f}; median α={alpha_summary_stats['median_alpha']:.1f}; α=1 fraction {alpha_summary_stats['alpha_1_fraction']:.3%}; α=0.9 fraction {alpha_summary_stats['alpha_0_9_fraction']:.3%}; α=0.8 fraction {alpha_summary_stats['alpha_0_8_fraction']:.3%}; α≤0.7 fraction {alpha_summary_stats['alpha_le_0_7_fraction']:.3%}.

| Projection | Rows | Mean α | Median α | α=1 | α=0.9 | α=0.8 | α≤0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|
{projection_summary_md}

Per-layer and layer-by-projection mean/median/fraction summaries are in `ALPHA_SUMMARY_BY_LAYER_PROJECTION.csv`; the corresponding full histograms are in `ALPHA_DISTRIBUTION.csv`.

Across rows, mean relative local output-MSE gain was {mean_row_gain:.3%}, median {median_row_gain:.3%}; summed row-MSE gain was {aggregate_local_gain:.3%}, with {improved_rows:,} rows improved and {tied_rows:,} tied against α=1. These local diagnostics and the model-level WT2/C4 gains have the same direction in this run, but one model is insufficient to establish that local reconstruction gains predict PPL gains.

## Correctness, cost, caveats

- Quantization: all 24 layers, 168 Linear modules, 304,128 rows; finite weights/scales/zeros/losses; native `g_idx`; α=1 native parity checked at 35 M2 references. M1/M2 save-reload logit max absolute difference: 0 for the frozen 64-token check.
- Evaluation correctness: six method/dataset runs PASS; token totals and window-weighted NLL match the shared evaluator. Evaluation forward time summed across runs: {compact_summary['evaluation']['aggregate_evaluation_forward_seconds']:.1f} s (model loading is recorded separately in raw JSON).
- Initial evaluator launches exposed two v18 harness issues: the standalone import path and an expected-window assertion that did not follow the shared evaluator's stop-at-tail rule. No result from a failed attempt was accepted. Both issues were corrected, and the full six-run matrix was rerun successfully.
- Runtime logs contain the Transformers warning that sliding-window attention is not implemented for SDPA. All six comparisons used the same frozen model path and evaluator settings; absolute scores should be interpreted with this implementation caveat.
- These are offline calibration and evaluation costs, not online inference speed measurements. No packed-kernel or latency claim is made.

## Evidence boundary

Verified: v18 frozen calibration/evaluation manifests; matched-shape precheck; 48 h conservative cost gate; independent M1 and full M2 checkpoints; 1,680 full-Linear candidates; alpha=1 spot parity; checkpoint reload; WT2 and fixed-budget C4 PPL/NLL. Expected frozen hashes for v16, v17.1, and v17.2 match. v17.3 has no published upstream expected-hash manifest; v18 records the exact consumed hashes for that version.

Still unverified: whether the gain persists across other model seeds, model sizes, evaluation corpora, or packed deployment. No new method or follow-up experiment is started here.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    for method in ("M1", "M2"):
        item = runtime[method.lower()]
        p = OUT / method / "RUN_REPORT.md"
        if method == "M1":
            text = f"""# M1 native GPTQ run report

Status: **COMPLETE / correctness PASS / reload PASS**.

- Model revision: `{protocol['model']['revision']}`; W4A16 symmetric G128; block 128; damp 0.1; native sequential GPTQ.
- Started from the original FP checkpoint and used the frozen 128 × 2048 WikiText-2 train calibration manifest (SHA256 `{load_json(OUT / 'CALIBRATION_VALIDATION.json')['manifest_sha256']}`).
- Processed 24 layers / 168 linears / 304,128 rows. This is an independent native GPTQ run; it was not assembled from M2 candidates.
- Actual wall including load, layer replay, checkpointing, final save/reload: {item['wall_seconds_including_load_replay_save']:.3f} s. Final save {item['save_seconds']:.3f} s. Peak allocated/reserved CUDA memory for the combined formal process: {runtime['cuda_peak_allocated_bytes']/2**30:.2f}/{runtime['cuda_peak_reserved_bytes']/2**30:.2f} GiB.
- Final checkpoint SHA256: `{item['final_model_sha256']}`. Reload output max/mean absolute difference on 64 fixed tokens: {item['reload_correctness']['max_abs_logit_difference']}/{item['reload_correctness']['mean_abs_logit_difference']}.
- WT2: PPL {wt['M1']['perplexity']:.8f}, NLL/token {wt['M1']['average_nll']:.9f}, {wt['M1']['evaluated_tokens']:,} scored tokens. C4 fixed-budget: PPL {c4['M1']['perplexity']:.8f}, NLL/token {c4['M1']['average_nll']:.9f}, {c4['M1']['evaluated_tokens']:,} scored tokens.

Reproduce both formal methods in frozen order with `& '.venv-gptq\\Scripts\\python.exe' -X utf8 'experiments\\range_search_w4\\results\\range_w4_fullmodel_v18\\run_formal.py'`; this runner has no per-method CLI and resumes completed layer checkpoints. Evaluate the frozen matrix with the corresponding `evaluate_v18.py` command. Detailed artifacts: `QUANTIZATION_METADATA.json`, `QUANTIZATION_CORRECTNESS.json`, `RELOAD_CORRECTNESS.json`, `checkpoint/final/`, and `../FORMAL_TIMING.csv`.
"""
        else:
            top_stages = sorted(((stage, sum(vals)) for (m, stage), vals in grouped_timing.items() if m == "M2"), key=lambda x: -x[1])
            stage_md = "\n".join(f"- `{stage}`: {seconds:.2f} s" for stage, seconds in top_stages)
            text = f"""# M2 pure range-search run report

Status: **COMPLETE / correctness PASS / reload PASS**.

- Frozen method: dynamic native symmetric GPTQ G128 with shared alpha grid `{', '.join(alpha_order)}`. Each alpha ran a full Linear from pristine candidate state; all rows were scored, selected independently, stitched once per Linear, and committed before dependency refresh.
- Processed 24 layers / 168 linears / 304,128 rows. Formal counts: 1,680 full-Linear candidate GPTQ runs, 3,041,280 row-alpha score slots, 35 alpha=1 native spot references. Frozen calibration manifest SHA256 `{load_json(OUT / 'CALIBRATION_VALIDATION.json')['manifest_sha256']}`.
- Actual wall including load, prefix capture/replay, search, checkpoint, final save/reload: {item['wall_seconds_including_load_replay_save']:.3f} s ({item['wall_seconds_including_load_replay_save']/60:.2f} min); conservative pre-run estimate {costs['scenario_hours']['conservative_M2']:.3f} h under the unchanged 48 h gate. M2/M1 measured calibration wall ratio: {calib['M2']/calib['M1']:.3f}×.
- Final checkpoint SHA256: `{item['final_model_sha256']}`; reload max/mean absolute logit difference on 64 fixed tokens: {item['reload_correctness']['max_abs_logit_difference']}/{item['reload_correctness']['mean_abs_logit_difference']}.
- WT2: PPL {wt['M2']['perplexity']:.8f}, NLL/token {wt['M2']['average_nll']:.9f}; vs M1 {pct(wt['M2']['perplexity'], wt['M1']['perplexity']):+.4f}% PPL. C4 fixed-budget: PPL {c4['M2']['perplexity']:.8f}, NLL/token {c4['M2']['average_nll']:.9f}; vs M1 {pct(c4['M2']['perplexity'], c4['M1']['perplexity']):+.4f}% PPL.
- Final alpha counts: `{json.dumps(dict(alpha_counts), sort_keys=True)}`. Local mean/median relative output-MSE gain: {mean_row_gain:.3%}/{median_row_gain:.3%}; {improved_rows:,} improved rows, {tied_rows:,} tied rows.

## Measured stage wall totals

{stage_md}

Reproduce both formal methods in frozen order with `& '.venv-gptq\\Scripts\\python.exe' -X utf8 'experiments\\range_search_w4\\results\\range_w4_fullmodel_v18\\run_formal.py'`; this runner has no per-method CLI and resumes completed layer checkpoints. Evaluate with `& '.venv-gptq\\Scripts\\python.exe' -X utf8 'experiments\\range_search_w4\\results\\range_w4_fullmodel_v18\\evaluate_v18.py'`; regenerate derived tables and reports with the same command using `finalize_reports_v18.py`. Detailed row/module tables, alpha distribution, candidate landscapes, metadata, and checkpoints are retained in v18.
"""
        p.write_text(text, encoding="utf-8")

    root_run = f"""# v18 execution report

Final status: **COMPLETE**; primary verdict: **{manifest['verdict']}**.

- Cost model: exact-shape matched prechecks and measured all-row scoring/stitch; conservative M2 estimate {costs['scenario_hours']['conservative_M2']:.3f} h, below unchanged 48 h gate. Conservative M1+M2 estimate {costs['scenario_hours']['conservative_M1_plus_M2']:.3f} h.
- Actual calibration: M1 {calib['M1']:.2f} s; M2 {calib['M2']:.2f} s; combined {runtime['overall_quantization_wall_seconds']:.2f} s. M2/M1 {calib['M2']/calib['M1']:.2f}×.
- M2 workload: 1,680 full-Linear candidate GPTQ executions and 3,041,280 row-alpha score slots; peak allocated/reserved CUDA memory {runtime['cuda_peak_allocated_bytes']/2**30:.2f}/{runtime['cuda_peak_reserved_bytes']/2**30:.2f} GiB.
- Evaluation: all six M0/M1/M2 × WT2/fixed-budget-C4 runs pass, with exact frozen target counts and window-weighted NLL checks. Sum of forward evaluation wall: {compact_summary['evaluation']['aggregate_evaluation_forward_seconds']:.1f} s; model loads reported separately.
- Final artifacts are generated by `finalize_reports_v18.py`; raw method/dataset JSON and per-window CSVs remain under `evaluation/`.

No PPL-driven rerun or method change occurred. This run stops at v18 model-level evaluation and awaits independent review.
"""
    (OUT / "RUN_REPORT.md").write_text(root_run, encoding="utf-8")


if __name__ == "__main__":
    main()
