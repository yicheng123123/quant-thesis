"""CPU-only summary/CSV builder; never launches quantization or GPU scoring."""
import csv
import json
import statistics
from pathlib import Path

import numpy as np

O = Path(__file__).resolve().parent


def write_json(name, data):
    (O / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")


def write_csv(name, fields, rows):
    with (O / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    run = json.loads((O / "RUN_REPORT.json").read_text("utf-8"))
    assert run["status"] == "COMPLETE" and run["gpu_wall_consumed_seconds"] <= 600
    raw_rows, selection_rows, sensitive_rows, timing_rows = [], [], [], []
    summary = {"status": "COMPLETE", "gpu_wall_seconds": run["gpu_wall_consumed_seconds"], "targets": {},
               "full_linear_gptq_candidate_executions": 20, "row_alpha_scores_per_scorer": 57600,
               "new_heldout_ppl_fullmodel": "NOT_RUN", "full_block_propagation": "NOT_RUN"}
    for name in ("up_proj", "down_proj"):
        v = run["targets"][name]
        assert len(v["candidate_costs"]) == 10
        assert all(c["source"] == "generated_once" for c in v["candidate_costs"])
        z = np.load(O / v["raw_scores_file"])
        direct, quad = z["direct"], z["quadratic"]
        di, qi = z["direct_selection"], z["quadratic_selection"]
        gap, qgap = z["direct_best_second_gap"], z["quadratic_best_second_gap"]
        err = np.abs(quad - direct)
        rel = err / np.maximum(np.abs(direct), 1e-30)
        assert direct.shape == quad.shape == (10, v["shape"][0])
        assert np.array_equal(di, qi) == (v["comparison"]["selection_difference_count"] == 0)
        assert np.isfinite(direct).all() and np.isfinite(quad).all()
        assert all(v["comparison"]["selected_tensor_exact"].values())
        per_row_error = np.max(err, axis=0)
        margin_certified = 2 * per_row_error < gap
        for i in range(direct.shape[1]):
            selection_rows.append({"projection": name, "row": i,
                                   "direct_alpha": (int(di[i]) + 1) / 10,
                                   "quadratic_alpha": (int(qi[i]) + 1) / 10,
                                   "direct_best_second_gap": float(gap[i]),
                                   "quadratic_best_second_gap": float(qgap[i]),
                                   "max_candidate_abs_score_error": float(per_row_error[i]),
                                   "margin_certified": bool(margin_certified[i])})
            if not margin_certified[i]:
                sensitive_rows.append(selection_rows[-1])
        for ai in range(10):
            for row in range(direct.shape[1]):
                raw_rows.append({"projection": name, "row": row, "alpha": (ai + 1) / 10,
                                 "direct_score": float(direct[ai, row]),
                                 "quadratic_score": float(quad[ai, row]),
                                 "absolute_error": float(err[ai, row]),
                                 "relative_error": float(rel[ai, row])})
        repeats = v["timing_repeats"]
        direct_times = [x["total_wall_s"] for x in repeats if x["method"] == "direct"]
        quadratic_times = [x["total_wall_s"] for x in repeats if x["method"] == "quadratic"]
        assert len(direct_times) == len(quadratic_times) == 2
        timing_rows.extend(dict(projection=name, **x) for x in repeats)
        direct_mean = statistics.mean(direct_times)
        quadratic_mean = statistics.mean(quadratic_times)
        summary["targets"][name] = {
            "rows": direct.shape[1], "candidates": 10,
            "max_absolute_score_error": float(err.max()),
            "median_absolute_score_error": float(np.median(err)),
            "max_relative_score_error": float(rel.max()),
            "p95_relative_score_error": float(np.quantile(rel, .95)),
            "median_relative_score_error": float(np.median(rel)),
            "selection_difference_rows": int(np.count_nonzero(di != qi)),
            "direct_exact_tie_rows": int(np.count_nonzero(gap == 0)),
            "quadratic_exact_tie_rows": int(np.count_nonzero(qgap == 0)),
            "minimum_direct_best_second_gap": float(np.min(gap)),
            "median_direct_best_second_gap": float(np.median(gap)),
            "margin_certified_rows": int(np.count_nonzero(margin_certified)),
            "sensitive_same_selection_rows": np.flatnonzero(~margin_certified).tolist(),
            "direct_selected_alpha_histogram": {(i + 1) / 10: int(x) for i, x in enumerate(np.bincount(di, minlength=10))},
            "quadratic_selected_alpha_histogram": {(i + 1) / 10: int(x) for i, x in enumerate(np.bincount(qi, minlength=10))},
            "selected_weight_scale_zero_g_idx_exact": v["comparison"]["selected_tensor_exact"],
            "direct_recheck": v["comparison"]["direct_recheck"],
            "validation_seconds": v["validation_timing"],
            "repeat_direct_full10_mean_seconds": direct_mean,
            "repeat_direct_full10_range_seconds": max(direct_times) - min(direct_times),
            "repeat_quadratic_C_plus_full10_mean_seconds": quadratic_mean,
            "repeat_quadratic_C_plus_full10_range_seconds": max(quadratic_times) - min(quadratic_times),
            "repeat_absolute_score_time_saved_seconds": direct_mean - quadratic_mean,
            "repeat_scoring_only_time_ratio": direct_mean / quadratic_mean,
            "C_bytes": v["validation_timing"]["C_bytes"],
            "direct_repeat_incremental_peak_bytes": [x["incremental_peak_bytes"] for x in repeats if x["method"] == "direct"],
            "quadratic_repeat_incremental_peak_bytes": [x["incremental_peak_bytes"] for x in repeats if x["method"] == "quadratic"],
        }
    summary["verdict"] = "A_LOCAL_SCORING_SELECTION_MATCH_AND_CLEAR_BUILD_INCLUDED_BENEFIT" if all(
        x["selection_difference_rows"] == 0 and x["repeat_absolute_score_time_saved_seconds"] >
        2 * max(x["repeat_direct_full10_range_seconds"], x["repeat_quadratic_C_plus_full10_range_seconds"])
        for x in summary["targets"].values()) else "C_OR_B_REVIEW_REQUIRED"
    write_csv("RAW_SCORE_COMPARISON.csv", ["projection", "row", "alpha", "direct_score", "quadratic_score", "absolute_error", "relative_error"], raw_rows)
    write_csv("ROW_SELECTION.csv", ["projection", "row", "direct_alpha", "quadratic_alpha", "direct_best_second_gap", "quadratic_best_second_gap", "max_candidate_abs_score_error", "margin_certified"], selection_rows)
    write_csv("SENSITIVE_ROWS.csv", ["projection", "row", "direct_alpha", "quadratic_alpha", "direct_best_second_gap", "quadratic_best_second_gap", "max_candidate_abs_score_error", "margin_certified"], sensitive_rows)
    write_csv("SCORING_TIMING.csv", ["projection", "method", "repeat", "C_build_wall_s", "score_10_wall_s", "total_wall_s", "C_bytes", "baseline_allocated_bytes", "peak_allocated_bytes", "incremental_peak_bytes", "peak_reserved_bytes"], timing_rows)
    write_json("SUMMARY.json", summary)
    print(json.dumps({"verdict": summary["verdict"], "targets": {k: {"selection_differences": v["selection_difference_rows"],
         "direct_seconds": v["repeat_direct_full10_mean_seconds"], "quadratic_seconds": v["repeat_quadratic_C_plus_full10_mean_seconds"]}
         for k, v in summary["targets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
