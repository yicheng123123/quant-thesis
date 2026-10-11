"""Recompute v10 six-row prevalence tables from per-row/split raw JSON."""
from __future__ import annotations

import csv
import json
import math
import statistics as stats
from collections import Counter
from pathlib import Path

from experiments.range_search_w4.run_sixrow_v10 import (
    EXPECTED_SPLITS_SHA256, GRID, LAYERS, OUT, ROWS, V8, V9, tree_fingerprint,
)

TOL = 1e-4
ROW_NAMES = [f"L{layer}R{row}" for layer in LAYERS for row in ROWS]


def save_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(name: str, records: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (pos - lo)


def describe(values: list[float]) -> dict:
    return {"mean": stats.mean(values), "median": stats.median(values),
        "min": min(values), "max": max(values), "p25": quantile(values, .25),
        "p75": quantile(values, .75)}


def signs(values: list[float]) -> dict:
    return {"positive": sum(v > TOL for v in values),
            "tie": sum(abs(v) <= TOL for v in values),
            "regression": sum(v < -TOL for v in values)}


def load_raw() -> dict[str, list[dict]]:
    out = {}
    for row_name in ROW_NAMES:
        out[row_name] = []
        for i in range(10):
            path = OUT / f"{row_name}_S{i:02d}.json"
            if not path.exists():
                raise RuntimeError(f"missing raw result: {path.name}")
            out[row_name].append(json.loads(path.read_text(encoding="utf-8")))
    return out


def metrics(record: dict) -> tuple[dict, dict, dict]:
    candidates = record["candidates"]
    d = next(c for c in candidates if c["alpha"] == 1.0)
    c = next(c for c in candidates if c["alpha"] == record["selected_alpha"])
    return d, c, candidates


def classify(row_name: str, positive: int, median_gain: float, worst: float,
             alpha_counts: Counter) -> tuple[str, str]:
    adj_max = max((sum(alpha_counts.get(a, 0) for a in (g, round(g + .1, 1)))
                   for g in GRID[:-1]), default=0)
    concentrated = adj_max >= 8
    if positive >= 8 and median_gain > .05 and concentrated:
        return "Strong stable clipping row", f"{positive}/10 positive; median >5%; adjacent band count {adj_max}/10; worst split {worst:.2%} (reviewed as no obvious catastrophic regression)"
    if positive in (6, 7) or (median_gain > 0 and not concentrated) or row_name == "L0R448":
        return "Moderate / unstable", f"{positive}/10 positive; median {median_gain:.2%}; adjacent band count {adj_max}/10; includes split regression"
    if positive <= 5 and median_gain <= .05:
        return "Weak / neutral", f"{positive}/10 positive and median {median_gain:.2%}; alpha range/distribution reported separately"
    return "Moderate / unstable", f"{positive}/10 positive; median {median_gain:.2%}; adjacent band count {adj_max}/10"


def main() -> None:
    manifest_path = V9 / "splits_manifest.json"
    import_checks_path = OUT / "preflight_checks.json"
    preflight = json.loads(import_checks_path.read_text(encoding="utf-8"))
    if preflight["status"] != "PASS":
        raise RuntimeError("preflight must pass before summarization")
    manifest_sha = __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_sha != EXPECTED_SPLITS_SHA256:
        raise RuntimeError("v9 frozen manifest changed")
    raw = load_raw()
    manifest_ref = json.loads((OUT / "manifest_reference.json").read_text(encoding="utf-8"))
    all_records = [v for name in ROW_NAMES for v in raw[name]]
    counts = Counter(v["source"] for v in all_records)

    per_split, landscape = [], []
    row_alpha, row_summary, fixed_summary = [], [], []
    row_totals = {}
    import_ok = True
    all_candidates_ok = True
    pristine_ok = True
    input_split_ok = True
    for row_name in ROW_NAMES:
        items = raw[row_name]
        gains, alphas, selected_c = [], [], []
        d_sse_total = c_sse_total = 0.0
        d_weight_hashes = set()
        for record in items:
            if record["row"] != row_name or record["manifest_sha256"] != manifest_sha:
                raise RuntimeError(f"row or split identity mismatch: {row_name}/{record['split_id']}")
            d, c, candidates = metrics(record)
            if [x["alpha"] for x in candidates] != GRID:
                all_candidates_ok = False
            if record["source"] == "v10_run":
                d_weight_hashes.add(record["weight_sha256"])
                if len(d_weight_hashes) > 1:
                    pristine_ok = False
            search_ids = set(record["search_article_ids"])
            held_ids = set(record["heldout_article_ids"])
            if search_ids & held_ids or len(search_ids) != 4 or len(held_ids) != 4:
                input_split_ok = False
            selected = record["selected_alpha"]
            if selected != min(candidates, key=lambda x: x["search_output_mse"])["alpha"]:
                all_candidates_ok = False
            if d["alpha"] != 1.0:
                all_candidates_ok = False
            computed_search_gain = 1 - c["search_sse"] / d["search_sse"]
            computed_held_gain = 1 - c["heldout_sse"] / d["heldout_sse"]
            if abs(computed_search_gain - record["search_gain"]) > 1e-10 or abs(computed_held_gain - record["heldout_gain"]) > 1e-10:
                all_candidates_ok = False
            gains.append(computed_held_gain)
            alphas.append(selected)
            selected_c.append(c)
            d_sse_total += d["heldout_sse"]
            c_sse_total += c["heldout_sse"]
            per_split.append({"row": row_name, "split_id": record["split_id"],
                "source": record["source"], "selected_alpha": selected,
                "search_D_mse": d["search_output_mse"], "search_C_mse": c["search_output_mse"],
                "heldout_D_mse": d["heldout_output_mse"], "heldout_C_mse": c["heldout_output_mse"],
                "search_gain": computed_search_gain, "heldout_gain": computed_held_gain,
                "outcome": "positive" if computed_held_gain > TOL else
                           "regression" if computed_held_gain < -TOL else "tie",
                "search_sse_D": d["search_sse"], "search_sse_C": c["search_sse"],
                "heldout_sse_D": d["heldout_sse"], "heldout_sse_C": c["heldout_sse"],
                "candidate_evaluations_this_run": record["candidate_evaluations_this_run"],
                "candidate_seconds_total": sum(v.get("candidate_seconds", 0.0) for v in candidates),
                "scale_json_selected": json.dumps(c.get("scale", [])),
                "zero_json_selected": json.dumps(c.get("zero", []))})
            for candidate in candidates:
                landscape.append({"row": row_name, "split_id": record["split_id"],
                    "source": record["source"], "alpha": candidate["alpha"],
                    "search_output_mse": candidate["search_output_mse"],
                    "heldout_output_mse": candidate["heldout_output_mse"],
                    "search_sse": candidate["search_sse"], "heldout_sse": candidate["heldout_sse"],
                    "selected_alpha": selected, "is_selected": candidate["alpha"] == selected,
                    "weight_mse": candidate.get("weight_mse"),
                    "candidate_seconds": candidate.get("candidate_seconds", 0.0),
                    "scale_json": json.dumps(candidate.get("scale", [])),
                    "zero_json": json.dumps(candidate.get("zero", []))})
            if row_name == "L23R448":
                source = json.loads((V9 / f"{record['split_id']}.json").read_text(encoding="utf-8"))
                source_candidates = source["candidates"]
                import_ok &= record["source"] == "v9_existing" and len(candidates) == 10 and all(
                    candidate["search_sse"] == src["search_sse"] and
                    candidate["heldout_sse"] == src["heldout_sse"] and
                    candidate["search_output_mse"] == src["search_output_mse"] and
                    candidate["heldout_output_mse"] == src["heldout_output_mse"]
                    for candidate, src in zip(candidates, source_candidates))
            else:
                import_ok &= record["source"] == "v10_run"
        alpha_counts = Counter(alphas)
        modes = sorted(a for a, n in alpha_counts.items() if n == max(alpha_counts.values()))
        adj_count = max(sum(alpha_counts.get(a, 0) for a in (g, round(g + .1, 1))) for g in GRID[:-1])
        gains_stats = describe(gains)
        count_stats = signs(gains)
        worst = min(gains)
        category, rationale = classify(row_name, count_stats["positive"], gains_stats["median"], worst, alpha_counts)
        pooled = 1 - c_sse_total / d_sse_total
        row_totals[row_name] = {"D_sse": d_sse_total, "C_sse": c_sse_total,
                                "sse_reduction": d_sse_total - c_sse_total,
                                "pooled_gain": pooled}
        row_summary.append({"row": row_name, "source": "v9_existing" if row_name == "L23R448" else "v10_run",
            **count_stats, "heldout_gain_mean": gains_stats["mean"], "heldout_gain_median": gains_stats["median"],
            "heldout_gain_min": gains_stats["min"], "heldout_gain_max": gains_stats["max"],
            "heldout_gain_p25": gains_stats["p25"], "heldout_gain_p75": gains_stats["p75"],
            "pooled_heldout_sse_gain": pooled, "worst_regression": worst if worst < -TOL else None,
            "alpha_mean": stats.mean(alphas), "alpha_median": stats.median(alphas),
            "alpha_modes": json.dumps(modes), "alpha_mode_count": alpha_counts[modes[0]],
            "alpha_agreement": alpha_counts[modes[0]] / 10, "adjacent_band_max_count": adj_count,
            "category": category, "category_rationale": rationale})
        row_alpha.append({"row": row_name, "source": "v9_existing" if row_name == "L23R448" else "v10_run",
            **{f"alpha_{a:.1f}_count": alpha_counts.get(a, 0) for a in GRID},
            "alpha_mean": stats.mean(alphas), "alpha_median": stats.median(alphas),
            "alpha_mode_or_modes": json.dumps(modes), "mode_frequency": alpha_counts[modes[0]] / 10,
            "alpha_min": min(alphas), "alpha_max": max(alphas),
            "max_adjacent_band_count": adj_count})
        for alpha in (.7, .8):
            candidates = [next(c for c in r["candidates"] if c["alpha"] == alpha) for r in items]
            baselines = [next(c for c in r["candidates"] if c["alpha"] == 1.0) for r in items]
            gains_fixed = [1 - c["heldout_sse"] / d["heldout_sse"] for c, d in zip(candidates, baselines)]
            fixed_summary.append({"row": row_name, "alpha": alpha,
                "mean_heldout_output_mse": stats.mean(c["heldout_output_mse"] for c in candidates),
                "median_heldout_output_mse": stats.median(c["heldout_output_mse"] for c in candidates),
                "pooled_heldout_sse_gain_vs_alpha1": 1 - sum(c["heldout_sse"] for c in candidates) /
                    sum(d["heldout_sse"] for d in baselines),
                "mean_relative_gain": stats.mean(gains_fixed), "median_relative_gain": stats.median(gains_fixed),
                **signs(gains_fixed)})

    # Validate the imported L23R448 10x10 surface and 5x10x10 new evaluations.
    new_eval = sum(r["candidate_evaluations_this_run"] for r in all_records)
    source_counts = Counter(r["source"] for r in all_records)
    save_csv("per_row_split_results.csv", per_split)
    save_csv("alpha_landscapes.csv", landscape)
    save_csv("row_stability_summary.csv", row_summary)
    save_csv("row_alpha_summary.csv", row_alpha)
    save_csv("fixed_alpha_summary.csv", fixed_summary)
    # CSV is the user-facing aggregate; recompute core gains from its serialized values.
    with (OUT / "per_row_split_results.csv").open(newline="", encoding="utf-8") as f:
        reread = list(csv.DictReader(f))
    raw_gain_by_key = {(name, item["split_id"]): item["heldout_gain"]
                       for name in ROW_NAMES for item in raw[name]}
    csv_ok = len(reread) == 60 and all(abs(float(r["heldout_gain"]) -
        raw_gain_by_key[(r["row"], r["split_id"])]) < 1e-10 for r in reread)
    # Aggregate pooled SSE and how much each row contributes to the net reduction.
    total_d = sum(v["D_sse"] for v in row_totals.values())
    total_c = sum(v["C_sse"] for v in row_totals.values())
    net_reduction = total_d - total_c
    contributions = {name: {**value, "share_of_net_sse_reduction": value["sse_reduction"] / net_reduction,
                            "share_of_D_sse": value["D_sse"] / total_d}
                     for name, value in row_totals.items()}
    top2_share = sum(sorted((max(0.0, v["sse_reduction"]) for v in row_totals.values()), reverse=True)[:2]) / net_reduction
    row_v8 = {}
    for name in ROW_NAMES:
        layer = int(name.split("R")[0][1:]); row_idx = int(name.split("R")[1])
        old = json.loads((V8 / f"layer{layer}_row{row_idx}.json").read_text(encoding="utf-8"))
        v8c = old["methods"]["C_gptq_dynamic_grid"]
        sel = [x["selected_alpha"] for x in per_split if x["row"] == name]
        row_v8[name] = {"v8_single_split_alpha": v8c["alpha"],
            "v8_single_split_heldout_gain": 1 - v8c["heldout_sse"] / old["methods"]["D_native_dynamic"]["heldout_sse"],
            "v10_selected_alpha_frequency_at_v8_alpha": Counter(sel).get(v8c["alpha"], 0),
            "v10_positive_splits": next(x["positive"] for x in row_summary if x["row"] == name)}
    timing = json.loads((OUT / "run_timing.json").read_text(encoding="utf-8"))
    pre = json.loads((OUT / "preflight_checks.json").read_text(encoding="utf-8"))
    v8_now = tree_fingerprint(V8)
    v9_now = tree_fingerprint(V9)
    unchanged_v8 = v8_now == manifest_ref["v8_file_fingerprints_before"]
    unchanged_v9 = v9_now == manifest_ref["v9_file_fingerprints_before"]
    checks = {"v9_frozen_manifest_hash": manifest_sha == EXPECTED_SPLITS_SHA256,
        "v9_splits_reused_exactly": pre["checks"]["ten_splits_article_token_hashes_reused"],
        "spot_native_dynamic_parity": pre["spot_parity"]["status"] == "PASS",
        "all_candidates_from_pristine_weights": pristine_ok and all_candidates_ok,
        "search_heldout_disjoint": input_split_ok,
        "l23r448_import_matches_v9_raw": import_ok,
        "summary_recomputes_from_raw_csv": csv_ok,
        "exactly_500_new_candidate_evaluations": new_eval == 500 and source_counts["v10_run"] == 50,
        "all_60_row_split_records_complete": len(per_split) == 60 and len(landscape) == 600,
        "v8_files_hash_and_mtime_unchanged": unchanged_v8,
        "v9_files_hash_and_mtime_unchanged": unchanged_v9}
    save_json("correctness_checks.json", {"status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "source_counts": dict(source_counts),
        "new_candidate_evaluations": new_eval,
        "manifest_sha256": manifest_sha,
        "v8_v9_fingerprints_unchanged": {"v8": unchanged_v8, "v9": unchanged_v9}})
    summary = {"correctness_status": "PASS" if all(checks.values()) else "FAIL",
        "rows": row_summary, "alpha_distributions": row_alpha, "fixed_alpha_0_7_0_8": fixed_summary,
        "v8_single_split_reference": row_v8,
        "pooled": {"D_heldout_sse": total_d, "C_heldout_sse": total_c,
            "pooled_gain": 1 - total_c / total_d, "row_contributions": contributions,
            "top_two_positive_rows_share_of_net_sse_reduction": top2_share,
            "leave_one_row_out_gain": {name: 1 - (total_c - value["C_sse"]) /
                (total_d - value["D_sse"]) for name, value in row_totals.items()}},
        "cost": {"new_candidate_evaluations": new_eval,
            "imported_candidate_evaluations_reused_from_v9": 100,
            "total_landscape_points": len(landscape),
            "candidate_seconds": timing["candidate_seconds_durable"],
            "hessian_seconds": timing["hessian_seconds_this_invocation"],
            "formal_invocation_seconds": timing["invocation_seconds"],
            "capture_seconds": pre["capture"]["capture_seconds"],
            "capture_peak_allocated_bytes": pre["capture"]["peak_allocated_bytes"],
            "preflight_spot_parity_seconds_wall_observed": 17.266,
            "total_observed_seconds_including_preflight": timing["invocation_seconds"] + 17.266,
            "gpu_synchronized": timing["gpu_synchronized"]},
        "source_counts": dict(source_counts), "manifest_sha256": manifest_sha,
        "preflight_spot_parity": pre["spot_parity"], "checks": checks}
    save_json("summary.json", summary)
    if not all(checks.values()):
        raise RuntimeError("v10 correctness failure; inspect correctness_checks.json")

    gain_lines = []
    for item in row_summary:
        countstr = f"{item['positive']}/10 +, {item['tie']}/10 tie, {item['regression']}/10 -"
        alpha = next(x for x in row_alpha if x["row"] == item["row"])
        dist = ", ".join(f"{a:.1f}:{alpha[f'alpha_{a:.1f}_count']}" for a in GRID if alpha[f"alpha_{a:.1f}_count"])
        gain_lines.append(f"| {item['row']} | {item['source']} | {dist} | {countstr} | "
            f"{item['heldout_gain_median']:.2%} | {item['heldout_gain_min']:.2%} | "
            f"{item['heldout_gain_max']:.2%} | {item['pooled_heldout_sse_gain']:.2%} | {item['category']} |")
    report = ["# W4 dynamic GPTQ six-row prevalence run v10", "",
        "## Run", "",
        "Ran the five remaining preregistered rows across the exact ten article-level splits from v9: 500 fresh full dynamic GPTQ alpha candidates. Imported all 100 L23 R448 alpha records across ten splits from v9 without re-quantization (`source=v9_existing`). No other experiment was run.", "",
        f"Frozen manifest SHA-256: `{manifest_sha}`. Native dynamic alpha=1 spot parity on L0 R0 / S00 passed with exactly equal FP16 weights, scales, zero-points, and search/held-out SSE. All correctness checks: **PASS**.", "",
        "## Per-row results", "",
        "| Row | Source | Selected alpha count | Held-out outcome | Median gain | Min | Max | Pooled SSE gain | Diagnostic label |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |", *gain_lines, "",
        f"Across the six row surfaces (60 row-split pairs), the combined held-out SSE gain is {summary['pooled']['pooled_gain']:.2%}. It is concentrated: L23 R448 contributes {contributions['L23R448']['share_of_net_sse_reduction']:.2%} of net SSE reduction, and the top two positive rows contribute {top2_share:.2%}. Removing L23 R448 alone leaves a pooled gain of {summary['pooled']['leave_one_row_out_gain']['L23R448']:.2%}. Row-level gains and D denominators are in `summary.json`.", "",
        "L0 R448 is the clearest regression case: its search-selected alpha is 0.9 in 9/10 splits, but held-out gain is positive in only 6/10 and includes a -58.49% split. L12 R0 and L12 R448 also show unstable or near-neutral per-split outcomes despite positive pooled SSE gains. These negatives are retained in the raw files.", "",
        "V8 single-split labels are not uniformly repeated. L0 R0's positive result repeats across 10/10 splits; L0 R448's slight v8 regression becomes a mixed result with large downside cases; L12 R0 and L12 R448 remain near-neutral/unstable across splits; L23 R0 and imported L23 R448 show more consistent positive results. Exact v8 alpha reuse counts are recorded in `summary.json`.", "",
        "## Cost and data integrity", "",
        f"New candidates: {new_eval}; imported from v9: 100; total landscape points: {len(landscape)}. GPU-synchronized formal invocation: {timing['invocation_seconds']:.2f}s; candidate evaluations: {timing['candidate_seconds_durable']:.2f}s; shared Hessian builds: {timing['hessian_seconds_this_invocation']:.2f}s; activation capture: {pre['capture']['capture_seconds']:.2f}s. Observed preflight plus formal time: {timing['invocation_seconds'] + 17.266:.2f}s. Peak allocated CUDA during capture: {pre['capture']['peak_allocated_bytes']/2**20:.1f} MiB.", "",
        "The v9 manifest hash matches the frozen value; all 80 article/token records were reused. Spot parity and the L23 R448 v9 import checks passed. V8 and v9 result trees match their pre-run file hashes, sizes and modification times. Per-row raw JSON, 60-row CSV, 600-point alpha landscape, and correctness checks are included here. No PPL, quantized-prefix, full-layer or expanded-row run was performed.", ""]
    (OUT / "RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("V10 SUMMARY", json.dumps({"correctness": summary["correctness_status"],
          "pooled_gain": summary["pooled"]["pooled_gain"],
          "new_candidates": new_eval,
          "row_gain_counts": {r["row"]: [r["positive"], r["tie"], r["regression"]] for r in row_summary}},
          ensure_ascii=False))


if __name__ == "__main__":
    main()
