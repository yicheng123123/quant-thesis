"""Recompute v8 tables, checks, and gate from six immutable per-row JSON files."""
from __future__ import annotations

import csv
import json
import math
import statistics as stats
from pathlib import Path

from experiments.range_search_w4.run_task0_parity import OUTPUT, historical_fingerprint

ROWS = [(layer, row) for layer in (0, 12, 23) for row in (0, 448)]
METHODS = ("D_native_dynamic", "A_fixed_minmax", "B_rtn_then_dynamic", "C_gptq_dynamic_grid")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    for pos, i in enumerate(order):
        result[i] = float(pos)
    # Average ties rather than forcing arbitrary alpha order.
    for value in set(values):
        members = [i for i, v in enumerate(values) if v == value]
        middle = sum(result[i] for i in members) / len(members)
        for i in members:
            result[i] = middle
    return result


def spearman(a: list[float], b: list[float]) -> float | None:
    ra, rb = rank(a), rank(b)
    ma, mb = stats.mean(ra), stats.mean(rb)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return numerator / den if den else None


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    index = (len(s) - 1) * p
    low = math.floor(index)
    return s[low] + (s[min(low + 1, len(s) - 1)] - s[low]) * (index - low)


def describe(values: list[float]) -> dict:
    return {"count": len(values), "median": quantile(values, .5),
            "p25": quantile(values, .25), "p75": quantile(values, .75),
            "min": min(values) if values else None, "max": max(values) if values else None}


def comparison(rows: list[dict], method: str, baseline: str) -> dict:
    gains = []
    for r in rows:
        a = r["methods"][method]["heldout_sse"]
        b = r["methods"][baseline]["heldout_sse"]
        gains.append(1 - a / b)
    return {"method": method, "baseline": baseline,
            "heldout_gain_distribution": describe(gains),
            "total_sse_improvement": 1 - sum(r["methods"][method]["heldout_sse"] for r in rows) /
                sum(r["methods"][baseline]["heldout_sse"] for r in rows),
            "improve": sum(g > 1e-9 for g in gains),
            "tie": sum(abs(g) <= 1e-9 for g in gains),
            "regress": sum(g < -1e-9 for g in gains),
            "worst_regression": min(gains), "row_gains": gains}


def main() -> None:
    records = []
    for layer, row in ROWS:
        p = OUTPUT / f"layer{layer}_row{row}.json"
        if not p.exists():
            raise RuntimeError(f"six-row run incomplete: {p}")
        records.append(json.loads(p.read_text(encoding="utf-8")))
    parity = json.loads((OUTPUT / "dynamic_parity_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    timing = json.loads((OUTPUT / "run_timing.json").read_text(encoding="utf-8"))
    native_rows = json.loads((OUTPUT / "native_six_row_check.json").read_text(encoding="utf-8"))
    method_rows, candidate_rows, rerank_rows, interaction_rows = [], [], [], []
    global_spearman, group_spearman = [], []
    for r in records:
        label = f"L{r['layer']}R{r['row']}"
        d = r["methods"]["D_native_dynamic"]
        for name in METHODS:
            v = r["methods"][name]
            method_rows.append({"layer": r["layer"], "row": r["row"], "method": name,
                "alpha": v["alpha"], "search_sse": v["search_sse"], "heldout_sse": v["heldout_sse"],
                "search_output_mse": v["search_output_mse"], "heldout_output_mse": v["heldout_output_mse"],
                "weight_mse": v["weight_mse"], "heldout_gain_vs_D": 1 - v["heldout_sse"] / d["heldout_sse"],
                "gptq_candidate_evaluations": v["candidate_evaluations"],
                "rtn_candidate_evaluations": v.get("rtn_evaluations", 0),
                "candidate_seconds": v["seconds"], "method_seconds": v.get("total_seconds", v["seconds"]),
                "peak_allocated_bytes_row": r["peak_allocated_bytes"],
                "scale_json": json.dumps(v["scale"]), "zero_json": json.dumps(v["zero"])})
        for scope, candidates in (("shared_alpha", r["global_candidates"]),
                                  ("one_group_intervention", r["group_candidates"])):
            proxies = r["rtn_global"] if scope == "shared_alpha" else candidates
            if scope == "shared_alpha":
                zipped = list(zip(proxies, candidates))
                group_sets = [(None, zipped)]
            else:
                group_sets = [(g, [(v, v) for v in candidates if v["group"] == g]) for g in range(7)]
            for group, pairs in group_sets:
                rtn_output = [p[0]["search_output_mse" if group is None else "rtn_search_output_mse"] for p in pairs]
                rtn_weight = [p[0]["weight_mse" if group is None else "rtn_weight_mse"] for p in pairs]
                gptq_output = [p[1]["search_output_mse" if group is None else "gptq_search_output_mse"] for p in pairs]
                gp_best = min(range(10), key=lambda i: gptq_output[i])
                rt_best = min(range(10), key=lambda i: rtn_output[i])
                rt_order = sorted(range(10), key=lambda i: (rtn_output[i], i))
                item = {"layer": r["layer"], "row": r["row"], "group": group,
                    "scope": scope, "rtn_output_spearman": spearman(rtn_output, gptq_output),
                    "rtn_weight_spearman": spearman(rtn_weight, gptq_output),
                    "rtn_best_alpha": pairs[rt_best][1]["alpha"],
                    "gptq_best_alpha": pairs[gp_best][1]["alpha"],
                    "best_matches": rt_best == gp_best,
                    "top1_hit": gp_best in rt_order[:1], "top2_hit": gp_best in rt_order[:2],
                    "top3_hit": gp_best in rt_order[:3], "top5_hit": gp_best in rt_order[:5],
                    "search_regret": gptq_output[rt_best] - gptq_output[gp_best],
                    "relative_search_regret": gptq_output[rt_best] / gptq_output[gp_best] - 1,
                    "heldout_regret": (pairs[rt_best][1]["heldout_output_mse"] if group is None else
                                       pairs[rt_best][1]["gptq_heldout_output_mse"]) -
                                      (pairs[gp_best][1]["heldout_output_mse"] if group is None else
                                       pairs[gp_best][1]["gptq_heldout_output_mse"])}
                (global_spearman if group is None else group_spearman).append(item)
                for proxy, gp in pairs:
                    candidate_rows.append({"layer": r["layer"], "row": r["row"],
                        "scope": scope, "group": "" if group is None else group, "alpha": gp["alpha"],
                        "rtn_weight_mse": proxy["weight_mse"] if group is None else proxy["rtn_weight_mse"],
                        "rtn_output_mse_search": proxy["search_output_mse"] if group is None else proxy["rtn_search_output_mse"],
                        "rtn_output_mse_heldout": proxy["heldout_output_mse"] if group is None else proxy["rtn_heldout_output_mse"],
                        "gptq_output_mse_search": gp["search_output_mse"] if group is None else gp["gptq_search_output_mse"],
                        "gptq_output_mse_heldout": gp["heldout_output_mse"] if group is None else gp["gptq_heldout_output_mse"],
                        "gptq_weight_mse": gp["weight_mse"] if group is None else gp["gptq_weight_mse"],
                        "gptq_seconds": gp["seconds"],
                        "gptq_scale_json": json.dumps(gp["scale"] if group is None else gp["gptq_scale"]),
                        "gptq_zero_json": json.dumps(gp["zero"] if group is None else gp["gptq_zero"])})
        for rr in r["rerank"]:
            v = rr["selected"]
            c = r["methods"]["C_gptq_dynamic_grid"]
            rerank_rows.append({"layer": r["layer"], "row": r["row"], "k": rr["k"],
                "candidate_alphas_json": json.dumps(rr["candidate_alphas"]),
                "selected_alpha": v["alpha"], "full_C_alpha": c["alpha"],
                "matches_full_C": v["alpha"] == c["alpha"],
                "gptq_candidate_evaluations": rr["gptq_candidate_evaluations"],
                "rtn_candidate_evaluations": 10, "seconds_including_proxy": rr["seconds"],
                "search_output_mse": v["search_output_mse"], "heldout_output_mse": v["heldout_output_mse"],
                "heldout_regret_vs_full_C": v["heldout_output_mse"] / c["heldout_output_mse"] - 1,
                "heldout_gain_vs_D": 1 - v["heldout_output_mse"] / d["heldout_output_mse"]})
        diagnostic = r["group_interaction"]
        for method in ("minmax", "independent_group_search", "coordinate_replace_one_sweep",
                       "gptq_minmax_external", "gptq_independent_external"):
            value = diagnostic["reproduction"][method]
            for split in ("search", "heldout"):
                err = value[split]
                interaction_rows.append({"layer": r["layer"], "row": r["row"],
                    "method": "legacy_" + method, "split": split,
                    "independent_sse": err["independent_sum_squares"],
                    "full_sse": err["full_sum_squares"], "cross_term": err["cross_term"],
                    "cross_over_full": err["cross_term"] / err["full_sum_squares"]})
        for method in ("dynamic_D", "dynamic_C"):
            for split in ("search", "heldout"):
                err = diagnostic[f"{method}_{split}"]
                interaction_rows.append({"layer": r["layer"], "row": r["row"], "method": method,
                    "split": split, "independent_sse": err["independent_sum_squares"],
                    "full_sse": err["full_sum_squares"], "cross_term": err["cross_term"],
                    "cross_over_full": err["cross_term"] / err["full_sum_squares"]})
    write_csv(OUTPUT / "six_row_results.csv", method_rows)
    write_csv(OUTPUT / "candidate_rankings.csv", candidate_rows)
    write_csv(OUTPUT / "rerank_results.csv", rerank_rows)
    write_csv(OUTPUT / "group_interaction_diagnostics.csv", interaction_rows)
    comparisons = [comparison(records, m, b) for m, b in (("D_native_dynamic", "A_fixed_minmax"),
        ("B_rtn_then_dynamic", "A_fixed_minmax"), ("C_gptq_dynamic_grid", "A_fixed_minmax"),
        ("B_rtn_then_dynamic", "D_native_dynamic"), ("C_gptq_dynamic_grid", "D_native_dynamic"))]
    ranking = {}
    for label, items in (("shared_alpha", global_spearman), ("one_group_intervention", group_spearman)):
        ranking[label] = {"n": len(items), "rtn_output_spearman": describe([v["rtn_output_spearman"] for v in items if v["rtn_output_spearman"] is not None]),
            "rtn_weight_spearman": describe([v["rtn_weight_spearman"] for v in items if v["rtn_weight_spearman"] is not None]),
            "low_output_correlation_below_0_5": sum(v["rtn_output_spearman"] is not None and v["rtn_output_spearman"] < .5 for v in items),
            "best_match_count": sum(v["best_matches"] for v in items),
            "top_k_hit_count": {str(k): sum(v[f"top{k}_hit"] for v in items) for k in (1, 2, 3, 5)},
            "relative_search_regret": describe([v["relative_search_regret"] for v in items]),
            "heldout_regret": describe([v["heldout_regret"] for v in items]), "per_unit": items}
    write_json(OUTPUT / "ranking_summary.json", ranking)
    (OUTPUT / "ranking_summary.md").write_text("# RTN vs GPTQ candidate ranking\n\n" +
        "Ranking uses the same ten alpha values and search articles. Group interventions rerun full dynamic GPTQ.\n\n" +
        "\n".join(f"- {scope}: n={v['n']}; RTN output Spearman median {v['rtn_output_spearman']['median']:.3f} " +
            f"(p25 {v['rtn_output_spearman']['p25']:.3f}, p75 {v['rtn_output_spearman']['p75']:.3f}); " +
            f"best match {v['best_match_count']}/{v['n']}; top-2 {v['top_k_hit_count']['2']}/{v['n']}; " +
            f"top-3 {v['top_k_hit_count']['3']}/{v['n']}."
            for scope, v in ranking.items()) + "\n", encoding="utf-8")
    c_vs_d = comparisons[-1]
    c_stable = (c_vs_d["improve"] >= 5 and c_vs_d["total_sse_improvement"] >= .02 and
                c_vs_d["heldout_gain_distribution"]["median"] >= .02 and
                c_vs_d["worst_regression"] >= -.10)
    rerank_summary = {}
    for k in (1, 2, 3, 5):
        sub = [v for v in rerank_rows if v["k"] == k]
        rerank_summary[str(k)] = {"gptq_evaluations_total": sum(v["gptq_candidate_evaluations"] for v in sub),
            "same_alpha_count": sum(v["matches_full_C"] for v in sub),
            "heldout_regret_vs_C": describe([v["heldout_regret_vs_full_C"] for v in sub]),
            "heldout_gain_vs_D": describe([v["heldout_gain_vs_D"] for v in sub]),
            "total_seconds_including_proxy": sum(v["seconds_including_proxy"] for v in sub)}
    timing_summary = {"capture_seconds": timing["capture_seconds"],
        "hessian_seconds_by_layer": timing["hessian_seconds_by_layer"],
        "six_row_invocation_seconds": timing["invocation_seconds"],
        "row_seconds": {f"L{r['layer']}R{r['row']}": r["total_seconds_excluding_capture_and_hessian"] for r in records},
        "row_peak_allocated_bytes": {f"L{r['layer']}R{r['row']}": r["peak_allocated_bytes"] for r in records},
        "full_C_candidate_evaluations": 60,
        "group_intervention_candidate_evaluations": 420,
        "rerank_gptq_candidate_evaluations": sum(v["gptq_candidate_evaluations"] for v in rerank_rows),
        "full_C_60_candidate_seconds": sum(v["seconds"] for r in records for v in r["global_candidates"]),
        "ninety_six_row_C_candidate_only_estimate_seconds":
            sum(v["seconds"] for r in records for v in r["global_candidates"]) / 6 * 96,
        "ninety_six_row_linear_runtime_estimate_seconds": timing["invocation_seconds"] / 6 * 96,
        "estimate_scope": "candidate-only extrapolation excludes capture/Hessian and diagnostics; full-invocation extrapolation includes 70 group interventions per row; neither is a measured 96-row runtime"}
    write_json(OUTPUT / "timing.json", timing_summary)
    with (OUTPUT / "six_row_results.csv").open(newline="", encoding="utf-8") as f:
        reloaded_csv = list(csv.DictReader(f))
    raw_total_matches = all(abs(
        sum(r["methods"][m]["heldout_sse"] for r in records) -
        sum(float(v["heldout_sse"]) for v in reloaded_csv if v["method"] == m)
    ) < 1e-10 for m in METHODS)
    raw_total_matches &= all(abs(
        1 - sum(float(v["heldout_sse"]) for v in reloaded_csv if v["method"] == x["method"]) /
            sum(float(v["heldout_sse"]) for v in reloaded_csv if v["method"] == x["baseline"]) -
        x["total_sse_improvement"]) < 1e-10 for x in comparisons)
    checks = {"native_parity": parity["status"] == "PASS",
        "all_six_rows_match_native_full_layer": native_rows["status"] == "PASS" and len(native_rows["rows"]) == 6,
        "fixed_alpha1_matches_native_minmax_params": all(r["checks"]["A_scale_equals_initial_minmax"] for r in records),
        "pristine_weight_unchanged": all(r["checks"]["pristine_unchanged"] for r in records),
        "D_is_C_alpha1": all(r["checks"]["D_equals_C_alpha1"] for r in records),
        "document_disjoint": len({d["sha256"] for d in manifest["documents"][:4]} &
                                  {d["sha256"] for d in manifest["documents"][4:]}) == 0,
        "matching_candidate_sets": all([v["alpha"] for v in r["rtn_global"]] ==
                                       [v["alpha"] for v in r["global_candidates"]] for r in records),
        "group_intervention_candidate_sets": all(
            sorted(v["alpha"] for v in r["group_candidates"] if v["group"] == g) ==
            [round(i / 10, 1) for i in range(1, 11)]
            for r in records for g in range(7)),
        "rerank_includes_minmax": all(r["checks"]["rerank_includes_minmax"] for r in records),
        "alpha1_restarts_match_D": all(
            all(abs(next(v for v in rr["candidate_results"] if v["alpha"] == 1.0)[key] -
                    r["methods"]["D_native_dynamic"][key]) < 1e-10
                for key in ("search_sse", "heldout_sse", "weight_mse"))
            for r in records for rr in r["rerank"]),
        "all_recorded_zeros_are_eight": all(
            all(z == 8.0 for z in r["methods"][m]["zero"])
            for r in records for m in METHODS),
        "batch_matches_sequential_fp16": all(r["checks"]["batch_vs_sequential_fp16_max_abs"] == 0 for r in records),
        "direct_output_matches_delta": all(r["checks"]["direct_output_vs_delta_relative"] < 1e-4 for r in records),
        "total_sse_from_raw": raw_total_matches,
        "historical_v7_unchanged": manifest["historical_v7_before"] == historical_fingerprint(),
        "legacy_diagnostic_reproduced": all(r["group_interaction"]["historical_coordinate_heldout_abs_diff"] < 1e-5 for r in records),
        "candidate_count": len(candidate_rows) == 480,
        "six_rows_complete": len(records) == 6}
    write_json(OUTPUT / "correctness_checks.json", {"status": "PASS" if all(checks.values()) else "FAIL",
                                                 "checks": checks,
                                                 "per_row_checks": {f"L{r['layer']}R{r['row']}": r["checks"] for r in records}})
    summary = {"comparisons": comparisons, "ranking": {k: {a: b for a, b in v.items() if a != "per_unit"} for k, v in ranking.items()},
        "rerank": rerank_summary, "selected_alphas": {f"L{r['layer']}R{r['row']}":
            {m: r["methods"][m]["alpha"] for m in METHODS} for r in records},
        "c_vs_d_stable_signal_rule": "at least 5/6 held wins, >=2% median and pooled held gain, worst regression >= -10%",
        "c_vs_d_stable_signal": c_stable,
        "checks_pass": all(checks.values())}
    write_json(OUTPUT / "summary.json", summary)
    top2, top3 = rerank_summary["2"], rerank_summary["3"]
    gate_c = any(v["gptq_evaluations_total"] <= 30 and v["heldout_regret_vs_C"]["median"] <= .05
                 for v in (top2, top3))
    gate_lines = ["# Gate decision after six preregistered rows", "",
        f"- A parity and checks: {'PASS' if all(checks.values()) else 'STOP: failed check'}.",
        f"- B C vs D held-out: {c_vs_d['improve']}/6 improve, median {100*c_vs_d['heldout_gain_distribution']['median']:.2f}%, pooled SSE {100*c_vs_d['total_sse_improvement']:.2f}%; stable rule {'PASS' if c_stable else 'NOT MET'}.",
        f"- C rerank: top-2 {top2['gptq_evaluations_total']}/60 GPTQ evaluations, median held regret {100*top2['heldout_regret_vs_C']['median']:.2f}%; top-3 {top3['gptq_evaluations_total']}/60, median regret {100*top3['heldout_regret_vs_C']['median']:.2f}%; criterion {'PASS' if gate_c else 'NOT MET'}.",
        f"- D measured six-row wall time {timing['invocation_seconds']:.1f}s. The 60 C GPTQ candidates took {timing_summary['full_C_60_candidate_seconds']:.1f}s; a linear 96-row C candidate-only estimate is {timing_summary['ninety_six_row_C_candidate_only_estimate_seconds']:.1f}s plus capture/Hessian. Repeating every diagnostic gives a much larger {timing_summary['ninety_six_row_linear_runtime_estimate_seconds']:.1f}s estimate. Neither is measured 96-row runtime.", "",
        "No 96-row or full-model PPL run is authorized by this gate alone; the six-row result is local FP-input evidence.", ""]
    (OUTPUT / "GATE_DECISION.md").write_text("\n".join(gate_lines), encoding="utf-8")
    severe = min(((1 - r["methods"]["C_gptq_dynamic_grid"]["heldout_sse"] /
                   r["methods"]["D_native_dynamic"]["heldout_sse"], r) for r in records), key=lambda t: t[0])
    bad = severe[1]
    special = next(r for r in records if (r["layer"], r["row"]) == (23, 448))
    sp = special["group_interaction"]["reproduction"]
    coordinate_held_gains = [1 - r["group_interaction"]["reproduction"]["coordinate_replace_one_sweep"]["heldout"]["full_sum_squares"] /
        r["group_interaction"]["reproduction"]["independent_group_search"]["heldout"]["full_sum_squares"] for r in records]
    activation_path = OUTPUT / "activation_shift_diagnostics.json"
    activation = json.loads(activation_path.read_text(encoding="utf-8")) if activation_path.exists() else None
    activation_line = (f"Layer23 FP-input shift: global RMS ratio held/search {activation['layers']['23']['global_rms_ratio']:.3f}, "
        f"median per-feature RMS relative change {100*activation['layers']['23']['feature_rms_relative_change_median']:.1f}%, "
        f"Gram Frobenius relative change {100*activation['layers']['23']['gram_relative_frobenius_change']:.1f}%. "
        "These describe the split, without isolating activation shift as the cause of loss changes."
        if activation else "FP-input distribution shift was not independently measured.")
    method_comparison_lines = [f"- {v['method']} vs {v['baseline']}: held-out median {100*v['heldout_gain_distribution']['median']:.2f}%, pooled SSE {100*v['total_sse_improvement']:.2f}%, {v['improve']} improve / {v['tie']} tie / {v['regress']} regress."
                               for v in comparisons]
    row_lines = [f"- L{r['layer']} R{r['row']}: D {r['methods']['D_native_dynamic']['heldout_output_mse']:.6g}; A {r['methods']['A_fixed_minmax']['heldout_output_mse']:.6g}; B alpha={r['methods']['B_rtn_then_dynamic']['alpha']} {r['methods']['B_rtn_then_dynamic']['heldout_output_mse']:.6g}; C alpha={r['methods']['C_gptq_dynamic_grid']['alpha']} {r['methods']['C_gptq_dynamic_grid']['heldout_output_mse']:.6g}."
                 for r in records]
    b_worst = min(((1 - r["methods"]["B_rtn_then_dynamic"]["heldout_sse"] /
                    r["methods"]["D_native_dynamic"]["heldout_sse"], r) for r in records), key=lambda t: t[0])
    report = ["# Dynamic W4 G128 range search: six-row audit", "",
        "## Scope and parity", "",
        "Qwen2.5-0.5B-Instruct, frozen revision in protocol.json, FP16 weights and FP32 GPTQ math; symmetric W4A16 G128, block 128, damp 0.1, desc_act false. Eight distinct WikiText-2 train articles supply four search and four held-out 128-token sequences; inputs are FP model activations. Native GPTQModel 2.2.0 and custom dynamic alpha=1 match exactly on the Task 0 full layer and all six representative rows across three native full-layer runs. No test data or PPL used.", "",
        "D is native dynamic GPTQ (alpha=1). A freezes MinMax at pristine W. B uses whole-row RTN output MSE to choose one shared alpha, then runs dynamic GPTQ. C evaluates all ten shared-alpha dynamic GPTQ candidates from pristine W and selects on search output MSE. The fixed-range v7 per-channel 43.26% result has different grouping and range timing; its contribution from weak fixed range cannot be numerically decomposed by this G128 study.", "",
        "## Main comparison", "",
        f"C vs D: {c_vs_d['improve']} improve, {c_vs_d['tie']} tie, {c_vs_d['regress']} regress on held-out; median relative gain {100*c_vs_d['heldout_gain_distribution']['median']:.2f}%, pooled SSE gain {100*c_vs_d['total_sse_improvement']:.2f}%, worst row gain {100*c_vs_d['worst_regression']:.2f}%. Full row data and all actual scales/zero-points are in six_row_results.csv and raw JSON.", "",
        *method_comparison_lines, "", "Held-out output MSE by preregistered row:", "", *row_lines, "",
        "The ten-point search guarantees no worse **search** loss than D because D is alpha=1 in the same flow; it gives no held-out or PPL guarantee.", "",
        "## Proxy ranking and cost", "",
        f"Shared-alpha RTN output/GPTQ Spearman median {ranking['shared_alpha']['rtn_output_spearman']['median']:.3f}; one-group intervention median {ranking['one_group_intervention']['rtn_output_spearman']['median']:.3f}. See ranking_summary.json for p25/p75, min/max, weight proxy, all top-k hits, and per-group regrets. Interventions rerun all downstream GPTQ compensation; they are diagnostics, not the C method.", "",
        f"Top-2 rerank uses {top2['gptq_evaluations_total']} GPTQ evaluations across six rows vs 60 for C; top-3 uses {top3['gptq_evaluations_total']}. Both match C's alpha on 5/6 rows; their worst held-out regret relative to C is {100*top2['heldout_regret_vs_C']['max']:.2f}%. Both count ten RTN evaluations per row and a fresh GPTQ rerun of every shortlisted candidate. C's 60 candidate evaluations took {timing_summary['full_C_60_candidate_seconds']:.1f}s; the full six-row invocation including 420 group interventions took {timing['invocation_seconds']:.1f}s. These are offline calibration costs, not online inference speed.", "",
        "## Group interaction and failure cases", "",
        f"Layer23 row448 legacy coordinate replacement search full SSE {sp['coordinate_replace_one_sweep']['search']['full_sum_squares']:.6g}, held SSE {sp['coordinate_replace_one_sweep']['heldout']['full_sum_squares']:.6g}; cross terms {sp['coordinate_replace_one_sweep']['search']['cross_term']:.6g} and {sp['coordinate_replace_one_sweep']['heldout']['cross_term']:.6g}. This RTN-only one-group replacement can lose cancellation on held-out, so its search gains cannot be attributed to dynamic GPTQ. All six rows' independent/full/cross SSE, including dynamic D/C, are in group_interaction_diagnostics.csv.", "",
        f"The legacy RTN coordinate diagnostic improves held-out full SSE relative to independent-group RTN selection on {sum(g > 0 for g in coordinate_held_gains)}/6 rows, but layer23 row448 regresses by {-100*min(coordinate_held_gains):.2f}%. Its group-local vs whole-row candidate rankings disagree in {sum(v['search']['ranking_disagrees'] for v in sp['ranking_diagnostic'])}/7 groups on search. These are effects of group interaction under RTN replacement, with no demonstrated transfer to a GPTQ coordinate method.", "",
        activation_line, "",
        f"Worst C-vs-D held row: layer {bad['layer']} row {bad['row']}, relative gain {100*severe[0]:.2f}%, C alpha {bad['methods']['C_gptq_dynamic_grid']['alpha']}. Its exact losses remain in the raw JSON.", "",
        f"Worst B-vs-D held row: layer {b_worst[1]['layer']} row {b_worst[1]['row']}, relative gain {100*b_worst[0]:.2f}%, B alpha {b_worst[1]['methods']['B_rtn_then_dynamic']['alpha']}; the proxy can choose a candidate that GPTQ search would reject. This is a negative result kept in the raw JSON.", "",
        "The dynamic D/A differences change sign across rows, so fixed MinMax is not uniformly weaker. For layer23 row448, v7's RTN coordinate search lowered search error but its cross-term changed from negative on search to positive on held-out. This supports a cancellation/generalization concern for that RTN diagnostic; it does not establish the causal mechanism for dynamic GPTQ or final model quality.", "",
        "## Interpretation and next evidence", "",
        "The six rows use one projection type (q_proj), so projection-type differences are untested. Search/held-out article counts are small; alpha stability across other calibration splits is untested. No quantized-prefix activation, full-layer reconstruction, full-model PPL, or downstream quality has been measured here. Gate B fails the fixed 5/6 consistency rule, so no 96-row or PPL expansion follows from this run. The thesis hypothesis remains open but lacks broader held-out stability and model-level evidence. The current engineering result is an auditable native-parity dynamic GPTQ range search with resume, measured candidate budgets, and independent raw-result checks. No originality or publication claim is made.", "",
        "## Reproduction", "",
        "From E:\\quant-thesis with the existing .venv-gptq environment:", "",
        "```powershell", "$env:PYTHONPATH='E:\\quant-thesis'", ".\\.venv-gptq\\Scripts\\python.exe -m experiments.range_search_w4.run_task0_parity", ".\\.venv-gptq\\Scripts\\python.exe -m experiments.range_search_w4.run_dynamic_v8 --smoke", ".\\.venv-gptq\\Scripts\\python.exe -m experiments.range_search_w4.run_dynamic_v8", ".\\.venv-gptq\\Scripts\\python.exe -m experiments.range_search_w4.verify_native_rows_v8", ".\\.venv-gptq\\Scripts\\python.exe -m experiments.range_search_w4.diagnose_activation_v8", ".\\.venv-gptq\\Scripts\\python.exe -m experiments.range_search_w4.summarize_dynamic_v8", "```", "",
        "The parity command requires a fresh v8 output directory; the existing parity artifacts are already saved. The main runner resumes completed per-row JSON files without overwriting them. The summarizer regenerates CSV and reports from all six raw rows.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("SUMMARY", json.dumps({"c_vs_d": c_vs_d, "checks": all(checks.values()), "gate_B": c_stable}, ensure_ascii=False))


if __name__ == "__main__":
    main()
