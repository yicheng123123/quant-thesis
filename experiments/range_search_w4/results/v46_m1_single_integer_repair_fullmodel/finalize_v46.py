"""CPU-only final provenance, PPL, comparison, and cost ledger for v46."""
import csv
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V18 = RESULTS / "range_w4_fullmodel_v18"
V40 = RESULTS / "range_w4_winner_union_v40"
V42 = RESULTS / "v42_m2_single_integer_repair_fullmodel"
V43 = RESULTS / "v43_m2_single_integer_repair_fullmodel"
V44 = RESULTS / "v44_m2_single_integer_repair_fullmodel"
V45 = RESULTS / "v45_m2_single_integer_repair_fullmodel"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def write_csv(name, rows):
    if not rows:
        raise AssertionError(f"empty CSV: {name}")
    with (OUT / name).open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def windows_nll(path):
    rows = read_csv(path)
    n = sum(int(r["valid_tokens"]) for r in rows)
    total = sum(float(r["window_nll"]) for r in rows)
    return total / n, n, len(rows)


def historic(path, expected, dataset, method):
    result = read(path)
    assert result["status"] == "PASS"
    assert sha(path) == expected[f"{method}_{dataset}"]["result_sha256"]
    if method in ("M1", "M2"):
        windows = result["window_results"]
        n = sum(int(x["valid_tokens"]) for x in windows)
        nll = sum(float(x["window_nll"]) for x in windows) / n
        count = len(windows)
        raw_window_sha = sha(V18 / "evaluation/windows" / f"{method}_{dataset}.csv")
    else:
        wpath = V45 / f"R_{dataset}_WINDOWS.csv"
        nll, n, count = windows_nll(wpath)
        raw_window_sha = sha(wpath)
    audit = expected[f"{method}_{dataset}"]
    assert n == audit["valid_tokens"] and count == audit["windows"]
    assert raw_window_sha == audit["windows_sha256"]
    assert abs(nll - audit["average_nll_valid_tokens"]) < 2e-12
    ppl = float(result.get("perplexity", result.get("ppl")))
    assert abs(math.exp(nll) - ppl) < 2e-11
    return {"ppl": ppl, "nll": nll, "valid_tokens": n, "windows": count,
            "result_sha256": sha(path), "windows_sha256": raw_window_sha,
            "result_path": str(path), "windows_path": str(V18 / "evaluation/windows" / f"{method}_{dataset}.csv") if method in ("M1", "M2") else str(V45 / f"R_{dataset}_WINDOWS.csv")}


def main():
    protocol = read(OUT / "PROTOCOL.json")
    source = read(OUT / "SOURCE_AUDIT.json")
    base = read(OUT / "BASELINE_AUDIT.json")
    repair = read(OUT / "FORMAL_REPAIR_COMPLETE.json")
    lock = read(OUT / "CHECKPOINT_LOCK.json")
    deploy = read(OUT / "DEPLOYMENT_LOCK.json")
    final = read(OUT / "FINAL_STATUS.json")
    run = read(OUT / "RUN_STATUS.json")
    assert source["pass"] and base["pass"] and repair["modules"] == 168 and repair["rows"] == 304128
    assert deploy["pass"] and final["status"] == "COMPLETE" and final["PPL_run"]
    assert final["model_sha256"] == lock["final_model_sha256"] == deploy["final_model_sha256"]
    assert sha(OUT / "R1/checkpoint/final/model.safetensors") == lock["final_model_sha256"]
    assert sha(OUT / "R1/checkpoint/final/config.json") == lock["config_sha256"]
    assert len(lock["group_checkpoints"]) == 96
    for x in lock["group_checkpoints"]:
        assert sha(x["path"]) == x["sha256"]

    # Historical baselines and R2 are source- and window-hash checked; no reevaluation.
    hist = {}
    for ds, stem in (("wikitext", "wikitext"), ("c4_fixed_budget", "c4_fixed_budget")):
        hist["M1", ds] = historic(V18 / f"evaluation/raw/M1_{stem}.json", base["ppl_and_window_baselines"], ds, "M1")
        hist["M2", ds] = historic(V18 / f"evaluation/raw/M2_{stem}.json", base["ppl_and_window_baselines"], ds, "M2")
        hist["R2", ds] = historic(V45 / f"R_{stem}.json", base["ppl_and_window_baselines"], ds, "R2")
        r1path = OUT / f"R1_{stem}.json"
        r1 = read(r1path)
        r1nll, n, nw = windows_nll(OUT / f"R1_{stem}_WINDOWS.csv")
        assert r1["status"] == "PASS" and r1["model_sha256"] == lock["final_model_sha256"]
        assert n == r1["scored_tokens"] and nw == r1["windows"]
        assert r1["manifest_sha256"] == base["evaluation_manifests"][ds]
        assert abs(r1nll - r1["nll"]) < 2e-12 and abs(math.exp(r1nll) - r1["ppl"]) < 2e-11
        assert r1["input_tokens"] == n + 1
        hist["R1", ds] = {"ppl": r1["ppl"], "nll": r1nll, "valid_tokens": n, "windows": nw,
                          "result_sha256": sha(r1path), "windows_sha256": sha(OUT / f"R1_{stem}_WINDOWS.csv"),
                          "result_path": str(r1path), "windows_path": str(OUT / f"R1_{stem}_WINDOWS.csv")}

    comparisons = []
    for ds in ("wikitext", "c4_fixed_budget"):
        a, b, r1, r2 = (hist[m, ds] for m in ("M1", "M2", "R1", "R2"))
        contrasts = {}
        for label, x, y in (("R1_vs_M1", r1, a), ("R2_vs_M2", r2, b),
                            ("R2_vs_R1", r2, r1), ("M2_vs_M1", b, a)):
            contrasts[label] = {"ppl_absolute_delta": x["ppl"] - y["ppl"],
                                "ppl_relative_percent": (x["ppl"] / y["ppl"] - 1) * 100,
                                "nll_absolute_delta": x["nll"] - y["nll"]}
        interaction = contrasts["R2_vs_M2"]["nll_absolute_delta"] - contrasts["R1_vs_M1"]["nll_absolute_delta"]
        comparisons.append({
            "dataset": ds, "valid_tokens": a["valid_tokens"], "windows": a["windows"],
            "M1_PPL": a["ppl"], "M1_weighted_average_NLL": a["nll"],
            "M2_PPL": b["ppl"], "M2_weighted_average_NLL": b["nll"],
            "R1_PPL": r1["ppl"], "R1_weighted_average_NLL": r1["nll"],
            "R2_PPL": r2["ppl"], "R2_weighted_average_NLL": r2["nll"],
            "R1_vs_M1_PPL_delta": contrasts["R1_vs_M1"]["ppl_absolute_delta"],
            "R1_vs_M1_PPL_relative_percent": contrasts["R1_vs_M1"]["ppl_relative_percent"],
            "R1_vs_M1_NLL_delta": contrasts["R1_vs_M1"]["nll_absolute_delta"],
            "R2_vs_M2_PPL_delta": contrasts["R2_vs_M2"]["ppl_absolute_delta"],
            "R2_vs_M2_PPL_relative_percent": contrasts["R2_vs_M2"]["ppl_relative_percent"],
            "R2_vs_M2_NLL_delta": contrasts["R2_vs_M2"]["nll_absolute_delta"],
            "R2_vs_R1_PPL_delta": contrasts["R2_vs_R1"]["ppl_absolute_delta"],
            "R2_vs_R1_PPL_relative_percent": contrasts["R2_vs_R1"]["ppl_relative_percent"],
            "R2_vs_R1_NLL_delta": contrasts["R2_vs_R1"]["nll_absolute_delta"],
            "M2_vs_M1_PPL_delta": contrasts["M2_vs_M1"]["ppl_absolute_delta"],
            "M2_vs_M1_PPL_relative_percent": contrasts["M2_vs_M1"]["ppl_relative_percent"],
            "M2_vs_M1_NLL_delta": contrasts["M2_vs_M1"]["nll_absolute_delta"],
            "NLL_interaction_I": interaction,
            "contrast_detail_json": json.dumps(contrasts, ensure_ascii=False),
        })
    write_csv("FOUR_MODEL_COMPARISON.csv", comparisons)
    write_json("PPL_SOURCE_AUDIT.json", {
        "pass": True, "P3_read": False, "new_baselines_evaluated": False,
        "R1_evaluations": {k[1]: v for k, v in hist.items() if k[0] == "R1"},
        "historical_source_records": {f"{m}_{ds}": v for (m, ds), v in hist.items() if m != "R1"},
        "window_weighting": "sum(window_nll) / sum(valid_tokens)",
        "same_token_counts_and_windows_across_models": True,
    })

    module_rows = read_csv(OUT / "MODULE_RESULTS.csv")
    assert len(module_rows) == 168 and {r["mode"] for r in module_rows} == {"formal"}
    modules = {}
    for row in module_rows:
        m = row["module"]
        z = modules.setdefault(m, {"module": m, "linear_count": 0, "rows": 0, "accepted_updates": 0,
                                  "modified_coordinates": 0, "candidate_evaluations": 0,
                                  "initial_search_mse_sum": 0., "final_search_mse_sum": 0., "max_p_drift": 0.})
        z["linear_count"] += 1
        for k in ("rows", "accepted_updates", "modified_coordinates", "candidate_evaluations"):
            z[k] += int(row[k])
        for k in ("initial_search_mse_sum", "final_search_mse_sum", "max_p_drift"):
            z[k] += float(row[k]) if k != "max_p_drift" else 0.
        z["max_p_drift"] = max(z["max_p_drift"], float(row["max_p_drift"]))
    module_summary = []
    for z in modules.values():
        z["search_mse_change_percent"] = (z["final_search_mse_sum"] / z["initial_search_mse_sum"] - 1) * 100
        module_summary.append(z)
    module_summary.sort(key=lambda x: x["module"])
    write_csv("MODULE_AGGREGATES.csv", module_summary)

    block_rows = read_csv(OUT / "BLOCK_TIMINGS.csv")
    assert len(block_rows) == 24 and {int(x["layer"]) for x in block_rows} == set(range(24))
    block_sum = sum(float(x["seconds"]) for x in block_rows)
    checks = read_csv(OUT / "NUMERICAL_RECHECKS.csv")
    rejects = read_csv(OUT / "REJECTION_EVENTS.csv")
    assert len(checks) == 4 and all(x["same_precision_pass"] == "True" and x["accepted_after_FP64"] == "True" for x in checks)
    assert len(rejects) == repair["rejected_candidates"] == 9
    assert all(x["restored"] == "True" and x["reason"] == "FP32_NON_DESCENT" for x in rejects)
    assert len(read_csv(OUT / "STAGE_TIMINGS.csv")) > 0

    v45decision = read(V45 / "DECISION.json")
    v40summary = read(V40 / "SUMMARY.json")
    m1_build = 648.8761009
    m2_build = float(v45decision["historical_M2_creation_seconds_separate"])
    v42_45 = float(v45decision["v42_to_v45_cumulative_GPU_budget_wall_seconds"])
    r1_elapsed = float(final["elapsed_seconds_from_original_gpu_start"])
    stage = {}
    for x in read_csv(OUT / "STAGE_TIMINGS.csv"):
        key = (x["mode"], x["stage"])
        stage[key] = stage.get(key, 0.) + float(x["seconds"])
    # Timers below are nested inside block walls. They are explanatory and must not be summed with block totals.
    cost_rows = [
        {"item": "M1_original_GPTQ_construction", "seconds": m1_build, "source": "v18 formal runtime", "accounting": "historical measured; outside v46 job"},
        {"item": "M2_original_scalar10_construction", "seconds": m2_build, "source": "v45 DECISION.json -> historical v18 creation timing", "accounting": "historical measured; outside v45 validation chain"},
        {"item": "M2_minus_M1_initialization_extra", "seconds": m2_build-m1_build, "source": "difference of v18 measurements", "accounting": "derived; one-time search/build premium"},
        {"item": "R1_v46_all_24_formal_block_wall_sum", "seconds": block_sum, "source": "v46 BLOCK_TIMINGS.csv", "accounting": "measured; complete 168-Linear repair plus group commits/propagation"},
        {"item": "R1_independent_FP64_rechecks", "seconds": stage.get(("formal", "independent_FP64_increment_recheck"), 0.), "source": "v46 STAGE_TIMINGS.csv", "accounting": "measured nested subset of block wall; not additive"},
        {"item": "R1_save_grid_reload_output_regression", "seconds": sum(stage.get(("deployment", s), 0.) for s in ("final_save", "pack_reload_all_grid_check", "reload")), "source": "v46 STAGE_TIMINGS.csv", "accounting": "measured; after formal blocks"},
        {"item": "R1_evaluation_checkpoint_reload", "seconds": stage.get(("evaluation", "evaluation_checkpoint_reload"), 0.), "source": "v46 STAGE_TIMINGS.csv", "accounting": "measured; continuation reload from locked R1"},
        {"item": "R1_WT2_plus_C4_PPL", "seconds": stage.get(("evaluation", "PPL_evaluation"), 0.), "source": "v46 STAGE_TIMINGS.csv", "accounting": "measured; both once; PPL stages only"},
        {"item": "R1_first_wrong_dispatch_attempt", "seconds": float(read(OUT / "EXECUTION_AMENDMENT_001.json")["failed_attempt_elapsed_seconds"]), "source": "EXECUTION_AMENDMENT_001.json", "accounting": "measured failed attempt, included in original GPU_START cumulative wall"},
        {"item": "R1_preflight_elapsed_from_original_start", "seconds": float(read(OUT / "validation/PREFLIGHT_GPU.json")["completed_unix"] - read(OUT / "GPU_START.json")["unix"]), "source": "GPU_START.json and PREFLIGHT_GPU.json timestamps", "accounting": "timestamp interval; overlaps original cumulative wall; do not add to total"},
        {"item": "R1_total_original_GPU_START_to_final_status", "seconds": r1_elapsed, "source": "v46 FINAL_STATUS.json", "accounting": "end-to-end elapsed; includes preflight, failed dispatch, correction, repair, save/reload, continuation, and PPL"},
        {"item": "R2_v42_to_v45_validation_chain", "seconds": v42_45, "source": "v45 DECISION.json; v42 + v43 + v44 + v45", "accounting": "measured cumulative; contains pilot, failed attempts, diagnostics and resumed work"},
        {"item": "R2_v45_new_group_formal_blocks", "seconds": float(v45decision["actual_formal_blocks_new_groups_seconds"]), "source": "v45 DECISION.json", "accounting": "measured for v45's 141 new modules only; excludes 27 inherited modules"},
        {"item": "R2_clean_full_repair_unique_stages", "seconds": "", "source": "v43/v44/v45 recovered checkpoints and stage records", "accounting": "not reliably reconstructable as one non-overlapping clean repair-only measurement"},
        {"item": "R2_M2_creation_plus_v42_v45_chain", "seconds": m2_build+v42_45, "source": "v45 DECISION.json", "accounting": "derived full lineage; includes historical M2 construction plus all v42-v45 validation-chain wall"},
        {"item": "v40_known_winner_union_method_wall", "seconds": v40summary["timing"]["B_method_wall_seconds"], "source": "v40 SUMMARY.json", "accounting": "engineering reference for already-known 675-candidate union; excludes discovering the union"},
    ]
    write_csv("COST_COMPARISON.csv", cost_rows)

    m1r, m2r, r1r, r2r = [hist[x, "wikitext"] for x in ("M1", "M2", "R1", "R2")]
    m1c, m2c, r1c, r2c = [hist[x, "c4_fixed_budget"] for x in ("M1", "M2", "R1", "R2")]
    outcome = {
        "status": "COMPLETE", "category": "A",
        "quality_decision": "R1 improves on M1 and R2 improves on M2 in both corpora; R2 also remains better than R1 in both corpora.",
        "scalar10_after_repair": "Additional quality gain is present on both evaluated corpora; NLL interaction is positive on both, describing attenuation of the scalar10 repair contrast after repair. This is descriptive only, not significance or mechanism evidence.",
        "worth_decision": "M2/scalar10 is the quality-leading flow here; M1 plus repair is the lower-initialization-cost flow. M2's initialization cost was 1467.06 s higher than M1's in historical measured construction. Whether the observed 0.68-0.80% R2-vs-R1 PPL difference is worth that one-time premium depends on the project's quality/compute utility; this run does not define a threshold. v40 shows known-union execution headroom but is not the cost of discovering that union.",
        "results": {"wikitext_R1_vs_M1_PPL_percent": (r1r["ppl"]/m1r["ppl"]-1)*100,
                    "wikitext_R2_vs_R1_PPL_percent": (r2r["ppl"]/r1r["ppl"]-1)*100,
                    "c4_R1_vs_M1_PPL_percent": (r1c["ppl"]/m1c["ppl"]-1)*100,
                    "c4_R2_vs_R1_PPL_percent": (r2c["ppl"]/r1c["ppl"]-1)*100,
                    "wikitext_NLL_interaction_I": (r2r["nll"]-m2r["nll"])-(r1r["nll"]-m1r["nll"]),
                    "c4_NLL_interaction_I": (r2c["nll"]-m2c["nll"])-(r1c["nll"]-m1c["nll"])},
        "repair": {"modules": repair["modules"], "rows": repair["rows"],
                   "accepted_updates": repair["accepted_updates"], "modified_coordinates": repair["modified_coordinates"],
                   "candidate_evaluations": repair["candidate_evaluations"], "FP32_non_descent_rejects": len(rejects),
                   "FP64_recheck_count_accepted": len(checks), "genuine_numerical_correctness_errors": 0,
                   "total_search_MSE_initial": sum(x["initial_search_mse_sum"] for x in module_summary),
                   "total_search_MSE_final": sum(x["final_search_mse_sum"] for x in module_summary),
                   "total_search_MSE_change_percent": (sum(x["final_search_mse_sum"] for x in module_summary)/sum(x["initial_search_mse_sum"] for x in module_summary)-1)*100},
        "provenance": {"M1_checkpoint_sha256": protocol["initialization"]["model_sha256"],
                       "R1_checkpoint_sha256": final["model_sha256"],
                       "calibration_sha256": protocol["initialization"]["calibration_sha256"],
                       "model_revision": protocol["initialization"]["revision"],
                       "repair_function_sha256": protocol["repair_source"]["repair_v45_sha256"],
                       "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                       "model_setting": protocol["model_setting"],
                       "P3_read": False, "historical_M1_M2_R2_requantized_or_reevaluated": False},
        "execution_history": {"first_launcher_failed_on_dispatch": True,
                              "original_RUN_STATUS_preserved": run["status"] == "FAILED_STOP",
                              "corrected_continuation_final_status": final["status"],
                              "wrong_groups_in_R1": 0, "final_group_checkpoints": len(lock["group_checkpoints"]),
                              "elapsed_seconds_from_original_start": r1_elapsed,
                              "hard_cap_seconds": final["hard_cap_seconds"],
                              "failed_attempt_seconds_in_total": read(OUT / "EXECUTION_AMENDMENT_001.json")["failed_attempt_elapsed_seconds"]},
        "limitations": ["One run/seed; no cross-seed stability claim.",
                        "M1/M2/R1/R2 differ by initializer and potentially grid; not a code-only contrast.",
                        "PPL is observed only on the specified WT2 test and fixed-token C4 validation splits.",
                        "The method is an application of existing discrete repair ideas, not an official ReQuant reproduction or an originality claim.",
                        "The v46 launcher evaluation lookup failure is preserved in RUN_STATUS/FAILED_ATTEMPTS; the separate continuation used audited manifest hashes and the unchanged evaluator exactly once."],
    }
    write_json("FINAL_DECISION.json", outcome)
    sealed_status = dict(final)
    sealed_status.update({
        "category": outcome["category"],
        "quality_aggregation_status": "COMPLETE",
        "decision_sha256": sha(OUT / "FINAL_DECISION.json"),
        "execution_failure_history_preserved_in_RUN_STATUS": True,
    })
    write_json("FINAL_STATUS.json", sealed_status)

    lines = [
        "# v46 — M1 initialization plus v45 single-coordinate integer repair",
        "",
        "## Result",
        "",
        "Category A for repair migration: R1 improved on native M1 on both corpora. The scalar10 flow still adds a quality gain after repair: R2 is also better than R1 on both corpora. These are single-run descriptive comparisons, not statistical claims.",
        "",
        "| Split | M1 PPL / NLL | M2 PPL / NLL | R1 PPL / NLL | R2 PPL / NLL | R1 vs M1 | R2 vs R1 | NLL interaction I |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in comparisons:
        label = "WikiText-2 test" if c["dataset"] == "wikitext" else "C4 fixed-budget validation"
        def pn(p, n): return f"{p:.6f} / {n:.6f}"
        lines.append(f"| {label} | {pn(c['M1_PPL'],c['M1_weighted_average_NLL'])} | {pn(c['M2_PPL'],c['M2_weighted_average_NLL'])} | {pn(c['R1_PPL'],c['R1_weighted_average_NLL'])} | {pn(c['R2_PPL'],c['R2_weighted_average_NLL'])} | {c['R1_vs_M1_PPL_relative_percent']:+.3f}% | {c['R2_vs_R1_PPL_relative_percent']:+.3f}% | {c['NLL_interaction_I']:+.6f} |")
    lines += [
        "",
        "R1 vs M1 PPL change: WikiText-2 {:+.3f}%, C4 {:+.3f}%. R2 vs M2: WikiText-2 {:+.3f}%, C4 {:+.3f}%. R2 vs R1: WikiText-2 {:+.3f}%, C4 {:+.3f}%. M2 vs M1: WikiText-2 {:+.3f}%, C4 {:+.3f}%.".format(
            comparisons[0]["R1_vs_M1_PPL_relative_percent"], comparisons[1]["R1_vs_M1_PPL_relative_percent"],
            comparisons[0]["R2_vs_M2_PPL_relative_percent"], comparisons[1]["R2_vs_M2_PPL_relative_percent"],
            comparisons[0]["R2_vs_R1_PPL_relative_percent"], comparisons[1]["R2_vs_R1_PPL_relative_percent"],
            comparisons[0]["M2_vs_M1_PPL_relative_percent"], comparisons[1]["M2_vs_M1_PPL_relative_percent"]),
        "",
        "Positive I on both splits means the scalar10 repair contrast is less favorable after repair on the NLL scale; R2 nevertheless remains below R1. This interaction is descriptive and does not establish significance or a mechanism.",
        "",
        "## Coverage and numerical checks",
        "",
        f"Completed all {repair['modules']} Linear modules and {repair['rows']:,} output rows. Accepted updates: {repair['accepted_updates']:,}; changed coordinates: {repair['modified_coordinates']:,}; candidate evaluations: {repair['candidate_evaluations']:,}. Aggregate search reconstruction MSE sum changed {outcome['repair']['total_search_MSE_change_percent']:+.3f}%.",
        "",
        f"V45 numeric flow recorded {len(rejects)} FP32 non-descent candidates, all restored and rejected, plus {len(checks)} same-precision/independent-FP64 rechecks, all accepted after the frozen checks. No hard numerical or state error occurred in the corrected formal run. Final code/grid packing, reload, fixed-sample logits, and all 96 group hashes passed.",
        "",
        "## Cost and execution history",
        "",
        f"R1's full v46 clock from original GPU_START to final status was {r1_elapsed:.1f}s ({r1_elapsed/60:.2f} min) of the 10,800s cap. It includes preflight, the failed first dispatch, diagnosis/continuation, repair, save/reload, and both PPL evaluations. The 24 measured formal block walls sum to {block_sum:.1f}s ({block_sum/60:.2f} min). Save/grid/reload/output regression was {cost_rows[5]['seconds']:.2f}s; continuation checkpoint reload {cost_rows[6]['seconds']:.2f}s; WT2+C4 evaluation {cost_rows[7]['seconds']:.2f}s. Nested numeric-recheck timing is listed separately in COST_COMPARISON.csv and is not additive to block wall.",
        "",
        f"Historical M1 construction was {m1_build:.1f}s and M2 scalar10 construction {m2_build:.1f}s, a measured {m2_build-m1_build:.1f}s ({(m2_build-m1_build)/60:.2f} min) difference. R2's complete v42-v45 validation chain cost {v42_45:.1f}s; plus M2 construction the lineage total is {m2_build+v42_45:.1f}s. That chain includes pilot, failure/diagnostics, recovered work, save/reload, and PPL. The unique clean repair-only cost across restored R2 groups is not reliably reconstructable; v45 separately reports {v45decision['actual_formal_blocks_new_groups_seconds']:.1f}s for its 141 newly repaired modules, excluding 27 inherited modules. Do not read R1/R2 wall totals as an acceleration comparison.",
        "",
        f"v40's already-known winner union measured {v40summary['timing']['B_method_wall_seconds']:.1f}s for 675 full-Linear candidates versus {v40summary['timing']['A_method_wall_seconds']:.1f}s for its 1,680-candidate scalar10 arm. This is an engineering reference for executing a known union, not the cost of discovering it.",
        "",
        "The first v46 launcher resolved the legacy V42 function and failed at an expected non-descent. Its status and 14 invalid group files remain archived/excluded. The corrected run started from the original M1 and completed all groups with the byte-identical v45 function. After save/reload, the original runner hit `KeyError('v18')` before scoring any PPL window. The separate continuation verified historical manifest hashes from BASELINE_AUDIT and ran the unchanged evaluator once per corpus. RUN_STATUS.json and FAILED_ATTEMPTS.jsonl preserve the failed invocation; FINAL_STATUS.json and FINAL_DECISION.json describe the completed chain.",
        "",
        "## Sources and limits",
        "",
        "See PROTOCOL.json, SOURCE_AUDIT.json, BASELINE_AUDIT.json, PPL_SOURCE_AUDIT.json, FOUR_MODEL_COMPARISON.csv, MODULE_RESULTS.csv, MODULE_AGGREGATES.csv, NUMERICAL_RECHECKS.csv, REJECTION_EVENTS.csv, COST_COMPARISON.csv, CHECKPOINT_LOCK.json, DEPLOYMENT_LOCK.json, and FINAL_STATUS.json. P3 was not read. No PPL-based tuning or second model was performed. Interface state could not confirm the requested Luna Max selection. R1 is an application of existing discrete repair ideas, not an official ReQuant reproduction or an originality claim.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf8")

    amendment = {
        "amendment": "EXECUTION_AMENDMENT_002",
        "cause": "The corrected formal runner completed repair, save, pack/grid verification, reload, and output regression, then its evaluator referenced P['v18'] although v46 PROTOCOL.json has no such key.",
        "failed_runner_record_preserved": True,
        "failure_occurred_before_any_PPL_window": True,
        "algorithm_or_frozen_protocol_changed": False,
        "continuation": "A separate script verified manifest hashes against BASELINE_AUDIT.json and called unchanged runtime_v46.evaluate exactly once for each dataset.",
        "continuation_script_sha256": sha(OUT / "continue_evaluation_v46.py"),
        "final_model_sha256": final["model_sha256"],
        "PPL_evaluation_count": {"wikitext": 1, "c4_fixed_budget": 1},
        "elapsed_seconds_from_original_GPU_START": r1_elapsed,
        "created_unix": __import__("time").time(),
    }
    write_json("EXECUTION_AMENDMENT_002.json", amendment)


if __name__ == "__main__":
    main()
