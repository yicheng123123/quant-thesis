"""Independent raw-artifact summarizer for the frozen v12 validation.

This post-run entry point deliberately reads Gate inputs from search metrics
only; held-out candidate metrics are joined after all row decisions exist.
It is separate from the frozen execution script so its hash remains intact.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.range_search_w4.results.range_w4_gate_v12 import execute_validation_v12 as run
from experiments.range_search_w4.results.range_w4_gate_v12.prepare_validation_v12 import (
    GATE_SHA, PROJECTIONS, digest_file, tensor_hash,
)

OUT = Path(__file__).resolve().parent
RAW = OUT / "raw_validation"
WEIGHTS = RAW / "candidate_weights"
V11 = OUT.parent / "range_w4_96row_v11"
GRID = [round(i / 10, 1) for i in range(1, 11)]


def write_csv(name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(OUT / name, index=False, float_format="%.15g")


def load_inputs():
    gate, row_manifest, splits, protocol = run.manifests()
    split_by_id = {s["split_id"]: s for s in splits["splits"]}
    row_ids = {run.row_key(r) for r in row_manifest["rows"]}
    files = sorted(RAW.glob("L*_R*_S??.json"))
    run.req(len(files) == 480, f"expected 480 raw records, found {len(files)}")
    raw = {}
    for path in files:
        record = run.load_json(path)
        key = (record["row_id"], record["split_id"])
        run.req(key not in raw, f"duplicate raw record {key}")
        run.req(record["row_id"] in row_ids and record["split_id"] in split_by_id,
                f"unknown row/split {key}")
        run.req(record["gate_v1_sha256"] == GATE_SHA and record["candidate_count"] == 10,
                f"Gate/candidate count {key}")
        run.req(record["row_manifest_sha256"] == protocol["row_manifest_sha256"] and
                record["split_manifest_sha256"] == protocol["split_manifest_sha256"],
                f"manifest identity {key}")
        run.req(record["pristine_weight_hash_before"] == record["pristine_weight_hash_after"] ==
                record["weight_sha256"], f"pristine W changed {key}")
        run.req([float(c["alpha"]) for c in record["candidates"]] == GRID,
                f"candidate grid mismatch {key}")
        qpath = WEIGHTS / f"{record['row_id']}_{record['split_id']}.pt"
        qobj = run.load_tensor_file(qpath)
        run.req(qobj["row_manifest_sha256"] == protocol["row_manifest_sha256"] and
                qobj["split_manifest_sha256"] == protocol["split_manifest_sha256"],
                f"candidate weight manifest {key}")
        for candidate in record["candidates"]:
            alpha = float(candidate["alpha"])
            run.req(tensor_hash(qobj["weights"][str(alpha)]) == candidate["weight_sha256"],
                    f"candidate tensor hash {key}/{alpha}")
        raw[key] = record
    run.req(len(raw) == 48 * 10, "raw row-split count")
    for row_id in row_ids:
        run.req(sum(key[0] == row_id for key in raw) == 10, f"split coverage {row_id}")
    return gate, row_manifest, splits, protocol, raw


def summarize() -> dict:
    gate, row_manifest, splits, protocol, raw = load_inputs()
    row_meta = {run.row_key(r): r for r in row_manifest["rows"]}

    # Gate receives only these four search-side fields. No held-out metric is
    # copied into this table or read while the decisions are calculated.
    search_rows = []
    for (row_id, split_id), record in raw.items():
        for candidate in record["candidates"]:
            search_rows.append({"row_id": row_id, "split_id": split_id,
                                "alpha": float(candidate["alpha"]),
                                "search_sse": float(candidate["search_sse"])})
    search = pd.DataFrame(search_rows)
    gate_fields = {"row_id", "split_id", "alpha", "search_sse"}
    run.req(set(search.columns) == gate_fields and len(search) == 4800,
            "Gate search-only input schema")
    decisions = []
    decision_map = {}
    alpha_by_row_split = {}
    for row_id, group in search.groupby("row_id", sort=True):
        split_stats = []
        for split_id, part in group.groupby("split_id", sort=True):
            ordered = part.sort_values(["search_sse", "alpha"], kind="stable")
            best, second = ordered.iloc[0], ordered.iloc[1]
            baseline = part.loc[part.alpha.eq(1.0)].iloc[0]
            run.req(baseline.search_sse > 0 and second.search_sse > 0,
                    f"zero search denominator {row_id}/{split_id}")
            item = {"split_id": split_id,
                    "search_best_alpha": float(best.alpha),
                    "search_gain": float(1.0 - best.search_sse / baseline.search_sse),
                    "search_margin": float((second.search_sse - best.search_sse) / second.search_sse)}
            split_stats.append(item)
            alpha_by_row_split[(row_id, split_id)] = item["search_best_alpha"]
        gain_med = float(np.median([x["search_gain"] for x in split_stats]))
        margin_med = float(np.median([x["search_margin"] for x in split_stats]))
        accepted = gain_med >= 0.10 and margin_med >= 0.06
        meta = row_meta[row_id]
        decision_map[row_id] = {"accept": accepted, "gain_median": gain_med,
                                "margin_median": margin_med, "splits": split_stats}
        decisions.append({"row_id": row_id, "layer": meta["layer"],
                          "projection": meta["projection"], "row_index": meta["row_index"],
                          "search_gain_median": gain_med, "search_margin_median": margin_med,
                          "gate_decision": "ACCEPT" if accepted else "FALLBACK",
                          "tau_gain": 0.10, "tau_margin": 0.06,
                          "search_best_alpha_by_split": json.dumps(
                              {x["split_id"]: x["search_best_alpha"] for x in split_stats},
                              separators=(",", ":"))})

    # Descriptive discovery/validation distribution comparison. The discovery
    # values are reused as saved; no threshold or row is changed from them.
    discovery_landscape = pd.read_csv(V11 / "full10_alpha_landscapes.csv")
    required_discovery_cols = {"row_id", "split_id", "alpha", "search_sse"}
    run.req(required_discovery_cols.issubset(discovery_landscape.columns) and
            len(discovery_landscape) == 96 * 10 * 10,
            "v11 full10 discovery landscape schema/dimensions")
    discovery_split_metrics = []
    for (row_id, split_id), part in discovery_landscape.groupby(["row_id", "split_id"], sort=True):
        ordered = part.sort_values(["search_sse", "alpha"], kind="stable")
        best, second = ordered.iloc[0], ordered.iloc[1]
        baseline = part.loc[part.alpha.eq(1.0)].iloc[0]
        run.req(baseline.search_sse > 0 and second.search_sse > 0,
                f"v11 zero search denominator {row_id}/{split_id}")
        discovery_split_metrics.append({"row_id": row_id, "split_id": split_id,
                                        "search_gain": 1 - best.search_sse / baseline.search_sse,
                                        "search_margin": (second.search_sse - best.search_sse) / second.search_sse})
    discovery_split_df = pd.DataFrame(discovery_split_metrics)
    discovery_rows = discovery_split_df.groupby("row_id", sort=True).agg(
        search_gain_median=("search_gain", "median"),
        search_margin_median=("search_margin", "median"),
        split_count=("split_id", "nunique"),
    ).reset_index()
    run.req(len(discovery_rows) == 96 and bool((discovery_rows.split_count == 10).all()),
            "v11 discovery row/split dimensions")
    discovery_rows["accepted"] = ((discovery_rows.search_gain_median >= .10) &
                                   (discovery_rows.search_margin_median >= .06))
    run.req(int(discovery_rows.accepted.sum()) == 52,
            "recomputed frozen Gate does not match discovery's 52 ACCEPT rows")
    validation_rows = pd.DataFrame(decisions)
    comparison_rows = []
    for name, frame, accepted_col in (("discovery_v11", discovery_rows, "accepted"),
                                      ("validation_v12", validation_rows, "gate_decision")):
        accept_mask = (frame[accepted_col] if accepted_col == "accepted"
                       else frame[accepted_col].eq("ACCEPT"))
        for metric in ("search_gain_median", "search_margin_median"):
            values = frame[metric].astype(float)
            comparison_rows.append({"dataset": name, "row_count": len(frame),
                                    "accepted_rows": int(accept_mask.sum()),
                                    "fallback_rows": int((~accept_mask).sum()),
                                    "accept_rate": float(accept_mask.mean()),
                                    "metric": metric, "mean": float(values.mean()),
                                    "median": float(values.median()),
                                    "q25": float(values.quantile(.25)),
                                    "q75": float(values.quantile(.75)),
                                    "min": float(values.min()), "max": float(values.max())})
    write_csv("discovery_validation_comparison.csv", comparison_rows)

    # Only after decision_map is complete are held-out candidate fields used.
    per_split, landscape = [], []
    for (row_id, split_id), record in sorted(raw.items()):
        candidates = record["candidates"]
        alpha1 = next(c for c in candidates if float(c["alpha"]) == 1.0)
        selected = next(c for c in candidates
                        if float(c["alpha"]) == alpha_by_row_split[(row_id, split_id)])
        accepted = decision_map[row_id]["accept"]
        gate_selected = selected if accepted else alpha1
        for c in candidates:
            landscape.append({"row_id": row_id, "layer": record["layer"],
                              "projection": record["projection"], "row_index": record["row_index"],
                              "split_id": split_id, "alpha": float(c["alpha"]),
                              "search_sse": c["search_sse"], "search_output_mse": c["search_output_mse"],
                              "heldout_sse": c["heldout_sse"], "heldout_output_mse": c["heldout_output_mse"],
                              "weight_mse": c["weight_mse"],
                              "scale": json.dumps(c["scale"], separators=(",", ":")),
                              "zero": json.dumps(c["zero"], separators=(",", ":")),
                              "candidate_seconds": c["candidate_seconds"],
                              "weight_sha256": c["weight_sha256"],
                              "selected_by_search": float(c["alpha"]) == alpha_by_row_split[(row_id, split_id)]})
        per_split.append({"row_id": row_id, "layer": record["layer"],
                          "projection": record["projection"], "row_index": record["row_index"],
                          "split_id": split_id,
                          "selected_alpha_C": float(selected["alpha"]),
                          "selected_alpha_G": float(gate_selected["alpha"]),
                          "gate_decision": "ACCEPT" if accepted else "FALLBACK",
                          "search_D_sse": alpha1["search_sse"], "search_C_sse": selected["search_sse"],
                          "heldout_D_sse": alpha1["heldout_sse"], "heldout_C_sse": selected["heldout_sse"],
                          "heldout_G_sse": gate_selected["heldout_sse"],
                          "search_gain_C": 1 - selected["search_sse"] / alpha1["search_sse"],
                          "heldout_gain_C": 1 - selected["heldout_sse"] / alpha1["heldout_sse"],
                          "heldout_gain_G": 1 - gate_selected["heldout_sse"] / alpha1["heldout_sse"],
                          "weight_mse_C": selected["weight_mse"],
                          "search_article_ids": json.dumps(record["search_article_ids"]),
                          "heldout_article_ids": json.dumps(record["heldout_article_ids"]),
                          "search_x_sha256": record["search_x_sha256"],
                          "heldout_x_sha256": record["heldout_x_sha256"]})
    perdf, landscape_df = pd.DataFrame(per_split), pd.DataFrame(landscape)
    run.req(len(perdf) == 480 and len(landscape_df) == 4800, "derived table dimensions")
    write_csv("validation_per_row_split.csv", per_split)
    write_csv("validation_alpha_landscapes.csv", landscape)
    write_csv("validation_gate_decisions.csv", decisions)

    d_total = float(perdf.heldout_D_sse.sum())
    method_rows = []
    for method, sse_col, gain_col in (("D", "heldout_D_sse", None),
                                      ("C", "heldout_C_sse", "heldout_gain_C"),
                                      ("G", "heldout_G_sse", "heldout_gain_G")):
        gains = (np.zeros(len(perdf)) if gain_col is None
                 else perdf[gain_col].to_numpy(dtype=float))
        part = perdf[["row_id"]].copy()
        part["gain"] = gains
        row_medians = part.groupby("row_id").gain.median().to_numpy()
        sse = float(perdf[sse_col].sum())
        pooled_gain = 0.0 if method == "D" else 1 - sse / d_total
        method_rows.append({"method": method, "heldout_sse": sse,
                            "pooled_gain_vs_D": pooled_gain,
                            "retained_C_gain": (pooled_gain / (1 - float(perdf.heldout_C_sse.sum()) / d_total)
                                                 if method == "G" and perdf.heldout_C_sse.sum() < d_total else None),
                            "row_count": 48, "row_split_count": 480,
                            "median_row_median_gain": float(np.median(row_medians)),
                            "q25_row_median_gain": float(np.quantile(row_medians, .25)),
                            "q75_row_median_gain": float(np.quantile(row_medians, .75)),
                            "gain_lt_0": int((gains < 0).sum()),
                            "gain_lt_minus_5pct": int((gains < -.05).sum()),
                            "gain_lt_minus_10pct": int((gains < -.10).sum()),
                            "gain_lt_minus_20pct": int((gains < -.20).sum()),
                            "gain_lt_minus_40pct": int((gains < -.40).sum()),
                            "gain_lt_minus_50pct": int((gains < -.50).sum()),
                            "worst_gain": float(gains.min()),
                            "accepted_rows": sum(x["gate_decision"] == "ACCEPT" for x in decisions) if method == "G" else (48 if method == "C" else 0),
                            "fallback_rows": sum(x["gate_decision"] == "FALLBACK" for x in decisions) if method == "G" else (0 if method == "C" else 48)})
    write_csv("D_C_GATE_summary.csv", method_rows)

    def stratified_summary(column: str) -> list[dict]:
        result = []
        for label, part in perdf.groupby(column, sort=True):
            d = float(part.heldout_D_sse.sum())
            c = float(part.heldout_C_sse.sum())
            g = float(part.heldout_G_sse.sum())
            result.append({"stratum": label, "row_count": int(part.row_id.nunique()),
                           "row_split_count": len(part),
                           "gate_accept_rows": int(part.loc[part.gate_decision.eq("ACCEPT"), "row_id"].nunique()),
                           "gate_fallback_rows": int(part.loc[part.gate_decision.eq("FALLBACK"), "row_id"].nunique()),
                           "C_pooled_gain_vs_D": 1 - c / d, "G_pooled_gain_vs_D": 1 - g / d,
                           "C_severe_failures": int((part.heldout_gain_C < -.10).sum()),
                           "G_severe_failures": int((part.heldout_gain_G < -.10).sum()),
                           "C_gain_lt_0": int((part.heldout_gain_C < 0).sum()),
                           "G_gain_lt_0": int((part.heldout_gain_G < 0).sum()),
                           "C_worst_gain": float(part.heldout_gain_C.min()),
                           "G_worst_gain": float(part.heldout_gain_G.min()),
                           "D_heldout_sse": d, "C_heldout_sse": c, "G_heldout_sse": g})
        return result

    projection_summary = stratified_summary("projection")
    layer_summary = stratified_summary("layer")
    write_csv("projection_summary.csv", projection_summary)
    write_csv("layer_summary.csv", layer_summary)
    regressions = perdf[(perdf.heldout_gain_C < -.10) | (perdf.heldout_gain_G < -.10)]
    regressions.to_csv(OUT / "regression_cases.csv", index=False, float_format="%.15g")

    concentration = {}
    contributions = []
    for method, sse_col in (("C", "heldout_C_sse"), ("G", "heldout_G_sse")):
        by_row = perdf.groupby("row_id").agg(D_sse=("heldout_D_sse", "sum"),
                                              method_sse=(sse_col, "sum"))
        by_row["net_sse_reduction"] = by_row.D_sse - by_row.method_sse
        by_row = by_row.sort_values("net_sse_reduction", ascending=False)
        net = float(by_row.net_sse_reduction.sum())
        by_row["share_of_net_reduction"] = by_row.net_sse_reduction / net if net else np.nan
        for rank, (row_id, value) in enumerate(by_row.iterrows(), 1):
            contributions.append({"method": method, "row_id": row_id, "rank": rank,
                                  "D_sse": value.D_sse, "method_sse": value.method_sse,
                                  "net_sse_reduction": value.net_sse_reduction,
                                  "share_of_net_reduction": value.share_of_net_reduction})
        removals = {}
        for n in (1, 5, 10):
            kept = by_row.iloc[n:]
            removals[str(n)] = {"removed_rows": list(by_row.index[:n]),
                                "pooled_gain": 1 - float(kept.method_sse.sum()) / float(kept.D_sse.sum())}
        concentration[method] = {"top_shares": {str(n): float(by_row.net_sse_reduction.iloc[:n].sum() / net)
                                                  for n in (1, 5, 10)} if net else {},
                                 "remove_top_rows": removals,
                                 "top_rows": list(by_row.index[:10])}
    write_csv("pooled_contribution.csv", contributions)

    methods = {x["method"]: x for x in method_rows}
    c_severe, g_severe = methods["C"]["gain_lt_minus_10pct"], methods["G"]["gain_lt_minus_10pct"]
    pass_a = 2 * g_severe <= c_severe
    pass_b = methods["G"]["pooled_gain_vs_D"] > 0
    pass_c = (methods["C"]["pooled_gain_vs_D"] <= 0 or
              methods["G"]["pooled_gain_vs_D"] / methods["C"]["pooled_gain_vs_D"] >= .5)
    decision_status = "GATE_V1_VALIDATION_PASS" if pass_a and pass_b and pass_c else "GATE_V1_VALIDATION_FAIL"

    # Independent aggregation directly from the raw JSON records.
    direct_d = sum(float(r["heldout_D_sse"]) for r in raw.values())
    direct_c = sum(float(r["heldout_C_sse"]) for r in raw.values())
    direct_g = sum(float(r["heldout_C_sse"] if decision_map[r["row_id"]]["accept"]
                          else r["heldout_D_sse"]) for r in raw.values())
    raw_to_summary = (math.isclose(direct_d, methods["D"]["heldout_sse"], rel_tol=1e-12) and
                       math.isclose(direct_c, methods["C"]["heldout_sse"], rel_tol=1e-12) and
                       math.isclose(direct_g, methods["G"]["heldout_sse"], rel_tol=1e-12))

    lock = run.load_json(OUT / "candidate_run_manifest_lock.json")
    frozen_ns = lock.get("locked_at_epoch_ns", 0)
    raw_files = sorted(RAW.glob("L*_R*_S??.json"))
    rows_before_run = (lock.get("row_manifest_mtime_ns", 2**63) < frozen_ns and
                       all(p.stat().st_mtime_ns >= frozen_ns for p in raw_files))
    splits_before_run = (lock.get("split_manifest_mtime_ns", 2**63) < frozen_ns and
                         all(p.stat().st_mtime_ns >= frozen_ns for p in raw_files))
    split_audit = run.verify_split_disjointness(splits)
    v11_rows = run.load_json(V11 / "row_manifest.json")["rows"]
    historical_layers = {int(row["layer"]) for row in v11_rows}
    validation_layers = {int(row["layer"]) for row in row_manifest["rows"]}
    snapshot = run.load_json(OUT / "phaseA_preservation_fingerprint.json")
    phase_a_unchanged = run.check_phase_a_snapshot(snapshot)
    preflight = run.load_json(OUT / "validation_preflight.json")
    gate_only_decisions = all(
        d["gate_decision"] == ("ACCEPT" if d["search_gain_median"] >= .10 and
                                d["search_margin_median"] >= .06 else "FALLBACK")
        for d in decisions)
    candidate_state = all(len(r["candidates"]) == 10 and
                          r["pristine_weight_hash_before"] == r["pristine_weight_hash_after"]
                          for r in raw.values()) and len(landscape_df) == 4800
    checks = {
        "gate_v1_sha256_exact": digest_file(OUT / "GATE_V1.json") == GATE_SHA,
        "validation_row_manifest_frozen_before_runs": rows_before_run,
        "validation_split_manifest_frozen_before_runs": splits_before_run,
        "validation_articles_disjoint_v8_v11": split_audit["historical_overlap"] is False,
        "validation_rows_use_unstudied_layers": validation_layers == {4, 8, 20} and not (validation_layers & historical_layers),
        "search_heldout_disjoint": split_audit["search_heldout_disjoint"],
        "four_projection_alpha1_native_parity": all(x["status"] == "PASS" for x in preflight["parity"].values()),
        "pristine_weight_isolation": all(r["pristine_weight_hash_before"] == r["pristine_weight_hash_after"] == r["weight_sha256"] for r in raw.values()),
        "candidate_state_isolation": candidate_state,
        "gate_reads_search_side_only": set(search.columns) == gate_fields and gate_only_decisions,
        "heldout_never_enters_gate_decision": set(search.columns) == gate_fields,
        "raw_to_summary_independent_recomputation": raw_to_summary,
        "candidate_weight_hashes_valid": len(landscape_df) == 4800,
        "historical_v8_v12_phase_a_files_unchanged": phase_a_unchanged,
        "thresholds_unchanged_after_gate_freeze": lock["gate_sha256"] == GATE_SHA and gate["tau_gain"] == .10 and gate["tau_margin"] == .06,
        "manifests_unchanged_after_first_candidate": lock["row_manifest_sha256"] == digest_file(OUT / "validation_row_manifest.json") and lock["split_manifest_sha256"] == digest_file(OUT / "validation_split_manifest.json") and lock["protocol_sha256"] == digest_file(OUT / "VALIDATION_PROTOCOL.json"),
    }
    correctness = {"status": "PASS" if all(checks.values()) else "FAIL",
                   "checks": checks, "all_checks_passed": sum(checks.values()),
                   "check_count": len(checks), "gate_sha256": digest_file(OUT / "GATE_V1.json"),
                   "row_manifest_sha256": digest_file(OUT / "validation_row_manifest.json"),
                   "split_manifest_sha256": digest_file(OUT / "validation_split_manifest.json"),
                   "raw_records": len(raw), "candidate_weight_records": len(landscape_df),
                   "created_at_utc": run.now_utc()}
    run.save_json(OUT / "VALIDATION_CORRECTNESS.json", correctness)
    status = decision_status if correctness["status"] == "PASS" else "VALIDATION_CORRECTNESS_FAIL_NO_SCIENTIFIC_PASS"
    accepted = sum(x["accept"] for x in decision_map.values())
    costs = run.load_json(OUT / "validation_run_timing.json")
    worst_c = perdf.loc[perdf.heldout_gain_C.idxmin(),
                        ["row_id", "split_id", "heldout_gain_C"]].to_dict()
    worst_g = perdf.loc[perdf.heldout_gain_G.idxmin(),
                        ["row_id", "split_id", "heldout_gain_G"]].to_dict()
    alpha_counts = {str(a): int(sum(x["search_best_alpha"] == a
                                    for item in decision_map.values() for x in item["splits"]))
                    for a in GRID}
    summary = {"status": status, "correctness_status": correctness["status"],
               "primary": {"A_severe_reduction_at_least_50pct": pass_a,
                           "B_gate_pooled_gain_positive": pass_b,
                           "C_retains_at_least_50pct_C_gain_when_applicable": pass_c},
               "metrics": {"D": methods["D"], "C": methods["C"], "G": methods["G"]},
               "accepted_rows": accepted, "fallback_rows": 48 - accepted,
               "search_best_alpha_counts": alpha_counts,
               "discovery_validation_comparison": comparison_rows,
               "worst_regressions": {"C": worst_c, "G": worst_g},
               "pooled_gain_concentration": concentration,
               "projection_summary": projection_summary, "layer_summary": layer_summary,
               "costs": costs, "correctness": correctness,
               "summary_execution_note": "The frozen execute_validation_v12.py --summarize entry point raised AttributeError because its search-only table lacked row metadata; no raw experiment artifact was affected. This companion summarizer independently regenerated all reports from frozen raw JSON and candidate tensors.",
               "created_at_utc": run.now_utc()}
    run.save_json(OUT / "VALIDATION_SUMMARY.json", summary)
    write_report(summary, decisions, regressions.to_dict(orient="records"), concentration)
    return summary


def write_report(summary: dict, decisions: list[dict], regressions: list[dict], concentration: dict) -> None:
    m, cost = summary["metrics"], summary["costs"]
    p = summary["primary"]
    severe_reduction = 1 - m["G"]["gain_lt_minus_10pct"] / max(1, m["C"]["gain_lt_minus_10pct"])
    projection_notes = []
    for x in summary["projection_summary"]:
        csev, gsev = x["C_severe_failures"], x["G_severe_failures"]
        reduction = 1 - gsev / csev if csev else float("nan")
        projection_notes.append(f"{x['stratum']}: severe failures {csev}→{gsev} ({reduction:.1%} reduction)")
    lines = ["# Frozen Gate v1 independent validation report", "",
             f"Final result: **{summary['status']}**. Correctness: **{summary['correctness_status']}**.", "",
             f"1. Gate SHA256 exact match: **{digest_file(OUT / 'GATE_V1.json') == GATE_SHA}** (`{digest_file(OUT / 'GATE_V1.json')}`).",
             f"2. Frozen new rows: **48**, from layers L4/L8/L20 and all four projections; these layers are disjoint from prior v11 discovery layers, with four equal-index buckets per projection/layer.",
             f"3. Validation documents: **80** globally unique full articles, 4 search + 4 held-out per split; disjoint from historical v8-v11 article IDs and content hashes. Split hash `{digest_file(OUT / 'validation_split_manifest.json')}`.",
             f"4. Four-projection native alpha=1 parity: **PASS** for q_proj, o_proj, up_proj and down_proj.",
             f"5. Gate decisions: **{summary['accepted_rows']} ACCEPT / {summary['fallback_rows']} FALLBACK** rows.",
             f"6. Uniform C pooled held-out SSE gain vs D: **{m['C']['pooled_gain_vs_D']:.4%}**.",
             f"7. Gate pooled held-out SSE gain vs D: **{m['G']['pooled_gain_vs_D']:.4%}**.",
             f"8. Gate retained C gain: **{m['G']['retained_C_gain'] if m['G']['retained_C_gain'] is not None else 'n/a (C pooled gain <= 0)'}**.",
             f"9. C severe failures (held gain < -10%): **{m['C']['gain_lt_minus_10pct']} / 480**.",
             f"10. Gate severe failures (held gain < -10%): **{m['G']['gain_lt_minus_10pct']} / 480**.",
             f"11. Severe failures reduced by at least 50%: **{p['A_severe_reduction_at_least_50pct']}** (reduction {severe_reduction:.2%}).",
             f"12. Gate pooled gain is strictly positive: **{p['B_gate_pooled_gain_positive']}**.",
             f"13. Gate retains at least half C gain when applicable: **{p['C_retains_at_least_50pct_C_gain_when_applicable']}**.",
             f"14. All applicable primary conditions pass: **{all(p.values())}**.",
             f"15. Per-projection severe-failure reduction: {'; '.join(projection_notes)}. Only up_proj reaches 50%; this is descriptive and does not alter the global Gate.",
             f"16. Worst held-out regression: **C {summary['worst_regressions']['C']['row_id']} / {summary['worst_regressions']['C']['split_id']} {m['C']['worst_gain']:.2%}; Gate {summary['worst_regressions']['G']['row_id']} / {summary['worst_regressions']['G']['split_id']} {m['G']['worst_gain']:.2%}**.", "",
             "## Discovery vs validation (descriptive only)", "",
             "Gate acceptance, search-gain median and search-margin median by row; this comparison does not retune the frozen thresholds.", "",
             "| Dataset | Rows | Accept/Fallback | Accept rate | Metric | Mean | Median | Q25–Q75 | Min–max |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for x in summary["discovery_validation_comparison"]:
        lines.append(f"| {x['dataset']} | {x['row_count']} | {x['accepted_rows']}/{x['fallback_rows']} | {x['accept_rate']:.1%} | {x['metric']} | {x['mean']:.4f} | {x['median']:.4f} | {x['q25']:.4f}–{x['q75']:.4f} | {x['min']:.4f}–{x['max']:.4f} |")
    lines += ["", "## D / C / Gate results", "",
             "| Method | Pooled gain vs D | <0 | <-5% | <-10% | <-20% | <-40% | <-50% | Median row-median gain | Worst |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ("D", "C", "G"):
        x = m[method]
        lines.append(f"| {method} | {x['pooled_gain_vs_D']:.3%} | {x['gain_lt_0']} | {x['gain_lt_minus_5pct']} | {x['gain_lt_minus_10pct']} | {x['gain_lt_minus_20pct']} | {x['gain_lt_minus_40pct']} | {x['gain_lt_minus_50pct']} | {x['median_row_median_gain']:.3%} | {x['worst_gain']:.2%} |")
    lines += ["", "## Projection breakdown", "",
              "| Projection | Rows | Accept/Fallback | C pooled | Gate pooled | C severe | Gate severe | Worst C | Worst Gate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in summary["projection_summary"]:
        lines.append(f"| {x['stratum']} | {x['row_count']} | {x['gate_accept_rows']}/{x['gate_fallback_rows']} | {x['C_pooled_gain_vs_D']:.3%} | {x['G_pooled_gain_vs_D']:.3%} | {x['C_severe_failures']} | {x['G_severe_failures']} | {x['C_worst_gain']:.2%} | {x['G_worst_gain']:.2%} |")
    lines += ["", "## Layer breakdown", "",
              "| Layer | Rows | Accept/Fallback | C pooled | Gate pooled | C severe | Gate severe | Worst C | Worst Gate |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in summary["layer_summary"]:
        lines.append(f"| L{x['stratum']} | {x['row_count']} | {x['gate_accept_rows']}/{x['gate_fallback_rows']} | {x['C_pooled_gain_vs_D']:.3%} | {x['G_pooled_gain_vs_D']:.3%} | {x['C_severe_failures']} | {x['G_severe_failures']} | {x['C_worst_gain']:.2%} | {x['G_worst_gain']:.2%} |")
    lines += ["", "## Search-best alpha counts across 480 row-splits", "",
              "```json", json.dumps(summary["search_best_alpha_counts"], indent=2), "```", "",
              "## Pooled gain concentration", ""]
    for method in ("C", "G"):
        x = concentration[method]
        if x["top_shares"]:
            lines.append(f"- {method}: top1/top5/top10 row shares = " +
                         "/".join(f"{100*x['top_shares'][str(k)]:.1f}%" for k in (1, 5, 10)) +
                         "; pooled gains after removing top1/top5 = " +
                         f"{x['remove_top_rows']['1']['pooled_gain']:.3%}/{x['remove_top_rows']['5']['pooled_gain']:.3%}.")
    lines.append("Shares use net SSE reduction as denominator; they can exceed 100% when negative row contributions offset positive reductions.")
    lines += ["", "## Cost", "",
              f"- Candidate slots: {cost['completed_candidate_slots']}/{cost['candidate_slots_nominal']}; actual candidate attempts including retries {cost['candidate_evaluation_attempts_including_resumed_retries']}; reused from completed cache {cost['candidate_slots_reused_from_completed_cache']}.",
              f"- Candidate time sum {cost['candidate_seconds_sum_cumulative']:.1f}s; Hessian time sum {cost['hessian_seconds_sum_cumulative_unique_keys']:.1f}s; preflight {cost['preflight_total_seconds']:.1f}s.",
              f"- End-to-end wall time including preflight and input-cache build {cost['total_wall_seconds_including_preflight_and_cache_build']:.1f}s.",
              f"- Peak CUDA allocated/reserved {cost['peak_cuda_allocated_bytes']/2**20:.1f}/{cost['peak_cuda_reserved_bytes']/2**20:.1f} MiB.", "",
              "## Correctness, anomaly and limits", "",
              f"Correctness: {summary['correctness']['all_checks_passed']}/{summary['correctness']['check_count']} checks; details in `VALIDATION_CORRECTNESS.json`.",
              summary["summary_execution_note"],
              "## Reproducibility commands", "",
              "These are the commands used to produce the artifacts; the frozen validation has now failed its preregistered criteria and must not be rerun on the same split.", "",
              "```powershell",
              ".\\.venv-gptq\\Scripts\\python.exe experiments\\range_search_w4\\results\\range_w4_gate_v12\\execute_validation_v12.py --preflight",
              ".\\.venv-gptq\\Scripts\\python.exe experiments\\range_search_w4\\results\\range_w4_gate_v12\\execute_validation_v12.py --run",
              ".\\.venv-gptq\\Scripts\\python.exe experiments\\range_search_w4\\results\\range_w4_gate_v12\\summarize_validation_v12_postrun.py",
              "```", "",
              "The 480 row-split outcomes are repeated measurements clustered within 48 rows, three layers and four projections. No PPL, quantized-prefix, downstream task or other model was run. This report ends at the frozen Gate v1 decision; preserve any negative result and await independent Sol review.", ""]
    (OUT / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    result = summarize()
    print(json.dumps({"status": result["status"],
                      "correctness": result["correctness_status"],
                      "accepted_rows": result["accepted_rows"],
                      "fallback_rows": result["fallback_rows"],
                      "C_pooled_gain": result["metrics"]["C"]["pooled_gain_vs_D"],
                      "G_pooled_gain": result["metrics"]["G"]["pooled_gain_vs_D"],
                      "C_severe": result["metrics"]["C"]["gain_lt_minus_10pct"],
                      "G_severe": result["metrics"]["G"]["gain_lt_minus_10pct"]}, indent=2))
