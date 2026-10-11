"""Independently recompute v14 summaries and preregistered coverage verdict."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from run_v14 import (OUT, RAW, CAND_DIR, CACHE, ACT_CACHE, ROW_PATH, ARTICLE_PATH,
                     K_VALUES, GRID, ROOT, load_json, load_torch, sha_file, tensor_hash,
                     row_key, module_key, candidate_path)

# Explicitly mirror the v14 source trees captured before any candidates.
RESULTS = OUT.parent
TREE_DIRS = {"v8": RESULTS / "range_w4_dynamic_v8", "v9": RESULTS / "range_w4_l23r448_v9",
             "v10": RESULTS / "range_w4_sixrow_v10", "v11": RESULTS / "range_w4_96row_v11",
             "v12": RESULTS / "range_w4_gate_v12", "v13": RESULTS / "range_w4_failure_mechanism_v13"}


def output_sse(x: torch.Tensor, w: torch.Tensor, q: torch.Tensor) -> float:
    err = x.float() @ (q.float() - w.float())
    return float(err.square().sum().item())


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty required CSV {path.name}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def hist_fingerprint(path: Path) -> dict:
    out = {}
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = str(p.relative_to(path)).replace("\\", "/")
        out[rel] = {"sha256": sha_file(p), "size_bytes": p.stat().st_size}
    return out


def compare(a: float, b: float, label: str, abs_tol=1e-6, rel_tol=1e-5) -> dict:
    err = abs(a - b); tol = abs_tol + rel_tol * abs(b)
    return {"label": label, "actual": float(a), "expected": float(b), "abs_error": err,
            "tolerance": tol, "pass": math.isfinite(a) and err <= tol}


def docs_for(cache: dict, row: dict, docs: list[dict], device="cuda") -> list[torch.Tensor]:
    xs = cache["inputs"][module_key(row["layer"], row["projection"])]
    return [xs[d["token_sha256"]].to(device) for d in docs]


def main() -> None:
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows_m, articles, protocol = load_json(ROW_PATH), load_json(ARTICLE_PATH), load_json(OUT / "PROTOCOL.json")
    preflight = load_json(OUT / "PREFLIGHT.json")
    assert protocol["row_manifest_sha256"] == sha_file(ROW_PATH)
    assert protocol["article_manifest_sha256"] == sha_file(ARTICLE_PATH)
    cache = load_torch(ACT_CACHE)
    meta = load_json(CACHE / "activation_cache_metadata.json")
    cache["metadata"].update(meta)
    rows = {row_key(r): r for r in rows_m["rows"]}
    k_articles = {k: articles["calibration"][:k] for k in K_VALUES}
    held_docs = articles["heldout"]
    raw_files = sorted(RAW.glob("L*_K*.json"))
    assert len(raw_files) == 48 * 4
    raw_records = [load_json(p) for p in raw_files]
    assert all(r.get("status") == "COMPLETE" and len(r["candidates"]) == 10 for r in raw_records)
    assert len({(r["row_id"], r["K"]) for r in raw_records}) == 192
    comparison_checks = []
    candidate_count = 0
    landscapes, per_row = [], []
    raw_by_key = {}
    for rec in raw_records:
        rid, k = rec["row_id"], rec["K"]
        row = rows[rid]
        assert rec["row_manifest_sha256"] == protocol["row_manifest_sha256"]
        assert rec["article_manifest_sha256"] == protocol["article_manifest_sha256"]
        assert rec["pristine_weight_sha256"] == row["weight_hash"]
        assert rec["calibration_article_ids"] == [d["article_id"] for d in k_articles[k]]
        assert rec["heldout_article_ids"] == [d["article_id"] for d in held_docs]
        assert len(set(rec["calibration_article_ids"]) & set(rec["heldout_article_ids"])) == 0
        w = cache["weights"][rid].cuda()
        assert tensor_hash(cache["weights"][rid]) == row["weight_hash"]
        xs_docs = docs_for(cache, row, k_articles[k])
        xs = torch.cat(xs_docs)
        cand_eval = []
        for cand in sorted(rec["candidates"], key=lambda x: x["alpha"]):
            cp = candidate_path(rid, k, cand["alpha"])
            qobj = load_torch(cp)
            assert qobj["row_manifest_sha256"] == protocol["row_manifest_sha256"]
            assert qobj["article_manifest_sha256"] == protocol["article_manifest_sha256"]
            assert qobj["row_id"] == rid and qobj["K"] == k and qobj["alpha"] == cand["alpha"]
            q = qobj["weight"]
            assert tensor_hash(q) == cand["weight_sha256"]
            assert cand["pristine_weight_hash_after"] == row["weight_hash"]
            assert cand["hinv_sha256"] == rec["hessian_sha256"]
            independent_sse = output_sse(xs, w, q.cuda())
            comparison_checks.append(compare(independent_sse, cand["search_sse"], f"{rid}/K{k}/alpha{cand['alpha']}/search"))
            cand_eval.append({**cand, "independent_search_sse": independent_sse, "weight": q})
            candidate_count += 1
        selected = min(cand_eval, key=lambda c: (c["independent_search_sse"], c["alpha"]))
        d_obj = next(c for c in cand_eval if c["alpha"] == 1.0)
        assert selected["alpha"] == rec["selected_alpha"], f"search-only alpha mismatch {rid}/K{k}"
        assert abs(1 - selected["independent_search_sse"] / d_obj["independent_search_sse"] - rec["search_gain_vs_D"]) < 1e-7
        assert rec["hessian_sha256"] == d_obj["hinv_sha256"]
        # Held-out recomputation is performed only for the frozen D_K and C_K weights.
        held_xdocs = docs_for(cache, row, held_docs)
        d_article, c_article = [], []
        for xdoc in held_xdocs:
            d_article.append(output_sse(xdoc, w, d_obj["weight"].cuda()))
            c_article.append(output_sse(xdoc, w, selected["weight"].cuda()))
        for i, (dval, cval) in enumerate(zip(d_article, c_article)):
            comparison_checks.append(compare(dval, rec["heldout_D"]["article_sse"][i], f"{rid}/K{k}/H{i+1}/D"))
            comparison_checks.append(compare(cval, rec["heldout_C"]["article_sse"][i], f"{rid}/K{k}/H{i+1}/C"))
        dsum, csum = sum(d_article), sum(c_article)
        comparison_checks.append(compare(dsum, rec["heldout_D"]["pooled_sse"], f"{rid}/K{k}/pooled_D"))
        comparison_checks.append(compare(csum, rec["heldout_C"]["pooled_sse"], f"{rid}/K{k}/pooled_C"))
        gain = 1 - csum / dsum
        comparison_checks.append(compare(gain, rec["heldout_gain_C_vs_D"], f"{rid}/K{k}/gain"))
        alpha_sse = {c["alpha"]: c["independent_search_sse"] for c in cand_eval}
        d_search = alpha_sse[1.0]
        for c in cand_eval:
            landscapes.append({"row_id": rid, "layer": row["layer"], "projection": row["projection"],
                               "row_index": row["row_index"], "K": k, "alpha": c["alpha"],
                               "search_sse": c["independent_search_sse"],
                               "search_mse": c["independent_search_sse"] / xs.shape[0],
                               "relative_search_gain_vs_alpha1": 1 - c["independent_search_sse"] / d_search,
                               "selected_alpha": selected["alpha"], "is_search_best": c["alpha"] == selected["alpha"],
                               "weight_mse": c["weight_mse"], "candidate_seconds": c["candidate_seconds"],
                               "weight_sha256": c["weight_sha256"]})
        article_gain = [1 - cval / dval if dval > 0 else 0.0 for cval, dval in zip(c_article, d_article)]
        sorted_shares = sorted((v / csum for v in c_article), reverse=True)
        per_row.append({"row_id": rid, "layer": row["layer"], "projection": row["projection"],
                        "row_index": row["row_index"], "bucket": row["bucket"], "K": k,
                        "selected_alpha": selected["alpha"], "search_gain_vs_D": 1 - selected["independent_search_sse"] / d_search,
                        "D_heldout_sse": dsum, "D_heldout_mse": dsum / (32 * 128),
                        "C_heldout_sse": csum, "C_heldout_mse": csum / (32 * 128),
                        "heldout_gain": gain, "severe_lt_minus10pct": gain < -.10,
                        "catastrophic_lt_minus40pct": gain < -.40,
                        "candidate_count": len(cand_eval), "hessian_sha256": rec["hessian_sha256"],
                        "candidate_seconds": sum(c["candidate_seconds"] for c in cand_eval),
                        "heldout_evaluation_seconds": rec["heldout_evaluation_seconds"],
                        "article_gain_min": min(article_gain), "article_gain_max": max(article_gain)})
        # preserve needed per-article SSE in a separate report row
        per_row[-1]["_d_article_sse"] = d_article
        per_row[-1]["_c_article_sse"] = c_article
        per_row[-1]["_article_gain"] = article_gain
        raw_by_key[(rid, k)] = rec
        del w, xs, xs_docs, held_xdocs, cand_eval, selected, d_obj
    # Fail closed before writing any scientific summaries if artifact-level correctness fails.
    assert all(c["pass"] for c in comparison_checks), next(c for c in comparison_checks if not c["pass"])
    assert candidate_count == 1920

    # Required primary row-level table.
    public_per_row = [{k: v for k, v in r.items() if not k.startswith("_")} for r in per_row]
    save_csv(OUT / "coverage_per_row.csv", public_per_row)
    save_csv(OUT / "coverage_alpha_landscapes.csv", landscapes)

    # D_K native baseline changes are reported separately from relative C_K gains.
    by_idk = {(r["row_id"], r["K"]): r for r in per_row}
    d4 = {r["row_id"]: r["D_heldout_sse"] for r in per_row if r["K"] == 4}
    baseline_rows = []
    for r in per_row:
        baseline_rows.append({"row_id": r["row_id"], "layer": r["layer"], "projection": r["projection"],
                              "row_index": r["row_index"], "K": r["K"], "D_heldout_sse": r["D_heldout_sse"],
                              "D_heldout_mse": r["D_heldout_mse"],
                              "improvement_vs_D4": 1 - r["D_heldout_sse"] / d4[r["row_id"]]})
    save_csv(OUT / "coverage_baseline_D.csv", baseline_rows)

    # Alpha stability across the frozen K schedule.
    stability = []
    by_row = defaultdict(dict)
    for r in per_row: by_row[r["row_id"]][r["K"]] = r
    for rid, vals in by_row.items():
        alphas = [vals[k]["selected_alpha"] for k in K_VALUES]
        stability.append({"row_id": rid, "layer": vals[4]["layer"], "projection": vals[4]["projection"],
                          "row_index": vals[4]["row_index"],
                          **{f"alpha_star_K{k}": vals[k]["selected_alpha"] for k in K_VALUES},
                          "all_four_exact_agreement": len(set(alphas)) == 1,
                          "absolute_change_K4_to_K32": abs(alphas[-1] - alphas[0]),
                          "closer_to_1_at_K32": abs(1 - alphas[-1]) < abs(1 - alphas[0]),
                          "stable_within_one_grid_K16_to_K32": abs(alphas[-1] - alphas[-2]) <= .1,
                          "adjacent_alpha_changes": ";".join(f"{abs(alphas[i+1]-alphas[i]):.1f}" for i in range(3))})
    save_csv(OUT / "coverage_alpha_stability.csv", stability)

    # Search landscapes at adjacent K. Ties use average ranks; top-2 overlap is Jaccard.
    ranking = []
    landscape_by_row_k = defaultdict(lambda: defaultdict(dict))
    for x in landscapes: landscape_by_row_k[x["row_id"]][x["K"]][x["alpha"]] = x["search_sse"]
    for rid, kvals in landscape_by_row_k.items():
        sample = by_row[rid][4]
        for ka, kb in [(4, 8), (8, 16), (16, 32)]:
            la, lb = kvals[ka], kvals[kb]
            aa = np.array([la[a] for a in GRID]); bb = np.array([lb[a] for a in GRID])
            rho = float(spearmanr(rankdata(aa), rankdata(bb)).statistic)
            topa, topb = set(np.argsort(aa, kind="stable")[:2]), set(np.argsort(bb, kind="stable")[:2])
            jaccard = len(topa & topb) / len(topa | topb)
            besta, bestb = GRID[int(np.argmin(aa))], GRID[int(np.argmin(bb))]
            ranking.append({"row_id": rid, "layer": sample["layer"], "projection": sample["projection"],
                            "row_index": sample["row_index"], "K_from": ka, "K_to": kb,
                            "spearman_alpha_ranking": rho, "best_alpha_from": besta,
                            "best_alpha_to": bestb, "best_alpha_agreement": besta == bestb,
                            "top2_jaccard": jaccard})
    save_csv(OUT / "coverage_ranking_stability.csv", ranking)

    # Article concentration uses outcomes only, after C_K and D_K have been selected/fixed.
    concentration = []
    for r in per_row:
        ds, cs, gains = r["_d_article_sse"], r["_c_article_sse"], r["_article_gain"]
        ctotal = sum(cs)
        shares = sorted((x / ctotal for x in cs), reverse=True)
        worst_idx = int(np.argmin(gains))
        concentration.append({"row_id": r["row_id"], "layer": r["layer"], "projection": r["projection"],
                              "row_index": r["row_index"], "K": r["K"], "selected_alpha": r["selected_alpha"],
                              "heldout_gain_pooled": r["heldout_gain"],
                              "top1_article_C_sse_share": shares[0], "top3_articles_C_sse_share": sum(shares[:3]),
                              "top5_articles_C_sse_share": sum(shares[:5]),
                              "worst_article_id": held_docs[worst_idx]["article_id"],
                              "worst_article_gain": gains[worst_idx],
                              "worst_article_D_sse": ds[worst_idx], "worst_article_C_sse": cs[worst_idx],
                              **{f"H{i+1:02d}_D_sse": ds[i] for i in range(32)},
                              **{f"H{i+1:02d}_C_sse": cs[i] for i in range(32)},
                              **{f"H{i+1:02d}_gain": gains[i] for i in range(32)}})
    save_csv(OUT / "article_concentration.csv", concentration)

    def agg(frame: list[dict]) -> dict:
        g = np.asarray([x["heldout_gain"] for x in frame], dtype=np.float64)
        d = sum(x["D_heldout_sse"] for x in frame); c = sum(x["C_heldout_sse"] for x in frame)
        return {"row_count": len(frame), "positive_row_count": int((g > 0).sum()),
                "gain_lt_0_count": int((g < 0).sum()), "gain_lt_minus5pct_count": int((g < -.05).sum()),
                "gain_lt_minus10pct_count": int((g < -.10).sum()), "gain_lt_minus20pct_count": int((g < -.20).sum()),
                "gain_lt_minus40pct_count": int((g < -.40).sum()), "median_row_gain": float(np.median(g)),
                "mean_row_gain": float(g.mean()), "worst_row_gain": float(g.min()),
                "pooled_D_sse": float(d), "pooled_C_sse": float(c),
                "pooled_D_mse": float(d / (len(frame) * 32 * 128)),
                "pooled_C_mse": float(c / (len(frame) * 32 * 128)),
                "pooled_C_vs_D_gain": float(1 - c / d),
                "selected_alpha_distribution": json.dumps({str(a): int(sum(x["selected_alpha"] == a for x in frame)) for a in GRID}, sort_keys=True),
                "mean_selected_alpha": float(np.mean([x["selected_alpha"] for x in frame]))}

    summary_rows = []
    for k in K_VALUES:
        fr = [x for x in per_row if x["K"] == k]
        summary_rows.append({"K": k, **agg(fr)})
    # D_K pooled change against D4 is separate from C_K relative gain.
    d4_total = next(r["pooled_D_sse"] for r in summary_rows if r["K"] == 4)
    for r in summary_rows:
        r["D_K_improvement_vs_D4"] = 1 - r["pooled_D_sse"] / d4_total
        r["coverage_activation_forward_seconds"] = sum(cache["metadata"]["forward_seconds_by_article"][f"C{i:02d}"] for i in range(1, r["K"]+1)) + sum(cache["metadata"]["forward_seconds_by_article"][f"H{i:02d}"] for i in range(1, 33))
    save_csv(OUT / "coverage_summary.csv", summary_rows)

    projection_summary, layer_summary = [], []
    for k in K_VALUES:
        for projection in ("q_proj", "o_proj", "up_proj", "down_proj"):
            fr = [x for x in per_row if x["K"] == k and x["projection"] == projection]
            projection_summary.append({"projection": projection, "K": k, **agg(fr)})
        for layer in (2, 6, 14, 18):
            fr = [x for x in per_row if x["K"] == k and x["layer"] == layer]
            layer_summary.append({"layer": layer, "K": k, **agg(fr)})
    save_csv(OUT / "projection_summary.csv", projection_summary)
    save_csv(OUT / "layer_summary.csv", layer_summary)

    regression_rows = []
    for r in per_row:
        if r["heldout_gain"] < 0:
            regression_rows.append({k: r[k] for k in ["row_id", "layer", "projection", "row_index", "bucket", "K",
                "selected_alpha", "search_gain_vs_D", "D_heldout_sse", "C_heldout_sse", "heldout_gain",
                "severe_lt_minus10pct", "catastrophic_lt_minus40pct", "article_gain_min", "article_gain_max"]})
    save_csv(OUT / "regression_cases.csv", regression_rows if regression_rows else [{"row_id": "NONE"}])

    # Cost per K: activation forward contribution + model setup/cache storage pro-rated by articles, Hessian, candidates, held-out scoring.
    hessian_file = load_json(OUT / "formal_hessian_timing.json")
    hessian_builds = hessian_file["builds"]
    activation_meta = cache["metadata"]
    fixed_capture_setup = activation_meta.get("model_load_seconds", 0.0) + activation_meta.get("cache_save_seconds", 0.0)
    cost_rows = []
    for k in K_VALUES:
        forwards = sum(activation_meta["forward_seconds_by_article"][f"C{i:02d}"] for i in range(1,k+1)) + sum(activation_meta["forward_seconds_by_article"][f"H{i:02d}"] for i in range(1,33))
        capture_sec = forwards + fixed_capture_setup * ((k + 32) / 64)
        h_rows = [x for x in hessian_builds if x["K"] == k]
        hess_sec = sum(x["hessian_build_seconds"] for x in h_rows)
        fr = [x for x in per_row if x["K"] == k]
        cand_sec = sum(x["candidate_seconds"] for x in fr)
        held_sec = sum(x["heldout_evaluation_seconds"] for x in fr)
        total = capture_sec + hess_sec + cand_sec + held_sec
        cand_raw = [c for r in raw_records if r["K"] == k for c in r["candidates"]]
        peak_alloc = max([activation_meta["peak_allocated_bytes"]] + [x["peak_allocated_bytes"] for x in h_rows] + [x["peak_allocated_bytes"] for x in cand_raw] + [r["heldout_peak_allocated_bytes"] for r in raw_records if r["K"] == k])
        peak_reserved = max([activation_meta["peak_reserved_bytes"]] + [x["peak_reserved_bytes"] for x in h_rows] + [x["peak_reserved_bytes"] for x in cand_raw] + [r["heldout_peak_reserved_bytes"] for r in raw_records if r["K"] == k])
        cost_rows.append({"K": k, "activation_capture_seconds_attributed": capture_sec,
                          "activation_forward_seconds_measured": forwards,
                          "capture_setup_and_cache_save_seconds_prorated": fixed_capture_setup * ((k+32)/64),
                          "hessian_build_seconds": hess_sec, "hessian_build_count": len(h_rows),
                          "candidate_seconds": cand_sec, "candidate_count": len(cand_raw),
                          "heldout_evaluation_seconds": held_sec, "total_calibration_seconds": total,
                          "cost_multiplier_vs_K4": None, "peak_allocated_bytes": peak_alloc,
                          "peak_reserved_bytes": peak_reserved})
    k4_total = cost_rows[0]["total_calibration_seconds"]
    for c in cost_rows: c["cost_multiplier_vs_K4"] = c["total_calibration_seconds"] / k4_total
    save_csv(OUT / "cost_summary.csv", cost_rows)

    # Frozen diagnostic coverage support rule; no model/threshold tuning occurs.
    s4, s8, s16, s32 = summary_rows
    if s4["gain_lt_minus10pct_count"] > 0:
        gate_a = s32["gain_lt_minus10pct_count"] <= .5 * s4["gain_lt_minus10pct_count"]
        gate_a_definition = "K32 severe count <= 50% of K4 severe count"
    else:
        gate_a = (s32["gain_lt_minus40pct_count"] < s4["gain_lt_minus40pct_count"] and
                  s32["worst_row_gain"] > s4["worst_row_gain"])
        gate_a_definition = "K4 has zero severe cases: require both fewer K32 catastrophic cases and a better K32 worst gain"
    checks_gate = {"A_severe_reduction": bool(gate_a), "B_K32_pooled_gain_positive": s32["pooled_C_vs_D_gain"] > 0,
                   "C_K32_median_row_gain_nonnegative": s32["median_row_gain"] >= 0,
                   "D_K16_and_K32_severe_below_K4": s16["gain_lt_minus10pct_count"] < s4["gain_lt_minus10pct_count"] and s32["gain_lt_minus10pct_count"] < s4["gain_lt_minus10pct_count"]}
    verdict = "COVERAGE_HYPOTHESIS_SUPPORTED" if all(checks_gate.values()) else "COVERAGE_HYPOTHESIS_NOT_SUPPORTED"

    # Verify all frozen historical result trees remain bit-identical.
    before = load_json(OUT / "historical_fingerprints_before.json")
    historical_matches, changed = {}, {}
    for name, path in TREE_DIRS.items():
        cur = hist_fingerprint(path)
        historical_matches[name] = cur == before[name]
        if cur != before[name]:
            old_keys, new_keys = set(before[name]), set(cur)
            changed[name] = {"missing": sorted(old_keys-new_keys), "new": sorted(new_keys-old_keys),
                             "modified": sorted(k for k in old_keys & new_keys if before[name][k] != cur[k])[:20]}
    all_hist_unchanged = all(historical_matches.values())
    article_old = articles["historical_blacklist"]
    new_docs = articles["calibration"] + articles["heldout"]
    old_ids = {x["source_article_index"] for x in article_old}
    old_hashes = {h for x in article_old for h in (x.get("article_sha256"), x.get("token_sha256")) if h}
    live_disjoint = not ({x["source_article_index"] for x in new_docs} & old_ids) and not ({h for x in new_docs for h in (x["article_sha256"],x["token_sha256"])} & old_hashes)
    projection_parity = len(preflight["projection_parity"]) == 4 and all(p["status"] == "PASS" and all(p["checks"].values()) for p in preflight["projection_parity"])
    check_obj = {"manifest": {"row_manifest_sha256": sha_file(ROW_PATH), "article_manifest_sha256": sha_file(ARTICLE_PATH),
                               "row_manifest_frozen_before_candidates": sha_file(ROW_PATH) == protocol["row_manifest_sha256"],
                               "article_manifest_frozen_before_candidates": sha_file(ARTICLE_PATH) == protocol["article_manifest_sha256"],
                               "rows_48_and_articles_64": len(rows) == 48 and len(new_docs) == 64,
                               "new_layers_and_rows_fixed": set(rows_m["layers"]) == {2,6,14,18},
                               "all_articles_disjoint_v8_v13": live_disjoint,
                               "calibration_heldout_disjoint": not ({d["source_article_index"] for d in articles["calibration"]} & {d["source_article_index"] for d in articles["heldout"]}),
                               "nested_K_sets_exact": articles["K_membership"] == {f"K{k}": [f"C{i:02d}" for i in range(1,k+1)] for k in K_VALUES}},
                  "preflight": {"four_projection_alpha1_parity": projection_parity,
                                "preflight_status": preflight["status"]},
                  "execution": {"nominal_candidates": 1920, "reusable_candidates": 0,
                                "completed_candidates": candidate_count, "candidate_hashes_valid": candidate_count == 1920,
                                "pristine_W_isolation": all(c["pristine_weight_hash_after"] == rr["pristine_weight_sha256"] for rr in raw_records for c in rr["candidates"]),
                                "candidate_H_isolation": all(c["hinv_sha256"] == rr["hessian_sha256"] for rr in raw_records for c in rr["candidates"]),
                                "candidate_state_isolation": all(c["pristine_weight_hash_after"] == rr["pristine_weight_sha256"] and c["hinv_sha256"] == rr["hessian_sha256"] for rr in raw_records for c in rr["candidates"]),
                                "C_K_only_reads_search_data": all(all(not any(key.startswith("heldout") for key in c) for c in rr["candidates"]) for rr in raw_records),
                                "C_K_selection_from_search_only": all(min(rr["candidates"], key=lambda c:(c["search_sse"],c["alpha"]))["alpha"] == rr["selected_alpha"] for rr in raw_records),
                                "heldout_never_selects_alpha": all(min(rr["candidates"], key=lambda c:(c["search_sse"],c["alpha"]))["alpha"] == rr["selected_alpha"] for rr in raw_records),
                                "D_K_and_C_K_share_H_and_calibration": all(rr["hessian_sha256"] == next(c["hinv_sha256"] for c in rr["candidates"] if c["alpha"] == 1.0) and rr["calibration_article_ids"] == [d["article_id"] for d in k_articles[rr["K"]]] for rr in raw_records),
                                "raw_to_summary_independent_recomputation": all(c["pass"] for c in comparison_checks),
                                "historical_v8_v13_trees_unchanged": all_hist_unchanged,
                                "no_gate_or_robust_selector_activated": all(term not in (OUT / "run_v14.py").read_text(encoding="utf-8") for term in ["GATE_V1.json", "CVaR", "worst_article_objective", "mean_plus_std"]),
                                "primary_table_rows": len(per_row)},
                  "historical_tree_matches": historical_matches,
                  "changed_historical_paths": changed,
                  "independent_comparison_count": len(comparison_checks),
                  "independent_comparison_max_abs_error": max(c["abs_error"] for c in comparison_checks),
                  "status": "PASS" if (projection_parity and all_hist_unchanged and all(c["pass"] for c in comparison_checks) and candidate_count == 1920 and live_disjoint) else "FAIL"}
    # If any key correctness gate failed, preserve the evidence and stop before scientific summary/report.
    save_json(OUT / "correctness_checks.json", check_obj)
    if check_obj["status"] != "PASS":
        raise SystemExit("Correctness check failed; scientific summary intentionally not generated")

    gate_provenance = {"record_type": "post_run_provenance_annotation",
                       "source": "user-provided v14 instruction attachment received before manifest preparation and candidate execution",
                       "source_attachment_name": "已粘贴的文本.txt",
                       "source_attachment_sha256": "EB3FE621BE642B775D2E11B404626A419F115963E574B80562CCDE01DB3165AF",
                       "protocol_json_note": "The frozen PROTOCOL.json omitted serialization of the coverage support gate. This annotation records the exact pre-run user-specified conditions and does not add or alter any threshold after observing results.",
                       "criteria": {"A": "if K4 severe count > 0: K32 severe count <= 50% of K4 severe count; if K4 severe count == 0, compare catastrophic count and worst gain instead of dividing by zero",
                                   "B": "K32 pooled C32 vs D32 gain > 0",
                                   "C": "K32 median row gain >= 0",
                                   "D": "K16 and K32 severe regression counts are both below K4"}}
    save_json(OUT / "coverage_support_gate_provenance.json", gate_provenance)
    summary_obj = {"status": verdict, "coverage_support_gate": {"rule": gate_provenance["criteria"],
                                                                     "provenance_file": "coverage_support_gate_provenance.json",
                                                                     "criterion_A_definition": gate_a_definition,
                                                                     "checks": checks_gate,
                                                                     "all_pass": all(checks_gate.values())},
                   "candidate_workload": {"nominal": 1920, "reusable": 0, "new": 1920, "completed": candidate_count},
                   "per_K": summary_rows,
                   "alpha_stability": {"rows": len(stability), "all_four_exact_agreement": sum(x["all_four_exact_agreement"] for x in stability),
                                       "median_abs_change_K4_to_K32": float(np.median([x["absolute_change_K4_to_K32"] for x in stability])),
                                       "closer_to_1_K32_count": sum(x["closer_to_1_at_K32"] for x in stability),
                                       "stable_within_one_grid_K16_K32_count": sum(x["stable_within_one_grid_K16_to_K32"] for x in stability)},
                   "ranking_stability": {f"K{a}_K{b}": {"median_spearman": float(np.median([x["spearman_alpha_ranking"] for x in ranking if x["K_from"]==a and x["K_to"]==b])),
                                                            "best_alpha_agreement_count": sum(x["best_alpha_agreement"] for x in ranking if x["K_from"]==a and x["K_to"]==b),
                                                            "median_top2_jaccard": float(np.median([x["top2_jaccard"] for x in ranking if x["K_from"]==a and x["K_to"]==b]))} for a,b in [(4,8),(8,16),(16,32)]},
                   "article_concentration": {str(k): {"median_top1_share": float(np.median([x["top1_article_C_sse_share"] for x in concentration if x["K"]==k])),
                                                       "median_top3_share": float(np.median([x["top3_articles_C_sse_share"] for x in concentration if x["K"]==k])),
                                                       "median_top5_share": float(np.median([x["top5_articles_C_sse_share"] for x in concentration if x["K"]==k])),
                                                       "median_worst_article_gain": float(np.median([x["worst_article_gain"] for x in concentration if x["K"]==k]))} for k in K_VALUES},
                   "costs": cost_rows,
                   "timing": {"analysis_seconds": time.perf_counter()-started,
                              "preflight_estimated_formal_seconds": preflight["cost_estimate"]["estimated_formal_seconds"],
                              "formal_runner_wall_seconds": load_json(OUT/"formal_run_timing.json")["wall_seconds_including_hessian_builds"],
                              "formal_hessian_wall_seconds": hessian_file["wall_seconds"]}}
    save_json(OUT / "SUMMARY.json", summary_obj)
    save_csv(OUT / "coverage_alpha_stability.csv", stability)
    save_csv(OUT / "coverage_ranking_stability.csv", ranking)
    save_csv(OUT / "article_concentration.csv", concentration)
    save_csv(OUT / "cost_summary.csv", cost_rows)
    # Store non-CSV readable map from artifact layer/projection strata as required.
    report = make_report(summary_obj, projection_summary, layer_summary, correctness=check_obj)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"correctness": check_obj["status"], "verdict": verdict,
                      "per_K": summary_rows, "costs": cost_rows}, indent=2))


def make_report(summary: dict, projection_summary: list[dict], layer_summary: list[dict], correctness: dict) -> str:
    rows = summary["per_K"]
    costs = summary["costs"]
    def pct(x): return f"{100*x:.2f}%"
    lines = ["# v14 — Calibration Coverage Scaling", "",
             "## Protocol and correctness", "",
             "Frozen experiment: Qwen/Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`, FP16 model weights, FP32 GPTQ/Hessian/output loss, symmetric W4A16, G128, block 128, damping 0.1, desc_act disabled, shared alpha grid 0.1–1.0. Search inputs are FP-model projection activations; no quantized prefix is used.",
             "", f"Correctness status: **{correctness['status']}**. Four projection alpha=1 native parity checks passed; {correctness['execution']['completed_candidates']} new candidate results completed and independently recomputed. Historical v8–v13 result trees are {('unchanged' if correctness['execution']['historical_v8_v13_trees_unchanged'] else 'CHANGED')}.",
             "", "Every K has its own H_K and D_K. C_K selects alpha only from mean search output MSE over the first K calibration articles, then D_K and C_K are evaluated on the same 32 held-out articles. Candidate weights are separate per alpha and the raw candidate tensors passed their hashes.",
             "", "## Coverage scaling", "",
             "| K | positive rows | <0 | <−5% | <−10% | <−20% | <−40% | median row gain | mean row gain | worst row gain | pooled D SSE/MSE | pooled C SSE/MSE | pooled C vs D | pooled D improvement vs D4 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['K']} | {r['positive_row_count']} | {r['gain_lt_0_count']} | {r['gain_lt_minus5pct_count']} | {r['gain_lt_minus10pct_count']} | {r['gain_lt_minus20pct_count']} | {r['gain_lt_minus40pct_count']} | {pct(r['median_row_gain'])} | {pct(r['mean_row_gain'])} | {pct(r['worst_row_gain'])} | {r['pooled_D_sse']:.6g} / {r['pooled_D_mse']:.8g} | {r['pooled_C_sse']:.6g} / {r['pooled_C_mse']:.8g} | {pct(r['pooled_C_vs_D_gain'])} | {pct(r['D_K_improvement_vs_D4'])} |")
    lines += ["", "The row is the unit of analysis: 48 fixed rows per K, 192 row×K observations total. Article-level measurements are diagnostics within each row and are not treated as independent rows.",
              "", "## Q1–Q5: severe regressions, worst case, pooled and median gain", ""]
    for r in rows:
        lines.append(f"- K{r['K']}: severe (<−10%) {r['gain_lt_minus10pct_count']}/48; catastrophic (<−40%) {r['gain_lt_minus40pct_count']}/48; worst {pct(r['worst_row_gain'])}; pooled C_K vs D_K gain {pct(r['pooled_C_vs_D_gain'])}; median row gain {pct(r['median_row_gain'])}.")
    lines += ["", "The user-prespecified Coverage Support Gate requires K32 severe cases to be no more than half of K4 (unless K4 has zero, when catastrophic and worst-case values are compared), K32 pooled gain positive, K32 median row gain nonnegative, and both K16 and K32 severe counts below K4. Verdict: **" + summary["status"] + "**. Criteria: `" + json.dumps(summary["coverage_support_gate"]["checks"], sort_keys=True) + "`.",
              "The conditions came from the v14 task instructions before any candidate run. Their exact source hash is preserved in `coverage_support_gate_provenance.json`; the already-frozen `PROTOCOL.json` omitted this field, so the provenance note documents that serialization gap without editing the frozen protocol.",
              "", "K4-versus-K32 comparisons separate two effects: D_K itself changes because each K builds its own Hessian, and the row-level clipping gain compares C_K only against that same K's D_K. The `coverage_baseline_D.csv` records per-row D_K changes relative to D4.",
              "", "## Q6–Q8: alpha stability, landscape agreement, held-out concentration", "",
              f"- Across 48 rows, exact selected-alpha agreement across all four K values occurs for {summary['alpha_stability']['all_four_exact_agreement']}/48; median |α*_32−α*_4| is {summary['alpha_stability']['median_abs_change_K4_to_K32']:.2f}. α*_32 is closer to 1 than α*_4 for {summary['alpha_stability']['closer_to_1_K32_count']}/48, and K16/K32 differ by at most one grid step for {summary['alpha_stability']['stable_within_one_grid_K16_K32_count']}/48."]
    for key, val in summary["ranking_stability"].items():
        lines.append(f"- Search landscape {key}: median Spearman {val['median_spearman']:.3f}, best-alpha agreement {val['best_alpha_agreement_count']}/48, median top-2 Jaccard {val['median_top2_jaccard']:.3f}.")
    for k, x in summary["article_concentration"].items():
        lines.append(f"- K{k} held-out C_K article concentration: median top1/top3/top5 SSE share {pct(x['median_top1_share'])}/{pct(x['median_top3_share'])}/{pct(x['median_top5_share'])}; median worst single-article gain {pct(x['median_worst_article_gain'])}.")
    lines += ["", "Article concentration is an outcome diagnostic only. It did not enter alpha selection.",
              "", "## Q9: calibration cost", ""]
    lines.append("Per-K capture cost attributes actual per-article forward time for C01..CK plus the common H01..H32 pool, then prorates the measured model-load and cache-write overhead by document count. Hessian, candidate, and held-out evaluation seconds are measured. The full 64-article capture was one shared pass, so per-K capture totals are attributed costs, not four repeated executions.")
    lines += ["", "| K | activation capture (s) | Hessian (s) | candidate (s) | held-out eval (s) | total calibration (s) | multiplier vs K4 | peak allocated (MiB) | peak reserved (MiB) |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in costs:
        lines.append(f"| {c['K']} | {c['activation_capture_seconds_attributed']:.2f} | {c['hessian_build_seconds']:.2f} | {c['candidate_seconds']:.2f} | {c['heldout_evaluation_seconds']:.2f} | {c['total_calibration_seconds']:.2f} | {c['cost_multiplier_vs_K4']:.2f}× | {c['peak_allocated_bytes']/2**20:.1f} | {c['peak_reserved_bytes']/2**20:.1f} |")
    lines += ["", f"Formal runner wall time including Hessian builds was {summary['timing']['formal_runner_wall_seconds']/60:.1f} minutes. These are offline calibration/evaluation costs, not inference costs.",
              "", "## Q10: conclusion and limits", "",
              f"Under the preregistered diagnostic support gate, this run is **{summary['status']}**. Read the full K4→K8→K16→K32 sequence: severe/catastrophic counts, worst row, pooled gain, median gain, D_K baseline changes, and cost move independently. No claim of PPL or final-model improvement follows; no PPL, Gate, robust objective, predictor, or new model/bitwidth was run.",
              "", "Projection- and layer-level counts are in `projection_summary.csv` and `layer_summary.csv`; all negative row×K cases are in `regression_cases.csv`. Raw JSON and candidate tensors preserve all 1,920 candidate records and per-article held-out metrics.",
              "", "## Reproduction", "",
              "```powershell", "& '.venv-gptq\\Scripts\\python.exe' 'experiments/range_search_w4/results/range_w4_calibration_coverage_v14/prepare_v14.py'", "& '.venv-gptq\\Scripts\\python.exe' 'experiments/range_search_w4/results/range_w4_calibration_coverage_v14/run_v14.py'", "& '.venv-gptq\\Scripts\\python.exe' 'experiments/range_search_w4/results/range_w4_calibration_coverage_v14/summarize_v14.py'", "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
