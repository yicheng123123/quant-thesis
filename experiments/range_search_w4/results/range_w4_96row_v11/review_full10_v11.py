"""Read-only raw audit and descriptive v11 full10 feature analysis.

Run from the repository root with .venv-gptq/Scripts/python.exe. This script
never rewrites the frozen experimental records or existing summaries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
V9_SPLITS = HERE.parents[0] / "range_w4_l23r448_v9" / "splits_manifest.json"
GRID = np.round(np.arange(0.1, 1.01, 0.1), 1)
PROJECTIONS = ("q_proj", "o_proj", "up_proj", "down_proj")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a: float, b: float) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=1e-10, atol=1e-10))


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError("CORRECTNESS FAIL: " + msg)


def q(values, p):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(np.quantile(values, p)) if len(values) else float("nan")


def rho(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5 or np.unique(a[mask]).size < 2 or np.unique(b[mask]).size < 2:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def main() -> None:
    summary = read_json(HERE / "FULL10_SUMMARY.json")
    correctness = read_json(HERE / "FULL10_CORRECTNESS.json")
    manifest = read_json(HERE / "row_manifest.json")
    splits = read_json(V9_SPLITS)
    require(correctness["status"] == "PASS", "existing correctness status")
    require(sha256(HERE / "row_manifest.json") == "c56eeaa1d1af6e4ed8dd78f811c811c66758b215659a98bf8550052c786e6106", "row manifest hash")
    require(sha256(V9_SPLITS) == "34a24ef8602b8ce1e38e7a284f112acf7f766962d87d737f69148b78bcfaba78", "split manifest hash")
    require(len(manifest["rows"]) == 96, "row manifest count")
    split_map = {x["split_id"]: x for x in splits["splits"]}
    require(set(split_map) == {f"S{i:02d}" for i in range(10)}, "frozen split IDs")
    for split in split_map.values():
        s = {d["source_article_index"] for d in split["search"]}
        h = {d["source_article_index"] for d in split["heldout"]}
        require(len(s) == len(h) == 4 and not s.intersection(h), "article-disjoint split")
    parity = correctness["stage2_parity"]
    require({x["projection"] for x in parity} == set(PROJECTIONS) and all(x["status"] == "PASS" for x in parity), "four projection native parity records")
    require(all(x["differences"]["weight_mismatched_elements"] == 0 and
                x["differences"]["search_sse_abs"] == 0 and
                x["differences"]["heldout_sse_abs"] == 0 for x in parity), "native parity quantities")
    require(all(correctness["historical_tree_match"].values()), "saved historical fingerprints")
    reference = read_json(HERE / "manifest_reference.json")
    for name, files in reference["historical_fingerprints_before_v11"].items():
        root = HERE.parent / f"range_w4_{'dynamic_v8' if name == 'v8' else 'l23r448_v9' if name == 'v9' else 'sixrow_v10'}"
        actual = {str(path.relative_to(root)).replace("/", "\\"): {
            "sha256": sha256(path), "mtime_ns": path.stat().st_mtime_ns,
            "size_bytes": path.stat().st_size} for path in root.rglob("*") if path.is_file()}
        require(actual == files, f"current {name} historical tree fingerprint")
    preservation = read_json(HERE / "stage1_preservation_fingerprint.json")
    require(all((HERE / rel).is_file() and sha256(HERE / rel) == expected
                for rel, expected in preservation["files"].items()), "current Stage 1 preservation fingerprint")

    per = pd.read_csv(HERE / "full10_per_row_split_results.csv")
    rows = pd.read_csv(HERE / "full10_row_summary.csv")
    feat = pd.read_csv(HERE / "row_feature_outcome_table.csv")
    landscape = pd.read_csv(HERE / "full10_alpha_landscapes.csv")
    require(len(per) == 960 and len(rows) == len(feat) == 96 and len(landscape) == 9600, "CSV dimensions")
    require(per[["row_id", "split_id"]].duplicated().sum() == 0, "duplicate row/split")
    require(landscape[["row_id", "split_id", "alpha"]].duplicated().sum() == 0, "duplicate candidate")
    require(set(per.row_id) == {f"L{r['layer']}_{r['projection']}_R{r['row_index']}" for r in manifest["rows"]}, "row IDs")

    raw_files = sorted((HERE / "raw").glob("L*_S??.json")) + sorted((HERE / "raw" / "stage2").glob("L*_S??.json"))
    require(len(raw_files) == 960, "raw record count")
    per_key = per.set_index(["row_id", "split_id"])
    landscapes = landscape.set_index(["row_id", "split_id", "alpha"])
    for path in raw_files:
        raw = read_json(path)
        key = (raw["row_id"], raw["split_id"])
        require(key in per_key.index, f"raw key {key}")
        require(raw["row_manifest_sha256"] == sha256(HERE / "row_manifest.json") and
                raw["v9_manifest_sha256"] == sha256(V9_SPLITS), f"manifest references {key}")
        split = split_map[key[1]]
        require(raw["search_article_ids"] == [d["source_article_index"] for d in split["search"]] and
                raw["heldout_article_ids"] == [d["source_article_index"] for d in split["heldout"]], f"frozen article indices {key}")
        candidates = raw["candidates"]
        require(np.array_equal([float(c["alpha"]) for c in candidates], GRID), f"grid {key}")
        chosen = min(candidates, key=lambda c: c["search_output_mse"])
        d = candidates[-1]
        rec = per_key.loc[key]
        require(close(chosen["alpha"], raw["selected_alpha"]) and
                close(chosen["alpha"], rec["selected_alpha"]), f"search-only argmin {key}")
        for col, value in (("search_D_sse", d["search_sse"]), ("heldout_D_sse", d["heldout_sse"]),
                           ("search_C_sse", chosen["search_sse"]), ("heldout_C_sse", chosen["heldout_sse"])):
            require(close(rec[col], value), f"raw/CSV {col} {key}")
        require(close(rec["held_gain"], 1 - chosen["heldout_sse"] / d["heldout_sse"]), f"held gain {key}")
        for c in candidates:
            lc = landscapes.loc[(key[0], key[1], float(c["alpha"]))]
            require(close(lc.search_sse, c["search_sse"]) and close(lc.heldout_sse, c["heldout_sse"]), f"raw/landscape {key}")

    gains = per.groupby("row_id").held_gain.agg(list)
    for rid, g in gains.items():
        rec = rows.set_index("row_id").loc[rid]
        med, worst = float(np.median(g)), min(g)
        pos, gt2 = sum(v > 0 for v in g), sum(v > .02 for v in g)
        if pos >= 8 and med > .05 and worst >= -.1:
            label = "Robust-positive"
        elif (any(v > 0 for v in g) and any(v < 0 for v in g)) or worst < -.1:
            label = "Mixed-risk"
        elif abs(med) <= .02 and gt2 <= 2:
            label = "Neutral"
        else:
            label = "Unclassified (frozen-rule gap)"
        require(label == rec.final_benefit_label and close(med, rec["median"]), f"frozen label {rid}")
    labels = rows.final_benefit_label.value_counts().to_dict()
    require(labels == {"Robust-positive": 26, "Mixed-risk": 61, "Neutral": 5,
                       "Unclassified (frozen-rule gap)": 4}, "label prevalence")
    dsum, csum = per.heldout_D_sse.sum(), per.heldout_C_sse.sum()
    require(close(dsum, summary["pooled_heldout_sse"]["alpha1_D_sse"]) and
            close(csum, summary["pooled_heldout_sse"]["search_selected_C_sse"]), "pooled raw-to-summary")
    for alpha, reported in zip((.7, .8, .9), summary["fixed_alpha_descriptive_only"]):
        sub = landscape[np.isclose(landscape.alpha, alpha)]
        require(close(sub.heldout_sse.sum(), reported["pooled_fixed_alpha_sse"]), "fixed alpha raw-to-summary")

    landscape_extra = []
    for (rid, sid), group in landscape.groupby(["row_id", "split_id"]):
        group = group.sort_values("search_sse")
        best, second = group.iloc[0], group.iloc[1]
        baseline = group[np.isclose(group.alpha, 1.0)].iloc[0]
        landscape_extra.append({
            "row_id": rid, "split_id": sid,
            "search_gain": 1 - best.search_sse / baseline.search_sse,
            "search_margin": (second.search_sse - best.search_sse) / second.search_sse,
            "near_best_alpha_count_2pct": int((group.search_sse <= 1.02 * best.search_sse).sum()),
            "alpha1_near_best_2pct": int(baseline.search_sse <= 1.02 * best.search_sse),
        })
    extra = pd.DataFrame(landscape_extra).groupby("row_id").agg(
        search_gain_median=("search_gain", "median"),
        search_gain_mean=("search_gain", "mean"),
        search_margin_median=("search_margin", "median"),
        near_best_alpha_count_2pct_mean=("near_best_alpha_count_2pct", "mean"),
        alpha1_near_best_2pct_rate=("alpha1_near_best_2pct", "mean"),
    ).reset_index()
    extra["log10_alpha1_search_sse_total"] = np.log10(feat.set_index("row_id").loc[extra.row_id, "alpha1_search_sse_total"].to_numpy(float))
    extra["selected_alpha_band_width"] = feat.set_index("row_id").loc[extra.row_id, "selected_alpha_max"].to_numpy(float) - feat.set_index("row_id").loc[extra.row_id, "selected_alpha_min"].to_numpy(float)
    joined = feat.merge(extra, on="row_id", validate="one_to_one").merge(
        per.groupby("row_id").agg(regression_rate=("held_gain", lambda x: float((x < 0).mean())),
                                  severe_regression_rate=("held_gain", lambda x: float((x < -.1).mean()))).reset_index(),
        on="row_id", validate="one_to_one")

    chosen_features = {
        "weight": ["weight_max_abs", "weight_rms", "weight_std", "weight_max_abs_over_rms",
                   "weight_kurtosis_excess", "weight_skewness", "weight_p90_abs", "weight_p95_abs",
                   "weight_p99_abs", "weight_p999_abs", "weight_outlier_ratio_2rms",
                   "weight_outlier_ratio_3rms", "weight_outlier_ratio_4rms", "group_max_abs_mean",
                   "group_max_abs_std", "group_max_abs_max", "group_rms_mean", "group_rms_std"],
        "activation_search": ["activation_search_global_rms_mean", "activation_search_global_rms_std",
                              "activation_search_feature_rms_p90_mean", "activation_search_feature_rms_p99_mean",
                              "activation_search_feature_rms_max_mean", "activation_search_feature_variance_p90_mean",
                              "activation_search_feature_variance_p99_mean", "activation_search_feature_variance_max_mean"],
        "hessian_search": ["hessian_diag_mean_mean", "hessian_diag_std_mean", "hessian_diag_p90_mean",
                           "hessian_diag_p99_mean", "hessian_diag_max_mean", "hinv_cholesky_diag_mean_mean",
                           "hinv_cholesky_diag_std_mean", "hinv_cholesky_diag_p90_mean",
                           "hinv_cholesky_diag_p99_mean", "hinv_cholesky_diag_max_mean"],
        "gptq_search": ["alpha1_search_sse_total", "log10_alpha1_search_sse_total",
                        "alpha1_search_output_mse_mean", "search_gain_median", "search_gain_mean",
                        "search_margin_median", "near_best_alpha_count_2pct_mean",
                        "alpha1_near_best_2pct_rate", "selected_alpha_mode", "selected_alpha_mode_frequency",
                        "selected_alpha_mean", "selected_alpha_band_width"],
    }
    require(all(c in joined for cols in chosen_features.values() for c in cols), "feature columns")
    non_gap = joined[joined.final_benefit_label != "Unclassified (frozen-rule gap)"].copy()
    robust = non_gap[non_gap.final_benefit_label == "Robust-positive"]
    other = non_gap[non_gap.final_benefit_label != "Robust-positive"]
    analysis, stratified = [], []
    for family, columns in chosen_features.items():
        for col in columns:
            riqr, oiqr = q(robust[col], .75) - q(robust[col], .25), q(other[col], .75) - q(other[col], .25)
            pooled_iqr = (riqr + oiqr) / 2
            analysis.append({
                "family": family, "feature": col, "feature_source": "search_or_weight_only",
                "n_robust": len(robust), "n_nonrobust_excluding_gap": len(other),
                "robust_median": q(robust[col], .5), "robust_q25": q(robust[col], .25), "robust_q75": q(robust[col], .75),
                "nonrobust_median": q(other[col], .5), "nonrobust_q25": q(other[col], .25), "nonrobust_q75": q(other[col], .75),
                "median_diff_over_mean_iqr": (q(robust[col], .5) - q(other[col], .5)) / pooled_iqr if pooled_iqr else float("nan"),
                "rho_median_held_gain": rho(joined[col], joined.held_gain_median),
                "rho_regression_rate": rho(joined[col], joined.regression_rate),
                "rho_severe_regression_rate": rho(joined[col], joined.severe_regression_rate),
                "rho_worst_gain": rho(joined[col], joined.held_gain_worst),
                "n_rho": len(joined),
            })
            for projection in PROJECTIONS:
                proj = joined[joined.projection == projection]
                pr = proj[proj.final_benefit_label == "Robust-positive"]
                po = proj[proj.final_benefit_label.isin(["Mixed-risk", "Neutral"])]
                stratified.append({
                    "family": family, "feature": col, "projection": projection,
                    "n_rows": len(proj), "n_robust": len(pr), "n_nonrobust_excluding_gap": len(po),
                    "robust_median": q(pr[col], .5), "robust_q25": q(pr[col], .25), "robust_q75": q(pr[col], .75),
                    "nonrobust_median": q(po[col], .5), "nonrobust_q25": q(po[col], .25), "nonrobust_q75": q(po[col], .75),
                    "rho_median_held_gain": rho(proj[col], proj.held_gain_median),
                    "rho_regression_rate": rho(proj[col], proj.regression_rate),
                    "rho_severe_regression_rate": rho(proj[col], proj.severe_regression_rate),
                    "rho_worst_gain": rho(proj[col], proj.held_gain_worst),
                })
    pd.DataFrame(analysis).to_csv(HERE / "feature_outcome_analysis.csv", index=False, float_format="%.12g")
    pd.DataFrame(stratified).to_csv(HERE / "feature_stratified_summary.csv", index=False, float_format="%.12g")

    # Representative rule: one robust row per projection nearest that stratum's
    # robust median gain; match a same-layer/projection mixed row by nearest
    # log10 search-side baseline SSE. Add global high/moderate robust anchors.
    cases = []
    seeds = []
    for projection in PROJECTIONS:
        candidates = robust[robust.projection == projection]
        target = candidates.held_gain_median.median()
        seeds.append(candidates.iloc[(candidates.held_gain_median - target).abs().argmin()])
    for criterion in ("max", "moderate"):
        if criterion == "max":
            seeds.append(robust.loc[robust.held_gain_median.idxmax()])
        else:
            target = robust.held_gain_median.median()
            seeds.append(robust.iloc[(robust.held_gain_median - target).abs().argmin()])
    seen = set()
    for seed in seeds:
        if seed.row_id in seen:
            continue
        seen.add(seed.row_id)
        pool = joined[(joined.layer == seed.layer) & (joined.projection == seed.projection) &
                      (joined.final_benefit_label == "Mixed-risk")]
        if pool.empty:
            continue
        match = pool.iloc[(pool.log10_alpha1_search_sse_total - seed.log10_alpha1_search_sse_total).abs().argmin()]
        for role, row in (("robust", seed), ("matched_mixed", match)):
            cases.append({"pair_id": seed.row_id, "role": role, "row_id": row.row_id,
                          "layer": row.layer, "projection": row.projection, "label": row.final_benefit_label,
                          "median_held_gain": row.held_gain_median, "worst_held_gain": row.held_gain_worst,
                          "regression_rate": row.regression_rate, "severe_regression_rate": row.severe_regression_rate,
                          "alpha1_search_sse_total": row.alpha1_search_sse_total,
                          "weight_max_abs_over_rms": row.weight_max_abs_over_rms,
                          "weight_kurtosis_excess": row.weight_kurtosis_excess,
                          "activation_search_global_rms_mean": row.activation_search_global_rms_mean,
                          "activation_search_feature_rms_max_mean": row.activation_search_feature_rms_max_mean,
                          "hessian_diag_mean_mean": row.hessian_diag_mean_mean,
                          "hinv_cholesky_diag_mean_mean": row.hinv_cholesky_diag_mean_mean,
                          "search_gain_median": row.search_gain_median,
                          "search_margin_median": row.search_margin_median,
                          "near_best_alpha_count_2pct_mean": row.near_best_alpha_count_2pct_mean,
                          "alpha1_near_best_2pct_rate": row.alpha1_near_best_2pct_rate,
                          "selected_alpha_mode": row.selected_alpha_mode,
                          "selected_alpha_band_width": row.selected_alpha_band_width})
    failures = joined[joined.final_benefit_label == "Mixed-risk"].sort_values("held_gain_worst")
    failure_seeds = []
    used_projection = set()
    for _, failure in failures.iterrows():
        if failure.projection in used_projection or not ((robust.layer == failure.layer) &
                                                         (robust.projection == failure.projection)).any():
            continue
        failure_seeds.append(failure)
        used_projection.add(failure.projection)
        if len(failure_seeds) == 3:
            break
    neutral_like = joined[(joined.final_benefit_label == "Mixed-risk") &
                          (joined.selected_alpha_mode == 1)].copy()
    neutral_like = neutral_like.iloc[(neutral_like.held_gain_median.abs()).argsort()]
    for _, failure in neutral_like.iterrows():
        if ((robust.layer == failure.layer) & (robust.projection == failure.projection)).any():
            failure_seeds.append(failure)
            break
    for failure in failure_seeds:
        pool = robust[(robust.layer == failure.layer) & (robust.projection == failure.projection)]
        match = pool.iloc[(pool.log10_alpha1_search_sse_total - failure.log10_alpha1_search_sse_total).abs().argmin()]
        for role, row in (("failure_anchor", failure), ("matched_robust", match)):
            cases.append({"pair_id": failure.row_id, "role": role, "row_id": row.row_id,
                          "layer": row.layer, "projection": row.projection, "label": row.final_benefit_label,
                          "median_held_gain": row.held_gain_median, "worst_held_gain": row.held_gain_worst,
                          "regression_rate": row.regression_rate, "severe_regression_rate": row.severe_regression_rate,
                          "alpha1_search_sse_total": row.alpha1_search_sse_total,
                          "weight_max_abs_over_rms": row.weight_max_abs_over_rms,
                          "weight_kurtosis_excess": row.weight_kurtosis_excess,
                          "activation_search_global_rms_mean": row.activation_search_global_rms_mean,
                          "activation_search_feature_rms_max_mean": row.activation_search_feature_rms_max_mean,
                          "hessian_diag_mean_mean": row.hessian_diag_mean_mean,
                          "hinv_cholesky_diag_mean_mean": row.hinv_cholesky_diag_mean_mean,
                          "search_gain_median": row.search_gain_median,
                          "search_margin_median": row.search_margin_median,
                          "near_best_alpha_count_2pct_mean": row.near_best_alpha_count_2pct_mean,
                          "alpha1_near_best_2pct_rate": row.alpha1_near_best_2pct_rate,
                          "selected_alpha_mode": row.selected_alpha_mode,
                          "selected_alpha_band_width": row.selected_alpha_band_width})
    pd.DataFrame(cases).to_csv(HERE / "matched_case_analysis.csv", index=False, float_format="%.12g")

    row_deltas = per.groupby("row_id").apply(lambda x: pd.Series({"d": x.heldout_D_sse.sum(),
                                                                   "c": x.heldout_C_sse.sum()}), include_groups=False)
    row_deltas["delta"] = row_deltas.d - row_deltas.c
    top = row_deltas.sort_values("delta", ascending=False)
    gap = rows[rows.final_benefit_label == "Unclassified (frozen-rule gap)"]
    proj_stats = {}
    for projection in PROJECTIONS:
        x = per[per.projection == projection]
        y = rows[rows.projection == projection]
        proj_stats[projection] = {"median_row_median_gain": float(y["median"].median()),
                                  "regression_splits": int((x.held_gain < 0).sum()),
                                  "severe_splits": int((x.held_gain < -.1).sum()),
                                  "alpha1_selections": int((x.selected_alpha == 1).sum()),
                                  "D_sse": float(x.heldout_D_sse.sum()),
                                  "pooled_gain": float(1 - x.heldout_C_sse.sum() / x.heldout_D_sse.sum()),
                                  "median_D_sse_per_split": float(x.heldout_D_sse.median())}
    layer_stats = {}
    for layer in (0, 12, 23):
        x = per[per.layer == layer]
        layer_stats[layer] = {"D_sse": float(x.heldout_D_sse.sum()),
                              "pooled_gain": float(1 - x.heldout_C_sse.sum() / x.heldout_D_sse.sum()),
                              "severe_splits": int((x.held_gain < -.1).sum())}
    metrics = {
        "correctness": "PASS", "raw_records": len(raw_files), "landscape_candidates": len(landscape),
        "labels": labels, "pooled_gain": float(1 - csum / dsum),
        "median_row_median_gain": float(rows["median"].median()),
        "median_split_gain": float(per.held_gain.median()),
        "regression_splits": int((per.held_gain < 0).sum()),
        "severe_splits": int((per.held_gain < -.1).sum()),
        "worst_gain": float(per.held_gain.min()),
        "top1_net_share": float(top.delta.iloc[:1].sum() / top.delta.sum()),
        "top5_net_share": float(top.delta.iloc[:5].sum() / top.delta.sum()),
        "remaining_top1_gain": float(1 - top.c.iloc[1:].sum() / top.d.iloc[1:].sum()),
        "remaining_top5_gain": float(1 - top.c.iloc[5:].sum() / top.d.iloc[5:].sum()),
        "projection": proj_stats, "layer": layer_stats,
        "gap_rows": [{"row_id": r.row_id, "median": r["median"], "worst": r.worst,
                      "positive": r.positive_split_count, "negative": r.negative_split_count,
                      "alpha_band": r.best_contiguous_three_alpha_band,
                      "alpha_band_count": r.best_contiguous_three_alpha_band_count,
                      "held_gain_by_split": json.loads(r.held_gain_by_split),
                      "selected_alpha_by_split": json.loads(r.selected_alpha_by_split)}
                     for _, r in gap.iterrows()],
        "matched_pairs": sorted(set(x["pair_id"] for x in cases)),
    }
    (HERE / "SOL_REVIEW_DERIVED_METRICS.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
