"""Produce compact descriptive summary from analyze_v13.py CSVs (no experiment runs)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
q = pd.read_csv(HERE / "quadratic_decomposition.csv")
r = pd.read_csv(HERE / "article_ranking_stability.csv")
a = pd.read_csv(HERE / "article_level_landscapes.csv")
t = pd.read_csv(HERE / "token_error_concentration.csv")
m = pd.read_csv(HERE / "matched_cases.csv")
z = q.merge(r.drop(columns=["layer", "projection", "selected_alpha", "search_gain", "heldout_gain", "gate_accept"]), on=["row_id", "split_id"])
assert len(z) == 480


def stat(frame: pd.DataFrame, cols: list[str]) -> dict:
    out = {"n": len(frame)}
    for col in cols:
        v = frame[col].astype(float)
        out[col] = {"median": float(v.median()), "p25": float(v.quantile(.25)),
                    "p75": float(v.quantile(.75)), "mean": float(v.mean())}
    return out


groups = {"robust_success_ge_10pct": z[z.heldout_gain >= .1],
          "near_neutral_minus10_to_10pct": z[(z.heldout_gain >= -.1) & (z.heldout_gain < .1)],
          "severe_lt_minus10pct": z[z.heldout_gain < -.1],
          "catastrophic_lt_minus40pct": z[z.heldout_gain < -.4]}
cols = ["search_gain", "heldout_gain", "relative_global_gram_shift_F", "directional_shift_C",
        "directional_shift_D", "directional_shift_D_minus_C", "generalization_shift",
        "pairwise_spearman_mean", "pairwise_spearman_min", "pairwise_top3_jaccard_mean",
        "articles_best_agree_aggregate", "selected_articles_improved", "selected_article_gain_std",
        "selected_article_gain_min", "e_C_norm", "e_D_norm"]
summary = {"groups": {name: stat(frame, cols) for name, frame in groups.items()}, "correlations": {},
           "article_dependence": {}, "token_concentration": {}, "worst_case": {}, "matched_cases": {}}
for col in [x for x in cols if x != "heldout_gain"]:
    res = spearmanr(z[col], z.heldout_gain)
    summary["correlations"][col] = {"spearman_vs_heldout_gain": float(res.statistic),
                                    "pvalue_descriptive_only": float(res.pvalue)}

selected = a[a.is_search_best]
assert len(selected) == 480
for name, frame in groups.items():
    keys = set(zip(frame.row_id, frame.split_id))
    ss = selected[[tuple(x) in keys for x in zip(selected.row_id, selected.split_id)]]
    positive = ss[ss.alpha < 1]
    advances = np.column_stack([positive[f"article{j}_mse"].values for j in range(1, 5)])
    base = a[a.alpha == 1].set_index(["row_id", "split_id"])
    dloss = np.column_stack([[base.loc[(row, split), f"article{j}_mse"] for row, split in zip(positive.row_id, positive.split_id)] for j in range(1, 5)])
    gain = dloss - advances
    pos = np.maximum(gain, 0)
    top_share = pos.max(axis=1) / pos.sum(axis=1) if len(pos) else np.array([])
    summary["article_dependence"][name] = {
        "selected_alpha_lt_1_count": len(positive),
        "selected_improved_article_counts": {str(k): int((ss.articles_improved == k).sum()) for k in range(5)},
        "selected_regressed_ge_1_article": int((positive.articles_regressed >= 1).sum()),
        "top_positive_article_share_median": float(np.median(top_share)) if len(top_share) else None,
        "top_positive_article_share_ge_0_75": int((top_share >= .75).sum()) if len(top_share) else 0,
        "article_best_agrees_aggregate_ge_2": int((frame.articles_best_agree_aggregate >= 2).sum()),
        "pairwise_rank_spearman_below_0_5": int((frame.pairwise_spearman_mean < .5).sum())}

for name, frame in groups.items():
    keys = set(zip(frame.row_id, frame.split_id))
    subset = t[(t.side == "heldout") & (t.method == "C")]
    subset = subset[[tuple(x) in keys for x in zip(subset.row_id, subset.split_id)]]
    summary["token_concentration"][name] = stat(subset, ["top1pct_sse_share", "top5pct_sse_share",
                                                              "top10pct_sse_share", "top1article_sse_share",
                                                              "top2articles_sse_share", "max"])

worst = z[(z.row_id == "L20_o_proj_R287") & (z.split_id == "S08")].iloc[0]
wa = selected[(selected.row_id == worst.row_id) & (selected.split_id == worst.split_id)].iloc[0]
wd = a[(a.row_id == worst.row_id) & (a.split_id == worst.split_id) & (a.alpha == 1)].iloc[0]
wt = t[(t.row_id == worst.row_id) & (t.split_id == worst.split_id)]
summary["worst_case"] = {
    "quadratic": {key: float(worst[key]) for key in ["search_gain", "heldout_gain", "selected_alpha", "qD_search", "qC_search", "qD_heldout", "qC_heldout", "A_search", "A_heldout", "generalization_shift", "directional_shift_C", "directional_shift_D", "relative_global_gram_shift_F", "pairwise_spearman_mean", "articles_best_agree_aggregate", "selected_articles_improved"]},
    "search_article": [{"article_id": int(wa[f"article{j}_id"]), "D_mse": float(wd[f"article{j}_mse"]),
                        "C_mse": float(wa[f"article{j}_mse"]), "gain": float(wa[f"article{j}_gain"])} for j in range(1, 5)],
    "heldout_token": [{"method": method, **{key: float(wt[(wt.side == "heldout") & (wt.method == method)].iloc[0][key]) for key in ["top1pct_sse_share", "top5pct_sse_share", "top1article_sse_share", "max"] + [f"article{j}_sse_share" for j in range(1, 5)]}} for method in ["D", "C"]]}
summary["matched_cases"] = {"count": len(m),
                            "failure_median_global_gram_shift": float(m.failure_relative_global_gram_shift_F.median()),
                            "success_median_global_gram_shift": float(m.success_relative_global_gram_shift_F.median()),
                            "failure_median_directional_shift_C": float(m.failure_directional_shift_C.median()),
                            "success_median_directional_shift_C": float(m.success_directional_shift_C.median()),
                            "failure_median_rank_spearman": float(z.set_index(["row_id", "split_id"]).loc[list(zip(m.failure_row_id, m.failure_split_id))].pairwise_spearman_mean.median()),
                            "success_median_rank_spearman": float(z.set_index(["row_id", "split_id"]).loc[list(zip(m.success_row_id, m.success_split_id))].pairwise_spearman_mean.median())}
(HERE / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"groups": {k: {"n": v["n"], "held_gain_median": v["heldout_gain"]["median"],
                                 "global_shift_median": v["relative_global_gram_shift_F"]["median"],
                                 "directional_C_median": v["directional_shift_C"]["median"],
                                 "rank_spearman_median": v["pairwise_spearman_mean"]["median"],
                                 "improved_articles_median": v["selected_articles_improved"]["median"]} for k,v in summary["groups"].items()},
                  "worst_case": summary["worst_case"], "matched_cases": summary["matched_cases"]}, indent=2))
