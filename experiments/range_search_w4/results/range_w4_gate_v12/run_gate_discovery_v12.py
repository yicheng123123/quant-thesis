"""Phase A only: search-side row gate discovery on frozen v11 full10 artifacts.

No GPTQ, model load, validation evaluation, or PPL is performed here. The
threshold grid and ranking rules are loaded from the already frozen protocol.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
V11 = OUT.parent / "range_w4_96row_v11"
V9 = OUT.parent / "range_w4_l23r448_v9" / "splits_manifest.json"
PROJECTIONS = ("q_proj", "o_proj", "up_proj", "down_proj")
LAYERS = (0, 12, 23)
VALIDATION_LAYERS = (4, 8, 20)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_once(path: Path, content: str) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite frozen output: {path}")
    path.write_text(content, encoding="utf-8", newline="\n")


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise RuntimeError("DISCOVERY INPUT FAIL: " + explanation)


def outcome(df: pd.DataFrame, method_sse_col: str) -> dict:
    gain = 1.0 - df[method_sse_col].to_numpy(float) / df.heldout_D_sse.to_numpy(float)
    dsum = float(df.heldout_D_sse.sum())
    msum = float(df[method_sse_col].sum())
    return {
        "pairs": int(len(df)),
        "D_sse": dsum,
        "method_sse": msum,
        "pooled_gain": 1.0 - msum / dsum,
        "gain_lt_0": int((gain < 0).sum()),
        "gain_lt_minus_5pct": int((gain < -.05).sum()),
        "gain_lt_minus_10pct": int((gain < -.10).sum()),
        "gain_lt_minus_20pct": int((gain < -.20).sum()),
        "gain_lt_minus_40pct": int((gain < -.40).sum()),
        "worst_gain": float(gain.min()),
        "median_split_gain": float(np.median(gain)),
    }


def breakdown(df: pd.DataFrame, method_col: str, group_col: str) -> dict:
    values = {}
    for key, part in df.groupby(group_col, sort=True):
        values[str(key)] = outcome(part, method_col)
    return values


def simplicity(x: float) -> int:
    return len(f"{x:.2f}".rstrip("0").split(".")[1]) if "." in f"{x:.2f}".rstrip("0") else 0


def frozen_validation_plan(gate_sha: str, chosen: dict, row_manifest: dict) -> str:
    dims = {}
    for projection in PROJECTIONS:
        found = {int(r["output_dim"]) for r in row_manifest["rows"] if r["projection"] == projection}
        require(len(found) == 1, f"projection dimension {projection}")
        dims[projection] = found.pop()
    rng = random.Random(20261001)
    lines = [
        "# Gate v1 independent validation protocol — FROZEN, NOT EXECUTED", "",
        f"Gate identity: SHA-256 `GATE_V1.json` = `{gate_sha}`. Use exactly tau_gain={chosen['tau_gain']:.2f} and tau_margin={chosen['tau_margin']:.2f}; no retuning.",
        "", "## Scope and row manifest rule", "",
        "Use the same Qwen2.5-0.5B-Instruct checkpoint/revision and the same symmetric W4A16 dynamic GPTQ G128, block 128, damping 0.1, desc_act=False, FP-model input protocol as v11. New layers only: L4, L8, L20. Projections in fixed order q_proj, o_proj, up_proj, down_proj. Four rows per layer/projection, 48 total.",
        "", "Split each output dimension into four index buckets `[floor(k*d/4), floor((k+1)*d/4))` for k=0..3. Use one `random.Random(20261001)` instance, iterate layers (4,8,20), projections (q,o,up,down), then buckets (low,low-mid,high-mid,high), drawing `randrange(start, stop)` once per bucket. The following IDs are fixed by that rule; none were selected by outcome:",
        "", "```text",
    ]
    for layer in VALIDATION_LAYERS:
        for projection in PROJECTIONS:
            dim = dims[projection]
            indices = [rng.randrange((k * dim) // 4, ((k + 1) * dim) // 4) for k in range(4)]
            lines.append(f"L{layer} {projection}: " + ", ".join(str(i) for i in indices))
    lines += [
        "```", "", "## Independent articles and split freezing", "",
        "Use only `Salesforce/wikitext:wikitext-2-raw-v1:train`, the existing v7/v9 full-article parser and the same checkpoint tokenizer. Enumerate eligible complete train articles in original source order; retain those with at least 128 tokens. Exclude every old v8–v11 source article index AND article/token hash. From the remaining articles, take the first 80 with distinct source index, article hash, and first-128-token hash. If fewer than 80 exist, stop and document the blocker; do not substitute WikiText test or fragments.",
        "", "Shuffle this fixed 80-article list once with `random.Random(20261001)`; partition sequentially into ten groups of eight; in each group the first four whole articles are search and the last four held-out. Use the first 128 tokens of each article. Before any GPTQ run, save a manifest with source indices, hashes, token IDs, seed, exact parser/tokenizer identities and split membership, then hash that manifest. Require global uniqueness of all 80 articles/tokens and no overlap with v8–v11. No search or held-out article may be selected after seeing any quantization outcome.",
        "", "## Three paired methods and Gate inputs", "",
        "For every new row and every frozen split, evaluate the same ten shared-alpha dynamic GPTQ candidates (0.1–1.0). Each candidate starts from pristine W and GPTQ initial state; alpha=1 receives native parity spot checks in each projection before formal work. D uses alpha=1, C uses the search-output-MSE-minimizing alpha for that split. Gate computes each row's ten search gains and ten top1/top2 search margins from search SSE only, takes the two medians, then accepts **all ten C split choices** iff both frozen thresholds pass; otherwise all ten splits use D. The Gate reads no validation held-out value, projection-specific rule, layer-specific rule, or third feature. Gate is a selection/safety experiment, not an offline-search acceleration.",
        "", "## Frozen validation decision", "",
        "A: Gate `held_gain < -0.10` count is at most half of C's count (`2*G_severe <= C_severe`). B: Gate pooled SSE gain versus D is strictly >0. C: if C pooled gain is >0, Gate retains at least 50% of it (`G_gain/C_gain >= 0.5`); if C gain is <=0, C is inapplicable and must be reported as such. Validation PASS requires all applicable conditions. Thresholds, rows, articles, splits, candidate grid, Gate structure and PASS conditions cannot change after seeing validation held-out values. If FAIL, retain the negative result and do not tune on these data.",
        "", "## Required secondary reporting and cost gate", "",
        "Report each method's pooled SSE gain, per-row median gain, counts for held_gain <0, <−5%, <−10%, <−20%, <−40%, worst gain, accepted/fallback rows, projection and layer breakdown, and top1/top5/top10 shares of net SSE reduction. Report exact failed and degenerate cases. Treat the 480 row/split values as repeated measurements, not 480 independent model samples; show row and layer/projection clustering. Record candidate evaluations, synchronized total wall time including precomputation/cache build, and peak allocated/reserved GPU memory. Do a small correctness/cost preflight before formal execution. Expected candidate slots: `48 rows × 10 splits × 10 alpha = 4,800`; this is a slot count, not a runtime guarantee.",
        "", "No PPL, quantized-prefix, other model, extra bitwidth, predictor training, or threshold adaptation belongs to this validation.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    protocol_path = OUT / "GATE_DISCOVERY_PROTOCOL.json"
    protocol = load_json(protocol_path)
    require(protocol["status"] == "FROZEN_BEFORE_THRESHOLD_EVALUATION", "protocol status")
    require(protocol["threshold_grid"]["pair_count"] == 36, "grid pair count")
    source = protocol["discovery_source"]
    for key, filename in (("row_manifest_sha256", "row_manifest.json"),
                          ("full10_alpha_landscapes_csv_sha256", "full10_alpha_landscapes.csv"),
                          ("full10_per_row_split_results_csv_sha256", "full10_per_row_split_results.csv")):
        require(digest(V11 / filename) == source[key], f"source hash {filename}")
    require(digest(V9) == source["v9_split_manifest_sha256"], "split manifest hash")
    require(load_json(V11 / "FULL10_CORRECTNESS.json")["status"] == "PASS", "v11 correctness")
    require(not (OUT / "gate_grid_results.csv").exists() and not (OUT / "GATE_V1.json").exists(), "immutable outputs absent")

    per = pd.read_csv(V11 / "full10_per_row_split_results.csv")
    # Deliberately read only search-side columns for the Gate feature calculation.
    search = pd.read_csv(V11 / "full10_alpha_landscapes.csv",
                         usecols=["row_id", "split_id", "alpha", "search_sse"])
    require(len(per) == 960 and len(search) == 9600, "source row counts")
    require(per[["row_id", "split_id"]].duplicated().sum() == 0, "unique row/split records")
    require(search[["row_id", "split_id", "alpha"]].duplicated().sum() == 0, "unique candidates")
    require(per.row_id.nunique() == 96 and per.groupby("row_id").split_id.nunique().eq(10).all(), "96 by 10 layout")
    require(per.heldout_D_sse.gt(0).all(), "positive D SSE")
    require(set(per.projection) == set(PROJECTIONS) and set(per.layer) == set(LAYERS), "frozen strata")

    features = []
    for (rid, sid), group in search.groupby(["row_id", "split_id"], sort=True):
        group = group.sort_values(["search_sse", "alpha"], kind="stable")
        require(len(group) == 10 and np.array_equal(np.sort(group.alpha.to_numpy()), np.array(source["alpha_grid"])), f"alpha grid {rid}/{sid}")
        best, second = group.iloc[0], group.iloc[1]
        d = group[group.alpha.eq(1.0)].iloc[0]
        require(d.search_sse > 0 and second.search_sse > 0, f"search SSE positive {rid}/{sid}")
        record = per[(per.row_id == rid) & (per.split_id == sid)].iloc[0]
        require(np.isclose(best.alpha, record.selected_alpha) and
                np.isclose(best.search_sse, record.search_C_sse) and
                np.isclose(d.search_sse, record.search_D_sse), f"search-only selection parity {rid}/{sid}")
        features.append({"row_id": rid, "split_id": sid,
                         "search_gain": 1.0 - best.search_sse / d.search_sse,
                         "search_margin": (second.search_sse - best.search_sse) / second.search_sse})
    feature = pd.DataFrame(features).groupby("row_id", sort=True).agg(
        search_gain_median=("search_gain", "median"),
        search_margin_median=("search_margin", "median")).reset_index()
    require(len(feature) == 96, "feature row count")
    row_manifest = load_json(V11 / "row_manifest.json")
    manifest_ids = {f"L{r['layer']}_{r['projection']}_R{r['row_index']}" for r in row_manifest["rows"]}
    require(set(feature.row_id) == manifest_ids, "feature row manifest")
    c = outcome(per, "heldout_C_sse")
    d = outcome(per, "heldout_D_sse")
    require(c["pooled_gain"] > 0, "C pooled gain must be positive for frozen discovery condition")
    require(c["gain_lt_minus_10pct"] == 97, "C severe count differs from audited v11")

    results = []
    for tg in protocol["threshold_grid"]["tau_gain"]:
        for tm in protocol["threshold_grid"]["tau_margin"]:
            accepted = feature.loc[(feature.search_gain_median >= tg) &
                                   (feature.search_margin_median >= tm), "row_id"]
            accept_set = set(accepted)
            current = per.copy()
            current["gate_sse"] = np.where(current.row_id.isin(accept_set),
                                           current.heldout_C_sse, current.heldout_D_sse)
            score = outcome(current, "gate_sse")
            retained = score["pooled_gain"] / c["pooled_gain"]
            pass_a = 2 * score["gain_lt_minus_10pct"] <= c["gain_lt_minus_10pct"]
            pass_b = score["pooled_gain"] > 0
            pass_c = retained >= .5
            results.append({
                "tau_gain": tg, "tau_margin": tm,
                "accepted_rows": len(accept_set), "fallback_rows": 96 - len(accept_set),
                "gate_heldout_sse": score["method_sse"], "D_heldout_sse": score["D_sse"],
                "Gate_pooled_gain_vs_D": score["pooled_gain"],
                "C_pooled_gain_vs_D": c["pooled_gain"], "retained_gain": retained,
                "C_severe_failures": c["gain_lt_minus_10pct"],
                "Gate_severe_failures": score["gain_lt_minus_10pct"],
                "severe_reduction_fraction": 1 - score["gain_lt_minus_10pct"] / c["gain_lt_minus_10pct"],
                "gain_lt_0": score["gain_lt_0"],
                "gain_lt_minus_5pct": score["gain_lt_minus_5pct"],
                "gain_lt_minus_10pct": score["gain_lt_minus_10pct"],
                "gain_lt_minus_20pct": score["gain_lt_minus_20pct"],
                "gain_lt_minus_40pct": score["gain_lt_minus_40pct"],
                "worst_gain": score["worst_gain"],
                "pass_A": pass_a, "pass_B": pass_b, "pass_C": pass_c,
                "PASS": pass_a and pass_b and pass_c,
                "projection_breakdown_json": json.dumps(breakdown(current, "gate_sse", "projection"), separators=(",", ":")),
                "layer_breakdown_json": json.dumps(breakdown(current, "gate_sse", "layer"), separators=(",", ":")),
                "accepted_row_ids_json": json.dumps(sorted(accept_set), separators=(",", ":")),
            })
    grid = pd.DataFrame(results)
    require(len(grid) == 36, "evaluated grid count")
    write_text_once(OUT / "gate_grid_results.csv", grid.to_csv(index=False, float_format="%.15g"))
    passing = grid[grid.PASS].copy()
    if passing.empty:
        report = (
            "# Gate v1 discovery: GATE_V1_DISCOVERY_FAIL\n\n"
            "No threshold in the pre-frozen 6×6 grid met all three criteria. "
            "No Gate v1 or validation protocol was frozen. Do not expand the grid "
            "or change the rule using these data. Inspect `gate_grid_results.csv` "
            "for the gain-risk tradeoff and stop.\n"
        )
        write_text_once(OUT / "GATE_DISCOVERY_REPORT.md", report)
        print(json.dumps({"status": "GATE_V1_DISCOVERY_FAIL", "pass_count": 0}, indent=2))
        return

    passing["simplicity"] = passing.tau_gain.map(simplicity) + passing.tau_margin.map(simplicity)
    passing = passing.sort_values(
        ["retained_gain", "Gate_severe_failures", "fallback_rows", "simplicity", "tau_gain", "tau_margin"],
        ascending=[False, True, False, True, True, True], kind="stable")
    chosen = passing.iloc[0].to_dict()
    accepted_ids = json.loads(chosen["accepted_row_ids_json"])
    gate = {
        "gate_id": "range_w4_gate_v12_gate_v1",
        "status": "FROZEN_DISCOVERY_ONLY_NOT_INDEPENDENTLY_VALIDATED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "tau_gain": float(chosen["tau_gain"]), "tau_margin": float(chosen["tau_margin"]),
        "search_gain_definition": protocol["search_features"]["search_gain_median"],
        "search_gain_per_split_definition": protocol["search_features"]["search_gain_per_split"],
        "search_margin_definition": protocol["search_features"]["search_margin_median"],
        "search_margin_per_split_definition": protocol["search_features"]["search_margin_per_split"],
        "decision_rule": protocol["decision_rule"],
        "decision_granularity": protocol["decision_granularity"],
        "discovery_dataset": "v11 full10: 96 frozen rows × 10 frozen article splits",
        "v11_row_manifest_sha256": source["row_manifest_sha256"],
        "v11_split_manifest_sha256": source["v9_split_manifest_sha256"],
        "discovery_protocol_sha256": digest(protocol_path),
        "discovery_script_sha256": digest(Path(__file__)),
        "accepted_row_count": int(chosen["accepted_rows"]),
        "fallback_row_count": int(chosen["fallback_rows"]),
        "discovery_accepted_row_ids": accepted_ids,
        "discovery_C_severe_failure_count": int(chosen["C_severe_failures"]),
        "discovery_Gate_severe_failure_count": int(chosen["Gate_severe_failures"]),
        "discovery_severe_failure_reduction_fraction": float(chosen["severe_reduction_fraction"]),
        "discovery_C_pooled_gain": float(chosen["C_pooled_gain_vs_D"]),
        "discovery_Gate_pooled_gain": float(chosen["Gate_pooled_gain_vs_D"]),
        "discovery_retained_gain": float(chosen["retained_gain"]),
        "validation_status": "NOT_RUN",
        "immutability": "Do not edit this file; verify its SHA-256 before any independent validation.",
    }
    gate_path = OUT / "GATE_V1.json"
    write_text_once(gate_path, json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    gate_sha = digest(gate_path)

    # The frozen validation plan is a design artifact only. It samples row IDs
    # from known projection dimensions without loading a model or evaluating W.
    write_text_once(OUT / "VALIDATION_PLAN_FROZEN.md", frozen_validation_plan(gate_sha, chosen, row_manifest))
    proj = json.loads(chosen["projection_breakdown_json"])
    layer = json.loads(chosen["layer_breakdown_json"])
    report = [
        "# Gate v1 discovery report — v11 only", "",
        f"Status: **DISCOVERY_PASS**. Independent validation: **NOT RUN**. Pre-frozen grid: 6×6 = 36 pairs; {len(passing)} passed A/B/C.",
        "", f"Frozen protocol: `GATE_DISCOVERY_PROTOCOL.json` SHA-256 `{digest(protocol_path)}`.",
        f"Frozen Gate: `GATE_V1.json` SHA-256 `{gate_sha}`. Script SHA-256 `{digest(Path(__file__))}`.",
        "", "## Chosen row-level rule", "",
        f"ACCEPT iff `search_gain_median >= {chosen['tau_gain']:.2f}` AND `search_margin_median >= {chosen['tau_margin']:.2f}`. Otherwise FALLBACK to native dynamic GPTQ alpha=1. Both medians use only the ten search landscapes for that row; ACCEPT uses each split's own search-best alpha. No held-out input enters the Gate.",
        "", f"Accepted {int(chosen['accepted_rows'])}/96 rows; fallback {int(chosen['fallback_rows'])}/96. Uniform C severe failures (<−10%): {c['gain_lt_minus_10pct']}/960; Gate: {int(chosen['Gate_severe_failures'])}/960, reduction {float(chosen['severe_reduction_fraction']):.2%}.",
        f"Uniform C pooled gain versus D: {c['pooled_gain']:.2%}. Gate: {float(chosen['Gate_pooled_gain_vs_D']):.2%}. Retained gain: {float(chosen['retained_gain']):.2%}.",
        "", "## Tail and stratum diagnostics", "",
        f"D is all zero gain. C: negative {c['gain_lt_0']}, <−5% {c['gain_lt_minus_5pct']}, <−10% {c['gain_lt_minus_10pct']}, <−20% {c['gain_lt_minus_20pct']}, <−40% {c['gain_lt_minus_40pct']}, worst {c['worst_gain']:.2%}.",
        f"Gate: negative {int(chosen['gain_lt_0'])}, <−5% {int(chosen['gain_lt_minus_5pct'])}, <−10% {int(chosen['gain_lt_minus_10pct'])}, <−20% {int(chosen['gain_lt_minus_20pct'])}, <−40% {int(chosen['gain_lt_minus_40pct'])}, worst {float(chosen['worst_gain']):.2%}.",
        "", "Projection breakdown below reports paired pooled gain, severe count, and worst gain; complete projection/layer metrics for every threshold are JSON fields in `gate_grid_results.csv`.", "",
    ]
    for projection in PROJECTIONS:
        x = proj[projection]
        report.append(f"- {projection}: pooled {x['pooled_gain']:.2%}, severe {x['gain_lt_minus_10pct']}/240, worst {x['worst_gain']:.2%}.")
    report += ["", "Layer breakdown:", ""]
    for l in LAYERS:
        x = layer[str(l)]
        report.append(f"- L{l}: pooled {x['pooled_gain']:.2%}, severe {x['gain_lt_minus_10pct']}/320, worst {x['worst_gain']:.2%}.")
    report += [
        "", "## Interpretation and limits", "",
        "The threshold pair maximized retained pooled gain among PASS candidates under the ranking rule frozen before evaluation. This is selection on the same 96-row discovery held-out outcomes; no unbiased effect size, safety guarantee, classifier accuracy, or model-level/PPL improvement follows. Search metrics are known only after the ten candidate GPTQ runs; Gate is not an offline candidate-search speedup. Stratum tails and concentrated SSE contributions remain relevant even if the aggregate PASS criteria hold.",
        "", "The frozen independent design is in `VALIDATION_PLAN_FROZEN.md`. Its rows/articles have not been quantized or evaluated. Do not alter `GATE_V1.json` after this run; validate its SHA-256 before any subsequent work.", "",
    ]
    write_text_once(OUT / "GATE_DISCOVERY_REPORT.md", "\n".join(report))
    print(json.dumps({"status": "DISCOVERY_PASS", "pass_count": len(passing),
                      "tau_gain": gate["tau_gain"], "tau_margin": gate["tau_margin"],
                      "accepted_rows": gate["accepted_row_count"],
                      "fallback_rows": gate["fallback_row_count"],
                      "severe_C": gate["discovery_C_severe_failure_count"],
                      "severe_Gate": gate["discovery_Gate_severe_failure_count"],
                      "Gate_pooled_gain": gate["discovery_Gate_pooled_gain"],
                      "retained_gain": gate["discovery_retained_gain"],
                      "gate_sha256": gate_sha, "validation_candidate_slots": 4800}, indent=2))


if __name__ == "__main__":
    main()
