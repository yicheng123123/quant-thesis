#!/usr/bin/env python3
"""Artifact-only scalar10 candidate budget precheck.

Reads only completed M2 search-choice outputs and saved v24/v30 direct search
scores. It does not import torch, initialize CUDA, run a model, GPTQ, PPL, or
held scoring. All generated files stay in this independent result directory.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path


START = time.monotonic()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = ROOT / "experiments" / "range_search_w4" / "results"
V18 = RESULTS / "range_w4_fullmodel_v18"
V24 = RESULTS / "range_w4_structured_group_e2_v24"
V30 = RESULTS / "range_w4_weight_error_regularization_v30"
V17 = RESULTS / "range_w4_m2_runtime_v17_3"
GRID = [round(x / 10, 1) for x in range(1, 11)]
GRID_STR = [f"{x:.1f}" for x in GRID]
ALPHA1 = "1.0"
BUDGETS = (3, 4, 5)


def fail_if_over_budget() -> None:
    elapsed = time.monotonic() - START
    if elapsed > 295:
        raise RuntimeError(f"CPU analysis exceeded 295-second guard: {elapsed:.3f}s")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: val(v) for k, v in row.items()})


def val(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (list, tuple, set)):
        if isinstance(value, set):
            value = sorted(value, key=alpha_num_or_text)
        return ";".join(str(x) for x in value)
    return value


def alpha_num_or_text(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def canonical_alpha(value) -> str:
    return f"{float(value):.1f}"


def alpha_set_string(values) -> str:
    return ";".join(sorted({canonical_alpha(a) for a in values}, key=float))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pctl(values, q: float):
    if not values:
        return None
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def direct_best(score_by_alpha, alphas=None):
    candidates = score_by_alpha.keys() if alphas is None else alphas
    a = min(candidates, key=lambda x: (score_by_alpha[x]["mse"], -float(x)))
    return a, score_by_alpha[a]


def top_frequency_subset(rows, k: int):
    """Maximize preserved historical row winners; alpha=1 is mandatory."""
    counts = Counter(r["selected_alpha"] for r in rows)
    others = [a for a in GRID_STR if a != ALPHA1 and counts[a] > 0]
    chosen = sorted(others, key=lambda a: (-counts[a], -float(a)))[: k - 1]
    return {ALPHA1, *chosen}


def winner_union(rows):
    return {ALPHA1, *(r["selected_alpha"] for r in rows)}


def sort_rows_by_key(rows):
    return sorted(rows, key=lambda r: (r["module"], int(r["row"])))


def m2_subset_metrics(rows, subset):
    total_base = math.fsum(r["best_mse"] for r in rows)
    retained = [r for r in rows if r["selected_alpha"] in subset]
    changed = [r for r in rows if r["selected_alpha"] not in subset]
    upper = []
    for r in changed:
        upper.append((max(0.0, r["alpha1_mse"] - r["best_mse"]), r))
    upper_sum = math.fsum(x[0] for x in upper)
    worst = max(upper, key=lambda x: (x[0] / max(abs(x[1]["best_mse"]), 1e-300), x[0]), default=None)
    exact = 0.0 if not changed else None
    return {
        "rows": len(rows),
        "retained": len(retained),
        "changed": len(changed),
        "retention_pct": 100.0 * len(retained) / len(rows),
        "baseline_mse_sum": total_base,
        "exact_delta_mse_sum": exact,
        "delta_lower_mse_sum": 0.0,
        "delta_upper_mse_sum": upper_sum,
        "delta_upper_pct": 100.0 * upper_sum / max(total_base, 1e-300),
        "delta_upper_sse_sum": upper_sum * M2_TOKENS,
        "worst_upper_mse": worst[0] if worst else 0.0,
        "worst_upper_pct": 100.0 * worst[0] / max(abs(worst[1]["best_mse"]), 1e-300) if worst else 0.0,
        "worst_row": f"{worst[1]['module']}#R{worst[1]['row']}" if worst else "",
        "changed_rows": changed,
    }


def sample_subset_metrics(rows, subset, direct_grid_by_unit):
    base_sum = math.fsum(r["score_sse"] for r in rows)
    deltas = []
    retained = 0
    for r in rows:
        score_map = direct_grid_by_unit[r["unit_key"]][r["row_key"]]
        best_a, best_score = direct_best(score_map)
        sub_a, sub_score = direct_best(score_map, subset)
        if sub_a == best_a:
            retained += 1
        delta = sub_score["sse"] - best_score["sse"]
        if delta < 0 and abs(delta) < max(1e-20, abs(best_score["sse"]) * 1e-12):
            delta = 0.0
        deltas.append((delta, r, best_score, sub_score, best_a, sub_a))
    delta_sum = math.fsum(x[0] for x in deltas)
    worst = max(deltas, key=lambda x: (x[0] / max(abs(x[2]["sse"]), 1e-300), x[0]), default=None)
    return {
        "rows": len(rows),
        "retained": retained,
        "changed": len(rows) - retained,
        "retention_pct": 100.0 * retained / len(rows),
        "baseline_sse_sum": base_sum,
        "delta_sse_sum": delta_sum,
        "delta_pct": 100.0 * delta_sum / max(base_sum, 1e-300),
        "worst_delta_sse": worst[0] if worst else 0.0,
        "worst_delta_pct": 100.0 * worst[0] / max(abs(worst[2]["sse"]), 1e-300) if worst else 0.0,
        "worst_row": f"{worst[1]['unit_key']}#R{worst[1]['row']}" if worst and worst[0] > 0 else "",
        "worst_original_alpha": worst[4] if worst else "",
        "worst_subset_alpha": worst[5] if worst else "",
    }


def choose_best_sample_subset(rows, k: int, direct_grid_by_unit):
    others = [a for a in GRID_STR if a != ALPHA1]
    best_key = None
    best_subset = None
    for combo in itertools.combinations(others, k - 1):
        subset = {ALPHA1, *combo}
        total = 0.0
        for r in rows:
            score_map = direct_grid_by_unit[r["unit_key"]][r["row_key"]]
            total += direct_best(score_map, subset)[1]["sse"]
        # Stable tie policy for subset design: prefer larger alphas in a tie.
        tie = tuple(-float(a) for a in sorted(subset, key=float, reverse=True))
        key = (total, tie)
        if best_key is None or key < best_key:
            best_key, best_subset = key, subset
    return best_subset, best_key[0]


def direct_margin(rows, direct_grid_by_unit):
    gaps = []
    strict_by_alpha = Counter()
    selected_by_alpha = Counter()
    selected_rows_by_alpha = defaultdict(list)
    tie_rows = 0
    for r in rows:
        scores = direct_grid_by_unit[r["unit_key"]][r["row_key"]]
        ordered = sorted(((v["mse"], -float(a), a) for a, v in scores.items()))
        best_mse, _, best_a = ordered[0]
        second_mse = ordered[1][0]
        gap = second_mse - best_mse
        gaps.append(gap)
        tied = sum(1 for v in scores.values() if v["mse"] == best_mse)
        if tied == 1:
            strict_by_alpha[best_a] += 1
        else:
            tie_rows += 1
        selected_by_alpha[best_a] += 1
        selected_rows_by_alpha[best_a].append(r["row_key"])
    return {
        "gaps": gaps,
        "strict_by_alpha": strict_by_alpha,
        "selected_by_alpha": selected_by_alpha,
        "selected_rows_by_alpha": selected_rows_by_alpha,
        "tie_rows": tie_rows,
    }


# Source protocol checks. The v17.3 runtime protocol and v18 formal protocol
# independently state the same scalar grid and row-wise direct-MSE semantics.
v18_protocol_path = V18 / "PROTOCOL.json"
v17_protocol_path = V17 / "PROTOCOL.json"
v24_protocol_path = V24 / "PROTOCOL.json"
v24_manifest_path = V24 / "E2_DOCUMENT_MANIFEST.json"
v24_scorer_path = V24 / "run_e2_v24.py"
m2_rows_path = V18 / "M2_ROW_RESULTS.csv"
m2_module_summary_path = V18 / "M2_MODULE_SUMMARY.csv"
m2_alpha_distribution_path = V18 / "M2_ALPHA_DISTRIBUTION.csv"
v30_search_path = V30 / "SEARCH_CANDIDATE_TABLE.csv"
v30_run_path = V30 / "RUN_SUMMARY.json"

for p in (v18_protocol_path, v17_protocol_path, v24_protocol_path, v24_manifest_path, v24_scorer_path,
          m2_rows_path, m2_module_summary_path, m2_alpha_distribution_path,
          v30_search_path, v30_run_path):
    if not p.is_file():
        raise FileNotFoundError(p)

v18_protocol = read_json(v18_protocol_path)
v17_protocol = read_json(v17_protocol_path)
v24_protocol = read_json(v24_protocol_path)
v24_manifest = read_json(v24_manifest_path)
v30_run = read_json(v30_run_path)
v24_scorer_source = v24_scorer_path.read_text(encoding="utf-8")

m2_grid = [canonical_alpha(a) for a in v18_protocol["m2"]["alpha_grid"]]
v17_grid = [canonical_alpha(a) for a in v17_protocol["search"]["alpha_grid"]]
v24_grid = [canonical_alpha(a) for a in v24_protocol["methods"]["M2_original_scalar"]["alpha_grid"]]
if m2_grid != GRID_STR or v17_grid != GRID_STR or v24_grid != GRID_STR:
    raise AssertionError(f"Grid mismatch: v18={m2_grid}, v17={v17_grid}, v24={v24_grid}")
if v18_protocol["model"]["id"] != v24_protocol["model"]["id"]:
    raise AssertionError("M2/v24 model id mismatch")
if v18_protocol["model"]["revision"] != v24_protocol["model"]["revision"]:
    raise AssertionError("M2/v24 model revision mismatch")
if v18_protocol["model"]["dtype"] != v24_protocol["model"]["dtype"] or v18_protocol["model"]["dtype"] != "FP16":
    raise AssertionError("M2/v24 target weight dtype mismatch")
if int(v18_protocol["quantization"]["group_size"]) != 128 or int(v24_protocol["quantization"]["group_size"]) != 128:
    raise AssertionError("M2/v24 G128 quantization semantics mismatch")
if not v24_protocol["quantization"].get("dynamic_groups", False):
    raise AssertionError("v24 candidate bank is not dynamic G128")
if v18_protocol["m2"]["selection_unit"] != "independent alpha per output row, shared across all G128 groups of that row":
    raise AssertionError("Unexpected M2 row/column selection semantics")
if v18_protocol["m2"]["tie_break"] != "exact loss ties prefer the larger alpha":
    raise AssertionError("Unexpected M2 tie rule")
if v24_protocol["objective"]["selection_data"] != "search documents only":
    raise AssertionError("v24 selection is not explicitly search-only")
if v24_protocol["methods"]["M2_original_scalar"]["selection"] != "per-row local search MSE":
    raise AssertionError("v24 row selection objective mismatch")
if "X @" not in v18_protocol["m2"]["objective"] or "X @" not in v24_protocol["objective"]["formula"]:
    raise AssertionError("M2/v24 direct scorer formula is not explicit")
if "return min(scores, key=lambda alpha: (scores[alpha], -alpha))" not in v24_scorer_source:
    raise AssertionError("v24 source no longer confirms exact ties prefer larger alpha")

M2_TOKENS = int(v18_protocol["calibration"]["tokens_per_document"]) * int(v18_protocol["calibration"]["documents"])
V24_SEARCH_DOCS = 16
V24_TOKENS = int(v24_protocol["documents"]["tokens_per_document"]) * V24_SEARCH_DOCS


# Complete M2 winners, per-row selected direct loss, and alpha=1 direct loss.
m2_rows = []
for raw in read_csv(m2_rows_path):
    row = {
        "module": raw["module"],
        "row": int(raw["row"]),
        "selected_alpha": canonical_alpha(raw["selected_alpha"]),
        "best_mse": float(raw["search_output_mse"]),
        "alpha1_mse": float(raw["alpha1_output_mse"]),
    }
    if row["selected_alpha"] not in GRID_STR:
        raise AssertionError(f"Out-of-grid M2 winner: {row}")
    if not all(math.isfinite(row[k]) for k in ("best_mse", "alpha1_mse")):
        raise AssertionError(f"Non-finite M2 direct score: {row}")
    if row["selected_alpha"] == ALPHA1 and abs(row["best_mse"] - row["alpha1_mse"]) > max(1e-15, abs(row["best_mse"]) * 1e-10):
        raise AssertionError(f"Alpha=1 selected row disagrees with alpha1 score: {row}")
    row["row_key"] = str(row["row"])
    m2_rows.append(row)
expected_rows = 304128
if len(m2_rows) != expected_rows:
    # The workload count is independently published by the completed v18 run.
    raise AssertionError(f"Expected {expected_rows} full-model M2 rows, got {len(m2_rows)}")
m2_by_module = defaultdict(list)
for r in m2_rows:
    m2_by_module[r["module"]].append(r)
if len(m2_by_module) != 168:
    raise AssertionError(f"Expected 168 M2 linears, got {len(m2_by_module)}")
for module, group in m2_by_module.items():
    row_ids = sorted(r["row"] for r in group)
    if row_ids != list(range(len(group))):
        raise AssertionError(f"Non-contiguous or duplicated rows for {module}")

m2_global_counts = Counter(r["selected_alpha"] for r in m2_rows)
m2_module_counts = {m: Counter(r["selected_alpha"] for r in group) for m, group in m2_by_module.items()}
m2_global_winner_set = winner_union(m2_rows)
m2_module_winner_sets = {m: winner_union(group) for m, group in m2_by_module.items()}
if sum(m2_global_counts.values()) != len(m2_rows):
    raise AssertionError("M2 global alpha counts do not sum to all rows")

# Cross-check the pre-existing M2 alpha distribution and module summary.
m2_distribution = read_csv(m2_alpha_distribution_path)
published_global = {
    canonical_alpha(r["alpha"]): int(r["rows"])
    for r in m2_distribution
    if r["scope"] == "full_model"
}
if published_global and published_global != {a: m2_global_counts.get(a, 0) for a in GRID_STR}:
    raise AssertionError("M2_ROW_RESULTS alpha counts disagree with saved M2_ALPHA_DISTRIBUTION")
published_modules = {r["module"]: r for r in read_csv(m2_module_summary_path)}
if set(published_modules) != set(m2_by_module):
    raise AssertionError("M2_MODULE_SUMMARY module coverage mismatch")
for module, raw in published_modules.items():
    hist = json.loads(raw["selected_alpha_histogram"])
    for a in GRID_STR:
        if int(hist.get(a, 0)) != m2_module_counts[module].get(a, 0):
            raise AssertionError(f"M2 module histogram mismatch for {module} alpha={a}")


# The saved v24/v30 direct table is complete for its fixed representative rows,
# and explicitly contains search-side scores only in this file.
v30_raw = read_csv(v30_search_path)
sample_grid = defaultdict(dict)
sample_alpha_rows = []
sample_meta = defaultdict(dict)
for raw in v30_raw:
    a = canonical_alpha(raw["alpha"])
    if a not in GRID_STR:
        raise AssertionError(f"Out-of-grid v24 score: {raw}")
    layer = int(raw["layer"])
    projection = raw["projection"]
    split_id = raw["split_id"]
    row_id = int(raw["row"])
    unit = f"L{layer}_{projection}_{split_id}"
    row_key = f"{unit}_R{row_id}"
    if a in sample_grid[unit].setdefault(row_key, {}):
        raise AssertionError(f"Duplicate v24 direct score for {row_key}, alpha={a}")
    mse = float(raw["search_output_mse"])
    sse = float(raw["search_output_sse"])
    if not math.isfinite(mse) or not math.isfinite(sse):
        raise AssertionError(f"Non-finite v24 score for {row_key}, alpha={a}")
    sample_grid[unit][row_key][a] = {"mse": mse, "sse": sse}
    sample_meta[unit][row_key] = {
        "unit_key": unit,
        "module_key": f"L{layer}_{projection}",
        "layer": layer,
        "projection": projection,
        "split_id": split_id,
        "row": row_id,
        "row_key": row_key,
        "published_original_alpha": canonical_alpha(raw["original_search_selected_alpha"]),
    }
    sample_alpha_rows.append({
        **sample_meta[unit][row_key], "alpha": a, "score_mse": mse, "score_sse": sse,
        "published_selected_flag": str(raw["is_original_search_selected"]).strip().lower() in ("true", "1", "yes"),
        "candidate_file_sha256": raw.get("candidate_tensor_file_sha256", ""),
        "search_input_sha256": raw.get("search_input_bytes_sha256", ""),
    })

sample_units = sorted(sample_grid)
if len(sample_alpha_rows) != 480 or len(sample_units) != 12:
    raise AssertionError(f"Expected v24/v30 direct score coverage 480 row-alpha scores/12 Linear-bank units; got {len(sample_alpha_rows)}/{len(sample_units)}")
for unit in sample_units:
    if len(sample_grid[unit]) != 4:
        raise AssertionError(f"Expected 4 sampled output rows for {unit}")
    for row_key, scores in sample_grid[unit].items():
        if set(scores) != set(GRID_STR):
            raise AssertionError(f"Incomplete candidate grid for {row_key}: {sorted(scores)}")

sample_by_unit = defaultdict(list)
sample_by_global_row = {}
sample_original_choices = {}
sample_margins_by_unit = {}
sample_rows = []
for unit in sample_units:
    for row_key, score_map in sorted(sample_grid[unit].items()):
        meta = sample_meta[unit][row_key]
        best_a, best_score = direct_best(score_map)
        sample_rows.append({
            **meta,
            "score_mse": best_score["mse"],
            "score_sse": best_score["sse"],
            "recomputed_best_alpha": best_a,
        })
sample_margin_global = direct_margin(sample_rows, sample_grid)
for r in sample_rows:
    sample_by_unit[r["unit_key"]].append(r)
    score_map = sample_grid[r["unit_key"]][r["row_key"]]
    best_a, best_score = direct_best(score_map)
    if best_a != r["published_original_alpha"]:
        raise AssertionError(f"Recomputed v24 direct selection {best_a} != saved choice {r['published_original_alpha']} for {r['row_key']}")
    flags = [x["published_selected_flag"] for x in sample_alpha_rows if x["row_key"] == r["row_key"]]
    if sum(flags) != 1:
        raise AssertionError(f"Expected one saved winner flag for {r['row_key']}, got {sum(flags)}")
    for candidate_row in (x for x in sample_alpha_rows if x["row_key"] == r["row_key"]):
        if candidate_row["published_selected_flag"] != (candidate_row["alpha"] == best_a):
            raise AssertionError(f"Saved winner flag mismatch for {r['row_key']} alpha={candidate_row['alpha']}")
    r["recomputed_best_alpha"] = best_a
    r["recomputed_best_mse"] = best_score["mse"]
    r["recomputed_best_sse"] = best_score["sse"]
    sample_original_choices[r["row_key"]] = best_a
for unit, rows in sample_by_unit.items():
    sample_margins_by_unit[unit] = direct_margin(rows, sample_grid)


# SUBSET_RESULTS: full-model subsets are ranked by saved search-only winner
# frequency; sample subsets are chosen by exact direct search SSE on saved rows.
subset_fields = [
    "scope", "unit", "budget", "subset_strategy", "subset_alpha_set",
    "subset_candidate_count", "rows", "baseline_search_loss_mse_sum", "baseline_search_loss_sse_sum",
    "original_selection_retained_rows",
    "selection_retention_pct", "rows_changed", "search_loss_delta_exact_mse_sum",
    "search_loss_delta_lower_bound_mse_sum", "search_loss_delta_upper_bound_mse_sum",
    "search_loss_delta_exact_sse_sum", "search_loss_delta_upper_bound_sse_sum",
    "search_loss_increase_pct_exact", "search_loss_increase_pct_upper_bound",
    "worst_row_delta_mse", "worst_row_delta_sse", "worst_row_delta_pct", "worst_row_delta_bound_type",
    "worst_row_id", "worst_row_original_alpha", "worst_row_subset_alpha",
    "best_second_gap_mean_mse", "best_second_gap_median_mse",
    "best_second_gap_min_mse", "zero_gap_rows", "strict_exclusive_winner_rows",
    "candidate_executions_in_scope", "loss_status", "subset_selection_basis",
]
subset_results = []

# Full M2: global fixed sets and per-Linear sets, all chosen from M2 search.
for k in BUDGETS:
    subset = top_frequency_subset(m2_rows, k)
    metrics = m2_subset_metrics(m2_rows, subset)
    subset_results.append({
        "scope": "M2_FULL_GLOBAL", "unit": "GLOBAL_FIXED",
        "budget": k, "subset_strategy": "mandatory_alpha1_plus_global_top_winner_frequency",
        "subset_alpha_set": alpha_set_string(subset), "subset_candidate_count": len(subset),
        "rows": metrics["rows"], "original_selection_retained_rows": metrics["retained"],
        "baseline_search_loss_mse_sum": metrics["baseline_mse_sum"],
        "baseline_search_loss_sse_sum": metrics["baseline_mse_sum"] * M2_TOKENS,
        "selection_retention_pct": metrics["retention_pct"], "rows_changed": metrics["changed"],
        "search_loss_delta_exact_mse_sum": metrics["exact_delta_mse_sum"],
        "search_loss_delta_lower_bound_mse_sum": metrics["delta_lower_mse_sum"],
        "search_loss_delta_upper_bound_mse_sum": metrics["delta_upper_mse_sum"],
        "search_loss_delta_exact_sse_sum": None,
        "search_loss_delta_upper_bound_sse_sum": metrics["delta_upper_sse_sum"],
        "search_loss_increase_pct_exact": 0.0 if metrics["changed"] == 0 else None,
        "search_loss_increase_pct_upper_bound": metrics["delta_upper_pct"],
        "worst_row_delta_mse": None if metrics["changed"] else 0.0,
        "worst_row_delta_sse": None if metrics["changed"] else 0.0,
        "worst_row_delta_pct": metrics["worst_upper_pct"],
        "worst_row_delta_bound_type": "alpha1_upper_bound" if metrics["changed"] else "exact_zero",
        "worst_row_id": metrics["worst_row"],
        "best_second_gap_mean_mse": None, "best_second_gap_median_mse": None,
        "best_second_gap_min_mse": None, "zero_gap_rows": None,
        "strict_exclusive_winner_rows": None,
        "candidate_executions_in_scope": len(m2_by_module) * k,
        "loss_status": "EXACT_ZERO" if metrics["changed"] == 0 else "UNKNOWN_WITH_ALPHA1_UPPER_BOUND",
        "subset_selection_basis": "search-only M2 selected-alpha counts; maximizes preserved row winners given alpha1",
    })
    fail_if_over_budget()

    for module, group in sorted(m2_by_module.items()):
        module_subset = top_frequency_subset(group, k)
        mm = m2_subset_metrics(group, module_subset)
        subset_results.append({
            "scope": "M2_FULL_PER_MODULE", "unit": module,
            "budget": k, "subset_strategy": "mandatory_alpha1_plus_module_top_winner_frequency",
            "subset_alpha_set": alpha_set_string(module_subset), "subset_candidate_count": len(module_subset),
            "rows": mm["rows"], "original_selection_retained_rows": mm["retained"],
            "baseline_search_loss_mse_sum": mm["baseline_mse_sum"],
            "baseline_search_loss_sse_sum": mm["baseline_mse_sum"] * M2_TOKENS,
            "selection_retention_pct": mm["retention_pct"], "rows_changed": mm["changed"],
            "search_loss_delta_exact_mse_sum": mm["exact_delta_mse_sum"],
            "search_loss_delta_lower_bound_mse_sum": mm["delta_lower_mse_sum"],
            "search_loss_delta_upper_bound_mse_sum": mm["delta_upper_mse_sum"],
            "search_loss_delta_exact_sse_sum": None,
            "search_loss_delta_upper_bound_sse_sum": mm["delta_upper_sse_sum"],
            "search_loss_increase_pct_exact": 0.0 if mm["changed"] == 0 else None,
            "search_loss_increase_pct_upper_bound": mm["delta_upper_pct"],
            "worst_row_delta_mse": None if mm["changed"] else 0.0,
            "worst_row_delta_sse": None if mm["changed"] else 0.0,
            "worst_row_delta_pct": mm["worst_upper_pct"],
            "worst_row_delta_bound_type": "alpha1_upper_bound" if mm["changed"] else "exact_zero",
            "worst_row_id": mm["worst_row"],
            "best_second_gap_mean_mse": None, "best_second_gap_median_mse": None,
            "best_second_gap_min_mse": None, "zero_gap_rows": None,
            "strict_exclusive_winner_rows": None,
            "candidate_executions_in_scope": len(module_subset),
            "loss_status": "EXACT_ZERO" if mm["changed"] == 0 else "UNKNOWN_WITH_ALPHA1_UPPER_BOUND",
            "subset_selection_basis": "search-only M2 selected-alpha counts; maximizes preserved row winners given alpha1",
        })
    fail_if_over_budget()

# Full M2 no-change reference sets, despite not being one of the requested 3-5 budgets.
global_union_metrics = m2_subset_metrics(m2_rows, m2_global_winner_set)
subset_results.append({
    "scope": "M2_FULL_GLOBAL", "unit": "GLOBAL_FIXED_ALL_WINNERS",
    "budget": len(m2_global_winner_set), "subset_strategy": "all_alpha1_plus_full_model_winner_union",
    "subset_alpha_set": alpha_set_string(m2_global_winner_set), "subset_candidate_count": len(m2_global_winner_set),
    "rows": global_union_metrics["rows"], "original_selection_retained_rows": global_union_metrics["retained"],
    "baseline_search_loss_mse_sum": global_union_metrics["baseline_mse_sum"],
    "baseline_search_loss_sse_sum": global_union_metrics["baseline_mse_sum"] * M2_TOKENS,
    "selection_retention_pct": global_union_metrics["retention_pct"], "rows_changed": global_union_metrics["changed"],
    "search_loss_delta_exact_mse_sum": 0.0, "search_loss_delta_lower_bound_mse_sum": 0.0,
    "search_loss_delta_upper_bound_mse_sum": 0.0, "search_loss_delta_exact_sse_sum": 0.0,
    "search_loss_delta_upper_bound_sse_sum": 0.0, "search_loss_increase_pct_exact": 0.0,
    "search_loss_increase_pct_upper_bound": 0.0, "worst_row_delta_mse": 0.0,
    "worst_row_delta_sse": 0.0, "worst_row_delta_pct": 0.0, "worst_row_id": "",
    "best_second_gap_mean_mse": None, "best_second_gap_median_mse": None,
    "best_second_gap_min_mse": None, "zero_gap_rows": None,
    "strict_exclusive_winner_rows": None, "candidate_executions_in_scope": len(m2_by_module) * len(m2_global_winner_set),
    "loss_status": "EXACT_ZERO_ON_SAVED_M2_CHOICES", "subset_selection_basis": "complete observed full-model winner union from search",
})
module_union_execs = sum(len(s) for s in m2_module_winner_sets.values())
subset_results.append({
    "scope": "M2_FULL_PER_MODULE", "unit": "ALL_168_MODULES_WINNER_UNIONS",
    "budget": max(map(len, m2_module_winner_sets.values())), "subset_strategy": "each_module_alpha1_plus_its_full_winner_union",
    "subset_alpha_set": "MODULE_SPECIFIC_SETS_IN_M2_FULL_MODULE_ROWS", "subset_candidate_count": module_union_execs,
    "rows": len(m2_rows), "original_selection_retained_rows": len(m2_rows), "selection_retention_pct": 100.0,
    "baseline_search_loss_mse_sum": math.fsum(r["best_mse"] for r in m2_rows),
    "baseline_search_loss_sse_sum": math.fsum(r["best_mse"] for r in m2_rows) * M2_TOKENS,
    "rows_changed": 0, "search_loss_delta_exact_mse_sum": 0.0,
    "search_loss_delta_lower_bound_mse_sum": 0.0, "search_loss_delta_upper_bound_mse_sum": 0.0,
    "search_loss_delta_exact_sse_sum": 0.0, "search_loss_delta_upper_bound_sse_sum": 0.0,
    "search_loss_increase_pct_exact": 0.0, "search_loss_increase_pct_upper_bound": 0.0,
    "worst_row_delta_mse": 0.0, "worst_row_delta_sse": 0.0, "worst_row_delta_pct": 0.0,
    "worst_row_id": "", "best_second_gap_mean_mse": None, "best_second_gap_median_mse": None,
    "best_second_gap_min_mse": None, "zero_gap_rows": None, "strict_exclusive_winner_rows": None,
    "candidate_executions_in_scope": module_union_execs,
    "loss_status": "EXACT_ZERO_ON_SAVED_M2_CHOICES", "subset_selection_basis": "full search winner union separately for each Linear",
})
for module, group in sorted(m2_by_module.items()):
    module_union = m2_module_winner_sets[module]
    mm = m2_subset_metrics(group, module_union)
    subset_results.append({
        "scope": "M2_FULL_PER_MODULE_WINNER_UNION", "unit": module,
        "budget": len(module_union), "subset_strategy": "alpha1_plus_all_winners_in_this_Linear",
        "subset_alpha_set": alpha_set_string(module_union), "subset_candidate_count": len(module_union),
        "rows": mm["rows"], "original_selection_retained_rows": mm["retained"],
        "baseline_search_loss_mse_sum": mm["baseline_mse_sum"],
        "baseline_search_loss_sse_sum": mm["baseline_mse_sum"] * M2_TOKENS,
        "selection_retention_pct": mm["retention_pct"], "rows_changed": mm["changed"],
        "search_loss_delta_exact_mse_sum": 0.0, "search_loss_delta_lower_bound_mse_sum": 0.0,
        "search_loss_delta_upper_bound_mse_sum": 0.0, "search_loss_delta_exact_sse_sum": 0.0,
        "search_loss_delta_upper_bound_sse_sum": 0.0, "search_loss_increase_pct_exact": 0.0,
        "search_loss_increase_pct_upper_bound": 0.0, "worst_row_delta_mse": 0.0,
        "worst_row_delta_sse": 0.0, "worst_row_delta_pct": 0.0,
        "worst_row_delta_bound_type": "exact_zero_on_saved_choices", "worst_row_id": "",
        "best_second_gap_mean_mse": None, "best_second_gap_median_mse": None,
        "best_second_gap_min_mse": None, "zero_gap_rows": None,
        "strict_exclusive_winner_rows": None, "candidate_executions_in_scope": len(module_union),
        "loss_status": "EXACT_ZERO_ON_SAVED_M2_CHOICES", "subset_selection_basis": "complete observed winner union for this Linear",
    })

# V24/V30 representative-row exact direct search subset design. The search
# score matrix is complete only for the 48 fixed rows, not complete Linears.
for k in BUDGETS:
    for scope_name, rows in [("V24_SAMPLE_GLOBAL", sample_rows)]:
        subset, _ = choose_best_sample_subset(rows, k, sample_grid)
        mm = sample_subset_metrics(rows, subset, sample_grid)
        mg = sample_margin_global
        subset_results.append({
            "scope": scope_name, "unit": "GLOBAL_FIXED",
            "budget": k, "subset_strategy": "minimum_pooled_direct_search_SSE_with_alpha1",
            "subset_alpha_set": alpha_set_string(subset), "subset_candidate_count": len(subset),
            "rows": mm["rows"], "original_selection_retained_rows": mm["retained"],
            "baseline_search_loss_mse_sum": None,
            "baseline_search_loss_sse_sum": mm["baseline_sse_sum"],
            "selection_retention_pct": mm["retention_pct"], "rows_changed": mm["changed"],
            "search_loss_delta_exact_mse_sum": None,
            "search_loss_delta_lower_bound_mse_sum": None,
            "search_loss_delta_upper_bound_mse_sum": None,
            "search_loss_delta_exact_sse_sum": mm["delta_sse_sum"],
            "search_loss_delta_upper_bound_sse_sum": mm["delta_sse_sum"],
            "search_loss_increase_pct_exact": mm["delta_pct"],
            "search_loss_increase_pct_upper_bound": mm["delta_pct"],
            "worst_row_delta_mse": None, "worst_row_delta_sse": mm["worst_delta_sse"],
            "worst_row_delta_pct": mm["worst_delta_pct"], "worst_row_delta_bound_type": "exact_direct_search_score",
            "worst_row_id": mm["worst_row"],
            "worst_row_original_alpha": mm["worst_original_alpha"],
            "worst_row_subset_alpha": mm["worst_subset_alpha"],
            "best_second_gap_mean_mse": math.fsum(mg["gaps"]) / len(mg["gaps"]),
            "best_second_gap_median_mse": pctl(mg["gaps"], .5), "best_second_gap_min_mse": min(mg["gaps"]),
            "zero_gap_rows": mg["tie_rows"],
            "strict_exclusive_winner_rows": sum(mg["strict_by_alpha"].values()),
            "candidate_executions_in_scope": len(sample_units) * k,
            "loss_status": "EXACT_DIRECT_SCORE_ON_48_FIXED_ROWS",
            "subset_selection_basis": "search-only v24 direct scores; alpha1 mandatory; held/oracle/PPL unused",
        })

    for unit, rows in sorted(sample_by_unit.items()):
        subset, _ = choose_best_sample_subset(rows, k, sample_grid)
        mm = sample_subset_metrics(rows, subset, sample_grid)
        mg = sample_margins_by_unit[unit]
        subset_results.append({
            "scope": "V24_SAMPLE_PER_MODULE_BANK", "unit": unit,
            "budget": k, "subset_strategy": "minimum_module_bank_direct_search_SSE_with_alpha1",
            "subset_alpha_set": alpha_set_string(subset), "subset_candidate_count": len(subset),
            "rows": mm["rows"], "original_selection_retained_rows": mm["retained"],
            "baseline_search_loss_mse_sum": None,
            "baseline_search_loss_sse_sum": mm["baseline_sse_sum"],
            "selection_retention_pct": mm["retention_pct"], "rows_changed": mm["changed"],
            "search_loss_delta_exact_mse_sum": None,
            "search_loss_delta_lower_bound_mse_sum": None,
            "search_loss_delta_upper_bound_mse_sum": None,
            "search_loss_delta_exact_sse_sum": mm["delta_sse_sum"],
            "search_loss_delta_upper_bound_sse_sum": mm["delta_sse_sum"],
            "search_loss_increase_pct_exact": mm["delta_pct"],
            "search_loss_increase_pct_upper_bound": mm["delta_pct"],
            "worst_row_delta_mse": None, "worst_row_delta_sse": mm["worst_delta_sse"],
            "worst_row_delta_pct": mm["worst_delta_pct"], "worst_row_delta_bound_type": "exact_direct_search_score",
            "worst_row_id": mm["worst_row"],
            "worst_row_original_alpha": mm["worst_original_alpha"],
            "worst_row_subset_alpha": mm["worst_subset_alpha"],
            "best_second_gap_mean_mse": math.fsum(mg["gaps"]) / len(mg["gaps"]),
            "best_second_gap_median_mse": pctl(mg["gaps"], .5), "best_second_gap_min_mse": min(mg["gaps"]),
            "zero_gap_rows": mg["tie_rows"],
            "strict_exclusive_winner_rows": sum(mg["strict_by_alpha"].values()),
            "candidate_executions_in_scope": len(subset),
            "loss_status": "EXACT_DIRECT_SCORE_ON_4_FIXED_ROWS",
            "subset_selection_basis": "search-only v24 direct scores per Linear-bank; alpha1 mandatory",
        })
    fail_if_over_budget()

# Baseline rows expose exact direct-score margins for the sample, not the full M2.
for scope_name, unit, rows, mg, executions in [
    ("V24_SAMPLE_GLOBAL", "GLOBAL_FIXED", sample_rows, sample_margin_global, len(sample_units) * 10),
    *[("V24_SAMPLE_PER_MODULE_BANK", u, r, sample_margins_by_unit[u], 10) for u, r in sorted(sample_by_unit.items())],
]:
    subset_results.append({
        "scope": scope_name, "unit": unit, "budget": 10, "subset_strategy": "original_scalar10_direct_scores",
        "subset_alpha_set": alpha_set_string(GRID_STR), "subset_candidate_count": 10,
        "rows": len(rows), "original_selection_retained_rows": len(rows), "selection_retention_pct": 100.0,
        "baseline_search_loss_mse_sum": None,
        "baseline_search_loss_sse_sum": math.fsum(r["score_sse"] for r in rows),
        "rows_changed": 0, "search_loss_delta_exact_mse_sum": 0.0,
        "search_loss_delta_lower_bound_mse_sum": 0.0, "search_loss_delta_upper_bound_mse_sum": 0.0,
        "search_loss_delta_exact_sse_sum": 0.0, "search_loss_delta_upper_bound_sse_sum": 0.0,
        "search_loss_increase_pct_exact": 0.0, "search_loss_increase_pct_upper_bound": 0.0,
        "worst_row_delta_mse": 0.0, "worst_row_delta_sse": 0.0, "worst_row_delta_pct": 0.0,
        "worst_row_id": "", "worst_row_delta_bound_type": "exact_zero",
        "best_second_gap_mean_mse": math.fsum(mg["gaps"]) / len(mg["gaps"]),
        "best_second_gap_median_mse": pctl(mg["gaps"], .5), "best_second_gap_min_mse": min(mg["gaps"]),
        "zero_gap_rows": mg["tie_rows"], "strict_exclusive_winner_rows": sum(mg["strict_by_alpha"].values()),
        "candidate_executions_in_scope": executions,
        "loss_status": "ORIGINAL_DIRECT_SCORE_BASELINE",
        "subset_selection_basis": "none",
    })

write_csv(OUT / "SUBSET_RESULTS.csv", subset_fields, subset_results)


# ALPHA_USAGE: full M2 selected winners and sample direct-score winners. The
# per-module M2 records list every served output row; no M2 second-best score
# is fabricated from the saved winner-only file.
usage_fields = [
    "scope", "unit", "alpha", "selected_rows", "row_fraction", "service_row_ids",
    "strict_exclusive_winner_rows", "best_second_gap_mean_mse", "best_second_gap_median_mse",
    "best_second_gap_min_mse", "zero_gap_rows", "removal_selection_retained_rows",
    "removal_search_delta_exact_sse_sum", "removal_search_delta_lower_bound_mse_sum",
    "removal_search_delta_upper_bound_mse_sum", "removal_search_delta_upper_bound_sse_sum",
    "removal_search_loss_upper_bound_pct", "removal_worst_row_delta_upper_pct",
    "removal_worst_row_id", "removal_status",
]
usage_rows = []

def m2_removal_metrics(rows, alpha):
    if alpha == ALPHA1:
        return {"retained": None, "lower": None, "upper": None, "upper_sse": None, "upper_pct": None, "worst_pct": None, "worst_row": "", "status": "ALPHA1_MANDATORY"}
    served = [r for r in rows if r["selected_alpha"] == alpha]
    total = math.fsum(r["best_mse"] for r in rows)
    bounds = [(max(0.0, r["alpha1_mse"] - r["best_mse"]), r) for r in served]
    upper = math.fsum(x[0] for x in bounds)
    worst = max(bounds, key=lambda x: x[0] / max(abs(x[1]["best_mse"]), 1e-300), default=None)
    return {
        "retained": len(rows) - len(served), "lower": 0.0, "upper": upper,
        "upper_sse": upper * M2_TOKENS, "upper_pct": 100.0 * upper / max(total, 1e-300),
        "worst_pct": 100.0 * worst[0] / max(abs(worst[1]["best_mse"]), 1e-300) if worst else 0.0,
        "worst_row": f"{worst[1]['module']}#R{worst[1]['row']}" if worst else "",
        "status": "LOSS_UNKNOWN_ALPHA1_FALLBACK_UPPER_BOUND" if served else "NO_ROWS_SELECTED_THIS_ALPHA",
    }


def sample_removal_metrics(rows, alpha, score_data):
    if alpha == ALPHA1:
        return {"retained": None, "delta": None, "delta_pct": None, "worst": None, "worst_pct": None, "worst_row": "", "status": "ALPHA1_MANDATORY"}
    original_base = math.fsum(score_data[r["unit_key"]][r["row_key"]][direct_best(score_data[r["unit_key"]][r["row_key"]])[0]]["sse"] for r in rows)
    remaining = [a for a in GRID_STR if a != alpha]
    deltas = []
    retained = 0
    for r in rows:
        scores = score_data[r["unit_key"]][r["row_key"]]
        old_a, old_score = direct_best(scores)
        new_a, new_score = direct_best(scores, remaining)
        if old_a == new_a:
            retained += 1
        delta = max(0.0, new_score["sse"] - old_score["sse"])
        deltas.append((delta, r, old_score, new_score, old_a, new_a))
    total_delta = math.fsum(x[0] for x in deltas)
    worst = max(deltas, key=lambda x: (x[0] / max(abs(x[2]["sse"]), 1e-300), x[0]), default=None)
    return {
        "retained": retained, "delta": total_delta,
        "delta_pct": 100.0 * total_delta / max(original_base, 1e-300),
        "worst": worst[0] if worst else 0.0,
        "worst_pct": 100.0 * worst[0] / max(abs(worst[2]["sse"]), 1e-300) if worst else 0.0,
        "worst_row": f"{worst[1]['unit_key']}#R{worst[1]['row']}" if worst and worst[0] > 0 else "",
        "status": "EXACT_DIRECT_SEARCH_SCORE_REMOVAL",
    }


def append_usage(scope, unit, alpha, count, total, service_ids, strict, margins,
                 remove_metrics, is_m2=False):
    usage_rows.append({
        "scope": scope, "unit": unit, "alpha": alpha,
        "selected_rows": count, "row_fraction": count / total if total else 0.0,
        "service_row_ids": ";".join(service_ids),
        "strict_exclusive_winner_rows": strict,
        "best_second_gap_mean_mse": (math.fsum(margins["gaps"]) / len(margins["gaps"])) if margins else None,
        "best_second_gap_median_mse": pctl(margins["gaps"], .5) if margins else None,
        "best_second_gap_min_mse": min(margins["gaps"]) if margins else None,
        "zero_gap_rows": margins["tie_rows"] if margins else None,
        "removal_selection_retained_rows": remove_metrics.get("retained"),
        "removal_search_delta_exact_sse_sum": None if is_m2 else remove_metrics.get("delta"),
        "removal_search_delta_lower_bound_mse_sum": remove_metrics.get("lower"),
        "removal_search_delta_upper_bound_mse_sum": remove_metrics.get("upper"),
        "removal_search_delta_upper_bound_sse_sum": remove_metrics.get("upper_sse"),
        "removal_search_loss_upper_bound_pct": remove_metrics.get("upper_pct") if is_m2 else remove_metrics.get("delta_pct"),
        "removal_worst_row_delta_upper_pct": remove_metrics.get("worst_pct"),
        "removal_worst_row_id": remove_metrics.get("worst_row"),
        "removal_status": remove_metrics.get("status"),
    })


for alpha in GRID_STR:
    served = [r for r in m2_rows if r["selected_alpha"] == alpha]
    ids = [f"{r['module']}#R{r['row']}" for r in served]
    rm = m2_removal_metrics(m2_rows, alpha)
    append_usage("M2_FULL_GLOBAL", "GLOBAL", alpha, len(served), len(m2_rows), [], None, None, rm, True)
    for module, group in sorted(m2_by_module.items()):
        mserved = [r for r in group if r["selected_alpha"] == alpha]
        mrm = m2_removal_metrics(group, alpha)
        append_usage("M2_FULL_PER_MODULE", module, alpha, len(mserved), len(group),
                     [str(r["row"]) for r in mserved], None, None, mrm, True)

for scope, unit, rows, margins in [
    ("V24_SAMPLE_GLOBAL", "GLOBAL", sample_rows, sample_margin_global),
    *[("V24_SAMPLE_PER_MODULE_BANK", u, r, sample_margins_by_unit[u]) for u, r in sorted(sample_by_unit.items())],
]:
    total = len(rows)
    for alpha in GRID_STR:
        count = margins["selected_by_alpha"].get(alpha, 0)
        ids = margins["selected_rows_by_alpha"].get(alpha, [])
        rm = sample_removal_metrics(rows, alpha, sample_grid)
        append_usage(scope, unit, alpha, count, total, ids,
                     margins["strict_by_alpha"].get(alpha, 0), margins, rm, False)

write_csv(OUT / "ALPHA_USAGE.csv", usage_fields, usage_rows)


# Full Linear candidate executions, never row-alpha score slots. Protocol
# counts are kept apart from projections; no wall-time extrapolation is made.
baseline_m2_exec = int(v18_protocol["m2"]["candidate_calls"])
if baseline_m2_exec != 1680:
    raise AssertionError(f"Unexpected M2 baseline candidate execution count: {baseline_m2_exec}")
baseline_m2_modules = len(m2_by_module)
baseline_sample_exec = int(v30_run["full_linear_candidate_tensor_states"])
if baseline_sample_exec != 120:
    raise AssertionError(f"Unexpected v24 sample full-Linear candidate state count: {baseline_sample_exec}")
cost_rows = [{
    "scope": "M2_FULL_MODEL", "scenario": "historical_scalar10_protocol_count",
    "budget_cap": 10, "modules": baseline_m2_modules,
    "baseline_candidate_executions": baseline_m2_exec,
    "projected_candidate_executions": baseline_m2_exec,
    "projected_executions_saved": 0, "projected_reduction_pct": 0.0,
    "count_basis": "saved v18 protocol; actual completed search workload",
    "time_status": "actual M2 calibration wall in existing run report is not used as a linear projection",
    "notes": "Candidate executions are complete Linear GPTQ candidates; no row-alpha slot conversion.",
}]
for k in BUDGETS:
    global_subset = top_frequency_subset(m2_rows, k)
    global_count = baseline_m2_modules * len(global_subset)
    module_subsets = {m: top_frequency_subset(group, k) for m, group in m2_by_module.items()}
    per_module_count = sum(len(s) for s in module_subsets.values())
    for scenario, count, desc in [
        ("global_fixed_frequency_subset", global_count, "same alpha set on each full Linear; alpha1 retained"),
        ("per_module_frequency_subset_no_fill", per_module_count, "per-Linear top-frequency winners; alpha1 retained; no unused filler candidates"),
    ]:
        cost_rows.append({
            "scope": "M2_FULL_MODEL", "scenario": scenario, "budget_cap": k,
            "modules": baseline_m2_modules, "baseline_candidate_executions": baseline_m2_exec,
            "projected_candidate_executions": count,
            "projected_executions_saved": baseline_m2_exec - count,
            "projected_reduction_pct": 100.0 * (baseline_m2_exec - count) / baseline_m2_exec,
            "count_basis": "projection from existing full-row M2 search choices; no new GPTQ run",
            "time_status": "count-only projection; runtime savings not measured or linearly extrapolated",
            "notes": desc,
        })
cost_rows.extend([
    {
        "scope": "M2_FULL_MODEL", "scenario": "global_all_observed_winner_alphas",
        "budget_cap": len(m2_global_winner_set), "modules": baseline_m2_modules,
        "baseline_candidate_executions": baseline_m2_exec,
        "projected_candidate_executions": baseline_m2_modules * len(m2_global_winner_set),
        "projected_executions_saved": baseline_m2_exec - baseline_m2_modules * len(m2_global_winner_set),
        "projected_reduction_pct": 100.0 * (baseline_m2_exec - baseline_m2_modules * len(m2_global_winner_set)) / baseline_m2_exec,
        "count_basis": "winner-preserving set derived from all 304128 saved M2 rows",
        "time_status": "count-only projection; same saved row choices are preserved by construction",
        "notes": f"Global set={alpha_set_string(m2_global_winner_set)}; exact on stored M2 row choices.",
    },
    {
        "scope": "M2_FULL_MODEL", "scenario": "per_module_all_observed_winner_unions",
        "budget_cap": max(map(len, m2_module_winner_sets.values())), "modules": baseline_m2_modules,
        "baseline_candidate_executions": baseline_m2_exec,
        "projected_candidate_executions": module_union_execs,
        "projected_executions_saved": baseline_m2_exec - module_union_execs,
        "projected_reduction_pct": 100.0 * (baseline_m2_exec - module_union_execs) / baseline_m2_exec,
        "count_basis": "winner-preserving module-specific sets derived from all 304128 saved M2 rows",
        "time_status": "count-only projection; same saved row choices are preserved by construction",
        "notes": "Per-Linear alpha set is alpha1 union the values that won at least one saved row.",
    },
    {
        "scope": "V24_SAMPLE_FIXED_ROWS", "scenario": "historical_scalar10_full_linear_candidates",
        "budget_cap": 10, "modules": len(sample_units), "baseline_candidate_executions": baseline_sample_exec,
        "projected_candidate_executions": baseline_sample_exec, "projected_executions_saved": 0,
        "projected_reduction_pct": 0.0, "count_basis": "saved v30 run metadata; historical candidate tensor states",
        "time_status": "historical count only; no additional candidate executions in this precheck",
        "notes": "12 target-bank Linear states × 10 alpha; representative row scores do not multiply GPTQ savings.",
    },
])
for k in BUDGETS:
    cost_rows.append({
        "scope": "V24_SAMPLE_FIXED_ROWS", "scenario": "global_fixed_subset_projection",
        "budget_cap": k, "modules": len(sample_units), "baseline_candidate_executions": baseline_sample_exec,
        "projected_candidate_executions": len(sample_units) * k,
        "projected_executions_saved": baseline_sample_exec - len(sample_units) * k,
        "projected_reduction_pct": 100.0 * (baseline_sample_exec - len(sample_units) * k) / baseline_sample_exec,
        "count_basis": "count projection for 12 complete Linear candidate states; no rerun performed",
        "time_status": "count-only projection; not measured time savings",
        "notes": "Direct-score cells reduced by 1-k/10 are not reported as GPTQ savings.",
    })
write_csv(OUT / "COST_PROJECTION.csv", [
    "scope", "scenario", "budget_cap", "modules", "baseline_candidate_executions",
    "projected_candidate_executions", "projected_executions_saved", "projected_reduction_pct",
    "count_basis", "time_status", "notes",
], cost_rows)


# Source audit / scope ledger, including explicit missing evidence.
v24_targets = v24_protocol["targets"]
v24_docs = v24_protocol["documents"]
source_files = [
    ("v18 M2 protocol", v18_protocol_path, "complete formal M2 full-model grid/input/selection semantics"),
    ("v17.3 M2 runtime protocol", v17_protocol_path, "independent confirmation of alpha grid, row unit, baseline count"),
    ("v18 M2 row results", m2_rows_path, "all 168 Linears and 304128 rows; selected alpha + selected loss + alpha1 loss only"),
    ("v18 M2 module summary", m2_module_summary_path, "per-Linear winner histograms crosschecked against row table"),
    ("v18 M2 alpha distribution", m2_alpha_distribution_path, "saved global winner counts crosschecked against row table"),
    ("v24 E2 protocol", v24_protocol_path, "same model/grid/direct row objective/tie rule; only fixed representative rows"),
    ("v24 E2 document manifest", v24_manifest_path, "source metadata for the 64-document exploratory candidate cohort"),
    ("v24 scalar selector source", v24_scorer_path, "source confirms exact direct-score ties select larger alpha; no scorer was executed"),
    ("v30 search candidate table", v30_search_path, "480 existing direct search scores: 48 rows × 10 alphas; no held columns used"),
    ("v30 run summary", v30_run_path, "historical candidate count and no-new-run audit"),
]
source_audit_rows = []
for name, path, role in source_files:
    source_audit_rows.append({
        "source_name": name, "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(path), "bytes": path.stat().st_size,
        "role_and_scope": role, "read_or_used": "yes",
    })
source_audit_rows.extend([
    {
        "source_name": "M2 full-row all-alpha score matrix", "relative_path": "NOT_FOUND_IN_V18_ARTIFACT_INDEX",
        "sha256": "", "bytes": "", "role_and_scope": "second-best gaps and exact reduced-grid losses unavailable for the full M2 rows",
        "read_or_used": "not present in inspected v18 output inventory",
    },
    {
        "source_name": "v24/v30 complete Linear row coverage", "relative_path": str(v30_search_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(v30_search_path), "bytes": v30_search_path.stat().st_size,
        "role_and_scope": f"only {len(sample_units)} target-bank Linears × 4 representative output rows; not all output rows",
        "read_or_used": "yes; sample-only exact direct score analysis",
    },
    {
        "source_name": "M2 versus v24/v30 input correspondence", "relative_path": "PROTOCOL_COMPARISON",
        "sha256": "", "bytes": "",
        "role_and_scope": f"same model revision, FP16, dynamic G128, scalar10 row-wise direct MSE and larger-alpha tie rule; M2 uses {v18_protocol['calibration']['documents']}×{v18_protocol['calibration']['tokens_per_document']} formal calibration tokens, while v24 uses {V24_SEARCH_DOCS} search articles per bank ({V24_TOKENS} tokens) and separate S0/S1 candidate Hessians. v24 candidate inputs are not the exact formal M2 scoring cohort/Hessian.",
        "read_or_used": "protocols only; no new input, model, or score generation",
    },
    {
        "source_name": "P3 artifacts", "relative_path": "EXCLUDED",
        "sha256": "", "bytes": "", "role_and_scope": "not enumerated, opened, hashed, or read",
        "read_or_used": "no",
    },
    {
        "source_name": "v39 historical artifacts", "relative_path": "EXCLUDED",
        "sha256": "", "bytes": "", "role_and_scope": "not opened or modified",
        "read_or_used": "no",
    },
])
write_csv(OUT / "SOURCE_AUDIT.csv", ["source_name", "relative_path", "sha256", "bytes", "role_and_scope", "read_or_used"], source_audit_rows)


def pct_s(n, d):
    return 100.0 * n / d if d else 0.0


alpha_hist = {a: m2_global_counts.get(a, 0) for a in GRID_STR}
module_size_hist = Counter(len(s) for s in m2_module_winner_sets.values())
per_module_budget_summary = {}
m2_global_budget_summary = {}
for k in BUDGETS:
    sets = {m: top_frequency_subset(rows, k) for m, rows in m2_by_module.items()}
    global_subset = top_frequency_subset(m2_rows, k)
    metrics = m2_subset_metrics(m2_rows, global_subset)
    module_metrics = [m2_subset_metrics(m2_by_module[m], sets[m]) for m in m2_by_module]
    m2_global_budget_summary[k] = {
        "subset_alpha_set": alpha_set_string(global_subset),
        "candidate_execs": baseline_m2_modules * len(global_subset),
        "retained_rows": metrics["retained"],
        "retention_pct": metrics["retention_pct"],
        "changed_rows": metrics["changed"],
        "loss_upper_pct": metrics["delta_upper_pct"],
        "worst_upper_pct": metrics["worst_upper_pct"],
    }
    per_module_budget_summary[k] = {
        "candidate_execs": sum(len(s) for s in sets.values()),
        "full_winner_union_preserved_modules": sum(len(m2_module_winner_sets[m]) <= k for m in m2_by_module),
        "module_subset_retention_rows": sum(x["retained"] for x in module_metrics),
        "module_subset_retention_pct": pct_s(sum(x["retained"] for x in module_metrics), len(m2_rows)),
        "module_subset_changed_rows": sum(x["changed"] for x in module_metrics),
        "module_subset_loss_upper_mse": math.fsum(x["delta_upper_mse_sum"] for x in module_metrics),
        "module_subset_loss_upper_pct": 100.0 * math.fsum(x["delta_upper_mse_sum"] for x in module_metrics) / math.fsum(r["best_mse"] for r in m2_rows),
        "module_subset_worst_upper_pct": max((x["worst_upper_pct"] for x in module_metrics), default=0.0),
        "global_subset_retention_rows": metrics["retained"],
        "global_subset_retention_pct": metrics["retention_pct"],
        "global_subset_changed_rows": metrics["changed"],
        "global_subset_loss_upper_pct": metrics["delta_upper_pct"],
    "global_subset_candidate_execs": baseline_m2_modules * len(global_subset),
    "global_subset_alpha_set": alpha_set_string(global_subset),
    }

summary = {
    "status": "COMPLETE_ARTIFACT_ONLY_CANDIDATE_BUDGET_PRECHECK",
    "analysis_cpu_seconds": time.monotonic() - START,
    "cpu_limit_seconds": 300,
    "gpu_initializations": 0,
    "model_forwards": 0,
    "gptq_runs": 0,
    "new_held_scoring": 0,
    "ppl_runs": 0,
    "p3_read": False,
    "v39_modified": False,
    "m2_full_model": {
        "linears": len(m2_by_module), "rows": len(m2_rows), "tokens_per_row": M2_TOKENS,
        "selected_alpha_counts": alpha_hist,
        "global_winner_union_plus_alpha1": alpha_set_string(m2_global_winner_set),
        "global_winner_union_size": len(m2_global_winner_set),
        "module_specific_winner_union_candidate_executions": module_union_execs,
        "module_winner_union_size_histogram": {str(k): module_size_hist[k] for k in sorted(module_size_hist)},
        "module_winner_union_sizes_le_budget": {str(k): sum(len(s) <= k for s in m2_module_winner_sets.values()) for k in BUDGETS},
        "budgets": per_module_budget_summary,
        "global_budget_summary": m2_global_budget_summary,
        "m2_candidate_execution_baseline": baseline_m2_exec,
    },
    "v24_v30_direct_search_scope": {
        "row_alpha_scores": len(sample_alpha_rows), "row_split_units": len(sample_grid),
        "complete_grid_per_row": True, "target_bank_full_linears": len(sample_units),
        "representative_output_rows_per_target_bank": 4,
        "search_tokens_per_bank": V24_TOKENS,
        "full_linear_candidate_tensor_states_historical": baseline_sample_exec,
        "input_cohort_matches_formal_m2": False,
    },
    "scope_limits": [
        "v18 stores full-model chosen alpha, chosen direct loss, and alpha1 direct loss, but no all-ten direct score matrix per row.",
        "Exact best-to-second-best gaps and exact k=3/4/5 restricted-grid loss are unavailable for full M2 rows when a selected alpha is removed; only alpha1-fallback upper bounds are computed.",
        "v24/v30 full-grid direct score analysis covers four fixed representative rows per target-bank Linear, not every output row, and uses a different search cohort/Hessian than formal v18 M2.",
        "All subset design results are retrospective search-data analysis, not new-data validation and do not save the already incurred scalar10 search cost.",
    ],
    "output_files": ["SOURCE_AUDIT.csv", "ALPHA_USAGE.csv", "SUBSET_RESULTS.csv", "COST_PROJECTION.csv", "DECISION.md"],
}
if summary["analysis_cpu_seconds"] > 295:
    raise RuntimeError(f"CPU analysis exceeded 295-second guard: {summary['analysis_cpu_seconds']:.3f}s")
(OUT / "RUN_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# A compact human-readable decision is generated from the audited outputs.
global5 = per_module_budget_summary[5]
module_size_counts = ", ".join(f"{n} alpha: {module_size_hist[n]} modules" for n in sorted(module_size_hist))
alpha_counts_text = ", ".join(f"{a}: {alpha_hist[a]}" for a in GRID_STR)
low_alpha = {a: alpha_hist[a] for a in GRID_STR if alpha_hist[a] <= 3044 and alpha_hist[a] > 0}
sample_global_k = {}
for r in subset_results:
    if r["scope"] == "V24_SAMPLE_GLOBAL" and r["unit"] == "GLOBAL_FIXED" and r["budget"] in BUDGETS:
        sample_global_k[int(r["budget"])] = r

global_budget_lines = [
    "| 候选数上限 | 固定全局子集 | 原选择保留 | 变化行 | search 损失上界 | 最坏行上界 | 完整 Linear 候选执行数 |",
    "|---:|---|---:|---:|---:|---:|---:|",
]
module_budget_lines = [
    "| 候选数上限 | 全赢家集合不超预算的 Linear 数 | 逐模块原选择保留 | 变化行 | search 损失上界 | 最坏行上界 | 完整 Linear 候选执行数 |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for k in BUDGETS:
    g = m2_global_budget_summary[k]
    m = per_module_budget_summary[k]
    global_budget_lines.append(
        f"| {k} | {g['subset_alpha_set']} | {g['retained_rows']}/{len(m2_rows)} ({g['retention_pct']:.6f}%) "
        f"| {g['changed_rows']} | ≤{g['loss_upper_pct']:.6f}% | ≤{g['worst_upper_pct']:.6f}% "
        f"| {g['candidate_execs']} |"
    )
    module_budget_lines.append(
        f"| {k} | {m['full_winner_union_preserved_modules']}/168 "
        f"| {m['module_subset_retention_rows']}/{len(m2_rows)} ({m['module_subset_retention_pct']:.6f}%) "
        f"| {m['module_subset_changed_rows']} | ≤{m['module_subset_loss_upper_pct']:.6f}% "
        f"| ≤{m['module_subset_worst_upper_pct']:.6f}% | {m['candidate_execs']} |"
    )

sample_module_lines = [
    "| 候选数 | 原选择保留 | 变化行 | search 损失增加 | 最坏行变化 | 完整 Linear 候选执行数 |",
    "|---:|---:|---:|---:|---|---:|",
]
for k in BUDGETS:
    sr = [r for r in subset_results if r["scope"] == "V24_SAMPLE_PER_MODULE_BANK" and r["budget"] == k]
    base = math.fsum(float(r["baseline_search_loss_sse_sum"]) for r in sr)
    delta = math.fsum(float(r["search_loss_delta_exact_sse_sum"]) for r in sr)
    retained = sum(int(r["original_selection_retained_rows"]) for r in sr)
    changed = sum(int(r["rows_changed"]) for r in sr)
    worst = max(sr, key=lambda r: float(r["worst_row_delta_pct"] or 0.0))
    executions = sum(int(r["candidate_executions_in_scope"]) for r in sr)
    sample_module_lines.append(
        f"| {k} | {retained}/48 ({100.0*retained/48:.2f}%) | {changed} | {100.0*delta/max(base,1e-300):.6f}% "
        f"| {float(worst['worst_row_delta_pct'] or 0.0):.6f}% ({worst['worst_row_id'] or '—'}) | {executions} |"
    )
sample_global_lines = [
    "| 候选数 | 全局直接分数子集 | 原选择保留 | search 损失增加 | 最坏行变化 | 完整 Linear 候选执行数 |",
    "|---:|---|---:|---:|---|---:|",
]
for k in BUDGETS:
    r = sample_global_k[k]
    sample_global_lines.append(
        f"| {k} | {r['subset_alpha_set']} | {r['original_selection_retained_rows']}/{r['rows']} "
        f"({float(r['selection_retention_pct']):.2f}%) | {float(r['search_loss_increase_pct_exact']):.6f}% "
        f"| {float(r['worst_row_delta_pct'] or 0.0):.6f}% ({r['worst_row_id'] or '—'}) "
        f"| {r['candidate_executions_in_scope']} |"
    )
margin_text = (
    f"best-second gap mean/median/min = "
    f"{math.fsum(sample_margin_global['gaps'])/len(sample_margin_global['gaps']):.9g}/"
    f"{pctl(sample_margin_global['gaps'], .5):.9g}/{min(sample_margin_global['gaps']):.9g} MSE; "
    f"strict exclusive winners = {sum(sample_margin_global['strict_by_alpha'].values())}/48; "
    f"exact-tie rows = {sample_margin_global['tie_rows']}"
)
sample_selection_text = ", ".join(
    f"{a}: {sample_margin_global['selected_by_alpha'].get(a, 0)} rows"
    for a in GRID_STR if sample_margin_global["selected_by_alpha"].get(a, 0)
)
low_alpha_usage = {
    (r["scope"], r["alpha"]): r for r in usage_rows
    if r["scope"] == "M2_FULL_GLOBAL" and r["alpha"] in low_alpha
}
def low_alpha_removal_text(alpha):
    usage = low_alpha_usage[("M2_FULL_GLOBAL", alpha)]
    pooled = float(usage["removal_search_loss_upper_bound_pct"] or 0)
    worst = float(usage["removal_worst_row_delta_upper_pct"] or 0)
    return f"{alpha}: {low_alpha[alpha]} rows, removal loss upper ≤{pooled:.6f}% pooled (worst row ≤{worst:.6f}%)"

low_alpha_text = "; ".join(low_alpha_removal_text(a) for a in low_alpha)

decision = f"""# candidate_budget_precheck 决策

## 结论

**值得冻结一个较小的、逐 Linear 候选集供后续独立验证；本次不启动验证。** 完整 v18 M2 历史表覆盖 168 个 Linear、304,128 个输出行。逐模块保留 alpha=1 与该 Linear 在原 M2 search 中实际获胜的 alpha，合计需要 **{module_union_execs} 个完整 Linear 候选执行**，相对 scalar10 的 1,680 次减少 **{100.0 * (baseline_m2_exec-module_union_execs)/baseline_m2_exec:.2f}%**。按这组候选子集重选，所有已保存 M2 行赢家都仍在候选集中，所以在这些既有 M2 输入/选择结果上行选择保留率为 100%，搜索损失变化为 0。

这是利用已完成十点搜索结果形成的探索性冻结建议；它没有节省已经付出的搜索成本，也不是新数据验证。没有在缩减网格上提交完整模型运行，因此不能据此宣称模型张量或 PPL 完全相同。

## 完整 M2 行结果上的 3/4/5 候选预算

完整 M2 的全局 alpha 计数为：{alpha_counts_text}。全局赢家并集加上必留 alpha=1 共 {len(m2_global_winner_set)} 个候选（{alpha_set_string(m2_global_winner_set)}）；所以固定全局 3、4、5 候选都不能保留所有原逐行选择。

逐模块原赢家并集大小分布为：{module_size_counts}。预算 3/4/5 的模块子集按 search 行赢家频数择优、alpha=1 必留；该规则最大化被保留的原行赢家数，不等于最小化未知的完整 M2 损失。逐模块精确赢家集合（每个 Linear 的 alpha 列表）列在 `SUBSET_RESULTS.csv` 的 `M2_FULL_PER_MODULE_WINNER_UNION` 行。

固定全局候选集和逐模块候选集的结果如下。M2 全模型只有赢家与 alpha=1 直接分数，因此有变化时给出的是 alpha=1 回退上界，实际变化未知。

{chr(10).join(global_budget_lines)}

{chr(10).join(module_budget_lines)}

直接分数代表行上的全局子集：

{chr(10).join(sample_global_lines)}

代表行上的逐模块-bank 子集汇总：

{chr(10).join(sample_module_lines)}

稀有 alpha 仍服务于已有行：{low_alpha_text}。`ALPHA_USAGE.csv` 列出它们在完整 M2 各 Linear 中服务的行号，以及移除该 alpha 后用 alpha=1 作回退得到的损失上界。不能把低频说成无用。

## 直接候选分数与覆盖范围

v18 M2 保留全模型每行的原赢家直接 MSE 和 alpha=1 直接 MSE，但没有保留每行十个 alpha 的完整分数。因此全模型的最优/次优间隔、被移除赢家的精确 3/4/5 子集损失不可恢复。没有用二次型代替这些分数。

v24/v30 可复核到 480 个完整网格直接分数（48 个固定代表行 × 10 alpha），覆盖 12 个目标 Linear-bank、每个仅 4 个输出行。代表行选择分布为 {sample_selection_text}；{margin_text}。该样本中固定 3 候选子集即可保留全部 48 行选择、search 损失增加为 0，但这只适用于这些代表行。表内复现了原逐行选择和“大 alpha 优先”的精确并列规则；其预算子集结果按原 direct search SSE 精确计算，并在 `SUBSET_RESULTS.csv` 给出逐行、逐模块和全局的最优/次优间隔、独占赢家数、原选择保留率、search 损失增量及最差行。它与正式 v18 M2 使用相同模型版本、FP16、scalar10 网格和行级 direct-MSE 选择语义，但 v24 候选 Hessian/search 输入来自每 bank 16 篇文章，不是 v18 的 128 篇正式校准输入；因此这些代表行结果不替代完整 M2 预算损失。

## 成本与传播边界

候选成本按完整 Linear GPTQ candidate executions 计数。正式 M2 历史基线为 168×10=1,680 次；预算 3/4/5 的全局固定网格投影为 504/672/840 次。预算 3/4/5 的逐模块变长集合计数见 `COST_PROJECTION.csv`。这些是候选次数投影，不是实测时间；原 v18 实测 calibration wall 为 2,115.94 秒，不按候选次数线性换算。

若缩减子集改变任何模块的 alpha 选择，后续输入与 Hessian 可能随前缀变化；v24 缓存上的逐行损失不代表缩减网格的独立整模型运行。即使逐模块赢家并集在现有 M2 行结果上保持所有选择一致，也只说明既有输入下的选择一致性；没有缩减网格的完整提交/传播证据，不宣称模型张量或 PPL 完全相同。

## 执行边界

仅读取已有 search 产物；未读取 P3，未修改 v39 或历史产物；没有 GPU 初始化、模型前向、GPTQ、新 held 评分、PPL 或整模型量化。CPU 分析用时 {summary['analysis_cpu_seconds']:.3f} 秒（上限 300 秒）。历史数据上的候选子集设计是探索性分析，不称为新数据验证。
"""
(OUT / "DECISION.md").write_text(decision, encoding="utf-8")

summary["analysis_cpu_seconds"] = time.monotonic() - START
if summary["analysis_cpu_seconds"] > 295:
    raise RuntimeError(f"CPU analysis exceeded 295-second guard: {summary['analysis_cpu_seconds']:.3f}s")
(OUT / "RUN_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "status": summary["status"],
    "analysis_cpu_seconds": summary["analysis_cpu_seconds"],
    "m2_rows": len(m2_rows),
    "m2_linears": len(m2_by_module),
    "m2_winner_union_candidate_executions": module_union_execs,
    "m2_global_winners": alpha_set_string(m2_global_winner_set),
    "v24_sample_scores": len(sample_rows),
    "output_dir": str(OUT),
}, ensure_ascii=False, indent=2))
