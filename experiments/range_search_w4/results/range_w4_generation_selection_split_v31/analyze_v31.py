"""Artifact-only analysis of the frozen v31 generation/selection split."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V24 = RESULTS / "range_w4_structured_group_e2_v24"
V30 = RESULTS / "range_w4_weight_error_regularization_v30"
PROTOCOL = json.loads((HERE / "FROZEN_PROTOCOL.json").read_text("utf-8"))
EPS = json.loads((V24 / "HELD_REPEAT_NOISE.json").read_text("utf-8"))["repeat_noise_mse_by_unit"]
ALPHAS = [float(x) for x in PROTOCOL["inheritance"]["v24_scalar_grid"]]
ROWS = {k: [int(x) for x in v] for k, v in PROTOCOL["inheritance"]["v24_fixed_rows"].items()}
GROUPS = PROTOCOL["documents"]["groups"]
BANKS = {
    "S0": {"original_selector": "A", "independent_selector": "B", "eval": ["C", "D"], "generation": list(range(16))},
    "S1": {"original_selector": "C", "independent_selector": "D", "eval": ["A", "B"], "generation": list(range(32, 48))},
}
TOKENS_PER_ROW_GROUP = 16 * 2048


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def choose(score_by_alpha: dict[float, float]) -> tuple[float, float]:
    # v24 rule: exact score ties resolve to the larger alpha.
    best = min(score_by_alpha.values())
    alpha = max(a for a, score in score_by_alpha.items() if score == best)
    return alpha, best


def get_unit_id(layer: int, projection: str, row: int, bank: str) -> str:
    return f"L{layer}_{projection}_R{row}_{bank}"


def summarize(comparisons: list[dict], group_keys: tuple[str, ...], label: str) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in comparisons:
        buckets[tuple(row[k] for k in group_keys)].append(row)
    out = []
    for key, items in sorted(buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        orig_sse = math.fsum(float(r["original_selected_mse"]) * TOKENS_PER_ROW_GROUP for r in items)
        indep_sse = math.fsum(float(r["independent_selected_mse"]) * TOKENS_PER_ROW_GROUP for r in items)
        alpha1_sse = math.fsum(float(r["alpha1_mse"]) * TOKENS_PER_ROW_GROUP for r in items)
        oracle_sse = math.fsum(float(r["oracle_mse_diagnostic_only"]) * TOKENS_PER_ROW_GROUP for r in items)
        gains = [float(r["relative_gain_independent_vs_original"]) for r in items]
        positive_savings = sorted((max(0.0, float(r["delta_sse_original_minus_independent"])) for r in items), reverse=True)
        positive_total = math.fsum(positive_savings)
        gross_harm = math.fsum(max(0.0, -float(r["delta_sse_original_minus_independent"])) for r in items)
        severe_vs_alpha1 = sum(float(r["relative_gain_independent_vs_alpha1"]) < -0.10 for r in items)
        severe_vs_original = sum(float(r["relative_gain_independent_vs_original"]) < -0.10 for r in items)
        result = {
            "summary_level": label,
            **{k: v for k, v in zip(group_keys, key)},
            "row_group_observations": len(items),
            "unique_row_units": len({(r["bank"], r["layer"], r["projection"], r["row"]) for r in items}),
            "original_selected_pooled_sse": orig_sse,
            "independent_selected_pooled_sse": indep_sse,
            "independent_improvement_vs_original_fraction": (orig_sse - indep_sse) / orig_sse if orig_sse else 0.0,
            "alpha1_pooled_sse": alpha1_sse,
            "original_improvement_vs_alpha1_fraction": (alpha1_sse - orig_sse) / alpha1_sse if alpha1_sse else 0.0,
            "independent_improvement_vs_alpha1_fraction": (alpha1_sse - indep_sse) / alpha1_sse if alpha1_sse else 0.0,
            "oracle_pooled_sse_diagnostic_only": oracle_sse,
            "original_oracle_regret_sse": orig_sse - oracle_sse,
            "independent_oracle_regret_sse": indep_sse - oracle_sse,
            "original_oracle_regret_fraction_of_original_sse": (orig_sse - oracle_sse) / orig_sse if orig_sse else 0.0,
            "independent_oracle_regret_fraction_of_independent_sse": (indep_sse - oracle_sse) / indep_sse if indep_sse else 0.0,
            "wins_vs_original": sum(r["outcome_vs_original"] == "win" for r in items),
            "ties_vs_original": sum(r["outcome_vs_original"] == "tie" for r in items),
            "losses_vs_original": sum(r["outcome_vs_original"] == "loss" for r in items),
            "independent_wins_vs_alpha1": sum(r["independent_vs_alpha1_outcome"] == "win" for r in items),
            "independent_ties_vs_alpha1": sum(r["independent_vs_alpha1_outcome"] == "tie" for r in items),
            "independent_losses_vs_alpha1": sum(r["independent_vs_alpha1_outcome"] == "loss" for r in items),
            "original_wins_vs_alpha1": sum(r["original_vs_alpha1_outcome"] == "win" for r in items),
            "original_ties_vs_alpha1": sum(r["original_vs_alpha1_outcome"] == "tie" for r in items),
            "original_losses_vs_alpha1": sum(r["original_vs_alpha1_outcome"] == "loss" for r in items),
            "median_relative_gain_vs_original": statistics.median(gains) if gains else 0.0,
            "worst_relative_gain_vs_original": min(gains) if gains else 0.0,
            "worst_unit_id": min(items, key=lambda r: float(r["relative_gain_independent_vs_original"]))["unit_id"] if items else "",
            "changed_alpha_observations": sum(r["alpha_changed"] for r in items),
            "severe_gt10pct_regressions_vs_original": severe_vs_original,
            "severe_gt10pct_regressions_vs_alpha1": severe_vs_alpha1,
            "gross_positive_savings_sse": positive_total,
            "gross_harm_sse": gross_harm,
            "top1_share_of_positive_savings": positive_savings[0] / positive_total if positive_total else 0.0,
            "top3_share_of_positive_savings": math.fsum(positive_savings[:3]) / positive_total if positive_total else 0.0,
            "most_negative_row_delta_sse": min(float(r["delta_sse_original_minus_independent"]) for r in items) if items else 0.0,
            "paired_observation_note": "Each candidate-bank row appears in two evaluation groups; pooled group observations are descriptive paired repeats, not independent trials.",
        }
        out.append(result)
    return out


def main() -> None:
    status = json.loads((HERE / "RUN_STATUS.json").read_text("utf-8"))
    budget = json.loads((HERE / "BUDGET.json").read_text("utf-8"))
    if status.get("status") != "COMPLETE" or status.get("score_calls_completed") != 240:
        raise RuntimeError("cross-group scores are incomplete; analysis is blocked")

    # v30 is an audited export of original v24 scalar10 scores and selected alphas.
    v30 = read_csv(V30 / "CANDIDATE_ROW_TABLE.csv")
    old_scores: dict[tuple[str, int, str, int, float], dict[str, float]] = {}
    original_expected: dict[tuple[str, int, str, int], float] = {}
    for r in v30:
        bank = r["split_id"]
        layer, projection, row = int(r["layer"]), r["projection"], int(r["row"])
        alpha = float(r["alpha"])
        old_scores[(bank, layer, projection, row, alpha)] = {
            "A" if bank == "S0" else "C": float(r["search_output_mse"]),
            "B" if bank == "S0" else "D": float(r["held_output_mse"]),
        }
        if r["is_original_search_selected"].lower() == "true":
            original_expected[(bank, layer, projection, row)] = alpha

    # Candidate sidecars are the original source for search scores and the frozen candidate bank metadata.
    candidate_scores: dict[tuple[str, int, str, int, float], dict[str, float]] = {}
    candidate_rows: dict[tuple[str, int, str, float], dict] = {}
    candidate_root = V24 / "raw/full_candidates"
    candidate_sidecars = list(candidate_root.glob("L*_*_S*_a*.json"))
    for p in candidate_sidecars:
        m = json.loads(p.read_text("utf-8"))
        a = float(m["alpha"])
        if not any(math.isclose(a, g, rel_tol=0, abs_tol=1e-12) for g in ALPHAS):
            continue
        stem = p.stem
        parts = stem.split("_")
        layer, projection, bank = int(parts[0][1:]), f"{parts[1]}_{parts[2]}", parts[3]
        qpt = p.with_suffix(".pt")
        candidate_rows[(bank, layer, projection, a)] = {
            "sidecar": p, "pt": qpt, "metadata": m,
            "candidate_file_sha256": sha(qpt),
        }
        for row in ROWS[projection]:
            candidate_scores[(bank, layer, projection, row, a)] = {
                "A" if bank == "S0" else "C": float(m["search_mse_by_row"][str(row)])
            }

    new_scores: dict[tuple[str, str, int, str, int, float], float] = {}
    new_score_records = []
    input_hashes = {(int(r["layer"]), r["projection"]): r["current_sha256"]
                    for r in read_csv(HERE / "ACTIVATION_CACHE_AUDIT.csv")}
    frozen_protocol_sha = sha(HERE / "FROZEN_PROTOCOL.json")
    for bank in ("S0", "S1"):
        for group in BANKS[bank]["eval"]:
            for f in (HERE / "raw/cross_group_scores" / bank / group).glob("L*_proj_a*.json"):
                rec = json.loads(f.read_text("utf-8"))
                if rec["bank"] != bank or rec["evaluation_group"] != group:
                    raise RuntimeError(f"cross-group metadata mismatch: {f}")
                layer, projection, alpha = int(rec["layer"]), rec["projection"], float(rec["alpha"])
                target_meta = candidate_rows[(bank, layer, projection, alpha)]
                if rec["protocol_sha256"] != frozen_protocol_sha:
                    raise RuntimeError(f"frozen protocol hash mismatch in score record: {f}")
                if rec["candidate_weight_sha256"] != target_meta["metadata"]["weight_sha256"]:
                    raise RuntimeError(f"candidate tensor mismatch in score record: {f}")
                if rec["candidate_hessian_sha256"] != target_meta["metadata"]["hessian_sha256"]:
                    raise RuntimeError(f"candidate Hessian lineage mismatch in score record: {f}")
                if rec["candidate_file_sha256"] != target_meta["candidate_file_sha256"]:
                    raise RuntimeError(f"candidate file mismatch in score record: {f}")
                if rec["candidate_generation_documents"] != BANKS[bank]["generation"]:
                    raise RuntimeError(f"candidate-generation document lineage mismatch: {f}")
                if rec["evaluation_doc_indices"] != GROUPS[group]:
                    raise RuntimeError(f"evaluation document lineage mismatch: {f}")
                if rec["input_cache_sha256"] != input_hashes[(layer, projection)]:
                    raise RuntimeError(f"input cache hash mismatch in score record: {f}")
                stored_rows = [int(x) for x in rec["rows"]]
                if stored_rows != ROWS[projection] or set(map(int, rec["score_mse_by_row"])) != set(range(len(stored_rows))):
                    raise RuntimeError(f"scored row-position mapping is invalid in {f}")
                for row_position, score in rec["score_mse_by_row"].items():
                    row = stored_rows[int(row_position)]
                    if row not in ROWS[projection] or not math.isfinite(float(score)):
                        raise RuntimeError(f"invalid row score in {f}")
                    key = (bank, group, layer, projection, row, alpha)
                    if key in new_scores:
                        raise RuntimeError(f"duplicate cross score: {key}")
                    new_scores[key] = float(score)
                new_score_records.append(rec)
    if len(new_scores) != 960 or len(new_score_records) != 240:
        raise RuntimeError(f"expected 960 new scores/240 score records, got {len(new_scores)}/{len(new_score_records)}")

    # Build a complete long table of all 10 candidates on all four groups, per bank/target/row.
    long_rows = []
    row_keys = []
    for bank in ("S0", "S1"):
        groups_for_bank = ["A", "B", "C", "D"]
        for layer in (0, 12, 23):
            for projection in ("up_proj", "down_proj"):
                for row in ROWS[projection]:
                    unit = get_unit_id(layer, projection, row, bank)
                    for alpha in ALPHAS:
                        sidecar_score = candidate_scores[(bank, layer, projection, row, alpha)]
                        old = old_scores[(bank, layer, projection, row, alpha)]
                        if old["A" if bank == "S0" else "C"] != sidecar_score["A" if bank == "S0" else "C"]:
                            raise RuntimeError(f"search score source mismatch: {unit}/a={alpha}")
                        for group in groups_for_bank:
                            if group in BANKS[bank]["eval"]:
                                score = new_scores[(bank, group, layer, projection, row, alpha)]
                                source = "new_v31_direct_score"
                            else:
                                score = old[("A" if bank == "S0" else "C") if group in ("A", "C") else ("B" if bank == "S0" else "D")]
                                source = "v30_audited_v24_saved_score"
                            docs = GROUPS[group]
                            rec = {
                                "bank": bank, "generation_documents": ";".join(map(str, BANKS[bank]["generation"])),
                                "layer": layer, "projection": projection, "row": row, "unit_id": unit,
                                "evaluation_group": group, "evaluation_doc_indices": ";".join(map(str, docs)),
                                "alpha": alpha, "score_mse": score, "score_sse": score * len(docs) * 2048,
                                "score_source": source,
                                "candidate_weight_sha256": candidate_rows[(bank, layer, projection, alpha)]["metadata"]["weight_sha256"],
                                "candidate_hessian_sha256": candidate_rows[(bank, layer, projection, alpha)]["metadata"]["hessian_sha256"],
                                "candidate_file_sha256": candidate_rows[(bank, layer, projection, alpha)]["candidate_file_sha256"],
                            }
                            long_rows.append(rec)
                    row_keys.append((bank, layer, projection, row))
    if len(long_rows) != 1920:
        raise RuntimeError(f"expected 1920 complete row-alpha-group records; got {len(long_rows)}")
    write_csv(HERE / "CROSS_GROUP_ROW_ALPHA_SCORES.csv", long_rows)

    score_lookup = {(r["bank"], r["evaluation_group"], int(r["layer"]), r["projection"], int(r["row"]), float(r["alpha"])): float(r["score_mse"]) for r in long_rows}
    selection_rows = []
    comparisons = []
    all_checks = []
    for bank in ("S0", "S1"):
        original_group = BANKS[bank]["original_selector"]
        independent_group = BANKS[bank]["independent_selector"]
        for layer in (0, 12, 23):
            for projection in ("up_proj", "down_proj"):
                for row in ROWS[projection]:
                    unit = get_unit_id(layer, projection, row, bank)
                    orig_scores = {a: score_lookup[(bank, original_group, layer, projection, row, a)] for a in ALPHAS}
                    ind_scores = {a: score_lookup[(bank, independent_group, layer, projection, row, a)] for a in ALPHAS}
                    orig_alpha, orig_select_score = choose(orig_scores)
                    ind_alpha, ind_select_score = choose(ind_scores)
                    expected = original_expected[(bank, layer, projection, row)]
                    orig_reproduced = orig_alpha == expected
                    all_checks.append({"check": f"original_choice_{unit}", "passed": orig_reproduced,
                                       "observed_alpha": orig_alpha, "expected_alpha": expected})
                    if not orig_reproduced:
                        raise RuntimeError(f"original choice failed reproduction: {unit} {orig_alpha} != {expected}")
                    eps = float(EPS[unit])
                    per_eval = {}
                    for group in BANKS[bank]["eval"]:
                        orig_eval = score_lookup[(bank, group, layer, projection, row, orig_alpha)]
                        ind_eval = score_lookup[(bank, group, layer, projection, row, ind_alpha)]
                        alpha1 = score_lookup[(bank, group, layer, projection, row, 1.0)]
                        oracle_alpha, oracle = choose({a: score_lookup[(bank, group, layer, projection, row, a)] for a in ALPHAS})
                        delta = orig_eval - ind_eval
                        gain = delta / orig_eval if orig_eval else 0.0
                        gain_vs_alpha1 = (alpha1 - ind_eval) / alpha1 if alpha1 else 0.0
                        gain_orig_vs_alpha1 = (alpha1 - orig_eval) / alpha1 if alpha1 else 0.0
                        outcome = "win" if delta > eps else "loss" if delta < -eps else "tie"
                        per_eval[group] = {"original": orig_eval, "independent": ind_eval, "alpha1": alpha1,
                                           "oracle": oracle, "oracle_alpha": oracle_alpha, "gain": gain,
                                           "gain_vs_alpha1": gain_vs_alpha1, "gain_orig_vs_alpha1": gain_orig_vs_alpha1,
                                           "outcome": outcome}
                        comparisons.append({
                            "bank": bank, "generation_group": "A" if bank == "S0" else "C",
                            "original_selector_group": original_group, "independent_selector_group": independent_group,
                            "evaluation_group": group, "layer": layer, "projection": projection, "row": row,
                            "unit_id": unit, "original_selected_alpha": orig_alpha, "independent_selected_alpha": ind_alpha,
                            "alpha_changed": orig_alpha != ind_alpha, "original_selector_mse": orig_select_score,
                            "independent_selector_mse": ind_select_score, "epsilon_mse": eps,
                            "original_selected_mse": orig_eval, "independent_selected_mse": ind_eval,
                            "delta_mse_original_minus_independent": delta,
                            "delta_sse_original_minus_independent": delta * TOKENS_PER_ROW_GROUP,
                            "relative_gain_independent_vs_original": gain,
                            "alpha1_mse": alpha1, "delta_mse_alpha1_minus_independent": alpha1 - ind_eval,
                            "relative_gain_independent_vs_alpha1": gain_vs_alpha1,
                            "relative_gain_original_vs_alpha1": gain_orig_vs_alpha1,
                            "independent_vs_alpha1_outcome": "win" if alpha1 - ind_eval > eps else "loss" if alpha1 - ind_eval < -eps else "tie",
                            "original_vs_alpha1_outcome": "win" if alpha1 - orig_eval > eps else "loss" if alpha1 - orig_eval < -eps else "tie",
                            "oracle_alpha_diagnostic_only": oracle_alpha, "oracle_mse_diagnostic_only": oracle,
                            "original_oracle_regret_mse": orig_eval - oracle,
                            "independent_oracle_regret_mse": ind_eval - oracle,
                            "original_oracle_regret_fraction": (orig_eval - oracle) / orig_eval if orig_eval else 0.0,
                            "independent_oracle_regret_fraction": (ind_eval - oracle) / ind_eval if ind_eval else 0.0,
                            "outcome_vs_original": outcome,
                            "severe_gt10pct_regression_vs_original": gain < -0.10,
                            "severe_gt10pct_regression_vs_alpha1": gain_vs_alpha1 < -0.10,
                        })
                    selection_rows.append({
                        "bank": bank, "layer": layer, "projection": projection, "row": row, "unit_id": unit,
                        "generation_articles": ";".join(map(str, BANKS[bank]["generation"])),
                        "original_selector_group": original_group, "independent_selector_group": independent_group,
                        "original_alpha": orig_alpha, "independent_alpha": ind_alpha, "alpha_changed": orig_alpha != ind_alpha,
                        "original_selector_mse": orig_select_score, "independent_selector_mse": ind_select_score,
                        "original_selector_margin_to_second_best": sorted(orig_scores.values())[1] - orig_select_score,
                        "independent_selector_margin_to_second_best": sorted(ind_scores.values())[1] - ind_select_score,
                        "original_alpha_ties": ";".join(str(a) for a, s in orig_scores.items() if s == orig_select_score),
                        "independent_alpha_ties": ";".join(str(a) for a, s in ind_scores.items() if s == ind_select_score),
                        **{f"{g}_original_eval_mse": per_eval[g]["original"] for g in BANKS[bank]["eval"]},
                        **{f"{g}_independent_eval_mse": per_eval[g]["independent"] for g in BANKS[bank]["eval"]},
                        **{f"{g}_alpha1_eval_mse": per_eval[g]["alpha1"] for g in BANKS[bank]["eval"]},
                        **{f"{g}_oracle_alpha_diagnostic_only": per_eval[g]["oracle_alpha"] for g in BANKS[bank]["eval"]},
                    })
    if len(comparisons) != 96 or len(selection_rows) != 48 or not all(x["passed"] for x in all_checks):
        raise RuntimeError("selection/comparison counts or original-choice reproduction failed")
    write_csv(HERE / "ROW_EVALUATION_COMPARISONS.csv", comparisons)
    write_csv(HERE / "SELECTIONS_BY_ROW.csv", selection_rows)
    write_json(HERE / "ANALYSIS_CHECKS.json", {"status": "PASS", "checks_total": len(all_checks), "checks_passed": len(all_checks),
                                                 "checks": all_checks, "cross_score_records": len(new_score_records),
                                                 "new_row_alpha_scores": len(new_scores), "complete_long_score_rows": len(long_rows),
                                                 "original_selected_choices_reproduced": len(selection_rows),
                                                 "comparisons": len(comparisons), "candidate_GPTQ_reruns": 0,
                                                 "forward_captures_or_model_forwards": 0})

    target_summary = summarize(comparisons, ("bank", "evaluation_group", "layer", "projection"), "bank_eval_target")
    target_pooled_summary = summarize(comparisons, ("bank", "layer", "projection"), "bank_target_pooled_two_eval_groups")
    bank_group_summary = summarize(comparisons, ("bank", "evaluation_group"), "bank_eval")
    bank_summary = summarize(comparisons, ("bank",), "bank_pooled_two_eval_groups")
    overall_summary = summarize(comparisons, tuple(), "all_banks_and_eval_groups_descriptive")
    write_csv(HERE / "TARGET_SUMMARY.csv", target_summary)
    write_csv(HERE / "TARGET_POOLED_SUMMARY.csv", target_pooled_summary)
    write_csv(HERE / "BANK_GROUP_SUMMARY.csv", bank_group_summary)
    write_csv(HERE / "BANK_SUMMARY.csv", bank_summary + overall_summary)

    alpha_counts = Counter()
    for row in selection_rows:
        for scope in ((row["bank"], "ALL", "ALL"), (row["bank"], str(row["layer"]), row["projection"])):
            alpha_counts[(*scope, "original_selector", row["original_alpha"])] += 1
            alpha_counts[(*scope, "independent_selector", row["independent_alpha"])] += 1
    alpha_distribution = []
    for bank in ("S0", "S1"):
        scopes = [("ALL", "ALL", 24)] + [(str(layer), projection, 4) for layer in (0, 12, 23) for projection in ("up_proj", "down_proj")]
        for layer, projection, denominator in scopes:
            for method in ("original_selector", "independent_selector"):
                for alpha in ALPHAS:
                    count = alpha_counts[(bank, layer, projection, method, alpha)]
                    alpha_distribution.append({"bank": bank, "layer": layer, "projection": projection,
                                               "selector": method, "alpha": alpha, "row_count": count,
                                               "fraction_of_scope_rows": count / denominator})
    write_csv(HERE / "ALPHA_DISTRIBUTION.csv", alpha_distribution)

    timing = read_csv(HERE / "FORMAL_TIMING.csv")
    if len(timing) != 240:
        raise RuntimeError(f"expected 240 unique timing records, got {len(timing)}")
    score_records = new_score_records
    score_seconds = math.fsum(float(r["scoring_wall_seconds"]) for r in score_records)
    candidate_load_seconds = math.fsum(float(r["candidate_load_seconds"]) for r in score_records)
    warmup = float(budget.get("warmup_seconds_this_process", 0.0))
    total_wall = float(budget["gpu_wall_elapsed_seconds"])
    cost = {
        "status": "COMPLETE",
        "new_gpu_wall_elapsed_seconds_including_cuda_init_warmup_candidate_reads_scoring_checkpoint_writes": total_wall,
        "budget_limit_seconds": float(budget["max_seconds"]),
        "budget_fraction_used": total_wall / float(budget["max_seconds"]),
        "warmup_seconds": warmup,
        "candidate_load_seconds_sum": candidate_load_seconds,
        "direct_scoring_seconds_sum": score_seconds,
        "unallocated_wall_seconds_including_cuda_init_hashes_metadata_io_checkpoint_io_and_loop_overhead": total_wall - warmup - candidate_load_seconds - score_seconds,
        "score_calls_full_candidate_per_target_alpha_group": len(score_records),
        "row_alpha_scores": len(new_scores),
        "candidate_GPTQ_executions": 0,
        "new_model_forwards": 0,
        "mean_score_seconds_per_call": score_seconds / len(score_records),
        "median_score_seconds_per_call": statistics.median(float(r["scoring_wall_seconds"]) for r in score_records),
        "min_score_seconds_per_call": min(float(r["scoring_wall_seconds"]) for r in score_records),
        "max_score_seconds_per_call": max(float(r["scoring_wall_seconds"]) for r in score_records),
        "peak_allocated_bytes": int(budget["peak_allocated_bytes"]),
        "peak_reserved_bytes": int(budget["peak_reserved_bytes"]),
        "score_process_retries": 0,
        "same_frozen_direct_scorer": True,
        "score_wall_is_not_full_calibration_wall": True,
    }
    write_json(HERE / "COST_SUMMARY.json", cost)

    # Summarize alpha movement and full-pair aggregation without treating the two eval groups as independent samples.
    alpha_movement = {}
    for bank in ("S0", "S1"):
        rows = [r for r in selection_rows if r["bank"] == bank]
        changed = [r for r in rows if r["alpha_changed"]]
        alpha_movement[bank] = {
            "unique_rows": len(rows), "changed_rows": len(changed), "changed_fraction": len(changed) / len(rows),
            "original_alpha_counts": dict(sorted(Counter(str(r["original_alpha"]) for r in rows).items())),
            "independent_alpha_counts": dict(sorted(Counter(str(r["independent_alpha"]) for r in rows).items())),
        }
    paired_rows = []
    for bank in ("S0", "S1"):
        by_id = defaultdict(list)
        for r in comparisons:
            if r["bank"] == bank:
                by_id[r["unit_id"]].append(r)
        for unit, entries in by_id.items():
            if len(entries) != 2:
                raise RuntimeError(f"paired eval count is not two for {unit}")
            paired_gain = statistics.mean(float(r["relative_gain_independent_vs_original"]) for r in entries)
            paired_rows.append({"bank": bank, "unit_id": unit, "paired_eval_groups": ";".join(r["evaluation_group"] for r in entries),
                                "mean_relative_gain_across_two_eval_groups": paired_gain,
                                "alpha_changed": entries[0]["alpha_changed"]})
    write_csv(HERE / "PAIRED_ROW_SUMMARY.csv", paired_rows)
    paired_summary = {}
    for bank in ("S0", "S1"):
        vals = [float(r["mean_relative_gain_across_two_eval_groups"]) for r in paired_rows if r["bank"] == bank]
        paired_summary[bank] = {
            "paired_row_units": len(vals), "median_mean_gain_across_two_eval_groups": statistics.median(vals),
            "mean_mean_gain_across_two_eval_groups": statistics.mean(vals),
            "positive_row_units": sum(x > 0 for x in vals), "negative_row_units": sum(x < 0 for x in vals),
            "zero_row_units": sum(x == 0 for x in vals),
            "note": "One paired summary per candidate-bank row, combining two evaluation groups; exploratory descriptive statistic only.",
        }

    main_report = {
        "status": "ANALYSIS_COMPLETE_EXPLORATORY",
        "protocol_sha256": sha(HERE / "FROZEN_PROTOCOL.json"),
        "data_relation_audit_sha256": sha(HERE / "DATA_RELATION_AUDIT.json"),
        "analysis_checks_sha256": sha(HERE / "ANALYSIS_CHECKS.json"),
        "candidate_generation": {"S0": "documents 0..15", "S1": "documents 32..47"},
        "prefix_v18_article_overlap_with_64_v24_documents": 0,
        "article_seen_exploratory_only": True,
        "new_candidate_generation": False,
        "new_gptq_candidate_executions": 0,
        "new_forward_captures": 0,
        "complete_candidate_row_alpha_group_records": len(long_rows),
        "new_vs_reused_score_record_counts": {
            "new_v31_direct_score": sum(r["score_source"] == "new_v31_direct_score" for r in long_rows),
            "v30_audited_v24_saved_score": sum(r["score_source"] == "v30_audited_v24_saved_score" for r in long_rows),
        },
        "new_ppl": False,
        "alpha_movement": alpha_movement,
        "paired_row_summary": paired_summary,
        "bank_group_summaries": bank_group_summary,
        "bank_pooled_summaries": bank_summary + overall_summary,
        "cost": cost,
        "caveat": "This retrospective data-reuse diagnostic is exploratory because all articles/results had been inspected. It does not establish blind generalization, multiple-comparison overfitting, or a theorem. Local output SSE is not PPL.",
    }
    write_json(HERE / "RUN_REPORT.json", main_report)
    make_report(main_report, target_summary, target_pooled_summary, bank_group_summary, bank_summary, overall_summary, paired_summary, cost)
    write_reproduce()
    write_output_manifest()
    print(json.dumps({"status": main_report["status"], "comparisons": len(comparisons), "scores": len(long_rows),
                      "cross_group_score_calls": len(score_records), "cost_seconds": total_wall,
                      "bank_summaries": bank_summary}, ensure_ascii=False, indent=2))


def fmt_pct(x: float) -> str:
    return f"{100*x:+.3f}%"


def make_report(main: dict, target_rows: list[dict], target_pooled_rows: list[dict], group_rows: list[dict], bank_rows: list[dict], overall_rows: list[dict], paired: dict, cost: dict) -> None:
    def cell(v): return f"{100*float(v):+.3f}%"
    lines = [
        "# v31 — 固定 GPTQ 候选的生成数据—选择数据分离诊断", "",
        "> 状态：完成的回顾性探索诊断。所有 64 篇文章及历史结果已被查看；这不是盲测，也不是正式泛化验证。", "",
        "## 数据关系与冻结定义", "",
        "- S0 候选的 GPTQ Hessian 由 WT2 train 完整文章 0–15 构建；原选择也在 A=0–15，独立选择在 B=16–31，评价在 C=32–47、D=48–63。",
        "- S1 候选 Hessian 由文章 32–47 构建；原选择在 C，独立选择在 D=48–63，评价在 A、B。",
        "- 每组 16 篇完整文章、每篇 2048 tokens；所有文章组不拆 token。v18 固定 M2 前缀的 128 篇校准文章与这 64 篇的文章哈希交集为 0。",
        "- 复用同一固定候选 bank、v24 直接输出误差、固定前缀输入和 scalar10 网格。GPTQ 的 W4A16 symmetric dynamic G128、block128、damp 0.1、desc_act=false 语义未变。没有重跑 GPTQ 或前向。",
        "- 原选择从 v24/v30 已存 A/C search 分数复现；独立选择使用已存 B/D 分数；四个新评价组只补 direct score，不重做候选。并列按精确分数相同后取较大 alpha。oracle 仅事后诊断。",
        "- 完整行×候选×组长表共 1,920 项：960 项来自本轮 direct score，960 项复用已审计的 v24/v30 A/C search 与 B/D held 分数；所有新增分数均校验候选、Hessian、协议及输入缓存哈希。",
        "- 原始 cross-score JSON 的 `score_mse_by_row` 键 0–3 表示顶层 `rows` 列表中的位置；`CROSS_GROUP_ROW_ALPHA_SCORES.csv` 已按该列表映射为实际输出行号，避免把位置编号当成模型行号。",
        "",
        "## 生成—选择数据分离结果", "",
        "改善定义：`(原选择 SSE - 独立选择 SSE) / 原选择 SSE`；正值表示独立选择更低。SSE 是预先固定的 4 个输出行上的 pooled SSE，逐行按 `MSE×32768` 汇总，不代表完整 Linear 的所有行。W/T/L 使用 v24 冻结的逐行重复 alpha=1 epsilon。",
        "同一 bank/row 在两个评价组重复出现，以下 pooled 描述统计按其共同评价范围汇总，不把两个评价组当成独立实验样本。",
        "", "### Bank × evaluation group", "",
        "| Bank | Eval | n row-group | Independent vs original pooled SSE | W/T/L vs original | W/T/L vs α=1 | Median row gain | Worst row gain | >10% worse vs original / α=1 | Top-3 share of positive savings | Original / independent regret to oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in group_rows:
        lines.append(f"| {r['bank']} | {r['evaluation_group']} | {r['row_group_observations']} | {cell(r['independent_improvement_vs_original_fraction'])} | {r['wins_vs_original']}/{r['ties_vs_original']}/{r['losses_vs_original']} | {r['independent_wins_vs_alpha1']}/{r['independent_ties_vs_alpha1']}/{r['independent_losses_vs_alpha1']} | {cell(r['median_relative_gain_vs_original'])} | {cell(r['worst_relative_gain_vs_original'])} ({r['worst_unit_id']}) | {r['severe_gt10pct_regressions_vs_original']} / {r['severe_gt10pct_regressions_vs_alpha1']} | {cell(r['top3_share_of_positive_savings'])} | {float(r['original_oracle_regret_fraction_of_original_sse']):.2%} / {float(r['independent_oracle_regret_fraction_of_independent_sse']):.2%} |")
    lines.extend(["", "### Bank pooled over both evaluation groups (descriptive paired repeats)", "",
                  "| Bank | Independent vs original pooled SSE | W/T/L | Median row-group gain | Worst row-group gain | >10% worse vs original / α=1 | Top-1 / top-3 positive-saving concentration | Original / independent oracle regret |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in bank_rows:
        if r["summary_level"] != "bank_pooled_two_eval_groups":
            continue
        lines.append(f"| {r['bank']} | {cell(r['independent_improvement_vs_original_fraction'])} | {r['wins_vs_original']}/{r['ties_vs_original']}/{r['losses_vs_original']} | {cell(r['median_relative_gain_vs_original'])} | {cell(r['worst_relative_gain_vs_original'])} ({r['worst_unit_id']}) | {r['severe_gt10pct_regressions_vs_original']} / {r['severe_gt10pct_regressions_vs_alpha1']} | {cell(r['top1_share_of_positive_savings'])} / {cell(r['top3_share_of_positive_savings'])} | {float(r['original_oracle_regret_fraction_of_original_sse']):.2%} / {float(r['independent_oracle_regret_fraction_of_independent_sse']):.2%} |")
    lines.extend(["", "### Target × bank（两评价组配对汇总）", "",
                  "| Bank | Target | Independent vs original pooled SSE | W/T/L | Median gain | Worst gain (row) | Severe >10% vs original / α=1 | Top-3 positive saving share | Oracle regret original / independent |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in target_pooled_rows:
        target = f"L{r['layer']} {r['projection']}"
        lines.append(f"| {r['bank']} | {target} | {cell(r['independent_improvement_vs_original_fraction'])} | {r['wins_vs_original']}/{r['ties_vs_original']}/{r['losses_vs_original']} | {cell(r['median_relative_gain_vs_original'])} | {cell(r['worst_relative_gain_vs_original'])} ({r['worst_unit_id']}) | {r['severe_gt10pct_regressions_vs_original']} / {r['severe_gt10pct_regressions_vs_alpha1']} | {cell(r['top3_share_of_positive_savings'])} | {float(r['original_oracle_regret_fraction_of_original_sse']):.2%} / {float(r['independent_oracle_regret_fraction_of_independent_sse']):.2%} |")
    o = overall_rows[0]
    lines.extend(["", f"两 bank × 两组的描述性总体 pooled 改善为 **{cell(o['independent_improvement_vs_original_fraction'])}**；W/T/L={o['wins_vs_original']}/{o['ties_vs_original']}/{o['losses_vs_original']}，最差行变化 {cell(o['worst_relative_gain_vs_original'])}（{o['worst_unit_id']}）。该总体值只是有限固定行和固定候选 bank 的汇总，不能作为独立样本统计推断。", "",
                  "### 对 α=1 原生候选", "",
                  "每个 target × bank × eval 的相对 α=1 指标见 `TARGET_SUMMARY.csv`；两评价组配对后的 target 汇总见 `TARGET_POOLED_SUMMARY.csv`；全部逐行结果、最差与负项见 `ROW_EVALUATION_COMPARISONS.csv`。完整 W/T/L 不通过 PPL 或量化误差代理推断。", "",
                  "### Alpha 分布与配对行变化", "",
                  "每个 bank 有 24 个预先固定行。原 selector 与 independent selector 的逐 alpha 计数见 `ALPHA_DISTRIBUTION.csv`；每行所选 alpha、选择 margin、两个评价组的误差与 oracle alpha 见 `SELECTIONS_BY_ROW.csv`。两个评价组按同一行配对的均值见 `PAIRED_ROW_SUMMARY.csv`。", "",
                  "| Bank | Changed rows | Change rate | Original selector alpha counts | Independent selector alpha counts | Paired row units positive / negative | Median paired-row mean gain |",
                  "|---|---:|---:|---|---|---:|---:|"])
    for b, r in paired.items():
        a = main["alpha_movement"][b]
        def alpha_counts(values):
            return ", ".join(f"{float(alpha):.1f}:{count}" for alpha, count in sorted(values.items(), key=lambda x: float(x[0])))
        lines.append(f"| {b} | {a['changed_rows']}/24 | {100*a['changed_fraction']:.1f}% | {alpha_counts(a['original_alpha_counts'])} | {alpha_counts(a['independent_alpha_counts'])} | {r['positive_row_units']} / {r['negative_row_units']} | {cell(r['median_mean_gain_across_two_eval_groups'])} |")
    lines.extend(["", "## 逐 target 结果", "", "每 target × bank × eval 的全部统计见 `TARGET_SUMMARY.csv`，不从 structured 模式推断 scalar 结果。低于 0 的提升表示退化；最差行和严重退化没有过滤。", "",
                  "## 成本与范围", "",
                  f"- 新 direct-score 调用：{cost['score_calls_full_candidate_per_target_alpha_group']}；row-alpha 分数：{cost['row_alpha_scores']}；GPTQ candidate executions=0，new model forwards=0。",
                  f"- CUDA init、warmup、候选读取、评分和检查点写入计入的总 wall：{cost['new_gpu_wall_elapsed_seconds_including_cuda_init_warmup_candidate_reads_scoring_checkpoint_writes']:.3f}s / 600s。",
                  f"- 实测评分 kernel 汇总 {cost['direct_scoring_seconds_sum']:.3f}s；候选读取汇总 {cost['candidate_load_seconds_sum']:.3f}s；其余未归因/初始化/校验/IO {cost['unallocated_wall_seconds_including_cuda_init_hashes_metadata_io_checkpoint_io_and_loop_overhead']:.3f}s。峰值 allocated/reserved={cost['peak_allocated_bytes']:,}/{cost['peak_reserved_bytes']:,} bytes。",
                  "- 此成本只适用于本次 240 个完整 Linear candidate × 16-document-group 的行级 direct scoring。它不是 GPTQ 校准成本、整模型成本或在线推理加速。", "",
                  "## 结论边界", "",
                  "本轮只检验：在已有固定 GPTQ 候选集合上，把选择分数从候选 Hessian/原 search 文章转移到独立文章，是否改变并改善有限固定行的后续文章输出 SSE。结果不能证明 multiple-comparison overfitting，也不能推出完整模型泛化、PPL 或定理。由于历史文章已被查看，后续如要作正式泛化主张，应事先冻结全新文章集合、选择和评价一次性流程及判据；本轮结果仅能作为预注册设计依据。", "",
                  "**质量评测/PPL：NOT_RUN。整模型量化：NOT_RUN。候选 GPTQ 重算：NOT_RUN（按冻结协议为 0 次）。**", "",
                  "复现与文件索引见 `REPRODUCE.md`。"])
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n", "utf-8")


def write_reproduce() -> None:
    text = """# v31 reproduction\n\nRun from repository root `E:/quant-thesis` in the existing `.venv-gptq` environment. This protocol performs no model forward and no GPTQ candidate generation.\n\n```powershell\n& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_generation_selection_split_v31/audit_v31.py'\n& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_generation_selection_split_v31/run_cross_group_scores_v31.py'\n& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_generation_selection_split_v31/analyze_v31.py'\n```\n\nThe score runner is resumable from per-candidate JSON checkpoints and enforces the frozen 600-second cumulative GPU wall budget. It reuses v24 candidate tensors and v23/v24 activation caches. Do not rerun `audit_v31.py` after scoring unless deliberately regenerating its frozen audit outputs. The score runner refuses any protocol status other than `FROZEN_BEFORE_CROSS_GROUP_SCORING`.\n\nKey raw artifacts: `CROSS_GROUP_ROW_ALPHA_SCORES.csv` (all 1,920 candidate-row-alpha-group values, saved and new), `ROW_EVALUATION_COMPARISONS.csv`, `SELECTIONS_BY_ROW.csv`, and `raw/cross_group_scores/`.\n"""
    (HERE / "REPRODUCE.md").write_text(text, "utf-8")


def write_output_manifest() -> None:
    rows = []
    for p in sorted(HERE.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.name == "OUTPUT_MANIFEST.csv" or p.name.endswith(".tmp"):
            continue
        rows.append({"relative_path": p.relative_to(HERE).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)})
    write_csv(HERE / "OUTPUT_MANIFEST.csv", rows)


if __name__ == "__main__":
    main()
