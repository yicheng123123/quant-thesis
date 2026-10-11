from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "validation"
AUDIT = RUN / "audit"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    stage_rows = list(csv.DictReader((RUN / "FORMAL_TIMING.csv").open(encoding="utf-8", newline="")))
    grouped: dict[str, list[dict]] = {}
    for row in stage_rows:
        grouped.setdefault(row["stage"], []).append(row)
    timing_summary = []
    for stage, rows in sorted(grouped.items()):
        timing_summary.append({
            "stage": stage,
            "calls": len(rows),
            "wall_seconds_sum": sum(float(row["wall_seconds"]) for row in rows),
            "max_peak_allocated_bytes": max(int(row["gpu_peak_allocated_bytes"]) for row in rows),
            "max_peak_reserved_bytes": max(int(row["gpu_peak_reserved_bytes"]) for row in rows),
        })
    with (AUDIT / "STAGE_TIMING_SUMMARY.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(timing_summary[0]))
        writer.writeheader(); writer.writerows(timing_summary)

    by_stage = {row["stage"]: row for row in timing_summary}
    get = lambda name: by_stage.get(name, {}).get("wall_seconds_sum", 0.0)
    run_manifest = read_json(RUN / "RUN_MANIFEST.json")
    runtime = read_json(RUN / "FORMAL_RUNTIME.json")
    budget = read_json(ROOT / "BUDGET.json")
    consistency = read_json(AUDIT / "CONSISTENCY_SUMMARY.json")
    named = read_json(AUDIT / "NORMALIZED_TENSOR_HASHES.json")
    diff = read_json(AUDIT / "NUMERICAL_SELECTION_DIFFERENCE.json")["selection_differences"]
    c_rows = list(csv.DictReader((AUDIT / "C_SHARING_AUDIT.csv").open(encoding="utf-8", newline="")))
    module_consistency = list(csv.DictReader((AUDIT / "MODULE_CONSISTENCY.csv").open(encoding="utf-8", newline="")))
    alpha_histograms = {}
    for scorer, col in (("direct", "direct_selected_alpha_histogram"),
                        ("quadratic", "quadratic_selected_alpha_histogram")):
        hist = {f"{i/10:.1f}": 0 for i in range(1, 11)}
        for row in module_consistency:
            for alpha, count in json.loads(row[col]).items():
                hist[alpha] += int(count)
        alpha_histograms[scorer] = hist
    first_pass = float(read_json(RUN / "RECOVERY_AUDIT.json")["budget_start_seconds"])
    # The runner writes its budget ledger after the last timed CUDA operation;
    # FORMAL_RUNTIME is emitted after row-artifact finalization and therefore
    # captures the full conservative process wall through report preparation.
    cumulative = max(float(budget["gpu_wall_elapsed_seconds"]),
                     float(runtime["cumulative_v28_gpu_wall_seconds"]))
    stage_sum = sum(float(row["wall_seconds_sum"]) for row in timing_summary)
    direct = get("all_row_direct_output_scoring")
    quad = get("all_row_quadratic_output_scoring")
    c_build = get("score_C_build_once_per_dependency_group")
    score_saving = direct - (quad + c_build)
    row_diff = diff[0] if diff else None
    used_alloc = max(int(row["max_peak_allocated_bytes"]) for row in timing_summary)
    used_reserved = max(int(row["max_peak_reserved_bytes"]) for row in timing_summary)

    summary = {
        "version": "v28",
        "frozen_protocol_sha256": run_manifest.get("protocol_sha256"),
        "checkpoint_execution_bundle_sha256": read_json(RUN / "RECOVERY_AUDIT.json")["checkpoint_code_bundle_sha256"],
        "recovery_output_bundle_sha256": read_json(RUN / "M2" / "QUANTIZATION_METADATA.json")["code_bundle_sha256"],
        "direct_path_vs_v18": {
            "modules": consistency["modules"], "rows": consistency["rows"],
            "input_hashes_equal": consistency["modules_input_hash_exact_vs_v18"] == consistency["modules"],
            "H_hashes_equal": consistency["modules_hessian_hash_exact_vs_v18"] == consistency["modules"],
            "direct_scores_exact": consistency["modules_direct_scores_exact_vs_v18"] == consistency["modules"],
            "direct_alpha_and_selected_tensors_exact": consistency["modules_direct_selected_tensors_exact_vs_v18"] == consistency["modules"],
            "boundary_hashes_exact": consistency["boundary_hashes_exact"] == consistency["boundary_hashes_total"],
            "normalized_named_tensor_hash_exact": named["v18_named_tensor_hash"] == named["v28_direct_named_tensor_hash"],
            "named_tensor_count": named["v18_tensor_count"],
        },
        "quadratic_scorer_substitution": {
            "status": "FAIL_SELECTION_MISMATCH" if consistency["selection_difference_rows"] else "PASS",
            "selection_difference_rows": consistency["selection_difference_rows"],
            "sensitive_rows": consistency["sensitive_rows"],
            "modules_with_exact_selected_W_scale_zero_g_idx": consistency["modules_quadratic_selected_tensors_exact_vs_direct"],
            "difference_detail": row_diff,
            "quadratic_only_full_m2": "NOT_RUN_GATED_BY_SELECTION_MISMATCH",
            "new_ppl": "NOT_RUN",
        },
        "work_units": {
            "linear_modules": 168,
            "output_rows": 304128,
            "candidate_gptq_executions": runtime["formal_work_units"]["m2_full_linear_alpha_candidate_executions"],
            "native_alpha1_parity_executions": runtime["formal_work_units"]["m2_auxiliary_alpha1_native_parity_references"],
            "row_alpha_scores_per_scorer": runtime["formal_work_units"]["m2_row_alpha_score_slots"],
            "row_alpha_scores_both_scorers": 2 * runtime["formal_work_units"]["m2_row_alpha_score_slots"],
            "C_builds": len(c_rows),
            "C_shared_dependency_groups_valid": sum(r["one_shared_C_for_exact_group"].lower() == "true" for r in c_rows),
        },
        "selection_summary": {
            "direct_exact_tie_rows": sum(int(r["direct_exact_tie_rows"]) for r in module_consistency),
            "quadratic_exact_tie_rows": sum(int(r["quadratic_exact_tie_rows"]) for r in module_consistency),
            "direct_alpha_histogram": alpha_histograms["direct"],
            "quadratic_alpha_histogram": alpha_histograms["quadratic"],
        },
        "timing_seconds": {
            "first_dual_formal_pass_wall": first_pass,
            "recovery_finalization_wall": cumulative - first_pass,
            "cumulative_v28_wall_including_recovery": cumulative,
            "dual_validation_stage_timing_sum_noninclusive": stage_sum,
            "unattributed_wall_difference": cumulative - stage_sum,
            "direct_score_all_rows_all_candidates": direct,
            "quadratic_score_all_rows_all_candidates": quad,
            "C_build_all_dependency_groups": c_build,
            "quadratic_plus_C_scoring_stage_total": quad + c_build,
            "direct_minus_quadratic_plus_C_scoring_stage_difference": score_saving,
            "score_stage_fraction_saved": score_saving / direct if direct else None,
            "candidate_full_linear_gptq": get("full_linear_alpha_candidate_gptq"),
            "native_H_build": get("unique_group_hessian_build"),
            "native_Hinv_preparation": get("shared_native_hinv_prepare"),
            "activation_capture_and_dependency_refresh": sum(get(s) for s in (
                "initial_embedding_capture", "qkv_shared_input_capture", "qkv_commit_to_o_input",
                "o_commit_to_up_gate_input", "up_gate_commit_to_down_input", "down_commit_to_next_layer_input")),
            "argmin_update": get("streaming_per_row_argmin_update"),
            "stitch_and_commit": get("full_linear_row_stitch") + get("whole_linear_commit"),
            "checkpoint_IO": get("atomic_dependency_group_checkpoint_io") + get("atomic_layer_checkpoint_io"),
            "alpha1_native_parity": get("alpha1_native_parity_reference"),
            "direct_difference_recheck": get("direct_difference_row_recheck"),
            "model_load_and_reload": get("model_load"),
            "final_dense_model_save": get("final_dense_model_save"),
            "quadratic_only_end_to_end_wall": "NOT_RUN",
            "end_to_end_calibration_speedup": "NOT_MEASURED",
        },
        "peak_gpu_memory": {"allocated_bytes": used_alloc, "allocated_gib": used_alloc / 2**30,
                             "reserved_bytes": used_reserved, "reserved_gib": used_reserved / 2**30},
        "budget": {"hard_cap_seconds": float(budget["max_seconds"]),
                   "actual_cumulative_wall_seconds": cumulative,
                   "remaining_seconds": float(budget["max_seconds"]) - cumulative,
                   "timer_semantics": "perf_counter process wall, including CPU and GPU portions; conservative for the GPU budget",
                   "ledger_note": "FINAL_SUMMARY uses FORMAL_RUNTIME after post-scoring artifact finalization; BUDGET was last persisted before the final CPU artifact pass"},
    }
    if cumulative > float(budget["gpu_wall_elapsed_seconds"]):
        budget["gpu_wall_elapsed_seconds"] = cumulative
        budget["last_recorded_stage"] = "finalized_runtime_including_post_scoring_artifacts"
        (ROOT / "BUDGET.json").write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")
    (AUDIT / "FINAL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    attempt_files = sorted(ROOT.glob("ATTEMPT_FAILURE_*.json"))
    execution_log = {
        "pre_run_source_protocol_freeze": "CPU-only repository/source/hash audit; no GPU work; duration not separately instrumented",
        "setup_failures": [read_json(p) for p in attempt_files],
        "standalone_warmup_stage": "NOT_RUN; first model/candidate use is included in the formal wall/stage counters",
        "native_alpha1_parity_references": get("alpha1_native_parity_reference"),
        "first_full_dual_pass_wall_seconds": first_pass,
        "first_pass_failure": read_json(RUN / "RECOVERY_AUDIT.json")["reason"],
        "checkpoint_recovery_finalization_seconds": cumulative - first_pass,
        "checkpoint_recovery_candidate_reruns": 0,
        "post_run_numeric_analysis": "CPU-only; outside the GPU execution budget",
        "cumulative_process_wall_seconds": cumulative,
        "hard_budget_seconds": budget["max_seconds"],
    }
    (RUN / "EXECUTION_LOG.json").write_text(json.dumps(execution_log, indent=2) + "\n", encoding="utf-8")

    r = row_diff
    lines = [
        "# v28 — Full M2 Quadratic Scoring Audit",
        "",
        "## Outcome",
        "",
        "The direct-score validation completed for all 168 Linear modules. The quadratic scorer did **not** pass the frozen selection-equivalence gate: one output row selects a different alpha. The conditional quadratic-only M2 run and PPL evaluation were therefore not run.",
        "",
        "## Frozen scope and formula",
        "",
        "- Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, FP16, RTX 4060 Laptop GPU, SDPA, TF32 disabled.",
        "- Same v18 M2: 168 Linear modules, W4A16, symmetric dynamic GPTQ G128, block 128, damping 0.1, original error compensation, 10 alpha values 0.1–1.0, 128 WikiText-2 train documents × 2048 tokens.",
        f"- Score target remains the unweighted local Linear output MSE over 262,144 tokens. {consistency['modules_with_bias']} modules have bias and each stored bias hash remained unchanged; {consistency['modules_without_bias']} have no bias. Any unchanged common bias cancels in the output difference. The frozen path has no padding mask or sample weighting. Inputs are the actual M2 prefix activations.",
        "- Quadratic scorer: `C = X.T @ X / N` with every entry retained, `e = Q.float() - W.float()`, score `e.T @ C @ e`. C was built from each exact dependency-group input, not GPTQ's damped/dead-column H or inverse.",
        "- Frozen alpha tie handling: strict minimum; exact ties choose the larger alpha. No tolerance or fallback changed the selection.",
        "",
        "## Full-model consistency",
        "",
        f"- Direct v28 path vs v18: input hashes {consistency['modules_input_hash_exact_vs_v18']}/168; H hashes {consistency['modules_hessian_hash_exact_vs_v18']}/168; direct candidate score arrays exact {consistency['modules_direct_scores_exact_vs_v18']}/168; direct selected alpha/W/scale/zero/g_idx exact {consistency['modules_direct_selected_tensors_exact_vs_v18']}/168; activation boundary hashes exact {consistency['boundary_hashes_exact']}/{consistency['boundary_hashes_total'] }.",
        f"- Normalized named tensor digest: `{named['v28_direct_named_tensor_hash']}`; all {named['v28_direct_tensor_count']} named tensors match v18 exactly. `config.json`, model revision and quantization format also match. Existing v18 M2 quality results refer to this same direct-selected model; no new PPL result or quality gain is claimed.",
        f"- Dual scorer: {consistency['row_alpha_scores']} row-alpha pairs per scorer across all 10 candidates. Score max absolute error `{consistency['max_abs_score_error']:.9g}`, max relative error `{consistency['max_relative_score_error']:.9g}`, pooled median absolute error `{consistency['median_abs_score_error_all_row_alpha_pairs']:.9g}`. The diagnostic bound `2 × max absolute score error >= direct best/second gap` flags {consistency['sensitive_rows']} rows; only {consistency['selection_difference_rows']} row changes selection. This bound is descriptive and never changes selection.",
        f"- Whole-model selected-alpha counts (direct vs quadratic) are in `FINAL_SUMMARY.json`; exact ties were {summary['selection_summary']['direct_exact_tie_rows']} and {summary['selection_summary']['quadratic_exact_tie_rows']} rows, respectively.",
        f"- Alpha 0.8/0.9 counts shift by exactly the changed row: direct `{alpha_histograms['direct']['0.8']:,}` / `{alpha_histograms['direct']['0.9']:,}`, quadratic `{alpha_histograms['quadratic']['0.8']:,}` / `{alpha_histograms['quadratic']['0.9']:,}`. Other alpha counts are unchanged.",
        "",
        "### Selection-changing row",
        "",
    ]
    if r:
        lines += [
            f"`model.layers.{r['layer']}.{r['module']}`, output row {r['row']}: direct score selects alpha **{r['direct_order_best_to_worst'][0]:.1f}**, quadratic score selects **{r['quadratic_order_best_to_worst'][0]:.1f}**.",
            f"- Direct best/second gap: `{r['direct_best_second_gap']:.12g}`; quadratic best/second gap: `{r['quadratic_best_second_gap']:.12g}`.",
            f"- Max absolute score error for this row: `{r['max_abs_score_error']:.12g}`. Full direct scoring ranks alpha 0.8 ahead of 0.9 by `{r['direct_best_second_gap']:.12g}`; quadratic summation reverses them.",
            f"- Selected W differs in {r['selected_weight_differing_elements']}/{896} elements (max absolute FP16 difference `{r['selected_weight_max_abs_difference']:.8g}`); scale differs; zero and G128 `g_idx` match.",
            "- Same-run direct recheck rescored the retained alpha 0.8 and 0.9 candidate rows. Subset GEMM differed from full-row direct scores by at most `4.03e-12` and `1.39e-12`, respectively, below the full-row direct winner gap; it was not bitwise-identical, so the original full-row direct result remains the reference. The recheck did not alter alpha selection.",
            "- Therefore the numerical formula is mathematically equivalent, but the implemented FP32 Gram/matmul accumulation is not selection-equivalent at this near tie. The pure quadratic scorer cannot replace direct scoring under the frozen exact-choice rule.",
        ]
    lines += [
        "",
        "The full arrays and all flagged rows are preserved in `audit/raw_scores/layer_XX_scores.npz`, `SENSITIVE_ROWS.csv`, and `SELECTION_DIFFERENCES.csv`; the expanded affected-row values and W/scale/zero/g_idx evidence are in `NUMERICAL_SELECTION_DIFFERENCE.json`.",
        "",
        "## Sharing and costs",
        "",
        f"- {summary['work_units']['candidate_gptq_executions']} full-Linear alpha GPTQ executions generated both scorers' candidate scores once each; {summary['work_units']['native_alpha1_parity_executions']} additional alpha=1 native parity references were run. Per scorer, there were {summary['work_units']['row_alpha_scores_per_scorer']:,} row-alpha slots.",
        f"- C was built {summary['work_units']['C_builds']} times, once for each of 96 dependency groups (q/k/v; o; up/gate; down in each layer). All {summary['work_units']['C_shared_dependency_groups_valid']} group hashes confirm exact input/H/C sharing only within the group.",
        f"- Scoring-only measurement over all rows/candidates: direct scorer `{direct:.3f}s`; C construction plus quadratic scorer `{c_build + quad:.3f}s` (`C={c_build:.3f}s`, quadratic scoring `{quad:.3f}s`). Operator-stage difference is `{score_saving:.3f}s` ({summary['timing_seconds']['score_stage_fraction_saved']*100:.2f}%). This is a scoring-stage comparison inside the dual-validation run, not a full calibration speedup.",
        f"- Candidate GPTQ `{get('full_linear_alpha_candidate_gptq'):.3f}s`; H preparation `{get('unique_group_hessian_build') + get('shared_native_hinv_prepare'):.3f}s`; activation capture/refresh `{summary['timing_seconds']['activation_capture_and_dependency_refresh']:.3f}s`; alpha=1 parity `{get('alpha1_native_parity_reference'):.3f}s`; checkpoint I/O `{summary['timing_seconds']['checkpoint_IO']:.3f}s`.",
        f"- End-to-end dual-validation wall, including finalization recovery: `{cumulative:.3f}s` ({cumulative/60:.2f} min). Targeted stage timers sum to `{stage_sum:.3f}s`; `{cumulative-stage_sum:.3f}s` remains outside those stage counters, so stage sums are not substituted for end-to-end wall.",
        f"- Peak recorded GPU memory: `{used_alloc / 2**30:.3f} GiB` allocated, `{used_reserved / 2**30:.3f} GiB` reserved. Budget timer counts `perf_counter` process wall (CPU and GPU), conservatively; cumulative is `{cumulative:.3f}/{budget['max_seconds']:.0f}s`.",
        "- Three setup retries stopped before model load and recorded zero GPU time/candidates. There was no separate warm-up stage; model loading, first captures, alpha=1 parity references, and first candidate calls are timed. CPU-only source freeze and post-run analysis are identified separately in `EXECUTION_LOG.json`.",
        "",
        "## Gate and unrun items",
        "",
        "- Conditional quadratic-only M2: **NOT_RUN** because one row changed the selected alpha and committed W/scale. It was not hidden by a fallback.",
        "- New WikiText-2/C4 PPL: **NOT_RUN**. The saved direct-selected validation model is tensor-identical to v18 M2, so existing v18 PPL can be attributed to that direct path. No PPL is attributed to an unrun quadratic-only model.",
        "- End-to-end calibration acceleration for an exclusively quadratic scorer: **NOT_MEASURED**. No offline scoring-stage saving is described as inference acceleration.",
        "- First full pass reached all 24 layers and 1,680 candidate executions, then final report writing failed because `validation/M2` had not been created. Finalization resumed from the immutable layer checkpoints without rerunning candidates; see `RECOVERY_AUDIT.json`. Three earlier setup attempts failed before model load and recorded zero GPU time/candidates.",
        "",
        "## Artifacts",
        "",
        "- `PROTOCOL.json` — frozen source/protocol hashes, formula, gates, and scope.",
        "- `validation/FORMAL_TIMING.csv` and `validation/audit/STAGE_TIMING_SUMMARY.csv` — raw and aggregated phase timings.",
        "- `validation/audit/MODULE_CONSISTENCY.csv`, `C_SHARING_AUDIT.csv`, `BOUNDARY_ACTIVATION_COMPARISON.csv` — full module/group evidence.",
        "- `validation/audit/raw_scores/`, `SENSITIVE_ROWS.csv`, `SELECTION_DIFFERENCES.csv`, `NUMERICAL_SELECTION_DIFFERENCE.json` — scores and row-level diagnostics.",
        "- `validation/audit/NORMALIZED_TENSOR_HASHES.json`, `FINAL_SUMMARY.json` — normalized final tensor/config comparison and run summary.",
        "- `run_v28.py`, `analyze_v28.py`, `recovery_finalize_v28.py`, `finalize_v28_report.py` — runner and analysis tools.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reproduce = """# v28 Reproduction\n\nRun from `E:/quant-thesis` in PowerShell with the existing `.venv-gptq` environment and local model/cache present.\n\n```powershell\n$env:V28_RUN_LABEL = 'validation'\n$env:V28_SCORE_MODE = 'dual'\n& .\\.venv-gptq\\Scripts\\python.exe experiments/range_search_w4/results/range_w4_quadratic_full_m2_v28/run_v28.py\n& .\\.venv-gptq\\Scripts\\python.exe experiments/range_search_w4/results/range_w4_quadratic_full_m2_v28/analyze_v28.py\n& .\\.venv-gptq\\Scripts\\python.exe experiments/range_search_w4/results/range_w4_quadratic_full_m2_v28/finalize_v28_report.py\n```\n\nThe completed artifacts include a post-run checkpoint finalization recovery. To reproduce that artifact from the already-generated validation checkpoints, use `recovery_finalize_v28.py`; it verifies the recorded historical checkpoint bundle plus every module/layer/alpha-chain hash, performs no GPTQ candidates, saves/reloads the direct-selected model, and preserves the cumulative budget ledger.\n\nThe conditional `quadratic_only` execution is intentionally not run because `CONSISTENCY_SUMMARY.json` records one selection-changing row. Do not bypass this gate to claim equivalence. PPL and downstream task evaluation were not run.\n"""
    (ROOT / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")
    print(json.dumps({"summary": str(AUDIT / "FINAL_SUMMARY.json"),
                      "report": str(ROOT / "REPORT.md"),
                      "cumulative_seconds": cumulative,
                      "status": summary["quadratic_scorer_substitution"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
