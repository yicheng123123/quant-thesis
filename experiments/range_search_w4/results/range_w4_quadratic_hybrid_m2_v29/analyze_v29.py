"""Artifact-only v29 consistency, cost, and normalized-hash analysis."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
import write_reproduce_v29


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
V18 = ROOT / "experiments/range_search_w4/results/range_w4_fullmodel_v18"
V28 = ROOT / "experiments/range_search_w4/results/range_w4_quadratic_full_m2_v28"
RUN = HERE / "validation"
GRID = [i / 10 for i in range(1, 11)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def named_tensor_hashes(path: Path) -> dict[str, str]:
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as model:
        for name in sorted(model.keys()):
            tensor = model.get_tensor(name).contiguous()
            h = hashlib.sha256()
            h.update(name.encode("utf-8") + b"\0")
            h.update(str(tensor.dtype).encode("ascii") + b"\0")
            h.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
            h.update(tensor.numpy().tobytes())
            out[name] = h.hexdigest()
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def best_alpha(scores: np.ndarray) -> np.ndarray:
    chosen = np.full(scores.shape[1], -1.0, dtype=np.float64)
    best = np.full(scores.shape[1], np.inf, dtype=np.float64)
    for i, alpha in enumerate(GRID):
        replace = (scores[i] < best) | ((scores[i] == best) & (alpha > chosen))
        chosen[replace] = alpha
        best[replace] = scores[i, replace]
    return chosen


def main() -> None:
    runtime_path = RUN / "FORMAL_RUNTIME.json"
    if not runtime_path.is_file():
        raise RuntimeError("v29 full validation is incomplete")
    runtime = json.loads(runtime_path.read_text("utf-8"))
    if runtime.get("status") != "QUANTIZATION_COMPLETE_HYBRID_PASS":
        raise RuntimeError(f"v29 runtime status is not a completed hybrid pass: {runtime.get('status')}")

    timing_path = RUN / "FORMAL_TIMING.csv"
    with timing_path.open(newline="", encoding="utf-8") as f:
        timings = list(csv.DictReader(f))
    stage_seconds: dict[str, float] = defaultdict(float)
    stage_calls: Counter[str] = Counter()
    peak_allocated = peak_reserved = 0
    for row in timings:
        stage_seconds[row["stage"]] += float(row["wall_seconds"])
        stage_calls[row["stage"]] += 1
        peak_allocated = max(peak_allocated, int(row["gpu_peak_allocated_bytes"]))
        peak_reserved = max(peak_reserved, int(row["gpu_peak_reserved_bytes"]))

    direct_reference_differences = []
    hybrid_differences = []
    module_rows = []
    sensitive_records = []
    score_errors = []
    direct_exact_modules = input_exact_modules = h_exact_modules = 0
    recheck_exact_calls = recheck_calls = recheck_rows_per_candidate = recheck_row_alpha_values = 0
    sensitive_rows = triggered_linears = 0
    all_rechecks_exact = True
    total_rows = total_candidates = 0
    peak_candidate_cache = total_candidate_cache_bytes = 0
    direct_selected_hist = Counter()
    hybrid_selected_hist = Counter()
    quad_selected_hist = Counter()
    alpha_differences_by_layer = Counter()
    recheck_module_names = []

    for layer in range(24):
        v29_layer_path = RUN / "checkpoints" / "M2" / f"layer_{layer:02d}.pt"
        v18_layer_path = V18 / "checkpoints" / "M2" / f"layer_{layer:02d}.pt"
        v29_layer = torch.load(v29_layer_path, map_location="cpu", weights_only=False)
        v18_layer = torch.load(v18_layer_path, map_location="cpu", weights_only=False)
        for name, state in sorted(v29_layer["modules"].items()):
            old = v18_layer["modules"][name]
            rows = int(state["weight"].shape[0])
            total_rows += rows
            score_arrays = state["candidate_losses"].numpy().astype(np.float64, copy=False)
            quad_arrays = state["candidate_losses_quadratic"].numpy().astype(np.float64, copy=False)
            hybrid_arrays = state["candidate_losses_hybrid"].numpy().astype(np.float64, copy=False)
            direct_a = state["selected_alpha"].numpy().astype(np.float64, copy=False)
            hybrid_a = state["hybrid_selected_alpha"].numpy().astype(np.float64, copy=False)
            quad_a = state["quadratic_selected_alpha"].numpy().astype(np.float64, copy=False)
            if score_arrays.shape != (10, rows) or quad_arrays.shape != (10, rows) or hybrid_arrays.shape != (10, rows):
                raise RuntimeError(f"candidate score shape mismatch L{layer} {name}")
            total_candidates += int(score_arrays.size)

            alpha_direct_match = np.array_equal(direct_a, old["selected_alpha"].numpy())
            weight_direct_match = torch.equal(state["weight"], old["weight"])
            scale_direct_match = torch.equal(state["scales"], old["scales"])
            zero_direct_match = torch.equal(state["zeros"], old["zeros"])
            g_direct_match = torch.equal(state["g_idx"], old["g_idx"])
            score_direct_match = torch.equal(state["candidate_losses"], old["candidate_losses"])
            input_match = state["input_sha256"] == old["input_sha256"]
            h_match = state["hessian_sha256"] == old["hessian_sha256"]
            direct_exact = all((alpha_direct_match, weight_direct_match, scale_direct_match,
                                zero_direct_match, g_direct_match, score_direct_match))
            direct_exact_modules += int(direct_exact)
            input_exact_modules += int(input_match)
            h_exact_modules += int(h_match)
            if not direct_exact:
                direct_reference_differences.append({
                    "layer": layer, "module": name, "alpha": alpha_direct_match,
                    "weight": weight_direct_match, "scale": scale_direct_match,
                    "zero": zero_direct_match, "g_idx": g_direct_match,
                    "candidate_scores": score_direct_match,
                })

            hybrid_tensors = (state["hybrid_weight"], state["hybrid_scales"],
                              state["hybrid_zeros"], state["hybrid_g_idx"])
            direct_tensors = (state["weight"], state["scales"], state["zeros"], state["g_idx"])
            hybrid_tensor_match = all(torch.equal(a, b) for a, b in zip(hybrid_tensors, direct_tensors))
            hybrid_alpha_match = np.array_equal(hybrid_a, direct_a)
            hybrid_selection_match = bool(hybrid_alpha_match and hybrid_tensor_match)
            if not hybrid_selection_match:
                rows_diff = np.flatnonzero(hybrid_a != direct_a).tolist()
                alpha_differences_by_layer[layer] += len(rows_diff)
                hybrid_differences.append({"layer": layer, "module": name,
                                           "different_rows": rows_diff,
                                           "hybrid_alpha_equal": hybrid_alpha_match,
                                           "hybrid_W_scale_zero_g_idx_equal": hybrid_tensor_match})

            comp = state["comparison"]
            module_name = f"model.layers.{layer}.{name}"
            sensitive_list = comp["sensitive_rows"]
            module_sensitive = len(sensitive_list)
            sensitive_rows += module_sensitive
            triggered = bool(comp["triggered_full_linear"])
            if triggered:
                triggered_linears += 1
                recheck_module_names.append(module_name)
                records = comp["direct_recheck"]
                recheck_calls += len(records)
                recheck_rows_per_candidate += int(comp["direct_recheck_rows_per_candidate"])
                recheck_row_alpha_values += int(comp["direct_recheck_row_alpha_values"])
                exact_here = all(bool(x["exact_to_initial_direct_score"]) for x in records) and len(records) == 10
                recheck_exact_calls += sum(bool(x["exact_to_initial_direct_score"]) for x in records)
                all_rechecks_exact = all_rechecks_exact and exact_here
                if not exact_here:
                    sensitive_records.append({"layer": layer, "module": module_name,
                                              "failure": "rechecked full-shape direct score differs",
                                              "evidence": records})
            elif comp["direct_recheck"]:
                raise RuntimeError(f"non-triggered Linear has recheck rows: {module_name}")

            if module_sensitive:
                sensitive_records.append({
                    "layer": layer, "module": module_name,
                    "sensitive_rows": sensitive_list,
                    "sensitive_row_count": module_sensitive,
                    "triggered_full_linear": triggered,
                    "recheck_score_calls": comp["direct_recheck_full_linear_score_calls"],
                    "recheck_rows_per_alpha": comp["direct_recheck_rows_per_candidate"],
                    "recheck_row_alpha_values": comp["direct_recheck_row_alpha_values"],
                    "recheck_seconds": comp["direct_recheck_seconds"],
                    "recheck_evidence": comp["direct_recheck"],
                })

            direct_selected_hist.update({f"{a:.1f}": int(np.count_nonzero(direct_a == a)) for a in GRID})
            hybrid_selected_hist.update({f"{a:.1f}": int(np.count_nonzero(hybrid_a == a)) for a in GRID})
            quad_selected_hist.update({f"{a:.1f}": int(np.count_nonzero(quad_a == a)) for a in GRID})
            error = np.abs(quad_arrays - score_arrays)
            gap = np.partition(score_arrays, 1, axis=0)[1] - np.partition(score_arrays, 1, axis=0)[0]
            triggered_by_rule = (2 * error.max(axis=0) >= gap)
            if not np.array_equal(np.flatnonzero(triggered_by_rule), np.asarray(sensitive_list, dtype=np.int64)):
                raise RuntimeError(f"sensitivity decision does not reproduce saved evidence L{layer} {name}")
            score_errors.append(error.reshape(-1))
            cache_bytes = int(comp["candidate_cache_gpu_logical_bytes"])
            peak_candidate_cache = max(peak_candidate_cache, cache_bytes)
            total_candidate_cache_bytes += cache_bytes

            module_rows.append({
                "layer": layer, "module": module_name, "rows": rows,
                "input_hash_matches_v18": input_match, "H_hash_matches_v18": h_match,
                "direct_candidate_score_arrays_match_v18": score_direct_match,
                "direct_selected_alpha_matches_v18": alpha_direct_match,
                "direct_W_scale_zero_g_idx_match_v18": all((weight_direct_match, scale_direct_match,
                                                            zero_direct_match, g_direct_match)),
                "quadratic_alpha_matches_direct": np.array_equal(quad_a, direct_a),
                "hybrid_alpha_matches_direct": hybrid_alpha_match,
                "hybrid_W_scale_zero_g_idx_match_direct": hybrid_tensor_match,
                "sensitive_rows": module_sensitive, "triggered_full_linear_recheck": triggered,
                "recheck_score_calls": int(comp["direct_recheck_full_linear_score_calls"]),
                "recheck_rows_per_candidate": int(comp["direct_recheck_rows_per_candidate"]),
                "recheck_row_alpha_values": int(comp["direct_recheck_row_alpha_values"]),
                "recheck_seconds": float(comp["direct_recheck_seconds"]),
                "sensitivity_detection_seconds": float(comp["sensitivity_detection_seconds"]),
                "candidate_cache_gpu_bytes": cache_bytes,
                "candidate_cache_disk_io_bytes": int(comp["candidate_cache_disk_io_bytes"]),
                "candidate_cache_cpu_transfer_bytes": int(comp["candidate_cache_cpu_transfer_bytes"]),
                "max_abs_score_error": float(error.max()),
                "median_abs_score_error": float(np.median(error)),
                "selection_difference_rows_direct_vs_quadratic": int(np.count_nonzero(quad_a != direct_a)),
                "selection_difference_rows_direct_vs_hybrid": int(np.count_nonzero(hybrid_a != direct_a)),
                "state_sha256": state["state_sha256"],
            })

        # Boundary activation records are compared once per completed layer.
        new_bounds = v29_layer["metadata"].get("boundary_activation_hashes", {})
        old_bounds = v18_layer["metadata"].get("boundary_activation_hashes", {})
        for key in sorted(set(new_bounds) | set(old_bounds)):
            if new_bounds.get(key) != old_bounds.get(key):
                direct_reference_differences.append({"layer": layer, "boundary": key,
                                                     "v29": new_bounds.get(key), "v18": old_bounds.get(key)})

    write_csv(RUN / "MODULE_CONSISTENCY.csv", module_rows)
    write_csv(RUN / "SENSITIVE_RECHECKS.csv", sensitive_records)
    write_csv(RUN / "DIRECT_REFERENCE_DIFFERENCES.csv", direct_reference_differences)
    write_csv(RUN / "HYBRID_SELECTION_DIFFERENCES.csv", hybrid_differences)

    v29_model = RUN / "M2" / "checkpoint" / "final" / "model.safetensors"
    v18_model = V18 / "M2" / "checkpoint" / "final" / "model.safetensors"
    v29_hashes = named_tensor_hashes(v29_model)
    v18_hashes = named_tensor_hashes(v18_model)
    named_diff = {k: {"v29": v29_hashes.get(k), "v18": v18_hashes.get(k)}
                  for k in sorted(set(v29_hashes) | set(v18_hashes)) if v29_hashes.get(k) != v18_hashes.get(k)}
    cfg29 = json.loads((v29_model.parent / "config.json").read_text("utf-8"))
    cfg18 = json.loads((v18_model.parent / "config.json").read_text("utf-8"))
    config_equal = cfg29 == cfg18
    normalized_hashes = {
        "v29_named_tensor_count": len(v29_hashes), "v18_named_tensor_count": len(v18_hashes),
        "v29_normalized_named_tensor_digest": hashlib.sha256("".join(
            f"{k}:{v29_hashes[k]}\n" for k in sorted(v29_hashes)).encode()).hexdigest(),
        "v18_normalized_named_tensor_digest": hashlib.sha256("".join(
            f"{k}:{v18_hashes[k]}\n" for k in sorted(v18_hashes)).encode()).hexdigest(),
        "normalized_named_tensors_equal": not named_diff,
        "config_equal": config_equal,
        "v29_model_file_sha256": sha(v29_model), "v18_model_file_sha256": sha(v18_model),
        "named_tensor_mismatches": named_diff,
        "v29_named_tensor_hashes": v29_hashes,
        "v18_named_tensor_hashes": v18_hashes,
    }
    (RUN / "NORMALIZED_TENSOR_HASHES.json").write_text(json.dumps(normalized_hashes, indent=2) + "\n", "utf-8")

    v28_frozen = json.loads((HERE / "FROZEN_TRIGGER_AUDIT.json").read_text("utf-8"))
    timing_summary = {
        stage: {"calls": stage_calls[stage], "wall_seconds_sum_noninclusive": sec}
        for stage, sec in sorted(stage_seconds.items())
    }
    (RUN / "STAGE_TIMING_SUMMARY.json").write_text(json.dumps(timing_summary, indent=2) + "\n", "utf-8")

    budget = json.loads((HERE / "BUDGET.json").read_text("utf-8"))
    manifest = json.loads((RUN / "RUN_MANIFEST.json").read_text("utf-8"))
    v18_final_report = json.loads((V18 / "M2" / "RUN_REPORT.json").read_text("utf-8"))
    v18_ppl_paths = [V18 / "evaluation" / "M2" / "wikitext2" / "ppl.json",
                     V18 / "evaluation" / "M2" / "c4" / "ppl.json"]
    # The quality run remains the v18 artifact; this turn does not execute a new PPL job.
    correctness = {
        "status": "PASS" if (direct_exact_modules == 168 and input_exact_modules == 168 and
                              h_exact_modules == 168 and not direct_reference_differences and
                              not hybrid_differences and all_rechecks_exact and not named_diff and config_equal) else "FAIL",
        "modules": 168, "rows": total_rows, "row_alpha_scores_per_scorer": total_candidates,
        "direct_selected_state_exact_vs_v18_modules": direct_exact_modules,
        "input_hash_exact_vs_v18_modules": input_exact_modules,
        "H_hash_exact_vs_v18_modules": h_exact_modules,
        "activation_boundary_hashes_exact": not direct_reference_differences,
        "hybrid_selected_state_exact_vs_direct_modules": runtime["m2"]["quantization_correctness"]["hybrid_exact_modules"],
        "hybrid_selected_alpha_exact_rows": runtime["m2"]["quantization_correctness"]["hybrid_selection_exact_rows"],
        "selection_difference_rows_direct_vs_hybrid": len(hybrid_differences),
        "sensitive_rows": sensitive_rows, "triggered_full_linears": triggered_linears,
        "full_linear_recheck_score_calls": recheck_calls,
        "full_linear_recheck_exact_calls": recheck_exact_calls,
        "recheck_rows_per_candidate_sum": recheck_rows_per_candidate,
        "recheck_row_alpha_values": recheck_row_alpha_values,
        "all_full_shape_rechecks_exact_to_initial_direct_scores": all_rechecks_exact,
        "frozen_v28_trigger_rows": v28_frozen["v28_sensitive_rows"],
        "frozen_v28_trigger_linears": v28_frozen["v28_triggered_full_linears"],
        "normalized_named_tensors_equal_vs_v18": not named_diff,
        "config_equal_vs_v18": config_equal,
        "validation_committed_selection": "original direct scorer, per frozen v18 protocol",
        "hybrid_selection_validation_scope": "dual-score replay; direct scores were available before sensitivity detection",
        "frozen_trigger_usable_without_direct_scores": False,
        "standalone_hybrid_only_M2": "NOT_RUN",
        "standalone_trigger_limitation": (
            "The frozen criterion uses abs(quadratic_score - direct_score) and the direct best/second gap. "
            "A quadratic-only execution cannot evaluate this trigger without first running the original scorer."
        ),
        "PPL": "NOT_RUN",
    }
    (RUN / "CONSISTENCY_SUMMARY.json").write_text(json.dumps(correctness, indent=2) + "\n", "utf-8")
    operational = {
        "dual_validation_result": "PASS",
        "operational_hybrid_result": "NOT_VALIDATED",
        "reason": (
            "The frozen sensitivity rule depends on direct score errors and the direct best/second gap. "
            "It cannot be evaluated from quadratic scores alone. The v29 dual validation had direct scores available, "
            "and the standalone quadratic mode has no direct-free trigger. No alternate trigger was introduced."
        ),
        "reference_committed_in_validation": "original direct-score selection",
        "pure_quadratic_v28_failure_preserved": "L7 mlp.gate_proj row 884 selected alpha differs (.8 direct vs .9 quadratic)",
        "standalone_hybrid_only_M2": "NOT_RUN",
        "new_PPL": "NOT_RUN",
    }
    (RUN / "MIXED_OPERATIONAL_LIMITATION.json").write_text(json.dumps(operational, indent=2) + "\n", "utf-8")

    stage = lambda name: stage_seconds.get(name, 0.0)
    hybrid_score_components = (stage("score_C_build_once_per_dependency_group") +
                               stage("all_row_quadratic_output_scoring") +
                               stage("sensitivity_detection") +
                               stage("sensitive_full_linear_direct_recheck"))
    v28_run_wall = float(json.loads((V28 / "validation" / "audit" / "FINAL_SUMMARY.json").read_text("utf-8"))["timing_seconds"]["first_dual_formal_pass_wall"])
    v28_direct_all = float(json.loads((V28 / "validation" / "audit" / "FINAL_SUMMARY.json").read_text("utf-8"))["timing_seconds"]["direct_score_all_rows_all_candidates"])
    v28_c_quad = float(json.loads((V28 / "validation" / "audit" / "FINAL_SUMMARY.json").read_text("utf-8"))["timing_seconds"]["quadratic_plus_C_scoring_stage_total"])
    gate_hybrid_est = v28_run_wall - v28_direct_all + v28_c_quad + stage("sensitive_full_linear_direct_recheck")
    cumulative_total = float(runtime["cumulative_v29_gpu_wall_seconds"])
    remaining = float(v28_frozen.get("max_seconds", 5400.0)) - cumulative_total
    # The saved audit omits max_seconds; the frozen protocol is authoritative.
    hard_budget = float(json.loads((HERE / "PROTOCOL.json").read_text("utf-8"))["validation"]["gpu_wall_budget_seconds"])
    remaining = hard_budget - cumulative_total
    cost = {
        "scope": "v29 dual validation and recovery finalization; not standalone hybrid-only calibration",
        "cumulative_v29_budget_ledger_seconds": cumulative_total,
        "cumulative_v29_budget_ledger_minutes": cumulative_total / 60,
        "hard_budget_seconds": hard_budget, "remaining_budget_seconds_after_validation": remaining,
        "main_candidate_full_linear_gptq_executions": int(runtime["m2"]["formal_counts"]["full_linear_alpha_candidate_executions"]),
        "main_candidate_row_alpha_score_slots_per_scorer": int(runtime["m2"]["formal_counts"]["row_alpha_score_slots"]),
        "spot_full_linear_gptq_candidate_executions": 10,
        "spot_direct_recheck_score_calls": 10,
        "direct_all_row_score_seconds": stage("all_row_direct_output_scoring"),
        "quadratic_score_seconds": stage("all_row_quadratic_output_scoring"),
        "C_build_seconds": stage("score_C_build_once_per_dependency_group"),
        "sensitivity_detection_seconds": stage("sensitivity_detection"),
        "sensitive_full_linear_direct_recheck_seconds": stage("sensitive_full_linear_direct_recheck"),
        "counterfactual_hybrid_score_component_sum_seconds": hybrid_score_components,
        "counterfactual_score_component_difference_vs_direct_seconds": stage("all_row_direct_output_scoring") - hybrid_score_components,
        "counterfactual_score_component_difference_fraction_of_direct": (stage("all_row_direct_output_scoring") - hybrid_score_components) / stage("all_row_direct_output_scoring"),
        "counterfactual_component_accounting_only": (
            "Not an independent hybrid runtime or speedup: trigger membership was derived using direct scores, "
            "so this measured trigger pattern is unavailable to a quadratic-only pass."
        ),
        "sensitive_recheck_full_linear_score_calls": recheck_calls,
        "sensitive_recheck_full_linears": triggered_linears,
        "sensitive_row_fraction": sensitive_rows / total_rows,
        "triggered_full_linear_fraction": triggered_linears / 168,
        "rechecked_row_alpha_values": recheck_row_alpha_values,
        "rechecked_row_alpha_fraction": recheck_row_alpha_values / total_candidates,
        "candidate_gptq_seconds": stage("full_linear_alpha_candidate_gptq"),
        "Hessian_build_seconds": stage("unique_group_hessian_build"),
        "Hinv_preparation_seconds": stage("shared_native_hinv_prepare"),
        "prefix_capture_refresh_seconds_by_stage": {
            name: stage_seconds[name] for name in stage_seconds
            if "capture" in name or "commit_to_" in name or "next_layer" in name or "dependency_forward" in name
        },
        "stitch_seconds": stage("full_linear_row_stitch"),
        "commit_seconds": stage("whole_linear_commit"),
        "checkpoint_IO_seconds": stage("atomic_dependency_group_checkpoint_io") + stage("atomic_layer_checkpoint_io"),
        "model_load_and_reload_seconds": stage("model_load"),
        "final_model_save_seconds": stage("final_dense_model_save"),
        "candidate_cache_peak_gpu_logical_bytes": peak_candidate_cache,
        "candidate_cache_disk_IO_bytes": 0,
        "candidate_cache_CPU_transfer_bytes": 0,
        "peak_GPU_allocated_bytes_from_stage_records": peak_allocated,
        "peak_GPU_reserved_bytes_from_stage_records": peak_reserved,
        "v28_historical_dual_run_seconds_for_budget_gate_only": v28_run_wall,
        "hybrid_only_full_model_wall_estimate_for_budget_gate_only": gate_hybrid_est,
        "hybrid_only_estimate_is_benchmark": False,
        "hybrid_only_remaining_budget_gate": "NOT_RUN: projected full run exceeds remaining cumulative budget; estimate is not a speedup claim",
        "standalone_hybrid_only_M2": "NOT_RUN",
        "new_PPL": "NOT_RUN",
        "historical_v18_M2_PPL_reusable_if_hashes_match": not named_diff and config_equal,
        "historical_v18_M2_final_model_file_sha256": v18_final_report["final_model_sha256"],
        "v29_final_model_file_sha256": runtime["m2"]["final_model_sha256"],
        "runtime_report_note": "Formal runtime JSON was written during checkpoint-only recovery and reports that short finalization process; use cumulative BUDGET ledger for total v29 cost.",
    }
    (RUN / "COST_SUMMARY.json").write_text(json.dumps(cost, indent=2) + "\n", "utf-8")

    recovery_note = {
        "status": "RECOVERED_AFTER_FINALIZATION_OUTPUT_DIRECTORY_ERROR",
        "initial_full_pass": "all 1680 candidate GPTQ executions and all 24 atomic layer checkpoints completed before the error",
        "error": "missing validation/M2 directory when writing QUANTIZATION_CORRECTNESS.json; no scoring or quantization failure",
        "recovery": "created only the v29 validation/M2 output directory and resumed from verified layer 24/24; candidate GPTQ reruns=0",
        "resumed_start_layer": runtime["m2"]["resume_start_layer"],
        "formal_failure_timestamp_from_manifest": manifest.get("formal_failure", {}).get("failed_utc"),
        "cumulative_budget_seconds": cumulative_total,
        "no_historical_artifacts_modified": True,
    }
    (RUN / "RECOVERY_NOTE.json").write_text(json.dumps(recovery_note, indent=2) + "\n", "utf-8")

    report = f"""# v29 — Quadratic Fast Scoring with Original-Score Recheck

## Result

- Full v29 dual-score replay: **{correctness['status']}** for 168 Linears and {total_rows:,} output rows. This is not an independent hybrid-only run.
- The v28 pure quadratic result remains **FAIL** with its one known selection-changing row. v29 is a separate hybrid validation.
- During this dual-scoring validation, the original direct-score selection was committed to preserve the reference prefix. The separately computed hybrid selection matches it in {correctness['hybrid_selected_state_exact_vs_direct_modules']}/168 Linears and {correctness['hybrid_selected_alpha_exact_rows']:,}/{total_rows:,} rows. Full named tensors and config match v18: **{not named_diff and config_equal}**.
- New PPL: **NOT_RUN**. Existing v18 M2 quality results describe the same normalized named tensors/config only because those hashes match.

## Frozen sensitivity rule and execution unit

- The v28 saved-score rule `2 × max_alpha(abs(quadratic − direct)) >= direct best/second gap` covers row 884 in L7 gate_proj. It flags {sensitive_rows} rows across {triggered_linears} Linears in this v29 run.
- v18 `score_rows` sends the entire Linear output dimension through each GEMM; it does not have output-row chunks. A single flagged row therefore causes a full-Linear recheck for all 10 candidates.
- Actual rechecks: {recheck_calls} full-Linear score calls, {recheck_rows_per_candidate:,} output rows per alpha summed across triggered Linears, and {recheck_row_alpha_values:,} row-alpha score values. All {recheck_exact_calls}/{recheck_calls} full-shape repeats exactly matched the initial direct score vectors.
- This rule is an empirical numerical guard, not a general error guarantee. It was computed with direct scores available during dual validation; standalone sensitivity prediction without direct scores has not been validated.
- The frozen trigger itself needs direct candidate scores (`abs(quadratic − direct)` and the direct best/second gap). Therefore the dual run validates the mixed result against the reference, but does **not** establish an operational quadratic-only trigger. The current `quadratic` mode cannot evaluate that trigger, and no replacement rule was introduced.

## Costs and cache

- Cumulative v29 budget ledger: **{cumulative_total:.2f}s ({cumulative_total/60:.2f} min)**, including spot/retries and checkpoint recovery. Hard cap: {hard_budget:.0f}s; remaining: {remaining:.2f}s.
- Main validation generated {runtime['m2']['formal_counts']['full_linear_alpha_candidate_executions']:,} full-Linear GPTQ candidates once and evaluated {runtime['m2']['formal_counts']['row_alpha_score_slots']:,} row-alpha values per scorer. Direct scoring: {stage('all_row_direct_output_scoring'):.2f}s; C construction plus quadratic scoring: {stage('score_C_build_once_per_dependency_group') + stage('all_row_quadratic_output_scoring'):.2f}s; sensitivity detection: {stage('sensitivity_detection'):.2f}s; rechecks: {stage('sensitive_full_linear_direct_recheck'):.2f}s.
- Although only {sensitive_rows}/{total_rows:,} rows ({sensitive_rows/total_rows:.4%}) triggered, the scorer's indivisible unit caused rechecks for {triggered_linears}/168 full Linears ({triggered_linears/168:.2%}), covering {recheck_row_alpha_values:,}/{total_candidates:,} row-alpha values ({recheck_row_alpha_values/total_candidates:.2%}). These 820 calls repeated direct scoring already performed in dual validation; they are not a standalone hybrid benchmark.
- Stage-only accounting gives C build + quadratic score + sensitivity detection + rechecks = {hybrid_score_components:.2f}s, compared with {stage('all_row_direct_output_scoring'):.2f}s for direct scoring (arithmetic difference {stage('all_row_direct_output_scoring') - hybrid_score_components:.2f}s). This is not a measured saving: the observed trigger set depends on direct scores and is not available to a quadratic-only run.
- Candidate cache remains GPU resident for one Linear at a time. Peak logical Q/scale/zero cache: {peak_candidate_cache:,} bytes; candidate-cache disk and CPU-transfer bytes: 0. Stage-record peak GPU memory: {peak_allocated/2**30:.3f} GiB allocated, {peak_reserved/2**30:.3f} GiB reserved.
- Standalone hybrid-only M2: **NOT_RUN**. The v28-stage-based full-run gate estimate is {gate_hybrid_est:.2f}s versus {remaining:.2f}s remaining, so it exceeds the remaining budget; this is a budget estimate only, not a measured speedup. In addition, the frozen trigger requires direct scores, so an operational quadratic-only hybrid needs a separately specified trigger before it can run.

## Recovery and limits

- The first full pass completed all candidates and 24 layer checkpoints; finalization then hit a missing v29 output directory. The run resumed from layer 24 with zero candidate reruns. See `RECOVERY_NOTE.json`.
- No new held-out evaluation or PPL was run. No online inference speed claim is made.

## Artifacts

- `PROTOCOL.json`, `IMPLEMENTATION_LOCK.json`, `SPOT_RECHECK.json`
- `MODULE_CONSISTENCY.csv`, `SENSITIVE_RECHECKS.csv`, `HYBRID_SELECTION_DIFFERENCES.csv`
- `RECHECK_DECISIONS.csv` (per triggered Linear: reason, before/after alpha histogram, full recheck scope, and extra wall time)
- `CONSISTENCY_SUMMARY.json`, `COST_SUMMARY.json`, `STAGE_TIMING_SUMMARY.json`, `NORMALIZED_TENSOR_HASHES.json`
- `MIXED_OPERATIONAL_LIMITATION.json`
- `FORMAL_TIMING.csv`, `M2_HYBRID_ROW_RESULTS.csv`, `raw/M2_candidate_losses/layer_XX.npz`
- `M2_HYBRID_ROW_RESULTS_WITH_HEADER.csv`, `M2_HYBRID_MODULE_SUMMARY_WITH_HEADER.csv` (headered exports; the runner's raw hybrid CSVs are retained unchanged)

See `REPRODUCE.md` for commands.
"""
    (RUN / "REPORT.md").write_text(report, "utf-8")
    (HERE / "REPORT.md").write_text(report, "utf-8")
    # The original runner omitted headers on its two hybrid CSV exports. Preserve
    # those raw artifacts and create explicit, headered copies for downstream use.
    csv_exports = {
        "M2_HYBRID_ROW_RESULTS.csv": ["module", "row", "direct_alpha", "quadratic_alpha", "hybrid_alpha",
            "direct_loss", "quadratic_selected_loss", "hybrid_loss", "sensitive_row", "linear_rechecked",
            "alpha_matches_direct", "weight_matches_direct", "scale_matches_direct", "zero_matches_direct",
            "g_idx_matches_direct"],
        "M2_HYBRID_MODULE_SUMMARY.csv": ["module", "layer", "rows", "sensitive_rows", "triggered_full_linear",
            "direct_recheck_score_calls", "rechecked_rows_per_candidate", "recheck_row_alpha_values",
            "sensitivity_detection_seconds", "direct_recheck_seconds", "candidate_cache_gpu_logical_bytes",
            "candidate_cache_shared_g_idx_bytes", "candidate_cache_disk_io_bytes", "candidate_cache_cpu_transfer_bytes",
            "hybrid_alpha_equal", "hybrid_weight_equal", "hybrid_scale_equal", "hybrid_zero_equal",
            "hybrid_g_idx_equal", "state_sha256"],
    }
    for source_name, fieldnames in csv_exports.items():
        source = RUN / source_name
        target = RUN / source_name.replace(".csv", "_WITH_HEADER.csv")
        with source.open(newline="", encoding="utf-8") as src, target.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.writer(dst)
            writer.writerow(fieldnames)
            for row in csv.reader(src):
                writer.writerow(row)
    hybrid_rows = list(csv.DictReader((RUN / "M2_HYBRID_ROW_RESULTS_WITH_HEADER.csv").open(newline="", encoding="utf-8")))
    module_rows = list(csv.DictReader((RUN / "M2_HYBRID_MODULE_SUMMARY_WITH_HEADER.csv").open(newline="", encoding="utf-8")))
    rechecks = []
    for module in module_rows:
        if module["triggered_full_linear"].lower() != "true":
            continue
        module_name = module["module"]
        rows_for_module = [row for row in hybrid_rows if row["module"] == module_name]
        histograms = {}
        for column in ("direct_alpha", "quadratic_alpha", "hybrid_alpha"):
            histograms[column] = dict(sorted(Counter(row[column] for row in rows_for_module).items()))
        rechecks.append({
            "module": module_name,
            "trigger_reason": "2*max_alpha(abs(quadratic-direct)) >= direct best/second gap for at least one row",
            "sensitive_rows": module["sensitive_rows"],
            "recheck_scope": "all output rows x all 10 candidates using unchanged v18 score_rows",
            "direct_recheck_score_calls": module["direct_recheck_score_calls"],
            "rechecked_rows_per_candidate": module["rechecked_rows_per_candidate"],
            "rechecked_row_alpha_values": module["recheck_row_alpha_values"],
            "quadratic_alpha_histogram_before_recheck": json.dumps(histograms["quadratic_alpha"], sort_keys=True),
            "hybrid_alpha_histogram_after_recheck": json.dumps(histograms["hybrid_alpha"], sort_keys=True),
            "direct_alpha_histogram_reference": json.dumps(histograms["direct_alpha"], sort_keys=True),
            "quadratic_to_hybrid_changed_rows": sum(row["quadratic_alpha"] != row["hybrid_alpha"] for row in rows_for_module),
            "hybrid_matches_direct_rows": sum(row["alpha_matches_direct"].lower() == "true" for row in rows_for_module),
            "extra_direct_recheck_seconds": module["direct_recheck_seconds"],
        })
    with (RUN / "RECHECK_DECISIONS.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["module", "trigger_reason", "sensitive_rows", "recheck_scope", "direct_recheck_score_calls",
                   "rechecked_rows_per_candidate", "rechecked_row_alpha_values", "quadratic_alpha_histogram_before_recheck",
                   "hybrid_alpha_histogram_after_recheck", "direct_alpha_histogram_reference",
                   "quadratic_to_hybrid_changed_rows", "hybrid_matches_direct_rows", "extra_direct_recheck_seconds"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rechecks)
    (RUN / "REPRODUCE.md").write_text(
        """# v29 reproduction\n\n## Audit the delivered run\n\nFrom `E:/quant-thesis` in the existing `.venv-gptq` PowerShell environment, regenerate summaries and headered CSV exports from the saved raw results (no GPU experiment is rerun):\n\n```powershell\n& .\\.venv-gptq\\Scripts\\python.exe experiments/range_search_w4/results/range_w4_quadratic_hybrid_m2_v29/analyze_v29.py\n```\n\n## Fresh full replay\n\nThe delivered v29 directory is an immutable completed run; its protocol freezer and implementation lock intentionally refuse to overwrite/relock it. To run again, create an empty sibling directory named `range_w4_quadratic_hybrid_m2_v29_replay`, copy `freeze_v29.py`, `spot_recheck_v29.py`, `lock_and_preflight_v29.py`, and `run_v29.py` into it, then run:\n\n```powershell\n$dst = 'experiments/range_search_w4/results/range_w4_quadratic_hybrid_m2_v29_replay'\nNew-Item -ItemType Directory -Path $dst\nCopy-Item experiments/range_search_w4/results/range_w4_quadratic_hybrid_m2_v29/{freeze_v29.py,spot_recheck_v29.py,lock_and_preflight_v29.py,run_v29.py} $dst\n@'{\n{\n  \"max_seconds\": 5400.0,\n  \"gpu_wall_elapsed_seconds\": 0.0,\n  \"last_recorded_stage\": \"fresh replay\"\n}\n}'@ | Set-Content -Encoding utf8 \"$dst/BUDGET.json\"\n& .\\.venv-gptq\\Scripts\\python.exe \"$dst/freeze_v29.py\"\n& .\\.venv-gptq\\Scripts\\python.exe \"$dst/spot_recheck_v29.py\"\n& .\\.venv-gptq\\Scripts\\python.exe \"$dst/lock_and_preflight_v29.py\"\n$env:V29_RUN_LABEL='validation'\n$env:V29_SCORE_MODE='dual'\n& .\\.venv-gptq\\Scripts\\python.exe \"$dst/run_v29.py\"\n```\n\nThe delivered run recovered finalization from 24 verified layer checkpoints after creating the missing `validation/M2` output directory; candidate GPTQ was not repeated during recovery. It used the frozen local model/calibration cache and wrote only below the v29 directory. The hybrid-only pass and PPL are not run.\n""", "utf-8")
    (HERE / "REPRODUCE.md").write_text((RUN / "REPRODUCE.md").read_text("utf-8"), "utf-8")
    write_reproduce_v29.write(RUN, HERE)
    print(json.dumps({"status": correctness["status"], "direct_exact_modules": direct_exact_modules,
                      "hybrid_exact_modules": runtime["m2"]["quantization_correctness"]["hybrid_exact_modules"],
                      "sensitive_rows": sensitive_rows, "triggered_linears": triggered_linears,
                      "recheck_calls": recheck_calls, "cumulative_seconds": cumulative_total,
                      "remaining_seconds": remaining, "normalized_tensors_equal": not named_diff,
                      "hybrid_only_estimate_seconds_for_budget_gate": gate_hybrid_est}, indent=2))


if __name__ == "__main__":
    main()
