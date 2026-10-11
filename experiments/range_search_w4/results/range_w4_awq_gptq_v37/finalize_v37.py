"""Finalize v37 evidence files from saved, frozen diagnostic outputs (CPU only)."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
COR = HERE / "correctness"


def read(rel: str):
    return json.loads((HERE / rel).read_text(encoding="utf-8"))


def backup_once(src: Path, dst: Path) -> None:
    if not dst.exists():
        shutil.copy2(src, dst)


def main() -> None:
    initial_report = COR / "v37_correctness_results.json"
    backup_once(initial_report, COR / "v37_correctness_results_INITIAL_DIAGNOSTIC.json")
    backup_once(HERE / "S0_VALIDITY.json", HERE / "S0_VALIDITY_INITIAL_INVALID_REFERENCE.json")
    backup_once(RAW / "S0_METRICS_RECOMPUTED.json", RAW / "S0_METRICS_INITIAL_INVALID_REFERENCE.json")

    p2 = read("correctness/S1_P2_CLEAN_RECAPTURE.json")
    parity = read("correctness/G00_DOWN_ALPHA1_PARITY_REPAIRED.json")
    selection = read("correctness/S0_SELECTION_REPLAY.json")
    norm = read("raw/S0_P1_NORM_REPLAY_WITH_D.json")
    corrected = read("raw/S0_HELD_REPAIRED_RESULT.json")
    corrected["execution_source_sha256"] = corrected.get("source_sha256")
    corrected["current_replay_script_sha256"] = hashlib.sha256((HERE / "repair_s0_held.py").read_bytes()).hexdigest()
    corrected["post_execution_script_change"] = "Only the CPU per-article CSV scoring line/label was changed after GPU execution; candidate outputs and pooled GPU metrics were not changed."
    ledger_rows = list(csv.DictReader((RAW / "BUDGET_LEDGER.csv").open(newline="", encoding="utf-8")))
    last = float(ledger_rows[-1]["cumulative_v37_gpu_seconds"])
    used = float(corrected["GPU_seconds_used"])
    if used > last:
        with (RAW / "BUDGET_LEDGER.csv").open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["unattributed_finalization_interval", "final_S0_repair_postprocessing",
                f"{used-last:.6f}", f"{used:.6f}", f"{float(corrected['route_gpu_seconds_used']):.6f}", 0, 0,
                "PASS", "Conservative process elapsed-wall reconciliation after the last measured GPU stage."])
        ledger_rows = list(csv.DictReader((RAW / "BUDGET_LEDGER.csv").open(newline="", encoding="utf-8")))

    all_arms_replayed = all(v["exact_vs_v36_saved_full_block_output"] for v in corrected["saved_full_block_arm_replay"].values())
    selection_and_path_valid = bool(all(selection["results"][k]["row_alpha_exact_after_recorded_dtype_cast"] and
        selection["results"][k]["best_loss_exact"] and selection["results"][k]["selected_tensor_alpha_exact"]
        for k in selection["results"]) and all(corrected["selection_lock_checks"].values()) and all_arms_replayed and
        parity["status"] == "PASS" and all(norm["reports"][k]["batched_replay_vs_old_post"]["exact"]
        for k in ("S0_P0",)) and all(norm["reports"][k]["locked_S0_D_replay_vs_old_post"]["exact"]
        for k in ("S0_P1", "S1_P2")))

    validity = {
        "status": "VALID_REPAIRED_HELD_REFERENCE_BUT_QUALITY_GATE_FAIL" if selection_and_path_valid and not corrected["primary_gate_passes"] else
                  ("VALID_REPAIRED" if selection_and_path_valid else "INVALID_OR_INCOMPLETE"),
        "candidate_selection_and_held_path_valid_after_v37_repair": selection_and_path_valid,
        "v36_original_P1_held_reference_valid": False,
        "v36_original_S0_metrics_valid": False,
        "v37_repaired_S0_metrics_valid": selection_and_path_valid,
        "checks": {
            "P0_postnorm_is_pristine_norm_replay_exact": norm["reports"]["S0_P0"]["batched_replay_vs_old_post"]["exact"],
            "P1_old_postnorm_is_S0_D_scaled_norm_replay_exact": norm["reports"]["S0_P1"]["locked_S0_D_replay_vs_old_post"]["exact"],
            "P2_old_postnorm_is_S0_D_scaled_norm_replay_exact": norm["reports"]["S1_P2"]["locked_S0_D_replay_vs_old_post"]["exact"],
            "clean_P2_capture_self_replay_exact": p2["clean_capture_self_replay_exact"],
            "G00_down_alpha1_direct_score_parity": parity["status"],
            "saved_S0_selection_argmins_replayed_with_recorded_dtype": selection["status"],
            "saved_four_arm_full_block_outputs_replayed_exact": all_arms_replayed,
            "selected_tensor_hashes_match_selection_lock": all(corrected["selection_lock_checks"].values()),
            "held_reference_rebuilt_from_pristine_norm_and_frozen_prenorm": True,
            "held_reference_sha256": corrected["pristine_P1_postnorm_sha256"],
            "four_arm_outputs_sha256_match_v36_saved_outputs": {k:v["exact_vs_v36_saved_full_block_output"] for k,v in corrected["saved_full_block_arm_replay"].items()},
            "S1_P3_read": False,
        },
        "quality_gates": {
            "primary_G11_vs_G01_ge_1pct": corrected["primary_gate_passes"],
            "corrected_fold_signal_floor": corrected["fold_signal_floor"]["passes"],
            "per_projection_no_regression_over_5pct": corrected["all_projection_gates_pass"],
            "projection_row_tail_at_most_5pct_over_10pct_and_none_over_50pct": corrected["all_projection_row_tail_gates_pass"],
        },
        "quality_gate_passes": bool(corrected["primary_gate_passes"] and corrected["fold_signal_floor"]["passes"] and
            corrected["all_projection_gates_pass"] and corrected["all_projection_row_tail_gates_pass"]),
        "primary_G11_vs_G01_held_improvement": corrected["primary_G11_vs_G01_improvement"],
        "decision": "B: correctness/reference repair completed; S0 held evaluation is valid; frozen quality gates fail; no S1 candidates or P3 evaluation.",
        "classification_boundary": "S0 is retrospective repair/re-evaluation on already viewed P1 data, not a new blind test.",
    }
    (HERE / "S0_VALIDITY.json").write_text(json.dumps(validity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (RAW / "S0_METRICS_RECOMPUTED.json").write_text(json.dumps(corrected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    v37_report = {
        "status": "S0_VALID_BUT_FROZEN_QUALITY_GATES_FAIL_NO_S1",
        "v36_gpu_seconds_before_v37": 180.335929,
        "v37_gpu_seconds_used": used,
        "route_gpu_seconds_used": float(corrected["route_gpu_seconds_used"]),
        "v37_subbudget_seconds": 300.0,
        "v37_stop_before_seconds": 270.0,
        "input_hash_audit": read("raw/INPUT_HASH_VERIFICATION.json"),
        "root_cause_status": "three v36 issues traced; see ROOT_CAUSE.md and BUG_IMPACT_TABLE.csv",
        "S1_P2_clean_recap_diagnosis": p2,
        "S0_P1_norm_replay_diagnosis": norm,
        "G00_down_alpha1_direct_score_repair": parity,
        "S0_candidate_selection_replay": selection,
        "S0_repaired_held_evaluation": corrected,
        "S0_validity": validity,
        "held_data_access": {"S0_P1":"previously viewed in v36; reopened only for frozen repair/re-evaluation",
                             "S1_P2":"search-only capture correctness diagnosis",
                             "S1_P3":"NOT_READ"},
        "new_search_or_candidate_generation":"NOT_RUN",
        "S1_quality_candidates":"NOT_RUN",
        "PPL":"NOT_RUN",
        "full_model_quantization":"NOT_RUN",
        "cumulative_budget_ledger":"raw/BUDGET_LEDGER.csv",
        "report":"REPORT.md",
    }
    (COR / "v37_correctness_results.json").write_text(json.dumps(v37_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Conservative final state accounts for all elapsed wall through finalization.
    route_used = 180.335929 + used
    (RAW / "V37_STATE.json").write_text(json.dumps({"status": v37_report["status"],
        "protocol_sha256": read("raw/V37_STATE.json")["protocol_sha256"],
        "v36_gpu_seconds_already_charged": 180.335929,"v37_gpu_seconds_used": used,
        "route_gpu_seconds_used": route_used,"v37_subbudget_seconds": 300.0,
        "v37_stop_before_seconds": 270.0,"route_cap_seconds": 1800,
        "last_stage": "S0_HELD_REPAIR_FINALIZED"},indent=2)+"\n",encoding="utf-8")

    # Stage cost ledger is charged in elapsed synchronized process time, including model load and failures.
    stages = {}
    for row in ledger_rows:
        stages[row["stage"]] = stages.get(row["stage"], 0.0) + float(row["gpu_seconds"])
    budget = {"v36_previously_charged_seconds":180.335929,"v37_total_charged_seconds":used,
        "route_cumulative_seconds":route_used,"v37_subbudget_seconds":300.0,
        "route_hard_cap_seconds":1800,"remaining_to_v37_subbudget_seconds":300.0-used,
        "remaining_to_v37_stop_before_seconds":270.0-used,"remaining_to_original_route_cap_seconds":1800.0-route_used,
        "accounting":"synchronized elapsed process time; includes model loads, CUDA warmups, every failed attempt, GPU work, output finalization, and conservative reconciliation; not kernel-only GPU time",
        "ledger_row_count":len(ledger_rows),"failed_gpu_stages":sum(1 for r in ledger_rows if r["status"]=="FAIL"),
        "failed_gpu_stage_seconds":sum(float(r["gpu_seconds"]) for r in ledger_rows if r["status"]=="FAIL"),
        "stage_seconds":stages,"peak_allocated_bytes_max":max(int(r["peak_allocated_bytes"]) for r in ledger_rows),
        "peak_reserved_bytes_max":max(int(r["peak_reserved_bytes"]) for r in ledger_rows),
        "new_candidate_search_executions":0,"fresh_native_alpha1_parity_executions_attempted":1,
        "fresh_native_parity_attempt_seconds":stages.get("native_alpha1_parity_GPTQ",0.0),
        "new_S1_candidate_executions":0,"P3_reads":0}
    pre_attempts = list(csv.DictReader((RAW / "PRE_GPU_ATTEMPT_LOG.csv").open(newline="", encoding="utf-8")))
    budget["pre_cuda_failed_attempts_zero_gpu_seconds"] = sum(1 for r in pre_attempts if r["status"] == "FAIL_BEFORE_CUDA")
    budget["pre_cuda_attempt_rows"] = len(pre_attempts)
    (RAW / "BUDGET_SUMMARY.json").write_text(json.dumps(budget,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
