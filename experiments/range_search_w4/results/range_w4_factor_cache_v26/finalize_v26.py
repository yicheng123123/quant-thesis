"""CPU-only, idempotent report builder from frozen v26 raw artifacts."""
import csv
import json
import statistics
from pathlib import Path

import run_v26 as r

O = Path(__file__).resolve().parent


def read(name):
    return json.loads((O / name).read_text(encoding="utf-8"))


def csv_write(name, rows):
    if not rows:
        raise ValueError(name)
    with (O / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    r.verify_sources()
    p = read("PROFILE_ORIGINAL.json")
    d = read("PROFILE_DETAIL.json")
    gate = read("PROFILE_DECISION.json")
    status = read("RUN_STATUS.json")
    assert p["status"] == d["status"] == "COMPLETE"
    assert not gate["implement_full_factor_cache"]
    assert set(p["targets"]) == set(d["targets"]) == {"up_proj", "down_proj"}
    assert status["gpu_wall_consumed_seconds"] <= 600

    timing_rows = []
    stage_rows = []
    summary = {}
    parity = {
        "status": "PASS_FOR_EXECUTED_ORIGINAL_CHECKS",
        "new_full_factor_cache": "NOT_IMPLEMENTED_BY_FROZEN_COST_GATE",
        "cached_vs_original": "NOT_RUN_BY_FROZEN_COST_GATE",
        "candidate_reverse_order": "NOT_RUN_BY_FROZEN_COST_GATE",
        "full_block_propagation": "NOT_RUN",
        "targets": {},
    }
    for name in ("up_proj", "down_proj"):
        v, detail, decision = p["targets"][name], d["targets"][name], gate["decision"][name]
        assert v["H_sha256"] == detail["H_sha256"]
        assert v["pristine_weight_sha256"] == detail["weight_sha256"]
        assert all(detail["alpha1_native_parity"].values())
        assert detail["H_unchanged"] and detail["W_unchanged"] and detail["score_finite"]
        assert len(v["records"]) == 4
        assert sorted((x["alpha"], x["replicate"]) for x in v["records"]) == [(.8, 1), (.8, 2), (1.0, 1), (1.0, 2)]
        candidate_times = [x["candidate_wall_seconds"] for x in v["records"]]
        score_times = [x["scoring"]["wall_seconds"] for x in v["records"]]
        all_rows = v["shape"][0]
        for c in v["records"]:
            assert c["counts"]["cached_inverse_hit"] == 1
            assert c["counts"]["lower_cholesky"] == 1
            assert c["counts"]["upper_cholesky"] == 1
            assert c["scoring"]["rows"] == all_rows
            timing_rows.append({
                "projection": name, "alpha": c["alpha"], "repeat": c["replicate"],
                "full_linear_gptq_calls": 1, "row_scores": all_rows,
                "candidate_wall_s": c["candidate_wall_seconds"],
                "candidate_gpu_stream_s": c["candidate_gpu_stream_elapsed_seconds"],
                "score_wall_s": c["scoring"]["wall_seconds"],
                "candidate_plus_score_wall_s": c["candidate_wall_seconds"] + c["scoring"]["wall_seconds"],
                "lower_cholesky_gpu_s": c["gpu_seconds"]["lower_cholesky"],
                "inverse_cache_hit_gpu_s": c["gpu_seconds"]["cached_inverse_hit"],
                "upper_cholesky_gpu_s": c["gpu_seconds"]["upper_cholesky"],
                "find_params_calls": c["counts"]["find_params"],
                "find_params_gpu_s": c["gpu_seconds"]["find_params"],
                "W_clone_gpu_s": c["gpu_seconds"]["W_clone_FP32"],
                "unattributed_gpu_s": c["residual_gpu_seconds"],
                "candidate_peak_allocated_bytes": c["cuda_peak_allocated_bytes"],
                "candidate_peak_reserved_bytes": c["cuda_peak_reserved_bytes"],
                "score_peak_allocated_bytes": c["scoring"]["peak_allocated_bytes"],
                "score_peak_reserved_bytes": c["scoring"]["peak_reserved_bytes"],
            })
        for stage, data in detail["H_and_factor_stages"].items():
            stage_rows.append({"projection": name, "part": "standalone_H_factor_decomposition", "stage": stage,
                               "wall_s": data["wall_seconds"], "gpu_stream_s": data["gpu_stream_seconds"],
                               "calls": 1, "additive_to_candidate": "NO"})
        for stage, data in detail["scoring_components"].items():
            stage_rows.append({"projection": name, "part": "separate_full_row_score_diagnostic", "stage": stage,
                               "wall_s": data["cpu_dispatch_envelope_seconds"], "gpu_stream_s": data["gpu_stream_seconds"],
                               "calls": data["calls"], "additive_to_candidate": "NO"})
        summary[name] = {
            "matrix_shape": v["shape"], "input_shape": v["input_shape"],
            "full_linear_gptq_calls_measured": 4,
            "row_scores_measured": 4 * all_rows,
            "full_row_score_tokens_per_candidate": 128 * 2048,
            "H_build_wall_s": v["H_build_seconds"],
            "shared_H_inverse_prepare_wall_s": v["H_preparation"]["wall_seconds"],
            "warmup_candidate_wall_s": v["warmup_seconds"],
            "candidate_mean_wall_s": statistics.mean(candidate_times),
            "candidate_range_wall_s": max(candidate_times) - min(candidate_times),
            "score_mean_wall_s": statistics.mean(score_times),
            "candidate_plus_score_mean_wall_s": statistics.mean(a + b for a, b in zip(candidate_times, score_times)),
            "two_alpha_stitch_wall_s": v["score_and_two_alpha_stitch_seconds"],
            "lower_upper_cholesky_fraction_of_candidate": decision["factor_share"],
            "factor_cache_cost_gate": decision["meets_both"],
            "peak_allocated_bytes": max(max(c["cuda_peak_allocated_bytes"], c["scoring"]["peak_allocated_bytes"]) for c in v["records"]),
            "peak_reserved_bytes": max(max(c["cuda_peak_reserved_bytes"], c["scoring"]["peak_reserved_bytes"]) for c in v["records"]),
        }
        parity["targets"][name] = {
            "historical_formal_input_hash_match": True,
            "historical_formal_H_hash_match": True,
            "alpha1_native_exact": detail["alpha1_native_parity"],
            "original_alpha1_and_08_repeat_exact": ["weight_fp16", "scales", "zeros", "g_idx", "all_row_score"],
            "pristine_W_and_H_unchanged": detail["W_unchanged"] and detail["H_unchanged"],
            "factor_decomposition_equals_existing_prepared_state_exact": True,
            "scored_rows": detail["score_rows"],
            "selected_08_rows_in_two_candidate_diagnostic": v["selected_08_rows"],
        }
    csv_write("CANDIDATE_TIMING.csv", timing_rows)
    csv_write("STAGE_TIMING.csv", stage_rows)
    r.write("PARITY.json", parity)
    report = {
        "verdict": "4_PROFILE_DOES_NOT_SUPPORT_FULL_FACTOR_CACHE",
        "cache_implemented": False,
        "cache_speedup_measured": False,
        "gpu_budget_seconds": 600,
        "conservative_gpu_time_including_two_failed_setup_attempts_seconds": status["gpu_wall_consumed_seconds"],
        "targets": summary,
        "candidate_calls_in_profile": sum(v["full_linear_gptq_calls_measured"] for v in summary.values()),
        "row_scores_in_profile": sum(v["row_scores_measured"] for v in summary.values()),
        "other_complete_gptq_calls": {"profile_warmups": 2, "detail_original_alpha1": 2, "detail_native_alpha1": 2},
        "all_complete_gptq_calls_this_run": 14,
        "other_full_row_score_calls": 2,
        "all_row_scores_this_run": sum(v["row_scores_measured"] for v in summary.values()) + 4864 + 896,
        "historical_M2_all_model_calibration_seconds": 2115.9353341000096,
        "historical_M2_all_model_candidate_calls": 1680,
        "historical_M2_all_model_row_alpha_score_slots": 3041280,
        "correctness": parity["status"],
        "non_run": ["full factor cache implementation/parity", "reverse candidate order", "full block propagation", "new held-out", "PPL", "whole-model quantization"],
        "failed_setup_attempts": ["PROFILE_ATTEMPT_FAILURE_01.json", "PROFILE_ATTEMPT_FAILURE_02.json"],
    }
    r.write("RUN_REPORT.json", report)
    print(json.dumps({"verdict": report["verdict"], "candidate_calls": report["candidate_calls_in_profile"],
                      "row_scores": report["row_scores_in_profile"], "gpu_seconds": report["conservative_gpu_time_including_two_failed_setup_attempts_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
