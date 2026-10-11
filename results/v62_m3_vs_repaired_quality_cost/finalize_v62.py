from __future__ import annotations

import csv, hashlib, json, math
from pathlib import Path

ROOT = Path(r"E:\quant-thesis")
OUT = ROOT / "results/v62_m3_vs_repaired_quality_cost"
V18 = ROOT / "experiments/range_search_w4/results/range_w4_fullmodel_v18"
V55 = ROOT / "experiments/range_search_w4/results/v55_m2_decoder_fixed_integer_repair"
V56 = ROOT / "experiments/range_search_w4/results/v56_m1_fast_integer_repair"
V59 = ROOT / "experiments/range_search_w4/results/v59_same_process_four_checkpoint_eval"
V61 = ROOT / "results/v61_m3_calibration_cost_memory"
V58 = ROOT / "experiments/range_search_w4/results/v58_m2_repaired_unified_eval"
DATASETS = ["wikitext", "c4_fixed_budget"]
ARMS = ["v61_M3", "v55_M2_plus_repair"]
WINDOWS = {"wikitext": 582, "c4_fixed_budget": 253}
VALID = {"wikitext": 299077, "c4_fixed_budget": 131071}

def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def read_csv(p: Path):
    return list(csv.DictReader(p.open(encoding="utf-8-sig", newline="")))

def write_json(p: Path, v):
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

def write_csv(p: Path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def to_int(x):
    return int(float(x))

def b_gib(x):
    return float(x) / (1024 ** 3) if x not in (None, "", "NOT_MEASURED") else x

def main():
    status = read_json(OUT / "RUN_STATUS.json")
    if status.get("status") != "COMPLETE" or status.get("complete_evaluations") != 4 or not status.get("same_process"):
        raise RuntimeError(f"cannot finalize incomplete/not-same-process run: {status}")
    source_rows = read_csv(OUT / "CHECKPOINT_SOURCE_AUDIT.csv")
    env = read_json(OUT / "PROCESS_ENVIRONMENT.json")
    lock = read_json(OUT / "RUN_LOCK.json")
    if len(source_rows) != 2 or env["process_id"] != status["process_ids"][0]:
        raise RuntimeError("source rows or main process identity mismatch")
    source_by_arm = {r["arm"]: r for r in source_rows}
    input_archives = read_json(OUT / "evaluation/INPUT_WINDOW_ARCHIVES.json")
    input_index = read_csv(OUT / "evaluation/INPUT_WINDOW_SOURCE_INDEX.csv")
    index_by = {(r["dataset"], to_int(r["window_index"])): r for r in input_index}
    # Verify raw token slices and the exact v59 label mask archive against its per-window hashes.
    import numpy as np
    archive_arrays = {}
    for ds, info in input_archives.items():
        if sha_file(Path(info["path"])) != info["sha256"]:
            raise RuntimeError(f"input archive hash mismatch: {ds}")
        z = np.load(info["path"], allow_pickle=False)
        archive_arrays[ds] = {k: z[k] for k in z.files}
        if len(archive_arrays[ds]["lengths"]) != WINDOWS[ds]:
            raise RuntimeError(f"input archive window count mismatch {ds}")

    checks, aggregates = [], []
    window_out_rows = []
    for arm in ARMS:
        for ds in DATASETS:
            csv_path = OUT / "evaluation" / f"{arm}_{ds}_WINDOWS.csv"
            rows = read_csv(csv_path)
            expected_result = read_json(OUT / "evaluation" / f"{arm}_{ds}.json")
            if len(rows) != WINDOWS[ds] or [to_int(r["window_index"]) for r in rows] != list(range(WINDOWS[ds])):
                raise RuntimeError(f"window count/order mismatch {arm}/{ds}")
            nll_sum, valid_sum = 0.0, 0
            ids_hash_ok, labels_hash_ok, windows_ok = True, True, True
            for r in rows:
                idx = to_int(r["window_index"])
                ix = index_by[(ds, idx)]
                a = archive_arrays[ds]
                length = int(a["lengths"][idx])
                ids = np.asarray(a["input_ids"][idx, :length], dtype="<i8")
                labels = np.asarray(a["labels"][idx, :length], dtype="<i8")
                ids_hash = sha_bytes(ids.tobytes())
                labels_hash = sha_bytes(labels.tobytes())
                ids_hash_ok &= ids_hash == r["input_ids_sha256"] == ix["input_ids_sha256_i64le"]
                labels_hash_ok &= labels_hash == r["labels_sha256"] == ix["labels_sha256_i64le"]
                windows_ok &= (to_int(r["begin"]) == to_int(ix["begin"]) and to_int(r["end"]) == to_int(ix["end"]) and
                               to_int(r["valid_tokens"]) == to_int(ix["valid_tokens"]) and
                               r["manifest_sha256"] == lock["input_manifest_sha256"][ds] and
                               r["model_sha256"] == lock["checkpoint_sha256"][arm] and
                               to_int(r["process_id"]) == env["process_id"])
                valid = to_int(r["valid_tokens"])
                nll_sum += float(r["window_loss"]) * valid
                if not math.isclose(float(r["window_nll"]), float(r["window_loss"]) * valid, rel_tol=0, abs_tol=1e-9):
                    raise RuntimeError(f"window NLL product mismatch {arm}/{ds}/{idx}")
                valid_sum += valid
                window_out_rows.append({"arm": arm, "dataset": ds, **r,
                                        "input_window_archive_sha256": input_archives[ds]["sha256"],
                                        "source_manifest_sha256": r["manifest_sha256"],
                                        "mask_rule": "labels before target_length are -100; next-token score mask; see INPUT_WINDOW_SOURCE_INDEX.csv"})
            if not ids_hash_ok or not labels_hash_ok or not windows_ok or valid_sum != VALID[ds]:
                raise RuntimeError(f"window source, mask, process or valid-token check failed {arm}/{ds}")
            nll = nll_sum / valid_sum
            ppl = math.exp(nll)
            if (expected_result["windows"] != WINDOWS[ds] or expected_result["valid_tokens"] != VALID[ds] or
                not math.isclose(nll, expected_result["weighted_nll"], rel_tol=0, abs_tol=1e-12) or
                not math.isclose(ppl, expected_result["ppl"], rel_tol=1e-12, abs_tol=1e-12)):
                raise RuntimeError(f"runner result differs from finalizer recalculation {arm}/{ds}")
            rec = {"arm": arm, "dataset": ds, "status": "PASS", "windows": len(rows), "valid_tokens": valid_sum,
                   "checkpoint_sha256": expected_result["checkpoint_sha256"],
                   "window_loss_times_valid_tokens_nll_sum": nll_sum, "weighted_nll": nll, "ppl": ppl,
                   "manifest_sha256": expected_result["manifest_sha256"], "token_ids_sha256_i64le": expected_result["token_ids_sha256_i64le"],
                   "input_archive_sha256": input_archives[ds]["sha256"], "all_input_slice_hashes_match": ids_hash_ok,
                   "all_labels_mask_hashes_match": labels_hash_ok, "all_windows_contiguous_and_source_bound": windows_ok,
                   "window_process_id": env["process_id"], "independent_from_runner_aggregate": True}
            checks.append(rec); aggregates.append(rec)
    write_csv(OUT / "INDEPENDENT_WINDOW_RECALC.csv", checks)
    write_csv(OUT / "evaluation/ALL_RAW_WINDOWS_WITH_PROVENANCE.csv", window_out_rows)

    by = {(r["arm"], r["dataset"]): r for r in aggregates}
    quality = []
    for ds in DATASETS:
        a, b = by[(ARMS[0], ds)], by[(ARMS[1], ds)]
        delta = a["weighted_nll"] - b["weighted_nll"]
        relp = (a["ppl"] / b["ppl"] - 1) * 100
        quality.append({"dataset": ds, "scope": "v62_same_process_primary", "v61_M3_NLL": a["weighted_nll"],
                        "v61_M3_PPL": a["ppl"], "v55_M2_plus_repair_NLL": b["weighted_nll"],
                        "v55_M2_plus_repair_PPL": b["ppl"], "delta_NLL_v61_minus_v55": delta,
                        "absolute_NLL_difference": abs(delta), "relative_PPL_change_v61_vs_v55_percent": relp,
                        "valid_tokens": a["valid_tokens"], "windows": a["windows"],
                        "lower_NLL": "v61_M3" if delta < 0 else ("v55_M2_plus_repair" if delta > 0 else "tie")})
    write_csv(OUT / "QUALITY_COMPARISON.csv", quality)

    # v61 planned-vs-executed optimizer step audit; retain exact source event rows.
    train_csv = V61 / "M3/logs/training_steps.csv"
    steps = read_csv(train_csv)
    skipped = [r for r in steps if r["optimizer_step_skipped"].lower() == "true"]
    executed = sum(r["optimizer_step_executed"].lower() == "true" for r in steps)
    unique = {(r["layer"], r["epoch_index"], r["sample_index"]) for r in steps}
    if len(steps) != 61440 or executed != 61438 or len(skipped) != 2 or len(unique) != 61440:
        raise RuntimeError(f"v61 step-count audit mismatch: rows={len(steps)}, executed={executed}, skipped={len(skipped)}, unique={len(unique)}")
    step_audit = {"status": "PASS", "planned_steps": 61440, "logged_scheduled_positions": len(steps), "unique_positions": len(unique),
                  "actual_optimizer_steps": executed, "skipped_steps": len(skipped), "difference": len(steps) - executed,
                  "source_csv": str(train_csv), "source_csv_sha256": sha_file(train_csv),
                  "interpretation": "Two native dynamic GradScaler updates saw found_inf=1 and skipped optimizer.step; the frozen runner consumes each scheduled sample once and does not retry or add compensating samples. Logs show finite loss but non-finite LWC gradient and gradient norm at those positions; the exact lower-level overflow source is not further decomposed.",
                  "original_implementation_behavior": "skip update and reduce scale, no retry/compensating sample per v61 frozen protocol",
                  "skips": [{k: r[k] for k in ["scheduled_iteration_count", "layer", "epoch_index", "sample_index", "loss", "scale_before", "found_inf", "grad_norm_finite", "optimizer_step_skipped", "scale_after", "optimizer_step_executed", "lwc_grad_finite", "lwc_parameters_unchanged_on_skip", "optimizer_state_unchanged_on_skip"]} for r in skipped],
                  "not_rerun_or_modified": True}
    write_json(OUT / "V61_STEP_COUNT_AUDIT.json", step_audit)

    # Reconcile v61 and v55 calibration token hashes. Their aggregate hash fields
    # use different documented byte/string constructions, so compare source rows and
    # raw token arrays before treating the differing digests as a data mismatch.
    v61_cal_source = read_json(V61 / "CALIBRATION_SOURCE.json")
    v61_cal_manifest = Path(v61_cal_source["manifest_path"])
    if not v61_cal_manifest.is_absolute():
        v61_cal_manifest = ROOT / v61_cal_manifest
    v55_cal_manifest = V18 / "CALIBRATION_MANIFEST.json"
    v61_manifest_doc = read_json(v61_cal_manifest)
    v55_manifest_doc = read_json(v55_cal_manifest)
    if sha_file(v61_cal_manifest) != v61_cal_source["manifest_sha256"] or sha_file(v55_cal_manifest) != v61_cal_source["manifest_sha256"]:
        raise RuntimeError("M3/M2 calibration manifest bytes differ from the frozen source hash")
    m3_cal_rows = read_csv(V61 / "CALIBRATION_SOURCE_ROWS.csv")
    samples = v61_manifest_doc["samples"]
    per_sample_hashes = []
    flat_ids = []
    bad_cal_rows = []
    for i, sample in enumerate(samples):
        ids = sample["token_ids"]
        digest = sha_bytes(json.dumps(ids).encode("utf-8"))
        flat_ids.extend(ids)
        per_sample_hashes.append(digest)
        row = m3_cal_rows[i]
        if len(ids) != 2048 or digest != sample.get("token_sha256") or digest != row["token_sha256"] or to_int(row["sample_index"]) != i:
            bad_cal_rows.append(i)
    m3_aggregate = sha_bytes(json.dumps(per_sample_hashes, separators=(",", ":")).encode("utf-8"))
    m2_flat_aggregate = sha_bytes(np.asarray(flat_ids, dtype="<i8").tobytes())
    v55_proto_for_hash = read_json(V55 / "PROTOCOL.json")
    if bad_cal_rows or len(samples) != 128 or m3_aggregate != v61_cal_source["ordered_token_hashes_sha256"] or m2_flat_aggregate != v55_proto_for_hash["calibration"]["ordered_token_ids_sha256"]:
        raise RuntimeError(f"calibration token source reconciliation failed: rows={bad_cal_rows[:8]}, M3={m3_aggregate}, M2={m2_flat_aggregate}")
    calibration_reconciliation = {"status": "PASS", "v61_manifest_path": str(v61_cal_manifest), "v55_manifest_path": str(v55_cal_manifest),
                                  "manifest_sha256": sha_file(v61_cal_manifest), "m2_manifest_sha256": sha_file(v55_cal_manifest),
                                  "documents": len(samples), "tokens_per_document": 2048, "total_tokens": len(flat_ids),
                                  "all_v61_per_sample_hashes_match_locked_manifest": True,
                                  "v61_ordered_hash": m3_aggregate, "v61_ordered_hash_definition": "SHA256 of compact JSON array of per-sample SHA256(JSON token_ids) strings",
                                  "m2_ordered_token_ids_hash": m2_flat_aggregate, "m2_hash_definition": "SHA256 of flattened little-endian int64 token IDs",
                                  "aggregate_hashes_are_not_directly_comparable_due_to_different_construction": True,
                                  "same_underlying_ordered_token_arrays": True}
    write_json(OUT / "CALIBRATION_HASH_RECONCILIATION.json", calibration_reconciliation)

    # Existing formal-build/repair costs; scopes are deliberately kept separate.
    v56_budget = read_json(V56 / "FINAL_BUDGET_ACCOUNTING.json")
    v56_repair = read_json(V56 / "FORMAL_REPAIR_COMPLETE.json")
    v55_budget = read_json(V55 / "FINAL_BUDGET_ACCOUNTING.json")
    v55_repair = read_json(V55 / "FORMAL_REPAIR_COMPLETE.json")
    v61_cost = read_json(V61 / "COST_ACCOUNTING.json")
    v61_memory = read_json(V61 / "MEMORY_PEAKS.json")
    v18_timing = read_csv(V18 / "FORMAL_TIMING.csv")
    v59_stages = read_csv(V59 / "STAGE_TIMINGS.csv")
    v59_quality = read_csv(V59 / "INDEPENDENT_WINDOW_RECALC.csv")
    v59q = {(r["arm"], r["dataset"]): r for r in v59_quality}
    v59pair = {a: sum(float(r["seconds"]) for r in v59_stages if r["arm"] == a and r["stage"] == "evaluation") for a in ["M1", "M2", "M1_plus_repair", "M2_plus_repair"]}
    build_m1 = float(v56_budget["components"]["historical_M1_build_seconds_separate"])
    build_m2 = float(v56_budget["components"]["historical_M2_build_seconds_separate"])

    def old_peak(rows, method):
        subset = [r for r in rows if r.get("method") == method]
        vals_a = [to_int(r["gpu_peak_allocated_bytes"]) for r in subset if r.get("gpu_peak_allocated_bytes")]
        vals_r = [to_int(r["gpu_peak_reserved_bytes"]) for r in subset if r.get("gpu_peak_reserved_bytes")]
        return (max(vals_a) if vals_a else "NOT_MEASURED", max(vals_r) if vals_r else "NOT_MEASURED")
    m1mem = old_peak(v18_timing, "M1"); m2mem = old_peak(v18_timing, "M2")
    def mode_peak(path: Path):
        rows = read_csv(path)
        rows = [r for r in rows if r.get("mode") == "formal"]
        a = [to_int(r["gpu_peak_allocated_bytes"]) for r in rows if r.get("gpu_peak_allocated_bytes")]
        b = [to_int(r["gpu_peak_reserved_bytes"]) for r in rows if r.get("gpu_peak_reserved_bytes")]
        return (max(a) if a else "NOT_MEASURED", max(b) if b else "NOT_MEASURED")
    v55mem = mode_peak(V55 / "STAGE_TIMINGS.csv")
    v56mem = mode_peak(V56 / "STAGE_TIMINGS.csv")
    v61_complete_mem = next(a for a in v61_memory["attempts"] if a["status"] == "COMPLETE")
    v61_alloc = max(a["peak_allocated_bytes"] for a in v61_memory["allocator_attempts"] if a.get("peak_allocated_bytes") is not None)
    v61_reserved = max(a["peak_reserved_bytes"] for a in v61_memory["allocator_attempts"] if a.get("peak_reserved_bytes") is not None)
    process_main = [p for p in v61_complete_mem["process_rss_peaks"] if p.get("effective_role") == "MODEL_MAIN"]
    v61_rss = max(p["peak_rss_bytes"] for p in process_main) if process_main else "NOT_MEASURED"
    v61_dev = v61_complete_mem.get("device_sampled_peak_bytes", "NOT_MEASURED")
    cur_mem = read_csv(OUT / "MODEL_LOAD_AUDIT.csv")
    curm = {r["arm"]: {"pytorch_peak_allocated_bytes": r["peak_allocated_after_pair_bytes"],
                        "pytorch_peak_reserved_bytes": r["peak_reserved_after_pair_bytes"],
                        "device_sampled_peak_bytes_global": r["device_sampled_peak_bytes_global"],
                        "process_rss_sampled_peak_bytes": r["rss_sampled_peak_bytes"]} for r in cur_mem}

    def quality_cols(source_arm, scope):
        rows = []
        for ds in DATASETS:
            r = by[(source_arm, ds)] if scope == "v62" else v59q[(source_arm, ds)]
            rows.append((float(r["weighted_nll"]), float(r["ppl"])))
        return rows
    m1q = [(float(v59q[("M1", d)]["weighted_nll"]), float(v59q[("M1", d)]["ppl"])) for d in DATASETS]
    m2q = [(float(v59q[("M2", d)]["weighted_nll"]), float(v59q[("M2", d)]["ppl"])) for d in DATASETS]
    r1q = [(float(v59q[("M1_plus_repair", d)]["weighted_nll"]), float(v59q[("M1_plus_repair", d)]["ppl"])) for d in DATASETS]
    r2q = [(float(by[("v55_M2_plus_repair", d)]["weighted_nll"]), float(by[("v55_M2_plus_repair", d)]["ppl"])) for d in DATASETS]
    m3q = [(float(by[("v61_M3", d)]["weighted_nll"]), float(by[("v61_M3", d)]["ppl"])) for d in DATASETS]

    costs = [
        {"solution": "M1_native_GPTQ", "quality_source": "v59 historical same-process background", "WT2_NLL": m1q[0][0], "WT2_PPL": m1q[0][1], "C4_fixed_budget_NLL": m1q[1][0], "C4_fixed_budget_PPL": m1q[1][1],
         "base_build_or_calibration_seconds": build_m1, "full_repair_seconds": 0, "repair_measured_scope": "not applicable", "evaluation_pair_seconds": v59pair["M1"], "evaluation_scope": "v59 synchronized WT2+C4 stages",
         "full_chain_wall_seconds": "NOT_MEASURED", "peak_allocated_bytes": m1mem[0], "peak_allocated_GiB": b_gib(m1mem[0]), "peak_reserved_bytes": m1mem[1], "peak_reserved_GiB": b_gib(m1mem[1]), "device_sampled_peak_bytes": "NOT_MEASURED", "main_process_RSS_peak_bytes": "NOT_MEASURED", "memory_scope": "v18 formal timing maxima; no RSS or device sampling"},
        {"solution": "M2_scalar10", "quality_source": "v59 historical same-process background", "WT2_NLL": m2q[0][0], "WT2_PPL": m2q[0][1], "C4_fixed_budget_NLL": m2q[1][0], "C4_fixed_budget_PPL": m2q[1][1],
         "base_build_or_calibration_seconds": build_m2, "full_repair_seconds": 0, "repair_measured_scope": "not applicable", "evaluation_pair_seconds": v59pair["M2"], "evaluation_scope": "v59 synchronized WT2+C4 stages",
         "full_chain_wall_seconds": "NOT_MEASURED", "peak_allocated_bytes": m2mem[0], "peak_allocated_GiB": b_gib(m2mem[0]), "peak_reserved_bytes": m2mem[1], "peak_reserved_GiB": b_gib(m2mem[1]), "device_sampled_peak_bytes": "NOT_MEASURED", "main_process_RSS_peak_bytes": "NOT_MEASURED", "memory_scope": "v18 formal timing maxima; no RSS or device sampling"},
        {"solution": "v61_repository_M3_LWC", "quality_source": "v62 same-process primary", "WT2_NLL": m3q[0][0], "WT2_PPL": m3q[0][1], "C4_fixed_budget_NLL": m3q[1][0], "C4_fixed_budget_PPL": m3q[1][1],
         "base_build_or_calibration_seconds": v61_cost["data_to_finalized_model_pipeline_seconds"], "full_repair_seconds": "NOT_APPLICABLE", "repair_measured_scope": "M3 LWC training/calibration, not integer repair",
         "evaluation_pair_seconds": sum(float(r["seconds"]) for r in read_csv(OUT / "STAGE_TIMINGS.csv") if r["arm"] == "v61_M3" and r["stage"] == "evaluation"), "evaluation_scope": "v62 fresh same-process WT2+C4",
         "full_chain_wall_seconds": v61_cost["total_all_attempt_wall_seconds"], "peak_allocated_bytes": v61_alloc, "peak_allocated_GiB": b_gib(v61_alloc), "peak_reserved_bytes": v61_reserved, "peak_reserved_GiB": b_gib(v61_reserved),
         "device_sampled_peak_bytes": v61_dev, "main_process_RSS_peak_bytes": v61_rss, "memory_scope": "v61 calibration; device total includes other processes; sampled, not instantaneous; WDDM per-process GPU unavailable"},
        {"solution": "v55_M2_scalar10_plus_integer_repair", "quality_source": "v62 same-process primary", "WT2_NLL": r2q[0][0], "WT2_PPL": r2q[0][1], "C4_fixed_budget_NLL": r2q[1][0], "C4_fixed_budget_PPL": r2q[1][1],
         "base_build_or_calibration_seconds": build_m2, "full_repair_seconds": v55_budget["valid_new_layer2_to23_block_wall_seconds"], "repair_measured_scope": "layers 2-23 only; block0/1 groups inherited; complete 24-block net repair duration not identifiable",
         "evaluation_pair_seconds": sum(float(r["seconds"]) for r in read_csv(OUT / "STAGE_TIMINGS.csv") if r["arm"] == "v55_M2_plus_repair" and r["stage"] == "evaluation"), "evaluation_scope": "v62 fresh same-process WT2+C4",
         "full_chain_wall_seconds": "NOT_IDENTIFIABLE", "known_added_chain_charged_seconds": v55_budget["v55_added_charged_seconds"], "v55_cumulative_history_seconds": v55_budget["final_cumulative_charged_seconds"],
         "peak_allocated_bytes": v55mem[0], "peak_allocated_GiB": b_gib(v55mem[0]), "peak_reserved_bytes": v55mem[1], "peak_reserved_GiB": b_gib(v55mem[1]), "device_sampled_peak_bytes": "NOT_MEASURED", "main_process_RSS_peak_bytes": "NOT_MEASURED", "memory_scope": "v55 stage maxima; RSS/device sampling unavailable"},
        {"solution": "v56_M1_native_plus_integer_repair", "quality_source": "v59 historical same-process background", "WT2_NLL": r1q[0][0], "WT2_PPL": r1q[0][1], "C4_fixed_budget_NLL": r1q[1][0], "C4_fixed_budget_PPL": r1q[1][1],
         "base_build_or_calibration_seconds": build_m1, "full_repair_seconds": v56_repair["elapsed_seconds"], "repair_measured_scope": "full 24 blocks, 168 Linear, 304128 rows from layer0",
         "evaluation_pair_seconds": v59pair["M1_plus_repair"], "evaluation_scope": "v59 synchronized WT2+C4; v56 original eval continuation separately 2283.576 s",
         "full_chain_wall_seconds": v56_budget["conservative_accounted_total_seconds"], "peak_allocated_bytes": v56mem[0], "peak_allocated_GiB": b_gib(v56mem[0]), "peak_reserved_bytes": v56mem[1], "peak_reserved_GiB": b_gib(v56mem[1]), "device_sampled_peak_bytes": "NOT_MEASURED", "main_process_RSS_peak_bytes": "NOT_MEASURED", "memory_scope": "v56 formal repair stage maxima; conservative all-attempt budget total includes interrupted eval/CPU/diagnostic; no device/RSS sampler"}
    ]
    # Current v62 peak memory is a separately measured inference scope; append detail rows.
    for arm, label in [("v61_M3", "v62_evaluation_v61_M3"), ("v55_M2_plus_repair", "v62_evaluation_v55_M2_plus_repair")]:
        r = curm[arm]
        costs.append({"solution": label, "quality_source": "v62 same-process primary; evaluation memory scope", "WT2_NLL": "see QUALITY_COMPARISON.csv", "WT2_PPL": "see QUALITY_COMPARISON.csv", "C4_fixed_budget_NLL": "see QUALITY_COMPARISON.csv", "C4_fixed_budget_PPL": "see QUALITY_COMPARISON.csv",
                      "base_build_or_calibration_seconds": "NOT_APPLICABLE", "full_repair_seconds": "NOT_APPLICABLE", "evaluation_pair_seconds": "see STAGE_TIMINGS.csv", "evaluation_scope": "v62 inference only",
                      "full_chain_wall_seconds": "NOT_APPLICABLE", "peak_allocated_bytes": r["pytorch_peak_allocated_bytes"], "peak_allocated_GiB": b_gib(r["pytorch_peak_allocated_bytes"]),
                      "peak_reserved_bytes": r["pytorch_peak_reserved_bytes"], "peak_reserved_GiB": b_gib(r["pytorch_peak_reserved_bytes"]),
                      "device_sampled_peak_bytes": r["device_sampled_peak_bytes_global"], "device_sampled_peak_GiB": b_gib(r["device_sampled_peak_bytes_global"]),
                      "main_process_RSS_peak_bytes": r["process_rss_sampled_peak_bytes"], "main_process_RSS_peak_GiB": b_gib(r["process_rss_sampled_peak_bytes"]),
                      "memory_scope": "v62 0.5-second samples; device total includes other processes; process GPU attribution unavailable under WDDM"})
    write_csv(OUT / "QUALITY_COST_MEMORY_TABLE.csv", costs)

    # Cost references and source hashes: different scopes are explicit; no speed ratios.
    cost_sources = [
        ("M1_M2_build_cost", V56 / "FINAL_BUDGET_ACCOUNTING.json", "historical M1/M2 build costs separately sourced by v56; not in v56 cap"),
        ("v18_formal_timing", V18 / "FORMAL_TIMING.csv", "M1/M2 original formal measured stage record; nested rows not summed into a new wall"),
        ("v55_repair_chain", V55 / "FINAL_BUDGET_ACCOUNTING.json", "22 newly repaired blocks and cumulative recovery-chain accounting"),
        ("v56_repair_chain", V56 / "FINAL_BUDGET_ACCOUNTING.json", "full-from-layer0 repair, evaluation continuation, failures and conservative total"),
        ("v61_M3_cost", V61 / "COST_ACCOUNTING.json", "full calibration attempt timing and failure"),
        ("v61_M3_memory", V61 / "MEMORY_PEAKS.json", "allocator and sampling peaks"),
        ("v59_runtime_eval", V59 / "STAGE_TIMINGS.csv", "same-process stage evaluation times for current-runtime planning/background"),
        ("v59_cost", V59 / "COST_ACCOUNTING.json", "all-eight evaluation run wall accounting"),
        ("v58_pair_cost", V58 / "COST_ACCOUNTING.json", "one M2+repair WT2+C4 pair wall, seconds"),
        ("v62_eval_cost", OUT / "COST_ACCOUNTING.json", "this run and all-in v62 charged activity"),
    ]
    write_csv(OUT / "COST_SOURCES.csv", [{"source": k, "path": str(p), "sha256": sha_file(p), "scope": scope} for k,p,scope in cost_sources])
    write_json(OUT / "COST_ACCOUNTING.json", {**read_json(OUT / "COST_ACCOUNTING.json"),
                "historical_reference": {"M1_native_GPTQ_build_seconds": build_m1, "M2_scalar10_build_seconds": build_m2,
                    "v61_M3_full_data_to_finalized_pipeline_seconds": v61_cost["data_to_finalized_model_pipeline_seconds"],
                    "v61_M3_successful_load_save_reload_invocation_seconds": v61_cost["full_invocation_wall_seconds"],
                    "v61_M3_all_attempts_including_failure_seconds": v61_cost["total_all_attempt_wall_seconds"],
                    "v61_M3_failure_seconds": v61_cost["failure_wall_seconds"],
                    "v55_M2_repair_measured_layer2_to23_seconds": v55_budget["valid_new_layer2_to23_block_wall_seconds"],
                    "v55_M2_repair_full24_net_seconds": "NOT_IDENTIFIABLE_FROM_RECOVERY_CHAIN",
                    "v55_continuation_runner_seconds": v55_budget["v55_formal_runner_process_wall_seconds"],
                    "v55_added_shared_chain_charge_seconds": v55_budget["v55_added_charged_seconds"],
                    "v55_cumulative_history_seconds": v55_budget["final_cumulative_charged_seconds"],
                    "v56_M1_repair_full_model_seconds": v56_repair["elapsed_seconds"],
                    "v56_original_eval_only_continuation_seconds": v56_budget["components"]["evaluation_only_continuation_wall_seconds"],
                    "v56_conservative_all_attempt_total_seconds": v56_budget["conservative_accounted_total_seconds"],
                    "v59_total_evaluation_stage_seconds_all_8": json.loads((V59 / "COST_ACCOUNTING.json").read_text(encoding="utf-8"))["evaluation_stage_sum_seconds"],
                    "v56_pair_seconds_are_seconds_not_minutes": True},
                "no_strict_speed_ratios_reported": True,
                "cost_interpretation": "M3 training/calibration, original M1/M2 quantization, integer repair, evaluation, and recovery/diagnostic totals are different work and timer scopes; rows are not added into algorithm speed ratios. v55 full 24-block net repair cannot be reconstructed from its inherited recovery chain; the missing block0/1 generation time is not replaced with cumulative debugging costs."})
    v62_accounting = read_json(OUT / "COST_ACCOUNTING.json")
    attempt_id = read_json(OUT / "RUN_STARTED.json")["attempt_id"]
    runner_attempt = read_json(OUT / "attempts" / attempt_id / "ATTEMPT_ACCOUNTING.json")
    # The timed runner starts after source/input freezing; charge a conservative pre-lock allowance
    # and all CPU report/finalizer attempts to the 5400-second task budget.
    v62_accounting["successful_attempt_prelock_source_and_input_archive_charge_seconds"] = 30.0
    v62_accounting["final_report_hash_and_CPU_finalizer_attempts_charge_seconds"] = 12.0
    v62_accounting["charged_v62_active_seconds"] = float(runner_attempt["cumulative_v62_active_seconds"]) + 42.0
    v62_accounting["remaining_seconds"] = float(v62_accounting["cap_seconds"]) - v62_accounting["charged_v62_active_seconds"]
    v62_accounting["runner_wall_scope_note"] = "runner_wall_seconds starts after pre-lock source verification and exact input-window archive creation; the conservative 30 s pre-lock charge is separate"
    v62_accounting["finalization_scope_note"] = "12 s conservative CPU charge covers report creation, manifest hashing, verification and earlier failed CPU finalizer passes"
    write_json(OUT / "COST_ACCOUNTING.json", v62_accounting)
    final_status = read_json(OUT / "RUN_STATUS.json")
    final_status["charged_active_seconds_including_precheck_prelock_and_closeout"] = v62_accounting["charged_v62_active_seconds"]
    final_status["remaining_seconds"] = v62_accounting["remaining_seconds"]
    final_status["budget_cap_seconds"] = v62_accounting["cap_seconds"]
    write_json(OUT / "RUN_STATUS.json", final_status)
    cost_sources_rows = read_csv(OUT / "COST_SOURCES.csv")
    for row in cost_sources_rows:
        if row["source"] == "v62_eval_cost":
            row["sha256"] = sha_file(OUT / "COST_ACCOUNTING.json")
    write_csv(OUT / "COST_SOURCES.csv", cost_sources_rows)

    # Compare v55's historical same-runtime v59 report to its new v62 eval as a checkpoint check.
    v55_hist = {(r["dataset"]): r for r in v59_quality if r["arm"] == "M2_plus_repair"}
    hist_check = []
    for ds in DATASETS:
        cur = by[("v55_M2_plus_repair", ds)]; old = v55_hist[ds]
        hist_check.append({"dataset": ds, "v62_v55_nll": cur["weighted_nll"], "v59_v55_nll": float(old["weighted_nll"]),
                           "delta_nll_v62_minus_v59": cur["weighted_nll"] - float(old["weighted_nll"]),
                           "v62_v55_ppl": cur["ppl"], "v59_v55_ppl": float(old["ppl"]),
                           "delta_ppl_v62_minus_v59": cur["ppl"] - float(old["ppl"]),
                           "same_checkpoint_sha": cur["checkpoint_sha256"] == old["checkpoint_sha256"],
                           "interpretation": "same locked checkpoint in same v59/v62 frozen runtime family; background consistency check"})
    write_csv(OUT / "V55_V59_HISTORICAL_CHECK.csv", hist_check)

    # Short, self-contained report and reproduction notes.
    qwt = quality[0]; qc4 = quality[1]
    v62_cost = read_json(OUT / "COST_ACCOUNTING.json")
    v62_env = read_json(OUT / "PROCESS_ENVIRONMENT.json")
    step_lines = "; ".join(f"layer{r['layer']} epoch{r['epoch_index']} sample{r['sample_index']} (iteration {r['scheduled_iteration_count']}, found_inf={r['found_inf']}, scale {r['scale_before']}→{r['scale_after']})" for r in skipped)
    report = f'''# v62 — v61 仓库 M3/LWC 与 v55 M2+整数修复同进程质量/成本比较

## 结果

两套 checkpoint 在同一 Python 进程按 M3→v55 顺序、同一冻结 evaluator 完成四项评分。窗口共 582 WT2 test 和 253 C4 fixed-budget validation，每项计分 token 分别为 299,077 与 131,071；窗口、输入切片和 labels mask 均逐项核对，最终质量只使用本轮完整窗口。

| 语料 | v61 M3 NLL / PPL | v55 M2+修复 NLL / PPL | ΔNLL（M3−v55） | M3 相对 PPL 变化 |
|---|---:|---:|---:|---:|
| WT2 test | {qwt['v61_M3_NLL']:.9f} / {qwt['v61_M3_PPL']:.9f} | {qwt['v55_M2_plus_repair_NLL']:.9f} / {qwt['v55_M2_plus_repair_PPL']:.9f} | {qwt['delta_NLL_v61_minus_v55']:+.9f} nats/token | {qwt['relative_PPL_change_v61_vs_v55_percent']:+.4f}% |
| C4 fixed-budget validation | {qc4['v61_M3_NLL']:.9f} / {qc4['v61_M3_PPL']:.9f} | {qc4['v55_M2_plus_repair_NLL']:.9f} / {qc4['v55_M2_plus_repair_PPL']:.9f} | {qc4['delta_NLL_v61_minus_v55']:+.9f} nats/token | {qc4['relative_PPL_change_v61_vs_v55_percent']:+.4f}% |

M3 在 C4 fixed-budget 上更低，WT2 上 v55 更低，出现语料间取舍；没有一个方案在两项都占优。负 ΔNLL 表示 M3 更低。C4 是冻结的固定预算验证流，不能称标准 full C4。两者使用相同模型 revision、token manifest 与有效 Qwen2 配置。M3 导出 tokenizer 与锁定 revision 在固定探针编码、词表长度和 special-token id 上一致；正式 evaluator 直接使用冻结 token-ID manifest。两个 calibration manifest 的 SHA 相同，128×2048 的逐篇 token 哈希逐项匹配。v61 与 M2 的 ordered-token 总 hash 字段采用不同构造（per-sample hash 列表 vs 扁平 int64 token 数组），所以摘要值不同但底层 token 顺序相同，详见 `CALIBRATION_HASH_RECONCILIATION.json`。量化方法不同：v61 为仓库 M3/LWC W4A16 非对称 affine G128；v55 为 M2 scalar10 加 W4A16 对称 G128 整数码修复。这里比较的是两套完整方案，不能把差异单独归因于 LWC、scalar10 或修复。没有官方 OmniQuant 完整复现声明。

## 评测与正确性

进程 PID `{v62_env['process_id']}`，Python `{v62_env['python']}`、PyTorch `{v62_env['pytorch']}`、Transformers `{v62_env['transformers']}`、CUDA `{v62_env['cuda_runtime']}`、GPU `{v62_env['gpu']}`；FP16、SDPA 请求、TF32 off、`use_cache=False`。两个有效配置指纹相同；全部评测窗口来自同一进程。SDPA 后端策略已记录，底层具体 kernel 未插桩。实际服务型号接口不可见，因此不能确认 Luna Max 是否被执行。

v61 的 61,440 个计划位置均有日志，实际优化器步为 61,438。两次跳步为：{step_lines}。冻结实现的动态 GradScaler 在 `found_inf=1` 时跳过优化器更新并降低 scale，不重试，也不补充样本；loss 有限，但日志中的 LWC gradient 与 gradient norm 标记非有限。更低层的 overflow 成因没有进一步拆解。v61 新模型未继承任何旧 M3 PPL。

每个窗口用 `window_loss × valid_tokens` 重算 NLL sum，除以该语料有效 token 总数得加权 NLL，再取 `exp(NLL)` 得 PPL。完整原始窗口、token 切片压缩档、mask 与逐窗哈希和独立复算在 `evaluation/`。

## 成本与内存

v62 四项评测阶段合计 {v62_cost['evaluation_stage_sum_seconds']:.3f} 秒；包含加载/哈希/卸载的正式 runner wall {v62_cost['runner_wall_seconds']:.3f} 秒。含 CPU 预检、两次失败诊断、首次 M3 加载核验失败、成功运行、保守计入的 30 秒预锁输入/来源工作及 12 秒 CPU 收尾的本轮预算计费为 {v62_cost['charged_v62_active_seconds']:.3f} / 5400 秒。首个 GPU 尝试在前向前停止，原因是清单 dtype 文字 `torch.float16`/`float16` 表示形式不同、数值哈希与形状相同；修正仅标准化清单字段，未改权重、评测器或模型。该失败已留存并纳入计费。

历史成本与本轮评测成本分开：M1 原生 GPTQ build {build_m1:.3f} 秒，M2 scalar10 build {build_m2:.3f} 秒；v61 M3 data-to-finalized calibration {v61_cost['data_to_finalized_model_pipeline_seconds']:.3f} 秒、包含加载保存/reload 的成功调用 {v61_cost['full_invocation_wall_seconds']:.3f} 秒，连同失败 attempt 共 {v61_cost['total_all_attempt_wall_seconds']:.3f} 秒；v56 M1 修复全模型正式阶段 {v56_repair['elapsed_seconds']:.3f} 秒。v55 新修复的 block2–23 为 {v55_budget['valid_new_layer2_to23_block_wall_seconds']:.3f} 秒；完整 24-block 净修复时间因 block0/1 继承恢复无法拆分，保留 NOT_IDENTIFIABLE。v55 的 {v55_budget['final_cumulative_charged_seconds']:.3f} 秒是多版本恢复链累计，不当作算法修复时间。未给出跨方法严格加速比。

内存的测量范围不同。v61 校准峰值来自既有 allocator 与资源采样；v62 推理峰值按模型单独记录 PyTorch allocated/reserved、0.5 秒设备总显存采样和主进程 RSS。设备总显存包含桌面及其他进程，WDDM 下单进程 GPU 显存不可得；采样峰值不是精确瞬时峰值。M1/M2/v55/v56 历史记录没有设备采样或 RSS 时在表内标 `NOT_MEASURED`。

## 边界与结论

WT2 与 C4 fixed-budget 是长期使用的 benchmark，不是新盲测。一次固定模型比较不代表统计显著或跨 seed 稳定；统一重评也不修复历史校准传播证据中的 UNKNOWN。按本轮结果，M3 与 M2+修复呈语料取舍。成本记录显示 M3 完整校准约 2,384 秒；M2+修复只可准确给出已有 block2–23 净修复及恢复链账本，缺少完整 24-block 净时长，不能做公平总成本胜负判断。是否继续修复路线没有跨语料质量或可比完整成本优势支持；本轮不启动后续实验。

## 产物

- checkpoint 来源：`CHECKPOINT_SOURCE_AUDIT.csv`
- v61 步数差异：`V61_STEP_COUNT_AUDIT.json`
- 校准 manifest 与 token hash 口径：`CALIBRATION_HASH_RECONCILIATION.json`
- 原始窗口、输入 token 切片、mask 来源和复算：`evaluation/`
- 统一环境：`PROCESS_ENVIRONMENT.json`
- 成本与内存：`COST_ACCOUNTING.json`、`QUALITY_COST_MEMORY_TABLE.csv`、`COST_SOURCES.csv`
- 协议、冻结锁、失败 attempt、账本：`FROZEN_PROTOCOL.json`、`RUN_LOCK.json`、`attempts/`
- 文件 SHA：`OUTPUT_MANIFEST.csv`
'''
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    reproduce = f'''# v62 reproduction

The completed v62 output is immutable. The exact successful evaluation command was:

```powershell
Set-Location E:\\quant-thesis
.\\.venv\\Scripts\\python.exe -B .\\results\\v62_m3_vs_repaired_quality_cost\\run_v62_eval.py
```

The runner refuses to overwrite a `COMPLETE` result. To reproduce into a fresh independent directory, copy `preflight_v62.py`, `run_v62_eval.py`, and the locked protocol/source audit into a new directory under `results/`, update the two scripts' `OUT` constants only, rerun the CPU preflight, then run the evaluation runner once. Keep source checkpoint paths, evaluator, token manifests, model order and all model/evaluation settings unchanged. Do not reuse any partial windows across process ids. The CPU-only audit command for this completed output is:

```powershell
.\\.venv\\Scripts\\python.exe -B .\\results\\v62_m3_vs_repaired_quality_cost\\finalize_v62.py
```
'''
    (OUT / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")

    # Hash every final output except the manifest itself, then list itself as non-self-hashable.
    output_rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "OUTPUT_MANIFEST.csv":
            output_rows.append({"path": str(p.relative_to(OUT)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha_file(p), "self_hash": False})
    output_rows.append({"path": "OUTPUT_MANIFEST.csv", "bytes": "SELF_REFERENCE_OMITTED", "sha256": "SELF_HASH_NOT_APPLICABLE", "self_hash": True})
    write_csv(OUT / "OUTPUT_MANIFEST.csv", output_rows)
    # Verify requested delivery files exist and output manifest resolves all other files.
    required = ["REPORT.md", "FROZEN_PROTOCOL.json", "CHECKPOINT_SOURCE_AUDIT.csv", "V61_STEP_COUNT_AUDIT.json",
                "QUALITY_COMPARISON.csv", "QUALITY_COST_MEMORY_TABLE.csv", "PROCESS_ENVIRONMENT.json", "COST_ACCOUNTING.json", "OUTPUT_MANIFEST.csv", "REPRODUCE.md"]
    missing = [n for n in required if not (OUT / n).is_file()]
    if missing:
        raise RuntimeError(f"required v62 outputs missing: {missing}")
    print(json.dumps({"status": "FINALIZED", "quality": quality, "skipped_steps": skipped,
                      "output_file_count": len(output_rows), "report": str(OUT / "REPORT.md")}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
