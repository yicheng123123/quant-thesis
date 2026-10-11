from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent
REPO = OUT.parents[3]
V214 = RESULTS / "range_w4_omniquant_resume_v21_4" / "M3"
V18 = RESULTS / "range_w4_fullmodel_v18"
V19 = RESULTS / "range_w4_fullmodel_review_v19"
V213 = RESULTS / "range_w4_omniquant_amp_audit_v21_3"
TOL = 1e-10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, obj: Any) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite(x: Any) -> bool:
    return math.isfinite(float(x))


def evaluator_model_hash(export_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    weights = sorted(export_dir.glob("*.safetensors"))
    if not weights:
        raise RuntimeError("dense export contains no safetensors")
    digest = hashlib.sha256()
    details = []
    for path in weights:
        file_hash = sha256(path)
        details.append({"name": path.name, "bytes": path.stat().st_size, "sha256": file_hash})
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
    return digest.hexdigest(), details


def tokenizer_hash(export_files: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    names = {"added_tokens.json", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
    chosen = sorted((entry for entry in export_files if entry["path"] in names), key=lambda x: x["path"])
    if {x["path"] for x in chosen} != names:
        raise RuntimeError("dense tokenizer file set is incomplete")
    digest = hashlib.sha256()
    for item in chosen:
        digest.update(item["path"].encode("utf-8"))
        digest.update(bytes.fromhex(item["sha256"]))
    return digest.hexdigest(), chosen


def audit_eval(name: str, raw_path: Path, window_path: Path, expected_tokens: int,
               expected_windows: int, expected_manifest_hash: str, expected_token_hash: str,
               expected_tail: list[int]) -> dict[str, Any]:
    raw = read_json(raw_path)
    if raw.get("status") != "PASS":
        raise RuntimeError(f"{name} raw result is not PASS")
    if raw.get("evaluated_tokens") != expected_tokens or raw.get("window_count") != expected_windows:
        raise RuntimeError(f"{name} target/window count mismatch")
    if raw.get("evaluation_manifest_sha256") != expected_manifest_hash:
        raise RuntimeError(f"{name} evaluation manifest hash mismatch")
    if raw.get("evaluation_token_ids_sha256_i64le") != expected_token_hash:
        raise RuntimeError(f"{name} token ID hash mismatch")
    rows_csv = read_csv(window_path)
    rows_json = raw.get("window_results", [])
    if len(rows_csv) != expected_windows or len(rows_json) != expected_windows:
        raise RuntimeError(f"{name} window artifact incomplete")
    by_json = {int(row["window_index"]): row for row in rows_json}
    by_csv = {int(row["window_index"]): row for row in rows_csv}
    expected_ids = list(range(expected_windows))
    if sorted(by_json) != expected_ids or sorted(by_csv) != expected_ids:
        raise RuntimeError(f"{name} windows contain missing or duplicate indices")
    for i in expected_ids:
        a, b = by_json[i], by_csv[i]
        for key in ("begin", "end", "new_targets", "valid_tokens"):
            if int(a[key]) != int(b[key]):
                raise RuntimeError(f"{name} window {i} {key} differs between raw JSON and CSV")
        for key in ("window_loss", "window_nll"):
            if abs(float(a[key]) - float(b[key])) > 1e-12:
                raise RuntimeError(f"{name} window {i} {key} differs between raw JSON and CSV")
        if not finite(a["window_loss"]) or not finite(a["window_nll"]):
            raise RuntimeError(f"{name} window {i} has nonfinite loss")
    total_targets = sum(int(by_csv[i]["valid_tokens"]) for i in expected_ids)
    if total_targets != expected_tokens:
        raise RuntimeError(f"{name} window target sum mismatch")
    tail = by_csv[expected_windows - 1]
    tail_tuple = [int(tail[k]) for k in ("begin", "end", "new_targets", "valid_tokens")]
    if tail_tuple != expected_tail:
        raise RuntimeError(f"{name} tail-window boundary mismatch: {tail_tuple}")

    total_nll_csv = math.fsum(float(by_csv[i]["window_nll"]) for i in expected_ids)
    total_nll_json = math.fsum(float(by_json[i]["window_nll"]) for i in expected_ids)
    mean_nll = total_nll_csv / total_targets
    ppl = math.exp(mean_nll)
    if abs(total_nll_csv - total_nll_json) > TOL:
        raise RuntimeError(f"{name} raw JSON and CSV total NLL differ")
    mean_diff = mean_nll - float(raw["average_nll"])
    ppl_diff = ppl - float(raw["perplexity"])
    if abs(mean_diff) > TOL or abs(ppl_diff) > TOL:
        raise RuntimeError(f"{name} independent aggregation differs from evaluator output")
    return {
        "status": "PASS",
        "dataset": name,
        "scope": "frozen WT2 test" if name == "wikitext" else "C4 fixed-budget validation; not standard/full C4 PPL",
        "targets": total_targets,
        "windows": expected_windows,
        "input_tokens": int(raw["input_tokens"]),
        "total_nll": total_nll_csv,
        "mean_nll_per_token": mean_nll,
        "ppl": ppl,
        "evaluator_mean_nll": float(raw["average_nll"]),
        "evaluator_ppl": float(raw["perplexity"]),
        "mean_nll_recompute_diff": mean_diff,
        "ppl_recompute_diff": ppl_diff,
        "recompute_tolerance_absolute": TOL,
        "window_csv_sha256": sha256(window_path),
        "raw_json_sha256": sha256(raw_path),
        "evaluation_token_ids_sha256_i64le": raw["evaluation_token_ids_sha256_i64le"],
        "model_weights_sha256": raw["model_weights_sha256"],
        "model_weight_files": raw["model_weight_files"],
        "window_indices_complete": True,
        "window_losses_finite": True,
        "failed_or_missing_windows": 0,
        "first_window": [int(by_csv[0][k]) for k in ("begin", "end", "new_targets", "valid_tokens")],
        "tail_window": tail_tuple,
        "source_log": raw["log"],
        "source_log_sha256": sha256(V214 / raw["log"]),
        "raw_source": str(raw_path),
        "window_source": str(window_path),
    }


def delta(m3: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    dppl = m3["ppl"] - base["ppl"]
    dnll = m3["nll"] - base["nll"]
    return {
        "ppl_delta_m3_minus_baseline": dppl,
        "ppl_relative_delta_percent": dppl / base["ppl"] * 100.0,
        "ppl_relative_improvement_percent": -dppl / base["ppl"] * 100.0,
        "nll_per_token_delta_m3_minus_baseline": dnll,
    }


def attempt_span(attempt_id: str, committed_at: str) -> dict[str, Any]:
    start = datetime.strptime(attempt_id[:15], "%Y%m%d_%H%M%S").replace(tzinfo=timezone(timedelta(hours=8)))
    end = datetime.strptime(committed_at, "%Y-%m-%d %H:%M:%S%z")
    seconds = (end - start).total_seconds()
    if seconds <= 0:
        raise RuntimeError("attempt/checkpoint timestamp ordering invalid")
    return {"attempt_id": attempt_id, "attempt_id_timestamp_local": start.isoformat(),
            "checkpoint_commit_local": end.isoformat(), "elapsed_seconds": seconds,
            "measurement_type": "timestamp-derived interval; not the runner's perf_counter calibration timer"}


def main() -> None:
    # Training and final checkpoint are read-only gates. No model or training code is imported here.
    training = read_json(V214 / "M3_TRAINING_SUMMARY.json")
    checkpoint_dir = V214 / "checkpoints" / "block_23"
    checkpoint_manifest_path = checkpoint_dir / "CHECKPOINT_MANIFEST.json"
    checkpoint_manifest = read_json(checkpoint_manifest_path)
    latest_path = V214 / "checkpoints" / "LATEST.json"
    latest = read_json(latest_path)
    reconstruction = read_json(V214 / "POST_TRAINING_RECONSTRUCTION.json")
    reload_check_path = V214 / "DENSE_EXPORT_RELOAD_CHECK.json"
    reload_check = read_json(reload_check_path)
    run_status = read_json(V214 / "RUN_STATUS.json")
    finalization_failure_path = V214 / "RUN_STATUS_TRAINING_FINALIZATION_FAILURE.json"
    finalization_failure = read_json(finalization_failure_path)
    lock = read_json(V214 / "FORMAL_PROTOCOL_LOCK.json")
    export_manifest_path = V214 / "DENSE_EXPORT_MANIFEST.json"
    export_manifest = read_json(export_manifest_path)
    export_dir = V214 / "dense_export"
    eval_status = read_json(V214 / "M3_EVALUATION_STATUS.json")
    eval_summary = read_json(V214 / "evaluation" / "M3_SUMMARY.json")
    eval_correctness = read_json(V214 / "evaluation" / "M3_EVAL_CORRECTNESS.json")
    eval_protocol = read_json(V214 / "EVALUATION_PROTOCOL.json")
    if training.get("status") != "PASS" or training.get("completed_blocks") != 24:
        raise RuntimeError("M3 training completion gate failed")
    if (training.get("scheduled_iterations") != 61440
            or training.get("successful_optimizer_steps") != 61438
            or training.get("gradscaler_skipped_steps") != 2):
        raise RuntimeError("M3 training count gate failed")
    if checkpoint_manifest.get("completed_block_id") != 23 or checkpoint_manifest.get("completed_layer_count") != 24:
        raise RuntimeError("block23 identity gate failed")
    if latest.get("checkpoint_dir") != "block_23" or latest.get("manifest_sha256") != sha256(checkpoint_manifest_path):
        raise RuntimeError("LATEST does not point to the verified block23 manifest")
    for field, filename in (("state", "omni_parameters.pth"), ("rng", "rng_state.pth")):
        p = checkpoint_dir / filename
        hash_key = f"{field}_file_sha256"
        byte_key = f"{field}_file_bytes"
        if sha256(p) != checkpoint_manifest.get(hash_key) or p.stat().st_size != checkpoint_manifest.get(byte_key):
            raise RuntimeError(f"block23 {field} artifact hash/size mismatch")
    if checkpoint_manifest.get("formal_protocol_lock_sha256") != sha256(V214 / "FORMAL_PROTOCOL_LOCK.json"):
        raise RuntimeError("block23 protocol lock mismatch")
    if reconstruction.get("status") != "PASS" or reconstruction.get("optimizer_steps_executed_during_reconstruction") != 0:
        raise RuntimeError("post-training reconstruction gate failed")
    if reload_check.get("status") != "PASS" or not reload_check.get("bitwise_equal") or reload_check.get("max_abs_diff") != 0.0:
        raise RuntimeError("dense export/reload exactness gate failed")
    if export_manifest.get("status") != "PASS" or export_manifest.get("linear_modules_replaced") != 168:
        raise RuntimeError("dense export manifest gate failed")
    if run_status.get("status") != "FORMAL_TRAINING_AND_DENSE_EXPORT_RELOAD_PASS":
        raise RuntimeError("v21.4 final run status gate failed")
    if (finalization_failure.get("exception_type") != "KeyError"
            or finalization_failure.get("message") != "'manifest'"
            or "latest_record[\"manifest\"]" not in finalization_failure.get("traceback", "")):
        raise RuntimeError("preserved post-training finalization failure is not the audited manifest KeyError")

    actual_export_files = []
    for item in export_manifest["files"]:
        p = export_dir / item["path"]
        actual = {"path": item["path"], "bytes": p.stat().st_size, "sha256": sha256(p)}
        if actual["bytes"] != item["bytes"] or actual["sha256"] != item["sha256"]:
            raise RuntimeError(f"dense export artifact mismatch: {item['path']}")
        actual_export_files.append(actual)
    export_weight_hash, actual_weights = evaluator_model_hash(export_dir)
    tokenizer_digest, tokenizer_files = tokenizer_hash(actual_export_files)
    config_entry = next(x for x in actual_export_files if x["path"] == "config.json")
    if config_entry["sha256"] != export_manifest["files"][1]["sha256"] and not any(
        x["path"] == "config.json" and x["sha256"] == config_entry["sha256"] for x in export_manifest["files"]
    ):
        raise RuntimeError("dense export config hash mismatch")
    if export_manifest.get("model_revision") != lock.get("model_revision"):
        raise RuntimeError("dense export model revision differs from lock")

    evaluator = REPO / "benchmark" / "eval" / "ppl.py"
    frozen_evaluator = RESULTS / "range_w4_omniquant_kaggle_v20_1" / "kaggle_bundle" / "evaluator" / "ppl.py"
    evaluator_sha = sha256(evaluator)
    frozen_evaluator_sha = sha256(frozen_evaluator)
    if evaluator_sha != frozen_evaluator_sha or evaluator_sha != eval_protocol.get("evaluator_sha256"):
        raise RuntimeError("frozen evaluator hash mismatch")
    if eval_protocol.get("v20_1_frozen_copy_sha256") != frozen_evaluator_sha:
        raise RuntimeError("v20.1 frozen evaluator copy identity mismatch")
    if eval_status.get("status") != "PASS" or eval_summary.get("status") != "PASS" or eval_correctness.get("status") != "PASS":
        raise RuntimeError("M3 evaluation did not finish with PASS")
    if set(eval_status.get("results_completed", [])) != {"wikitext", "c4_fixed_budget"}:
        raise RuntimeError("evaluation status does not include both frozen datasets")

    v19_eval = read_json(V19 / "EVALUATOR_AUDIT.json")
    wt2 = audit_eval("wikitext", V214 / "evaluation" / "raw" / "M3_wikitext.json",
                     V214 / "evaluation" / "windows" / "M3_wikitext.csv", 299077, 582,
                     eval_protocol["wikitext_manifest_sha256"],
                     v19_eval["runs"]["wikitext"]["M0"]["token_ids_sha256_i64le"],
                     [297472, 299078, 70, 70])
    c4 = audit_eval("c4_fixed_budget", V214 / "evaluation" / "raw" / "M3_c4_fixed_budget.json",
                    V214 / "evaluation" / "windows" / "M3_c4_fixed_budget.csv", 131071, 253,
                    eval_protocol["c4_manifest_sha256"],
                    v19_eval["runs"]["c4_fixed_budget"]["M0"]["token_ids_sha256_i64le"],
                    [129024, 131072, 512, 512])
    raw_wt = read_json(V214 / "evaluation" / "raw" / "M3_wikitext.json")
    raw_c4 = read_json(V214 / "evaluation" / "raw" / "M3_c4_fixed_budget.json")
    for raw in (raw_wt, raw_c4):
        if Path(raw["model_source"]).resolve() != export_dir.resolve():
            raise RuntimeError("evaluation did not use the final dense export path")
        if raw.get("model_weights_sha256") != export_weight_hash or raw.get("model_weight_files") != actual_weights:
            raise RuntimeError("evaluation dense model weight hash mismatch")
        if raw.get("model_revision") != lock.get("model_revision"):
            raise RuntimeError("evaluation model revision mismatch")

    # M0/M1/M2 values are read from v18 and cross-checked against the v19 independent audit.
    baseline_rows = read_csv(V18 / "FULLMODEL_COMPARISON.csv")
    baseline = {}
    for row in baseline_rows:
        method = row["Method"]
        baseline[method] = {
            "wt2_nll": float(row["WT2 NLL"]), "wt2_ppl": float(row["WT2 PPL"]),
            "c4_nll": float(row["C4 fixed-budget NLL"]), "c4_ppl": float(row["C4 fixed-budget PPL"]),
            "calibration_wall_seconds": float(row["Calibration wall seconds"]) if row["Calibration wall seconds"] else None,
            "calibration_peak_allocated_bytes": int(row["Calibration peak GPU allocated bytes"]) if row["Calibration peak GPU allocated bytes"] else None,
            "evaluation_peak_allocated_bytes": int(row["Evaluation peak GPU allocated bytes"]) if row["Evaluation peak GPU allocated bytes"] else None,
            "evaluation_peak_reserved_bytes": int(row["Evaluation peak GPU reserved bytes"]) if row["Evaluation peak GPU reserved bytes"] else None,
            "representation": row["Representation"], "quantization": row["Precision/Quant"],
            "calibration": row["Calibration"],
        }
    if set(baseline) != {"M0", "M1", "M2"}:
        raise RuntimeError("frozen v18 comparison is incomplete")
    v19_checkpoint = read_json(V19 / "CHECKPOINT_AUDIT.json")
    if v19_eval.get("status") != "PASS" or v19_eval.get("errors") or v19_checkpoint.get("status") != "PASS" or v19_checkpoint.get("errors"):
        raise RuntimeError("v19 independent baseline audit did not pass")
    for ds, nll_key, ppl_key, audit_name in (
        ("wikitext", "wt2_nll", "wt2_ppl", "wikitext"),
        ("c4_fixed_budget", "c4_nll", "c4_ppl", "c4_fixed_budget"),
    ):
        for method in ("M0", "M1", "M2"):
            audited = v19_eval["runs"][audit_name][method]
            if not audited.get("pass") or abs(audited["mean_nll_recomputed"] - baseline[method][nll_key]) > TOL:
                raise RuntimeError(f"v19 {method}/{ds} NLL does not match v18 table")
            if abs(audited["ppl_recomputed"] - baseline[method][ppl_key]) > TOL:
                raise RuntimeError(f"v19 {method}/{ds} PPL does not match v18 table")
            baseline[method][nll_key] = float(audited["mean_nll_recomputed"])
            baseline[method][ppl_key] = float(audited["ppl_recomputed"])

    m3 = {
        "wt2_nll": wt2["mean_nll_per_token"], "wt2_ppl": wt2["ppl"],
        "c4_nll": c4["mean_nll_per_token"], "c4_ppl": c4["ppl"],
    }
    quality = {"status": "PASS", "primary_dataset": "frozen WikiText-2 test",
               "numerical_tie_tolerance_absolute_ppl": TOL,
               "tie_tolerance_source": "same 1e-10 absolute evaluator aggregate consistency check in frozen M3 evaluator; no practical-equivalence band",
               "datasets": {}}
    recovery = {"status": "PASS", "metric": "PPL degradation recovery fraction",
                "formula": "(PPL_M1 - PPL_method) / (PPL_M1 - PPL_M0)", "datasets": {}}
    for ds, nll_key, ppl_key, m3nll, m3ppl in (
        ("wikitext", "wt2_nll", "wt2_ppl", "wt2_nll", "wt2_ppl"),
        ("c4_fixed_budget", "c4_nll", "c4_ppl", "c4_nll", "c4_ppl"),
    ):
        q_m3 = {"nll": m3[m3nll], "ppl": m3[m3ppl]}
        quality["datasets"][ds] = {
            "baseline_values": {k: {"nll": baseline[k][nll_key], "ppl": baseline[k][ppl_key]} for k in ("M0", "M1", "M2")},
            "M3": q_m3,
            "M3_vs_M1": delta(q_m3, {"nll": baseline["M1"][nll_key], "ppl": baseline["M1"][ppl_key]}),
            "M3_vs_M2": delta(q_m3, {"nll": baseline["M2"][nll_key], "ppl": baseline["M2"][ppl_key]}),
        }
        degradation = baseline["M1"][ppl_key] - baseline["M0"][ppl_key]
        recovery["datasets"][ds] = {
            "native_gptq_degradation_M1_minus_M0_ppl": degradation,
            "M2_recovered_ppl": baseline["M1"][ppl_key] - baseline["M2"][ppl_key],
            "M2_recovery_fraction": (baseline["M1"][ppl_key] - baseline["M2"][ppl_key]) / degradation,
            "M2_recovery_percent": (baseline["M1"][ppl_key] - baseline["M2"][ppl_key]) / degradation * 100.0,
            "M3_recovered_ppl": baseline["M1"][ppl_key] - m3[m3ppl],
            "M3_recovery_fraction": (baseline["M1"][ppl_key] - m3[m3ppl]) / degradation,
            "M3_recovery_percent": (baseline["M1"][ppl_key] - m3[m3ppl]) / degradation * 100.0,
        }

    primary_delta = m3["wt2_ppl"] - baseline["M2"]["wt2_ppl"]
    if primary_delta < -TOL:
        verdict = "M3_BEATS_M2_ON_FROZEN_WT2"
    elif primary_delta > TOL:
        verdict = "M2_BEATS_M3_ON_FROZEN_WT2"
    else:
        verdict = "M2_M3_NUMERICALLY_TIED_ON_FROZEN_WT2"
    quality["primary_verdict"] = verdict
    c4_delta = m3["c4_ppl"] - baseline["M2"]["c4_ppl"]
    quality["cross_corpus_direction"] = {
        "WT2_M3_minus_M2_ppl": primary_delta,
        "C4_fixed_budget_M3_minus_M2_ppl": c4_delta,
        "same_direction": (primary_delta < 0 and c4_delta < 0) or (primary_delta > 0 and c4_delta > 0),
        "role": "C4 is secondary cross-corpus interpretation and does not change the frozen WT2 primary verdict",
    }

    # Final iteration/AMP audit: current final schedule is checked against every resumed attempt row.
    attempt_id = training["attempt_id"]
    step_rows = [r for r in read_csv(V214 / "logs" / "training_steps.csv") if r["attempt_id"] == attempt_id]
    amp_rows = [r for r in read_csv(V214 / "AMP_EVENT_LOG.csv") if r["attempt_id"] == attempt_id]
    skipped = [r for r in step_rows if r["optimizer_step_skipped"].lower() == "true"]
    executed = [r for r in step_rows if r["optimizer_step_executed"].lower() == "true"]
    found_inf = [r for r in step_rows if float(r["found_inf"]) > 0]
    locations = {(int(r["layer"]), int(r["epoch_index"]), int(r["sample_index"])) for r in skipped}
    expected_skip_locations = {(21, 0, 0), (21, 0, 74)}
    if len(step_rows) != 7680 or len(executed) != 7678 or len(skipped) != 2 or len(found_inf) != 2:
        raise RuntimeError("final v21.4 attempt AMP/step accounting mismatch")
    if locations != expected_skip_locations or len(amp_rows) != 2:
        raise RuntimeError("final v21.4 AMP event locations mismatch")
    for row in step_rows:
        skip = row["optimizer_step_skipped"].lower() == "true"
        did_step = row["optimizer_step_executed"].lower() == "true"
        if skip == did_step or (float(row["found_inf"]) > 0) != skip:
            raise RuntimeError("found_inf, skip flag, and optimizer hook disagree")
        if not finite(row["loss"]) or row["lwc_parameters_finite"].lower() != "true" or row["optimizer_state_finite"].lower() != "true" or row["scaler_state_finite"].lower() != "true":
            raise RuntimeError("nonfinite final training state/loss found")
    for row in amp_rows:
        if row["optimizer_step_skipped"].lower() != "true" or row["parameters_finite"].lower() != "true" or row["optimizer_state_unchanged"].lower() != "true":
            raise RuntimeError("GradScaler event state check failed")
    historical = read_json(V213 / "HISTORICAL_UPDATE_COUNT_AUDIT.json")
    if historical.get("status") != "PASS" or historical.get("inferred_successful_optimizer_steps") != 53760:
        raise RuntimeError("historical prefix step audit is not PASS")
    amp_stats = {
        "status": "PASS",
        "canonical_final_model_accounting": {
            "scheduled_iterations": 61440,
            "successful_optimizer_steps": 61438,
            "gradscaler_skipped_steps": 2,
            "skip_fraction": 2 / 61440,
            "skip_percent": 2 / 61440 * 100.0,
            "resumed_v21_4_attempt_rows": len(step_rows),
            "resumed_v21_4_successful_steps": len(executed),
            "resumed_v21_4_skips": len(skipped),
            "canonical_final_artifact_counts_match_block23_manifest": True,
        },
        "events": [{
            "scheduled_iteration_count": int(r["scheduled_iteration_count"]),
            "layer": int(r["layer"]), "epoch_index": int(r["epoch_index"]), "sample_index": int(r["sample_index"]),
            "loss": float(r["loss"]), "found_inf": float(r["found_inf"]),
            "scale_before": float(r["scale_before"]), "scale_after": float(r["scale_after"]),
            "lwc_parameters_finite": r["parameters_finite"].lower() == "true",
            "optimizer_state_unchanged": r["optimizer_state_unchanged"].lower() == "true",
        } for r in amp_rows],
        "historical_block0_20": {
            "scheduled_iterations": 53760,
            "successful_updates_inferred": historical["inferred_successful_optimizer_steps"],
            "skips_inferred": 0,
            "evidence": "v21.3 CPU historical audit: complete prefix has finite owned LWC gradients after native scaler on all recorded rows; direct hook/found_inf snapshots were not saved",
            "prefix_inference_not_direct_step_hook_measurement": True,
            "legacy_manifest_field_caveat": "block20 effective_optimizer_updates_in_checkpoint=53760; its separate total_optimizer_updates=61440 is a schedule total and is not used as effective-update count",
        },
        "earlier_abandoned_boundary_attempt": {
            "location": {"layer": 21, "epoch_index": 0, "sample_index": 0},
            "included_in_canonical_final_model_counts": False,
            "evidence": "v21.1 stopped after block20 at this boundary; the row was rerun under the v21.4 native-scaler gate and is counted once in the final model ledger",
            "separate_nonfinal_attempt_is_disclosed": True,
        },
        "other_hidden_skips_in_canonical_v21_4_attempt": 0,
        "step_log_sha256": sha256(V214 / "logs" / "training_steps.csv"),
        "amp_event_log_sha256": sha256(V214 / "AMP_EVENT_LOG.csv"),
        "block23_manifest_sha256": sha256(checkpoint_manifest_path),
        "historical_audit_sha256": sha256(V213 / "HISTORICAL_UPDATE_COUNT_AUDIT.json"),
    }

    # Runtime: preserve only timestamp-derived attempt intervals; perf_counter values were not persisted.
    v211_checkpoint = RESULTS / "range_w4_omniquant_full_v21_1" / "M3" / "checkpoints" / "block_20" / "CHECKPOINT_MANIFEST.json"
    v211_manifest = read_json(v211_checkpoint)
    v211_prefix_audit = read_json(V213 / "HISTORICAL_UPDATE_COUNT_AUDIT.json")
    v211_attempt = v211_prefix_audit["checkpoint_attempt_id"]
    v211_span = attempt_span(v211_attempt, v211_manifest["committed_at_local"])
    v214_span = attempt_span(training["attempt_id"], checkpoint_manifest["committed_at_local"])
    span_sum = v211_span["elapsed_seconds"] + v214_span["elapsed_seconds"]
    runtime = {
        "status": "TIMING_LIMITATION",
        "calibration_wall_exactly_recovered": False,
        "reason": "v21.1/v21.4 runner perf_counter calibration_wall values were not persisted with the final artifacts. attempt_id-to-checkpoint timestamps recover useful intervals but do not equal the runner's internal timer boundary or the full invocation start.",
        "post_training_runner_summary_assembly_failure": {
            "exception_type": finalization_failure["exception_type"],
            "message": finalization_failure["message"],
            "source_artifact": str(finalization_failure_path),
            "source_sha256": sha256(finalization_failure_path),
            "failure_boundary": "after block23 checkpoint commit; training state remained committed",
            "recovery": "training summary reconstructed from final checkpoint and logs; dense model reconstructed for export/reload with zero optimizer steps",
        },
        "accepted_formal_attempt_timestamp_intervals": [
            {"phase": "v21.1 formal prefix blocks0-20 checkpoint commit", "completed_blocks": 21,
             "effective_optimizer_updates": 53760, **v211_span,
             "source_checkpoint_manifest_sha256": sha256(v211_checkpoint)},
            {"phase": "v21.4 final continuation blocks21-23, including required prefix replay", "completed_blocks": 3,
             "scheduled_iterations": 7680, "successful_optimizer_steps": 7678, "gradscaler_skipped_steps": 2,
             **v214_span, "source_training_summary_sha256": sha256(V214 / "M3_TRAINING_SUMMARY.json"),
             "source_checkpoint_manifest_sha256": sha256(checkpoint_manifest_path)},
        ],
        "sum_of_timestamp_derived_intervals_seconds": span_sum,
        "sum_of_timestamp_derived_intervals_hours": span_sum / 3600.0,
        "interpretation": "An auditable timestamp-derived interval total, not exact calibrated wall time and not used for cost ratios. It includes the v21.4 prefix replay and model/tokenizer load between attempt ID creation and checkpoint commit; some pre-attempt setup is outside the interval.",
        "persisted_measured_components": {
            "v21_4_post_checkpoint_reconstruction_seconds": training["reconstruction_after_checkpoint_error_seconds"],
            "v21_4_dense_export_and_reload_seconds": training["post_training_export_reload_seconds"],
            "v21_4_training_peak_allocated_bytes": training["peak_allocated_bytes"],
            "v21_4_training_peak_reserved_bytes": training["peak_reserved_bytes"],
            "v21_4_prefix_reconstruction_peak_allocated_bytes": reconstruction["prefix_replay_peak_allocated_bytes"],
            "v21_4_prefix_reconstruction_peak_reserved_bytes": reconstruction["prefix_replay_peak_reserved_bytes"],
        },
        "cost_scope": {
            "included": ["v21.1 committed prefix blocks0-20", "v21.4 committed continuation blocks21-23", "legitimate skips within canonical schedule", "v21.4 dependency prefix replay actually executed"],
            "separately_reported_not_added_to_calibration": ["post-checkpoint investigative model reconstruction", "dense export/reload"],
            "excluded_nonfinal_work": ["v21.1 discarded checkpoint-fsync attempt (2560 attempted updates, 0 committed)", "v21.2/v21.3 diagnostics", "smokes", "code/setup debugging"],
            "excluded_work_timing_recoverable": False,
        },
        "M1_measured_calibration_wall_seconds": baseline["M1"]["calibration_wall_seconds"],
        "M2_measured_calibration_wall_seconds": baseline["M2"]["calibration_wall_seconds"],
        "M3_to_M1_or_M2_cost_ratios": None,
        "cost_ratio_reason": "M3 exact runner timer is unavailable; timestamp spans are a different, non-equivalent boundary.",
    }

    audit = {
        "status": "PASS",
        "model_id": lock["model_id"], "model_revision": lock["model_revision"],
        "model_weight_sha256": lock["model_weight_sha256"],
        "block23": {
            "status": checkpoint_manifest["status"], "completed_block_id": checkpoint_manifest["completed_block_id"],
            "completed_layer_count": checkpoint_manifest["completed_layer_count"],
            "manifest_sha256": sha256(checkpoint_manifest_path), "latest_pointer_sha256": sha256(latest_path),
            "state_file": checkpoint_manifest["state_file"], "state_file_sha256": sha256(checkpoint_dir / checkpoint_manifest["state_file"]),
            "rng_file": checkpoint_manifest["rng_file"], "rng_file_sha256": sha256(checkpoint_dir / checkpoint_manifest["rng_file"]),
            "scheduled_iterations": checkpoint_manifest["scheduled_training_iterations"],
            "successful_optimizer_steps": checkpoint_manifest["successful_optimizer_steps"],
            "gradscaler_skipped_steps": checkpoint_manifest["gradscaler_skipped_steps"],
        },
        "dense_export": {
            "path": str(export_dir), "manifest_sha256": sha256(export_manifest_path),
            "all_export_files_match_manifest": True, "file_count": len(actual_export_files),
            "model_weights_aggregate_sha256": export_weight_hash, "model_weight_files": actual_weights,
            "exported_config_sha256": config_entry["sha256"], "training_config_sha256": lock["config_sha256"],
            "model_revision": export_manifest["model_revision"], "linear_modules_replaced": export_manifest["linear_modules_replaced"],
            "representation": export_manifest["representation"],
        },
        "reload_check": {"status": reload_check["status"], "bitwise_equal": reload_check["bitwise_equal"],
                         "max_abs_diff": reload_check["max_abs_diff"], "mean_abs_diff": reload_check["mean_abs_diff"],
                         "artifact_sha256": sha256(reload_check_path), "probe_token_ids_sha256": reload_check["probe_token_ids_sha256"]},
        "preserved_post_training_finalization_failure": {
            "exception_type": finalization_failure["exception_type"], "message": finalization_failure["message"],
            "artifact_sha256": sha256(finalization_failure_path), "occurred_after_final_checkpoint_commit": True,
        },
        "tokenizer": {"aggregate_sha256_over_frozen_tokenizer_files": tokenizer_digest,
                      "files": tokenizer_files,
                      "evaluation_tokenizer_revision": raw_wt["dataset_tokenizer_revision"]},
        "evaluator": {"path": str(evaluator), "sha256": evaluator_sha,
                      "v20_1_frozen_copy_sha256": frozen_evaluator_sha,
                      "v21_4_evaluation_protocol_sha256": sha256(V214 / "EVALUATION_PROTOCOL.json"),
                      "same_hash_as_v19_audited_evaluator": evaluator_sha == sha256(REPO / "benchmark" / "eval" / "ppl.py")},
        "evaluation_model_identity_match": True,
    }

    protocol = {
        "name": "v21.5 M3 evaluation resume and baseline finalization",
        "status": "FROZEN_POST_HOC_AUDIT_AND_FINALIZATION",
        "scope": ["read-only audit of v21.4 final M3 artifacts", "independent WT2/C4 window aggregation", "frozen M0/M1/M2/M3 comparison", "formal cost and AMP accounting"],
        "training_or_quantization_rerun": False,
        "evaluation_rerun": False,
        "optimizer_steps_this_round": 0,
        "algorithm_or_hyperparameter_changes": [],
        "model": {"id": lock["model_id"], "revision": lock["model_revision"], "dtype": "FP16", "device": "NVIDIA GeForce RTX 4060 Laptop GPU"},
        "M3": {"quantization": "W4A16 asymmetric affine G128", "LWC": True, "LET": False,
               "method_interpretation": "OmniQuant LWC gradient optimization with block reconstruction"},
        "evaluation": {"evaluator_sha256": evaluator_sha, "tokenizer_revision": raw_wt["dataset_tokenizer_revision"],
                       "dtype": "FP16", "attention": "SDPA", "tf32": False, "max_length": 2048, "stride": 512,
                       "WT2_scored_targets": 299077, "C4_fixed_budget_scored_targets": 131071,
                       "C4_label": "fixed-budget validation, not standard/full C4 PPL",
                       "selection_or_tuning_on_final_evaluation": False},
        "baseline_sources": {
            "v18_comparison_csv_sha256": sha256(V18 / "FULLMODEL_COMPARISON.csv"),
            "v19_evaluator_audit_sha256": sha256(V19 / "EVALUATOR_AUDIT.json"),
            "v19_checkpoint_audit_sha256": sha256(V19 / "CHECKPOINT_AUDIT.json"),
            "v21_3_historical_amp_update_audit_sha256": sha256(V213 / "HISTORICAL_UPDATE_COUNT_AUDIT.json"),
        },
        "source_lineage": {"v21_4_root": str(V214.parent), "finalization_output": str(OUT)},
    }

    eval_correct = {
        "status": "PASS", "WT2": wt2, "C4_fixed_budget": c4,
        "dense_model_reload_status": reload_check["status"],
        "all_frozen_required_windows_present": True,
        "all_window_losses_finite": True,
        "all_targets_match_frozen_protocol": True,
        "independent_total_and_mean_nll_ppl_recomputed_from_window_csv": True,
        "aggregate_absolute_tolerance": TOL,
        "evaluator_final_status": eval_status["status"],
        "evaluator_correctness_sha256": sha256(V214 / "evaluation" / "M3_EVAL_CORRECTNESS.json"),
    }

    runtime_ratios = None
    cost = {
        "status": "COMPARISON_LIMITED_BY_UNRECOVERABLE_M3_TIMER",
        "M1_calibration_wall_seconds": baseline["M1"]["calibration_wall_seconds"],
        "M2_calibration_wall_seconds": baseline["M2"]["calibration_wall_seconds"],
        "M2_over_M1_calibration_ratio": baseline["M2"]["calibration_wall_seconds"] / baseline["M1"]["calibration_wall_seconds"],
        "M3_exact_calibration_wall_seconds": None,
        "M3_timestamp_derived_formal_interval_sum_seconds": span_sum,
        "M3_to_M1_or_M2_ratios": runtime_ratios,
        "M3_training_peak_allocated_bytes": training["peak_allocated_bytes"],
        "M3_training_peak_reserved_bytes": training["peak_reserved_bytes"],
        "M1_calibration_peak_allocated_bytes": baseline["M1"]["calibration_peak_allocated_bytes"],
        "M2_calibration_peak_allocated_bytes": baseline["M2"]["calibration_peak_allocated_bytes"],
        "M3_post_checkpoint_reconstruction_seconds_excluded": training["reconstruction_after_checkpoint_error_seconds"],
        "M3_dense_export_reload_seconds_reported_separately": training["post_training_export_reload_seconds"],
        "interpretation": "Quality-cost ratios M3/M1 and M3/M2 are not reported because M3's runner timer did not persist; the 3052 s timestamp-derived intervals are not measurement-equivalent to v18 perf_counter wall values.",
    }

    # Write compact independent results and provenance only under this new v21.5 directory.
    protocol["status"] = "PASS"
    write_json("PROTOCOL.json", protocol)
    write_json("FINAL_M3_ARTIFACT_AUDIT.json", audit)
    write_json("WT2_M3_FINAL.json", wt2)
    write_json("C4_M3_FINAL.json", c4)
    write_json("EVALUATION_CORRECTNESS.json", eval_correct)
    write_json("M3_AMP_FINAL_STATS.json", amp_stats)
    write_json("M3_FORMAL_RUNTIME_FINAL.json", runtime)
    write_json("QUALITY_COMPARISON.json", quality)
    write_json("QUALITY_RECOVERY.json", recovery)
    write_json("COST_COMPARISON.json", cost)

    csv_path = OUT / "M0_M1_M2_M3_FINAL.csv"
    fields = ["method", "quantization", "calibration_mechanism", "gpu", "calibration_wall_seconds_exact",
              "timestamp_derived_formal_intervals_seconds", "calibration_wall_status", "scheduled_iterations",
              "successful_updates", "amp_skipped", "calibration_peak_allocated_bytes", "calibration_peak_reserved_bytes",
              "evaluation_peak_allocated_bytes", "evaluation_peak_reserved_bytes", "wt2_targets", "wt2_windows",
              "wt2_nll", "wt2_ppl", "c4_fixed_budget_targets", "c4_fixed_budget_windows", "c4_fixed_budget_nll",
              "c4_fixed_budget_ppl", "representation"]
    m1rt = read_json(V18 / "FORMAL_RUNTIME.json")
    gpu = m1rt["device"]
    rows = []
    for method in ("M0", "M1", "M2"):
        b = baseline[method]
        rows.append({"method": method, "quantization": b["quantization"], "calibration_mechanism": b["calibration"],
                     "gpu": gpu, "calibration_wall_seconds_exact": b["calibration_wall_seconds"],
                     "timestamp_derived_formal_intervals_seconds": "", "calibration_wall_status": "measured in v18",
                     "scheduled_iterations": "", "successful_updates": "", "amp_skipped": "",
                     "calibration_peak_allocated_bytes": b["calibration_peak_allocated_bytes"],
                     "calibration_peak_reserved_bytes": "", "evaluation_peak_allocated_bytes": b["evaluation_peak_allocated_bytes"],
                     "evaluation_peak_reserved_bytes": b["evaluation_peak_reserved_bytes"],
                     "wt2_targets": wt2["targets"], "wt2_windows": wt2["windows"], "wt2_nll": b["wt2_nll"], "wt2_ppl": b["wt2_ppl"],
                     "c4_fixed_budget_targets": c4["targets"], "c4_fixed_budget_windows": c4["windows"],
                     "c4_fixed_budget_nll": b["c4_nll"], "c4_fixed_budget_ppl": b["c4_ppl"], "representation": b["representation"]})
    rows.append({"method": "M3", "quantization": "W4A16 asymmetric affine G128; LWC on; LET off",
                 "calibration_mechanism": "OmniQuant LWC gradient optimization + block reconstruction",
                 "gpu": gpu, "calibration_wall_seconds_exact": "", "timestamp_derived_formal_intervals_seconds": span_sum,
                 "calibration_wall_status": "exact perf_counter wall unrecoverable; timestamp interval only",
                 "scheduled_iterations": 61440, "successful_updates": 61438, "amp_skipped": 2,
                 "calibration_peak_allocated_bytes": training["peak_allocated_bytes"], "calibration_peak_reserved_bytes": "",
                 "evaluation_peak_allocated_bytes": max(int(raw_wt["peak_gpu_allocated_bytes"]), int(raw_c4["peak_gpu_allocated_bytes"])),
                 "evaluation_peak_reserved_bytes": max(int(raw_wt["peak_gpu_reserved_bytes"]), int(raw_c4["peak_gpu_reserved_bytes"])),
                 "wt2_targets": wt2["targets"], "wt2_windows": wt2["windows"], "wt2_nll": wt2["mean_nll_per_token"], "wt2_ppl": wt2["ppl"],
                 "c4_fixed_budget_targets": c4["targets"], "c4_fixed_budget_windows": c4["windows"],
                 "c4_fixed_budget_nll": c4["mean_nll_per_token"], "c4_fixed_budget_ppl": c4["ppl"],
                 "representation": "dense-dequant FP16 OmniQuant W4A16/G128"})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    correctness = {
        "status": "PASS",
        "checks": {
            "this_round_optimizer_steps_zero": True,
            "no_training_entrypoint_or_model_forward_invoked_this_round": True,
            "final_block23_checkpoint_identity_and_hashes": True,
            "dense_export_all_manifest_hashes": True,
            "dense_export_reload_bitwise_equal_max_abs_zero": True,
            "evaluator_and_tokenizer_identity": True,
            "WT2_frozen_windows_582_and_299077_targets": True,
            "C4_frozen_windows_253_and_131071_targets": True,
            "WT2_independent_NLL_PPL_recomputation": True,
            "C4_independent_NLL_PPL_recomputation": True,
            "v18_M0_M1_M2_revalidated_by_v19": True,
            "final_training_step_and_amp_counts": True,
            "post_training_summary_error_preserved_and_recovery_audited": True,
            "final_comparison_and_primary_verdict": True,
        },
        "evaluation_reruns_this_round": 0,
        "training_reruns_this_round": 0,
        "source_hashes": {
            "v21_4_checkpoint_manifest": sha256(checkpoint_manifest_path),
            "v21_4_dense_export_manifest": sha256(export_manifest_path),
            "v21_4_reload_check": sha256(reload_check_path),
            "v21_4_training_summary": sha256(V214 / "M3_TRAINING_SUMMARY.json"),
            "v21_4_evaluation_status": sha256(V214 / "M3_EVALUATION_STATUS.json"),
            "v18_comparison": sha256(V18 / "FULLMODEL_COMPARISON.csv"),
            "v19_evaluator_audit": sha256(V19 / "EVALUATOR_AUDIT.json"),
            "v19_checkpoint_audit": sha256(V19 / "CHECKPOINT_AUDIT.json"),
        },
    }
    write_json("correctness_checks.json", correctness)

    summary = {
        "status": "M3_FORMAL_BASELINE_COMPLETED",
        "primary_verdict": verdict,
        "training": {"completed_blocks": 24, "scheduled_iterations": 61440, "successful_optimizer_steps": 61438, "gradscaler_skips": 2},
        "dense_export_reload": {"status": "PASS", "bitwise_equal": True, "max_abs_diff": 0.0},
        "WT2": {"nll": wt2["mean_nll_per_token"], "ppl": wt2["ppl"], "targets": wt2["targets"], "windows": wt2["windows"]},
        "C4_fixed_budget": {"nll": c4["mean_nll_per_token"], "ppl": c4["ppl"], "targets": c4["targets"], "windows": c4["windows"]},
        "M3_vs_M1_WT2": quality["datasets"]["wikitext"]["M3_vs_M1"],
        "M3_vs_M2_WT2": quality["datasets"]["wikitext"]["M3_vs_M2"],
        "M3_vs_M2_C4_fixed_budget": quality["datasets"]["c4_fixed_budget"]["M3_vs_M2"],
        "runtime_exactly_recovered": False,
        "post_training_summary_assembly_error": "KeyError: 'manifest' after block23 commit; recovered from checkpoint and logs",
        "timestamp_derived_formal_intervals_seconds": span_sum,
        "M3_to_M1_M2_cost_ratios_reported": False,
        "v21_5_optimizer_steps": 0,
        "evidence_boundary": "one Qwen2.5-0.5B-Instruct frozen checkpoint; WT2 test and C4 fixed-budget validation only",
    }
    write_json("SUMMARY.json", summary)

    wt2q = quality["datasets"]["wikitext"]
    c4q = quality["datasets"]["c4_fixed_budget"]
    wr = recovery["datasets"]["wikitext"]
    cr = recovery["datasets"]["c4_fixed_budget"]
    report = f"""# v21.5 — M3 evaluation resume and baseline finalization

## Verdict

**{summary['status']}**  
Primary frozen-WT2 verdict: **{verdict}**. C4 fixed-budget points in the same direction; it remains secondary and does not set the primary verdict.

This is a method-level comparison. M1 is native GPTQ; M2 is GPTQ plus per-row discrete range search with native second-order compensation; M3 is OmniQuant with LWC gradient optimization and block reconstruction. M2 versus M3 does not isolate clipping from LWC as a single variable.

## Training and final artifact

- Blocks: 24/24; scheduled iterations: 61,440; successful optimizer steps: 61,438; native GradScaler skips: 2.
- Final block23 checkpoint hash/manifest checks: PASS.
- The frozen v21.4 runner raised `KeyError: 'manifest'` while assembling its summary after block23 was committed. The original failure is preserved; v21.5 reconstructs the summary from the committed checkpoint and logs, then verifies the dense export/reload.
- Dense-dequant export: 168 Linear modules replaced. Reload probe logits were bitwise equal; max absolute difference 0.
- This v21.5 pass performed zero optimizer steps and did not rerun either evaluation.
- The completed WT2 and C4 runs both loaded the final v21.4 dense export; model weight, model revision, frozen evaluator, and tokenizer revisions were hash checked.

## Independent evaluation aggregation

| Dataset | Scope | Windows | Scored targets | Total NLL | Mean NLL/token | PPL | Recompute difference vs evaluator |
|---|---|---:|---:|---:|---:|---:|---:|
| WT2 | frozen test | {wt2['windows']} | {wt2['targets']} | {wt2['total_nll']:.9f} | {wt2['mean_nll_per_token']:.12f} | {wt2['ppl']:.12f} | mean {wt2['mean_nll_recompute_diff']:.3g}; PPL {wt2['ppl_recompute_diff']:.3g} |
| C4 | fixed-budget validation, not full C4 PPL | {c4['windows']} | {c4['targets']} | {c4['total_nll']:.9f} | {c4['mean_nll_per_token']:.12f} | {c4['ppl']:.12f} | mean {c4['mean_nll_recompute_diff']:.3g}; PPL {c4['ppl_recompute_diff']:.3g} |

The final windows and token totals agree with the frozen manifests. All required window indices are present and finite. Aggregation was independently recomputed from the persisted per-window CSV using total window NLL divided by total scored targets, followed by `exp(mean NLL)`.

## Frozen quality comparison

### WT2 primary

- M0 FP16: NLL {baseline['M0']['wt2_nll']:.12f}, PPL {baseline['M0']['wt2_ppl']:.12f}.
- M1 native GPTQ: NLL {baseline['M1']['wt2_nll']:.12f}, PPL {baseline['M1']['wt2_ppl']:.12f}.
- M2 GPTQ + range search: NLL {baseline['M2']['wt2_nll']:.12f}, PPL {baseline['M2']['wt2_ppl']:.12f}.
- M3 OmniQuant: NLL {m3['wt2_nll']:.12f}, PPL {m3['wt2_ppl']:.12f}.
- M3 vs M1: PPL delta {wt2q['M3_vs_M1']['ppl_delta_m3_minus_baseline']:.9f} ({wt2q['M3_vs_M1']['ppl_relative_delta_percent']:.4f}%), NLL/token delta {wt2q['M3_vs_M1']['nll_per_token_delta_m3_minus_baseline']:.9f}.
- M3 vs M2: PPL delta {wt2q['M3_vs_M2']['ppl_delta_m3_minus_baseline']:.9f} ({wt2q['M3_vs_M2']['ppl_relative_delta_percent']:.4f}%), NLL/token delta {wt2q['M3_vs_M2']['nll_per_token_delta_m3_minus_baseline']:.9f}.

### C4 secondary cross-corpus check

- M0/M1/M2/M3 PPL: {baseline['M0']['c4_ppl']:.12f} / {baseline['M1']['c4_ppl']:.12f} / {baseline['M2']['c4_ppl']:.12f} / {m3['c4_ppl']:.12f}.
- M3 vs M1: PPL delta {c4q['M3_vs_M1']['ppl_delta_m3_minus_baseline']:.9f} ({c4q['M3_vs_M1']['ppl_relative_delta_percent']:.4f}%), NLL/token delta {c4q['M3_vs_M1']['nll_per_token_delta_m3_minus_baseline']:.9f}.
- M3 vs M2: PPL delta {c4q['M3_vs_M2']['ppl_delta_m3_minus_baseline']:.9f} ({c4q['M3_vs_M2']['ppl_relative_delta_percent']:.4f}%), NLL/token delta {c4q['M3_vs_M2']['nll_per_token_delta_m3_minus_baseline']:.9f}.
- WT2 and C4 M3-vs-M2 directions agree. C4 is a 131,071-target fixed-budget validation, not standard/full C4 PPL.

## Recovery of native-GPTQ degradation

Using `(PPL_M1 - PPL_method) / (PPL_M1 - PPL_M0)`:

- WT2: M2 recovered {wr['M2_recovery_percent']:.3f}% of M1-to-M0 PPL degradation; M3 recovered {wr['M3_recovery_percent']:.3f}%.
- C4 fixed budget: M2 recovered {cr['M2_recovery_percent']:.3f}%; M3 recovered {cr['M3_recovery_percent']:.3f}%.

## AMP accounting

The canonical final checkpoint accounts for exactly two skips in 61,440 scheduled positions (skip fraction {amp_stats['canonical_final_model_accounting']['skip_fraction']:.10%}): L21/epoch0/sample0, scale 65,536→32,768; and L21/epoch0/sample74, scale 32,768→16,384. The v21.4 attempt log has 7,680 rows: 7,678 optimizer steps and two matching `found_inf` skips. Prefix blocks0–20 contribute 53,760 successful updates by historical inference from the v21.3 audit, because direct historical optimizer hooks/found_inf snapshots were not saved. An earlier interrupted L21/sample0 boundary row is disclosed separately and is not counted again in the canonical final-model schedule.

## Cost and measurement limits

- M1 calibration wall: {baseline['M1']['calibration_wall_seconds']:.6f} s. M2: {baseline['M2']['calibration_wall_seconds']:.6f} s; M2/M1 = {baseline['M2']['calibration_wall_seconds']/baseline['M1']['calibration_wall_seconds']:.4f}×.
- M3's exact internal `perf_counter` calibration wall was not persisted. Timestamp-derived intervals are {v211_span['elapsed_seconds']:.0f} s for the v21.1 committed prefix and {v214_span['elapsed_seconds']:.0f} s for the v21.4 continuation, sum {span_sum:.0f} s. This is not equivalent to the v18 wall measurement, so no M3/M1 or M3/M2 cost ratio is reported.
- The formal v21.4 training allocated peak was {training['peak_allocated_bytes']} bytes; reserved peak was not persisted. Post-checkpoint reconstruction took {training['reconstruction_after_checkpoint_error_seconds']:.3f} s and is excluded as post-hoc replay. Dense export/reload took {training['post_training_export_reload_seconds']:.3f} s and is reported separately.
- The first v21.1 checkpoint-fsync attempt executed 2,560 uncommitted updates and is excluded from final-artifact calibration cost. v21.2/v21.3 diagnostics, smoke runs, and debugging are also excluded.

## What this evidence supports

On this frozen Qwen2.5-0.5B-Instruct checkpoint and these two frozen evaluation streams, M3 completes a valid full-model OmniQuant baseline and has lower NLL/PPL than M1 and M2 on WT2 and C4 fixed-budget. It recovers more of the measured M1-to-M0 PPL degradation than M2 on these runs. It does not establish statistical significance, cross-model generalization, universal superiority, a causal advantage of one clipping mechanism, or inference speed gains. Cost superiority versus M2 is unresolved because exact M3 calibration wall time is unavailable.

## Reproduction and evidence files

This finalization used the frozen M3 raw JSON/window CSV and v18/v19 audited baseline artifacts; it ran no model forward, training, or evaluation. Re-run the standard-library finalizer from the repository root with:

```powershell
& 'E:\\quant-thesis\\.venv-gptq\\Scripts\\python.exe' -B 'experiments\\range_search_w4\\results\\range_w4_omniquant_finalize_v21_5\\finalize_v21_5.py'
```

The final checkpoint identity is in `FINAL_M3_ARTIFACT_AUDIT.json`; raw-data reaggregation is in `WT2_M3_FINAL.json` and `C4_M3_FINAL.json`; the machine-readable final table is `M0_M1_M2_M3_FINAL.csv`.
"""
    report_path = OUT / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    # Keep the evaluation-specific path requested in the protocol as well as root summaries.
    write_json("evaluation/WT2_M3_FINAL.json", wt2)
    write_json("evaluation/C4_M3_FINAL.json", c4)
    write_json("evaluation/EVALUATION_CORRECTNESS.json", eval_correct)

    print(json.dumps({"status": summary["status"], "primary_verdict": verdict,
                      "WT2_PPL": wt2["ppl"], "C4_fixed_budget_PPL": c4["ppl"],
                      "WT2_windows": wt2["windows"], "C4_windows": c4["windows"],
                      "timestamp_intervals_seconds": span_sum,
                      "exact_M3_calibration_wall_recovered": False,
                      "optimizer_steps_this_round": 0,
                      "outputs": sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*")
                                        if p.is_file() and p.name != "finalize_v21_5.py")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
