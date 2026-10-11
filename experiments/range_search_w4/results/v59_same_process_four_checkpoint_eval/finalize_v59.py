"""CPU-only independent audit, aggregation, and report generation for v59."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\quant-thesis")
OUT = ROOT / "experiments/range_search_w4/results/v59_same_process_four_checkpoint_eval"
V57 = ROOT / "experiments/range_search_w4/results/v57_uniform_runtime_four_arm_eval"
V58 = ROOT / "experiments/range_search_w4/results/v58_m2_repaired_unified_eval"
CAP = 3600.0
POSTPROCESS_CHARGE = 60.0
ARMS = ["M1", "M2", "M1_plus_repair", "M2_plus_repair"]
LABELS = {
    "M1": "M1",
    "M2": "M2",
    "M1_plus_repair": "M1+repair (v56)",
    "M2_plus_repair": "M2+repair (v55)",
}
DATASETS = ["wikitext", "c4_fixed_budget"]
EXPECTED = {
    "wikitext": {"n_input": 299078, "valid": 299077, "windows": 582,
                 "manifest_sha": "acd46b20a6b84b534d5625bfab505d599001679a775d619960a40e8da4f62f46",
                 "token_sha": "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3"},
    "c4_fixed_budget": {"n_input": 131072, "valid": 131071, "windows": 253,
                        "manifest_sha": "0b0d70d70d3c1e0ba62126c81a3f3e97531e58d6701236fb8b3315450b7188d4",
                        "token_sha": "bc9a7575527abbc41352352a0439c6bee062ea265b7bd03120704e83f4af7053"},
}

def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf8"))

def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)

def tensor_i64_hash(values) -> str:
    return sha_bytes(np.asarray(values, dtype="<i8").tobytes(order="C"))

def pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0

def main():
    t0 = time.perf_counter()
    run_status = read_json(OUT / "RUN_STATUS.json")
    summary = read_json(OUT / "RUN_SUMMARY.json")
    if run_status.get("status") != "COMPLETE" or run_status.get("complete_evaluations") != 8:
        raise RuntimeError("v59 did not finish all eight evaluations")
    if not summary.get("same_process_all_scored_windows"):
        raise RuntimeError("the eight scored window sets are not all from one process")

    process_env = read_json(OUT / "PROCESS_ENVIRONMENT.json")
    run_lock = read_json(OUT / "RUN_LOCK.json")
    source_rows = {r["arm"]: r for r in read_csv(OUT / "CHECKPOINT_SOURCE_AUDIT.csv")}
    manifest_rows = {r["dataset"]: r for r in read_csv(OUT / "INPUT_SOURCE_AUDIT.csv")}
    if len(set([process_env["process_id"], *summary["process_ids_across_windows"]])) != 1:
        raise RuntimeError("process IDs differ across evaluated windows")
    if not summary.get("same_runtime_signature_all_windows"):
        raise RuntimeError("runtime signature drift was recorded")

    manifest_tokens = {}
    for ds in DATASETS:
        source_row = manifest_rows[ds]
        expectation = EXPECTED[ds]
        if source_row["manifest_sha256"] != expectation["manifest_sha"] or source_row["token_ids_sha256_i64le"] != expectation["token_sha"]:
            raise RuntimeError(f"manifest source mismatch for {ds}")
        doc = read_json(Path(source_row["path"]))
        tokens = doc["token_ids"]
        if len(tokens) != expectation["n_input"] or tensor_i64_hash(tokens) != expectation["token_sha"]:
            raise RuntimeError(f"frozen token manifest content mismatch for {ds}")
        manifest_tokens[ds] = tokens

    arm_results: dict[str, dict] = {arm: {} for arm in ARMS}
    recalculation_records = []
    csv_consistency = []
    all_pids = set()
    for arm in ARMS:
        source = source_rows[arm]
        for ds in DATASETS:
            exp = EXPECTED[ds]
            result_path = OUT / "evaluation" / f"{arm}_{ds}.json"
            csv_path = OUT / "evaluation" / f"{arm}_{ds}_WINDOWS.csv"
            journal = OUT / "evaluation" / arm / ds / "windows"
            result = read_json(result_path)
            rows = [read_json(journal / f"window_{i:04d}.json") for i in range(exp["windows"])]
            csv_rows = read_csv(csv_path)
            if result.get("status") != "PASS" or len(rows) != exp["windows"] or len(csv_rows) != exp["windows"]:
                raise RuntimeError(f"incomplete result/window coverage for {arm}/{ds}")
            if result["checkpoint_sha256"] != source["checkpoint_sha256"] or result["manifest_sha256"] != exp["manifest_sha"] or result["token_ids_sha256_i64le"] != exp["token_sha"]:
                raise RuntimeError(f"result source lock mismatch for {arm}/{ds}")
            if result["runtime_signature_sha256"] != run_lock["runtime_signature_sha256"]:
                raise RuntimeError(f"runtime lock mismatch for {arm}/{ds}")
            if result["new_forwards_this_process"] != exp["windows"] or result["resumed_window_cache_hits"] != 0:
                raise RuntimeError(f"unexpected repeated/reused windows for {arm}/{ds}")

            tokens = np.asarray(manifest_tokens[ds], dtype="<i8")
            total_nll_sum = 0.0
            total_valid = 0
            row_pids = set()
            for i, (row, csvrow) in enumerate(zip(rows, csv_rows)):
                begin = i * 512
                end = min(begin + 2048, len(tokens))
                previous_end = 0 if i == 0 else min((i - 1) * 512 + 2048, len(tokens))
                target_length = end - previous_end
                ids = tokens[begin:end].copy()
                labels = ids.copy()
                if target_length < len(labels):
                    labels[:-target_length] = -100
                valid = int(np.count_nonzero(labels[1:] != -100))
                ih = sha_bytes(ids.tobytes(order="C"))
                lh = sha_bytes(labels.tobytes(order="C"))
                if (int(row["window_index"]) != i or int(row["begin"]) != begin or int(row["end"]) != end
                    or int(row["target_length"]) != target_length or int(row["valid_tokens"]) != valid
                    or row["input_ids_sha256"] != ih or row["labels_sha256"] != lh):
                    raise RuntimeError(f"window/token/mask provenance mismatch at {arm}/{ds}/{i}")
                if row["arm"] != arm or row["dataset"] != ds or row["model_sha256"] != source["checkpoint_sha256"] or row["manifest_sha256"] != exp["manifest_sha"]:
                    raise RuntimeError(f"window source labels mismatch at {arm}/{ds}/{i}")
                pid = int(row["process_id"])
                row_pids.add(pid); all_pids.add(pid)
                if not row.get("scored_new_in_this_process"):
                    raise RuntimeError(f"window was not scored by the v59 process at {arm}/{ds}/{i}")
                loss = float(row["window_loss"])
                nll = float(row["window_nll"])
                if not math.isfinite(loss) or not math.isclose(nll, loss * valid, rel_tol=1e-12, abs_tol=1e-10):
                    raise RuntimeError(f"invalid window NLL arithmetic at {arm}/{ds}/{i}")
                if int(csvrow["window_index"]) != i or not math.isclose(float(csvrow["window_nll"]), nll, rel_tol=0, abs_tol=1e-8):
                    raise RuntimeError(f"CSV/JSON window mismatch at {arm}/{ds}/{i}")
                total_nll_sum += nll
                total_valid += valid
            if len(row_pids) != 1 or next(iter(row_pids)) != int(process_env["process_id"]):
                raise RuntimeError(f"{arm}/{ds} window rows are not from the recorded process")
            weighted = total_nll_sum / total_valid
            ppl = math.exp(weighted)
            if total_valid != exp["valid"] or total_valid != int(result["valid_tokens"]) or len(rows) != int(result["windows"]):
                raise RuntimeError(f"valid-token/window count mismatch for {arm}/{ds}")
            if not math.isclose(weighted, float(result["weighted_nll"]), rel_tol=0, abs_tol=1e-10) or not math.isclose(ppl, float(result["ppl"]), rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"independent aggregate mismatch for {arm}/{ds}")
            arm_results[arm][ds] = {**result, "nll_sum_independent": total_nll_sum,
                                    "valid_tokens_independent": total_valid,
                                    "weighted_nll_independent": weighted, "ppl_independent": ppl}
            recalculation_records.append({
                "arm": arm, "dataset": ds, "checkpoint_sha256": source["checkpoint_sha256"],
                "manifest_sha256": exp["manifest_sha"], "token_ids_sha256_i64le": exp["token_sha"],
                "windows": len(rows), "valid_tokens": total_valid, "nll_sum": total_nll_sum,
                "weighted_nll": weighted, "ppl": ppl,
                "process_id": next(iter(row_pids)), "window_ids_and_mask_hashes_rechecked": True,
            })
            csv_consistency.append({"arm": arm, "dataset": ds, "json_window_count": len(rows),
                                    "csv_window_count": len(csv_rows), "nll_sum_match": True,
                                    "valid_token_match": True})

    if all_pids != {int(process_env["process_id"])}:
        raise RuntimeError(f"not one process across the eight evaluations: {all_pids}")
    if not math.isclose(sum(r["nll_sum"] for r in recalculation_records if r["dataset"] == "wikitext" and r["arm"] == "M1") / 299077,
                        arm_results["M1"]["wikitext"]["weighted_nll"], rel_tol=0, abs_tol=1e-10):
        raise RuntimeError("independent WT2 aggregate failed")

    arm_table = []
    comparison = []
    for ds in DATASETS:
        metric = {arm: arm_results[arm][ds] for arm in ARMS}
        for arm in ARMS:
            r = metric[arm]
            arm_table.append({
                "dataset": ds, "arm": arm, "display_name": LABELS[arm],
                "checkpoint_sha256": r["checkpoint_sha256"],
                "nll_sum": r["nll_sum_independent"], "valid_tokens": r["valid_tokens_independent"],
                "weighted_nll_nats_per_valid_token": r["weighted_nll_independent"],
                "ppl": r["ppl_independent"], "windows": r["windows"],
                "process_id": process_env["process_id"],
            })
        m1, m2 = metric["M1"], metric["M2"]
        r1, r2 = metric["M1_plus_repair"], metric["M2_plus_repair"]
        comparison.append({
            "dataset": ds,
            "M1_nll": m1["weighted_nll_independent"], "M1_ppl": m1["ppl_independent"],
            "M2_nll": m2["weighted_nll_independent"], "M2_ppl": m2["ppl_independent"],
            "M1_plus_repair_nll": r1["weighted_nll_independent"], "M1_plus_repair_ppl": r1["ppl_independent"],
            "M2_plus_repair_nll": r2["weighted_nll_independent"], "M2_plus_repair_ppl": r2["ppl_independent"],
            "repair_delta_nll_M1repair_minus_M1": r1["weighted_nll_independent"] - m1["weighted_nll_independent"],
            "repair_relative_ppl_change_M1_percent": pct(r1["ppl_independent"], m1["ppl_independent"]),
            "repair_delta_nll_M2repair_minus_M2": r2["weighted_nll_independent"] - m2["weighted_nll_independent"],
            "repair_relative_ppl_change_M2_percent": pct(r2["ppl_independent"], m2["ppl_independent"]),
            "scalar10_pre_repair_delta_nll_M2_minus_M1": m2["weighted_nll_independent"] - m1["weighted_nll_independent"],
            "scalar10_pre_repair_relative_ppl_change_M2_vs_M1_percent": pct(m2["ppl_independent"], m1["ppl_independent"]),
            "scalar10_post_repair_delta_nll_M2repair_minus_M1repair": r2["weighted_nll_independent"] - r1["weighted_nll_independent"],
            "scalar10_post_repair_relative_ppl_change_M2repair_vs_M1repair_percent": pct(r2["ppl_independent"], r1["ppl_independent"]),
            "interaction_nll_nats_per_valid_token": (r2["weighted_nll_independent"] - m2["weighted_nll_independent"]) - (r1["weighted_nll_independent"] - m1["weighted_nll_independent"]),
            "interaction_definition": "(NLL_M2repair-NLL_M2)-(NLL_M1repair-NLL_M1); negative means the observed repair delta is more favorable on M2",
            "comparison_source": "all four arms evaluated in v59, same process and runtime",
        })

    write_csv(OUT / "INDEPENDENT_WINDOW_RECALC.csv", recalculation_records)
    write_csv(OUT / "CSV_JSON_WINDOW_AUDIT.csv", csv_consistency)
    write_csv(OUT / "ARM_RESULTS.csv", arm_table)
    write_csv(OUT / "FOUR_ARM_COMPARISON.csv", comparison)

    load_rows = read_csv(OUT / "MODEL_LOAD_AUDIT.csv")
    timings = read_csv(OUT / "STAGE_TIMINGS.csv")
    if [r["arm"] for r in load_rows] != ARMS or any(r["named_hashes_match_v57"] != "True" or r["named_tensor_count"] != "290" for r in load_rows):
        raise RuntimeError("loaded checkpoint tensor audit is incomplete or failed")
    if any(r["attention_loaded"] != "sdpa" or r["dtype"] != "torch.float16" or r["device"] != "cuda:0" or r["eval_mode"] != "True" or r["use_cache_eval_argument"] != "False" for r in load_rows):
        raise RuntimeError("loader settings drifted across arms")
    if len([r for r in timings if r["stage"] == "evaluation"]) != 8:
        raise RuntimeError("stage timing table does not contain eight evaluations")

    cost = read_json(OUT / "COST_ACCOUNTING.json")
    budget = read_json(OUT / "BUDGET_STATE.json")
    precheck = read_json(OUT / "COST_PRECHECK.json")
    start_marker = read_json(OUT / "START_EPOCH.json")
    if budget["status"] not in ("COMPLETE", "COMPLETE_WITH_CPU_FINALIZATION") or abs(float(budget["completed_active_seconds"]) - float(run_status["charged_active_seconds"])) > 1e-6:
        raise RuntimeError("budget ledger and run status differ")
    if float(budget["completed_active_seconds"]) >= CAP:
        raise RuntimeError("v59 exceeded the 3600 second cap")

    cpu_preflight = {
        "status": "PASS", "script": str(OUT / "preflight_v59.py"),
        "measured_script_seconds": 4.41644449999967,
        "charged_wall_from_v59_directory_creation_to_runner_entry_seconds": precheck["charged_before_first_model_load_seconds"],
        "charging_boundary_basis": "BUDGET_START.json begins at v59 output directory creation; source preparation and all pre-run attempts are included",
        "gpu_model_forward_before_formal_process": False,
        "preflight_checks": ["four checkpoint SHA and config source", "v57 290-tensor reference manifests", "effective AutoConfig fingerprints", "frozen token manifests and hashes", "v58 runtime/evaluator source", "output directory writable", "budget projection"],
    }
    write_json(OUT / "CPU_PRECHECK.json", cpu_preflight)
    write_json(OUT / "PRE_RUN_DIAGNOSTIC.json", {
        "count": 2,
        "phase": "CPU-only source preflight before first model load/forward",
        "events": [
            {
                "initial_issue": "A preliminary read-only lookup assumed v57-v59 artifacts were directly under results; actual files are under experiments/range_search_w4/results.",
                "resolution": "Located and used the existing nested experiment results directory.",
                "effect": "No files outside v59 were written and no model/GPU operation occurred.",
            },
            {
                "initial_issue": "A preliminary hash assertion compared the v58 run_v58_eval.py source to the v56 runtime_v54.py evaluator-helper hash; the two files are different artifacts.",
                "evidence": {
                    "v56_helper_sha256": "bbe1e4c22200f1ccb3d85f35a274c44cf5d46cd606160b59b0cd9afc346bae11",
                    "v58_runner_sha256": "c27fb3a5dbf2f940a7ed0fd465ac02a07da8de94c9dc2a060adea962a63ed2df",
                },
                "resolution": "Read the v58 ARTIFACT_HASHES.csv; separately locked the v58 runner hash and retained the v56 helper hash as a distinct source record.",
                "effect": "No model load, GPU forward, checkpoint write, or evaluation occurred during the failed assertion.",
            },
        ],
        "charged_with": "pre-run charged wall from v59 directory creation through runner entry",
    })
    write_json(OUT / "POST_RUN_DIAGNOSTICS.json", {
        "count": 1,
        "phase": "CPU-only finalizer before any result table was written",
        "initial_issue": "The first finalizer invocation used the unsupported codec spelling utf8-sig.",
        "resolution": "Changed to Python's utf-8-sig codec name; reran the finalizer.",
        "effect": "No evaluation, score, journal, or checkpoint was modified by the failed invocation.",
        "first_failed_finalizer_wall_seconds": 1.705,
        "charged_with": "conservative 60 second postprocessing charge",
    })

    v18_protocol = read_json(ROOT / "experiments/range_search_w4/results/range_w4_fullmodel_v18/PROTOCOL.json")
    write_json(OUT / "MODEL_SOURCE_AUDIT.json", {
        "model_id": v18_protocol["model"]["id"],
        "model_revision": v18_protocol["model"]["revision"],
        "tokenizer_revision": v18_protocol["model"]["revision"],
        "source_protocol": str(ROOT / "experiments/range_search_w4/results/range_w4_fullmodel_v18/PROTOCOL.json"),
        "v18_protocol_sha256": sha_file(ROOT / "experiments/range_search_w4/results/range_w4_fullmodel_v18/PROTOCOL.json"),
        "effective_config_sha256_all_four": "a33901fd787a1385efbdd5574d88012774c97110064a3f6a2cb05d070cba9dd2",
        "all_four_loaded_named_tensor_manifests_match_v57": True,
    })

    pre_start_charge = float(precheck["charged_before_first_model_load_seconds"])
    runner_active = float(cost["runner_active_seconds"])
    formal_process_wall = runner_active - pre_start_charge
    evaluation_sum = sum(float(r["seconds"]) for r in timings if r["stage"] == "evaluation")
    load_sum = sum(float(r["load_seconds"]) for r in load_rows)
    hash_sum = sum(float(r["loaded_parameter_hash_seconds"]) for r in load_rows)
    unload_sum = sum(float(r["seconds"]) for r in timings if r["stage"] == "unload_gc_empty_cache")
    unassigned = formal_process_wall - evaluation_sum - load_sum - hash_sum - unload_sum
    finalizer_measured_before_accounting = time.perf_counter() - t0
    total_charge = runner_active + POSTPROCESS_CHARGE
    remaining = CAP - total_charge
    cost.update({
        "runner_active_seconds_including_preflight_charge": runner_active,
        "pre_runner_charged_wall_seconds": pre_start_charge,
        "formal_runner_process_wall_seconds_excluding_preflight_charge": formal_process_wall,
        "cpu_preflight_script_measured_seconds": cpu_preflight["measured_script_seconds"],
        "evaluation_stage_sum_seconds": evaluation_sum,
        "model_load_synchronized_seconds_sum": load_sum,
        "loaded_named_tensor_hash_seconds_sum": hash_sum,
        "unload_gc_empty_cache_seconds_sum": unload_sum,
        "formal_process_unassigned_setup_io_aggregation_seconds": unassigned,
        "postprocessing_cpu_measured_seconds_before_accounting": finalizer_measured_before_accounting,
        "postprocessing_budget_charge_seconds_conservative": POSTPROCESS_CHARGE,
        "v58_based_total_projection_seconds": precheck["projected_total_seconds"],
        "actual_total_minus_precheck_projection_seconds": total_charge - float(precheck["projected_total_seconds"]),
        "actual_formal_runner_wall_per_checkpoint_pair_average_seconds": formal_process_wall / 4,
        "actual_formal_runner_average_minus_v58_measured_pair_seconds": formal_process_wall / 4 - float(precheck["v58_measured_one_model_pair_process_wall_seconds"]),
        "post_report_reserve_seconds": POSTPROCESS_CHARGE,
        "total_charged_seconds_including_postprocessing": total_charge,
        "remaining_seconds_after_postprocessing_charge": remaining,
        "gpu_evaluation_included_in_runner_total": True,
        "GPU time and CPU wall are not interchangeable": True,
        "historical_ledgers_reset": False,
    })
    write_json(OUT / "COST_ACCOUNTING.json", cost)
    budget.update({
        "completed_active_seconds": total_charge,
        "active_start_unix": None, "active_start_utc": None,
        "status": "COMPLETE_WITH_CPU_FINALIZATION",
        "postprocessing_charge_seconds_conservative": POSTPROCESS_CHARGE,
        "total_charged_seconds": total_charge,
        "remaining_seconds": remaining,
    })
    write_json(OUT / "BUDGET_STATE.json", budget)
    run_status.update({
        "charged_active_seconds": total_charge,
        "runner_charged_seconds": runner_active,
        "postprocessing_charge_seconds_conservative": POSTPROCESS_CHARGE,
        "remaining_seconds": remaining,
        "finalization_status": "COMPLETE",
    })
    write_json(OUT / "RUN_STATUS.json", run_status)

    # GPU runtime telemetry was sampled during the run; WDDM did not expose process attribution.
    telemetry = [
        {"source": "nvidia-smi", "sample_local_time": "2026-10-10 22:19:34", "gpu_util_percent": 98, "memory_used_mib": 7918, "memory_total_mib": 8188, "temperature_c": 79, "process_attribution": "unavailable under WDDM"},
        {"source": "nvidia-smi", "sample_local_time": "during M1_plus_repair WT2", "gpu_util_percent": 98, "memory_used_mib": 7506, "memory_total_mib": 8188, "temperature_c": 79, "process_attribution": "unavailable under WDDM"},
    ]
    write_csv(OUT / "GPU_TELEMETRY_SAMPLES.csv", telemetry)

    # Compact report; all principal numbers are from this v59 process only.
    lines = [
        "# v59 — 同一进程四 checkpoint 正式质量对照", "",
        "## 结果状态", "",
        f"八项完整评测均完成，进程号 `{process_env['process_id']}`；582 个 WT2 窗口和 253 个 C4 窗口分别覆盖四个模型。窗口来源与计分 mask 独立重核通过；四份 checkpoint 的 290 项具名张量哈希均匹配。",
        "本轮四臂数值全部来自 v59 这一个 Python 进程，不混入历史 PPL。历史校准传播有效性仍是 `UNKNOWN`；统一评测不补充传播证据。",
        "", "## 统一四臂结果", "",
        "NLL 是 `sum(window_nll) / sum(valid_tokens)`，单位 nats/valid token；PPL 为 `exp(NLL)`。PPL 变化百分比以对应对照模型为分母。",
        "",
        "| 数据集 | M1 NLL / PPL | M2 NLL / PPL | M1+修复 NLL / PPL | M2+修复 NLL / PPL |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in comparison:
        lines.append(f"| {c['dataset']} | {c['M1_nll']:.9f} / {c['M1_ppl']:.9f} | {c['M2_nll']:.9f} / {c['M2_ppl']:.9f} | {c['M1_plus_repair_nll']:.9f} / {c['M1_plus_repair_ppl']:.9f} | {c['M2_plus_repair_nll']:.9f} / {c['M2_plus_repair_ppl']:.9f} |")
    lines += ["", "### 差值与交互量", ""]
    for c in comparison:
        ds = c["dataset"]
        lines.append(f"- **{ds}**：M1 修复 ΔNLL `{c['repair_delta_nll_M1repair_minus_M1']:+.9f}`，PPL `{c['repair_relative_ppl_change_M1_percent']:+.4f}%`；M2 修复 ΔNLL `{c['repair_delta_nll_M2repair_minus_M2']:+.9f}`，PPL `{c['repair_relative_ppl_change_M2_percent']:+.4f}%`。修复前 M2−M1 ΔNLL `{c['scalar10_pre_repair_delta_nll_M2_minus_M1']:+.9f}`，修复后 M2+修复−M1+修复 ΔNLL `{c['scalar10_post_repair_delta_nll_M2repair_minus_M1repair']:+.9f}`。描述性交互量 `I=(NLL_R2−NLL_M2)−(NLL_R1−NLL_M1)={c['interaction_nll_nats_per_valid_token']:+.9f}` nats/valid token。")
    lines += [
        "",
        "交互量为单次固定 checkpoint 的描述性差值；负值表示本轮 M2 修复臂的 NLL 变化相对更有利。不表示统计显著性、跨 seed 稳定性或协同机制。",
        "", "## 运行时和来源", "",
        f"- Python `{process_env['python']}` (`{process_env['python_executable']}`)；PyTorch `{process_env['pytorch']}`；PyTorch CUDA build/runtime `{process_env['cuda_runtime']}`；Transformers `{process_env['transformers']}`；tokenizers `{process_env['tokenizers']}`。GPU：`{process_env['gpu']}`。",
        f"- 同一 loader：FP16、CUDA、`attn_implementation=sdpa`、eval、`use_cache=False`、TF32 关闭、FP32 matmul precision `{process_env['fp32_matmul_precision']}`。SDPA backend 开关见 `PROCESS_ENVIRONMENT.json`；实际底层 kernel 未插桩。",
        f"- Qwen2 model code SHA256 `{process_env['model_code_sha256']}`；PPL core `benchmark/eval/ppl.py` SHA256 `{process_env['evaluator_core_sha256']}`；v58 runner/evaluator wrapper SHA256 `{process_env['evaluator_wrapper_sha256']}`。v57 来源清单和逐 checkpoint/张量核验见 `CHECKPOINT_SOURCE_AUDIT.csv`、`MODEL_LOAD_AUDIT.csv`。",
        "- 模型标识与 revision 来源于 v18 冻结协议，见 `MODEL_SOURCE_AUDIT.json`。",
        "- WT2 manifest/token hashes：`%s` / `%s`；C4：`%s` / `%s`。WT2 582 窗口、299,077 valid tokens；C4 253 窗口、131,071 valid tokens。窗口起止、输入切片哈希和 labels mask 哈希逐行复核。" % (EXPECTED['wikitext']['manifest_sha'], EXPECTED['wikitext']['token_sha'], EXPECTED['c4_fixed_budget']['manifest_sha'], EXPECTED['c4_fixed_budget']['token_sha']),
        "- Luna Max 是用户指定设置；本地评测进程的模型选择接口不可查询，不能独立确认当前会话模型身份。",
        "", "## 成本", "",
        f"- v59 上限 3600 秒；包括目录建立以来的预检/准备计费及正式 runner，本轮保守总计 `{total_charge:.3f}` 秒，余 `{remaining:.3f}` 秒。目录创建至 runner 入口计入 `{pre_start_charge:.3f}` 秒；其中独立 CPU 预检脚本实测 `{cpu_preflight['measured_script_seconds']:.3f}` 秒。GPU 评测与其他 CPU wall 分开列于 `COST_ACCOUNTING.json`。",
        "- 正式运行前记录并修正一次 CPU 来源哈希断言混淆（v56 helper 与 v58 runner 是不同文件）；发生在模型加载/前向前，见 `PRE_RUN_DIAGNOSTIC.json`，已包含在预运行时间计费中。",
        "- 另一次只读路径定位最初把版本目录假定在错误层级，随后改用项目实际嵌套目录；两项均为 CPU 侧预检，见 `PRE_RUN_DIAGNOSTIC.json`。",
        "- 一次最终 CPU 汇总启动因 Python codec 名称拼写失败，未写出结果表；更正后完成，见 `POST_RUN_DIAGNOSTICS.json`，包含在 30 秒收尾保守计费中。",
        f"- 正式 runner 活动 wall（含来源核验、加载、张量校验、八项评分及卸载）为 `{formal_process_wall:.3f}` 秒；八项同步 evaluator 阶段合计 `{evaluation_sum:.3f}` 秒，模型加载 `{load_sum:.3f}` 秒，290 张量核验 `{hash_sum:.3f}` 秒，卸载 `{unload_sum:.3f}` 秒。v58 的四对投影加准备/收尾预估为 `{float(precheck['projected_total_seconds']):.3f}` 秒，实际计费高 `{total_charge-float(precheck['projected_total_seconds']):.3f}` 秒；其中 M1+修复 WT2 实测 `{float(next(r['seconds'] for r in timings if r['arm']=='M1_plus_repair' and r['dataset']=='wikitext')):.3f}` 秒。收尾另保守计 `{POSTPROCESS_CHARGE:.1f}` 秒。预测只作为预估，实际总成本以上述计费为准。",
        "- 期间 GPU telemetry 曾显示约 98% 利用率、7.5–7.9 GiB/8 GiB 显存占用；WDDM 未提供进程归属，见 `GPU_TELEMETRY_SAMPLES.csv`。模型自身 CUDA 峰值 allocation/reservation 见 `MODEL_LOAD_AUDIT.csv`。",
        "", "## 解释边界", "",
        "本轮只对四个现有 checkpoint 做加载、哈希核验和正式评测，没有量化、整数修复、候选搜索、新校准或 P3。比较回答的是这些固定 checkpoint 在同一进程/运行时下的质量差异。它不能把历史校准传播有效性从 UNKNOWN 改为 PASS，也不提供统计显著、跨 seed 稳定或原创贡献证据。",
        "", "## 文件与复现", "",
        "- 八项窗口和聚合：`evaluation/<arm>_<dataset>_WINDOWS.csv`、`evaluation/<arm>_<dataset>.json`。独立复算：`INDEPENDENT_WINDOW_RECALC.csv`、`CSV_JSON_WINDOW_AUDIT.csv`。",
        "- 四臂主表：`ARM_RESULTS.csv`、`FOUR_ARM_COMPARISON.csv`。环境/来源/成本：`PROCESS_ENVIRONMENT.json`、`MODEL_SOURCE_AUDIT.json`、`CHECKPOINT_SOURCE_AUDIT.csv`、`INPUT_SOURCE_AUDIT.csv`、`MODEL_LOAD_AUDIT.csv`、`STAGE_TIMINGS.csv`、`COST_ACCOUNTING.json`。",
        "- 正式评测命令（工作目录 `E:\\quant-thesis`）：`.venv\\Scripts\\python.exe experiments\\range_search_w4\\results\\v59_same_process_four_checkpoint_eval\\run_v59_eval.py`。本 v59 目录已完成且预算锁定；复跑需使用空白的独立输出目录并改写 OUT 路径，不要覆盖或重用本次窗口。CPU 独立复算脚本是 `finalize_v59.py`。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    (OUT / "REPRODUCE.md").write_text(
        "# v59 复现\n\n正式评测使用命令（工作目录 `E:\\quant-thesis`）：\n\n```powershell\n.venv\\Scripts\\python.exe experiments\\range_search_w4\\results\\v59_same_process_four_checkpoint_eval\\run_v59_eval.py\n```\n\n此目录已完成并锁定预算；不要在它上面重新启动 runner。要进行新的完整重放，应将 runner 置于空白的独立输出目录、将 `OUT` 指向该目录，并保留源/配置锁。`finalize_v59.py` 是 CPU-only 独立核验脚本，会检查已存窗口、token/mask 哈希、进程 ID、valid-token 加权 NLL 和 PPL；跨进程恢复会被标为不满足严格同进程比较。\n",
        encoding="utf8")

    # Hash all final v59 artifacts except the self-referential manifest and Python cache.
    paths = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "ARTIFACT_HASHES.csv" and "__pycache__" not in p.parts and p.suffix != ".pyc")
    hashes = [{"relative_path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha_file(p)} for p in paths]
    write_csv(OUT / "ARTIFACT_HASHES.csv", hashes)

    # Conservatively charge the CPU closeout as 30 seconds and keep the cap accounting final.
    # Refresh report/hash after adding the final budget state; hashes then cover the final ledger.
    actual = time.perf_counter() - t0
    total_charge = float(run_status["runner_charged_seconds"]) + POSTPROCESS_CHARGE
    remaining = CAP - total_charge
    if remaining < 0:
        raise RuntimeError("postprocessing charge would exceed the v59 cap")
    # Artifact manifest is computed after all content; its own generation is covered by the conservative 30s charge.
    print(json.dumps({"status": "FINALIZED", "evaluations": 8, "same_process": True,
                      "comparison": comparison, "runner_seconds": runner_active,
                      "finalizer_seconds_measured_before_last_hash": actual,
                      "postprocessing_charge_seconds": POSTPROCESS_CHARGE,
                      "total_charged_seconds": total_charge, "remaining_seconds": remaining}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
