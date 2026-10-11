"""CPU-only final audit and D outcome report for the stopped v51 attempt."""
import csv, hashlib, json, math, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"E:\quant-thesis")
BASE = ROOT / "experiments/range_search_w4/results"
OUT = BASE / "v51_m2_fast_integer_repair_fullmodel"
V18 = BASE / "range_w4_fullmodel_v18"
V45 = BASE / "v45_m2_single_integer_repair_fullmodel"
V46 = BASE / "v46_m1_single_integer_repair_fullmodel"
V49 = BASE / "v49_fullmodel_fast_integer_repair"

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def readj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def writej(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

def writecsv(name, rows, fields=None):
    p = OUT / name
    if not rows and not fields:
        p.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0])
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    audit = readj(OUT / "SOURCE_AUDIT.json")
    protocol = readj(OUT / "PROTOCOL.json")
    start = readj(OUT / "GPU_START.json")
    run = readj(OUT / "RUN_STATUS.json")
    failure = readj(OUT / "diagnostics/INITIAL_QUAD_FAILURE.json")
    prep = readj(OUT / "PREPARATION.json")
    resume = readj(OUT / "PRE_RESUME_AUDIT.json")
    v49_status = readj(V49 / "RUN_STATUS.json")
    v18_runtime = readj(V18 / "FORMAL_RUNTIME.json")
    ppl_audit = readj(V46 / "PPL_SOURCE_AUDIT.json")
    assert audit["pass"] and prep["status"] == "CPU_SOURCE_AUDIT_PASS" and protocol["status"] == "FROZEN_BEFORE_GPU"
    assert start["protocol_sha256"] == sha(OUT / "PROTOCOL.json")
    assert run["status"] == "FAILED_STOP" and not run["complete_model"] and not run["PPL_run"]
    assert not (OUT / "CHECKPOINT_LOCK.json").exists() and not (OUT / "COMPLETE.json").exists()
    assert not (OUT / "FORMAL_REPAIR_COMPLETE.json").exists()
    assert failure["status"] == "DIAGNOSTIC_COMPLETE_NO_SEARCH_RESUMED"
    assert failure["row_count_failing_frozen_FP32_direct_vs_quadratic"] == 1
    assert failure["row_count_failing_independent_rawX_FP64_vs_quadratic_at_same_tolerance"] == 0
    assert failure["algorithm_or_tolerance_changed"] is False
    assert failure["no_search_or_model_commit_or_PPL"] is True
    assert len(list((OUT / "formal/groups").glob("layer_*.pt"))) == 4
    assert {p.name for p in (OUT / "formal/groups").glob("layer_*.pt")} == {f"layer_00_{g}.pt" for g in ["qkv", "o", "up_gate", "down"]}
    assert not list((OUT / "formal/groups").glob("layer_01_*.pt"))

    # Historical result files are hash-checked before inclusion. No v51 PPL is synthesized.
    historical = ppl_audit["historical_source_records"]
    rows = []
    for model, prefix in [("M1", "M1"), ("M2 scalar10", "M2"), ("original R2 (v45 M2 + v45 repair)", "R2")]:
        for dataset, key in [("WT2 test", "wikitext"), ("C4 fixed validation", "c4_fixed_budget")]:
            rec = historical[f"{prefix}_{key}"]
            assert sha(rec["result_path"]) == rec["result_sha256"] and sha(rec["windows_path"]) == rec["windows_sha256"]
            rows.append({"dataset": dataset, "model": model, "status": "historical source audited", "PPL": rec["ppl"], "valid_token_weighted_NLL": rec["nll"], "valid_tokens": rec["valid_tokens"], "windows": rec["windows"], "result_sha256": rec["result_sha256"], "windows_sha256": rec["windows_sha256"], "PPL_delta_vs_M1_pct": None, "PPL_delta_vs_M2_pct": None, "PPL_delta_vs_original_R2_pct": None})
    v49_source = audit["v49_r1_ppl_sources"]
    for dataset, key in [("WT2 test", "wikitext"), ("C4 fixed validation", "c4_fixed_budget")]:
        rec = v49_source[key]
        assert sha(rec["result_path"]) == rec["result_sha256"] and sha(rec["windows_path"]) == rec["windows_sha256"]
        rows.append({"dataset": dataset, "model": "v49 R1 (M1 + v49 fast repair)", "status": "historical source audited", "PPL": rec["ppl"], "valid_token_weighted_NLL": rec["nll"], "valid_tokens": rec["valid_tokens"], "windows": rec["windows"], "result_sha256": rec["result_sha256"], "windows_sha256": rec["windows_sha256"], "PPL_delta_vs_M1_pct": None, "PPL_delta_vs_M2_pct": None, "PPL_delta_vs_original_R2_pct": None})
    v51_rows = []
    for dataset in ["WT2 test", "C4 fixed validation"]:
        v51_rows.append({"dataset": dataset, "model": "v51 M2 + v49 fast repair", "status": "D - stopped before complete model and PPL", "PPL": None, "valid_token_weighted_NLL": None, "valid_tokens": None, "windows": None, "result_sha256": None, "windows_sha256": None, "PPL_delta_vs_M1_pct": None, "PPL_delta_vs_M2_pct": None, "PPL_delta_vs_original_R2_pct": None})
    rows.extend(v51_rows)
    by = {(r["dataset"], r["model"]): r for r in rows}
    for dataset in ["WT2 test", "C4 fixed validation"]:
        base1 = by[(dataset, "M1")]["PPL"]
        base2 = by[(dataset, "M2 scalar10")]["PPL"]
        baser2 = by[(dataset, "original R2 (v45 M2 + v45 repair)")]["PPL"]
        for r in rows:
            if r["dataset"] != dataset or r["PPL"] is None: continue
            r["PPL_delta_vs_M1_pct"] = (r["PPL"] / base1 - 1) * 100
            r["PPL_delta_vs_M2_pct"] = (r["PPL"] / base2 - 1) * 100
            r["PPL_delta_vs_original_R2_pct"] = (r["PPL"] / baser2 - 1) * 100
    writecsv("HISTORICAL_QUALITY_COMPARISON.csv", rows)

    stage_data = list(csv.DictReader((OUT / "STAGE_TIMINGS.csv").open(encoding="utf-8", newline="")))
    grouped = defaultdict(list)
    for x in stage_data: grouped[(x["mode"], x["stage"])].append(x)
    stage_rows = []
    for (mode, stage), vals in sorted(grouped.items()):
        stage_rows.append({"mode": mode, "stage": stage, "calls": len(vals), "synchronized_wall_seconds_sum": sum(float(x["seconds"]) for x in vals), "peak_allocated_bytes_max": max(int(x["gpu_peak_allocated_bytes"]) for x in vals), "peak_reserved_bytes_max": max(int(x["gpu_peak_reserved_bytes"]) for x in vals), "measurement": "sum of per-stage CUDA-synchronized timers; not process wall"})
    writecsv("PARTIAL_STAGE_COSTS.csv", stage_rows)
    mode_totals = defaultdict(float)
    for x in stage_data: mode_totals[x["mode"]] += float(x["seconds"])
    synchronized_total = sum(float(x["seconds"]) for x in stage_data)
    block_rows = list(csv.DictReader((OUT / "BLOCK_TIMINGS.csv").open(encoding="utf-8", newline="")))
    formal_block0 = sum(float(x["seconds"]) for x in block_rows if int(x["layer"]) == 0 and int(x["new_groups"]) > 0)
    restore_only_block0 = sum(float(x["seconds"]) for x in block_rows if int(x["layer"]) == 0 and int(x["new_groups"]) == 0)
    gpu_lifecycle_elapsed = float(failure["diagnostic_cumulative_elapsed_seconds"])
    m2_build = float(v18_runtime["m2"]["wall_seconds_including_load_replay_save"])
    v49_formal = 4543.0753269
    v49_full_wall = float(v49_status["elapsed_seconds_from_gpu_start"])
    total_stages = [
        {"item": "M2 scalar10 initialization", "seconds": m2_build, "basis": "historical measured v18 FORMAL_RUNTIME; outside v51 GPU job"},
        {"item": "v49 M1 + fast-repair formal full-model repair", "seconds": v49_formal, "basis": "historical measured v49 report/block accounting; different initialization and inputs"},
        {"item": "v49 complete job including failures/save/evaluation", "seconds": v49_full_wall, "basis": "historical measured v49 RUN_STATUS"},
        {"item": "v51 completed formal block0 repair once", "seconds": formal_block0, "basis": "measured BLOCK_TIMINGS; all 7 L0 Linear, 12672 rows, group commits/terminal checks included"},
        {"item": "v51 block0 restore/replay without repair", "seconds": restore_only_block0, "basis": "measured BLOCK_TIMINGS new_groups=0; committed groups restored and propagation replayed"},
        {"item": "v51 all CUDA-synchronized timer intervals", "seconds": synchronized_total, "basis": "measured STAGE_TIMINGS sum across formal, diagnostic, preflight scopes; excludes untimed gaps"},
        {"item": "v51 cumulative job wall through final diagnostic", "seconds": gpu_lifecycle_elapsed, "basis": "measured elapsed from GPU_START through final diagnostic; includes failures/recovery/diagnosis and intervening wall"},
        {"item": "v51 new full-model save/reload", "seconds": 0.0, "basis": "not reached"},
        {"item": "v51 new WT2/C4 PPL evaluation", "seconds": 0.0, "basis": "not reached"},
        {"item": "historical M2 build plus v51 attempted job through stop", "seconds": m2_build + gpu_lifecycle_elapsed, "basis": "sum of historical measured M2 construction plus incomplete v51 job window; not a completed pipeline cost"},
    ]
    writecsv("COST_LEDGER.csv", total_stages)

    rowfail = list(csv.DictReader((OUT / "diagnostics/INITIAL_QUAD_FAILURE_ROWS.csv").open(encoding="utf-8", newline="")))
    assert len(rowfail) == 1 and rowfail[0]["row"] == "122"
    failed_value = rowfail[0]
    current_runner = sha(OUT / "run_v51.py")
    amend = readj(OUT / "RUNNER_AMENDMENT_LOG.json")
    assert amend["current_runner_sha256"] == current_runner and amend["protocol_sha256_before"] == amend["protocol_sha256_after"] == sha(OUT / "PROTOCOL.json")
    m1 = (historical["M1_wikitext"]["result_sha256"], historical["M1_c4_fixed_budget"]["result_sha256"])
    m2 = (historical["M2_wikitext"]["result_sha256"], historical["M2_c4_fixed_budget"]["result_sha256"])
    r2 = (historical["R2_wikitext"]["result_sha256"], historical["R2_c4_fixed_budget"]["result_sha256"])
    v49_results = (v49_source["wikitext"]["result_sha256"], v49_source["c4_fixed_budget"]["result_sha256"])
    decision = {
        "status": "D", "complete_model": False, "quality_verified": False, "PPL_evaluations_v51": 0,
        "stop_reason": "The unchanged v48 fast_accept.batch_repair initialization consistency guard failed on layer1 self_attn.k_proj before candidate search. No permitted frozen path remained; no tolerance change or bypass was applied.",
        "fixed_guard": {"rtol": 2e-5, "atol": 1e-12, "failing_rows": 1, "total_rows": 128, "row": int(failed_value["row"]), "fp32_direct": float(failed_value["fp32_direct"]), "quadratic": float(failed_value["quadratic"]), "independent_rawX_fp64": float(failed_value["rawX_fp64"]), "fp32_minus_quadratic": float(failed_value["fp32_minus_quad"]), "frozen_tolerance": float(failed_value["frozen_tolerance"]), "raw64_minus_quadratic": float(failed_value["raw64_minus_quad"]), "fp32_relative_error": failure["max_relative_fp32_minus_quadratic"], "independent_raw64_relative_error": failure["max_relative_raw64_minus_quadratic"], "interpretation": "FP32 direct rounding failed the fixed check for a near-zero row; independent raw-X FP64 matches the full FP64 Gram quadratic. This rules against a detected code/grid/C/p state divergence in the deterministic replay, but the frozen v48 entry gate still fails."},
        "coverage": {"full_model_blocks": "1/24 committed", "linear_modules": "7/168", "output_rows": "12672/304128", "committed_groups": 4, "layer1_uncommitted_groups": 0, "all_stage_l1_failure_before_search": True, "full_checkpoint": False, "reload": False, "PPL": False},
        "run_hashes": {"protocol_sha256": sha(OUT / "PROTOCOL.json"), "M2_checkpoint_sha256": audit["m2_model_sha256"], "fast_accept_sha256": protocol["algorithm_source"]["fast_accept_sha256"], "rules_sha256": protocol["algorithm_source"]["acceptance_rules_sha256"], "runner_sha256": current_runner, "M1_PPL_result_hashes": m1, "M2_PPL_result_hashes": m2, "original_R2_PPL_result_hashes": r2, "v49_R1_PPL_result_hashes": v49_results},
        "P3_read": False, "v50_mixed_gram_path_used": False, "algorithm_or_tolerance_changed": False,
        "gpu_cap_seconds": 10800, "cumulative_elapsed_through_final_diagnostic_seconds": gpu_lifecycle_elapsed,
        "synchronized_stage_timer_totals_by_mode_seconds": dict(mode_totals), "all_synchronized_stage_timer_seconds": synchronized_total,
        "created_unix": time.time(),
    }
    writej("FINAL_DECISION.json", decision)

    lines = [
        "# v51 M2 scalar10 + v49 fast integer repair — stopped (D)", "",
        "## 结果", "",
        "v51 未完成全模型修复，未保存完整模型，也未运行 WT2/C4。质量迁移结论为 D（运行/评测未完成）。block0 是正式流程的一部分，已完整修复并提交一次；layer1 的第一个 qkv Linear 在进入候选搜索前触发 v48 的冻结初始一致性检查，随后停止。没有绕过检查、改容差或切换评分路径。",
        "", "M2 的 code/网格接入预检通过。正式 M2 保存的是 dense FP16，但其逐层参数含可信 FP16 权重和原 scale/zero/g_idx；v42/v45 的历史审计对 16 个合法 code 做逐值位模式枚举并证明每个坐标映射唯一。v51 再次逐层核对 168 个 Linear、304128 行、pack/unpack 和 M2 权重位模式，没有套用 M1 网格或近似舍入反推。",
        "", "停止点诊断重建了原 M2 + v51 已提交 block0 的确定性前缀。layer1 `self_attn.k_proj` 第 122 行的 FP32 direct loss 为 `%.12g`，FP64 quadratic 为 `%.12g`，差 `%.6g`，超过冻结误差界 `%.6g`。独立 raw-X FP64 为 `%.12g`，与 quadratic 只差 `%.6g`；其余 127/128 行通过。这个近零行的 FP32 direct 舍入误差导致原 guard 拒绝开始修复。它没有显示 code/grid、C 或 p 的状态偏差，但仍不能绕过冻结 guard，所以没有全模型结果。" % (float(failed_value["fp32_direct"]), float(failed_value["quadratic"]), float(failed_value["fp32_minus_quad"]), float(failed_value["frozen_tolerance"]), float(failed_value["rawX_fp64"]), float(failed_value["raw64_minus_quad"])),
        "", "历史 PPL 只用于上下文，不能代替 v51：",
        "", "| 数据集 | M1 | M2 scalar10 | 原 R2（M2+v45 修复） | v49 R1（M1+v49 快修） | v51 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ds in ["WT2 test", "C4 fixed validation"]:
        items = [by[(ds, "M1")], by[(ds, "M2 scalar10")], by[(ds, "original R2 (v45 M2 + v45 repair)")], by[(ds, "v49 R1 (M1 + v49 fast repair)")], by[(ds, "v51 M2 + v49 fast repair")]]
        vals = [f"{x['PPL']:.6f}" if x["PPL"] is not None else "未评测 (D)" for x in items]
        lines.append(f"| {ds} | " + " | ".join(vals) + " |")
    lines += [
        "", "基线 NLL 是按 valid tokens 加权的历史值，完整结果及哈希见 `HISTORICAL_QUALITY_COMPARISON.csv`。v49 使用 M1 初始化；R2 使用旧 v45 修复，比较均为描述性历史结果，不能推断 v51 相对质量。v51 没有 PPL/NLL，也没有锁定 checkpoint。",
        "", "## 覆盖与正确性", "",
        "- 正式 block0：7/168 Linear、12,672/304,128 行；四个组 checkpoint 均提交，码/网格/终态检查通过。接受更新共 379,055。恢复重放核对了四个 block0 输入哈希并跳过修复，没有重复提交。",
        "- layer1：`self_attn.k_proj` 首次初始化 direct/quadratic guard 失败，尚未评分任何候选，也没有提交 layer1 组。code 范围、M2 scale/zero/g_idx、FP16 实际网格没有发现不一致。",
        "- 未产生完整模型、reload/output regression、PPL 或新的质量结论；没有读取 P3。block0 局部重构记录不作为质量替代。",
        "", "## 成本", "",
        f"M2 scalar10 正式初始化历史实测 {m2_build:.3f}s，单列且不计入本轮 GPU cap。v51 从 GPU_START 到最后一项诊断累计 {gpu_lifecycle_elapsed:.3f}s（含失败、恢复和诊断），低于 10,800s 硬上限；所有已打点 CUDA 同步阶段合计 {synchronized_total:.3f}s，按 mode 的分解在 `PARTIAL_STAGE_COSTS.csv`。两种口径不同：同步阶段计时不含模型加载外的所有间隔，生命周期 wall 会计入等待/脚本间隔。",
        f"block0 一次完整正式修复实测 {formal_block0:.3f}s；恢复时只读重放 block0 {restore_only_block0:.3f}s，没有再次修复。v51 保存/reload/PPL 均未运行。历史 v49 M1+快修 formal repair 为 {v49_formal:.3f}s，完整 job wall {v49_full_wall:.3f}s。v51 只完成 block0，不能据此声称完整作业加速或标量范围质量收益。",
        f"M2 初始化 + v51 已发生不完整 job 的账面合计为 {m2_build + gpu_lifecycle_elapsed:.3f}s；这不是完成的初始化+全模型修复流水线。原 winner-union 重复执行成本没有计作首次候选发现成本。",
        "", "## 冻结与复现", "",
        f"协议 SHA256 `{sha(OUT / 'PROTOCOL.json')}`；runner SHA256 `{current_runner}`；v48 `fast_accept.py` SHA256 `{protocol['algorithm_source']['fast_accept_sha256']}`；规则 SHA256 `{protocol['algorithm_source']['acceptance_rules_sha256']}`。v49 runner 的唯一运行后 amendment 修正只读 v46 成本预测文件的路径；没有修改算法或协议，协议 hash 前后相同。",
        "", "原始失败状态与日志保持在 `RUN_STATUS.json`、`FAILED_ATTEMPTS.jsonl`、`run.log` 和 `run_resume.log`。数值诊断为重建状态，不覆盖原失败记录。复现及证据路径见 `REPRODUCE.md`。",
        "", "Luna Max 是用户请求的设置，但本接口无法确认或切换模型设置。结果没有被解释为 ReQuant 官方复现，也没有原创贡献主张。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reproduce = f'''# v51 evidence and reproduction\n\nThis run stopped at the frozen v48 initial direct/quadratic guard. Do not resume or bypass the guard. `RUN_STATUS.json` and failure logs remain the original stop records.\n\n## Reproduce the CPU evidence audit\n\n```powershell\ncd E:\\quant-thesis\n& .\\.venv\\Scripts\\python.exe experiments/range_search_w4/results/v51_m2_fast_integer_repair_fullmodel/finalize_incomplete_v51.py\n```\n\n## Original run entry\n\nThe frozen run entry was:\n\n```powershell\n& .\\.venv\\Scripts\\python.exe experiments/range_search_w4/results/v51_m2_fast_integer_repair_fullmodel/run_v51.py\n```\n\nThe attempt then resumed once after the read-only cost-projection path correction:\n\n```powershell\n& .\\.venv\\Scripts\\python.exe experiments/range_search_w4/results/v51_m2_fast_integer_repair_fullmodel/run_v51.py --resume\n```\n\nBoth calls, the block0 checkpoints, failure attempts, and the no-search raw-X diagnostic are recorded. Re-running the stopped search is not part of this deliverable.\n\n## Evidence\n\n- `PROTOCOL.json`, `PROTOCOL_AMENDMENT_001.json`, `RUNNER_DERIVATION.json`, `RUNNER_AMENDMENT_LOG.json`, `RUNTIME_PREFLIGHT.json`\n- `SOURCE_AUDIT.json`, `M2_MODULE_GRID_AUDIT.csv`, `PRE_RESUME_AUDIT.json`\n- `RUNTIME_BINDING.json`, `formal/groups/layer_00_*.pt`, `PROPAGATION_RESTORE_CHECKS.csv`\n- `diagnostics/INITIAL_QUAD_FAILURE.json`, `diagnostics/INITIAL_QUAD_FAILURE_ROWS.csv`, `diagnostics/INITIAL_QUAD_FAILURE_REPLAY2.log`\n- `PARTIAL_STAGE_COSTS.csv`, `COST_LEDGER.csv`, `HISTORICAL_QUALITY_COMPARISON.csv`, `FINAL_DECISION.json`\n'''
    (OUT / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")

    manifest_excluded = {"DELIVERY_MANIFEST.json", "finalize.log"}
    files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in manifest_excluded:
            files.append({"path": str(p.relative_to(OUT)), "sha256": sha(p), "bytes": p.stat().st_size})
    writej("DELIVERY_MANIFEST.json", {"status": "STOPPED_D", "files": files, "historical_artifacts_modified": False, "full_model_checkpoint": False, "new_ppl": False, "created_unix": time.time()})
    print("V51_FINAL_AUDIT_WRITTEN", len(files), "files", "D", flush=True)

if __name__ == "__main__":
    main()
