from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent


def read_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write_json(name: str, value):
    (OUT / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    budget = read_json("BUDGET.json")
    phase_seconds = sum(float(x["wall_seconds"]) for x in budget["phase_ledger"])
    cpu_ledger = budget.setdefault("cpu_correctness_audit_ledger", [])
    if not any(x.get("phase") == "B_FULL_PARITY_AND_TENSOR_HASHES" for x in cpu_ledger):
        cpu_ledger.append({
            "phase": "B_FULL_PARITY_AND_TENSOR_HASHES",
            "wall_seconds": 14.574195,
            "measurement": "PowerShell Measure-Command around analyze_v40.py B",
        })
    if not any(x.get("phase") == "A_FINAL_AUDIT_REFRESH" for x in cpu_ledger):
        cpu_ledger.append({
            "phase": "A_FINAL_AUDIT_REFRESH",
            "wall_seconds": 12.1,
            "measurement": "functions.exec command wall rounded to 0.1s around CPU-only analyze_v40.py A refresh after B completed",
        })
    cpu_seconds = sum(float(x["wall_seconds"]) for x in cpu_ledger)
    budget.update({
        "status": "ALL_RUNS_AND_AUDITS_COMPLETE",
        "active_phase": None,
        "active_phase_wall_seconds": 0.0,
        "phase_ledger_elapsed_seconds": phase_seconds,
        "gpu_runner_wall_seconds": phase_seconds,
        "cpu_correctness_audit_seconds": cpu_seconds,
        "gpu_experiment_elapsed_seconds": phase_seconds + cpu_seconds,
        "total_budget_accounted_seconds": phase_seconds + cpu_seconds,
        "total_budget_accounting_definition": (
            "Conservative sum of phase-supervisor wall for PILOT/A/B plus all measured CPU parity/hash audit walls; "
            "it is budget accounting, not active GPU kernel time."
        ),
        "final_audit_utc": datetime.now(timezone.utc).isoformat(),
        "final_audit": {
            "pilot_parity": "PASS",
            "A_run_validity": "PASS",
            "B_run_validity": "PASS",
            "all_286_boundary_hash_comparisons": "PASS",
            "normalized_final_tensor_hashes_v18_A_B": "PASS",
            "held_or_P3_read": False,
            "PPL_run": False,
        },
    })
    write_json("BUDGET.json", budget)

    a_report = read_json("A/M2/RUN_REPORT.json")
    b_report = read_json("B/M2/RUN_REPORT.json")
    pilot = read_json("PILOT_PARITY.json")
    a_valid = read_json("A_RUN_VALIDITY.json")
    b_valid = read_json("B_RUN_VALIDITY.json")
    hashes = read_json("FINAL_TENSOR_HASHES.json")

    a_wall = float(a_report["wall_seconds_including_load_replay_save"])
    b_wall = float(b_report["wall_seconds_including_load_replay_save"])
    a_wrapper = float(budget["phase_ledger"][1]["wall_seconds"])
    b_wrapper = float(budget["phase_ledger"][2]["wall_seconds"])
    pilot_wrapper = float(budget["phase_ledger"][0]["wall_seconds"])
    calls_a = int(a_report["formal_counts"]["full_linear_alpha_candidate_executions"])
    calls_b = int(b_report["formal_counts"]["full_linear_alpha_candidate_executions"])
    slots_a = int(a_report["formal_counts"]["row_alpha_score_slots"])
    slots_b = int(b_report["formal_counts"]["row_alpha_score_slots"])
    stage_rows = list(csv.DictReader((OUT / "STAGE_TIMING.csv").open(encoding="utf-8", newline="")))
    stage_by_arm = {(r["arm"], r["stage"]): r for r in stage_rows}

    def seconds(arm: str, stage: str) -> float:
        return float(stage_by_arm[(arm, stage)]["stage_wall_seconds_sum_nonadditive"])

    def peak_gib(arm: str, field: str) -> float:
        values = [int(r[field]) for r in stage_rows if r["arm"] == arm]
        return max(values) / (1024**3)

    names = ["PILOT", "A", "B"]
    wrapper_walls = [pilot_wrapper, a_wrapper, b_wrapper]
    method_walls = [None, a_wall, b_wall]
    candidate_counts = [27, calls_a, calls_b]
    score_slots = [49792, slots_a, slots_b]
    with (OUT / "RUN_WALLS.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "method_run_wall_seconds", "supervisor_wall_seconds", "primary_candidate_gptq_calls", "row_alpha_score_slots", "separate_alpha1_parity_calls", "gpu_budget_accounted_wall"])
        for i, name in enumerate(names):
            parity_calls = 7 if name == "PILOT" else 35
            writer.writerow([name, "" if method_walls[i] is None else f"{method_walls[i]:.6f}", f"{wrapper_walls[i]:.6f}", candidate_counts[i], score_slots[i], parity_calls, "yes"])

    summary = {
        "status": "PASS",
        "experiment": "v40 independent per-Linear historical winner-union candidate execution",
        "candidate_manifest_sha256": read_json("FREEZE_LOCK.json")["candidate_manifest_sha256"],
        "counts": {
            "linears": 168,
            "rows": 304128,
            "A_scalar10_full_linear_candidate_gptq_executions": calls_a,
            "B_frozen_winner_union_full_linear_candidate_gptq_executions": calls_b,
            "candidate_execution_reduction_fraction": (calls_a - calls_b) / calls_a,
            "A_row_alpha_score_slots": slots_a,
            "B_row_alpha_score_slots": slots_b,
            "row_alpha_score_slot_reduction_fraction": (slots_a - slots_b) / slots_a,
            "pilot_primary_candidate_calls": 27,
            "alpha1_native_parity_calls": {"pilot": 7, "A": 35, "B": 35},
        },
        "correctness": {
            "pilot": pilot["status"],
            "A": a_valid["status"],
            "B": b_valid["status"],
            "modules_each_full_run": 168,
            "layers_each_full_run": 24,
            "boundary_hashes_each_full_run": 119,
            "layer_state_hashes_each_full_run": 24,
            "A_all_candidate_scores_and_selected_state_exact_to_v18": True,
            "B_all_scored_union_candidates_and_selected_state_exact_to_v18": True,
            "boundary_parity_rows": 286,
            "boundary_parity_all_equal": True,
        },
        "final_model": {
            "named_tensor_count": 290,
            "v18_A_B_normalized_named_tensor_sha256": hashes["runs"]["A"]["normalized_named_tensor_sha256"],
            "v18_A_B_model_file_sha256": hashes["runs"]["A"]["model_file_sha256"],
            "same_quantization_inference_configuration": True,
            "historical_v18_quality_result_applicable_to_identical_model": True,
            "new_PPL_run": False,
        },
        "timing": {
            "A_method_wall_seconds": a_wall,
            "B_method_wall_seconds": b_wall,
            "paired_speedup_A_over_B": a_wall / b_wall,
            "wall_saved_seconds": a_wall - b_wall,
            "wall_reduction_fraction": (a_wall - b_wall) / a_wall,
            "A_supervisor_wall_seconds": a_wrapper,
            "B_supervisor_wall_seconds": b_wrapper,
            "paired_supervisor_speedup_A_over_B": a_wrapper / b_wrapper,
            "A_full_linear_gptq_seconds_instrumented": seconds("A", "full_linear_alpha_candidate_gptq"),
            "B_full_linear_gptq_seconds_instrumented": seconds("B", "full_linear_alpha_candidate_gptq"),
            "A_direct_scoring_seconds_instrumented": seconds("A", "all_row_direct_output_scoring"),
            "B_direct_scoring_seconds_instrumented": seconds("B", "all_row_direct_output_scoring"),
            "stage_sums_are_nonadditive": True,
            "peak_allocated_gib": {"A": peak_gib("A", "gpu_peak_allocated_bytes"), "B": peak_gib("B", "gpu_peak_allocated_bytes")},
            "peak_reserved_gib": {"A": peak_gib("A", "gpu_peak_reserved_bytes"), "B": peak_gib("B", "gpu_peak_reserved_bytes")},
        },
        "budget": {
            "hard_cap_seconds": 5400.0,
            "phase_wall_seconds": phase_seconds,
            "CPU_correctness_audit_seconds": cpu_seconds,
            "total_accounted_seconds": phase_seconds + cpu_seconds,
            "failed_attempts": budget.get("failed_attempts", []),
        },
        "not_run": ["new held evaluation", "P3", "new PPL", "other model/revision/data migration"],
    }
    write_json("SUMMARY.json", summary)

    report = f"""# v40 — 逐 Linear 赢家并集独立运行

## 结论

**正确性：PASS。** 完整的 B 缩减候选运行与原 M2 在全部 168 个 Linear 的候选分数、逐行选择、量化权重、scale、zero、g_idx、输入/Hessian 和依赖边界上匹配。A 与 B 的最终 290 个具名模型张量及推理配置都与历史 v18 M2 相同。

**成本：配对实测校准 wall 从 {a_wall:.2f}s 降到 {b_wall:.2f}s，节省 {a_wall-b_wall:.2f}s（{(a_wall-b_wall)/a_wall*100:.2f}%），加速比 {a_wall/b_wall:.3f}×。** 这是本轮各一次完整独立运行的配对实测，不是由候选数推算。按监督进程计时，A={a_wrapper:.2f}s、B={b_wrapper:.2f}s，加速比 {a_wrapper/b_wrapper:.3f}×。

## 冻结比较

| 项目 | A：原 scalar10 | B：历史赢家并集 |
|---|---:|---:|
| 全部 Linear | 168 | 168 |
| 主候选完整 Linear GPTQ 次数 | {calls_a:,} | {calls_b:,} |
| 相对 A 的候选执行减少 | — | {(calls_a-calls_b)/calls_a*100:.2f}% |
| row-alpha score slots | {slots_a:,} | {slots_b:,} |
| 相对 A 的 score slots 减少 | — | {(slots_a-slots_b)/slots_a*100:.2f}% |
| 全运行 alpha=1 parity 辅助调用 | 35 | 35 |
| 方法计时 wall（含模型加载、capture/replay、保存/重载检查） | {a_wall:.3f}s | {b_wall:.3f}s |
| 观测峰值显存 allocated / reserved | {peak_gib('A','gpu_peak_allocated_bytes'):.3f} / {peak_gib('A','gpu_peak_reserved_bytes'):.3f} GiB | {peak_gib('B','gpu_peak_allocated_bytes'):.3f} / {peak_gib('B','gpu_peak_reserved_bytes'):.3f} GiB |

B 的 675 个候选来自运行前冻结的 candidate_budget_precheck 逐 Linear 赢家并集，强制包含 alpha=1，并遵守原 scalar10 顺序；没有在本轮根据结果补充候选。预检集合本身依赖此前已经完成的 10 点搜索，因此 675 次只表示在候选集合已知时的重执行量，不消除首次发现赢家并集的成本。

## 正确性证据

- 完整 block0 pilot 完成 7 个模块，27 个候选 GPTQ、49,792 个 row-alpha score slots，另有 7 个 alpha=1 原生 parity 调用；逐行 alpha/分数、量化状态及 5 个边界通过。
- A 与 B 分别从新加载的 pristine 模型启动；两者不共享候选张量、Hessian、activation cache 或 checkpoints。共同使用的只有冻结的原始 calibration manifest。
- A 复现 v18 scalar10 全部 {slots_a:,} 个直接候选分数；B 复核本轮全部 {slots_b:,} 个赢家并集候选分数。两臂 168/168 模块的 alpha、直接损失、W、scale、zero、g_idx、输入及 Hessian 对应值均精确匹配 v18。
- 两臂各检查 24 个 layer state、119 个依赖边界；`BOUNDARY_PARITY.csv` 中 286 条比较全部相等。
- v18、A、B 均为 290 个具名张量；规范化聚合 SHA256=`{hashes['runs']['A']['normalized_named_tensor_sha256']}`。A/B 最终模型文件 SHA256=`{hashes['runs']['A']['model_file_sha256']}`。config、generation config、模型 revision、校准 manifest 与量化推理语义相同。元数据文件自身包含不同的 v40 来源记录，其元数据文件 hash 不同，但规范化张量和推理配置一致。
- A/B 保存后重载的 64-token logits 检查最大及平均绝对差均为 0（各自相对运行内参考）。未运行新的 held、P3 或 PPL。因为最终模型张量和推理配置精确等同，历史 M2 的质量评测可对应此模型；本轮没有产生新的质量评测或质量提升结论。

## 成本拆分

| 阶段（阶段时间为 instrumentation 累计，非端到端加和项） | A | B |
|---|---:|---:|
| 完整 Linear 候选 GPTQ | {seconds('A','full_linear_alpha_candidate_gptq'):.3f}s | {seconds('B','full_linear_alpha_candidate_gptq'):.3f}s |
| 原直接全行评分 | {seconds('A','all_row_direct_output_scoring'):.3f}s | {seconds('B','all_row_direct_output_scoring'):.3f}s |
| 唯一组 Hessian 构建 | {seconds('A','unique_group_hessian_build'):.3f}s | {seconds('B','unique_group_hessian_build'):.3f}s |
| 共享原生 H-inverse 因子准备 | {seconds('A','shared_native_hinv_prepare'):.3f}s | {seconds('B','shared_native_hinv_prepare'):.3f}s |
| qkv → o 输入刷新 | {seconds('A','qkv_commit_to_o_input'):.3f}s | {seconds('B','qkv_commit_to_o_input'):.3f}s |
| o → up/gate 输入刷新 | {seconds('A','o_commit_to_up_gate_input'):.3f}s | {seconds('B','o_commit_to_up_gate_input'):.3f}s |
| up/gate → down 输入刷新 | {seconds('A','up_gate_commit_to_down_input'):.3f}s | {seconds('B','up_gate_commit_to_down_input'):.3f}s |
| down → 下一层输入刷新 | {seconds('A','down_commit_to_next_layer_input'):.3f}s | {seconds('B','down_commit_to_next_layer_input'):.3f}s |
| 额外 alpha=1 parity | {seconds('A','alpha1_native_parity_reference'):.3f}s | {seconds('B','alpha1_native_parity_reference'):.3f}s |
| 层 checkpoint 写入 / 最终模型保存 | {seconds('A','atomic_layer_checkpoint_io'):.3f}s / {seconds('A','final_dense_model_save'):.3f}s | {seconds('B','atomic_layer_checkpoint_io'):.3f}s / {seconds('B','final_dense_model_save'):.3f}s |

阶段累计值来自内层同步计时，`STAGE_TIMING.csv` 明确标注为 non-additive，不应相加替代 run wall。B 的主要节省来自候选 GPTQ 与直接评分：Hessian/因子准备、前缀输入刷新和提交开销基本不变，因此端到端 wall 降幅小于候选调用降幅。

本轮没有单独追加 kernel warm-up 调用；顺序在冻结前确定为 pilot → A → B。每个完整运行 wall 包含新模型加载、输入捕获/传播和最终保存/重载检查；全局预算还保守计入 phase supervisor wall 与 CPU parity/hash 审计。日志记录 `expandable_segments not supported` 的 PyTorch allocator 警告和 SDPA sliding-window 提示；没有异常退出或 OOM，最终张量与边界校验通过。

## 预算与结论边界

- pilot/A/B phase supervisor wall 总计 **{phase_seconds:.3f}s**；CPU correctness/hash 审计 **{cpu_seconds:.3f}s**；预算账本累计 **{phase_seconds+cpu_seconds:.3f}s / 5,400s**。失败尝试为 0。
- B 在冻结的历史模型 revision、calibration、GPTQ 配置和数据协议上执行正确且复现 M2。
- 节省适用于“逐 Linear 赢家并集已由历史 10 点搜索获得、配置与输入协议保持相同、需要再次执行校准”的复用情形。它不证明该集合迁移到新数据、模型或 revision，也没有抵扣首次搜索成本。
- 不读新 held/P3，不运行新 PPL；不把离线校准节省表述为在线推理加速。

## 文件索引

- 冻结协议：`FROZEN_PROTOCOL.json`
- 候选清单及来源：`CANDIDATE_MANIFEST.json`
- 任务核算：`JOB_TABLE.csv`
- pilot：`PILOT_PARITY.json`
- 逐模块核验：`MODULE_PARITY.csv`、`A_RUN_VALIDITY.json`、`B_RUN_VALIDITY.json`
- 激活边界：`BOUNDARY_PARITY.csv`
- 最终张量与配置 hash：`FINAL_TENSOR_HASHES.json` / `.csv`
- 阶段计时：`STAGE_TIMING.csv`；端到端计时：`RUN_WALLS.csv`
- 预算：`BUDGET.json`
- 汇总：`SUMMARY.json`
- 复现步骤：`REPRODUCE.md`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    reproduce = """# v40 reproduction and audit commands

Run from `E:\\quant-thesis` with the frozen GPTQ environment.

The new GPU runs are complete. Do not rerun them over this finished result directory. The following read-only check validates the frozen manifest and saved source hashes:

```powershell
.\\.venv-gptq\\Scripts\\python.exe experiments\\range_search_w4\\results\\range_w4_winner_union_v40\\verify_v40_freeze.py
```

The recorded GPU-run command for each phase was the supervisor invocation below. If repeating the experiment, first make a clean sibling copy of this frozen v40 directory and run from that copy, preserving this completed output:

```powershell
.\\.venv-gptq\\Scripts\\python.exe experiments\\range_search_w4\\results\\range_w4_winner_union_v40\\run_v40_stage.py PILOT
.\\.venv-gptq\\Scripts\\python.exe experiments\\range_search_w4\\results\\range_w4_winner_union_v40\\run_v40_stage.py A
.\\.venv-gptq\\Scripts\\python.exe experiments\\range_search_w4\\results\\range_w4_winner_union_v40\\run_v40_stage.py B
```

Each phase is resumable at completed decoder-layer checkpoints, and the supervisor enforces the frozen cumulative execution cutoff. Phase B uses only the immutable `CANDIDATE_MANIFEST.json`; candidate order must remain the original scalar10 order and alpha=1 must remain included. Do not rebuild the manifest from current-run outcomes.

Saved parity/hash tables were generated with:

```powershell
.\\.venv-gptq\\Scripts\\python.exe experiments\\range_search_w4\\results\\range_w4_winner_union_v40\\analyze_v40.py pilot
.\\.venv-gptq\\Scripts\\python.exe experiments\\range_search_w4\\results\\range_w4_winner_union_v40\\analyze_v40.py A
.\\.venv-gptq\\Scripts\\python.exe experiments\\range_search_w4\\results\\range_w4_winner_union_v40\\analyze_v40.py B
```

The executed command lines, environments, phase logs, source hashes, raw module records, checkpoints and final model files are preserved under this v40 directory. The current artifact review does not run held evaluation, P3, or PPL.
"""
    (OUT / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")

    assert calls_a == 1680 and calls_b == 675
    assert slots_a == 3041280 and slots_b == 1287040
    assert pilot["status"] == "PASS" and a_valid["status"] == "PASS" and b_valid["status"] == "PASS"
    assert hashes["status"] == "PASS" and hashes["model_inference_quality_reusable_from_v18"]
    assert phase_seconds + cpu_seconds < float(budget["hard_cap_seconds"])
    print(json.dumps({
        "status": "FINALIZED",
        "budget_elapsed_seconds": phase_seconds + cpu_seconds,
        "candidate_reduction_percent": (calls_a - calls_b) / calls_a * 100,
        "wall_reduction_percent": (a_wall - b_wall) / a_wall * 100,
        "paired_speedup": a_wall / b_wall,
        "report": str(OUT / "REPORT.md"),
        "reproduce": str(OUT / "REPRODUCE.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
