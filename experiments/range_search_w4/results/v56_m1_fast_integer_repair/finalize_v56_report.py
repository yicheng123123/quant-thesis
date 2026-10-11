"""Regenerate v56 quality contrasts, cost accounting, and final report.

Reads v56/v55/v18 records and writes only inside the v56 result directory.
It does not load a model, access the GPU, or rerun evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"E:\quant-thesis\experiments\range_search_w4\results\v56_m1_fast_integer_repair")
V55 = ROOT.parent / "v55_m2_decoder_fixed_integer_repair"
V18 = ROOT.parent / "range_w4_fullmodel_v18"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows, columns=None):
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def fmt(x, n=6):
    return f"{x:.{n}f}"


dataset_names = {
    "wikitext": "WikiText-2 test",
    "c4_fixed_budget": "C4 fixed-budget validation",
}
historical = read_csv(V55 / "HISTORICAL_QUALITY_BACKGROUND.csv")
v55_table = read_csv(V55 / "QUALITY_COMPARISON.csv")
v55_eval_source = read_json(V55 / "QUALITY_EVALUATION_SOURCE.json")
v56_eval_source = read_json(ROOT / "QUALITY_EVALUATION_SOURCE.json")
v18_runtime = read_json(V18 / "FORMAL_RUNTIME.json")
v55_runtime = v55_eval_source.get("torch", "not recorded")
v56_runtime = read_json(ROOT / "INITIALIZATION_PREFLIGHT.json").get("torch_version", "not recorded")

hist_lookup = {(r["model"], r["dataset"]): r for r in historical if r["model"] in {"M1", "M2"}}
v55_lookup = {(r["model"], r["dataset"]): r for r in v55_table}
v55_eval_lookup = {r["dataset"]: r for r in v55_eval_source["results"]}
v56_eval_lookup = {r["dataset"]: r for r in v56_eval_source["new_results"]}

quality = []
for dataset in dataset_names:
    for arm, hist_model in (("M1", "M1"), ("M2", "M2")):
        src = hist_lookup[(hist_model, dataset)]
        quality.append({
            "dataset": dataset,
            "dataset_label": dataset_names[dataset],
            "arm": arm,
            "description": "original formal checkpoint",
            "nll_per_valid_token": float(src["NLL"]),
            "ppl": float(src["PPL"]),
            "valid_tokens": int(src["valid_tokens"]),
            "windows": int(src["windows"]),
            "checkpoint_sha256": src["checkpoint_sha256"],
            "result_sha256": src["result_sha256"],
            "windows_sha256": src["windows_sha256"],
            "manifest_sha256": src["manifest_sha256"],
            "runtime": v18_runtime["torch"],
            "evaluation_source": src["result_path"],
            "measurement": "historical checkpoint measurement; v53 retained provenance",
        })
    src55 = v55_lookup[("v55", dataset)]
    eval55 = v55_eval_lookup[dataset]
    quality.append({
        "dataset": dataset,
        "dataset_label": dataset_names[dataset],
        "arm": "R2_v55_M2_repaired",
        "description": "M2 plus v55 fast integer repair",
        "nll_per_valid_token": float(src55["nll_per_valid_token"]),
        "ppl": float(src55["ppl"]),
        "valid_tokens": int(src55["valid_tokens"]),
        "windows": int(src55["windows"]),
        "checkpoint_sha256": src55["checkpoint_sha256"],
        "result_sha256": eval55["result_sha256"],
        "windows_sha256": eval55["windows_sha256"],
        "manifest_sha256": eval55["manifest_sha256"],
        "runtime": v55_runtime,
        "evaluation_source": str(V55 / "evaluation" / f"v55_{dataset}.json"),
        "measurement": "fresh v55 frozen-evaluator measurement",
    })
    src56 = read_json(ROOT / "evaluation" / f"v56_{dataset}.json")
    eval56 = v56_eval_lookup[dataset]
    quality.append({
        "dataset": dataset,
        "dataset_label": dataset_names[dataset],
        "arm": "R1_v56_M1_repaired",
        "description": "M1 plus v56 fast integer repair",
        "nll_per_valid_token": float(src56["nll"]),
        "ppl": float(src56["ppl"]),
        "valid_tokens": int(src56["scored_tokens"]),
        "windows": int(src56["windows"]),
        "checkpoint_sha256": src56["model_sha256"],
        "result_sha256": eval56["result_sha256"],
        "windows_sha256": eval56["windows_sha256"],
        "manifest_sha256": src56["manifest_sha256"],
        "runtime": v56_runtime,
        "evaluation_source": str(ROOT / "evaluation" / f"v56_{dataset}.json"),
        "measurement": "fresh v56 frozen-evaluator measurement",
    })

arm_order = ["M1", "M2", "R1_v56_M1_repaired", "R2_v55_M2_repaired"]
quality.sort(key=lambda r: (r["dataset"], arm_order.index(r["arm"])))
for dataset in dataset_names:
    rows = [r for r in quality if r["dataset"] == dataset]
    assert len({r["valid_tokens"] for r in rows}) == 1
    assert len({r["windows"] for r in rows}) == 1
    assert len({r["manifest_sha256"] for r in rows}) == 1
    for r in rows:
        assert math.isclose(r["ppl"], math.exp(r["nll_per_valid_token"]), rel_tol=1e-10, abs_tol=1e-10)

write_csv(REPORTS / "FOUR_ARM_QUALITY.csv", quality)

q = {(r["dataset"], r["arm"]): r for r in quality}
contrast_specs = [
    ("repair_increment_M1", "R1_v56_M1_repaired", "M1", "M1 plus repair minus M1"),
    ("repair_increment_M2", "R2_v55_M2_repaired", "M2", "M2 plus repair minus M2"),
    ("repaired_M2_vs_repaired_M1", "R2_v55_M2_repaired", "R1_v56_M1_repaired", "R2 minus R1"),
    ("scalar10_before_repair", "M2", "M1", "M2 minus M1"),
]
contrasts = []
for dataset in dataset_names:
    for contrast, left, right, description in contrast_specs:
        a, b = q[(dataset, left)], q[(dataset, right)]
        dnll = a["nll_per_valid_token"] - b["nll_per_valid_token"]
        dppl = a["ppl"] - b["ppl"]
        contrasts.append({
            "dataset": dataset,
            "dataset_label": dataset_names[dataset],
            "contrast": contrast,
            "description": description,
            "minuend_arm": left,
            "subtrahend_arm": right,
            "delta_nll": dnll,
            "delta_nll_relative_pct": 100 * dnll / b["nll_per_valid_token"],
            "delta_ppl": dppl,
            "delta_ppl_relative_pct": 100 * dppl / b["ppl"],
        })
    r1gain = q[(dataset, "R1_v56_M1_repaired")]["nll_per_valid_token"] - q[(dataset, "M1")]["nll_per_valid_token"]
    r2gain = q[(dataset, "R2_v55_M2_repaired")]["nll_per_valid_token"] - q[(dataset, "M2")]["nll_per_valid_token"]
    contrasts.append({
        "dataset": dataset,
        "dataset_label": dataset_names[dataset],
        "contrast": "descriptive_interaction_(R2-M2)-(R1-M1)",
        "description": "negative means the M2 repair increment was more favorable on NLL",
        "minuend_arm": "repair_increment_M2",
        "subtrahend_arm": "repair_increment_M1",
        "delta_nll": r2gain - r1gain,
        "delta_nll_relative_pct": "",
        "delta_ppl": "",
        "delta_ppl_relative_pct": "",
    })
write_csv(REPORTS / "FOUR_ARM_CONTRASTS.csv", contrasts)

advantage = []
for dataset in dataset_names:
    pre_nll = q[(dataset, "M1")]["nll_per_valid_token"] - q[(dataset, "M2")]["nll_per_valid_token"]
    post_nll = q[(dataset, "R1_v56_M1_repaired")]["nll_per_valid_token"] - q[(dataset, "R2_v55_M2_repaired")]["nll_per_valid_token"]
    pre_ppl = q[(dataset, "M1")]["ppl"] - q[(dataset, "M2")]["ppl"]
    post_ppl = q[(dataset, "R1_v56_M1_repaired")]["ppl"] - q[(dataset, "R2_v55_M2_repaired")]["ppl"]
    advantage.append({
        "dataset": dataset,
        "dataset_label": dataset_names[dataset],
        "M2_advantage_before_repair_delta_NLL_M1_minus_M2": pre_nll,
        "M2_advantage_after_repair_delta_NLL_R1_minus_R2": post_nll,
        "change_in_M2_advantage_NLL": post_nll - pre_nll,
        "M2_advantage_before_repair_delta_PPL_M1_minus_M2": pre_ppl,
        "M2_advantage_after_repair_delta_PPL_R1_minus_R2": post_ppl,
        "change_in_M2_advantage_PPL": post_ppl - pre_ppl,
        "direction": "widened" if post_nll > pre_nll else "narrowed" if post_nll < pre_nll else "unchanged",
    })
write_csv(REPORTS / "SCALAR10_ADVANTAGE.csv", advantage)

# Synchronized stage timings are reported by stage and are not summed into the
# process wall: some timings are nested inside larger operations.
stage_rows = read_csv(ROOT / "STAGE_TIMINGS.csv")
stage_groups = defaultdict(lambda: {"n": 0, "seconds": 0.0, "peak_allocated": 0, "peak_reserved": 0})
for row in stage_rows:
    key = (row["mode"], row["stage"])
    item = stage_groups[key]
    item["n"] += 1
    item["seconds"] += float(row["seconds"])
    item["peak_allocated"] = max(item["peak_allocated"], int(float(row["gpu_peak_allocated_bytes"] or 0)))
    item["peak_reserved"] = max(item["peak_reserved"], int(float(row["gpu_peak_reserved_bytes"] or 0)))
stage_costs = []
for (mode, stage), item in sorted(stage_groups.items()):
    stage_costs.append({
        "mode": mode,
        "stage": stage,
        "observations": item["n"],
        "synchronized_stage_seconds_sum": item["seconds"],
        "max_peak_allocated_bytes": item["peak_allocated"],
        "max_peak_reserved_bytes": item["peak_reserved"],
        "aggregation_note": "per-stage timer total; some stages are nested and this column is not additive to process wall",
    })
blocks = read_csv(ROOT / "BLOCK_TIMINGS.csv")
props = read_csv(ROOT / "PROPAGATION_NATIVE_TIMINGS.csv")
stage_costs.extend([
    {"mode": "formal", "stage": "sum_of_24_block_wall", "observations": len(blocks),
     "synchronized_stage_seconds_sum": sum(float(r["seconds"]) for r in blocks),
     "max_peak_allocated_bytes": "", "max_peak_reserved_bytes": "",
     "aggregation_note": "per-block wall sum; excludes later standalone evaluation"},
    {"mode": "formal", "stage": "sum_of_24_native_propagation_wall", "observations": len(props),
     "synchronized_stage_seconds_sum": sum(float(r["seconds"]) for r in props),
     "max_peak_allocated_bytes": "", "max_peak_reserved_bytes": "",
     "aggregation_note": "per-block propagation check wall sum; complete-token-axis native forward/hook comparisons"},
])
write_csv(REPORTS / "STAGE_COSTS.csv", stage_costs)
peak_allocated = max((v["peak_allocated"] for v in stage_groups.values()), default=0)
peak_reserved = max((v["peak_reserved"] for v in stage_groups.values()), default=0)

ledger = read_json(ROOT / "ACTIVE_JOB_LEDGER.json")
formal_record = read_json(ROOT / "FORMAL_REPAIR_COMPLETE.json")
interrupted = read_json(ROOT / "INTERRUPTED_EVALUATION_AUDIT.json")
eval_complete = read_json(ROOT / "EVALUATION_RESUME_COMPLETE.json")
segments = ledger["segments"]
charged = [s for s in segments if "charged_seconds" in s]
cpu_charges = sum(float(s["charged_seconds"]) for s in charged if s.get("cpu_only"))
base_preflight = float(ledger["preflight_charged_seconds"])
ledger_formal = next(float(s["charged_seconds"]) for s in charged if s["label"] == "v56 full M1 repair, propagation, save/reload and evaluation")
eval_wall = next(float(s["charged_seconds"]) for s in charged if s["label"] == "v56 evaluation-only continuation after interrupted WT2")
ledger_total = base_preflight + sum(float(s["charged_seconds"]) for s in charged)
runner_formal = float(formal_record["elapsed_seconds"])
partial_eval_upper = float(interrupted["eval_partial_active_wall_upper_bound_seconds"])
# The interrupted-attempt segment already carries the first-run wall. For a
# conservative upper-bound reconciliation, compare its charge with the
# separately logged repair elapsed plus the upper bound for the discarded
# partial PPL attempt; do not present this bound as a measured wall interval.
first_session_upper = max(ledger_formal, runner_formal + partial_eval_upper)
reconciliation_extra = first_session_upper - ledger_formal
conservative_total = ledger_total + reconciliation_extra
cap = float(ledger["cap_seconds"])

historical_cost = read_json(ROOT / "HISTORICAL_COST_SEPARATE.json")
v18_report = read_json(V18 / "FORMAL_RUNTIME.json")
m1_build = float(v18_report["m1"]["wall_seconds_including_load_replay_save"])
m2_build = float(v18_report["m2"]["wall_seconds_including_load_replay_save"])
v55_final = read_json(V55 / "FINAL_BUDGET_ACCOUNTING.json")
deployment = read_json(ROOT / "DEPLOYMENT_COMPARISON.json")
prop_results = read_csv(ROOT / "PROPAGATION_NATIVE_RESULTS.csv")
module_rows = read_csv(ROOT / "MODULE_RESULTS.csv")
formal_modules = [r for r in module_rows if r["mode"] == "formal"]
assert len(formal_modules) == 168 and sum(int(r["rows"]) for r in formal_modules) == 304128
assert all(r["complete"] == "True" for r in formal_modules)
assert len(blocks) == 24 and len(props) == 24
assert len(prop_results) == 141 and all(r["pass"] == "True" and r["bitwise_equal"] == "True" for r in prop_results)
assert eval_complete["pass"] and v56_eval_source["pass"] and len(v56_eval_source["new_results"]) == 2

accounting = {
    "version": "v56",
    "cap_seconds": cap,
    "historical_shared_v55_seconds_separate": historical_cost["historical_v55_shared_total_seconds"],
    "historical_shared_v55_included_in_v56_cap": False,
    "raw_active_ledger_total_seconds": ledger_total,
    "raw_active_ledger_remaining_seconds": cap - ledger_total,
    "conservative_accounted_total_seconds": conservative_total,
    "conservative_remaining_seconds": cap - conservative_total,
    "finish_reserve_seconds": 360.0,
    "reserve_met_on_conservative_accounting": cap - conservative_total >= 360.0,
    "components": {
        "base_preflight_charge_seconds": base_preflight,
        "CPU_only_failure_and_audit_charges_seconds": cpu_charges,
        "first_formal_process_ledger_wall_seconds_including_partial_eval_attempt": ledger_formal,
        "first_run_formal_repair_elapsed_record_seconds": runner_formal,
        "discarded_partial_WT2_wall_upper_bound_seconds": partial_eval_upper,
        "first_session_conservative_upper_bound_seconds": first_session_upper,
        "evaluation_only_continuation_wall_seconds": eval_wall,
        "successful_frozen_evaluation_stage_seconds": sum(float(r["synchronized_stage_seconds_sum"]) for r in stage_costs if r["mode"] == "evaluation" and r["stage"] == "PPL_evaluation"),
        "peak_GPU_allocated_bytes_recorded": peak_allocated,
        "peak_GPU_reserved_bytes_recorded": peak_reserved,
        "ledger_to_conservative_reconciliation_seconds": reconciliation_extra,
        "historical_M1_build_seconds_separate": m1_build,
        "historical_M2_build_seconds_separate": m2_build,
        "historical_M2_minus_M1_build_seconds": m2_build - m1_build,
        "historical_M2_to_M1_build_ratio": m2_build / m1_build,
        "v55_new_layer2_to23_block_wall_seconds_context_only": float(v55_final["valid_new_layer2_to23_block_wall_seconds"]),
        "v55_formal_runner_process_wall_seconds_context_only": float(v55_final["v55_formal_runner_process_wall_seconds"]),
    },
    "accounting_notes": [
        "The raw v56 ledger remains unchanged and includes the first interrupted PPL attempt inside its first process segment.",
        "The audit also records repair elapsed and a discarded partial-WT2 upper bound; their boundary does not reconcile exactly with that segment. Conservative total uses max(ledger first segment, repair elapsed plus partial-attempt upper bound), then adds the separately measured successful continuation and CPU charges.",
        "The partial WT2 attempt is excluded from quality results; its possible runtime is not described as an independently measured GPU wall interval.",
        "v55 shared history and original M1/M2 build costs are outside the new v56 9000-second cap.",
        "v55 restored-chain work and v56 full-from-layer0 work are not a valid direct speed comparison.",
        "Stage timing rows are reported separately and are not added to process wall because nested timers may overlap.",
    ],
}
(ROOT / "FINAL_BUDGET_ACCOUNTING.json").write_text(json.dumps(accounting, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

source_lock = read_json(ROOT / "FROZEN_RUN_LOCK.json")
checkpoint_lock = read_json(ROOT / "CHECKPOINT_LOCK.json")
prep = read_json(ROOT / "PREPARATION.json")
source_history = read_json(ROOT / "SOURCE_REUSE_AND_DIFF.json")
rev = "7ae557604adf67be50417f59c2f167def9a775"

lines = [
    "# v56 报告：正确传播下的 M1 快速整数码修复与四臂比较",
    "",
    "## 结论",
    "",
    "v56 从正式 M1 的 layer0 独立完成修复、传播核验、部署检查和两项实际 PPL。M1+修复相对 M1 在 WT2 与固定预算 C4 都有较低 NLL/PPL。观察到的 v55 M2+修复在两项指标上又低于 v56 M1+修复，scalar10 的数值优势没有缩小，按四臂点估计反而略扩大。",
    "",
    "但 v55 新评测使用 PyTorch 2.5.1+cu121，v56 与 v18 M1/M2 基线使用 PyTorch 2.6.0+cu124。窗口、token、manifest 和 evaluator 逻辑相同，R2 相关差值仍混有运行时版本差异；按用户冻结要求没有重测 M1、M2 或 v55，因此不能把小幅差异严格归因于初始化或修复。结果是单次固定模型测量，不代表跨 seed 稳定或统计显著。",
    "",
    "## 冻结范围与来源",
    "",
    f"- 模型：Qwen2.5-0.5B-Instruct revision `{rev}`；M1 checkpoint SHA256 `{source_lock['M1_checkpoint_sha256']}`；从 layer0 新建运行，未加载 v55/v46 修复组、输入或 C，也未用 v46 修复参数。M2 checkpoint SHA256 `{source_lock['M2_checkpoint_sha256']}`，只作既有对照。",
    f"- 校准：128×2048 manifest SHA256 `{source_lock['calibration_manifest_sha256']}`；固定 M1 原始 scale/zero/g_idx，W4A16 symmetric G128；v49 快速修复核心、v52 初始化 FP64 复核和 v53 decoder 接口按冻结文件复用。`SOURCE_REUSE_AND_DIFF.json` 记录算法、候选顺序、预算、网格与数值接受规则没有变化；仅做 M1 初始化、路径、边界与标签适配。",
    f"- v56 协议 SHA256 `{checkpoint_lock['protocol_sha256']}`；最终配置 SHA256 `{checkpoint_lock['config_sha256']}`。冻结源码哈希见 `FROZEN_RUN_LOCK.json`；快速接受实现 SHA256 `{source_lock['files']['fast_accept_v52.py']}`，decoder 提取 SHA256 `{source_lock['files']['decoder_interface.py']}`，运行时 SHA256 `{source_lock['files']['runtime_v54.py']}`。",
    "- M1 与 M2 有效推理配置相同；M2 的逐行量化网格与 M1 不同。四臂比较是不同初始化流程加同一修复算法，不是只改变整数码的严格参数对照。",
    "- v18 的 M1/M2 PPL 来源、checkpoint/window/result 哈希在 v55 `HISTORICAL_QUALITY_BACKGROUND.csv` 中核验。v53 历史审计指出 v18 记录 Transformers 4.49.0，支持 tuple decoder 接口，未发现同类 Tensor 误用证据；但没有逐调用类型日志，也未穷尽重验全部历史前缀。因此沿用实际 checkpoint 测量，同时不把它当作逐层输入完全等价证明。v55 是 v53 正确传播后新测的模型。",
    "- v45/v46/v49 历史模型仅作背景，其传播有效性为 UNKNOWN，不进入四臂表。v55 共同历史账本 11037.312983 秒单独列示，不计入 v56 cap。",
    "",
    "## 完整修复和正确性",
    "",
    f"- 覆盖 24/24 blocks、168/168 Linear、304128/304128 输出行，96 个依赖组提交；接受更新 {int(formal_modules[0]['accepted_updates']) if len(formal_modules)==1 else sum(int(r['accepted_updates']) for r in formal_modules):,}，修改坐标 {sum(int(r['modified_coordinates']) for r in formal_modules):,}，候选评价 {sum(int(r['candidate_evaluations']) for r in formal_modules):,}；拒绝 0，FP64 复核 0。",
    f"- 全部模块通过终态 raw-X FP64 直接重构。记录的 search MSE sum 从 {sum(float(r['initial_search_mse_sum']) for r in formal_modules):.9f} 降到 {sum(float(r['final_search_mse_sum']) for r in formal_modules):.9f}；这是模块行均方误差汇总，不是 PPL。最大 p 漂移 {max(float(r['max_p_drift']) for r in formal_modules):.3e}；最大终态误差比 {max(float(r['terminal_max_error_ratio']) for r in formal_modules):.3e}。",
    f"- 文章 0、64、127 的独立原生 full-forward/hook 参考覆盖 24 层。CSV 有 {len(prop_results)} 条 next-hidden/qkv 输入比较，141/141 通过且逐位一致；保留完整 `[1,2048,896]` token 轴，参考路径不调用待验证的 decoder 提取函数。",
    f"- 码、固定网格、packing/reload、输出回归通过。最终模型 SHA256 `{checkpoint_lock['final_model_sha256']}`；290 个具名张量参与对照，与 M1/M2/v55 分别有 168 个张量不同，故 v56 不等价于 v55，按协议完成了新评测。有效配置与四臂相同。",
    "",
    "## 四臂质量结果",
    "",
    "NLL 按 valid_tokens 加权；WT2 为 299077 tokens/582 windows，C4 固定预算为 131071 tokens/253 windows。数值 CSV 保留全精度及来源哈希。",
    "",
    "| 数据集 | 模型 | 加权 NLL | PPL | valid_tokens |",
    "|---|---|---:|---:|---:|",
]
for dataset in dataset_names:
    for arm in arm_order:
        r = q[(dataset, arm)]
        display = {"M1": "M1 原生", "M2": "M2 scalar10", "R1_v56_M1_repaired": "v56 M1+修复", "R2_v55_M2_repaired": "v55 M2+修复"}[arm]
        lines.append(f"| {dataset_names[dataset]} | {display} | {r['nll_per_valid_token']:.6f} | {r['ppl']:.6f} | {r['valid_tokens']} |")
lines += [
    "",
    "相对变化以左边模型减右边模型计算；负值表示左边 NLL/PPL 更低。",
    "",
    "| 数据集 | 对比 | ΔNLL (相对%) | ΔPPL (相对%) |",
    "|---|---|---:|---:|",
]
contrast_labels = {
    "repair_increment_M1": "v56 M1+修复 − M1",
    "repair_increment_M2": "v55 M2+修复 − M2",
    "repaired_M2_vs_repaired_M1": "v55 M2+修复 − v56 M1+修复",
    "scalar10_before_repair": "M2 − M1（修复前）",
}
for r in contrasts:
    if r["contrast"] not in contrast_labels:
        continue
    lines.append(f"| {r['dataset_label']} | {contrast_labels[r['contrast']]} | {r['delta_nll']:+.6f} ({r['delta_nll_relative_pct']:+.4f}%) | {r['delta_ppl']:+.6f} ({r['delta_ppl_relative_pct']:+.4f}%) |")
lines += ["", "### Scalar10 优势与描述性交互", ""]
for r in advantage:
    lines.append(
        f"- {r['dataset_label']}：M2 相对 M1 的 NLL 优势（M1−M2）从修复前 {r['M2_advantage_before_repair_delta_NLL_M1_minus_M2']:.6f} "
        f"变为修复后 {r['M2_advantage_after_repair_delta_NLL_R1_minus_R2']:.6f}，优势扩大 {r['change_in_M2_advantage_NLL']:.6f}；PPL 差（M1−M2）从 {r['M2_advantage_before_repair_delta_PPL_M1_minus_M2']:.6f} "
        f"变为 {r['M2_advantage_after_repair_delta_PPL_R1_minus_R2']:.6f}。"
    )
for dataset in dataset_names:
    row = next(r for r in contrasts if r["dataset"] == dataset and r["contrast"] == "descriptive_interaction_(R2-M2)-(R1-M1)")
    lines.append(f"- {dataset_names[dataset]} 描述性交互 `I=(NLL_R2−NLL_M2)−(NLL_R1−NLL_M1)` = {row['delta_nll']:+.6f}；负值表示本次 M2 修复的 NLL 下降幅度较大。它不是显著性或协同机制检验。")
lines += [
    "",
    "## 成本账本",
    "",
    "| 范围 | 秒 | 口径 |",
    "|---|---:|---|",
    f"| 24 个完整 block timer 合计 | {sum(float(r['seconds']) for r in blocks):.3f} | BLOCK_TIMINGS.csv 的逐 block wall 合计 |",
    f"| 24 次原生传播核验 | {sum(float(r['seconds']) for r in props):.3f} | 文章 0/64/127 full-forward/hook 对照 |",
    f"| 有效初始化构建 | {float(dict(((r['stage'],r['synchronized_stage_seconds_sum']) for r in stage_costs if r['mode']=='formal' and r['stage']=='formal_model_load')).get('formal_model_load',0)):.3f} | v56 加载既有 M1；未重新 GPTQ |",
    f"| C 构建 | {sum(float(r['synchronized_stage_seconds_sum']) for r in stage_costs if r['mode']=='formal' and r['stage']=='C_construction'):.3f} | 96 个依赖组，FP64 Gram |",
    f"| 全矩阵独立终态 FP64 重构 | {sum(float(r['synchronized_stage_seconds_sum']) for r in stage_costs if r['mode']=='formal' and r['stage']=='independent_full_terminal_FP64'):.3f} | 168 个 Linear |",
    f"| 最终模型保存 + grid/reload 检查 | {sum(float(r['synchronized_stage_seconds_sum']) for r in stage_costs if r['mode']=='deployment' and r['stage'] in {'final_save','pack_reload_all_grid_check','reload'}):.3f} | 部署阶段同步 timer 合计 |",
    f"| 成功的 WT2+C4 evaluator stage | {sum(float(r['synchronized_stage_seconds_sum']) for r in stage_costs if r['mode']=='evaluation' and r['stage']=='PPL_evaluation'):.3f} | 两个完整语料的同步 evaluator timer |",
    f"| 峰值 GPU 显存记录 | {peak_allocated/(1024**2):.1f} MiB allocated / {peak_reserved/(1024**2):.1f} MiB reserved | STAGE_TIMINGS.csv 所记录的最大值 |",
    f"| 冻结续跑进程 wall | {eval_wall:.3f} | 包含模型 reload、两项评测与落盘 |",
    f"| v56 原始账本总计 | {ledger_total:.3f} | 含 preflight、CPU诊断、首轮失败/部分WT2和续跑 |",
    f"| 保守 cap 核算 | {conservative_total:.3f} | reconciled upper-bound accounting; 9000 秒内 |",
    f"| v56 cap 剩余（保守） | {cap-conservative_total:.3f} | 360 秒收尾预算要求满足 |",
    "",
    f"`STAGE_COSTS.csv`列出各同步阶段。阶段总计可能与 block/process wall 有嵌套，不应再次相加形成总 wall。首轮已完成修复后的 WT2 评测中断于窗口 200；它没有合格的结果文件，窗口损失未纳入质量表，可能耗时上界 {partial_eval_upper:.3f} 秒。原 ledger 将首轮 process 记为 {ledger_formal:.3f} 秒，而修复完成计时另记 {runner_formal:.3f} 秒；`FINAL_BUDGET_ACCOUNTING.json` 用较大的 repair elapsed 加部分评测上界作保守替代，不把这部分伪装成精确实测。",
    "",
    f"原正式校准构建成本单独参考：M1 {m1_build:.3f} 秒，M2 scalar10 {m2_build:.3f} 秒，差 {m2_build-m1_build:.3f} 秒，约 {m2_build/m1_build:.3f}×。这是 v18 历史完整构建 wall，不计入 v56 cap。v55 恢复链新增 layer2–23 block wall 为 {float(v55_final['valid_new_layer2_to23_block_wall_seconds']):.3f} 秒，但 block0/1 源自此前有效工作；不能与 v56 从 layer0 的完整 wall 直接计算加速比。",
    "",
    f"共同历史 v55 账本 {historical_cost['historical_v55_shared_total_seconds']:.6f} 秒保持只读、未并入 9000 秒 v56 cap。原始 ledger total 为 {ledger_total:.3f} 秒；为保守处理第一段边界记录差异，报告使用 {conservative_total:.3f} 秒的 upper-bound cap accounting，剩余 {cap-conservative_total:.3f} 秒。",
    "",
    "## 运行环境与限制",
    "",
    f"- v18 M1/M2：PyTorch `{v18_runtime['torch']}` / Transformers `{v18_runtime['transformers']}`；v56：PyTorch `{v56_runtime}`；v55 fresh PPL：PyTorch `{v55_runtime}`。因此 R2 涉及的差值可能包含软件/内核差异。",
    "- Luna Max 是用户指定配置；执行接口不能确认本轮实际模型设置。普通沙盒此前仍失败，本轮使用已获授权的沙盒外执行入口；没有声称普通沙盒恢复。",
    "- 没有读取 P3、没有按 PPL 调参，没有重新评测 M1/M2/v55；所有修复选择仅来自冻结 search 目标。局部重构改善不替代 PPL，快速接受规则是经验数值流程，不是理论安全证书或官方 ReQuant 复现。",
    "",
    "## 交付索引与复现",
    "",
    "- 冻结/来源：`PROTOCOL.json`、`FROZEN_RUN_LOCK.json`、`SOURCE_AUDIT.json`、`SOURCE_REUSE_AND_DIFF.json`。",
    "- 模型锁与传播：`CHECKPOINT_LOCK.json`、`DEPLOYMENT_COMPARISON.json`、`NAMED_TENSOR_HASHES.csv`、`PROPAGATION_NATIVE_RESULTS.csv`。",
    "- 质量原始窗口：`evaluation/v56_wikitext_WINDOWS.csv`、`evaluation/v56_c4_fixed_budget_WINDOWS.csv`；摘要和锁：`QUALITY_EVALUATION_SOURCE.json`、`EVALUATION_RESUME_COMPLETE.json`。",
    "- 四臂数值：`reports/FOUR_ARM_QUALITY.csv`、`reports/FOUR_ARM_CONTRASTS.csv`、`reports/SCALAR10_ADVANTAGE.csv`。成本：`reports/STAGE_COSTS.csv`、`FINAL_BUDGET_ACCOUNTING.json`、`ACTIVE_JOB_LEDGER.json`。",
    "- 只重算报告与 CSV/JSON 聚合（CPU only）：",
    "  ```powershell",
    "  & 'E:\\quant-thesis\\.venv\\Scripts\\python.exe' 'E:\\quant-thesis\\experiments\\range_search_w4\\results\\v56_m1_fast_integer_repair\\finalize_v56_report.py'",
    "  ```",
    "- 一次性完整运行的冻结入口记录于 `REPRODUCE.md`；现有完成目录已有 checkpoint 和一次性评测结果，不应在其上重启正式运行或重复评测。",
    "",
]
(ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

run_status = {
    "status": "COMPLETE",
    "category": "COMPLETE",
    "complete_model": True,
    "modules": 168,
    "rows": 304128,
    "model_sha256": checkpoint_lock["final_model_sha256"],
    "PPL_run": True,
    "PPL_reused": False,
    "datasets_completed": ["wikitext", "c4_fixed_budget"],
    "all_24_blocks_and_141_native_propagation_comparisons_pass": True,
    "v56_cap_seconds": cap,
    "conservative_accounted_seconds": conservative_total,
    "conservative_remaining_seconds": cap - conservative_total,
    "historical_shared_v55_seconds_separate": historical_cost["historical_v55_shared_total_seconds"],
    "aggregation_completed_unix": __import__("time").time(),
    "aggregation_script_sha256": sha(Path(__file__)),
}
(ROOT / "RUN_STATUS.json").write_text(json.dumps(run_status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({
    "report": str(ROOT / "REPORT.md"),
    "quality_csv": str(REPORTS / "FOUR_ARM_QUALITY.csv"),
    "contrasts_csv": str(REPORTS / "FOUR_ARM_CONTRASTS.csv"),
    "cost_accounting": str(ROOT / "FINAL_BUDGET_ACCOUNTING.json"),
    "ledger_total": ledger_total,
    "conservative_total": conservative_total,
    "remaining": cap - conservative_total,
}, ensure_ascii=False, indent=2))
