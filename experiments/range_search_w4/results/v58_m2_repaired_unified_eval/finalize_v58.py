"""Independent CPU aggregation and report generation for the completed v58 eval."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(r"E:\quant-thesis")
BASE = ROOT / "experiments/range_search_w4/results"
OUT = BASE / "v58_m2_repaired_unified_eval"
V55 = BASE / "v55_m2_decoder_fixed_integer_repair"
V56 = BASE / "v56_m1_fast_integer_repair"
POST_REPORTING_CONSERVATIVE_SECONDS = 5.0

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

expected = {
    "wikitext": (299078, 299077, 582, "acd46b20a6b84b534d5625bfab505d599001679a775d619960a40e8da4f62f46", "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3"),
    "c4_fixed_budget": (131072, 131071, 253, "0b0d70d70d3c1e0ba62126c81a3f3e97531e58d6701236fb8b3315450b7188d4", "bc9a7575527abbc41352352a0439c6bee062ea265b7bd03120704e83f4af7053"),
}
comparison = []
history = []
artifact_rows = []
for ds, (input_tokens, valid_tokens, windows_n, manifest_sha, token_sha) in expected.items():
    result_path = OUT / "evaluation" / f"v58_{ds}.json"
    windows_path = OUT / "evaluation" / f"v58_{ds}_WINDOWS.csv"
    result = read_json(result_path)
    with windows_path.open(encoding="utf8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert result["status"] == "PASS"
    assert result["manifest_sha256"] == manifest_sha
    assert result["token_ids_sha256_i64le"] == token_sha
    assert int(result["input_tokens"]) == input_tokens
    assert int(result["valid_tokens"]) == valid_tokens
    assert len(rows) == windows_n
    assert len({int(r["window_index"]) for r in rows}) == windows_n
    assert [int(r["window_index"]) for r in rows] == list(range(windows_n))
    for i, r in enumerate(rows):
        expected_begin = i * 512
        expected_end = min(expected_begin + 2048, input_tokens)
        assert int(r["begin"]) == expected_begin and int(r["end"]) == expected_end
        assert int(r["valid_tokens"]) > 0
    nll_sum = sum(float(r["window_nll"]) for r in rows)
    token_sum = sum(int(r["valid_tokens"]) for r in rows)
    nll = nll_sum / token_sum
    ppl = math.exp(nll)
    assert token_sum == valid_tokens
    assert math.isclose(nll, float(result["weighted_nll"]), rel_tol=0, abs_tol=1e-10)
    assert math.isclose(ppl, float(result["ppl"]), rel_tol=1e-12, abs_tol=1e-12)

    v55_hist = read_json(V55 / "evaluation" / f"v55_{ds}.json")
    v56_ref = read_json(V56 / "evaluation" / f"v56_{ds}.json")
    history.append({
        "dataset": ds,
        "v55_historical_nll": float(v55_hist["nll"]),
        "v58_current_nll": nll,
        "nll_delta_v58_minus_v55_history": nll - float(v55_hist["nll"]),
        "v55_historical_ppl": float(v55_hist["ppl"]),
        "v58_current_ppl": ppl,
        "ppl_absolute_delta": ppl - float(v55_hist["ppl"]),
        "ppl_relative_delta_percent": (ppl / float(v55_hist["ppl"]) - 1.0) * 100.0,
        "interpretation": "runtime/execution difference only; not attributed to a specific software component",
    })
    comparison.append({
        "dataset": ds,
        "M1_plus_repair_v56_nll_context_only": float(v56_ref["nll"]),
        "M1_plus_repair_v56_ppl_context_only": float(v56_ref["ppl"]),
        "M2_plus_repair_v55_v58_nll": nll,
        "M2_plus_repair_v55_v58_ppl": ppl,
        "absolute_nll_delta_M2repair_minus_M1repair": nll - float(v56_ref["nll"]),
        "absolute_ppl_delta_M2repair_minus_M1repair": ppl - float(v56_ref["ppl"]),
        "relative_ppl_delta_percent": (ppl / float(v56_ref["ppl"]) - 1.0) * 100.0,
        "both_lower_in_observed_values": nll < float(v56_ref["nll"]) and ppl < float(v56_ref["ppl"]),
        "strict_same_runtime_admissible": False,
        "status": "context only; v56 eval process runtime incomplete",
    })
    artifact_rows += [
        {"path": str(result_path.relative_to(OUT)), "sha256": sha(result_path), "kind": "complete_result_json"},
        {"path": str(windows_path.relative_to(OUT)), "sha256": sha(windows_path), "kind": "complete_window_table"},
    ]

write_csv(OUT / "HISTORICAL_V55_DIFFERENCE.csv", history)
write_csv(OUT / "UNIFIED_COMPARISON.csv", comparison)

env = read_json(OUT / "ENVIRONMENT_LOCK.json")
load = read_json(OUT / "LOADER_CHECK.json")
cost = read_json(OUT / "COST_ACCOUNTING.json")
budget = read_json(OUT / "BUDGET_STATE.json")
precheck = read_json(OUT / "COST_PRECHECK.json")
source = read_json(OUT / "SOURCE_AUDIT.json")
diag_path = OUT / "DIAGNOSTIC_FAILURES.jsonl"
diagnostics = [json.loads(x) for x in diag_path.read_text(encoding="utf8").splitlines() if x.strip()] if diag_path.exists() else []

v18_protocol = read_json(BASE / "range_w4_fullmodel_v18/PROTOCOL.json")
model_revision = v18_protocol["model"]["revision"]
env.update({
    "model_id": v18_protocol["model"]["id"],
    "model_revision": model_revision,
    "tokenizer_revision": model_revision,
    "tokenizer_source": "frozen token-ID manifests; no retokenization in the evaluator",
    "attention_requested_and_loaded": load["loaded_config_attention_implementation"],
    "device_used_for_v56": "NVIDIA GeForce RTX 4060 Laptop GPU per v56 protocol",
    "v56_attention_actual_kernel": "NOT_MEASURED",
})
(OUT / "ENVIRONMENT_LOCK.json").write_text(json.dumps(env, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")

source_rows = [
    {"arm": "M2_plus_repair_v55_rerun", "checkpoint": str(V55 / "R1/checkpoint/final/model.safetensors"), "checkpoint_sha256": source["v55_model_sha256"], "named_tensor_manifest_sha256": source["v55_named_tensor_manifest_sha256"], "named_tensor_count": source["v55_named_tensor_count"], "runtime": env["pytorch"] + "; Transformers " + env["transformers"], "evaluation_source": "v58 actual evaluation", "source_status": "PASS"},
    {"arm": "M1_plus_repair_v56_context", "checkpoint": str(V56 / "M1_REPAIRED/checkpoint/final/model.safetensors"), "checkpoint_sha256": source["v56_checkpoint_sha256"], "named_tensor_manifest_sha256": "v56 manifest audited in v57", "named_tensor_count": 290, "runtime": "torch 2.6.0+cu124; Python and Transformers evaluation-process versions missing", "evaluation_source": "v56 complete windows; context only", "source_status": "strict runtime identity UNKNOWN"},
    {"arm": "M1_M2_historical", "checkpoint": "v18 formal checkpoints; see v57 CHECKPOINT_SOURCE_AUDIT.csv", "checkpoint_sha256": "audited in v57", "named_tensor_manifest_sha256": "not available in v18; v57 recomputed", "named_tensor_count": 290, "runtime": "torch 2.6.0+cu124; Transformers 4.49.0; Python not recorded", "evaluation_source": "historical v18 windows", "source_status": "background only; runtime differs"},
    {"arm": "M2_plus_repair_v55_historical", "checkpoint": str(V55 / "R1/checkpoint/final/model.safetensors"), "checkpoint_sha256": source["v55_model_sha256"], "named_tensor_manifest_sha256": source["v55_named_tensor_manifest_sha256"], "named_tensor_count": source["v55_named_tensor_count"], "runtime": "torch 2.5.1+cu121; Python and Transformers versions missing", "evaluation_source": "v55 historical windows", "source_status": "background only; PyTorch runtime differs"},
]
write_csv(OUT / "SOURCE_AUDIT.csv", source_rows)

runtime_rows = [
    {"dimension": "Python", "v56_successful_eval": "NOT_RECORDED", "v58": env["python"], "same_confirmed": False},
    {"dimension": "PyTorch", "v56_successful_eval": source["v56_protocol_torch"], "v58": env["pytorch"], "same_confirmed": True},
    {"dimension": "CUDA runtime/build", "v56_successful_eval": "12.4 per frozen protocol", "v58": env["cuda_build"], "same_confirmed": True},
    {"dimension": "Transformers", "v56_successful_eval": "NOT_RECORDED; saved config version is save-time metadata", "v58": env["transformers"], "same_confirmed": False},
    {"dimension": "tokenizers package", "v56_successful_eval": "NOT_RECORDED", "v58": env["tokenizers"], "same_confirmed": False},
    {"dimension": "model revision/effective config", "v56_successful_eval": model_revision + "; effective config " + load["config_effective_sha256"], "v58": model_revision + "; effective config " + load["config_effective_sha256"], "same_confirmed": True},
    {"dimension": "dtype/device", "v56_successful_eval": "FP16 / RTX 4060 Laptop GPU per protocol", "v58": load["parameter_dtype"] + " / " + env["device"], "same_confirmed": True},
    {"dimension": "attention", "v56_successful_eval": "SDPA requested; lower kernel NOT_MEASURED", "v58": load["loaded_config_attention_implementation"] + " requested/loaded; lower kernel NOT_MEASURED", "same_confirmed": "requested_config_only"},
    {"dimension": "TF32/use_cache", "v56_successful_eval": "TF32 off; use_cache false", "v58": "TF32 off; use_cache false", "same_confirmed": True},
    {"dimension": "token manifests/scoring", "v56_successful_eval": "frozen WT2/C4 manifests; maxlen2048 stride512 valid-token weighted", "v58": "same hashes, windows and score", "same_confirmed": True},
    {"dimension": "evaluator wrapper", "v56_successful_eval": source["v56_evaluator_wrapper_sha256"], "v58": env["evaluator_wrapper_sha256"], "same_confirmed": True},
]
write_csv(OUT / "RUNTIME_COMPARISON.csv", runtime_rows)
input_rows = []
for ds in ("wikitext", "c4_fixed_budget"):
    r = read_json(OUT / "evaluation" / f"v58_{ds}.json")
    input_rows.append({"dataset": ds, "manifest_sha256": r["manifest_sha256"], "token_ids_sha256_i64le": r["token_ids_sha256_i64le"], "input_tokens": r["input_tokens"], "valid_tokens": r["valid_tokens"], "windows": r["windows"], "max_length": r["max_length"], "stride": r["stride"], "all_windows_complete": True})
write_csv(OUT / "INPUT_SOURCE_AUDIT.csv", input_rows)

measured_runner_seconds = float(budget["completed_active_seconds"])
budget_charged_seconds = measured_runner_seconds + POST_REPORTING_CONSERVATIVE_SECONDS
cost.update({
    "measured_runner_active_seconds": measured_runner_seconds,
    "post_evaluation_report_and_audit_conservative_seconds": POST_REPORTING_CONSERVATIVE_SECONDS,
    "total_budget_charged_seconds_including_postprocessing": budget_charged_seconds,
    "total_budget_remaining_seconds": 3600.0 - budget_charged_seconds,
    "postprocessing_charge_is_conservative_not_device_time": True,
})
budget.update({
    "post_evaluation_report_and_audit_conservative_seconds": POST_REPORTING_CONSERVATIVE_SECONDS,
    "budget_charged_active_seconds_including_postprocessing": budget_charged_seconds,
    "remaining_seconds_after_postprocessing_charge": 3600.0 - budget_charged_seconds,
})
status_path = OUT / "RUN_STATUS.json"
status = read_json(status_path)
status["budget_charged_seconds_including_postprocessing"] = budget_charged_seconds
status["remaining_seconds_including_postprocessing_charge"] = 3600.0 - budget_charged_seconds
status["postprocessing_charge_seconds"] = POST_REPORTING_CONSERVATIVE_SECONDS
status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
for p, doc in [(OUT / "COST_ACCOUNTING.json", cost), (OUT / "BUDGET_STATE.json", budget)]:
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")

table = "\n".join(
    f"| {r['dataset']} | {r['M2_plus_repair_v55_v58_nll']:.9f} | {r['M2_plus_repair_v55_v58_ppl']:.9f} | {r['M1_plus_repair_v56_nll_context_only']:.9f} | {r['M1_plus_repair_v56_ppl_context_only']:.9f} | {r['absolute_nll_delta_M2repair_minus_M1repair']:.9f} | {r['relative_ppl_delta_percent']:.4f}% |"
    for r in comparison
)
hist_table = "\n".join(
    f"| {r['dataset']} | {r['v55_historical_ppl']:.9f} | {r['v58_current_ppl']:.9f} | {r['ppl_absolute_delta']:+.9f} | {r['ppl_relative_delta_percent']:+.6f}% |"
    for r in history
)

report = f"""# v58 — v55 M2 修复模型统一评测运行时重评

## 结论

v55 checkpoint 在本轮冻结输入上完成 WT2 test 与固定预算 C4 validation 两项实际评测。相对 v56 的 M1＋修复历史评测，v55 M2＋修复的本轮点估计在两项数据上都更低（WT2 NLL 低 {comparison[0]['absolute_nll_delta_M2repair_minus_M1repair']:.9f} nats/token，C4 低 {comparison[1]['absolute_nll_delta_M2repair_minus_M1repair']:.9f} nats/token）。

但这不是可确认的同运行时主对照：v56 成功评测进程没有记录 Python 和 Transformers 版本，且未记录实际选择的底层 SDPA kernel。按本轮冻结规则，**不能据此断言 M2 初始化修复模型在两项指标上已被证明优于 M1 初始化修复模型**。观察值方向一致，比较资格不足。v56 数值只作上下文参考；未计算四臂交互量。

## 完整评测结果与上下文比较

本轮按完整窗口的 `sum(window_nll) / sum(valid_tokens)` 计算加权 NLL，再以 `exp(NLL)` 计算 PPL。两个本轮数据集全部窗口均完成，独立 CSV 汇总通过。

| 数据集 | M2＋修复 v55/v58 NLL | M2＋修复 PPL | M1＋修复 v56 NLL（上下文） | M1＋修复 PPL（上下文） | NLL 差（M2−M1） | PPL 相对差 |
|---|---:|---:|---:|---:|---:|---:|
{table}

本轮 v55 实测窗口：WT2 为 {comparison[0]['M2_plus_repair_v55_v58_nll']:.9f} NLL、{comparison[0]['M2_plus_repair_v55_v58_ppl']:.9f} PPL、299,077 valid tokens、582 窗口；C4 为 {comparison[1]['M2_plus_repair_v55_v58_nll']:.9f} NLL、{comparison[1]['M2_plus_repair_v55_v58_ppl']:.9f} PPL、131,071 valid tokens、253 窗口。

### v55 checkpoint 历史评测差异

| 数据集 | v55 历史 PPL | v58 当前 PPL | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
{hist_table}

差异只描述运行时/执行结果变化，不归因于某一软件组件。历史校准传播有效性仍为 `UNKNOWN`；本轮统一评测不补充该证据。

## 来源与运行时

- v55 `model.safetensors` SHA256：`{source['v55_model_sha256']}`；checkpoint lock、config SHA 和 290 个具名张量来源清单均核对通过。v57 已对 290 个张量逐项重算并与 v55 清单匹配；本轮再次核对 checkpoint 与清单 SHA。
- 模型与 tokenizer revision 均为 `{model_revision}`；实际评分直接使用冻结 token-ID manifest，不重新分词。
- loader：Transformers `AutoModelForCausalLM.from_pretrained`，`local_files_only=True`、FP16、`attn_implementation="sdpa"`、`eval()`、CUDA。加载后 dtype/device 为 `{load['parameter_dtype']}` / `{load['parameter_device']}`，模型处于 eval，归一化配置 SHA 为 `{load['config_effective_sha256']}`，与预检一致。evaluator 显式使用 `use_cache=False`，TF32 关闭。
- 当前评测进程：Python {env['python']}、PyTorch {env['pytorch']}、CUDA build {env['cuda_build']}、Transformers {env['transformers']}、tokenizers {env['tokenizers']}、设备 {env['device']}。
- v56 协议记录 PyTorch 2.6.0+cu124，并请求 SDPA；v56 成功评测进程的 Python/Transformers 版本缺失。v56 保存 config 内 `transformers_version` 属于保存时元数据，不能证明评测进程版本。底层 SDPA kernel 两边都未实测。因此完整 runtime identity 为未证实。
- v56 wrapper evaluator SHA 与本轮相同：`{env['evaluator_wrapper_sha256']}`。本轮当前 core evaluator `benchmark/eval/ppl.py` SHA 为 `{env['core_evaluator_sha256_current']}`；v56 未单独保存当时 core 文件 SHA。
- tokenizer 不重新分词；使用锁定 manifest 中的原始 token ID。WT2 manifest/token SHA 与 C4 manifest/token SHA均匹配冻结记录，窗口边界为 max_length=2048、stride=512，mask 仅计新增目标 token，不添加 special token，不使用 chat template。
- v18 的 M1/M2 历史测量使用 Transformers 4.49.0；v55 历史 M2＋修复使用 PyTorch 2.5.1+cu121。四臂统一运行时证据不成立，不计算交互量。

## 成本与失败

- v58 预算上限：3600 秒；runner 实测活动时间：{measured_runner_seconds:.3f} 秒；再保守计入报告与审计 CPU 时间 {POST_REPORTING_CONSERVATIVE_SECONDS:.1f} 秒，总预算计费 {budget_charged_seconds:.3f} 秒，余额 {3600.0-budget_charged_seconds:.3f} 秒。历史账本未重置、未混算。
- 本轮实际模型加载：{cost['model_load_synchronized_wall_seconds']:.3f} 秒；WT2 评测：{cost['dataset_eval_wall_seconds_excluding_load']['wikitext']:.3f} 秒；C4 评测：{cost['dataset_eval_wall_seconds_excluding_load']['c4_fixed_budget']:.3f} 秒。另有约束准备、写盘、聚合等进程时间；完整 runner process wall {cost['process_wall_including_loader_and_evaluation']:.3f} 秒。
- 另保守计入预运行只读核查 30 秒；首次 CPU 预检遇到 v57 CSV UTF-8 BOM 表头，发生在模型加载及前向前，失败尝试约 {diagnostics[0]['elapsed_seconds_including_conservative_readonly_audit_charge']:.3f} 秒（其中包含上述 30 秒保守核查额）；修正 BOM 读取后成功完成。未发生 GPU 模型前向失败。
- v56 实测 eval-only 成功 continuation wall 为 2283.576 秒（含加载），只用于预估；v58 本轮实测活动时间为 {cost['total_charged_active_seconds']:.3f} 秒。两者差异不据此归因于特定软件或硬件因素。

## 交付和复现

- 原始窗口：`evaluation/v58_wikitext_WINDOWS.csv`、`evaluation/v58_c4_fixed_budget_WINDOWS.csv`；逐窗口可恢复 journal 位于同名数据集子目录。
- 结果与独立聚合：`evaluation/v58_wikitext.json`、`evaluation/v58_c4_fixed_budget.json`。
- 来源、环境、loader、预算、适格性：`SOURCE_AUDIT.json`、`ENVIRONMENT_LOCK.json`、`LOADER_CHECK.json`、`COST_ACCOUNTING.json`、`FOUR_ARM_ADMISSIBILITY.json`。
- 对照明细：`SOURCE_AUDIT.csv`、`RUNTIME_COMPARISON.csv`、`INPUT_SOURCE_AUDIT.csv`。
- 两模型数值上下文表：`UNIFIED_COMPARISON.csv`（明确标为不适格的上下文比较）；v55 历史差异：`HISTORICAL_V55_DIFFERENCE.csv`。
- 复现命令：`.venv\\Scripts\\python.exe experiments\\range_search_w4\\results\\v58_m2_repaired_unified_eval\\run_v58_eval.py`，工作目录为 `E:\\quant-thesis`。断点续跑时保持相同 lock 与 manifest，完成窗口将通过输入/掩码/lock 哈希校验后跳过。

本轮没有量化、修复、候选搜索、新校准或 P3 读取；不声称统计显著、跨 seed 稳定、传播正确性已由本轮证明，或算法原创性。
"""
(OUT / "REPORT.md").write_text(report, encoding="utf8")
(OUT / "REPRODUCE.md").write_text(
    "# Reproduce v58\n\n"
    "PowerShell, from `E:\\quant-thesis`:\n\n"
    "```powershell\n"
    ".venv\\Scripts\\python.exe experiments\\range_search_w4\\results\\v58_m2_repaired_unified_eval\\run_v58_eval.py\n"
    "```\n\n"
    "The runner verifies the v55 lock and frozen manifests, uses the unchanged `benchmark.eval.ppl.evaluate_ppl`, and atomically journals each complete window. On resume it validates the same run lock, token slice and label mask before skipping a completed window. It does not read P3 or modify any checkpoint. `finalize_v58.py` performs CPU-only CSV aggregation and report generation.\n",
    encoding="utf8",
)

# Hashes are generated last and exclude themselves to avoid recursive identity.
files = [OUT / p for p in [
    "PROTOCOL.json", "SOURCE_AUDIT.json", "ENVIRONMENT_LOCK.json", "LOADER_CHECK.json",
    "COST_PRECHECK.json", "COST_ACCOUNTING.json", "RUN_STATUS.json", "BUDGET_STATE.json", "SOURCE_AUDIT.csv", "RUNTIME_COMPARISON.csv", "INPUT_SOURCE_AUDIT.csv",
    "UNIFIED_COMPARISON.csv", "HISTORICAL_V55_DIFFERENCE.csv", "REPORT.md", "REPRODUCE.md",
    "run_v58_eval.py", "finalize_v58.py", "evaluation/v58_wikitext.json", "evaluation/v58_wikitext_WINDOWS.csv",
    "evaluation/v58_c4_fixed_budget.json", "evaluation/v58_c4_fixed_budget_WINDOWS.csv",
]]
rows_hash = [{"path": str(p.relative_to(OUT)), "sha256": sha(p), "bytes": p.stat().st_size} for p in files]
write_csv(OUT / "ARTIFACT_HASHES.csv", rows_hash)
print(json.dumps({"status": "FINALIZED", "comparison_rows": comparison, "history_rows": history}, indent=2))
