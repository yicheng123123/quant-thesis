
from __future__ import annotations
import csv, hashlib, json, math, pathlib, platform, time
from typing import Any

T0 = time.perf_counter()
ROOT = pathlib.Path(r"E:\quant-thesis\experiments\range_search_w4\results")
OUT = ROOT / "v57_uniform_runtime_four_arm_eval"
V18 = ROOT / "range_w4_fullmodel_v18"
V55 = ROOT / "v55_m2_decoder_fixed_integer_repair"
V56 = ROOT / "v56_m1_fast_integer_repair"
if not OUT.is_dir():
    raise RuntimeError("v57 directory must be created before running this preflight")
generated_names = {
    "preflight_v57.py", "NAMED_TENSOR_HASHES.csv", "CHECKPOINT_SOURCE_AUDIT.csv",
    "CONFIG_RUNTIME_COMPARISON.json", "INPUT_SOURCE_AUDIT.csv", "HISTORICAL_WINDOW_RECALC.csv",
    "V57_EIGHT_EVAL_STATUS.csv", "UNIFIED_COMPARISON_PENDING.csv", "HISTORICAL_VS_V57_PENDING.csv",
    "COST_PRECHECK.json", "ENVIRONMENT_LOCK.json", "RUN_STATUS.json", "REPORT.md",
    "REPRODUCE.md", "PREFLIGHT_CPU_SECONDS.txt", "CPU_DIAGNOSTIC_LEDGER.csv"
}
if any(p.name not in generated_names for p in OUT.iterdir()):
    raise RuntimeError("Refusing to overwrite existing v57 artifacts")
progress_path = OUT / "V57_EIGHT_EVAL_STATUS.csv"
if progress_path.is_file():
    old_progress = list(csv.DictReader(progress_path.open(encoding="utf-8-sig", newline="")))
    if any(r.get("completed_windows", "0") not in ("", "0") for r in old_progress):
        raise RuntimeError("Refusing to overwrite v57 window progress")
progress_path = OUT / "V57_EIGHT_EVAL_STATUS.csv"
if progress_path.is_file():
    old_progress = list(csv.DictReader(progress_path.open(encoding="utf-8-sig", newline="")))
    if any(r.get("completed_windows", "0") not in ("", "0") for r in old_progress):
        raise RuntimeError("Refusing to overwrite v57 window progress")

def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def dump_json(name: str, value: Any) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def dump_csv(name: str, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def safe_manifest(path: pathlib.Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    headers = list(rows[0]) if rows else []
    name_col = next((x for x in ("tensor", "name", "key") if x in headers), None)
    hash_col = next((x for x in ("sha256", "tensor_sha256", "hash") if x in headers), None)
    if not name_col or not hash_col:
        return {}
    return {r[name_col]: r[hash_col] for r in rows}

arms = [
    {"arm":"M1","path":V18/"M1/checkpoint/final/model.safetensors","config":V18/"M1/checkpoint/final/config.json",
     "expected":"3678f867d064abc9730d392f279516223ef17de16351c577e0f1977e9b7bba33","lock":V18/"evaluation/SUMMARY.json","named":None,"origin":"v18"},
    {"arm":"M2","path":V18/"M2/checkpoint/final/model.safetensors","config":V18/"M2/checkpoint/final/config.json",
     "expected":"810d6329cdc51d995543dbb701e2368568b19459a44f340df1a8a0019d447eed","lock":V18/"evaluation/SUMMARY.json","named":None,"origin":"v18"},
    {"arm":"M1_plus_repair","path":V56/"M1_REPAIRED/checkpoint/final/model.safetensors","config":V56/"M1_REPAIRED/checkpoint/final/config.json",
     "expected":"9d768808bf1fd3581406df3538aebc00ed177eafc6e46372da679e9db85ccac0","lock":V56/"CHECKPOINT_LOCK.json","named":V56/"NAMED_TENSOR_HASHES.csv","origin":"v56"},
    {"arm":"M2_plus_repair","path":V55/"R1/checkpoint/final/model.safetensors","config":V55/"R1/checkpoint/final/config.json",
     "expected":"0fa87105e36deac2ead4c871e6e0a802590418e6ccacd616929cdf91ecc8ec0c","lock":V55/"FINAL_CHECKPOINT_LOCK.json","named":V55/"NAMED_TENSOR_HASHES.csv","origin":"v55"},
]
if any(not a["path"].is_file() or not a["config"].is_file() for a in arms):
    raise FileNotFoundError("Required model or config file missing")

# Package inventory does not initialize CUDA or execute a model.
import torch, transformers, tokenizers
from transformers import AutoConfig
from safetensors import safe_open
env_live={"python":platform.python_version(),"torch":torch.__version__,"torch_cuda_runtime":torch.version.cuda,
          "transformers":transformers.__version__,"tokenizers":tokenizers.__version__,
          "gpu_query_performed":False}

v18_protocol=read_json(V18/"PROTOCOL.json")
v56_protocol=read_json(V56/"PROTOCOL.json")
v56_quality=read_json(V56/"QUALITY_EVALUATION_SOURCE.json")
tensor_rows=[]; source_rows=[]; effective_configs={}; tensor_manifest_pass={}; raw_configs={}
volatile={"_name_or_path","name_or_path","transformers_version","_commit_hash"}

for a in arms:
    model_path=a["path"]; cfg_path=a["config"]
    model_sha=sha_file(model_path); config_sha=sha_file(cfg_path); cfg=read_json(cfg_path)
    raw_configs[a["arm"]]=cfg
    effective=AutoConfig.from_pretrained(str(cfg_path.parent),local_files_only=True).to_dict()
    effective={k:v for k,v in effective.items() if k not in volatile}
    effective_sha=hashlib.sha256(json.dumps(effective,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    effective_configs[a["arm"]]=effective_sha
    expected_named=safe_manifest(a["named"]) if a["named"] else None
    named_sha=sha_file(a["named"]) if a["named"] else ""
    tensors={}
    with safe_open(str(model_path),framework="pt",device="cpu") as sf:
        for key in sorted(sf.keys()):
            t=sf.get_tensor(key).contiguous()
            digest=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
            tensors[key]=digest
            tensor_rows.append({"arm":a["arm"],"tensor":key,"dtype":str(t.dtype).replace("torch.",""),
                                "shape":json.dumps(list(t.shape),separators=(",",":")),"sha256":digest})
    if expected_named is not None:
        tensor_manifest_pass[a["arm"]]=(expected_named==tensors)
    else:
        tensor_manifest_pass[a["arm"]]="new_v57_cpu_recomputation"
    if a["origin"]=="v18":
        lock_src="v18 evaluation SUMMARY lists exact model.safetensors SHA; standalone checkpoint-lock JSON absent"
        hist_env="torch=2.6.0+cu124; transformers=4.49.0; Python not recorded"
    elif a["origin"]=="v56":
        lock_src="v56 CHECKPOINT_LOCK plus NAMED_TENSOR_HASHES.csv"
        hist_env="torch=2.6.0+cu124; transformers=5.16.1; historical Python version not recorded"
    else:
        lock_src="v55 FINAL_CHECKPOINT_LOCK plus NAMED_TENSOR_HASHES.csv"
        hist_env="torch=2.5.1+cu121; transformers and Python not recorded in v55 evaluation source"
    source_rows.append({
        "arm":a["arm"],"checkpoint_path":str(model_path),"bytes":model_path.stat().st_size,
        "expected_checkpoint_sha256":a["expected"],"actual_checkpoint_sha256":model_sha,
        "checkpoint_sha_match":model_sha==a["expected"],"config_path":str(cfg_path),
        "config_sha256":config_sha,"effective_config_sha256":effective_sha,"model_type":cfg.get("model_type"),
        "architectures":json.dumps(cfg.get("architectures"),ensure_ascii=False),
        "config_torch_dtype":cfg.get("torch_dtype"),"config_transformers_version":cfg.get("transformers_version"),
        "named_tensor_count_recomputed":len(tensors),"named_tensor_manifest_path":str(a["named"]) if a["named"] else "computed by v57; no v18 named manifest",
        "named_tensor_manifest_sha256":named_sha,"named_tensor_manifest_matches":tensor_manifest_pass[a["arm"]],
        "checkpoint_source_record":lock_src,"historical_eval_runtime":hist_env})
dump_csv("NAMED_TENSOR_HASHES.csv",tensor_rows,["arm","tensor","dtype","shape","sha256"])
dump_csv("CHECKPOINT_SOURCE_AUDIT.csv",source_rows)
raw_base=raw_configs["M1"]
raw_config_differences={}
for arm,raw in raw_configs.items():
    changed={k:{"M1":raw_base.get(k),arm:raw.get(k)} for k in sorted(set(raw_base)|set(raw))
             if raw_base.get(k)!=raw.get(k)}
    raw_config_differences[arm]=changed
dump_json("CONFIG_RUNTIME_COMPARISON.json",{
    "loader":"Transformers AutoConfig.from_pretrained(local_files_only=True).to_dict()",
    "ignored_serialization_keys":sorted(volatile),
    "all_effective_configs_equal":len(set(effective_configs.values()))==1,
    "effective_config_sha256_by_arm":effective_configs,
    "raw_serialized_config_differences_vs_M1":raw_config_differences,
    "interpretation":"Raw JSON aliases may vary by Transformers serialization; compare normalized AutoConfig under the one current runtime."})
for r in source_rows:
    if not r["checkpoint_sha_match"]:
        raise RuntimeError("Checkpoint file SHA mismatch: "+r["arm"])
for arm,passed in tensor_manifest_pass.items():
    if passed is False:
        raise RuntimeError("Named tensor hash mismatch: "+arm)
if len(set(effective_configs.values()))!=1:
    raise RuntimeError("Effective configs differ")

# Check frozen manifest byte hashes and source-token identity references.
input_specs=[
("wikitext",V18/"evaluation/manifests/WIKITEXT_TEST_MANIFEST.json",
 "acd46b20a6b84b534d5625bfab505d599001679a775d619960a40e8da4f62f46",
 "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3",299078,299077,582),
("c4_fixed_budget",V18/"evaluation/manifests/C4_EVAL_MANIFEST.json",
 "0b0d70d70d3c1e0ba62126c81a3f3e97531e58d6701236fb8b3315450b7188d4",
 "bc9a7575527abbc41352352a0439c6bee062ea265b7bd03120704e83f4af7053",131072,131071,253)]
input_rows=[]
for dataset,path,expected,tokhash,inputs,valid,nwin in input_specs:
    actual=sha_file(path)
    input_rows.append({"dataset":dataset,"manifest_path":str(path),"expected_sha256":expected,"actual_sha256":actual,
                       "manifest_hash_match":actual==expected,"tokenizer_revision":v18_protocol["model"]["revision"],
                       "token_ids_sha256_i64le":tokhash,"input_tokens":inputs,"valid_tokens":valid,"expected_windows":nwin,
                       "max_length":2048,"stride":512,"add_special_tokens":False,"chat_template":False,
                       "score_mask":"new valid labels only; first input token unscored"})
dump_csv("INPUT_SOURCE_AUDIT.csv",input_rows)
if not all(r["manifest_hash_match"] for r in input_rows):
    raise RuntimeError("Evaluation manifest SHA mismatch")

# Historical raw windows arithmetic only; no new inference is run.
v18sum=read_json(V18/"evaluation/SUMMARY.json")
hist_specs=[
("M1","wikitext",V18/"evaluation/windows/M1_wikitext.csv",V18/"evaluation/SUMMARY.json","v18"),
("M1","c4_fixed_budget",V18/"evaluation/windows/M1_c4_fixed_budget.csv",V18/"evaluation/SUMMARY.json","v18"),
("M2","wikitext",V18/"evaluation/windows/M2_wikitext.csv",V18/"evaluation/SUMMARY.json","v18"),
("M2","c4_fixed_budget",V18/"evaluation/windows/M2_c4_fixed_budget.csv",V18/"evaluation/SUMMARY.json","v18"),
("M1_plus_repair","wikitext",V56/"evaluation/v56_wikitext_WINDOWS.csv",V56/"evaluation/v56_wikitext.json","v56"),
("M1_plus_repair","c4_fixed_budget",V56/"evaluation/v56_c4_fixed_budget_WINDOWS.csv",V56/"evaluation/v56_c4_fixed_budget.json","v56"),
("M2_plus_repair","wikitext",V55/"evaluation/v55_wikitext_WINDOWS.csv",V55/"evaluation/v55_wikitext.json","v55"),
("M2_plus_repair","c4_fixed_budget",V55/"evaluation/v55_c4_fixed_budget_WINDOWS.csv",V55/"evaluation/v55_c4_fixed_budget.json","v55")]
recalc=[]; hist={}
for arm,ds,path,result_path,origin in hist_specs:
    rows=list(csv.DictReader(path.open(encoding="utf-8-sig",newline="")))
    nll_sum=sum(float(x["window_nll"]) for x in rows); valid=sum(int(x["valid_tokens"]) for x in rows)
    nll=nll_sum/valid; ppl=math.exp(nll); obj=read_json(result_path)
    if origin=="v18":
        ref=obj[ds][arm]; ref_nll=float(ref["exact_window_weighted_average_nll"]); ref_ppl=float(ref["perplexity"])
        ref_valid=int(ref["evaluated_tokens"]); ref_count=int(ref["window_count"])
        runtime=f'torch={ref.get("torch_version")}; transformers={ref.get("transformers_version")}'
    else:
        ref_nll=float(obj["nll"]); ref_ppl=float(obj["ppl"]); ref_valid=valid; ref_count=int(obj["windows"])
        runtime="torch=2.6.0+cu124; transformers=5.16.1" if origin=="v56" else "torch=2.5.1+cu121; transformers=unknown"
    contiguous=all(int(row["window_index"])==i for i,row in enumerate(rows))
    recalc.append({"arm":arm,"dataset":ds,"record_type":"historical_source_recalc_not_v57",
        "window_file":str(path),"window_file_sha256":sha_file(path),"window_count":len(rows),
        "window_index_contiguous":contiguous,"nll_sum":format(nll_sum,".17g"),"valid_tokens":valid,
        "weighted_nll_recomputed":format(nll,".17g"),"ppl_recomputed":format(ppl,".17g"),
        "source_nll":format(ref_nll,".17g"),"source_ppl":format(ref_ppl,".17g"),
        "nll_abs_diff":format(abs(nll-ref_nll),".17g"),"ppl_abs_diff":format(abs(ppl-ref_ppl),".17g"),
        "valid_and_window_counts_match":valid==ref_valid and len(rows)==ref_count,"historical_runtime":runtime})
    hist[(arm,ds)]={"nll":nll,"ppl":ppl,"runtime":runtime}
dump_csv("HISTORICAL_WINDOW_RECALC.csv",recalc)

eval_status=[{"arm":a,"dataset":ds,"status":"NOT_STARTED_BUDGET_INFEASIBLE",
              "new_windows_file":"","completed_windows":0,"expected_windows":582 if ds=="wikitext" else 253,
              "new_nll_sum":"","new_valid_tokens":"","new_nll":"","new_ppl":""}
             for a in ("M1","M2","M1_plus_repair","M2_plus_repair") for ds in ("wikitext","c4_fixed_budget")]
dump_csv("V57_EIGHT_EVAL_STATUS.csv",eval_status)
compare=[]
for ds in ("wikitext","c4_fixed_budget"):
    defs=[
    ("scalar10_pre_repair_advantage","NLL(M1)-NLL(M2); positive favors M2"),
    ("repair_increment_M1","NLL(M1_plus_repair)-NLL(M1); negative is improvement"),
    ("repair_increment_M2","NLL(M2_plus_repair)-NLL(M2); negative is improvement"),
    ("scalar10_post_repair_advantage","NLL(M1_plus_repair)-NLL(M2_plus_repair); positive favors scalar10"),
    ("interaction","(NLL(M2_plus_repair)-NLL(M2))-(NLL(M1_plus_repair)-NLL(M1)); negative means larger repair NLL reduction under M2")]
    compare += [{"dataset":ds,"estimand":e,"definition":d,"unit":"nats per valid token",
                 "v57_value":"","status":"NOT_COMPUTABLE_NO_V57_EVALUATION"} for e,d in defs]
dump_csv("UNIFIED_COMPARISON_PENDING.csv",compare)
histdiff=[{"arm":a,"dataset":ds,"historical_nll":hist[(a,ds)]["nll"],"historical_ppl":hist[(a,ds)]["ppl"],
           "historical_runtime":hist[(a,ds)]["runtime"],"v57_nll":"","v57_ppl":"",
           "delta_nll_v57_minus_history":"","delta_ppl_v57_minus_history":"",
           "status":"NOT_COMPUTABLE_V57_NOT_EVALUATED"}
          for a in ("M1","M2","M1_plus_repair","M2_plus_repair") for ds in ("wikitext","c4_fixed_budget")]
dump_csv("HISTORICAL_VS_V57_PENDING.csv",histdiff)

# Measured v56 evaluation durations, keeping stage timers distinct from process wall.
stages=list(csv.DictReader((V56/"STAGE_TIMINGS.csv").open(encoding="utf-8-sig",newline="")))
ppl=[float(r["seconds"]) for r in stages if r["mode"]=="evaluation" and r["stage"]=="PPL_evaluation"]
pair_wall=float(v56_quality["eval_only_wall_seconds"])
four_est=4*pair_wall
cpu_sec=time.perf_counter()-T0
cpu_ledger=[
 {"attempt":1,"task":"raw config JSON equality precheck","status":"FAILED_PRECHECK_ONLY_ALIAS_DIFFERENCE","cpu_wall_seconds":12.7,"source":"observed command wall: 10.2s initial wait plus 2.5s completion poll","gpu_seconds":0.0},
 {"attempt":2,"task":"normalized AutoConfig, checkpoint/tensor/input/window audit","status":"PASS","cpu_wall_seconds":19.206932,"source":"script perf_counter output","gpu_seconds":0.0},
 {"attempt":3,"task":"reproducible rerun after progress guard/report finalization","status":"PASS","cpu_wall_seconds":18.2554042,"source":"script perf_counter output","gpu_seconds":0.0},
 {"attempt":4,"task":"final reproducible source/hash/window preflight rerun","status":"PASS","cpu_wall_seconds":19.2313565,"source":"script perf_counter output from final successful run","gpu_seconds":0.0},
 {"attempt":"other","task":"read-only source discovery and code inspection","status":"CPU_ONLY_NOT_CUMULATIVELY_METERED","cpu_wall_seconds":"NOT_AGGREGATED","source":"short PowerShell/metadata reads in this v57 turn","gpu_seconds":0.0}
]
dump_csv("CPU_DIAGNOSTIC_LEDGER.csv",cpu_ledger)
known_cpu=sum(float(r["cpu_wall_seconds"]) for r in cpu_ledger if isinstance(r["cpu_wall_seconds"],(float,int)))
cost={"budget_cap_seconds":1800.0,"scope":"v57 only; includes audits, loading, diagnostics and all eight evaluations",
      "cpu_preflight_script_seconds":cpu_sec,"gpu_seconds":0.0,"full_model_load_seconds":0.0,
      "formal_evaluation_seconds":0.0,"completed_evaluations":"0/8",
      "known_recorded_cpu_seconds_lower_bound":known_cpu,
      "other_cpu_source_inspection":"occurred but not cumulatively timed; see CPU_DIAGNOSTIC_LEDGER.csv",
      "v56_measured_reference":{"wikitext_stage_seconds":ppl[0],"c4_stage_seconds":ppl[1],
       "synchronized_stage_sum_seconds":sum(ppl),"successful_eval_only_process_wall_seconds":pair_wall},
      "estimate":{"one_pair_seconds":pair_wall,"four_pairs_seconds":four_est,
       "four_pair_estimate_over_cap_seconds":four_est-1800,"multiple_of_cap":four_est/1800,
       "single_wikitext_stage_exceeds_cap":ppl[0]>1800,
       "decision":"DO_NOT_START_FORMAL_EVALUATION"}}
dump_json("COST_PRECHECK.json",cost)

env={"python":env_live["python"],"pytorch":env_live["torch"],"cuda_runtime":env_live["torch_cuda_runtime"],
 "transformers":env_live["transformers"],"tokenizers":env_live["tokenizers"],
 "model_revision":v18_protocol["model"]["revision"],"model_id":v18_protocol["model"]["id"],
 "dtype":v18_protocol["model"]["dtype"],"device":"CUDA; NVIDIA GeForce RTX 4060 Laptop GPU in frozen runs",
 "device_query_in_v57":False,"attention_requested":"SDPA via attn_implementation=sdpa",
 "actual_attention_kernel":"NOT_MEASURED; selected lower-level SDPA backend is not present in logs",
 "tf32":False,"fp32_matmul_precision":"highest per v56 protocol","eval_mode":True,"use_cache":False,
 "max_length":2048,"stride":512,"add_special_tokens":False,"chat_template":False,
 "score_aggregation":"sum raw window NLL / sum valid_tokens, then exp for PPL",
 "tokenizer_revision":v18_protocol["model"]["revision"],
 "history_runtime":{"M1_M2":"torch 2.6.0+cu124; Transformers 4.49.0; Python not recorded",
                   "M1_plus_repair":"torch 2.6.0+cu124; Transformers 5.16.1; Python version not recorded",
                   "M2_plus_repair":"torch 2.5.1+cu121; Transformers and Python unknown"},
 "v57_full_hf_model_loads":"not attempted; preflight stopped on measured budget infeasibility"}
dump_json("ENVIRONMENT_LOCK.json",env)
status={"version":"v57","status":"PRECHECK_COMPLETE_FORMAL_EVALUATION_NOT_STARTED_BUDGET_INFEASIBLE",
 "formal_evaluations_requested":8,"formal_evaluations_completed":0,"P3_read":False,
 "quantization_or_repair_run":False,"new_calibration_run":False,"historical_files_modified":False,
 "checkpoint_hashes_match":{r["arm"]:r["checkpoint_sha_match"] for r in source_rows},
 "effective_configs_all_equal":len(set(effective_configs.values()))==1,
 "named_tensor_manifest_checks":tensor_manifest_pass,
 "input_manifest_checks":{r["dataset"]:r["manifest_hash_match"] for r in input_rows},
 "full_runtime_model_loads_verified_in_v57":False,
 "stop_reason":"A v56 measured WT2+C4 pair takes 2283.58 s process wall; v57 hard cap is 1800 s. The WT2 synchronized stage alone is 2146.70 s."}
dump_json("RUN_STATUS.json",status)

report=f"""# v57 统一运行时四模型正式质量对照 — 预检报告

## 状态

**正式评测未启动，0/8 项完成。** 本轮仅完成 CPU 侧来源、哈希、配置和历史窗口算术核查。没有完整模型的 CUDA 加载或前向，没有量化、整数码修复、新校准，也没有读取 P3。历史文件保持只读。

## 预算判断

v56 的阶段表中 WT2 同步评测计时为 {ppl[0]:.3f} 秒，C4 为 {ppl[1]:.3f} 秒，阶段计时合计 {sum(ppl):.3f} 秒。v56 成功评测 continuation 的进程 wall 为 {pair_wall:.3f} 秒。阶段计时和进程 wall 是不同口径，分别记录。

按每个模型一对数据的实测 eval-only wall 外推，四模型约需 {four_est:.1f} 秒，约为 v57 1800 秒上限的 {four_est/1800:.2f} 倍，且还未计入 v57 的加载和来源检查。仅 WT2 阶段本身就需要 {ppl[0]:.1f} 秒，已经超出整轮上限。因此没有启动任何一个会留下半套窗口的 GPU 评测。已计时 CPU 检查下界为 {known_cpu:.3f} 秒，其中最终可复现预检脚本为 {cpu_sec:.3f} 秒；另有少量只读路径定位未累计计时，明细见 CPU_DIAGNOSTIC_LEDGER.csv。v57 GPU、完整模型加载和正式评测均为 0 秒。

## 环境和 checkpoint 来源

当前 Python 环境包版本实测为 Python {env_live["python"]}、PyTorch {env_live["torch"]}、CUDA runtime {env_live["torch_cuda_runtime"]}、Transformers {env_live["transformers"]}、tokenizers {env_live["tokenizers"]}。冻结协议指定 Qwen2.5-0.5B-Instruct revision {v18_protocol["model"]["revision"]}，FP16、CUDA、SDPA、TF32 off、FP32 matmul highest、eval 模式、use_cache=false、max_length=2048、stride=512、不添加 special tokens、不套 chat template，并以 valid_tokens 加权。记录证明请求了 SDPA，但没有记录 PyTorch 最终采用的底层 attention kernel。

四个 safetensors 文件的 SHA 与各自历史来源记录均匹配；按当前 Transformers AutoConfig loader 归一化后，四个有效 config 指纹一致。第一次比较原始 config JSON 时发现序列化别名差异：旧配置使用 rope_theta/torch_dtype，新配置使用 rope_parameters/dtype，并带有 layer_types；按当前 AutoConfig 统一解析后，这些配置映射到相同有效指纹。这是已记录并修正的 CPU 预检判断，不涉及权重修改。v57 在 CPU 上计算了四个 checkpoint 的全部具名 tensor SHA，并验证 v55/v56 已有 named-hash 清单与文件张量吻合；M1/M2 没有 v18 named-tensor 清单，本轮补算并保存在 v57。完整 Hugging Face model loader 没有在本轮对四个模型实际执行，所以本轮不能声称四个 checkpoint 均经过统一运行时加载通过。历史 M1/M2 测量使用 PyTorch 2.6.0+cu124、Transformers 4.49.0；M1 修复历史评测使用 PyTorch 2.6.0+cu124、Transformers 5.16.1；M2 修复历史评测使用 PyTorch 2.5.1+cu121，Transformers 版本未记录。静态配置与 checkpoint 格式相符，但 R2 在当前 runtime 的完整加载仍未实测。

## 输入和历史窗口

两个冻结 manifest 文件摘要核对通过，tokenizer revision 与模型 revision 相同。WT2 token-ID hash 为 {input_rows[0]["token_ids_sha256_i64le"]}，C4 为 {input_rows[1]["token_ids_sha256_i64le"]}。八组历史 windows CSV 已用窗口 NLL 总和除以 valid token 总数独立复算，并用 exp(weighted NLL) 复算 PPL。结果及来源哈希在 HISTORICAL_WINDOW_RECALC.csv。此项是历史文件算术审计，不是 v57 新评测。

由于没有 v57 窗口，统一四臂差异、交互量和本轮相对历史的环境/执行差异均不可计算。待算交互量定义为 (NLL(M2+repair)-NLL(M2)) - (NLL(M1+repair)-NLL(M1))，单位是 nats/valid token；负值表示固定本轮测量下 M2 臂修复的 NLL 降幅较大。当前不填入跨环境历史估计。

## 交付文件

ENVIRONMENT_LOCK.json、CHECKPOINT_SOURCE_AUDIT.csv、NAMED_TENSOR_HASHES.csv、INPUT_SOURCE_AUDIT.csv、HISTORICAL_WINDOW_RECALC.csv、V57_EIGHT_EVAL_STATUS.csv、UNIFIED_COMPARISON_PENDING.csv、HISTORICAL_VS_V57_PENDING.csv、COST_PRECHECK.json、CPU_DIAGNOSTIC_LEDGER.csv、RUN_STATUS.json、preflight_v57.py 与 REPRODUCE.md 均保存在本独立目录。

## 结论

本轮没有统一 runtime 的 PPL 数据，不能回答 scalar10 修复前后优势或交互量。按 v56 的现有同运行时实测，原定八项评测需约 {four_est/3600:.2f} 小时的 eval-only wall，之后还要留出加载和核验时间；1800 秒预算不足。历史传播有效性 UNKNOWN 继续保留，统一重评也不能替代校准输入传播审计。
"""
(OUT/"REPORT.md").write_text(report,encoding="utf-8")
(OUT/"REPRODUCE.md").write_text(
 "从 PowerShell 运行 CPU-only 预检：\n\n"
 "  & 'E:\\quant-thesis\\.venv\\Scripts\\python.exe' "
 "'E:\\quant-thesis\\experiments\\range_search_w4\\results\\v57_uniform_runtime_four_arm_eval\\preflight_v57.py'\n\n"
 "脚本只读 v18/v55/v56 checkpoint、config、manifest 与历史窗口；只在 v57 写入哈希、算术复核、来源审计和预算预检。"
 "不运行完整 HF forward、CUDA 评测或任何量化/修复。\n",encoding="utf-8")
(OUT/"PREFLIGHT_CPU_SECONDS.txt").write_text(f"{cpu_sec:.9f}\n",encoding="ascii")
print(json.dumps({"out":str(OUT),"tensor_hash_rows":len(tensor_rows),
 "checkpoint_hashes_match":{r["arm"]:r["checkpoint_sha_match"] for r in source_rows},
 "named_manifest_checks":tensor_manifest_pass,
 "effective_configs_equal":len(set(effective_configs.values()))==1,
 "input_manifest_checks":{r["dataset"]:r["manifest_hash_match"] for r in input_rows},
 "historical_windows_recalculated":len(recalc),"cpu_preflight_seconds":cpu_sec,
 "gpu_seconds":0,"formal_evaluations":"0/8"},indent=2))

