"""Generate v47 cost, coverage, and decision artifacts from saved evidence."""
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"E:\quant-thesis\experiments\range_search_w4\results")
OUT = ROOT / "v47_integer_repair_cost_audit"
V46 = ROOT / "v46_m1_single_integer_repair_fullmodel"


def readj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def readcsv(p):
    with Path(p).open("r", newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
def writecsv(p, rows, fields):
    with Path(p).open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()


def main():
    s=readj(OUT/"SUMMARY.json")
    prof=readj(OUT/"PROFILE_RESULTS.json")
    raw=readcsv(OUT/"STAGE_TIMINGS.csv")
    target_groups=[("up_gate","mlp.up_proj"),("down","mlp.down_proj")]
    current=None;capture_i=0;tagged=[]
    for r in raw:
        if r["stage"]=="dependency_input_capture":
            capture_i+=1
            if capture_i<=len(target_groups): current=target_groups[capture_i-1]
        rr=dict(r)
        rr["profile_target_group"]=current[0] if current else ""
        rr["profile_target_module"]=current[1] if current else ""
        tagged.append(rr)
    fields=list(tagged[0]) if tagged else ["mode","layer","module","stage"]
    writecsv(OUT/"PROFILE_STAGE_TIMINGS_TAGGED.csv",tagged,fields)

    sums=defaultdict(lambda:{"calls":0,"seconds":0.0})
    direct=defaultdict(list)
    for r in tagged:
        key=(r["profile_target_group"],r["profile_target_module"],r["stage"])
        if not r["stage"].startswith("profile_R1_model_load") and r["profile_target_group"]:
            sums[key]["calls"]+=1;sums[key]["seconds"]+=float(r["seconds"])
        if r["profile_target_module"] and r["stage"]=="direct_recompute":
            direct[r["profile_target_module"]].append(float(r["seconds"]))
    bytarget={}
    for _,module in target_groups:
        vals=direct[module]
        if len(vals)<2: raise RuntimeError(f"Missing direct timing sequence for {module}")
        bytarget[module]={"initial_loss":vals[0],"proposal_direct":sum(vals[1:-1]),
                          "final_loss":vals[-1],"calls":len(vals)}
        sums[("",module,"direct_initial_loss")]["calls"]=1
        sums[("",module,"direct_initial_loss")]["seconds"]=vals[0]
        sums[("",module,"direct_proposal_checks")]["calls"]=len(vals)-2
        sums[("",module,"direct_proposal_checks")]["seconds"]=sum(vals[1:-1])
        sums[("",module,"direct_final_loss")]["calls"]=1
        sums[("",module,"direct_final_loss")]["seconds"]=vals[-1]
    pstage=[]
    for (group,module,stage),v in sorted(sums.items()):
        if stage=="profile_input_hash":
            # In this run the timing context enclosed a no-op; hashing happened
            # outside it. Do not present that near-zero number as hash cost.
            pstage.append({"target_group":group,"target_module":module,"stage":stage,
                           "calls":v["calls"],"seconds":"NOT_MEASURED",
                           "coverage":"hash bytes were verified; wrapper timed a no-op, full process wall includes the actual CPU hash"})
            continue
        pstage.append({"target_group":group,"target_module":module,"stage":stage,
                       "calls":v["calls"],"seconds":v["seconds"],
                       "coverage":"measured synchronized wall; p timer nested in accepted_update timer" if stage=="profile_p_vector_update" else "measured synchronized wall"})
    writecsv(OUT/"PROFILE_STAGE_COSTS.csv",pstage,
             ["target_group","target_module","stage","calls","seconds","coverage"])

    # Baseline clean path stage sums for only L0 up_proj and down_proj.
    v46t=readcsv(V46/"STAGE_TIMINGS.csv")
    baseline=defaultdict(lambda:{"calls":0,"seconds":0.0})
    for r in v46t:
        if r["mode"]!="formal" or r["layer"]!="0": continue
        stage=r["stage"];mod=r["module"]
        if mod not in ("mlp.up_proj","mlp.down_proj","up_gate","down"): continue
        label={"up_gate":"up_gate","down":"down"}.get(mod,mod)
        baseline[(label,stage)]["calls"]+=1;baseline[(label,stage)]["seconds"]+=float(r["seconds"])
    baseline_direct={}
    for r in readcsv(OUT/"DIRECT_CALL_ATTRIBUTION.csv"):
        if r["layer"]=="0" and r["module"] in ("mlp.up_proj","mlp.down_proj"):
            baseline_direct[r["module"]]=r
    comparisons=[]
    for group,module in target_groups:
        pairs=[("dependency_input_capture","dependency_input_capture"),
               ("C_construction","C_construction"),("candidate_scoring","candidate_scoring"),
               ("direct_initial_loss","direct_initial_loss"),
               ("direct_proposal_checks","direct_proposal_checks"),
               ("direct_final_loss","direct_final_loss"),
               ("direct_recompute_total","direct_recompute_total"),
               ("accepted_update_and_rejection_restore","accepted_update_and_rejection_restore"),
               ("profile_p_vector_update","profile_p_vector_update"),
               ("p_drift_check","p_drift_check"),("final_correctness","final_correctness")]
        for original_stage,profile_stage in pairs:
            # Runtime uses group names for capture/C and module names for repair.
            if original_stage in ("dependency_input_capture","C_construction"):
                bv=baseline.get((group,original_stage),{"calls":0,"seconds":None})
            elif original_stage.startswith("direct_"):
                bd=baseline_direct[module]
                keymap={"direct_initial_loss":"initial_loss_seconds",
                        "direct_proposal_checks":"proposal_direct_seconds",
                        "direct_final_loss":"final_loss_seconds"}
                callmap={"direct_initial_loss":1,"direct_proposal_checks":int(bd["direct_calls"])-2,
                         "direct_final_loss":1}
                if original_stage=="direct_recompute_total":
                    bv={"calls":int(bd["direct_calls"]),"seconds":sum(float(bd[k]) for k in ("initial_loss_seconds","proposal_direct_seconds","final_loss_seconds"))}
                else:
                    bv={"calls":callmap[original_stage],"seconds":float(bd[keymap[original_stage]])}
            else:
                bv=baseline.get((module,original_stage),{"calls":0,"seconds":None})
            pv=defaultdict(lambda:{"calls":0,"seconds":0.0})
            for rr in tagged:
                if rr["profile_target_group"]!=group: continue
                st=rr["stage"]
                matched=st==profile_stage
                if matched:
                    pv[profile_stage]["calls"]+=1;pv[profile_stage]["seconds"]+=float(rr["seconds"])
            if profile_stage.startswith("direct_"):
                values=bytarget[module]
                profile_parts={"direct_initial_loss":(1,values["initial_loss"]),
                               "direct_proposal_checks":(values["calls"]-2,values["proposal_direct"]),
                               "direct_final_loss":(1,values["final_loss"]),
                               "direct_recompute_total":(values["calls"],values["initial_loss"]+values["proposal_direct"]+values["final_loss"])}
                pv[profile_stage]={"calls":profile_parts[profile_stage][0],"seconds":profile_parts[profile_stage][1]}
            if profile_stage=="profile_p_vector_update":
                pp=[r for r in tagged if r["profile_target_module"]==module and r["stage"]==profile_stage]
                pv[profile_stage]={"calls":len(pp),"seconds":sum(float(x["seconds"]) for x in pp)}
            pvalue=pv[profile_stage]
            comparisons.append({"target_group":group,"target_module":module,"stage":original_stage,
                                "v46_full_run_calls":bv.get("calls",0),
                                "v46_full_run_seconds":bv.get("seconds", "NOT_MEASURED"),
                                "v47_L0_replay_calls":pvalue["calls"],
                                "v47_L0_replay_seconds":pvalue["seconds"],
                                "accounting":"same v45 decisions reproduced; v47 adds synchronized p child timer; environment and per-run variation apply"})
    writecsv(OUT/"PROFILE_STAGE_COMPARISON.csv",comparisons,
             ["target_group","target_module","stage","v46_full_run_calls","v46_full_run_seconds",
              "v47_L0_replay_calls","v47_L0_replay_seconds","accounting"])

    # Cost lines deliberately distinguish stage, attempt and process clocks.
    failed=readj(V46/"EXECUTION_AMENDMENT_001.json")
    final=readj(V46/"FINAL_STATUS.json")
    costs=[
        {"item":"v46 valid formal repair: 24 run_layer blocks","seconds":s["block_wall_seconds"],"status":"MEASURED_SUCCESS","scope":"168 Linear, 304128 rows; includes valid capture/C/repair/commit/prefix work; excludes the 14 wrong-dispatch groups"},
        {"item":"v46 first wrong-dispatch attempt","seconds":failed["failed_attempt_elapsed_seconds"],"status":"MEASURED_FAILURE","scope":"14 invalid groups archived and excluded from formal workload; time remains inside original process wall"},
        {"item":"v46 preflight interval from original start","seconds":21.379425048828125,"status":"MEASURED_OVERLAPPING_INTERVAL","scope":"Overlaps process wall and failed attempt; do not add"},
        {"item":"v46 save/grid/reload/output regression","seconds":11.039314699999522,"status":"MEASURED_SUCCESS","scope":"After formal blocks"},
        {"item":"v46 continuation model reload","seconds":1.2471799000049941,"status":"MEASURED_SUCCESS","scope":"Separate PPL continuation after evaluator wrapper failure"},
        {"item":"v46 WT2 plus C4 evaluation","seconds":131.31646400000318,"status":"MEASURED_SUCCESS","scope":"One evaluation each; after R1 lock"},
        {"item":"v46 original GPU_START to final status process wall","seconds":final["elapsed_seconds_from_original_gpu_start"],"status":"MEASURED_END_TO_END","scope":"Contains preflight, failed attempt, diagnosis/restart gaps, valid repair, save/reload, continuation and PPL; not a pure repair stage"},
        {"item":"v47 two-matrix synchronized instrumentation replay","seconds":prof["elapsed_seconds"],"status":"MEASURED_SUCCESS","scope":"Independent local replay, includes warmup/model load/input reconstruction/C/repair/diagnostics; 149.94s of 600s cap; no submit/checkpoint/PPL"},
        {"item":"v42-v45 R2 validation chain","seconds":7064.804290294647,"status":"HISTORICAL_CUMULATIVE","scope":"Contains pilots, failures, diagnostics, inherited/recovered groups, save/reload and PPL; not comparable to one clean full repair"},
        {"item":"v45 new 141 Linear stage wall","seconds":4717.0275438999815,"status":"HISTORICAL_PARTIAL_SUCCESS","scope":"Excludes 27 inherited modules; not full R2 repair cost"},
        {"item":"R2 unique clean full repair-only cost","seconds":"NOT_RECONSTRUCTABLE","status":"MISSING","scope":"Do not infer from v45 new-group stage or total recovery-chain wall"},
    ]
    writecsv(OUT/"EXECUTION_ACCOUNTING.csv",costs,["item","seconds","status","scope"])

    # Historical phases/statuses are kept separate from v46 R1.
    hist=[]
    specs=[("v42","v42_m2_single_integer_repair_fullmodel", "pilot"),
           ("v43","v43_m2_single_integer_repair_fullmodel", "formal attempt"),
           ("v44","v44_m2_single_integer_repair_fullmodel", "failed numerical attempt and diagnosis"),
           ("v45","v45_m2_single_integer_repair_fullmodel", "full resumed completion")]
    for ver,folder,kind in specs:
        d=readj(ROOT/folder/"DECISION.json")
        if ver=="v42":
            status=d["status"];elapsed_s=status.get("elapsed_seconds");mods=0;rows=0;full=status.get("complete_model",False);ppl=status.get("PPL_run",False)
            detail="Block0 pilot completed, budget stop; not a complete model"
        elif ver=="v43":
            status=d["status"];elapsed_s=status.get("elapsed_seconds");mods=status.get("committed_modules",0);rows=status.get("committed_rows",0);full=status.get("complete_model",False);ppl=status.get("PPL_run",False)
            detail="NUMERICAL_STOP_FROZEN_RULE; committed groups were later used as v44/v45 recovery inputs; no quality evaluation"
        elif ver=="v44":
            status=d["original_failure_status"];elapsed_s=status.get("elapsed_seconds");mods=d["coverage"]["modules"];rows=d["coverage"]["rows"];full=status.get("complete_model",False);ppl=status.get("PPL_run",False)
            detail="Original FP32/precision inconsistency stopped model; coverage is 27 committed modules from v43 plus 2 new; failed tentative down group was not committed"
        else:
            status=d["status"];elapsed_s=status.get("elapsed_seconds");mods=d["coverage"]["modules"];rows=d["coverage"]["rows"];full=status.get("complete_model",False);ppl=status.get("PPL_run",False)
            detail="Completed all 168; 27 inherited plus 141 new; cumulative v42-v45 wall is listed separately in execution accounting"
        hist.append({"version":ver,"stage_kind":kind,"status":status.get("status"),"elapsed_seconds_for_status_scope":elapsed_s,
                     "committed_or_completed_modules":mods,"committed_or_completed_rows":rows,
                     "complete_model":full,"PPL_run":ppl,"detail":detail})
    writecsv(OUT/"HISTORICAL_CHAIN.csv",hist,["version","stage_kind","status","elapsed_seconds_for_status_scope",
             "committed_or_completed_modules","committed_or_completed_rows","complete_model","PPL_run","detail"])

    # Count-only direct-row opportunity; no time claim.
    rw=readcsv(OUT/"REPEATED_DIRECT_WORK.csv")[0]
    detail=[{"scope":"whole v46 formal repair","matrix":"all 168 Linear modules",
             "proposal_direct_calls":rw["proposal_direct_calls"],"full_output_rows_computed":rw["full_linear_row_outputs_computed"],
             "proposal_rows_needed":rw["proposed_row_outputs_needed"],"unchanged_rows_recomputed":rw["unchanged_row_outputs_recomputed"],
             "unconsumed_output_fraction":rw["fraction_full_outputs_not_consumed"],
             "interpretation":"operation-count projection only; not a measured time saving"}]
    for item in prof["targets"]:
        n=item["rows"]; calls=32
        # Each of these two L0 matrices had no rejected proposals and every
        # proposal was accepted; accepted_updates therefore gives proposal rows.
        full=n*calls; needed=item["accepted_updates"]
        detail.append({"scope":"v47 local L0 profile","matrix":item["module"],
                       "proposal_direct_calls":calls,"full_output_rows_computed":full,
                       "proposal_rows_needed":needed,"unchanged_rows_recomputed":full-needed,
                       "unconsumed_output_fraction":(full-needed)/full,
                       "interpretation":"row count only; FP32 batch shape change can change rounding; no time estimate"})
    writecsv(OUT/"DIRECT_ROW_WORK.csv",detail,["scope","matrix","proposal_direct_calls","full_output_rows_computed",
              "proposal_rows_needed","unchanged_rows_recomputed","unconsumed_output_fraction","interpretation"])

    # Include historical decision records in the source inventory.
    src_path=OUT/"SOURCE_AUDIT.csv";src=readcsv(src_path)
    for ver in ("v42_m2_single_integer_repair_fullmodel","v43_m2_single_integer_repair_fullmodel",
                "v44_m2_single_integer_repair_fullmodel","v45_m2_single_integer_repair_fullmodel"):
        for rel in (["DECISION.json"] if ver!="v45_m2_single_integer_repair_fullmodel" else ["DECISION.json","REPORT.md","BLOCK_TIMINGS.csv"]):
            p=ROOT/ver/rel
            src.append({"source":str(p),"sha256":sha(p),"bytes":p.stat().st_size,
                        "role":"read-only historical R2 chain status/cost reference; not v46 formal repair timing"})
    for rel in ["PROFILE_RESULTS.json","PROFILE_STATUS.json","PROFILE_START.json","STAGE_TIMINGS.csv",
                "profile_v46_l0_matrices.py","instrumented_repair_v45.py","INSTRUMENTATION_DIFF.txt",
                "analyze_v47.py","finalize_v47.py"]:
        p=OUT/rel
        if p.exists():
            role="v47 analysis/reproduction source" if rel in ("analyze_v47.py","finalize_v47.py") else "v47 bounded profile raw status/timing/profiler source"
            src.append({"source":str(p),"sha256":sha(p),"bytes":p.stat().st_size,"role":role})
    writecsv(src_path,src,["source","sha256","bytes","role"])
    writecsv(OUT/"PROFILE_SOURCE_AUDIT.csv",[r for r in src if "v47 bounded profile" in r["role"]],
             ["source","sha256","bytes","role"])

    d={"decision":"NO_OPTIMIZATION_RECOMMENDED_YET",
       "reason":"Measured bottlenecks are direct FP32 proposal scoring (54.03% of valid full-model block wall) and FP64 C construction (34.80%). Candidate scoring/minimum is 1.45%; p update is only isolated on two L0 matrices and is nested in a small combined full-run stage. A selected-row direct path may save only 2.16% of whole-run proposal output rows by count, while changing GEMM shape can alter FP32 decisions. No safe implementation saving has been demonstrated.",
       "priority_direction":"evidence_insufficient_hold_optimization",
       "answers":{"A_candidate_scoring_is_major":False,"B_acceptance_still_needs_full_p_update":True,
                  "C_direct_recompute_is_major":True,"D_C_or_capture_limit": "C construction is the second largest block cost; input capture/hash/refresh is small",
                  "E_observed_repeated_work":True},
       "next_test_if_reopened":"Keep this as a future bounded implementation test only: on complete L0 up_proj and down_proj, compare any selected-row or cached-output direct scorer with the frozen v45 scorer at every proposal; require identical accept/reject decisions and terminal code/evaluation counts, then compare synchronized runtime. Do not treat row-count ratios as time savings.",
       "gpu_profiling":{"completed":True,"seconds":prof["elapsed_seconds"],"cap_seconds":prof["cap_seconds"],
                        "input_C_hashes_match_v46":True,"terminal_codes_updates_evaluations_and_losses_match_v46":True,
                        "checkpoint_written":False,"committed_to_model":False,"PPL_or_held_scored":False},
       "model_setting":"Requested Luna Max; interface cannot verify or change model setting.",
       "no_new_search_algorithm":True,"v46_and_older_artifacts_read_only":True}
    (OUT/"DECISION.json").write_text(json.dumps(d,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

    # Explicitly revise timer-coverage interpretation for the added L0 p timer.
    cov=readcsv(OUT/"TIMER_COVERAGE.csv")
    for r in cov:
        if r["requested_component"]=="p update only":
            r["coverage"]="V46 full-model timer is merged; v47 adds isolated synchronized p-update timer on both complete L0 matrices"
    writecsv(OUT/"TIMER_COVERAGE.csv",cov,["requested_component","observed_timer","coverage","limit"])

    r1=readj(V46/"FINAL_STATUS.json")
    report=f'''# v47 — 单点整数码修复成本分解与优化判断

## 结论

v46 正式路径的主耗时是 proposal 的 FP32 直接输出重算：**{s['direct_proposal_seconds']:.2f}s，占24个完整 block 修复 wall 的{100*s['direct_proposal_seconds']/s['block_wall_seconds']:.2f}%**。FP64 完整 C 构建为 **{s['C_construction_seconds']:.2f}s（{100*s['C_construction_seconds']/s['block_wall_seconds']:.2f}%）**。候选二次型评分与最小选择合计 **{s['candidate_score_min_seconds']:.2f}s（1.45%）**；更新、恢复和拒绝处理合并计时 **{s['accepted_update_restore_seconds']:.2f}s（0.54%）**。

最终只推荐：**证据不足，暂不优化**。候选筛选即使理想地删掉整段候选评分，对完整修复的 Amdahl 上限也只有 {1/(1-s['candidate_score_min_seconds']/s['block_wall_seconds']):.3f}×；它不能省掉被接受行的全维 p 更新。直接重算和 C 是真实瓶颈，但减少直接重算会碰到冻结 FP32 接受规则，当前只有输出行计数上的可观察冗余，尚未证明可安全换成另一执行形状并保持逐候选决策。

## 覆盖与运行口径

有效 v46 R1 正式结果覆盖 {s['formal_linears']} 个 Linear、{s['formal_rows']:,} 行，接受 {s['accepted_updates']:,} 次修改，二次型候选评价 {s['candidate_evaluations']:,} 次，拒绝 {s['rejected_candidates']} 个候选。逐行预算、投影/层分布见 `UPDATE_DISTRIBUTION.csv`、`PROJECTION_AND_BLOCK_SUMMARY.csv` 与 `LAYER_PROJECTION_UPDATES.csv`。

24 个 block `run_layer` 的完整同步 wall 合计 **{s['block_wall_seconds']:.3f}s**。这不是纯 GPU kernel 时间，含 CPU 调度、同步、数据拷贝、提交和各层真实前缀刷新。逐阶段排他计时合计 {s['exclusive_stage_timer_seconds']:.3f}s，差额 **{s['unattributed_block_seconds']:.3f}s（1.06%）**归为未归因的 timer 外工作。FP64 独立复核11.247s 是接受/拒绝处理计时内的嵌套子计时，不能再次相加。

首次错误分派的14个组被封存并排除在有效覆盖和5485.7秒正式修复 wall 之外；其失败尝试时间 **{failed['failed_attempt_elapsed_seconds']:.3f}s**仍包含于 v46 从原 GPU_START 到 final status 的 **{r1['elapsed_seconds_from_original_gpu_start']:.3f}s**进程 wall。该 process wall 还包括预检、诊断/重启空档、保存/reload、续跑加载及一次 WT2/C4评测。不要把 process wall 当作正式修复时间，也不要将重叠阶段再加一次。详细口径见 `EXECUTION_ACCOUNTING.csv`。

v42–v45 仅作 R2 历史参照：v42 是完成 block0 pilot 后预算停止；v43、v44分别保留数值停止及失败诊断，部分已提交组作为恢复输入；v45 才补全168个模块并完成PPL。v42–v45累计链成本记录为7064.804s，含 pilot、失败、诊断、恢复与评测；v45新增141个模块阶段为4717.028s，另有27个继承模块。因此R2完整 clean repair-only 时间不能从这些wall可靠重建，详见 `HISTORICAL_CHAIN.csv`。这些记录没有并入 v46 正式阶段。

## 阶段计时

下表分母是有效 v46 完整修复 **{s['block_wall_seconds']:.3f}s**；FP64 C 和 direct 的占比按这个正式修复口径。

| 阶段 | 计时 | 完整修复占比 | 覆盖说明 |
|---|---:|---:|---|
| 输入捕获 | {s['input_capture_seconds']:.2f}s | {100*s['input_capture_seconds']/s['block_wall_seconds']:.2f}% | 96次依赖输入捕获 |
| FP64 C 构建 | {s['C_construction_seconds']:.2f}s | {100*s['C_construction_seconds']/s['block_wall_seconds']:.2f}% | 含X分块搬运、FP64 GEMM、归一化/对称化/有限值检查 |
| 初始 e/p | {s['repair_initialization_seconds'] if 'repair_initialization_seconds' in s else 9.23:.2f}s | 0.17% | Q、e、p的合并初始化计时；初始loss单独在direct计时首调用中 |
| 初始 direct loss | {s['direct_initial_loss_seconds']:.2f}s | {100*s['direct_initial_loss_seconds']/s['block_wall_seconds']:.2f}% | 每Linear首个direct调用 |
| 候选评分＋最小选择 | {s['candidate_score_min_seconds']:.2f}s | {100*s['candidate_score_min_seconds']/s['block_wall_seconds']:.2f}% | 同一计时器内，不能进一步声称评分/argmin各自实测 |
| proposal direct FP32评分 | {s['direct_proposal_seconds']:.2f}s | {100*s['direct_proposal_seconds']/s['block_wall_seconds']:.2f}% | 5376次，全Linear行矩阵重算；FP32输出与FP64 SSE合并 |
| 最终 direct loss | {s['direct_final_loss_seconds']:.2f}s | {100*s['direct_final_loss_seconds']/s['block_wall_seconds']:.2f}% | 每Linear末次直接复算 |
| code/p更新与拒绝恢复 | {s['accepted_update_restore_seconds']:.2f}s | {100*s['accepted_update_restore_seconds']/s['block_wall_seconds']:.2f}% | 原v46合并计时，p没有独立全模型计时 |
| p漂移检查 | {50.42:.2f}s | {100*50.42/s['block_wall_seconds']:.2f}% | 840次独立缓存正确性检查，不是p更新 |
| 输入哈希 | {64.18:.2f}s | {100*64.18/s['block_wall_seconds']:.2f}% | 96次CPU SHA-256 |
| block前缀刷新 | {s['prefix_refresh_seconds']:.2f}s | {100*s['prefix_refresh_seconds']/s['block_wall_seconds']:.2f}% | 24次完整下游block前向 |
| checkpoint写入＋组提交 | {10.79+0.92:.2f}s | {100*(10.79+0.92)/s['block_wall_seconds']:.2f}% | packing/checkpoint与工作模型提交分计时 |
| 未归因 | {s['unattributed_block_seconds']:.2f}s | {100*s['unattributed_block_seconds']/s['block_wall_seconds']:.2f}% | loop/status、日志、timer间隙等，不按运算量摊销 |

`STAGE_COSTS.csv`、`TIMER_COVERAGE.csv`保留精确秒数和每项覆盖边界。完整直接路径总计 {s['direct_initial_loss_seconds']+s['direct_proposal_seconds']+s['direct_final_loss_seconds']:.2f}s，即完整修复的{100*(s['direct_initial_loss_seconds']+s['direct_proposal_seconds']+s['direct_final_loss_seconds'])/s['block_wall_seconds']:.2f}%。它实际包含逐输入块的CPU→GPU拷贝、FP32矩阵乘及FP64平方求和，无法把这些内部操作继续当作分开的实测阶段。

## 有界 L0 全矩阵重放

由于全模型阶段表把 p 更新合并在 code 更新/拒绝恢复内，按协议完成一次600秒硬上限内的 profiling。锁定 R1 重放后，up_proj (4864×896) 与 down_proj (896×4864) 的输入和未加 damping C 哈希均逐项匹配 v46 正式组；v45终态 code、scale、g_idx、zero、每行接受数、候选评价数及初末loss全部精确匹配。仅在 p 向量更新语句周围新增同步timer，计时器差异见 `INSTRUMENTATION_DIFF.txt`。

本次完整小矩阵重放耗时 **{prof['elapsed_seconds']:.3f}s / 600s**，峰值allocated显存 {prof['peak_allocated_bytes']/1024**3:.2f} GiB。up_proj的p更新计时{sum(float(r['seconds']) for r in raw if r['stage']=='profile_p_vector_update' and r['module']=='mlp.up_proj'):.4f}s，down_proj为{sum(float(r['seconds']) for r in raw if r['stage']=='profile_p_vector_update' and r['module']=='mlp.down_proj'):.4f}s，合计约0.11s（仅这两个完整矩阵，且含每个timer的同步开销）。它是合并更新阶段的子计时，不得与外层update时间相加。v46这两个完整矩阵历史update合计0.29s、候选评分1.72s、direct总计64.36s、C构建68.73s；v47仪表化stage明细见 `PROFILE_STAGE_COMPARISON.csv`。带instrumentation时间不是优化后速度，不能从一次相邻运行做因果加速结论。

跨所有v46 proposal direct调用，运行时为每次新候选计算Linear全部输出行，但只消费被提议修改行的loss。代码/已保存轨迹给出9,732,096个全行输出槽位，其中9,521,788个候选行被消费，210,308个未修改行被重复计算，占2.16%。L0 up_proj未消费15150/155648行槽位（9.73%），down_proj为0/28672。此处是输出行计数，不是时间节省估计；改变GEMM输出行batch可能改变FP32舍入。复算/未使用行细节见 `DIRECT_ROW_WORK.csv`。

## 更新预算与局部轨迹

272,996/304,128行（89.76%）达到32次接受上限；0行接受0次，180行在1–8次，2,482行在9–16次，28,470行在17–31次。投影平均接受数为：down_proj 32.00、k_proj 32.00、v_proj 32.00、q_proj 32.00、gate_proj 31.56、up_proj 31.54、o_proj 27.10。逐层/投影行数与32次上限数见表格文件。

保存轨迹直接loss增量按每行accepted update序号截断：1、4、8、16、32次的累计search MSE下降分别为12.874、35.281、54.993、81.346、111.408（相对初始总和882.156分别1.46%、4.00%、6.23%、9.22%、12.63%）。这是已固定X下逐行轨迹的描述性聚合，不是新预算实验或8/16步完整模型PPL；不据此选新预算。v46全程终值与总search MSE下降12.629%来自已保存模块汇总。

## 优化判断

| 问题 | 判断 |
|---|---|
| 候选评分和最小选择占主要成本？ | 否；79.28s，1.45%完整修复。候选/最小合并的全模型理想无限加速上限1.015×。 |
| 减候选后还需更新完整p？ | 是；每个被接受的坐标变化要加上相应C列乘实际δ，维护当前p仍需长度为输入维度的向量更新。筛选只省掉候选打分，不省掉已接受更新。 |
| FP32直接重算是主成本？ | 是；proposal check 54.03%，全初始/提议/终态direct合计57.41%。但这些严格接受复核不能直接删掉。 |
| C或前缀捕获是否限制整体加速？ | C是第二瓶颈34.80%，不能与direct share合并；输入捕获、hash和prefix刷新合计3.22%。 |
| 可直接观察重复计算？ | 是，所有输出行被重算而只有proposal rows被读取；但全体多余输出槽位占2.16%，不是“省90%候选就快90%”。 |

理想单部分上限均以完整修复5485.675s为分母：proposal direct全去除上限2.175×；全部direct全去除2.347×；C全去除1.534×；候选评分全去除1.015×；整个update/reject合并计时全去除1.005×；输入capture+hash+refresh全去除1.033×。这些是假设部分耗时为零的Amdahl上限，不是可实现预测。候选评分＋proposal direct＋update局部计时分母为{ s['direct_proposal_seconds']+s['candidate_score_min_seconds']+s['accepted_update_restore_seconds']:.3f}s；对应理想上限也仅是该局部计时口径，详见 `OPTIMIZATION_BOUNDS.csv`。

**最终决策：证据不足，暂不优化。** 下一次若重开，只应先用同一完整L0矩阵验证逐proposal保留原FP32 strict-decline决策的direct路径替代，再看同步实测耗时；本轮不实施优化、不启动整模型、不新增PPL。

## 来源与复现

`SOURCE_AUDIT.csv`列出v46冻结协议、实现、完整阶段计时、模块汇总和96组已提交checkpoint链，以及只读的v42–v45历史引用。`PROFILE_SOURCE_AUDIT.csv`列出v47重放源码和原始profile状态/计时。v46正确覆盖来自formal group目录；最初14个错误分派组不在该目录，且执行修订记录明确排除。

复现命令和“CPU汇总 vs 需GPU的完整L0 profile”区分见 `REPRODUCE.md`。requested model setting Luna Max无法由当前接口确认；报告的是本地执行证据，不声称模型设置已切换。
'''
    (OUT/"REPORT.md").write_text(report,encoding="utf-8")
    (OUT/"REPRODUCE.md").write_text(f'''# v47 reproduction

## CPU-only aggregation

Run from `E:\\quant-thesis` with the environment that can import PyTorch:

```powershell
python experiments/range_search_w4/results/v47_integer_repair_cost_audit/analyze_v47.py
python experiments/range_search_w4/results/v47_integer_repair_cost_audit/finalize_v47.py
```

The analyzer reads committed v46 result checkpoints and timing CSVs. It does not load a model or use CUDA. It verifies 96 committed groups and reconciles row/update/evaluation totals.

## Bounded full-matrix GPU profile

The already completed profile command was:

```powershell
.venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v47_integer_repair_cost_audit/profile_v46_l0_matrices.py
```

This reconstructs the exact L0 up_proj/down_proj inputs from locked R1, verifies v46 input and C hashes, and replays full rows under a 600-second hard cap. It adds only a synchronized timer around p update. It writes timing/status diagnostics in v47, does not commit weights, save a quantized checkpoint, score held data, or run PPL. Re-running it incurs a new GPU profiling run; it is not needed to read the saved results.

The v47 p timer is nested inside `accepted_update_and_rejection_restore`; never add parent and child timer totals. `profile_input_hash` is a no-op timer in this recorded profile; actual hashes were verified and included in total process wall, while full-model hash cost is measured from v46's `input_hash` timer.
''',encoding="utf-8")

    # Manifest, including report and reproduction instructions last.
    manifest=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name!="ARTIFACT_MANIFEST.csv":
            manifest.append({"file":str(p),"bytes":p.stat().st_size,"sha256":sha(p)})
    writecsv(OUT/"ARTIFACT_MANIFEST.csv",manifest,["file","bytes","sha256"])


if __name__=="__main__": main()
