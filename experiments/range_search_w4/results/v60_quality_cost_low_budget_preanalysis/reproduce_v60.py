from __future__ import annotations
import csv, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"E:\quant-thesis")
BASE = ROOT / "experiments/range_search_w4/results"
OUT = BASE / "v60_quality_cost_low_budget_preanalysis"
SOURCES = {
    "v55": BASE / "v55_m2_decoder_fixed_integer_repair",
    "v56": BASE / "v56_m1_fast_integer_repair",
    "v59": BASE / "v59_same_process_four_checkpoint_eval",
    "v54": BASE / "v54_m2_decoder_fixed_integer_repair",
    "v51": BASE / "v51_m2_fast_integer_repair_fullmodel",
    "v53": BASE / "v53_decoder_interface_propagation_audit",
}
BUDGETS = (1, 4, 8, 16, 32)
CPU0, WALL0 = time.process_time(), time.perf_counter()

def read_csv(p):
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(name, rows, fields=None):
    p = OUT / name
    if fields is None:
        fields = list(rows[0]) if rows else []
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def dump_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def relpct(a, b):
    return 100.0 * (a / b - 1.0)

# Recompute the v59 four-arm table from complete-window NLL sums and valid-token counts.
v59 = SOURCES["v59"]
raw = read_csv(v59 / "INDEPENDENT_WINDOW_RECALC.csv")
q, qrows = {}, []
for x in raw:
    nll_sum, tokens = float(x["nll_sum"]), int(x["valid_tokens"])
    nll = nll_sum / tokens
    ppl = math.exp(nll)
    q[x["arm"], x["dataset"]] = dict(nll_sum=nll_sum, tokens=tokens, nll=nll, ppl=ppl)
    qrows.append(dict(arm=x["arm"], dataset=x["dataset"], checkpoint_sha256=x["checkpoint_sha256"],
        windows=int(x["windows"]), valid_tokens=tokens, nll_sum=nll_sum,
        weighted_nll_nats_per_valid_token=nll, ppl_exp_weighted_nll=ppl,
        manifest_sha256=x["manifest_sha256"], token_ids_sha256=x["token_ids_sha256_i64le"],
        window_masks_rechecked=x["window_ids_and_mask_hashes_rechecked"]))
assert len(q) == 8 and all(x["window_masks_rechecked"] == "True" for x in qrows)
write_csv("FOUR_ARM_RESULTS.csv", qrows)
contrasts = []
for ds in ("wikitext", "c4_fixed_budget"):
    pairs = [
        ("M1_plus_repair_vs_M1", "M1_plus_repair", "M1"),
        ("M2_plus_repair_vs_M2", "M2_plus_repair", "M2"),
        ("M2_vs_M1_before_repair", "M2", "M1"),
        ("M2_plus_repair_vs_M1_plus_repair", "M2_plus_repair", "M1_plus_repair"),
    ]
    for name, a, b in pairs:
        aa, bb = q[a, ds], q[b, ds]
        contrasts.append(dict(dataset=ds, contrast=name, model_a=a, model_b=b,
            formula=f"{a} - {b}", delta_nll_nats_per_valid_token=aa["nll"]-bb["nll"],
            delta_nll_relative_pct=relpct(aa["nll"],bb["nll"]), delta_ppl=aa["ppl"]-bb["ppl"],
            delta_ppl_relative_pct=relpct(aa["ppl"],bb["ppl"])))
    m1,m2,r1,r2 = [q[a,ds]["nll"] for a in ("M1","M2","M1_plus_repair","M2_plus_repair")]
    contrasts.append(dict(dataset=ds,contrast="descriptive_interaction_I",model_a="R2/M2 and R1/M1 changes",model_b="",
        formula="I=(NLL_R2-NLL_M2)-(NLL_R1-NLL_M1)",delta_nll_nats_per_valid_token=(r2-m2)-(r1-m1),
        delta_nll_relative_pct="",delta_ppl="",delta_ppl_relative_pct=""))
write_csv("FOUR_ARM_CONTRASTS.csv", contrasts)

# Replay saved accepted moves, all on CPU. Each saved move has row, update number,
# C-quadratic delta and cumulative candidate-evaluation counter.
trace_audit, endpoints, budgets, strata, marginals, update_distributions = [], [], [], [], [], []
for version in ("v55", "v56"):
    folder = SOURCES[version]
    row_files = sorted((folder/"formal/rows").glob("*.csv"))
    groups = sorted((folder/"formal/groups").glob("layer_*.pt"))
    assert len(row_files)==168 and len(groups)==96, (version,len(row_files),len(groups))
    rowcsv = {}
    for fp in row_files:
        for r in read_csv(fp):
            key=(int(r["layer"]),r["module"],int(r["row"]))
            if key in rowcsv: raise AssertionError(f"duplicate row {key}")
            rowcsv[key]=r
    assert len(rowcsv)==304128
    rec=[]; module_count=0; move_total=0; max_reconstruction=0.0; statuses={}
    group_hash_rows=[]
    for gp in groups:
        group_hash_rows.append(dict(name=gp.name,bytes=gp.stat().st_size,sha256=sha(gp)))
        cp=torch.load(gp,map_location="cpu",weights_only=False)
        assert cp.get("committed") is True and cp.get("mode")=="formal"
        layer=int(cp["layer"])
        for module,r in cp["modules"].items():
            module_count += 1
            n,d=map(int,r["codes"].shape)
            updates=np.asarray(r["updates"].numpy(),dtype=np.int64)
            evaluations=np.asarray(r["evaluations"].numpy(),dtype=np.int64)
            status=np.asarray(r["status"].numpy(),dtype=np.int64)
            ini_raw=np.asarray(r["terminal_initial_raw64"].numpy(),dtype=np.float64)
            fin_raw=np.asarray(r["terminal_final_raw64"].numpy(),dtype=np.float64)
            fin_quad=np.asarray(r["final_mse"].numpy(),dtype=np.float64)
            delta=np.zeros((n,33),dtype=np.float64)
            eval_at=np.full((n,33),-1,dtype=np.int64)
            seen=np.zeros((n,33),dtype=np.bool_)
            for mv in r.get("moves",[]):
                rr=np.asarray(mv["rows"].numpy(),dtype=np.int64)
                ss=np.asarray(mv["updates_after"].numpy(),dtype=np.int64)
                dd=np.asarray(mv["predicted_delta"].numpy(),dtype=np.float64)
                ee=np.asarray(mv["evaluations"].numpy(),dtype=np.int64)
                if not np.isfinite(dd).all() or np.any(rr<0) or np.any(rr>=n) or np.any(ss<1) or np.any(ss>32):
                    raise AssertionError(f"invalid move trace {gp.name}/{module}")
                if seen[rr,ss].any(): raise AssertionError(f"duplicate row/update {gp.name}/{module}")
                seen[rr,ss]=True
                delta[rr,ss]=dd
                eval_at[rr,ss]=ee
                move_total += len(rr)
            if not np.array_equal(seen[:,1:].sum(1),updates):
                raise AssertionError(f"move/update count mismatch {gp.name}/{module}")
            for i,u in enumerate(updates.tolist()):
                if u and not seen[i,1:u+1].all(): raise AssertionError(f"noncontiguous trace {gp.name}/{module}/{i}")
                if seen[i,u+1:].any(): raise AssertionError(f"trace after terminal update {gp.name}/{module}/{i}")
            # final_mse is the searcher's accumulated C-quadratic loss; reverse the saved
            # deltas to recover its exact start, then replay every requested checkpoint.
            init_quad=fin_quad-delta[:,1:].sum(1)
            err=np.abs(init_quad+delta[:,1:].sum(1)-fin_quad)
            max_reconstruction=max(max_reconstruction,float(err.max(initial=0.0)))
            for i in range(n):
                key=(layer,module,i)
                saved=rowcsv.pop(key,None)
                if saved is None: raise AssertionError(f"missing row summary {key}")
                if int(saved["updates"])!=int(updates[i]) or int(saved["evaluations"])!=int(evaluations[i]):
                    raise AssertionError(f"row CSV/trace mismatch {key}")
                if abs(float(saved["initial_raw64_mse"])-ini_raw[i])>1e-14*max(1.0,abs(ini_raw[i])):
                    raise AssertionError(f"raw-X initial mismatch {key}")
                if abs(float(saved["final_raw64_mse"])-fin_raw[i])>1e-14*max(1.0,abs(fin_raw[i])):
                    raise AssertionError(f"raw-X final mismatch {key}")
                statuses[int(status[i])]=statuses.get(int(status[i]),0)+1
                cum=np.cumsum(delta[i,1:])
                budget_info={}
                for k in BUDGETS:
                    step=min(k,int(updates[i]))
                    loss=float(init_quad[i]+(cum[step-1] if step else 0.0))
                    if updates[i]>=k:
                        ev=int(eval_at[i,k])
                        if ev<0: raise AssertionError(f"missing per-step eval counter {key}/k{k}")
                    else:
                        ev=int(evaluations[i])
                    if ev<0 or ev>128*d: raise AssertionError(f"candidate cap exceeded {key}/k{k}")
                    budget_info[k]=(loss,ev,step)
                rec.append(dict(layer=layer,module=module,projection=module.split(".")[-1],input_dim=d,row=i,
                    updates=int(updates[i]),evaluations=int(evaluations[i]),eval_cap=128*d,status=int(status[i]),
                    initial_quad=float(init_quad[i]),final_quad=float(fin_quad[i]),initial_raw64=float(ini_raw[i]),
                    final_raw64=float(fin_raw[i]),delta=delta[i,1:].copy(),budget=budget_info))
        del cp
    if rowcsv: raise AssertionError(f"unmatched row summaries {version}: {len(rowcsv)}")
    assert module_count==168 and len(rec)==304128
    mh=hashlib.sha256()
    for r in sorted(group_hash_rows,key=lambda x:x["name"]):
        mh.update(f'{r["name"]}\t{r["bytes"]}\t{r["sha256"]}\n'.encode())
    dump_json(f"{version}_TRACE_MANIFEST.json",dict(group_count=len(groups),modules=module_count,rows=len(rec),
        manifest_sha256=mh.hexdigest(),files=group_hash_rows))
    trace_audit.append(dict(version=version,groups=len(groups),modules=module_count,rows=len(rec),
        accepted_move_events=move_total,max_final_quadratic_reconstruction_residual=max_reconstruction,
        move_count_matches_row_updates=True,per_step_eval_counts_recoverable=True,
        intermediate_rawX_direct_loss="NOT_STORED",recorded_C_quadratic_loss="RECOVERABLE",
        row_csv_rawX_endpoint_match=True,status_1_no_candidate_or_strict_descent=statuses.get(1,0),
        status_2_32_update_cap=statuses.get(2,0),status_3_candidate_eval_cap=statuses.get(3,0)))
    write_csv(f"{version}_ROW_UPDATE_DISTRIBUTION.csv",[
        dict(version=version,updates_per_row=k,rows=sum(x["updates"]==k for x in rec),
             fraction=sum(x["updates"]==k for x in rec)/len(rec)) for k in range(33)])
    initial_raw=sum(x["initial_raw64"] for x in rec); final_raw=sum(x["final_raw64"] for x in rec)
    initial_quad=sum(x["initial_quad"] for x in rec); final_quad=sum(x["final_quad"] for x in rec)
    endpoints.append(dict(version=version,rows=len(rec),rawX_initial_sum=initial_raw,rawX_final_sum=final_raw,
        rawX_full_path_drop=initial_raw-final_raw,rawX_drop_pct_initial=100*(initial_raw-final_raw)/initial_raw,
        C_quadratic_initial_reconstructed_sum=initial_quad,C_quadratic_final_sum=final_quad,
        C_quadratic_full_path_drop=initial_quad-final_quad,
        C_quadratic_drop_pct_initial=100*(initial_quad-final_quad)/initial_quad))
    full_drop=initial_quad-final_quad
    for k in BUDGETS:
        after=sum(x["budget"][k][0] for x in rec)
        evals=sum(x["budget"][k][1] for x in rec)
        acc=sum(x["budget"][k][2] for x in rec)
        reach=sum(x["updates"]>=k for x in rec)
        full_evals=sum(x["evaluations"] for x in rec)
        drop=initial_quad-after
        budgets.append(dict(version=version,accept_update_cap=k,rows=len(rec),rows_reaching_budget_cap=reach,
            rows_at_original_32_cap=sum(x["updates"]==32 for x in rec),accepted_updates_under_budget=acc,
            candidate_evaluations_under_budget=evals,candidate_evaluations_pct_of_32_run=100*evals/full_evals,
            initial_C_quadratic_loss_sum=initial_quad,C_quadratic_loss_after_budget=after,
            C_quadratic_loss_drop=drop,C_quadratic_drop_pct_initial=100*drop/initial_quad,
            full32_C_quadratic_drop=full_drop,retained_full32_drop_pct=100*drop/full_drop,
            measured_runtime="NOT_MEASURED"))
        buckets={}
        for x in rec:
            key=(x["layer"],x["module"],x["input_dim"])
            z=buckets.setdefault(key,dict(rows=0,reach=0,accepted=0,evals=0,initial=0.0,after=0.0,full=0.0))
            loss,ev,steps=x["budget"][k]
            z["rows"]+=1;z["reach"]+=x["updates"]>=k;z["accepted"]+=steps;z["evals"]+=ev
            z["initial"]+=x["initial_quad"];z["after"]+=loss;z["full"]+=x["initial_quad"]-x["final_quad"]
        for (layer,module,d),z in sorted(buckets.items()):
            drop=z["initial"]-z["after"]
            strata.append(dict(version=version,accept_update_cap=k,layer=layer,module=module,
                projection=module.split(".")[-1],input_dim=d,rows=z["rows"],rows_reaching_budget_cap=z["reach"],
                accepted_updates=z["accepted"],candidate_evaluations=z["evals"],
                initial_C_quadratic_loss=z["initial"],budget_C_quadratic_loss=z["after"],
                C_quadratic_drop=drop,full32_C_quadratic_drop=z["full"],
                retained_full32_drop_pct=100*drop/z["full"] if z["full"] else ""))
    for lo,hi,label in ((1,1,"1"),(2,4,"2-4"),(5,8,"5-8"),(9,16,"9-16"),(17,32,"17-32")):
        gain=moves_n=0
        for x in rec:
            top=min(hi,x["updates"])
            if top>=lo:
                gain += float(-x["delta"][lo-1:top].sum())
                moves_n += top-lo+1
        marginals.append(dict(version=version,update_interval=label,accepted_move_count=moves_n,
            C_quadratic_loss_drop=gain,pct_of_full32_drop=100*gain/full_drop,
            intermediate_rawX_direct_loss="NOT_STORED"))

write_csv("TRACE_RECOVERY_AUDIT.csv",trace_audit)
write_csv("LOCAL_FULL_PATH_ENDPOINTS.csv",endpoints)
write_csv("BUDGET_TRAJECTORY.csv",budgets)
write_csv("BUDGET_TRAJECTORY_BY_LAYER_PROJECTION_DIM.csv",strata)
write_csv("LATE_MARGINAL_RETURNS.csv",marginals)
layer_summary=[]
proj_summary=[]
for version in ("v55","v56"):
    for k in BUDGETS:
        source=[r for r in strata if r["version"]==version and int(r["accept_update_cap"])==k]
        for layer in sorted({int(r["layer"]) for r in source}):
            ss=[r for r in source if int(r["layer"])==layer]
            ini=sum(float(r["initial_C_quadratic_loss"]) for r in ss)
            aft=sum(float(r["budget_C_quadratic_loss"]) for r in ss)
            full=sum(float(r["full32_C_quadratic_drop"]) for r in ss)
            dr=ini-aft
            layer_summary.append(dict(version=version,accept_update_cap=k,layer=layer,
                rows=sum(int(r["rows"]) for r in ss),accepted_updates=sum(int(r["accepted_updates"]) for r in ss),
                candidate_evaluations=sum(int(r["candidate_evaluations"]) for r in ss),
                C_quadratic_drop=dr,full32_C_quadratic_drop=full,
                retained_full32_drop_pct=100*dr/full if full else ""))
        for projection in sorted({r["projection"] for r in source}):
            ss=[r for r in source if r["projection"]==projection]
            ini=sum(float(r["initial_C_quadratic_loss"]) for r in ss)
            aft=sum(float(r["budget_C_quadratic_loss"]) for r in ss)
            full=sum(float(r["full32_C_quadratic_drop"]) for r in ss)
            dr=ini-aft
            proj_summary.append(dict(version=version,accept_update_cap=k,projection=projection,
                input_dim=int(ss[0]["input_dim"]),rows=sum(int(r["rows"]) for r in ss),
                accepted_updates=sum(int(r["accepted_updates"]) for r in ss),
                candidate_evaluations=sum(int(r["candidate_evaluations"]) for r in ss),
                C_quadratic_drop=dr,full32_C_quadratic_drop=full,
                retained_full32_drop_pct=100*dr/full if full else ""))
write_csv("BUDGET_TRAJECTORY_BY_LAYER.csv",layer_summary)
write_csv("BUDGET_TRAJECTORY_BY_PROJECTION_DIM.csv",proj_summary)

# Synchronized stage wall. Some timers are nested; the source report explicitly forbids
# treating their sum as process wall. Missing internal splits are represented as NOT_MEASURED.
stage_rows=[]
category={
 "C_construction":("C build","FP64 Gram build"),
 "fast_initialization":("initial state","code/grid/e/p initialization"),
 "initial_rawX_FP64_review":("numerical review","initial raw-X FP64 initialization review"),
 "direct_recompute":("direct scoring","initial direct output loss; not per-step scoring"),
 "fast_candidate_scoring":("candidate scoring + selection","score all candidates and select minimum in one timer"),
 "fast_propose_and_settle":("proposal/accept/update","proposal, settle, accepted code and p updates combined"),
 "fast_p_check_refresh":("numerical review","periodic p/cache refresh, separately timed"),
 "fast_final_cache_grid_pack":("terminal check","final cache/grid/pack check"),
 "independent_full_terminal_FP64":("terminal check","independent full-matrix raw-X FP64 reconstruction"),
 "dependency_input_capture":("input propagation","formal dependency input capture"),
 "down_commit_prefix_refresh":("input propagation","committed block prefix refresh"),
 "sum_of_24_native_propagation_wall":("input propagation","fixed-article native forward/hook checks"),
 "packing_checkpoint_IO":("I/O","group pack/checkpoint write"),
 "whole_group_commit":("I/O","group commit"),
 "final_save":("I/O","final checkpoint save"),
 "reload":("I/O","reload"),
 "pack_reload_all_grid_check":("I/O","pack/reload/grid check"),
 "input_hash":("source validation","input hash"),
 "v52_Gram_source_hash":("source validation","C provenance hash"),
 "v52_input_pre_Gram_source_check":("source validation","input source check before Gram"),
 "v54_input_source_hash":("source validation","inherited input hash"),
 "committed_group_prefix_restore_validation":("resume validation","restored group and prefix validation"),
 "formal_model_load":("fixed overhead","model load"),
 "initial_embedding_capture":("input propagation","initial embedding capture"),
 "zero_model_output":("deployment check","zero model output check"),
 "unpack_pack_zero_commit":("deployment check","zero unpack/pack round trip"),
 "formal_warmup":("fixed overhead","warmup")}
def add_stage(ver,scope,stage,calls,seconds,source,kind,notes=""):
 cat,desc=category.get(stage,("other measured",stage))
 stage_rows.append(dict(version=ver,scope=scope,category=cat,stage=stage,calls=calls,
     measured_seconds=seconds,measurement=kind,source=source,description=desc,notes=notes))

v56cost=read_csv(SOURCES["v56"]/"reports/STAGE_COSTS.csv")
for r in v56cost:
    if r["mode"]=="formal":
        add_stage("v56 M1 full valid run","all 24 blocks",r["stage"],int(r["observations"]),
            float(r["synchronized_stage_seconds_sum"]),"v56/reports/STAGE_COSTS.csv","synchronized nested stage wall",
            r.get("aggregation_note",""))
for r in read_csv(SOURCES["v56"]/"BLOCK_TIMINGS.csv"):
    add_stage("v56 M1 full valid run",f"block {int(r['layer']):02d}","whole_block_wall",
        int(r["new_groups"]),float(r["seconds"]),"v56/BLOCK_TIMINGS.csv","block wall",
        "block timer includes complete module repair/terminal and propagation; do not add internal timers")
add_stage("v56 M1 full valid run","formal completion","formal_repair_elapsed",1,
    float(json.loads((SOURCES["v56"]/"FORMAL_REPAIR_COMPLETE.json").read_text())["elapsed_seconds"]),
    "v56/FORMAL_REPAIR_COMPLETE.json","runner elapsed","separate scope from block timer sum")
for r in read_csv(SOURCES["v55"]/"STAGE_COST_SUMMARY.csv"):
    if r["mode"]=="formal":
        add_stage("v55 M2 valid recovery chain","new layers 2-23",r["stage"],int(r["calls"]),
            float(r["measured_seconds"]),"v55/STAGE_COST_SUMMARY.csv","synchronized nested stage wall",
            "new v55 formal segment; restored groups have earlier source costs")
for r in read_csv(SOURCES["v54"]/"STAGE_TIMINGS.csv"):
    if r["mode"]=="formal" and int(r["layer"])==1:
        add_stage("v54 segment inherited by valid v55","layer 1",r["stage"],1,float(r["seconds"]),
            "v54/STAGE_TIMINGS.csv",r.get("type","synchronized stage wall"),
            "partial run; layer1 only, propagation check passed")
for r in read_csv(SOURCES["v51"]/"BLOCK_TIMINGS.csv"):
    if int(r["layer"])==0 and int(r["new_groups"])==4:
        add_stage("v51 block0 inherited by valid v55","layer 0","whole_block_wall",4,float(r["seconds"]),
            "v51/BLOCK_TIMINGS.csv","block wall later admitted by v53",
            "block0 only; v51 layer1+ / v52 downstream excluded")
for ver,scope,stg,note in [
 ("v56","all 24 blocks","selection separate from candidate score","combined in fast_candidate_scoring"),
 ("v56","all 24 blocks","code update separate from p update","combined in fast_propose_and_settle"),
 ("v56","all 24 blocks","pure GPU kernel time","not measured; synchronized timer is wall"),
 ("v56","all 24 blocks","low-budget runtime","not measured"),
 ("v55","new layers 2-23","selection separate from candidate score","combined in fast_candidate_scoring"),
 ("v55","new layers 2-23","code update separate from p update","combined in fast_propose_and_settle"),
 ("v55","new layers 2-23","low-budget runtime","not measured")]:
    stage_rows.append(dict(version=ver,scope=scope,category="NOT_MEASURED",stage=stg,calls="",
       measured_seconds="",measurement="NOT_MEASURED",source="",description="",notes=note))
write_csv("STAGE_COSTS.csv",stage_rows)

# Cross-run valid M2 block timer composition is not a single process wall.
v51_l0=sum(float(r["seconds"]) for r in read_csv(SOURCES["v51"]/"BLOCK_TIMINGS.csv") if int(r["layer"])==0 and int(r["new_groups"])==4)
v54_l1=sum(float(r["seconds"]) for r in read_csv(SOURCES["v54"]/"BLOCK_TIMINGS.csv") if int(r["layer"])==1 and int(r["new_groups"])==4)
v55_l2=sum(float(r["seconds"]) for r in read_csv(SOURCES["v55"]/"BLOCK_TIMINGS.csv") if int(r["layer"])>=2 and int(r["new_groups"])==4)
v56_block=sum(float(r["seconds"]) for r in read_csv(SOURCES["v56"]/"BLOCK_TIMINGS.csv"))
dump_json("COST_SCOPE_NOTES.json",dict(v51_valid_block0_seconds=v51_l0,v54_valid_layer1_seconds=v54_l1,
    v55_valid_new_layers2_23_seconds=v55_l2,v55_effective_M2_cross_run_block_timer_sum=v51_l0+v54_l1+v55_l2,
    v56_valid_24block_timer_sum=v56_block,
    note="M2 sum composes validated block timers from three runs; not a single process wall and excludes some setup, restore, CPU, failure and diagnosis time."))
formal={r["stage"]:float(r["synchronized_stage_seconds_sum"]) for r in v56cost if r["mode"]=="formal"}
score,propose,refresh=[formal[x] for x in ("fast_candidate_scoring","fast_propose_and_settle","fast_p_check_refresh")]
rowscore=int(next(r["observations"] for r in v56cost if r["stage"]=="fast_candidate_scoring"))
rowrefresh=int(next(r["observations"] for r in v56cost if r["stage"]=="fast_p_check_refresh"))
assert rowscore==168*32 and rowrefresh==168*5
conditional=[]
for k in (8,16,32):
    refresh_points=sum(t<=k for t in (1,8,16,24,32))
    estimate=(score+propose)*k/32+refresh*refresh_points/5
    baseline=score+propose+refresh
    save=baseline-estimate
    conditional.append(dict(accept_update_cap=k,baseline_score_propose_refresh_seconds=baseline,
        same_call_cost_conditional_estimate_seconds=estimate,conditional_saving_seconds=save,
        v56_24block_wall_seconds=v56_block,conditional_v56_block_wall_saving_pct=100*save/v56_block,
        conditional_block_wall_seconds=v56_block-save,
        assumptions="scoring and propose/settle time scale linearly with loop count; p-refresh count follows code checkpoints [1,8,16,24,32]; C, input/propagation, initialization, direct initial score, terminal check, IO and deployment stay fixed; not measured"))
write_csv("CONDITIONAL_COST_SCENARIO.csv",conditional)

# Evidence status matrix, preserving v53's scoped findings.
evidence=[
 dict(version="v18 M1/M2 initialization + base measurements",status="PROTOCOL_SUPPORTED_WITH_LIMITED_RECHECK",
  propagation="Recorded Transformers 4.49 tuple API supports expected return type; per-call type trace and exhaustive prefix validation absent; erroneous Tensor branch NOT_EVIDENCED.",
  checkpoint_quality="Source/checkpoint/window hashes audited; v59 independently remeasured the same fixed checkpoints in one process.",
  use="Use v59 for fixed-checkpoint quality; qualify historical calibration-input proof."),
 dict(version="v45/v46/v49 repair",status="UNKNOWN_PROPAGATION",
  propagation="Same indexing exists; old per-call return type is not evidenced; neither actual Tensor branch nor correct execution is established.",
  checkpoint_quality="Saved checkpoint and PPL are actual measurements; calibration propagation remains UNKNOWN.",
  use="Background only; exclude from primary trajectory and timing."),
 dict(version="v51/v52",status="PARTIAL; DOWNSTREAM_INVALID",
  propagation="v53 confirmed the block0-to-layer1 broadcast boundary; only block0 groups were independently admitted. v52 layer1+ outputs are invalid.",
  checkpoint_quality="Incomplete; no complete valid model PPL. Preserve failed/diagnostic cost separately.",
  use="Use v51 block0 trace/cost only where v53 gate admits; exclude all downstream."),
 dict(version="v53",status="INTERFACE_AND_SCOPED_PROPAGATION_PASS",
  propagation="Correct decoder extraction; 128-article block0 group input and block0-to-layer1 boundary checked; three fixed articles compared through 24 blocks.",
  checkpoint_quality="Audit only; no full model/PPL.",
  use="Admit validated block0 inheritance only."),
 dict(version="v54",status="VALID_PARTIAL_M2",
  propagation="Block0 restored; layer1 seven Linears/12,672 rows repaired; 12/12 fixed-article hidden/qkv comparisons passed at block1 boundary.",
  checkpoint_quality="Partial; no complete model/PPL.",
  use="Layer1 trajectory/cost inherited by v55."),
 dict(version="v55",status="VALID_COMPLETE_M2_PLUS_REPAIR",
  propagation="Corrected decoder; formal 128-article source hashes; all 24 blocks and 141 recorded fixed-article native comparisons passed.",
  checkpoint_quality="Deployment checks passed; v59 supplies the main quality table.",
  use="M2 trace; split recovery-chain cost by valid source run."),
 dict(version="v56",status="VALID_COMPLETE_M1_PLUS_REPAIR",
  propagation="Started from layer0; 141/141 fixed-article native-forward/hook comparisons passed through 24 blocks.",
  checkpoint_quality="Deployment checks passed; v59 supplies main quality table.",
  use="M1 trace and primary complete-run cost."),
 dict(version="v57-v59",status="EVALUATION_ONLY",
  propagation="No calibration or repair; unified evaluation cannot retroactively verify calibration propagation.",
  checkpoint_quality="v59 eight complete evaluation results in one process; independent window/token recalc passed.",
  use="v59 four-arm main table only.")]
write_csv("VERSION_EVIDENCE.csv",evidence)

# Source file hashes and trace manifests. Hashing and torch loading are included in CPU_COST.
srcpaths=[
 SOURCES["v59"]/"REPORT.md",SOURCES["v59"]/"INDEPENDENT_WINDOW_RECALC.csv",SOURCES["v59"]/"RUN_LOCK.json",SOURCES["v59"]/"PROCESS_ENVIRONMENT.json",
 SOURCES["v55"]/"REPORT.md",SOURCES["v55"]/"PROTOCOL.json",SOURCES["v55"]/"SOURCE_AUDIT.json",SOURCES["v55"]/"FORMAL_REPAIR_COMPLETE.json",SOURCES["v55"]/"MODULE_RESULTS.csv",SOURCES["v55"]/"BLOCK_TIMINGS.csv",SOURCES["v55"]/"STAGE_COST_SUMMARY.csv",SOURCES["v55"]/"PROPAGATION_NATIVE_RESULTS.csv",SOURCES["v55"]/"HISTORICAL_QUALITY_BACKGROUND.csv",
 SOURCES["v56"]/"REPORT.md",SOURCES["v56"]/"PROTOCOL.json",SOURCES["v56"]/"SOURCE_AUDIT.json",SOURCES["v56"]/"FORMAL_REPAIR_COMPLETE.json",SOURCES["v56"]/"MODULE_RESULTS.csv",SOURCES["v56"]/"BLOCK_TIMINGS.csv",SOURCES["v56"]/"reports/STAGE_COSTS.csv",SOURCES["v56"]/"PROPAGATION_NATIVE_RESULTS.csv",
 SOURCES["v54"]/"REPORT.md",SOURCES["v54"]/"STAGE_TIMINGS.csv",SOURCES["v54"]/"BLOCK_TIMINGS.csv",
 SOURCES["v51"]/"BLOCK_TIMINGS.csv",SOURCES["v53"]/"REPORT.md",SOURCES["v53"]/"HISTORICAL_IMPACT.csv",SOURCES["v53"]/"VALID_RESUME_BOUNDARY.json",
 OUT/"reproduce_v60.py"]
write_csv("SOURCE_AUDIT.csv",[dict(source_path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in srcpaths if p.exists()])

cpu=time.process_time()-CPU0
wall=time.perf_counter()-WALL0
dump_json("CPU_COST.json",dict(mode="CPU_ONLY",cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    torch_version=torch.__version__,numpy_version=np.__version__,cuda_api_used=False,
    groups_loaded=192,row_csv_files_read=336,trace_rows=sum(x["rows"] for x in trace_audit),
    python_process_cpu_seconds=cpu,python_process_wall_seconds=wall,
    note="Times include source hashing, torch.load(map_location=cpu), trace replay, CSV/JSON analyses; excludes prior manual read-only inspection and authoring."))
print(json.dumps(dict(trace_audit=trace_audit,budget_rows=budgets,endpoints=endpoints,
    conditional=conditional,cpu_seconds=cpu,wall_seconds=wall),ensure_ascii=False,indent=2))
