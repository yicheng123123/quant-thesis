import csv, hashlib, json, shutil, time
from pathlib import Path
import torch
ROOT=Path(r"E:\quant-thesis"); BASE=ROOT/"experiments/range_search_w4/results"
OUT=BASE/"v55_m2_decoder_fixed_integer_repair"; V54=BASE/"v54_m2_decoder_fixed_integer_repair"
V53=BASE/"v53_decoder_interface_propagation_audit"; V52=BASE/"v52_m2_fast_integer_repair_fp64_initial_review"
V18=BASE/"range_w4_fullmodel_v18"; GROUPS=["qkv","o","up_gate","down"]; t0=time.perf_counter()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()
def rd(p): return json.loads(Path(p).read_text(encoding="utf8"))
def wr(p,x):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
assert OUT.exists() and not (OUT/"PREPARATION.json").exists() and {q.name for q in OUT.iterdir()} <= {"prepare_v55.py","prepare_v55.log"},"refusing overwrite"
for d in ["formal/groups","formal/rows","R1/checkpoint/final","evaluation","reports","diagnostics"]: (OUT/d).mkdir(parents=True,exist_ok=True)
for n in ["runtime_v54.py","decoder_interface.py","audit_sources.py","fast_accept_v52.py","initialization_review.py","SOURCE_AUDIT.json","NUMERICAL_VALIDATION.json","CPU_TESTS.json","INITIALIZATION_REVIEW_REUSE.json","PROTOCOL.json","HISTORICAL_SOURCE_LOCK.json","HISTORICAL_QUALITY_BACKGROUND.csv","PRE_GPU_CHECK.json"]:
 src=V54/n
 if src.exists(): shutil.copy2(src,OUT/n); assert sha(src)==sha(OUT/n)
assert sha(OUT/"runtime_v54.py")==sha(V53/"runtime_v53.py")
assert sha(OUT/"fast_accept_v52.py")==sha(V52/"fast_accept_v52.py")
assert sha(OUT/"initialization_review.py")==sha(V52/"initialization_review.py")
assert sha(OUT/"PROTOCOL.json")==sha(V54/"PROTOCOL.json")
assert rd(OUT/"PROTOCOL.json")["initialization"]["model_sha256"]=="810d6329cdc51d995543dbb701e2368568b19459a44f340df1a8a0019d447eed"
assert rd(OUT/"PROTOCOL.json")["calibration"]["manifest_sha256"]=="664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3"
shutil.copytree(V54/"original_codes",OUT/"original_codes")
lock=rd(V54/"PARTIAL_CHECKPOINT_LOCK.json"); assert lock["validated_blocks"]==[0,1] and len(lock["group_checkpoints"])==8
expected={(int(x["layer"]),x["group"]):x for x in lock["group_checkpoints"]}
for li in range(2):
 for g in GROUPS:
  rec=expected[(li,g)]; src=V54/"formal/groups"/f"layer_{li:02d}_{g}.pt"; dst=OUT/"formal/groups"/src.name
  assert sha(src)==rec["sha256"]; shutil.copy2(src,dst); assert sha(dst)==rec["sha256"]
  cp=torch.load(dst,map_location="cpu",weights_only=False)
  assert cp["committed"] and cp["layer"]==li and cp["group"]==g and cp["protocol_sha256"]==sha(OUT/"PROTOCOL.json")
  assert cp["input_sha256"]==rec["input_sha256"]
assert len(list((OUT/"formal/groups").glob("*.pt")))==8 and not list((OUT/"formal/groups").glob("layer_02_*.pt"))
for n in ["COMMIT_STATUS.json","PREFIX_STATUS_layer_00.json","PREFIX_STATUS_layer_01.json"]: shutil.copy2(V54/"formal"/n,OUT/"formal"/n)
assert rd(OUT/"formal/COMMIT_STATUS.json")["last_committed_layer"]==1
for li in range(24): assert (OUT/"original_codes"/f"layer_{li:02d}.pt").exists()
runner=(V54/"run_v54.py").read_text(encoding="utf8")
repls=[
("OUT=BASE/'v54_m2_decoder_fixed_integer_repair'","OUT=BASE/'v55_m2_decoder_fixed_integer_repair'"),
("CAP=10800;","CAP=12600;"),("V54_COST_PRECHECK.json","V55_COST_PRECHECK.json"),
("RUN_ID='v54-'","RUN_ID='v55-'"),("version='v54'","version='v55'"),
("evaluation/v54_","evaluation/v55_"),("V54_BLOCK_COMPLETE","V55_BLOCK_COMPLETE"),
("frozen v54 source changed","frozen v55 source changed"),
("v52 180-minute cumulative GPU budget exhausted","v55 12600-second cumulative shared budget exhausted"),
("formal complete repair and deployment/evaluation","v55 complete repair and deployment/evaluation")]
for a,b in repls: assert a in runner,a; runner=runner.replace(a,b)
(OUT/"run_v55.py").write_text(runner,encoding="utf8")
budget=(V54/"budget.py").read_text(encoding="utf8"); assert "CAP=10800.;VERSION='v54'" in budget
(OUT/"budget.py").write_text(budget.replace("CAP=10800.;VERSION='v54'","CAP=12600.;VERSION='v55'",1),encoding="utf8")
resume={"status":"V54_EIGHT_COMMITTED_GROUPS_SOURCE_VERIFIED","base_M2_checkpoint_sha256":"810d6329cdc51d995543dbb701e2368568b19459a44f340df1a8a0019d447eed",
"v53_correct_decoder_runtime_sha256":sha(OUT/"runtime_v54.py"),"frozen_v54_protocol_sha256":sha(OUT/"PROTOCOL.json"),
"calibration_manifest_sha256":"664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3",
"groups":[{"layer":li,"group":g,"checkpoint_sha256":sha(OUT/"formal/groups"/f"layer_{li:02d}_{g}.pt"),"expected_v54_sha256":expected[(li,g)]["sha256"],
"input_sha256":expected[(li,g)]["input_sha256"],"C_sha256_source_only":expected[(li,g)]["C_sha256"],"restore_without_repair":True} for li in range(2) for g in GROUPS],
"group_count":8,"next_repair_group":"layer2/qkv","v52_layer2_plus_state_copied":False,"repair_implementation_changed":False}
wr(OUT/"RESUME_SOURCE_LOCK.json",resume)
wr(OUT/"BLOCK0_INHERITANCE.json",{"pass":True,"source":"v54 PARTIAL_CHECKPOINT_LOCK and exact committed group files","blocks":[0,1],"groups_per_block":GROUPS,"group_count":8,
"restore_without_repair":True,"last_committed":"layer1/down","next_start":"layer2/qkv","corrected_decoder_runtime_sha256":sha(OUT/"runtime_v54.py"),
"formal_128_article_boundary_check_required_after_replay":True,"articles":[0,64,127],"no_v52_layer2plus_parameters_inputs_or_C":True})
wr(OUT/"RECOVERY_HISTORY.json",{"entries":[{"version":"v55","source_version":"v54","reused_groups":8,"source_lock_sha256":sha(OUT/"RESUME_SOURCE_LOCK.json"),"repair_restart":"layer2/qkv","v52_layer2plus_state_used":False}]})
ledger=rd(V54/"ACTIVE_JOB_LEDGER.json")
assert abs(ledger["v51_charged_seconds"]+sum(s["charged_seconds"] for s in ledger["segments"])-5980.039842538834)<1e-6
ledger["cap_seconds"]=12600.0
ledger["budget_extension"]={"old_cap_seconds":10800.0,"added_seconds":1800.0,"new_cap_seconds":12600.0,"inherited_used_seconds":5980.039842538834,
"inherited_remaining_under_new_cap_seconds":6619.960157461166,"not_a_reset":True}
cpu_audit_observed=19.5; setup_failure_charge=30.0; assertion_failure_wall=10.5
for label,secs,basis in [("v55 pre-run read-only source/protocol/checkpoint and GPU process audit",cpu_audit_observed,"CPU-only measured tool wall"),("v55 preparation assertion failure before any model/GPU call",assertion_failure_wall,"CPU-only measured failed process wall"),
("v55 failed first preparation attempt before v55 directory existed",setup_failure_charge,"CPU-only conservative cap charge; actual elapsed unavailable; no artifacts or GPU calls")]:
 now=time.time(); ledger["segments"].append({"version":"v55","label":label,"start_epoch":now-secs,"end_epoch":now,"charged_seconds":secs,"cpu_only":True,"GPU_calls":0,"charge_basis":basis})
stages=list(csv.DictReader((V54/"STAGE_TIMINGS.csv").open(encoding="utf8"))); sums={}
for row in stages:sums[row["stage"]]=sums.get(row["stage"],0.0)+float(row["seconds"])
native=list(csv.DictReader((V54/"PROPAGATION_NATIVE_TIMINGS.csv").open(encoding="utf8")))
restart=sum(sums.get(k,0.0) for k in ["formal_model_load","initial_embedding_capture","committed_group_prefix_restore_validation","formal_warmup"])
restart+=sum(float(x["seconds"]) for x in native if int(x["block_completed"]) in [0,1])
old_online=4851.937226799946; cpu_sec=time.perf_counter()-t0
now=time.time(); ledger["segments"].append({"version":"v55","label":"CPU-only isolated v55 preparation and resume-source audit","start_epoch":now-cpu_sec,"end_epoch":now,
"charged_seconds":cpu_sec,"cpu_only":True,"GPU_calls":0,"measurement":"preparation script perf_counter wall"})
total=ledger["v51_charged_seconds"]+sum(s["charged_seconds"] for s in ledger["segments"])
ledger["v55_pre_run_shared_elapsed_seconds"]=total; ledger["v55_cpu_only_preparation_seconds"]=cpu_sec
wr(OUT/"ACTIVE_JOB_LEDGER.json",ledger)
wr(OUT/"BUDGET_EXTENSION.json",{"baseline_used_seconds":5980.039842538834,"baseline_user_rounded_seconds":5980.040,"old_cap_seconds":10800.0,
"explicit_extension_seconds":1800.0,"new_cap_seconds":12600.0,"inherited_available_after_extension_seconds":6619.960157461166,
"new_cpu_preparation_charges_seconds":cpu_audit_observed+setup_failure_charge+assertion_failure_wall+cpu_sec,"current_shared_elapsed_seconds":total,
"current_shared_remaining_seconds":12600.0-total,"old_online_remaining_projection_seconds":old_online,
"old_projection_includes_finish_and_two_PPL_reserve_seconds":360.0,"restart_replay_added_projection_seconds":restart,
"updated_total_remaining_projection_seconds":old_online+restart,"updated_headroom_seconds":12600.0-total-old_online-restart,
"projection_not_measurement":True,"GPU_model_setting":"Luna Max requested; interface cannot verify",
"ordinary_sandbox":"previously failed; approved outside-sandbox entry used","CPU_and_GPU_accounted_separately":True,"ledger_not_reset":True})
pre={"pass":True,"go":12600.0-total>=old_online+restart,"historical_online_projection_after_block1_seconds":old_online,
"historical_online_projection_already_includes_finish_reserve_seconds":360.0,"fresh_process_restart_replay_measured_projection_seconds":restart,
"projected_remaining_seconds_including_all_reserves":old_online+restart,"finish_reserve_seconds":360.0,"uncertainty_reserve_inside_old_projection_seconds":120.0,
"current_shared_elapsed_seconds":total,"current_shared_remaining_seconds":12600.0-total,"projected_headroom_seconds":12600.0-total-old_online-restart,
"basis":"v54 ONLINE_BUDGET_STATUS plus measured v54 model load, embedding, group restore, warmup, and two native validation timers; no double subtraction of 360s.",
"preserved_search_protocol_sha256":sha(OUT/"PROTOCOL.json"),"only_cap_extension_outside_protocol":True,"projection_is_not_measurement":True}
wr(OUT/"V55_COST_PRECHECK.json",pre); assert pre["go"],"preflight says remaining work does not fit; no GPU started"
prep={"status":"CPU_SOURCE_AUDIT_PASS","version":"v55","directories_precreated_and_writable":True,"M2_checkpoint_hash_verified":True,
"calibration_manifest_hash_verified":True,"v53_decoder_runtime_byte_equal":True,"v49_fast_acceptance_and_v52_initialization_review_byte_equal":True,
"exact_v54_protocol_preserved":True,"only_eight_v54_groups_present":True,"all_24_original_M2_code_maps_present":True,
"v52_layer2plus_repair_state_used":False,"GPU_job_started":False,"cpu_seconds":cpu_sec,"cpu_observed_audit_tool_wall_seconds":cpu_audit_observed,
"failed_setup_attempt_cpu_cap_charge_seconds":setup_failure_charge,"failed_preparation_assertion_measured_seconds":assertion_failure_wall,"torch_version_preparation":torch.__version__,
"torch_cuda_available_checked_without_cuda_work":torch.cuda.is_available(),"resume_next_group":"layer2/qkv"}
wr(OUT/"PREPARATION.json",prep)
(OUT/"V55_BUDGET_EXTENSION.md").write_text("# v55 shared-budget extension\n\nThe frozen v54 search protocol is copied byte-for-byte so all eight committed group checkpoints retain their protocol hash. The user explicitly extends the shared cumulative cap from 10,800 to 12,600 seconds (+1,800 seconds). The inherited charged baseline is 5,980.039842538834 seconds; this is an extension, not a reset. The historical 4,851.937226799946-second online estimate already includes a 360-second save/reload/evaluation reserve. Fresh-process restore and replay is added separately from measured v54 stage timings. The new cap and CPU-only preparation charges are recorded in BUDGET_EXTENSION.json and ACTIVE_JOB_LEDGER.json. Search, numerical rules, inputs, grid, and evaluator remain frozen.\n",encoding="utf8")
probe=OUT/"diagnostics"/".write_probe"; probe.write_text("ok",encoding="utf8"); probe.unlink()
free=shutil.disk_usage(OUT).free; assert free>3_000_000_000,f"insufficient free disk space: {free}"
files=["PROTOCOL.json","run_v55.py","runtime_v54.py","decoder_interface.py","audit_sources.py","fast_accept_v52.py","initialization_review.py","budget.py",
"INITIALIZATION_REVIEW_REUSE.json","BLOCK0_INHERITANCE.json","RESUME_SOURCE_LOCK.json","BUDGET_EXTENSION.json","V55_COST_PRECHECK.json","PREPARATION.json",
"SOURCE_AUDIT.json","NUMERICAL_VALIDATION.json","CPU_TESTS.json"]
wr(OUT/"FROZEN_RUN_LOCK.json",{"pass":True,"files":{n:sha(OUT/n) for n in files},"v53_runtime_sha256":sha(V53/"runtime_v53.py"),
"v53_validation_result_sha256":sha(V53/"GPU_VALIDATION_RESULT.json"),"v53_boundary_sha256":sha(V53/"VALID_RESUME_BOUNDARY.json"),
"v52_core_sha256":sha(V52/"fast_accept_v52.py"),"v52_initialization_review_sha256":sha(V52/"initialization_review.py"),
"v52_acceptance_rules_sha256":sha(BASE/"v48_fast_numerical_acceptance/ACCEPTANCE_RULES.md"),"protocol_sha256":sha(OUT/"PROTOCOL.json"),
"algorithm_and_tolerance_change":False,"v54_runtime_and_search_behavior_reused":True,"only_shared_cap_extended":True,
"no_v50_gram":True,"no_P3":True,"resume_group_hashes":{f"layer_{li:02d}_{g}":sha(OUT/"formal/groups"/f"layer_{li:02d}_{g}.pt") for li in range(2) for g in GROUPS}})
assert all(sha(OUT/n)==h for n,h in rd(OUT/"FROZEN_RUN_LOCK.json")["files"].items())
print(json.dumps({"preparation":"PASS","precheck":pre,"cpu_preparation_seconds":cpu_sec,"restart_projection_seconds":restart,"disk_free_bytes":free,"groups":8,"next":"layer2/qkv"},ensure_ascii=False))

