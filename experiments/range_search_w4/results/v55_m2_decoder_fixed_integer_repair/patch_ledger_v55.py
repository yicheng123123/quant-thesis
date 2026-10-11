import csv,hashlib,json,time
from pathlib import Path
O=Path(r"E:\quant-thesis\experiments\range_search_w4\results\v55_m2_decoder_fixed_integer_repair")
t0=time.perf_counter()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(n): return json.loads((O/n).read_text(encoding="utf8"))
def wr(n,x): (O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
# Preserve the v54 budget arithmetic while avoiding eager evaluation of a missing legacy timestamp.
p=O/"budget.py"; s=p.read_text(encoding="utf8")
old="vals=[(s.get('version'),s.get('charged_seconds',max(0,time.time()-s['start_epoch']))) for s in x['segments']]"
new="vals=[(s.get('version'),s['charged_seconds'] if 'charged_seconds' in s else max(0,time.time()-s['start_epoch'])) for s in x['segments']]"
assert old in s
p.write_text(s.replace(old,new,1),encoding="utf8")
ledger=rd("ACTIVE_JOB_LEDGER.json")
def set_charge(label,secs):
 found=[x for x in ledger["segments"] if x.get("label")==label]
 assert len(found)==1,(label,len(found))
 found[0]["charged_seconds"]=secs
set_charge("v55 pre-run read-only source/protocol/checkpoint and GPU process audit",22.3)
set_charge("CPU-only isolated v55 preparation and resume-source audit",10.7)
# CPU-only orchestration/preflight actions, measured command wall from this turn.
known=[
 ("v55 static source/ledger audit before preparation",3.3),
 ("v55 frozen diff and environment preflight",7.6),
 ("v55 formal runner startup failure before budget hook",13.841,"KeyError from legacy fixed-charge record lacking start_epoch; no model load/GPU compute"),
 ("v55 failure log and shared-ledger inspection",3.1)]
for item in known:
 label,secs=item[0],item[1]; now=time.time()
 row={"version":"v55","label":label,"start_epoch":now-secs,"end_epoch":now,"charged_seconds":secs,"cpu_only":True,"GPU_calls":0}
 if len(item)>2: row["failure_reason"]=item[2]
 ledger["segments"].append(row)
# Charge this repair/setup process itself with a 1s conservative final-write guard.
patch_seconds=time.perf_counter()-t0+1.0
now=time.time()
ledger["segments"].append({"version":"v55","label":"v55 ledger reader compatibility fix and lock refresh","start_epoch":now-patch_seconds,"end_epoch":now,
"charged_seconds":patch_seconds,"cpu_only":True,"GPU_calls":0,"basis":"measured patch script wall plus 1s guard"})
ledger["budget_extension"]["not_a_reset"]=True
total=ledger["v51_charged_seconds"]+sum(x["charged_seconds"] for x in ledger["segments"])
ledger["v55_pre_run_shared_elapsed_seconds"]=total
ledger["v55_cpu_only_charges_before_formal_start"]=total-5980.039842538834
# A small fixed allowance covers the final shell/Python launcher before budget.start.
launch_preflight=5.0; now=time.time()
ledger["segments"].append({"version":"v55","label":"v55 final pre-budget launcher and GPU-idle recheck","start_epoch":now-launch_preflight,"end_epoch":now,
"charged_seconds":launch_preflight,"cpu_only":True,"GPU_calls":0,"basis":"conservative fixed pre-start allowance"})
total=ledger["v51_charged_seconds"]+sum(x["charged_seconds"] for x in ledger["segments"])
ledger["v55_pre_run_shared_elapsed_seconds"]=total
wr("ACTIVE_JOB_LEDGER.json",ledger)
prep=rd("PREPARATION.json")
prep["ledger_reader_failure"]="CPU-only failure before model load; v55 ledger helper updated to lazily read start_epoch only for active records"
prep["pre_formal_runner_failure_seconds_charged"]=13.841
prep["v55_budget_helper_change"]="ledger-only conditional evaluation; no quantization/numeric/search changes"
wr("PREPARATION.json",prep)
ext=rd("BUDGET_EXTENSION.json")
ext["new_cpu_preparation_charges_seconds"]=total-5980.039842538834
ext["current_shared_elapsed_seconds"]=total
ext["current_shared_remaining_seconds"]=12600-total
ext["ledger_helper_patch_charge_seconds"]=patch_seconds
ext["launcher_allowance_seconds"]=launch_preflight
ext["updated_headroom_seconds"]=12600-total-ext["updated_total_remaining_projection_seconds"]
wr("BUDGET_EXTENSION.json",ext)
pre=rd("V55_COST_PRECHECK.json")
pre["current_shared_elapsed_seconds"]=total
pre["current_shared_remaining_seconds"]=12600-total
pre["projected_headroom_seconds"]=12600-total-pre["projected_remaining_seconds_including_all_reserves"]
pre["CPU_only_runner_startup_failure_seconds_charged"]=13.841
pre["ledger_compatibility_patch"]="conditional lazy evaluation only; numerical/search path unchanged"
pre["go"]=pre["projected_headroom_seconds"]>=0
wr("V55_COST_PRECHECK.json",pre)
lock=rd("FROZEN_RUN_LOCK.json")
lock["files"]["budget.py"]=sha(O/"budget.py")
lock["files"]["PREPARATION.json"]=sha(O/"PREPARATION.json")
lock["files"]["BUDGET_EXTENSION.json"]=sha(O/"BUDGET_EXTENSION.json")
lock["files"]["V55_COST_PRECHECK.json"]=sha(O/"V55_COST_PRECHECK.json")
wr("FROZEN_RUN_LOCK.json",lock)
# Preserve the actual failed attempt as a separate CPU-only event.
failed={"time_unix":time.time(),"elapsed_shared_seconds":total,"error":"KeyError('start_epoch') in budget.seconds before budget.start completed",
"cpu_only":True,"GPU_compute":False,"model_loads":0,"last_committed_source":"v54 layer1/down, no v55 state modified"}
with (O/"FAILED_ATTEMPTS.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(failed,allow_nan=False)+"\n")
assert pre["go"] and all(sha(O/n)==h for n,h in lock["files"].items())
print(json.dumps({"patched":"PASS","budget_total":total,"remaining":12600-total,"forecast":pre["projected_remaining_seconds_including_all_reserves"],
"headroom":pre["projected_headroom_seconds"],"cpu_charged":total-5980.039842538834,"budget_py_sha256":sha(O/"budget.py")},ensure_ascii=False))
