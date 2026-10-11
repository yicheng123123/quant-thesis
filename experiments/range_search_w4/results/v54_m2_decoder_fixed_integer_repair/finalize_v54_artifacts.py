from pathlib import Path
import csv, json, hashlib
O=Path(__file__).resolve().parent
ROOT=Path(r'E:\\quant-thesis\\experiments\\range_search_w4\\results')
V53=ROOT/'v53_decoder_interface_propagation_audit'
V46=ROOT/'v46_m1_single_integer_repair_fullmodel'
V45=ROOT/'v45_m2_single_integer_repair_fullmodel'
V49=ROOT/'v49_fullmodel_fast_integer_repair'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def readj(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def writej(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def writecsv(p,fields,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

# Book the logged first launcher failure into the shared cap ledger. It is CPU-only
# and remains separately labeled; no GPU call or model load occurred.
ledger_p=O/'ACTIVE_JOB_LEDGER.json'
ledger=readj(ledger_p)
failed=14.442328929901123
label='v54 first runner failure before inner handler (CPU process wall; zero model/GPU calls)'
if not any(s.get('label')==label for s in ledger.get('segments',[])):
    ledger['segments'].append({'version':'v54','label':label,'charged_seconds':failed,'cpu_only':True,'model_loads':0,'GPU_calls':0,'basis':'FAILED_ATTEMPTS.jsonl elapsed_since_process_start; STARTUP_DIAGNOSTICS.json confirms zero model loads and GPU calls','conservative_common_cap_charge':True})
ledger['v54_accounting_adjustment_note']='The first 14.44232893s CPU-only launcher failure is included in the shared cap total per v54 instruction; it is not GPU execution. The later 0.028126s startup segment and 288.14117384s formal segment are already recorded.'
writej(ledger_p,ledger)
segments=ledger['segments']; v51=float(ledger.get('v51_charged_seconds',0.0))
pre_v54=v51+sum(float(s.get('charged_seconds',0.0)) for s in segments if s.get('version')!='v54')
v54gpu=sum(float(s.get('charged_seconds',0.0)) for s in segments if s.get('version')=='v54' and not s.get('cpu_only'))
v54cpu=sum(float(s.get('charged_seconds',0.0)) for s in segments if s.get('version')=='v54' and s.get('cpu_only'))
total=pre_v54+v54gpu+v54cpu
cap=float(ledger.get('cap_seconds',10800.0))
account={'status':'STOPPED_D_BUDGET_GATE_AFTER_BLOCK1','cap_seconds':cap,'pre_v54_exact_ledger_seconds':pre_v54,'v54_gpu_or_active_runner_seconds':v54gpu,'v54_cpu_only_failure_wall_seconds_charged_conservatively':v54cpu,'v54_total_increment_including_cpu_failure':v54gpu+v54cpu,'common_cumulative_seconds':total,'common_remaining_seconds':cap-total,'user_rounded_start_seconds':5677.428,'user_rounded_remaining_before_v54_seconds':5122.572,'v54_startup_attempt_2_seconds_already_in_active_ledger':0.028126001358032227,'main_formal_runner_active_seconds':288.1411738395691,'first_failure_seconds_source':'FAILED_ATTEMPTS.jsonl elapsed_since_process_start; CPU-only, no model load/GPU call','process_startup_import_wall_before_main_budget_hook':'not separately measured; excluded from GPU active time','idle_or_UI_wait_seconds':0,'block1_gate_snapshot':readj(O/'ONLINE_BUDGET_STATUS.json'),'budget_note':'RUN_STATUS and ONLINE_BUDGET_STATUS retain raw in-run counters; this file additionally books the documented CPU-only failed launcher into the shared cap because v54 requested all failures be counted. CPU seconds are not represented as GPU compute.'}
writej(O/'FINAL_BUDGET_ACCOUNTING.json',account)

# Partial checkpoint lock: hashes and origin for the only two validated full blocks.
restore=readj(V53/'VALID_RESUME_BOUNDARY.json')
restore_rows=list(csv.DictReader((O/'PROPAGATION_RESTORE_CHECKS.csv').open(encoding='utf-8')))
cp_entries=[]
for li in (0,1):
    for group in ('qkv','o','up_gate','down'):
        p=O/'formal'/'groups'/f'layer_{li:02d}_{group}.pt'
        rec=next((r for r in restore_rows if r.get('layer')==str(li) and r.get('group')==group and r.get('checkpoint_sha256')),None)
        source=None
        if li==0:
            source=next(x for x in restore['block0_groups_to_restore_without_repair'] if x['group']==group)
        cp_entries.append({'layer':li,'group':group,'path':str(p),'sha256':sha(p),'committed':p.exists(),'restored_without_repair':(li==0),'source_v51_sha256':source['sha256'] if source else None,'input_sha256':(rec or {}).get('input_sha256'),'C_sha256':(rec or {}).get('C_sha256'),'code_grid_terminal_checks_pass':True})
block0=readj(O/'BLOCK0_GATE.json')
commit=readj(O/'formal'/'COMMIT_STATUS.json')
prefix=readj(O/'formal'/'PREFIX_STATUS_layer_01.json')
prop=[r for r in csv.DictReader((O/'PROPAGATION_NATIVE_RESULTS.csv').open(encoding='utf-8'))]
partial={'status':'VALID_GROUP_CHECKPOINTS_THROUGH_BLOCK1_ONLY','complete_model':False,'deployable':False,'final_model_checkpoint_exists':False,'final_model_checkpoint_sha256':None,'full_model_lock_exists':False,'evaluation_authorized_or_run':False,'initialization':{'M2_checkpoint_sha256':'810d6329cdc51d995543dbb701e2368568b19459a44f340df1a8a0019d447eed','revision':'7ae557604adf67be50417f59c2c2f167def9a775'},'reused_boundary':{'source':'v53 VALID_RESUME_BOUNDARY.json','only_v51_block0_groups':True,'block0_independent_128_article_input_validation_pass':block0['pass'],'layer1_qkv_input_sha256':'70cd41104db3769a3d434c27d4612a41becf4be98439c6328c6010e5b6eae7ac','true_boundary_hidden_sha256':'9488e56e397608a5d19291595f43645135ad6b6ed477cd6d97eeba7398bfeed6','v52_layer1plus_state_used':False},'last_committed':commit,'last_complete_prefix':prefix,'group_checkpoints':cp_entries,'validated_blocks':[0,1],'native_propagation_comparisons':{'articles':[0,64,127],'tokens_each':2048,'comparisons':len(prop),'all_pass':all(r.get('pass')=='True' for r in prop),'all_bitwise_equal':all(r.get('bitwise_equal')=='True' for r in prop),'max_abs_error':max(float(r['max_abs_error']) for r in prop)},'next_restart_boundary':'If a later run is authorized, use this exact M2 + groups through block1 state and v54 corrected runtime; start at layer2/qkv. Do not import any v52 layer1+ state. Revalidate these group hashes/input boundary before continuing.'}
writej(O/'PARTIAL_CHECKPOINT_LOCK.json',partial)

# Full historical measurement provenance rows for context only. v45/v46/v49
# measured actual saved checkpoints; v53 labels their calibration propagation UNKNOWN.
prov=list(csv.DictReader((V53/'HISTORICAL_PPL_PROVENANCE.csv').open(encoding='utf-8')))
wanted={'M1','M2','v45_original_R2','v46_R1','v49_fast_R1'}
rows=[]
for r in prov:
    if r['model'] not in wanted: continue
    if r['model'] in {'v45_original_R2','v46_R1','v49_fast_R1'}:
        prop_status='UNKNOWN historical decoder return type and calibration propagation (v53 audit)'
    else:
        prop_status='v18 recorded Transformers 4.49 tuple API supported; per-call historical type/unexhaustive prefix validation remains limited'
    rows.append({**r,'calibration_propagation_status':prop_status,'usage_note':'historical checkpoint measurement only; does not stand in for v54 evaluation'})
fields=list(rows[0].keys())
writecsv(O/'HISTORICAL_QUALITY_BACKGROUND.csv',fields,rows)

# v54 intentionally has no new quality result. Keep historical numbers alongside
# blanks for the new model and explicitly encode the D status.
base={(r['model'],r['dataset']):r for r in rows}
out=[]
for ds,key in [('wikitext','wikitext'),('c4_fixed_budget','c4_fixed_budget')]:
    m1=base[('M1',key)];m2=base[('M2',key)]
    out.append({'dataset':ds,'v54_status':'D_not_evaluated_budget_stop','v54_checkpoint_sha256':'','v54_ppl':'','v54_valid_token_weighted_nll':'','v54_valid_tokens':'','v54_windows':'','M2_checkpoint_sha256':m2['checkpoint_sha256'],'M2_historical_ppl':m2['PPL'],'M2_historical_nll':m2['NLL'],'M1_checkpoint_sha256':m1['checkpoint_sha256'],'M1_historical_ppl':m1['PPL'],'M1_historical_nll':m1['NLL'],'v54_delta_vs_M2_ppl':'','v54_delta_vs_M2_nll':'','v54_delta_vs_M1_ppl':'','v54_delta_vs_M1_nll':'','p3_read':False})
writecsv(O/'QUALITY_COMPARISON.csv',list(out[0].keys()),out)

# Cost table separates measured active GPU job from the CPU-only failure.
block=list(csv.DictReader((O/'BLOCK_TIMINGS.csv').open(encoding='utf-8')))
layer1=next(r for r in block if r['layer']=='1')
prop_tim=list(csv.DictReader((O/'PROPAGATION_NATIVE_TIMINGS.csv').open(encoding='utf-8')))
prop1=next(r for r in prop_tim if r['block_completed']=='1')
mods=list(csv.DictReader((O/'MODULE_RESULTS.csv').open(encoding='utf-8')))
layer1mods=[r for r in mods if r['mode']=='formal' and r['layer']=='1']
stages=list(csv.DictReader((O/'STAGE_TIMINGS.csv').open(encoding='utf-8')))
def stage_sum(s):return sum(float(r['seconds']) for r in stages if r.get('mode')=='formal' and r.get('layer')=='1' and r.get('stage')==s)
costrows=[
 {'scope':'shared','stage':'inherited_before_v54','seconds':f'{pre_v54:.9f}','measurement':'ledger sum; includes v51/v52/v53','gpu_active':'historical mixed lineage'},
 {'scope':'v54','stage':'first_runner_failure_cpu_wall','seconds':f'{failed:.9f}','measurement':'measured process wall from FAILED_ATTEMPTS','gpu_active':'no; conservatively charged to shared cap'},
 {'scope':'v54','stage':'second_runner_failure','seconds':f'{0.028126001358032227:.9f}','measurement':'active ledger segment','gpu_active':'no model load; included in active ledger'},
 {'scope':'v54','stage':'main_formal_runner_active','seconds':f'{288.1411738395691:.9f}','measurement':'closed active ledger segment','gpu_active':'includes model load, restore, block1, propagation, stop'},
 {'scope':'v54','stage':'block0_restore_and_replay','seconds':block[0]['seconds'],'measurement':'BLOCK_TIMINGS.csv','gpu_active':'synchronized block wall; no repair'},
 {'scope':'v54','stage':'layer1_full_block_wall','seconds':layer1['seconds'],'measurement':'BLOCK_TIMINGS.csv','gpu_active':'synchronized complete matrix block wall'},
 {'scope':'v54','stage':'layer1_three_article_native_full_forward','seconds':prop1['seconds'],'measurement':'PROPAGATION_NATIVE_TIMINGS.csv','gpu_active':'included in block1 wall; not additive'},
 {'scope':'v54','stage':'layer1_C_construction_timer_sum','seconds':f'{stage_sum("C_construction"):.9f}','measurement':'STAGE_TIMINGS.csv','gpu_active':'nested per-matrix timer sum; not additive'},
 {'scope':'v54','stage':'layer1_independent_terminal_FP64_timer_sum','seconds':f'{stage_sum("independent_full_terminal_FP64"):.9f}','measurement':'STAGE_TIMINGS.csv','gpu_active':'nested per-matrix timer sum; not additive'},
 {'scope':'v54','stage':'layer1_candidate_scoring_timer_sum','seconds':f'{stage_sum("fast_candidate_scoring"):.9f}','measurement':'STAGE_TIMINGS.csv','gpu_active':'nested timer sum; not additive'},
 {'scope':'v54','stage':'save_reload_full_checkpoint_and_PPL','seconds':'','measurement':'NOT_RUN','gpu_active':'no complete model'},
 {'scope':'v54','stage':'historical_M2_construction','seconds':'2115.935','measurement':'historical reported measurement','gpu_active':'reported separately; not v54'}]
writecsv(O/'COST_LEDGER_V54.csv',list(costrows[0].keys()),costrows)

# Final artifact hashes for evidence integrity. Manifest does not hash itself.
keyfiles=['REPORT.md','REPRODUCE.md','finalize_v54_artifacts.py','PROTOCOL.json','FROZEN_RUN_LOCK.json','SOURCE_AUDIT.json','RUNTIME_BINDING.json','BLOCK0_GATE.json','PROPAGATION_RESTORE_CHECKS.csv','PROPAGATION_NATIVE_RESULTS.csv','PROPAGATION_NATIVE_TIMINGS.csv','MODULE_RESULTS.csv','BLOCK_TIMINGS.csv','MATRIX_TIMINGS.csv','STAGE_TIMINGS.csv','RUN_STATUS.json','ONLINE_BUDGET_STATUS.json','ACTIVE_JOB_LEDGER.json','FINAL_BUDGET_ACCOUNTING.json','PARTIAL_CHECKPOINT_LOCK.json','QUALITY_COMPARISON.csv','HISTORICAL_QUALITY_BACKGROUND.csv','COST_LEDGER_V54.csv','CPU_TESTS.json','NUMERICAL_VALIDATION.json']
manifest={'status':'D_INCOMPLETE','artifacts':{x:sha(O/x) for x in keyfiles if (O/x).is_file()},'partial_group_checkpoints':{x['path']:x['sha256'] for x in cp_entries},'historical_sources':{'v53_HISTORICAL_IMPACT.csv':sha(V53/'HISTORICAL_IMPACT.csv'),'v53_HISTORICAL_PPL_PROVENANCE.csv':sha(V53/'HISTORICAL_PPL_PROVENANCE.csv'),'v46_BASELINE_AUDIT.json':sha(V46/'BASELINE_AUDIT.json'),'v45_PPL_COMPARISON.csv':sha(V45/'reports'/'PPL_COMPARISON.csv')},'historical_files_modified':False,'full_model_checkpoint_lock':False,'new_PPL_windows':False}
writej(O/'FINAL_ARTIFACT_MANIFEST.json',manifest)
print(json.dumps({'status':'finalized','common_cumulative_seconds':total,'remaining_seconds':cap-total,'v54_gpu_active_seconds':v54gpu,'v54_cpu_failure_seconds':v54cpu,'partial_group_count':len(cp_entries),'history_rows':len(rows),'layer1_modules':len(layer1mods),'history_sum_warning':'stage timers are nested and non-additive'},indent=2))
