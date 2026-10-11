import csv,hashlib,json,time,subprocess,sys
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=OUT.parents[3];V18=BASE/'range_w4_fullmodel_v18';V55=BASE/'v55_m2_decoder_fixed_integer_repair';V53=BASE/'v53_decoder_interface_propagation_audit';V48=BASE/'v48_fast_numerical_acceptance';V46=BASE/'v46_m1_single_integer_repair_fullmodel'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf8'))
def writej(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def readcsv(p):
 with Path(p).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def main():
 t=time.perf_counter();sys.path.insert(0,str(OUT));import budget
 prep=readj(OUT/'PREPARATION.json');src=readj(OUT/'SOURCE_AUDIT.json');init=readj(OUT/'INITIALIZATION_PREFLIGHT.json')
 assert prep['status']=='CPU_SOURCE_AUDIT_PASS' and src['pass'] and init['pass']
 # Cost-only historical v46 block timing proxy. No v46 checkpoint or repair-group parameters are opened.
 v46rows=readcsv(V46/'BLOCK_TIMINGS.csv');v46sum=sum(float(x['seconds']) for x in v46rows);assert len(v46rows)==24
 v55native=readcsv(V55/'PROPAGATION_NATIVE_TIMINGS.csv');native_sum=sum(float(x['seconds']) for x in v55native);assert len(v55native)==24
 continuation=readcsv(V55/'STAGE_TIMINGS_EVAL_CONTINUATION.csv');ppl_seconds=sum(float(x['seconds']) for x in continuation if x.get('stage')=='PPL_evaluation')
 if ppl_seconds<=0:
  q=readj(V55/'QUALITY_EVALUATION_SOURCE.json');ppl_seconds=sum(float(x.get('seconds',0.0)) for x in q.get('timings',[]))
 if ppl_seconds<=0:ppl_seconds=488.1707661151886
 v55cost=readcsv(V55/'STAGE_COST_SUMMARY.csv');
 deployment_cost=sum(float(x['measured_seconds']) for x in v55cost if x.get('mode')=='deployment' and x.get('stage') in {'final_save','pack_reload_all_grid_check','reload'})
 model_setup=sum(float(x['measured_seconds']) for x in v55cost if x.get('mode')=='formal' and x.get('stage') in {'formal_model_load','initial_embedding_capture','unpack_pack_zero_commit','zero_model_output'})
 # Charge this preflight before computing the standalone cap gate.
 budget.charge_cpu('v56 CPU-only cost projection and source freeze preflight',time.perf_counter()-t,modules=168,rows=304128)
 prior,local=budget.seconds()
 projected_repair=v46sum*1.15
 projected_propagation=native_sum*1.15
 projected_evaluation=ppl_seconds*1.15
 projected_setup=max(30.0,model_setup)
 projected_save_reload=max(20.0,deployment_cost*1.20)
 uncertainty=300.0;reserve=360.0
 projected_additional=projected_repair+projected_propagation+projected_evaluation+projected_setup+projected_save_reload+uncertainty+reserve
 projected_total=prior+projected_additional;go=projected_total<=9000.0
 pre={'version':'v56','go':go,'cap_seconds':9000.0,'preflight_charged_seconds':prior,'remaining_before_gpu_seconds':9000.0-prior,'historical_M1_v46_24block_timing_proxy_seconds':v46sum,'historical_M1_v46_propagation_validity':'UNKNOWN; timing used only as conservative matrix-shape cost proxy; no group/checkpoint parameters read','v55_correct_propagation_24block_validation_seconds':native_sum,'v55_fresh_two_dataset_evaluation_seconds':ppl_seconds,'v55_measured_save_reload_seconds':deployment_cost,'v55_measured_setup_seconds':model_setup,'projection':{'repair_with_15pct_margin':projected_repair,'three_article_propagation_with_15pct_margin':projected_propagation,'two_frozen_evaluations_with_15pct_margin':projected_evaluation,'model_load_zero_model_embed':projected_setup,'save_reload_and_grid_checks':projected_save_reload,'uncertainty_reserve':uncertainty,'finish_reserve':reserve},'projected_additional_seconds':projected_additional,'projected_total_including_precheck':projected_total,'projected_remaining_after_completion':9000.0-projected_total,'v55_historical_shared_ledger_seconds_separate':11037.312983,'prior_v55_cost_not_in_v56_budget':True,'M1_build_seconds_historical_separate':float(readj(V18/'FORMAL_RUNTIME.json')['m1']['wall_seconds_including_load_replay_save']),'M2_build_seconds_historical_separate':float(readj(V18/'FORMAL_RUNTIME.json')['m2']['wall_seconds_including_load_replay_save']),'prediction_not_measurement':True,'GPU_calls_preflight':0}
 writej(OUT/'V56_COST_PRECHECK.json',pre)
 # Freeze runner and source identities only after all adaptation and prechecks are complete.
 corefiles=['PROTOCOL.json','run_v56.py','runtime_v54.py','decoder_interface.py','fast_accept_v52.py','initialization_review.py','repair_control.py','precision_check.py','budget.py','audit_sources.py','SOURCE_AUDIT.json','SOURCE_MODULES.csv','INITIALIZATION_GRID_COMPARISON.csv','INITIALIZATION_PREFLIGHT.json','NUMERICAL_VALIDATION.json','CPU_TESTS.json','PREPARATION.json','SOURCE_REUSE_AND_DIFF.json','adapt_runner.py','prepare_v56.py','freeze_preflight.py','static_binding_check.py','STATIC_BINDING_PREFLIGHT.json','HISTORICAL_COST_SEPARATE.json','V55_ARM_SOURCE_AUDIT.json','V56_COST_PRECHECK.json','REPRODUCE.md']
 files={f:sha(OUT/f) for f in corefiles}
 v53lock=readj(V55/'FROZEN_RUN_LOCK.json')
 lock={'pass':True,'version':'v56','files':files,'v53_runtime_sha256':sha(V53/'runtime_v53.py'),'v53_validation_result_sha256':sha(V53/'GPU_VALIDATION_RESULT.json'),'v48_fast_accept_sha256':sha(V48/'fast_accept.py'),'v48_acceptance_rules_sha256':sha(V48/'ACCEPTANCE_RULES.md'),'frozen_search_helper_sha256':sha(OUT/'repair_control.py'),'frozen_precision_check_sha256':sha(OUT/'precision_check.py'),'search_helper_is_code_only_no_parameters':True,'precision_check_is_code_only_no_parameters':True,'v55_protocol_sha256':sha(V55/'PROTOCOL.json'),'v55_runtime_source_sha256':sha(V55/'runtime_v54.py'),'M1_checkpoint_sha256':sha(V18/'M1/checkpoint/final/model.safetensors'),'M1_checkpoint_expected_sha256':src['M1_model_sha256'],'M2_checkpoint_sha256':sha(V18/'M2/checkpoint/final/model.safetensors'),'calibration_manifest_sha256':src['calibration_manifest_sha256'],'no_v46_repair_group_files_read':True,'new_full_run_starts_at_layer0':True,'reused_v55_repair_groups':0,'algorithm_or_acceptance_rule_changed':False,'v56_cap_seconds':9000.0,'v55_shared_history_separate_seconds':11037.312983}
 writej(OUT/'FROZEN_RUN_LOCK.json',lock)
 # Recompile and inspect forbidden legacy checkpoint paths before authorizing the run.
 runner=OUT/'run_v56.py';subprocess.run([sys.executable,'-m','py_compile',str(runner)],check=True)
 text=runner.read_text(encoding='utf8')
 for forbidden in ["V46/'formal/groups'","V45/'formal/groups'","V52/'formal/groups'","M2/checkpoint/final')"]:assert forbidden not in text,forbidden
 assert "str(V18/'M1/checkpoint/final')" in text and "V55/'R1/checkpoint/final/model.safetensors'" in text
 assert go,'9000-second pre-run projection does not fit; do not launch'
 print('V56_FROZEN_PREFLIGHT_PASS',json.dumps(pre,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
