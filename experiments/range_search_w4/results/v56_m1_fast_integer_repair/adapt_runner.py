from pathlib import Path
import hashlib, json, shutil, time, textwrap
OUT=Path(__file__).resolve().parent; BASE=OUT.parent; V55=BASE/'v55_m2_decoder_fixed_integer_repair'; V46_HELPER=BASE/'v46_m1_single_integer_repair_fullmodel'/'repair_v45.py'; V46_PRECISION=BASE/'v46_m1_single_integer_repair_fullmodel'/'precision_check.py'; V53=BASE/'v53_decoder_interface_propagation_audit'; V48=BASE/'v48_fast_numerical_acceptance'; V18=BASE/'range_w4_fullmodel_v18'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def replace_once(s,a,b,label):
 n=s.count(a)
 if n!=1: raise RuntimeError(f'{label}: expected exactly one source match, found {n}')
 return s.replace(a,b,1)
def main():
 t0=time.perf_counter()
 # Adapt only source/init/output identity, historical-input assertions, and budget/finalization plumbing.
 src=(V55/'run_v55.py').read_text(encoding='utf8')
 shutil.copyfile(V46_HELPER,OUT/'repair_control.py')
 shutil.copyfile(V46_PRECISION,OUT/'precision_check.py')
 src=src.replace('"""v49 full M1 repair. Exact fast_accept is bound explicitly before rt.run_layer."""','"""v56 M1 full repair with the v55 frozen fast integer repair implementation."""',1)
 line="ROOT=Path(r'E:\\quant-thesis');BASE=ROOT/'experiments/range_search_w4/results';OUT=BASE/'v55_m2_decoder_fixed_integer_repair';V52=BASE/'v52_m2_fast_integer_repair_fp64_initial_review';V53=BASE/'v53_decoder_interface_propagation_audit';V46=BASE/'v46_m1_single_integer_repair_fullmodel';V45=BASE/'v45_m2_single_integer_repair_fullmodel';V49=BASE/'v49_fullmodel_fast_integer_repair';V48=BASE/'v48_fast_numerical_acceptance';V18=BASE/'range_w4_fullmodel_v18'"
 repl="ROOT=Path(r'E:\\quant-thesis');BASE=ROOT/'experiments/range_search_w4/results';OUT=BASE/'v56_m1_fast_integer_repair';V53=BASE/'v53_decoder_interface_propagation_audit';V48=BASE/'v48_fast_numerical_acceptance';V41=BASE/'v41_integer_pair_repair_pilot';V18=BASE/'range_w4_fullmodel_v18';V55=BASE/'v55_m2_decoder_fixed_integer_repair'"
 src=replace_once(src,line,repl,'top-level paths')
 src=replace_once(src,'CAP=12600;FINISH_RESERVE=360;UNCERTAINTY_RESERVE=120;GROUP_ORDER=', 'CAP=9000;FINISH_RESERVE=360;UNCERTAINTY_RESERVE=240;GROUP_ORDER=','budget constants')
 src=src.replace('v55 12600-second cumulative shared budget exhausted','v56 standalone 9000-second budget exhausted')
 src=replace_once(src,'sys.path.insert(0,str(ROOT));sys.path.insert(0,str(V46));sys.path.insert(0,str(V48));sys.path.insert(0,str(OUT))','sys.path.insert(0,str(ROOT));sys.path.insert(0,str(V48));sys.path.insert(0,str(OUT))','sys.path')
 src=replace_once(src,"old=loadmod('repair_control',V46/'repair_v45.py')\n",'','remove historical repair import')
 src=replace_once(src,"fast=loadmod('v52_fast_accept',OUT/'fast_accept_v52.py')","repair_control=loadmod('repair_control',OUT/'repair_control.py')\nfast=loadmod('v52_fast_accept',OUT/'fast_accept_v52.py')",'load frozen pure search helper from v56 local copy')
 src=src.replace("RUN_ID='v55-'", "RUN_ID='v56-'")
 src=replace_once(src,"V53_GROUP_HASHES={x['group']:x['native_reference_input_sha256'] for x in read(V53/'BLOCK0_GROUP_SOURCE_CHECKS.json')['groups']}\nV53_BOUNDARY=read(V53/'VALID_RESUME_BOUNDARY.json')['expected_boundary']\n",'','do not reuse M2 prefix hashes')
 src=replace_once(src,"  if li==0:assert xh==V53_GROUP_HASHES[group],f'v53 approved block0 input mismatch: {group}'\n  if li==1 and group=='qkv':assert xh==V53_BOUNDARY['layer1_qkv_X_raw_content_sha256'], 'corrected full-token layer1/qkv boundary mismatch'\n",'','do not enforce different initializer hashes')
 src=src.replace("v53_block0_source_match=(li==0),v53_layer1_boundary_match=(li==1 and group=='qkv')", "initializer='M1',source_is_current_run=True")
 src=src.replace('initial M2 grid source mismatch','initial M1 grid source mismatch')
 src=src.replace("'v51 block0 restored after v53 gate; not repaired again in v54'", "'fresh M1 block0 repaired exactly once by v56; no historical group reused'")
 src=src.replace("'v53 M2 approved block0 inputs'", "'v56 M1 current-run inputs'")
 src=src.replace('v42 hard budget/reserve exhausted','v56 hard budget/reserve exhausted')
 # Evaluation manifest audit uses v18 protocol directly, never a v46 baseline file.
 a=src.index('def in_memory_eval_protocol():'); b=src.index('def group_restore_audit(',a)
 func="""def in_memory_eval_protocol():
  global EVAL_KEY_MAP
  protocol=json.loads((V18/'PROTOCOL.json').read_text(encoding='utf8'))
  names={'wikitext':'WIKITEXT_TEST_MANIFEST.json','c4_fixed_budget':'C4_EVAL_MANIFEST.json'};hashes={}
  for ds,name in names.items():
   p=V18/'evaluation/manifests'/name;key='v16_wikitext_manifest_sha256' if ds=='wikitext' else 'v16_c4_manifest_sha256'
   expected=protocol['source_protocol'][key];assert sha(p)==expected
   data=json.loads(p.read_text(encoding='utf8'));assert len(data['token_ids']) in (299078,131072)
   hashes[ds]=expected
  rt.P.setdefault('v18',{})['source_protocol']={'v16_wikitext_manifest_sha256':hashes['wikitext'],'v16_c4_manifest_sha256':hashes['c4_fixed_budget']}
  EVAL_KEY_MAP=hashes
  return hashes

"""
 src=src[:a]+func+src[b:]
 src=src.replace("read('BLOCK0_INHERITANCE.json')['pass']", "read('INITIALIZATION_PREFLIGHT.json')['pass']")
 src=src.replace("read('V55_COST_PRECHECK.json')['go']", "read('V56_COST_PRECHECK.json')['go']")
 src=src.replace("f'frozen v55 source changed: {n}'", "f'frozen v56 source changed: {n}'")
 src=src.replace("budget.start('v55 complete repair and deployment/evaluation')", "budget.start('v56 full M1 repair, propagation, save/reload and evaluation')")
 src=src.replace("'v55 complete repair and deployment/evaluation'", "'v56 full M1 repair, propagation, save/reload and evaluation'")
 src=src.replace("inherited_v51_block0_without_repair=True", "fresh_M1_from_layer0=True,reused_groups=0,reused_M2_M1_repair_state=False")
 src=src.replace("prior_shared_charged_seconds=budget.seconds()[0]", "preflight_charged_seconds=budget.seconds()[0],historical_v55_shared_seconds_separate=11037.312983")
 src=src.replace("v46_evaluator_key_prebound_to_audited_v18_manifests=True", "evaluator_keys_bound_directly_to_v18_audited_manifests=True")
 src=src.replace("'missing_v46_P_v18_key_supplied_in_memory':True", "'v18_manifests_checked_directly':True")
 src=src.replace("str(V18/'M2/checkpoint/final')", "str(V18/'M1/checkpoint/final')")
 src=src.replace("source_m2=V18/'M2/checkpoint/final/model.safetensors';assert sha(source_m2)==read('PROTOCOL.json')['initialization']['model_sha256']", "source_m1=V18/'M1/checkpoint/final/model.safetensors';assert sha(source_m1)==read('PROTOCOL.json')['initialization']['model_sha256']")
 src=src.replace("later blocks remain original M2", "later blocks remain original M1")
 # Remove cross-run group comparisons against repaired v46 checkpoints.
 a=src.index('    if key not in VERIFIED_GROUPS:')
 b=src.index("   PROPAGATION.append({'layer':li,'boundary_after_down':hid",a)
 newblock = "    if key not in VERIFIED_GROUPS:\n     PROPAGATION.append({'run_id':RUN_ID,'layer':li,'group':g,'input_sha256':cp['input_sha256'],'C_sha256':cp['C_sha256'],'source':'v56 current M1 repair run','restored_without_repair':False,'current_run_group_completed':True,'checkpoint_sha256':sha(p)})\n     VERIFIED_GROUPS.add(key)\n"
 src=src[:a]+newblock+src[b:]
 # Remove the unused legacy timing helper; the runner never opens v46 data.
 a=src.index('def v46_measurement():')
 b=src.index('def in_memory_eval_protocol():',a)
 src=src[:a]+src[b:]
 # Online budget forecast is based on this run's measured blocks and propagation, with explicit uncertainty and 360-second close reserve.
 a=src.index("   timing_rows=list(csv.DictReader((V52/'BLOCK_TIMINGS.csv').open(encoding='utf8')))" )
 b=src.index("   if not gate:",a)
 budget_code="""   measured_blocks=list(csv.DictReader((OUT/'BLOCK_TIMINGS.csv').open(encoding='utf8')))
   done_block_walls=[float(x['seconds']) for x in measured_blocks if int(x['layer'])<=li]
   measured_native=[float(x['seconds']) for x in csv.DictReader((OUT/'PROPAGATION_NATIVE_TIMINGS.csv').open(encoding='utf8'))]
   mean_repair=sum(done_block_walls)/max(1,len(done_block_walls))
   mean_native=sum(measured_native)/max(1,len(measured_native))
   remaining_layers=list(range(li+1,24))
   remaining_repair=len(remaining_layers)*mean_repair*1.15
   remaining_propagation=len(remaining_layers)*mean_native*1.15
   uncertainty=max(180.0,0.10*(remaining_repair+remaining_propagation))
   predicted=remaining_repair+remaining_propagation+FINISH_RESERVE+uncertainty
   gate=elapsed()+predicted<=CAP
   dump('ONLINE_BUDGET_STATUS.json',{'after_completed_block':li,'elapsed_v56_seconds':elapsed(),'remaining_v56_seconds':CAP-elapsed(),'measured_mean_block_repair_wall_seconds':mean_repair,'remaining_repair_projection_with_15pct_margin_seconds':remaining_repair,'measured_mean_three_article_propagation_seconds':mean_native,'remaining_propagation_projection_with_15pct_margin_seconds':remaining_propagation,'uncertainty_reserve_seconds':uncertainty,'finish_and_evaluation_reserve_seconds':FINISH_RESERVE,'predicted_required_seconds':predicted,'go':gate,'projection_basis':'current M1 v56 completed blocks and current native-forward checks; no v46 repair parameters or prefix cache used'})
"""
 src=src[:a]+budget_code+src[b:]
 src=src.replace("'stop_reason':'remaining measured-history projection plus propagation and frozen 360s finish reserve exceeds shared budget'", "'stop_reason':'remaining v56 measured projection plus propagation, uncertainty and 360s finish reserve exceeds standalone cap'")
 # Include AutoConfig for normalized config comparison, avoiding serializer-specific raw JSON mismatch.
 src=src.replace('from transformers import AutoModelForCausalLM', 'from transformers import AutoConfig, AutoModelForCausalLM')
 # Replace old R2-only deployment comparison with all four arm tensor/config checks.
 a=src.index("  mcfg=V18/'M2/checkpoint/final/config.json'")
 b=src.index("  copies=[]",a)
 final="""   modelpath=OUT/'M1_REPAIRED/checkpoint/final/model.safetensors';outcfg=OUT/'M1_REPAIRED/checkpoint/final/config.json'
   refs={'M1':V18/'M1/checkpoint/final/model.safetensors','M2':V18/'M2/checkpoint/final/model.safetensors','v55_M2_plus_repair':V55/'R1/checkpoint/final/model.safetensors'}
   def normalized_config(path):
    obj=AutoConfig.from_pretrained(str(path.parent),local_files_only=True).to_dict()
    return {k:v for k,v in obj.items() if k not in {'_name_or_path','transformers_version','_commit_hash'}}
   own_cfg=normalized_config(outcfg);cfgrows=[]
   for label,path in [('M1',refs['M1'].parent/'config.json'),('M2',refs['M2'].parent/'config.json'),('v55_M2_plus_repair',refs['v55_M2_plus_repair'].parent/'config.json')]:
    cfg=normalized_config(path);cfgrows.append({'arm':label,'effective_config_equal':cfg==own_cfg,'effective_config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
   config_equal_all=all(x['effective_config_equal'] for x in cfgrows)
   rt.dump('CONFIG_COMPARISON.json',{'loader':'same local AutoConfig loader for all four arms','ignored_serialization_keys':['_name_or_path','transformers_version','_commit_hash'],'all_effective_configs_equal':config_equal_all,'comparisons':cfgrows})
   tensor_rows=[];diff_counts={};maxdiff={};comp_start=time.perf_counter();tensor_key_count=0
   with safe_open(str(modelpath),framework='pt',device='cpu') as ours:
    keys=set(ours.keys());tensor_key_count=len(keys)
    for label,path in refs.items():
     count=0;md=0.0
     with safe_open(str(path),framework='pt',device='cpu') as ref:
      assert keys==set(ref.keys()),f'named tensor key mismatch vs {label}'
      for key in sorted(keys):
       a=ours.get_tensor(key);b=ref.get_tensor(key);ha=hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();hb=hashlib.sha256(b.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();equal=torch.equal(a.view(torch.uint8),b.view(torch.uint8));delta=float((a.float()-b.float()).abs().max()) if a.numel() else 0.0
       if not equal:count+=1;md=max(md,delta)
       tensor_rows.append({'reference_arm':label,'tensor':key,'v56_sha256':ha,'reference_sha256':hb,'equal_bitwise':equal,'max_abs_difference':delta})
     diff_counts[label]=count;maxdiff[label]=md
   rt.csvwrite('NAMED_TENSOR_HASHES.csv',[{'tensor':key,'sha256':hashlib.sha256(safe_open(str(modelpath),framework='pt',device='cpu').get_tensor(key).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()} for key in sorted(keys)])
   for label in refs:rt.csvwrite(f'reports/TENSOR_DIFFERENCES_VS_{label.replace("+","_")}.csv',[x for x in tensor_rows if x['reference_arm']==label])
   rt.TIMINGS.append({'mode':'deployment','layer':-1,'module':'all_model_tensors','stage':'named_tensor_comparison_vs_M1_M2_v55','seconds':time.perf_counter()-comp_start,'elapsed_seconds':elapsed(),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'type':'measured CPU wall'})
   quant_rows=[]
   for li in range(24):
    source=ORIG.get(li) or torch.load(OUT/f'original_codes/layer_{li:02d}.pt',map_location='cpu',weights_only=True)
    for g in GROUP_ORDER:
     cp=torch.load(OUT/'formal/groups'/f'layer_{li:02d}_{g}.pt',map_location='cpu',weights_only=False);assert_group(cp,li,g,source)
     for name,res in cp['modules'].items():
      quant_rows.append({'layer':li,'group':g,'module':name,'M1_grid_scales_equal':torch.equal(res['scales'],source[name]['scales']),'M1_grid_zeros_equal':torch.equal(res['zeros'],source[name]['zeros']),'M1_grid_g_idx_equal':torch.equal(res['g_idx'],source[name]['g_idx']),'codes_differ_from_M1':not torch.equal(res['codes'],source[name]['codes']),'input_sha256':cp['input_sha256'],'C_sha256_source_only':cp['C_sha256']})
   rt.csvwrite('reports/MODULE_GRID_AUDIT.csv',quant_rows)
   equal_v55=(diff_counts['v55_M2_plus_repair']==0 and config_equal_all)
   group_locks=[{'path':str(p.relative_to(OUT)),'sha256':sha(p)} for p in sorted((OUT/'formal/groups').glob('*.pt'))]
   assert len(group_locks)==96 and len(quant_rows)==168
   lock={'final_model_sha256':sha(modelpath),'config_sha256':sha(outcfg),'protocol_sha256':sha(OUT/'PROTOCOL.json'),'group_checkpoints':group_locks,'linear_modules':len(quant_rows),'rows':304128,'named_tensor_count':tensor_key_count,'named_model_tensor_differences_vs_M1':diff_counts['M1'],'named_model_tensor_differences_vs_M2':diff_counts['M2'],'named_model_tensor_differences_vs_v55':diff_counts['v55_M2_plus_repair'],'effective_inference_configs_equal_all_arms':config_equal_all,'equivalent_to_v55':equal_v55,'locked_before_evaluation_unix':time.time()}
   dump('CHECKPOINT_LOCK.json',lock)
   deployment={'pass':True,'all_168_grid_pack_reload_and_sample_output_checks':True,'model_sha256':sha(modelpath),'config_sha256':sha(outcfg),'tensor_keys_compared':tensor_key_count,'tensor_differences_by_arm':diff_counts,'max_abs_diff_by_arm':maxdiff,'effective_configs_equal_all_arms':config_equal_all,'equivalent_to_v55_M2_plus_repair':equal_v55,'source_checkpoints':{k:sha(v) for k,v in refs.items()},'completed_unix':time.time()};dump('DEPLOYMENT_COMPARISON.json',deployment)
"""
 final=textwrap.dedent(final)
 final="\n".join(("  "+line if line else "") for line in final.splitlines())+"\n"
 src=src[:a]+final+src[b:]
 src=src.replace("dstres=OUT/f'evaluation/v55_{ds}.json';dstwin=OUT/f'evaluation/v55_{ds}_WINDOWS.csv'", "dstres=OUT/f'evaluation/v56_{ds}.json';dstwin=OUT/f'evaluation/v56_{ds}_WINDOWS.csv'")
 src=src.replace("'new_PPL_evaluations':2,'reuse_history':False", "'new_PPL_evaluations':2,'reuse_history':False")
 src=src.replace("'v55 complete repair and deployment/evaluation'", "'v56 full M1 repair, propagation, save/reload and evaluation'")
 src=src.replace("'equivalent_to_original_R2':equal", "'equivalent_to_v55_M2_plus_repair':equal_v55")
 src=src.replace("'all_96_committed_groups':True", "'all_96_committed_groups':True")
 src=src.replace("'status':'COMPLETE','category':'COMPLETE_PENDING_AGGREGATION'", "'status':'COMPLETE','category':'COMPLETE_PENDING_AGGREGATION'")
 src=src.replace("'hard_cap_seconds':CAP", "'hard_cap_seconds':CAP,'historical_v55_shared_seconds_separate':11037.312983")
 src=src.replace("'equivalent_to_original_R2':equal", "'equivalent_to_v55_M2_plus_repair':equal_v55")
 src=src.replace("'all_96_committed_groups':True,'all_168_modules':True,'all_304128_rows':True,'equivalent_to_original_R2':equal", "'all_96_committed_groups':True,'all_168_modules':True,'all_304128_rows':True,'equivalent_to_v55_M2_plus_repair':equal_v55")
 src=src.replace("print('V55_BLOCK_COMPLETE'", "print('V56_BLOCK_COMPLETE'")
 src=src.replace("v55 12600-second cumulative shared budget exhausted", "v56 9000-second standalone cap exhausted")
 src=src.replace("v46_R1", "historical_v46_background_UNKNOWN")
 src=src.replace("OUT/'R1","OUT/'M1_REPAIRED").replace("OUT/f'R1_","OUT/f'M1_REPAIRED_").replace('no_old_R1_activation_cache_substituted','no_historical_activation_cache_substituted')
 # No V45/V46/V52 checkpoint/repair or activation paths may remain in runner.
 forbidden=['V46/','V45/','V52/','repair_v45.py','original_R2_input_sha256']
 for token in forbidden:
  if token in src: raise RuntimeError(f'forbidden legacy data access remains: {token}')
 (OUT/'run_v56.py').write_text(src,encoding='utf8')
 # Runtime code is byte-equal to v55 except output model/evaluation labels.
 runtime=(OUT/'runtime_v54.py').read_text(encoding='utf8').replace('R1','M1_REPAIRED')
 (OUT/'runtime_v54.py').write_text(runtime,encoding='utf8')
 # Reproduction launcher and transformation/source map are part of the new artifact.
 (OUT/'REPRODUCE.md').write_text('# v56 reproducibility\n\nUse the project frozen runtime:\n\n    E:\\quant-thesis\\.venv\\Scripts\\python.exe run_v56.py\n\nThe run starts from formal M1 at layer0; no v55/v46 repair groups, activations, or C are loaded. v55 frozen fast repair, initialization review, decoder interface, and evaluator are reused.\n',encoding='utf8')
 j={'pass':True,'adapted_runner_sha256':sha(OUT/'run_v56.py'),'copied_frozen_code_dependencies_from_v46_source':{'repair_control_source':'repair_v45.py','repair_control_sha256':sha(OUT/'repair_control.py'),'precision_check_source':'precision_check.py','precision_check_sha256':sha(OUT/'precision_check.py'),'loaded_from_v56_local_copy':True,'parameters_loaded':False},'copied_exact_from_v55':{'fast_accept_v52.py':sha(OUT/'fast_accept_v52.py')==sha(V55/'fast_accept_v52.py'),'initialization_review.py':sha(OUT/'initialization_review.py')==sha(V55/'initialization_review.py'),'decoder_interface.py':sha(OUT/'decoder_interface.py')==sha(V55/'decoder_interface.py')},'runtime_changes':['output directory labels R1 -> M1_REPAIRED only'],'runner_changes':['M2 source -> official M1','fresh layer0 start; no inherited group checkpoints','drop different-initializer v53 input hash assertions','read v18 eval manifests directly','remove v46 repair checkpoint comparison/import','standalone v56 budget and measured online estimate','four-arm final checkpoint/config comparison'],'algorithm_changes':False,'search_helper_behavior_reused_byte_for_byte':True,'C_builder_changes':False,'candidate_order_or_budget_changes':False,'acceptance_or_tolerance_changes':False,'source_map_only_from_v55_algorithm':True}
 (OUT/'SOURCE_REUSE_AND_DIFF.json').write_text(json.dumps(j,indent=2)+'\n',encoding='utf8')
 import sys;sys.path.insert(0,str(OUT));import budget;budget.charge_cpu('v56 CPU-only frozen runner source adaptation',time.perf_counter()-t0,algorithm_core_changed=False)
 print('RUNNER_ADAPTED',j['adapted_runner_sha256'],flush=True)
if __name__=='__main__':main()
