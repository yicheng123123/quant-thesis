"""Deployment-only continuation; no repair, no change to frozen numeric rules."""
import run_v52 as r
from run_v52 import *
from transformers import AutoConfig
model=None

def main():
 global model,EVAL_KEY_MAP
 assert not (OUT/'COMPLETE.json').exists(), 'completed run may not be repeated'
 for n,h in read('FROZEN_VALIDATION_LOCK.json')['files'].items():assert sha(OUT/n)==h, n
 for n,h in read('DEPLOYMENT_ADAPTER_LOCK.json')['files'].items():assert sha(OUT/n)==h,n
 diag=read('CONFIG_SCHEMA_DIAGNOSTIC.json');assert diag['all_loaded_fields_equal_excluding_original_ignore']
 oldstop=read('diagnostics/CONFIG_ASSERTION_STOP.json');assert oldstop['status']=='FAILED_STOP' and 'config_equal' in oldstop['traceback']
 lock0=read('diagnostics/PRE_CONFIG_CHECKPOINT_LOCK.json')
 assert sha(OUT/'R1/checkpoint/final/model.safetensors')==lock0['final_model_sha256']
 assert sha(OUT/'R1/checkpoint/final/config.json')==lock0['config_sha256']
 configpaths=[V18/'M2/checkpoint/final',OUT/'R1/checkpoint/final',V45/'R/checkpoint/final']
 ignore={'_name_or_path','transformers_version'}
 loaded=[AutoConfig.from_pretrained(str(p),local_files_only=True).to_dict() for p in configpaths]
 canon=[{k:v for k,v in c.items() if k not in ignore} for c in loaded]
 config_equal=canon[0]==canon[1]==canon[2];assert config_equal,'true loaded inference configuration mismatch'
 budget.start('deployment continuation after config schema diagnosis')
 rt.WALL_START=time.time();rt.START=time.perf_counter();rt.MODE='deployment';rt.LAYER=-1;rt.MODULE='finish';rt.elapsed=elapsed;rt.deadline=deadline
 def hardstop():
  dump('DEPLOYMENT_FINISH_FAILURE.json',{'error':'shared cumulative hard cap','cumulative_seconds':elapsed()});budget.finish();os._exit(124)
 watcher=threading.Timer(max(1,CAP-elapsed()),hardstop);watcher.daemon=True;watcher.start()
 try:
  with rt.timer('finish_deployment_warmup'):
   z=torch.zeros((64,64),device='cuda');z@z
  EVAL_KEY_MAP=r.in_memory_eval_protocol();hashes=EVAL_KEY_MAP
  modelpath=OUT/'R1/checkpoint/final/model.safetensors';refpath=V45/'R/checkpoint/final/model.safetensors';m2path=V18/'M2/checkpoint/final/model.safetensors';diffs=[];diff_m2=[];comp_start=time.perf_counter()
  with safe_open(str(modelpath),framework='pt',device='cpu') as ours,safe_open(str(refpath),framework='pt',device='cpu') as r2,safe_open(str(m2path),framework='pt',device='cpu') as m2:
   assert set(ours.keys())==set(r2.keys())==set(m2.keys())
   for key in ours.keys():
    a=ours.get_tensor(key);b=r2.get_tensor(key);c=m2.get_tensor(key)
    if not torch.equal(a.view(torch.uint8),b.view(torch.uint8)):
     diffs.append({'key':key,'v52_tensor_sha256':tensor_hash(a),'original_R2_tensor_sha256':tensor_hash(b),'max_abs_v52_minus_R2':float((a.float()-b.float()).abs().max()),'unequal_bytes':int((a.view(torch.uint8)!=b.view(torch.uint8)).sum())})
    if not torch.equal(a.view(torch.uint8),c.view(torch.uint8)):
     diff_m2.append(key)
  tensor_key_count=0
  with safe_open(str(modelpath),framework='pt',device='cpu') as f:tensor_key_count=len(f.keys())
  rt.TIMINGS.append({'mode':'deployment','layer':-1,'module':'all_model_tensors','stage':'named_tensor_comparison_vs_original_R2','seconds':time.perf_counter()-comp_start,'elapsed_seconds':elapsed(),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'type':'measured CPU wall'})
  quant_rows=[]
  for li in range(24):
   src=ORIG.get(li) or torch.load(OUT/f'original_codes/layer_{li:02d}.pt',map_location='cpu',weights_only=True)
   for g in GROUP_ORDER:
    ck=torch.load(OUT/'formal/groups'/f'layer_{li:02d}_{g}.pt',map_location='cpu',weights_only=False);assert_group(ck,li,g,src)
    refcpgroup=torch.load(V45/'formal/groups'/f'layer_{li:02d}_{g}.pt',map_location='cpu',weights_only=False)
    for name,res in ck['modules'].items():
     refcp=refcpgroup['modules'][name]
     quant_rows.append({'layer':li,'group':g,'module':name,'codes_equal_original_R2':torch.equal(res['codes'],refcp['codes']),'scales_equal_M2_R2':torch.equal(res['scales'],refcp['scales']),'zeros_equal_M2_R2':torch.equal(res['zeros'],refcp['zeros']),'g_idx_equal_M2_R2':torch.equal(res['g_idx'],refcp['g_idx']),'FP16_grid_weights_equal_R2':torch.equal(res['weight'],refcp['weight']),'v52_input_sha256':ck['input_sha256'],'original_R2_input_sha256':refcpgroup['input_sha256'],'input_matches_original_R2':ck['input_sha256']==refcpgroup['input_sha256'],'v52_C_sha256':ck['C_sha256'],'original_R2_C_sha256':refcpgroup['C_sha256']})
  rt.csvwrite('reports/MODULE_VS_ORIGINAL_R2.csv',quant_rows);rt.csvwrite('reports/TENSOR_DIFFERENCES_VS_ORIGINAL_R2.csv',diffs)
  equal=(not diffs) and config_equal
  lock={'final_model_sha256':sha(modelpath),'config_sha256':sha(outcfg),'protocol_sha256':sha(OUT/'PROTOCOL.json'),'group_checkpoints':[{'path':str(p),'sha256':sha(p)} for p in sorted((OUT/'formal/groups').glob('*.pt'))],'modules_compared_with_original_R2':len(quant_rows),'named_model_tensor_differences_vs_original_R2':len(diffs),'inference_config_equal_original_R2':config_equal,'equivalent_to_original_R2':equal,'locked_before_evaluation_or_reuse_unix':time.time()}
  assert len(lock['group_checkpoints'])==96;dump('CHECKPOINT_LOCK.json',lock)
  deployment={'pass':True,'all_168_grid_pack_reload_and_sample_output_checks':True,'model_sha256':sha(modelpath),'config_sha256':sha(outcfg),'named_model_tensors_equal_original_R2':not diffs,'M2_quantized_target_tensors_different':len(diff_m2),'original_R2_model_sha256':sha(refpath),'M2_checkpoint_sha256':sha(m2path),'config_equal_original_R2':config_equal,'max_abs_diff_summary':max((d['max_abs_v52_minus_R2'] for d in diffs),default=0.0),'tensor_keys_compared':tensor_key_count,'completed_unix':time.time()};dump('DEPLOYMENT_COMPARISON.json',deployment)
  audit=read(V46/'PPL_SOURCE_AUDIT.json');copies=[]
  if equal:
   # The only PPL reuse path is exact named-tensor and normalized-config equality to the audited old R2.
   for ds,key in [('wikitext','wikitext'),('c4_fixed_budget','c4_fixed_budget')]:
    record=audit['historical_source_records'][f'R2_{key}'];srcres=Path(record['result_path']);srcwin=Path(record['windows_path'])
    assert sha(srcres)==record['result_sha256'] and sha(srcwin)==record['windows_sha256']
    dstres=OUT/f'evaluation/M2_FAST_{ds}.json';dstwin=OUT/f'evaluation/M2_FAST_{ds}_WINDOWS.csv'
    shutil.copy2(srcres,dstres);shutil.copy2(srcwin,dstwin)
    copies.append({'dataset':ds,'result_source_sha256':sha(srcres),'result_copy_sha256':sha(dstres),'windows_source_sha256':sha(srcwin),'windows_copy_sha256':sha(dstwin),'PPL':json.loads(srcres.read_text())['ppl'],'NLL':json.loads(srcres.read_text())['nll'],'valid_tokens':record['valid_tokens'],'windows':record['windows'],'reused_because_all_named_tensors_and_config_equal_original_R2':True})
   rt.dump('QUALITY_REUSE_SOURCE.json',{'pass':True,'new_PPL_evaluations':0,'reason':'All named dense model tensors and normalized inference config equal v45 original R2 bitwise/config-normalized.','P3_read':False,'source_audit_sha256':sha(V46/'PPL_SOURCE_AUDIT.json'),'reused_results':copies})
  else:
   rt.MODE='evaluation';assert rt.P['v18']['source_protocol']=={'v16_wikitext_manifest_sha256':EVAL_KEY_MAP['wikitext'],'v16_c4_manifest_sha256':EVAL_KEY_MAP['c4_fixed_budget']}
   rt.evaluate(model)
   for ds in ('wikitext','c4_fixed_budget'):
    srcres=OUT/f'R1_{ds}.json';srcwin=OUT/f'R1_{ds}_WINDOWS.csv';dstres=OUT/f'evaluation/M2_FAST_{ds}.json';dstwin=OUT/f'evaluation/M2_FAST_{ds}_WINDOWS.csv'
    assert srcres.exists() and srcwin.exists();srcres.replace(dstres);srcwin.replace(dstwin)
    res=json.loads(dstres.read_text(encoding='utf8'))
    assert res['model_sha256']==sha(modelpath) and res['manifest_sha256']==EVAL_KEY_MAP['wikitext' if ds=='wikitext' else 'c4_fixed_budget']
    copies.append({'dataset':ds,'result_sha256':sha(dstres),'windows_sha256':sha(dstwin),'PPL':res['ppl'],'NLL':res['nll'],'valid_tokens':res['scored_tokens'],'windows':res['windows'],'fresh_frozen_protocol_evaluation':True})
   rt.dump('QUALITY_REUSE_SOURCE.json',{'pass':True,'new_PPL_evaluations':2,'reason':'At least one named model tensor differed from original R2; one frozen evaluation per corpus after lock.','P3_read':False,'evaluator_code_sha256':sha(V46/'runtime_v46.py'),'source_manifests':hashes,'new_results':copies})

  dump('DEPLOYMENT_CONFIG_EQUIVALENCE.json',{'pass':True,'comparison':'all AutoConfig loaded fields; original excluded fields only','excluded_fields':sorted(ignore),'raw_json_equal':False,'transformers_version':diag['transformers_version'],'diagnostic_sha256':sha(OUT/'CONFIG_SCHEMA_DIAGNOSTIC.json'),'repair_core_unchanged':True,'weights_and_saved_config_unchanged':sha(OUT/'R1/checkpoint/final/model.safetensors')==lock0['final_model_sha256'] and sha(OUT/'R1/checkpoint/final/config.json')==lock0['config_sha256']})
  newlock=read('CHECKPOINT_LOCK.json');newlock.update(deployment_config_equivalence_basis='exact loaded AutoConfig equality with original ignored metadata only',deployment_adapter_lock_sha256=sha(OUT/'DEPLOYMENT_ADAPTER_LOCK.json'));dump('CHECKPOINT_LOCK.json',newlock)
  status={'status':'COMPLETE','category':'COMPLETE_PENDING_AGGREGATION','complete_model':True,'modules':168,'rows':304128,'blocks_completed':24,'model_sha256':sha(modelpath),'equivalent_to_original_R2':equal,'PPL_run':not equal,'PPL_reused':equal,'elapsed_seconds_from_gpu_start':elapsed(),'hard_cap_seconds':CAP,'original_config_assertion_stop_preserved':str(OUT/'diagnostics/CONFIG_ASSERTION_STOP.json'),'deployment_only_continuation':True,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'peak_allocated_bytes':max(torch.cuda.max_memory_allocated(),2210208256),'peak_reserved_bytes':max(torch.cuda.max_memory_reserved(),2657091584)}
  dump('RUN_STATUS.json',status)
  dump('COMPLETE.json',{'pass':True,'status_sha256':sha(OUT/'RUN_STATUS.json'),'all_96_committed_groups':True,'all_168_modules':True,'all_304128_rows':True,'equivalent_to_original_R2':equal,'model_ready_unix':read('DEPLOYMENT_COMPARISON.json')['completed_unix'],'full_job_elapsed_seconds':elapsed(),'no_repair_repeated_in_continuation':True})
  print('V52_DEPLOYMENT_COMPLETE',json.dumps(status),flush=True)
 except BaseException as ex:
  dump('DEPLOYMENT_FINISH_FAILURE.json',{'error':repr(ex),'traceback':traceback.format_exc(),'cumulative_seconds':elapsed(),'no_repair_rerun':True})
  raise
 finally:
  watcher.cancel();budget.finish()
if __name__=='__main__':main()
