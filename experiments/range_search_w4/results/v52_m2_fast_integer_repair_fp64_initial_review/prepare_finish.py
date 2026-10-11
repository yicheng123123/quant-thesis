from pathlib import Path
import hashlib,json,time
O=Path(__file__).resolve().parent
s=(O/'run_v52.py').read_text(encoding='utf8')
tail=s[s.index('  modelpath=OUT/'):s.index('  if model is not None:del model')]
tail='\n'.join(x[2:] for x in tail.splitlines())
tail=tail.replace("  rt.MODE='evaluation';assert", "  rt.MODE='deployment'\n  with rt.timer('finish_evaluation_model_reload'):\n   model=AutoModelForCausalLM.from_pretrained(str(OUT/'R1/checkpoint/final'),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()\n  rt.MODE='evaluation';assert")
head='''"""Deployment-only continuation; no repair, no change to frozen numeric rules."""
import run_v52 as r
from run_v52 import *
from transformers import AutoConfig
model=None

def main():
 global model,EVAL_KEY_MAP
 outcfg=OUT/'R1/checkpoint/final/config.json'
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
'''
body='\n'.join('  '+x for x in tail.splitlines())
foot='''
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
'''
(O/'finish_v52.py').write_text(head+body+'\n'+foot,encoding='utf8')
# Preserve the original formal stop and intermediate lock exactly; never rewrite these copies.
for src,dst in [('RUN_STATUS.json','diagnostics/CONFIG_ASSERTION_STOP.json'),('CHECKPOINT_LOCK.json','diagnostics/PRE_CONFIG_CHECKPOINT_LOCK.json')]:
 p=O/dst
 if p.exists():assert p.read_bytes()==(O/src).read_bytes()
 else:p.write_bytes((O/src).read_bytes())
(O/'DEPLOYMENT_CONFIG_COMPATIBILITY.md').write_text('''v52 deployment serialization adaptation, frozen before continuation.

The original run stopped on raw JSON equality after all repair, terminal checks, pack/grid checks, reload, and sample output regression passed. The exact stop is retained in diagnostics/CONFIG_ASSERTION_STOP.json; no repair is repeated.

Transformers 5.16.1 maps the legacy torch_dtype, rope_theta/rope_scaling and disabled sliding_window configuration to dtype, rope_parameters and 24 full_attention layer_types. It also serializes default pad_token_id=null. All three source configurations, when loaded by the same actual AutoConfig, have exactly equal to_dict fields after excluding ONLY the original _name_or_path and transformers_version metadata. No new field is ignored and no source/config/model is changed. See CONFIG_SCHEMA_DIAGNOSTIC.json and local installed configuration_qwen2.py.

finish_v52.py repeats that full loaded-config comparison, then continues the original tensor/grid comparisons and unchanged evaluator. This is a deployment schema adapter, not a numerical protocol amendment. Frozen run_v52.py, repair core, numerical thresholds and output checkpoint bytes remain unchanged. If loaded fields differ, stop. If named model tensors differ from R2, run each frozen evaluator exactly once after locking. The original failed run and all continuation costs remain in the ledger.
''',encoding='utf8')
files=['finish_v52.py','CONFIG_SCHEMA_DIAGNOSTIC.json','DEPLOYMENT_CONFIG_COMPATIBILITY.md','diagnostics/CONFIG_ASSERTION_STOP.json','diagnostics/PRE_CONFIG_CHECKPOINT_LOCK.json']
(O/'DEPLOYMENT_ADAPTER_LOCK.json').write_text(json.dumps({'frozen_before_deployment_continue_unix':time.time(),'files':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in files},'no_numeric_rules_changed':True},indent=2)+'\n')
compile((O/'finish_v52.py').read_text(),str(O/'finish_v52.py'),'exec')
print('DEPLOYMENT_ADAPTER_FROZEN_NO_REPAIR_CHANGE')


