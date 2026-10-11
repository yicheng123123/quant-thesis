"""Budget/output coordinator only; v42 repair/propagation/deployment functions unchanged."""
import csv,gc,hashlib,json,os,sys,threading,time,traceback
from pathlib import Path
import torch
import run_v42 as rt
OUT=Path(__file__).resolve().parent
CAP=10800;RESERVE=360
def deadline(reserve=0):
 if rt.elapsed()>CAP-reserve:raise TimeoutError(f'v43 hard budget/reserve exhausted: {rt.elapsed()}s, reserve{reserve}')
def restore_stats():
 # Committed checkpoint statistics are authoritative. Never repeat a committed group.
 rt.STATS.clear()
 for path in sorted((OUT/'formal/groups').glob('*.pt')):
  ck=torch.load(path,map_location='cpu',weights_only=False)
  for name,s in ck['modules'].items():
   rt.STATS.append({'mode':'formal','layer':ck['layer'],'module':name,'rows':s['codes'].shape[0],'input_dim':s['codes'].shape[1],'accepted_updates':int(s['updates'].sum()),'modified_coordinates':int((s['codes']!=torch.load(OUT/f'original_codes/layer_{ck["layer"]:02d}.pt',map_location='cpu',weights_only=True)[name]['codes']).sum()),'candidate_evaluations':int(s['evaluations'].sum()),'initial_search_mse_sum':float(s['initial_mse'].sum()),'final_search_mse_sum':float(s['final_mse'].sum()),'N':262144,'max_p_drift':s['max_p_drift'],'complete':True})
 rt.csvwrite('MODULE_RESULTS.csv',rt.STATS)
def main():
 assert rt.OUT==OUT and json.loads((OUT/'SOURCE_REUSE_AUDIT.json').read_text())['pass']
 proj=json.loads((OUT/'COST_PROJECTION.json').read_text());assert proj['go']
 identity={'coordinator_sha256':rt.sha(__file__),'implementation_sha256':rt.sha(OUT/'run_v42.py'),'audit_sources_sha256':rt.sha(OUT/'audit_sources.py'),'protocol_sha256':rt.sha(OUT/'PROTOCOL.json'),'source_reuse_audit_sha256':rt.sha(OUT/'SOURCE_REUSE_AUDIT.json')}
 assert identity['implementation_sha256']==json.loads((OUT/'SOURCE_REUSE_AUDIT.json').read_text())['runtime_sha256']
 resume='--resume' in sys.argv
 if resume:
  start=json.loads((OUT/'GPU_START.json').read_text());assert all(start[k]==v for k,v in identity.items()),'frozen resume identity mismatch'
  assert not (OUT/'DEPLOYMENT_LOCK.json').exists(),'deployment lock is terminal; no extra model or repeat PPL'
  rt.WALL_START=start['unix'];restore_stats()
  with (OUT/'RECOVERY_LOG.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'unix':time.time(),'pid':os.getpid(),'elapsed_seconds':rt.elapsed(),'committed_groups':len(list((OUT/'formal/groups').glob('*.pt'))),'skip_committed_repair':True})+'\n')
 else:
  assert not (OUT/'GPU_START.json').exists(),'preserve existing run; use identical --resume before original deadline'
  rt.WALL_START=time.time();rt.dump('GPU_START.json',{'unix':rt.WALL_START,'hard_cap_seconds':CAP,'pid':os.getpid(),**identity})
 rt.START=time.perf_counter();rt.deadline=deadline;deadline()
 def hardstop():
  rt.dump('RUN_STATUS.json',{'status':'INCOMPLETE_HARD_CAP','category':'D','elapsed_seconds':rt.elapsed(),'complete_model':False,'PPL_run':False});os._exit(124)
 threading.Timer(CAP-rt.elapsed(),hardstop).start()
 rt.dump('IMPLEMENTATION_LOCK.json',identity)
 rt.MODE='precheck'
 with rt.timer('warmup'):z=torch.zeros((64,64),device='cuda');z@z
 rt.MODE='formal';rt.LAYER=-1;model=rt.model_load();ids=torch.tensor(rt.CAL['samples'][0]['token_ids'][:64],device='cuda')[None];hidden=rt.embed(model)
 measured_blocks=[];formal_start=time.perf_counter()
 for li in range(24):
  remaining=24-li;base=max(proj['v42_measured_block0_seconds'],sum(measured_blocks)/len(measured_blocks) if measured_blocks else 0)
  projected_remaining=remaining*base*1.35+RESERVE
  rt.dump('BUDGET_STATUS.json',{'elapsed_seconds':rt.elapsed(),'hard_cap_seconds':CAP,'next_layer':li,'completed_group_checkpoints':len(list((OUT/'formal/groups').glob('*.pt'))),'conservative_remaining_seconds':projected_remaining,'same_safety_multiplier':1.35,'reserve_seconds':RESERVE,'go':rt.elapsed()+projected_remaining<=CAP})
  if rt.elapsed()+projected_remaining>CAP:raise TimeoutError('new measured cost does not fit unchanged conservative forecast and finish reserve')
  t=time.perf_counter();hidden=rt.run_layer(model,li,hidden);seconds=time.perf_counter()-t;measured_blocks.append(seconds)
  rt.dump('formal/PREFIX_STATUS.json',{'completed_layer':li,'hidden_sha256':hashlib.sha256(hidden.tobytes()).hexdigest(),'elapsed_seconds':rt.elapsed()})
  with (OUT/'BLOCK_TIMINGS.csv').open('a',encoding='utf8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['layer','seconds','elapsed_seconds','resumed_run']);
   if f.tell()==0:w.writeheader()
   w.writerow({'layer':li,'seconds':seconds,'elapsed_seconds':rt.elapsed(),'resumed_run':resume})
  print('BLOCK_COMPLETE',li,'seconds',round(seconds,3),'elapsed',round(rt.elapsed(),3),flush=True)
 formal_seconds=time.perf_counter()-formal_start
 assert len(rt.STATS)==168 and sum(s['rows'] for s in rt.STATS)==304128
 rt.dump('FORMAL_REPAIR_COMPLETE.json',{'modules':168,'rows':304128,'formal_repair_this_attempt_wall_seconds':formal_seconds,'elapsed_seconds':rt.elapsed(),'prefix_source':'current R only','fresh_original_M2_initialization':True})
 deadline(RESERVE)
 rt.MODE='deployment';t=time.perf_counter();model=rt.final_save_check(model,ids)
 # Output/grid checks are inherited unchanged; configuration and ready-lock follow them.
 with rt.timer('configuration_ready_lock'):
  old=json.loads((rt.V18/'M2/checkpoint/final/config.json').read_text());new=json.loads((OUT/'R/checkpoint/final/config.json').read_text())
  ignored={'_name_or_path','transformers_version'}
  assert {k:v for k,v in old.items() if k not in ignored}=={k:v for k,v in new.items() if k not in ignored},'saved inference/model configuration changed'
  lock=json.loads((OUT/'CHECKPOINT_LOCK.json').read_text());assert lock['final_model_sha256']==rt.sha(OUT/'R/checkpoint/final/model.safetensors')
  rt.dump('DEPLOYMENT_LOCK.json',{**lock,'deployment_correctness_pass':True,'configuration_equal_except_source_metadata':True,'ready_before_PPL_unix':time.time()})
 save_reload_seconds=time.perf_counter()-t
 t=time.perf_counter();rt.evaluate(model);eval_seconds=time.perf_counter()-t
 rt.dump('RUN_STATUS.json',{'status':'COMPLETE','elapsed_seconds':rt.elapsed(),'hard_cap_seconds':CAP,'complete_model':True,'modules':168,'output_rows':304128,'formal_repair_this_attempt_seconds':formal_seconds,'save_reload_seconds':save_reload_seconds,'PPL_evaluation_seconds':eval_seconds,'PPL_run':True,'gpu_name':torch.cuda.get_device_name(),'torch':torch.__version__,'resume':resume})
if __name__=='__main__':
 try:main()
 except Exception as ex:
  traceback.print_exc();failure={'status':'FAILED_STOP','category':'D','error':repr(ex),'traceback':traceback.format_exc(),'elapsed_seconds':rt.elapsed() if rt.WALL_START else 0,'complete_model':False,'PPL_run':False,'pid':os.getpid()}
  rt.dump('RUN_STATUS.json',failure)
  with (OUT/'FAILED_ATTEMPTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(failure)+'\n')
 finally:os._exit(0)
