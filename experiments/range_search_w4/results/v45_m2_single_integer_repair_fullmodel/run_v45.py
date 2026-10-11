"""Formal continuation after locked same-precision checks. One final R and PPL."""
import csv,hashlib,json,os,sys,threading,time,traceback
import torch
import run_v42 as rt
import repair_v45 as repair
OUT=rt.OUT;CAP=10800;RESERVE=360
ORIGINAL_CAPTURE=rt.capture;ORIGINAL_C=rt.build_C
def deadline(reserve=0):
 if rt.elapsed()>CAP-reserve:raise TimeoutError('v45 original180min clock/reserve exhausted')
def capture(model,li,hidden,group):
 X=ORIGINAL_CAPTURE(model,li,hidden,group)
 with rt.timer('input_provenance'):
  xh=hashlib.sha256(X.tobytes()).hexdigest();rt.dump('ACTIVE_GROUP_META.json',{'mode':rt.MODE,'layer':li,'group':group,'input_sha256':xh,'input_dim':X.shape[1],'N':len(X),'elapsed_seconds':rt.elapsed()})
  if li==3 and group=='down':assert xh==rt.P['recovery_boundary_input_sha256'];rt.note('recovery_boundary_input',input_sha256=xh,pass_all=True)
 return X
def build_C(X):
 C=ORIGINAL_C(X)
 with rt.timer('C_provenance'):
  ch=rt.th(C.cpu());meta=json.loads((OUT/'ACTIVE_GROUP_META.json').read_text());meta.update(C_sha256=ch,undamped=True,dtype='FP64');rt.dump('ACTIVE_GROUP_META.json',meta)
  if meta['layer']==3 and meta['group']=='down':assert ch==rt.P['recovery_boundary_C_sha256']
 return C
def restored_stats():
 rt.STATS.clear()
 for path in sorted((OUT/'formal/groups').glob('*.pt')):
  ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['committed'];orig=torch.load(OUT/f'original_codes/layer_{ck["layer"]:02d}.pt',map_location='cpu',weights_only=True)
  for n,s in ck['modules'].items():
   rt.STATS.append({'mode':'formal','layer':ck['layer'],'module':n,'rows':s['codes'].shape[0],'input_dim':s['codes'].shape[1],'accepted_updates':int(s['updates'].sum()),'modified_coordinates':int((s['codes']!=orig[n]['codes']).sum()),'candidate_evaluations':int(s['evaluations'].sum()),'initial_search_mse_sum':float(s['initial_mse'].sum()),'final_search_mse_sum':float(s['final_mse'].sum()),'N':262144,'max_p_drift':s['max_p_drift'],'rejected_candidates':int(s['rejections'].sum()) if 'rejections' in s else 0,'rows_with_rejection':int((s['rejections']>0).sum()) if 'rejections' in s else 0,'inherited_from_v44':ck.get('reused_without_new_repair',False),'complete':True})
 rt.csvwrite('MODULE_RESULTS.csv',rt.STATS)
def precision_stage_cost():
 return sum(float(r['seconds']) for r in csv.DictReader((OUT/'STAGE_TIMINGS.csv').open()) if r['mode']=='formal' and r['stage']=='independent_FP64_increment_recheck')
def main():
 assert json.loads((OUT/'CPU_TESTS.json').read_text())['pass'] and json.loads((OUT/'PREFLIGHT_GATE.json').read_text())['pass'] and json.loads((OUT/'SOURCE_REUSE_AUDIT.json').read_text())['pass']
 start=json.loads((OUT/'GPU_START.json').read_text());rt.WALL_START=start['unix'];rt.START=time.perf_counter();rt.MODE='formal';rt.deadline=deadline;deadline()
 for fn,k in [('PROTOCOL.json','protocol_sha256'),('precision_check.py','precision_sha256'),('run_v42.py','legacy_runtime_sha256')]:assert rt.sha(OUT/fn)==start[k]
 identity={fn:rt.sha(OUT/fn) for fn in ['PROTOCOL.json','precision_check.py','repair_v45.py','run_v45.py','CPU_TESTS.json','SOURCE_REUSE_AUDIT.json','PREFLIGHT_GATE.json','preflight_v45.py']}
 lock=OUT/'IMPLEMENTATION_LOCK.json'
 if lock.exists():
  assert json.loads(lock.read_text())==identity and '--resume' in sys.argv
  assert not (OUT/'CHECKPOINT_LOCK.json').exists()
  with (OUT/'RECOVERY_LOG.jsonl').open('a') as f:f.write(json.dumps({'unix':time.time(),'elapsed_seconds':rt.elapsed(),'skip_committed_groups':True})+'\n')
 else:rt.dump('IMPLEMENTATION_LOCK.json',identity)
 rt.dump('NUMERICAL_PROCESS_LOCK.json',{'frozen_before_formal_resume_unix':time.time(),'protocol_sha256':rt.sha(OUT/'PROTOCOL.json'),'precision_sha256':rt.sha(OUT/'precision_check.py'),'empirical_not_strict_theorem':True,'no_later_threshold_changes':True})
 def hardstop():rt.dump('RUN_STATUS.json',{'status':'INCOMPLETE_HARD_CAP','category':'D','elapsed_seconds':rt.elapsed(),'complete_model':False,'PPL_run':False});os._exit(124)
 threading.Timer(CAP-rt.elapsed(),hardstop).start();rt.capture=capture;rt.build_C=build_C;rt.batch_repair=repair.batch_repair;restored_stats()
 model=rt.model_load();hidden=rt.embed(model);ids=torch.tensor(rt.CAL['samples'][0]['token_ids'][:64],device='cuda')[None];tformal=time.perf_counter();measured=[]
 projection=json.loads((OUT/'COST_PROJECTION.json').read_text())
 for li in range(24):
  missing=[g for g in rt.GROUPS if not (OUT/f'formal/groups/layer_{li:02d}_{g}.pt').exists()]
  if missing:
   base=max(projection['v42_measured_block0_seconds'],sum(measured)/len(measured) if measured else 0)
   extra=max(0,projection['new_precision_recheck_seconds_allowance_prediction']-precision_stage_cost())
   forecast=(24-li)*base*1.35+RESERVE+extra
   rt.dump('BUDGET_STATUS.json',{'elapsed_seconds':rt.elapsed(),'layer':li,'safety_multiplier':1.35,'remaining_base_seconds_prediction':(24-li)*base*1.35,'remaining_numerical_allowance_seconds_prediction':extra,'finish_reserve_seconds':RESERVE,'conservative_remaining_seconds':forecast,'go':rt.elapsed()+forecast<=CAP})
   if rt.elapsed()+forecast>CAP:raise TimeoutError('observed cost cannot support conservative completion reserve')
  t=time.perf_counter();hidden=rt.run_layer(model,li,hidden);sec=time.perf_counter()-t
  if missing:measured.append(sec)
  with (OUT/'BLOCK_TIMINGS.csv').open('a',encoding='utf8',newline='') as f:
   fields=['layer','seconds','elapsed_seconds','new_groups'];w=csv.DictWriter(f,fieldnames=fields)
   if f.tell()==0:w.writeheader()
   w.writerow(dict(zip(fields,[li,sec,rt.elapsed(),len(missing)])))
  rt.dump('formal/PREFIX_STATUS.json',{'completed_layer':li,'hidden_sha256':hashlib.sha256(hidden.tobytes()).hexdigest(),'elapsed_seconds':rt.elapsed()});print('BLOCK_COMPLETE',li,'new_groups',len(missing),'seconds',round(sec,3),'elapsed',round(rt.elapsed(),3),flush=True)
 assert len(rt.STATS)==168 and sum(s['rows'] for s in rt.STATS)==304128
 rt.dump('FORMAL_REPAIR_COMPLETE.json',{'modules':168,'rows':304128,'inherited_modules':27,'inherited_groups':15,'newly_repaired_modules':141,'elapsed_seconds':rt.elapsed(),'formal_continuation_wall_seconds':time.perf_counter()-tformal});deadline(RESERVE)
 rt.MODE='deployment';model=rt.final_save_check(model,ids)
 with rt.timer('configuration_deployment_lock'):
  old=json.loads((rt.V18/'M2/checkpoint/final/config.json').read_text());new=json.loads((OUT/'R/checkpoint/final/config.json').read_text());ignore={'_name_or_path','transformers_version'};assert {k:v for k,v in old.items() if k not in ignore}=={k:v for k,v in new.items() if k not in ignore}
  ready=json.loads((OUT/'CHECKPOINT_LOCK.json').read_text());rt.dump('DEPLOYMENT_LOCK.json',{**ready,'configuration_pass':True,'reload_output_pass':True,'ready_before_PPL_unix':time.time()})
 rt.evaluate(model);rt.dump('RUN_STATUS.json',{'status':'COMPLETE','complete_model':True,'PPL_run':True,'modules':168,'rows':304128,'elapsed_seconds':rt.elapsed(),'hard_cap_seconds':CAP,'gpu_name':torch.cuda.get_device_name(),'torch':torch.__version__})
if __name__=='__main__':
 try:main()
 except Exception as ex:
  traceback.print_exc();f={'status':'FAILED_STOP','category':'D','error':repr(ex),'traceback':traceback.format_exc(),'elapsed_seconds':rt.elapsed() if rt.WALL_START else 0,'complete_model':False,'PPL_run':False};rt.dump('RUN_STATUS.json',f)
  with (OUT/'FAILED_ATTEMPTS.jsonl').open('a',encoding='utf8') as out:out.write(json.dumps(f)+'\n')
 finally:os._exit(0)
