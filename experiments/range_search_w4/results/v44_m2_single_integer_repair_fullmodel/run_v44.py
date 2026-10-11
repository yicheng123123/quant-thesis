"""v44 validation, trusted-group recovery and one full-model/PPL continuation."""
import csv,gc,hashlib,json,os,sys,threading,time,traceback
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import run_v42 as rt
import rejection_repair as repair
OUT=Path(__file__).resolve().parent;V43=OUT.parent/'v43_m2_single_integer_repair_fullmodel';CAP=10800;RESERVE=360
ORIGINAL_CAPTURE=rt.capture;ORIGINAL_C=rt.build_C
def deadline(reserve=0):
 if rt.elapsed()>CAP-reserve:raise TimeoutError('v44 original10800s wall budget/reserve exhausted')
def capture(model,li,hidden,group):
 X=ORIGINAL_CAPTURE(model,li,hidden,group)
 with rt.timer('input_provenance'):
  xh=hashlib.sha256(X.tobytes()).hexdigest();rt.dump('ACTIVE_GROUP_META.json',{'mode':rt.MODE,'layer':li,'group':group,'input_sha256':xh,'input_dim':X.shape[1],'N':len(X),'elapsed_seconds':rt.elapsed()})
  if li==3 and group=='up_gate' and rt.MODE in ['validation','formal']:
   assert xh==rt.P['recovery_boundary_input_sha256'],'recovered-prefix boundary input mismatch'
   rt.note('recovery_boundary_input',input_sha256=xh,pass_all=True)
 return X
def build_C(X):
 C=ORIGINAL_C(X)
 with rt.timer('C_provenance'):
  ch=rt.th(C.cpu());meta=json.loads((OUT/'ACTIVE_GROUP_META.json').read_text());meta.update(C_sha256=ch,undamped=True,dtype='FP64');rt.dump('ACTIVE_GROUP_META.json',meta)
  if meta['layer']==3 and meta['group']=='up_gate' and rt.MODE in ['validation','formal']:
   assert ch==rt.P['recovery_boundary_C_sha256'],'recovered boundary C mismatch'
 return C
def restored_stats():
 rt.STATS.clear()
 for path in sorted((OUT/'formal/groups').glob('*.pt')):
  ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['committed']
  orig=torch.load(OUT/f'original_codes/layer_{ck["layer"]:02d}.pt',map_location='cpu',weights_only=True)
  for n,s in ck['modules'].items():
   rt.STATS.append({'mode':'formal','layer':ck['layer'],'module':n,'rows':s['codes'].shape[0],'input_dim':s['codes'].shape[1],'accepted_updates':int(s['updates'].sum()),'modified_coordinates':int((s['codes']!=orig[n]['codes']).sum()),'candidate_evaluations':int(s['evaluations'].sum()),'initial_search_mse_sum':float(s['initial_mse'].sum()),'final_search_mse_sum':float(s['final_mse'].sum()),'N':262144,'max_p_drift':s['max_p_drift'],'rejected_candidates':int(s['rejections'].sum()) if 'rejections' in s else 0,'rows_with_rejection':int((s['rejections']>0).sum()) if 'rejections' in s else 0,'inherited_from_v43':ck.get('reused_without_new_repair',False),'complete':True})
 rt.csvwrite('MODULE_RESULTS.csv',rt.STATS)
def fixture_validation():
 rt.MODE='validation';model=rt.model_load();hidden=rt.embed(model);changed={}
 for li in range(4):
  rt.LAYER=li
  for group in rt.GROUPS:
   cp=OUT/f'formal/groups/layer_{li:02d}_{group}.pt'
   if cp.exists():
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    with torch.no_grad():
     for n,s in ck['modules'].items():model.model.layers[li].get_submodule(n).weight.copy_(s['weight'].cuda());changed[f'model.layers.{li}.{n}.weight']=s['weight']
  if li<3:hidden=rt.advance_block(model,li,hidden)
 with rt.timer('restored_parameter_integrity'):
  checked=[]
  with safe_open(str(rt.V18/'M2/checkpoint/final/model.safetensors'),framework='pt',device='cpu') as sf:
   for k in sf.keys():
    expected=changed[k] if k in changed else sf.get_tensor(k);actual=model.get_parameter(k).detach().cpu();assert torch.equal(actual.view(torch.int16),expected.view(torch.int16)),f'parameter pollution: {k}';checked.append(k)
  rt.dump('validation/PARAMETER_RESTORE_CHECK.json',{'pass':True,'saved_tensor_keys_verified':len(checked),'committed_weight_keys':len(changed),'uncommitted_parameters_exact_original_M2':True,'failed_up_gate_original_M2':True})
 rt.LAYER=3;rt.MODULE='up_gate';X=capture(model,3,hidden,'up_gate');C=build_C(X)
 s=torch.load(OUT/'original_codes/layer_03.pt',map_location='cpu',weights_only=True)['mlp.up_proj']
 failure=torch.load(V43/'NUMERICAL_FAILURE_formal_3_mlp_up_proj.pt',map_location='cpu',weights_only=True)
 with safe_open(str(rt.FP/'model.safetensors'),framework='pt',device='cpu') as sf:W=sf.get_tensor('model.layers.3.mlp.up_proj.weight').half()
 sc=s['scales'].cuda()[:,s['g_idx'].long().cuda()];original_E=(sc*(s['codes'].cuda().float()-8)).half().float().double()-W.cuda().double();original_loss=rt.direct(X,original_E);del sc,original_E
 # Test-only one-accept bound. All4864 rows and full original FP32 scorer shape remain.
 result=repair.batch_repair(failure['code'].byte(),s['scales'],s['g_idx'],W,X,C,'mlp.up_proj_failure_fixture',False,fixture_accept_cap=1,fixture_original_initial_loss=original_loss)
 first=result['rejection_events'][0];assert first['attempt']==1
 for j,row in enumerate(failure['rows'].tolist()):
  idx=(first['rows']==row).nonzero().flatten();assert idx.numel()==1;k=int(idx[0]);assert int(first['columns'][k])==490 and float(first['direct_delta'][k])>=0
  assert abs(float(first['direct_delta'][k])-float(failure['actual'][j]))<=1e-15
  assert torch.equal(result['codes'][row],failure['code'][row].byte()) and int(result['updates'][row])==0
 assert int(result['updates'].sum())>0 and int(result['evaluations'].max())<=128*896
 # Every rejected-state candidate must be unique until an accepted update changes state.
 seen=set()
 for event in result['rejection_events']:
  for k,row in enumerate(event['rows'].tolist()):
   key=(row,int(event['state_accept_count'][k]),int(event['columns'][k]),int(event['offsets'][k]));assert key not in seen;seen.add(key)
 torch.save(result,OUT/'validation/FAILURE_FIXTURE_RESULT.pt')
 rt.dump('VALIDATION_GATE.json',{'pass':True,'cpu_tests_pass':json.loads((OUT/'CPU_TESTS.json').read_text())['pass'],'original_failed_rows':failure['rows'].tolist(),'all3_original_candidates_rejected':True,'rejected_code_p_loss_accept_count_bitwise_restored':True,'legal_other_rows_independently_accepted':int(result['updates'].sum()),'repeated_candidate_in_same_state':False,'final_original_direct_check_pass':True,'batch_output_rows':4864,'input_dim':896,'N':262144,'input_chunk':4096,'FP32_full_batch_scorer_unchanged':True,'fixture_accept_limit':1,'formal_accept_limit':32,'fixture_counts_start_zero_not_original_v43_partial_counts':True,'elapsed_seconds':rt.elapsed()})
 del model,hidden,X,C,W,result,original_loss;gc.collect();torch.cuda.empty_cache()
def main():
 assert json.loads((OUT/'CPU_TESTS.json').read_text())['pass'] and json.loads((OUT/'SOURCE_REUSE_AUDIT.json').read_text())['pass']
 projection=json.loads((OUT/'COST_PROJECTION.json').read_text());assert projection['go']
 identity={'coordinator_sha256':rt.sha(__file__),'repair_sha256':rt.sha(OUT/'rejection_repair.py'),'legacy_runtime_sha256':rt.sha(OUT/'run_v42.py'),'protocol_sha256':rt.sha(OUT/'PROTOCOL.json'),'cpu_tests_sha256':rt.sha(OUT/'CPU_TESTS.json'),'source_audit_sha256':rt.sha(OUT/'SOURCE_REUSE_AUDIT.json')}
 resume='--resume' in sys.argv
 if resume:
  start=json.loads((OUT/'GPU_START.json').read_text());assert all(start[k]==v for k,v in identity.items());assert not (OUT/'DEPLOYMENT_LOCK.json').exists();rt.WALL_START=start['unix']
  with (OUT/'RECOVERY_LOG.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'unix':time.time(),'pid':os.getpid(),'elapsed_seconds':rt.elapsed(),'skip_committed_groups':True})+'\n')
 else:
  assert not (OUT/'GPU_START.json').exists();rt.WALL_START=time.time();rt.dump('GPU_START.json',{'unix':rt.WALL_START,'pid':os.getpid(),'hard_cap_seconds':CAP,**identity})
 rt.START=time.perf_counter();rt.deadline=deadline;rt.capture=capture;rt.build_C=build_C;rt.batch_repair=repair.batch_repair;deadline()
 def hardstop():rt.dump('RUN_STATUS.json',{'status':'INCOMPLETE_HARD_CAP','category':'D','elapsed_seconds':rt.elapsed(),'complete_model':False,'PPL_run':False});os._exit(124)
 threading.Timer(CAP-rt.elapsed(),hardstop).start();rt.dump('IMPLEMENTATION_LOCK.json',identity)
 rt.MODE='warmup'
 with rt.timer('warmup'):z=torch.zeros((64,64),device='cuda');z@z
 if not (OUT/'VALIDATION_GATE.json').exists():fixture_validation()
 assert json.loads((OUT/'VALIDATION_GATE.json').read_text())['pass'];restored_stats()
 rt.MODE='formal';rt.LAYER=-1;model=rt.model_load();hidden=rt.embed(model);ids=torch.tensor(rt.CAL['samples'][0]['token_ids'][:64],device='cuda')[None];tformal=time.perf_counter();measured=[]
 for li in range(24):
  missing=[g for g in rt.GROUPS if not (OUT/f'formal/groups/layer_{li:02d}_{g}.pt').exists()]
  if missing:
   base=max(projection['v42_measured_block0_seconds'],sum(measured)/len(measured) if measured else 0);forecast=(24-li)*base*1.35+RESERVE
   rt.dump('BUDGET_STATUS.json',{'elapsed_seconds':rt.elapsed(),'layer':li,'same_safety_multiplier':1.35,'conservative_remaining_seconds':forecast,'go':rt.elapsed()+forecast<=CAP,'remaining_new_groups':sum(not (OUT/f'formal/groups/layer_{l:02d}_{g}.pt').exists() for l in range(li,24) for g in rt.GROUPS)})
   if rt.elapsed()+forecast>CAP:raise TimeoutError('measured cost no longer supports conservative completion reserve')
  t=time.perf_counter();hidden=rt.run_layer(model,li,hidden);sec=time.perf_counter()-t
  if missing:measured.append(sec)
  with (OUT/'BLOCK_TIMINGS.csv').open('a',encoding='utf8',newline='') as f:
   fields=['layer','seconds','elapsed_seconds','new_groups'];w=csv.DictWriter(f,fieldnames=fields)
   if f.tell()==0:w.writeheader()
   w.writerow(dict(zip(fields,[li,sec,rt.elapsed(),len(missing)])))
  rt.dump('formal/PREFIX_STATUS.json',{'completed_layer':li,'hidden_sha256':hashlib.sha256(hidden.tobytes()).hexdigest(),'elapsed_seconds':rt.elapsed()});print('BLOCK_COMPLETE',li,'new_groups',len(missing),'seconds',round(sec,3),'elapsed',round(rt.elapsed(),3),flush=True)
 assert len(rt.STATS)==168 and sum(s['rows'] for s in rt.STATS)==304128
 rt.dump('FORMAL_REPAIR_COMPLETE.json',{'modules':168,'rows':304128,'inherited_modules':25,'inherited_groups':14,'newly_repaired_modules':143,'elapsed_seconds':rt.elapsed(),'formal_continuation_wall_seconds':time.perf_counter()-tformal});deadline(RESERVE)
 rt.MODE='deployment';model=rt.final_save_check(model,ids)
 with rt.timer('configuration_deployment_lock'):
  old=json.loads((rt.V18/'M2/checkpoint/final/config.json').read_text());new=json.loads((OUT/'R/checkpoint/final/config.json').read_text());ignore={'_name_or_path','transformers_version'};assert {k:v for k,v in old.items() if k not in ignore}=={k:v for k,v in new.items() if k not in ignore}
  lock=json.loads((OUT/'CHECKPOINT_LOCK.json').read_text());rt.dump('DEPLOYMENT_LOCK.json',{**lock,'configuration_pass':True,'reload_output_pass':True,'ready_before_PPL_unix':time.time()})
 rt.evaluate(model);rt.dump('RUN_STATUS.json',{'status':'COMPLETE','complete_model':True,'PPL_run':True,'modules':168,'rows':304128,'elapsed_seconds':rt.elapsed(),'hard_cap_seconds':CAP,'gpu_name':torch.cuda.get_device_name(),'torch':torch.__version__})
if __name__=='__main__':
 try:main()
 except Exception as ex:
  traceback.print_exc();failure={'status':'FAILED_STOP','category':'D','error':repr(ex),'traceback':traceback.format_exc(),'elapsed_seconds':rt.elapsed() if rt.WALL_START else 0,'complete_model':False,'PPL_run':False};rt.dump('RUN_STATUS.json',failure)
  with (OUT/'FAILED_ATTEMPTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(failure)+'\n')
 finally:os._exit(0)
