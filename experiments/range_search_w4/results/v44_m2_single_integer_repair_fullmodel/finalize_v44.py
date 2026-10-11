"""CPU-only v44 persisted state-machine, deployment, window and cost audit."""
import csv,json,math,time
from collections import Counter,defaultdict
from pathlib import Path
import torch
from safetensors import safe_open
from audit_sources import OUT,V18,sha,th,pack,unpack
V42=OUT.parent/'v42_m2_single_integer_repair_fullmodel'
V43=OUT.parent/'v43_m2_single_integer_repair_fullmodel'
torch.set_num_threads(4)
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def dump(n,v):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def csvread(n):return list(csv.DictReader((OUT/n).open(encoding='utf8')))
def csvwrite(n,rr):
 if not rr:return
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

def verify_delta(s,loss,rr,pred,actual,accept):
 threshold=torch.maximum(s['initial_mse'][rr]*1e-8,torch.full_like(pred,1e-14))
 assert bool(torch.isfinite(pred).all() and torch.isfinite(actual).all())
 assert bool((pred < -threshold).all())
 assert bool(((actual<0) if accept else (actual>=0)).all())
 assert bool(((actual-pred).abs()<=torch.maximum(loss[rr]*2e-6,pred.abs()*1e-3)).all())

def replay_legacy(s,o):
 code=o['codes'].long().clone();rows,d=code.shape
 counts=torch.zeros(rows,dtype=torch.long);updates=counts.clone();loss=s['initial_mse'].clone();active=torch.ones(rows,dtype=torch.bool)
 moves={m['round']:m for m in s['moves']};assert len(moves)==len(s['moves'])
 for attempt in range(1,33):
  if not active.any():break
  counts[active]+=(code[active]>0).sum(1)+(code[active]<15).sum(1)
  m=moves.get(attempt);active[:]=False
  if m is None:break
  rr,col,off=m['rows'],m['columns'],m['offsets'];assert rr.unique().numel()==rr.numel()
  assert torch.equal(m['evaluations'],counts[rr]);verify_delta(s,loss,rr,m['predicted_delta'],m['direct_delta'],True)
  code[rr,col]+=off;loss[rr]+=m['direct_delta'];updates[rr]+=1;active[rr]=True
 assert torch.equal(code.byte(),s['codes']) and torch.equal(counts,s['evaluations']) and torch.equal(updates,s['updates'])
 assert bool(torch.isclose(loss,s['final_mse'],rtol=2e-6,atol=1e-12).all())
 return counts,updates,torch.zeros_like(counts)

def replay_revision(s,o):
 code=o['codes'].long().clone();rows,d=code.shape;cap=128*d
 counts=torch.zeros(rows,dtype=torch.long);updates=counts.clone();rejections=counts.clone();status=counts.clone();loss=s['initial_mse'].clone()
 blocked=torch.zeros((rows,d,2),dtype=torch.bool)
 moves={m['round']:m for m in s['moves']};events={e['attempt']:e for e in s['rejection_events']}
 assert len(moves)==len(s['moves']) and len(events)==len(s['rejection_events'])
 attempts=set();rejection_rows=[]
 for sp in s['score_passes']:
  attempt=sp['attempt'];assert attempt not in attempts;attempts.add(attempt)
  status[(status==0)&(updates>=32)]=2;status[(status==0)&(counts>=cap)]=3
  active=status==0;legal=torch.stack((code>0,code<15),-1);eligible=(legal&~blocked).sum((1,2));remaining=cap-counts
  expected=torch.where(active,torch.minimum(eligible,remaining),torch.zeros_like(counts))
  assert torch.equal(expected,sp['evaluations_added']),'actual reevaluation accounting mismatch'
  counts+=expected;after=sp['status_after_scoring']
  assert bool(((after==0)|(after==1)|(after==2)|(after==3)).all())
  assert torch.equal(after[~active],status[~active])
  assert bool((after[active&(eligible==0)]==1).all())
  assert bool((after[active&(eligible>remaining)]==3).all())
  assert bool(((after[active&(eligible>0)&(eligible<=remaining)]==0)|(after[active&(eligible>0)&(eligible<=remaining)]==1)).all())
  m=moves.get(attempt);e=events.get(attempt)
  accepted=m['rows'] if m is not None else torch.empty(0,dtype=torch.long)
  rejected=e['rows'] if e is not None else torch.empty(0,dtype=torch.long)
  proposed=torch.cat((accepted,rejected));assert proposed.unique().numel()==proposed.numel()
  assert torch.equal(torch.sort(proposed).values,(after==0).nonzero().flatten())
  # Both events originate from the same independent-row tentative batch.
  for obj,rr,is_accept in [(m,accepted,True),(e,rejected,False)]:
   if obj is None:continue
   col,off=obj['columns'],obj['offsets'];assert bool(((off==-1)|(off==1)).all())
   kk=(off+1)//2;assert not bool(blocked[rr,col,kk].any())
   assert bool(legal[rr,col,kk].all()) and torch.equal(obj['evaluations'],counts[rr])
   verify_delta(s,loss,rr,obj['predicted_delta'],obj['direct_delta'],is_accept)
   if is_accept:
    code[rr,col]+=off;loss[rr]+=obj['direct_delta'];updates[rr]+=1;blocked[rr]=False
    assert torch.equal(obj['updates_after'],updates[rr])
   else:
    assert obj['restored'] and torch.equal(obj['old_loss'],loss[rr]) and torch.equal(obj['state_accept_count'],updates[rr])
    blocked[rr,col,kk]=True;rejections[rr]+=1
    for k,row in enumerate(rr.tolist()):
     old=float(loss[row]);pred=float(obj['predicted_delta'][k]);actual=float(obj['direct_delta'][k])
     rejection_rows.append({'attempt':attempt,'row':row,'coordinate':int(col[k]),'offset':int(off[k]),'state_accept_count':int(updates[row]),'evaluations':int(counts[row]),'predicted_delta':pred,'direct_delta':actual,'old_loss':old,'relative_predicted_delta':pred/old if old else 0,'relative_direct_delta':actual/old if old else 0,'restored':True})
  status=after.clone()
  assert bool(((code>=0)&(code<=15)).all() and (counts<=cap).all() and (updates<=32).all())
 status[(status==0)&(updates>=32)]=2;status[(status==0)&(counts>=cap)]=3
 assert set(moves)|set(events)<=attempts
 assert torch.equal(code.byte(),s['codes']) and torch.equal(counts,s['evaluations']) and torch.equal(updates,s['updates']) and torch.equal(rejections,s['rejections']) and torch.equal(status,s['status'])
 assert not bool((status==0).any()) and bool(torch.isclose(loss,s['final_mse'],rtol=2e-6,atol=1e-12).all())
 return counts,updates,rejections,rejection_rows

def persisted_audit():
 identity=read('GPU_START.json');assert sha(OUT/'run_v42.py')==identity['legacy_runtime_sha256']==sha(V42/'run_v42.py')
 for fn,key in [('run_v44.py','coordinator_sha256'),('rejection_repair.py','repair_sha256'),('PROTOCOL.json','protocol_sha256'),('CPU_TESTS.json','cpu_tests_sha256'),('SOURCE_REUSE_AUDIT.json','source_audit_sha256')]:assert sha(OUT/fn)==identity[key]
 source=read('SOURCE_AUDIT.json');reuse=read('SOURCE_REUSE_AUDIT.json')
 for s in source['sources']:assert sha(s['source'])==s['source_sha256'] and sha(s['derived'])==s['derived_sha256']
 assert sha(V18/'M2/checkpoint/final/model.safetensors')==source['model_sha256']
 assert sha(V18/'CALIBRATION_MANIFEST.json')==reuse['calibration_sha256']
 for s in reuse['historical_v43_watched']:assert sha(s['path'])==s['sha256']
 for s in reuse['inherited_groups']:assert sha(s['source_path'])==s['source_sha256']
 assert json.loads((V43/'RUN_STATUS.json').read_text())['status']=='NUMERICAL_STOP_FROZEN_RULE'
 modules=[];locks=[];rowrecords=[];rejection_records=[]
 for path in sorted((OUT/'formal/groups').glob('*.pt')):
  ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['committed'] and ck['protocol_sha256']==identity['protocol_sha256']
  orig=torch.load(OUT/f'original_codes/layer_{ck["layer"]:02d}.pt',map_location='cpu',weights_only=True)
  inherited=ck.get('reused_without_new_repair',False)
  if inherited:
   assert sha(ck['source_v43_path'])==ck['source_v43_sha256'];old=torch.load(ck['source_v43_path'],map_location='cpu',weights_only=False)
   assert ck['input_sha256']==old['input_sha256'] and ck['C_sha256']==old['C_sha256']
  for name,s in ck['modules'].items():
   c=s['codes'];o=orig[name];assert bool((c<=15).all()) and torch.equal(unpack(s['packed_u4']),c) and torch.equal(pack(c),s['packed_u4'])
   for key in ['scales','zeros','g_idx']:assert torch.equal(s[key],o[key])
   q=(s['scales'][:,s['g_idx'].long()]*(c.float()-8)).half();assert torch.equal(q.view(torch.int16),s['weight'].view(torch.int16))
   assert bool(torch.isfinite(s['initial_mse']).all() and torch.isfinite(s['final_mse']).all() and (s['final_mse']<=s['initial_mse']).all())
   if inherited:
    for key in ['codes','weight','scales','zeros','g_idx','updates','evaluations','status','initial_mse','final_mse']:assert torch.equal(s[key],old['modules'][name][key])
    counts,updates,rejections=replay_legacy(s,o)
   else:
    counts,updates,rejections,rr=replay_revision(s,o)
    rejection_records.extend({'layer':ck['layer'],'module':name,**r} for r in rr)
   rows,dim=c.shape;assert bool((counts<=128*dim).all() and (updates<=32).all())
   initial=float(s['initial_mse'].sum());final=float(s['final_mse'].sum())
   modules.append({'layer':ck['layer'],'module':name,'inherited_from_v43':inherited,'rows':rows,'input_dim':dim,'accepted_updates':int(updates.sum()),'modified_coordinates':int((c!=o['codes']).sum()),'actual_candidate_evaluations':int(counts.sum()),'rejected_candidates':int(rejections.sum()),'rows_with_rejection':int((rejections>0).sum()),'initial_search_mse_sum':initial,'final_search_mse_sum':final,'search_gain_percent':100*(initial-final)/initial,'initial_pooled_SSE':initial*262144,'final_pooled_SSE':final*262144,'max_p_drift':s['max_p_drift'],'codes_sha256':th(c),'scale_sha256':th(s['scales']),'zero_sha256':th(s['zeros']),'g_idx_sha256':th(s['g_idx']),'input_sha256':ck['input_sha256'],'C_sha256':ck['C_sha256'],'persisted_moves_replayed':True})
   for row in range(rows):rowrecords.append({'layer':ck['layer'],'module':name,'inherited_from_v43':inherited,'row':row,'input_dim':dim,'initial_search_mse':float(s['initial_mse'][row]),'final_search_mse':float(s['final_mse'][row]),'updates':int(updates[row]),'evaluations':int(counts[row]),'eval_cap':128*dim,'rejections':int(rejections[row]),'status':{1:'NO_REMAINING_PREDICTED_DESCENT',2:'UPDATE_CAP',3:'EVALUATION_CAP'}[int(s['status'][row])]})
  locks.append({'path':str(path),'sha256':sha(path),'layer':ck['layer'],'group':ck['group'],'inherited_from_v43':inherited,'input_sha256':ck['input_sha256'],'C_sha256':ck['C_sha256']})
 csvwrite('reports/MODULE_RESULTS.csv',modules);csvwrite('reports/ROW_RESULTS.csv',rowrecords);csvwrite('reports/REJECTION_EVENTS.csv',rejection_records)
 labels=['NO_REMAINING_PREDICTED_DESCENT','EVALUATION_CAP','UPDATE_CAP']
 stopcounts={k:sum(r['status']==k for r in rowrecords) for k in labels};newstopcounts={k:sum(r['status']==k and not r['inherited_from_v43'] for r in rowrecords) for k in labels}
 dump('reports/OUTPUT_AUDIT.json',{'pass':True,'modules':len(modules),'rows':len(rowrecords),'persisted_integer_moves_replayed':True,'reevaluation_and_state_blacklist_replayed':True,'rejected_state_conservation_checked':True,'grid_unchanged':True,'budget_caps_verified':True,'historical_source_hashes_unchanged':True,'v43_status_preserved':'NUMERICAL_STOP_FROZEN_RULE','original_FP32_direct_scorer_byte_identical':True,'stop_counts_full_model':stopcounts,'stop_counts_new_v44_rows':newstopcounts,'checkpoints':locks})
 return modules,rowrecords,rejection_records

def ppl_audit():
 baseline=[]
 for m in ['M1','M2']:
  for ds in ['wikitext','c4_fixed_budget']:
   path=V18/f'evaluation/raw/{m}_{ds}.json';r=json.loads(path.read_text());assert r['status']=='PASS'
   for wf in r['model_weight_files']:assert sha(Path(r['model_source'])/wf['name'])==wf['sha256']
   wf=V18/r['window_csv'];wr=list(csv.DictReader(wf.open()));nt=sum(int(w['valid_tokens']) for w in wr);nll=sum(float(w['window_nll']) for w in wr)/nt
   assert nt==r['evaluated_tokens']==r['input_tokens']-1 and abs(nll-r['average_nll'])<1e-10 and abs(math.exp(nll)-r['perplexity'])<1e-10
   mf=V18/'evaluation/manifests'/('WIKITEXT_TEST_MANIFEST.json' if ds=='wikitext' else 'C4_EVAL_MANIFEST.json');assert sha(mf)==r['evaluation_manifest_sha256']
   baseline.append({'method':m,'dataset':ds,'ppl':r['perplexity'],'raw_sha256':sha(path),'windows_sha256':sha(wf),'manifest_sha256':sha(mf),'model_files_verified':True})
 dump('reports/BASELINE_AUDIT.json',{'pass':True,'baselines':baseline});comparison=[]
 for ds in ['wikitext','c4_fixed_budget']:
  p=OUT/f'R_{ds}.json'
  if not p.exists():continue
  r=json.loads(p.read_text());wr=csvread(f'R_{ds}_WINDOWS.csv');nt=sum(int(w['valid_tokens']) for w in wr);nll=sum(float(w['window_nll']) for w in wr)/nt
  assert nt==r['scored_tokens']==r['input_tokens']-1 and len(wr)==r['windows'] and abs(nll-r['nll'])<1e-10 and abs(math.exp(nll)-r['ppl'])<1e-10
  ready=read('DEPLOYMENT_LOCK.json');assert r['model_sha256']==ready['final_model_sha256']==sha(OUT/'R/checkpoint/final/model.safetensors')
  mf=V18/'evaluation/manifests'/('WIKITEXT_TEST_MANIFEST.json' if ds=='wikitext' else 'C4_EVAL_MANIFEST.json');manifest=json.loads(mf.read_text());assert sha(mf)==r['manifest_sha256']
  expected=[];prev=0
  for begin in range(0,r['input_tokens'],512):
   end=min(begin+2048,r['input_tokens']);target=end-prev;valid=target-1 if begin==0 else target
   expected.append((begin,end,valid));prev=end
   if end==r['input_tokens']:break
  assert [(int(w['begin']),int(w['end']),int(w['valid_tokens'])) for w in wr]==expected
  assert r['max_length']==2048 and r['stride']==512 and not r['add_special_tokens'] and not r['chat_template']
  assert len(manifest['token_ids'])==r['input_tokens']
  mm={b['method']:b['ppl'] for b in baseline if b['dataset']==ds}
  comparison.append({'dataset':ds,'M1_ppl':mm['M1'],'M2_ppl':mm['M2'],'R_ppl':r['ppl'],'R_minus_M1':r['ppl']-mm['M1'],'R_vs_M1_relative_percent':100*(r['ppl']/mm['M1']-1),'R_minus_M2':r['ppl']-mm['M2'],'R_vs_M2_relative_percent':100*(r['ppl']/mm['M2']-1),'improved_vs_M2':r['ppl']<mm['M2'],'scored_tokens':nt,'windows':len(wr),'raw_window_NLL_reaggregation_pass':True})
 csvwrite('reports/PPL_COMPARISON.csv',comparison);return comparison

def partial_failure():
 result=[]
 for path in sorted(OUT.glob('CRITICAL_FAILURE*.pt')):
  f=torch.load(path,map_location='cpu',weights_only=False);st=f['state'];pr=f['proposal'];rr=pr['rows'];actual=f['direct_new_loss'][rr]-st['loss'][rr];pred=pr['pred'];tol=torch.maximum(st['loss'][rr]*2e-6,pred.abs()*1e-3)
  bad=(actual-pred).abs()>tol
  assert bool(torch.isfinite(f['direct_new_loss']).all())
  result.append({'path':str(path),'sha256':sha(path),'layer':3,'module':'mlp.down_proj','group_committed':False,'accepted_updates_in_uncommitted_state':int(st['updates'].sum()),'candidate_evaluations_in_uncommitted_state':int(st['evaluations'].sum()),'rejections_in_uncommitted_state':int(st['rejections'].sum()),'bad_rows':rr[bad].tolist(),'proposed_rows_in_failed_batch':rr.numel(),'failed_batch_not_accepted':True,'codes_legal':bool(((st['code']>=0)&(st['code']<=15)).all()),'counts_under_cap':bool((st['evaluations']<=128*st['code'].shape[1]).all()),'bad_predicted_delta':pred[bad].tolist(),'bad_direct_delta':actual[bad].tolist(),'bad_allowed_error':tol[bad].tolist(),'bad_absolute_error':(actual-pred).abs()[bad].tolist(),'bad_error_over_tolerance':((actual-pred).abs()/tol)[bad].tolist()})
 dump('reports/PARTIAL_FAILURE_AUDIT.json',{'uncommitted_failures':result,'excluded_from_committed_quality_and_coverage':True,'preserved_not_deleted':True});return result

def magnitude_summary(rejections):
 result={'count':len(rejections),'unique_rows':len(set((r['layer'],r['module'],r['row']) for r in rejections))}
 if rejections:
  for key in ['predicted_delta','direct_delta','relative_predicted_delta','relative_direct_delta']:
   v=torch.tensor([r[key] for r in rejections],dtype=torch.float64)
   result[key]={'minimum':float(v.min()),'median':float(v.median()),'maximum':float(v.max()),'maximum_absolute':float(v.abs().max())}
 return result

def main():
 t=time.perf_counter();status=read('RUN_STATUS.json');modules,rows,rejections=persisted_audit();comparison=ppl_audit();partial=partial_failure();complete=status['status']=='COMPLETE'
 if complete:assert len(modules)==168 and len(rows)==304128 and len(comparison)==2 and not partial
 else:assert not (OUT/'DEPLOYMENT_LOCK.json').exists() and not (OUT/'R/checkpoint/final/model.safetensors').exists() and not comparison
 category='D' if not complete else 'A' if all(r['improved_vs_M2'] for r in comparison) else 'B' if any(r['improved_vs_M2'] for r in comparison) else 'C'
 verdict={'A':'两语料PPL均改善，已有离散修复路线的本次迁移得到支持','B':'两语料方向不一致，本次迁移不稳定','C':'两语料均未改善','D':'正确性停止，完整模型与PPL质量未验证'}[category]
 checks=[json.loads(l) for l in (OUT/'CORRECTNESS.jsonl').read_text().splitlines()];assert all(r.get('pass_all',True) for r in checks)
 validation=read('VALIDATION_GATE.json');assert validation['pass']
 diag=read('NUMERICAL_DIAGNOSIS.json') if (OUT/'NUMERICAL_DIAGNOSIS.json').exists() else None
 if partial:assert diag and diag['pass_diagnosis'] and not diag['failed_group_checkpoint_exists']
 projection=read('COST_PROJECTION.json');times=csvread('STAGE_TIMINGS.csv');cost=defaultdict(float)
 for r in times:cost[(r['mode'],r['stage'])]+=float(r['seconds'])
 csvwrite('reports/COST_BILL.csv',[{'mode':m,'stage':s,'actual_synchronized_wall_seconds':v,'type':'measurement, not GPU kernel-only time'} for (m,s),v in sorted(cost.items())])
 modes={m:sum(v for (mm,s),v in cost.items() if mm==m) for m in set(m for m,s in cost)}
 blocks=csvread('BLOCK_TIMINGS.csv');layer=[]
 for li in sorted(set(r['layer'] for r in modules)):
  rr=[r for r in modules if r['layer']==li];a=sum(r['initial_search_mse_sum'] for r in rr);b=sum(r['final_search_mse_sum'] for r in rr)
  layer.append({'layer':li,'modules':len(rr),'rows':sum(r['rows'] for r in rr),'inherited_modules':sum(r['inherited_from_v43'] for r in rr),'accepted_updates':sum(r['accepted_updates'] for r in rr),'modified_coordinates':sum(r['modified_coordinates'] for r in rr),'actual_evaluations':sum(r['actual_candidate_evaluations'] for r in rr),'rejected_candidates':sum(r['rejected_candidates'] for r in rr),'initial_search_mse_sum':a,'final_search_mse_sum':b,'gain_percent':100*(a-b)/a})
 csvwrite('reports/LAYER_RESULTS.csv',layer)
 new=[r for r in modules if not r['inherited_from_v43']];inherited=[r for r in modules if r['inherited_from_v43']]
 initial=sum(r['initial_search_mse_sum'] for r in modules);final=sum(r['final_search_mse_sum'] for r in modules)
 labels=['NO_REMAINING_PREDICTED_DESCENT','EVALUATION_CAP','UPDATE_CAP']
 stopped={k:sum(r['status']==k for r in rows) for k in labels};stoppednew={k:sum(r['status']==k and not r['inherited_from_v43'] for r in rows) for k in labels}
 fixture=torch.load(OUT/'validation/FAILURE_FIXTURE_RESULT.pt',map_location='cpu',weights_only=False)
 fixturecounts={'accepted_updates':int(fixture['updates'].sum()),'candidate_evaluations':int(fixture['evaluations'].sum()),'rejections':int(fixture['rejections'].sum()),'test_accept_cap':1,'formal_cap':32,'quality_evidence':False};del fixture
 rejects=magnitude_summary(rejections)
 csvrejections=csvread('REJECTION_EVENTS.csv');formalrecords=[r for r in csvrejections if r['mode']=='formal'];assert len(formalrecords)==len(rejections)+sum(r['rejections_in_uncommitted_state'] for r in partial)
 dump('reports/NUMERICAL_INFORMATION.json',{'committed_formal_rejections':rejects,'formal_raw_rejections_including_uncommitted':len(formalrecords),'stop_counts_full_committed_model':stopped,'stop_counts_new_v44_committed_rows':stoppednew,'incomplete_rows_excluded_from_terminal_stop_counts':sum(r['rows'] for r in modules) != 304128,'incomplete_down_rows':896 if partial else 0,'fixture_counts_separate':fixturecounts,'material_error_observed':bool(partial),'material_error_details':diag,'candidate_error_tolerance_unchanged':True,'rejection_does_not_cover_actual_descending_but_tolerance_failure':True,'v43_flow_difference':'v43 aborts the Linear/model on any FP32 direct non-decrease; v44 restores and blacklists that row candidate within its current code state, while retaining other valid rows, and continues until no candidates/cap. Nonfinite/grid/restore/material tolerance failures still stop.','not_stepwise_equivalent_to_v43':True})
 globalwall=diag['original_gpu_budget_global_wall_seconds_including_diagnosis'] if diag else status['elapsed_seconds']
 historical42=projection['historical_v42_GPU_budget_wall_seconds'];historical43=projection['historical_v43_GPU_budget_wall_seconds'];cumulative=historical42+historical43+globalwall
 peak=max(int(r['gpu_peak_allocated_bytes']) for r in times);reserved=max(int(r['gpu_peak_reserved_bytes']) for r in times)
 failures=(OUT/'FAILED_ATTEMPTS.jsonl').read_text().splitlines() if (OUT/'FAILED_ATTEMPTS.jsonl').exists() else [];recoveries=(OUT/'RECOVERY_LOG.jsonl').read_text().splitlines() if (OUT/'RECOVERY_LOG.jsonl').exists() else []
 totals=lambda rr:{'modules':len(rr),'rows':sum(r['rows'] for r in rr),'accepted_updates':sum(r['accepted_updates'] for r in rr),'modified_coordinates':sum(r['modified_coordinates'] for r in rr),'actual_candidate_evaluations':sum(r['actual_candidate_evaluations'] for r in rr)}
 decision={'category':category,'verdict':verdict,'original_failure_status':status,'coverage':totals(modules),'new_v44_committed':totals(new),'reused_v43_committed':totals(inherited),'counts_scope':'committed groups only; validation and failed down group are separate','local_reconstruction_gain_percent':100*(initial-final)/initial,'committed_rejections':rejects,'stop_counts_committed_rows':stopped,'incomplete_states':partial,'numerical_diagnosis':diag,'PPL_comparison':comparison,'PPL_unverified':not complete,'v44_attempt_wall_seconds':status['elapsed_seconds'],'v44_gpu_budget_global_wall_seconds_including_diagnosis':globalwall,'v44_synchronized_stage_seconds_by_mode':modes,'v44_all_synchronized_stages_seconds':sum(cost.values()),'v44_unitemized_wall_seconds':globalwall-sum(cost.values()),'v42_GPU_budget_wall_seconds':historical42,'v42_pilot_block_seconds_subset_of_v42_wall':projection['historical_v42_pilot_block_seconds'],'v43_GPU_budget_wall_seconds_includes_failure':historical43,'v43_failed_repair_attempt_wall_seconds':projection['historical_v43_repair_attempt_wall_seconds'],'cumulative_v42_v43_v44_GPU_budget_wall_seconds':cumulative,'historical_M2_build_seconds_separate':projection['historical_M2_build_seconds'],'M2_plus_repair_validation_chain_wall_seconds':cumulative+projection['historical_M2_build_seconds'],'peak_GPU_allocated_bytes':peak,'peak_GPU_reserved_bytes':reserved,'v44_failures':len(failures),'v44_recoveries':len(recoveries),'pre_run_forecast_seconds_prediction_only':projection['conservative_total_projection_seconds'],'GPU_hard_cap_seconds':10800,'historical_v43_preserved':True,'automatic_followup':False,'model_setting':'Sol high requested; interface cannot confirm or switch.'}
 dump('DECISION.json',decision);dump('reports/CUMULATIVE_COST.json',{k:v for k,v in decision.items() if 'seconds' in k or 'stage' in k or 'peak_' in k})
 text=f'''# v44 REPORT

## 结果与停止判断

**{category}：{verdict}。** 正式已提交 {len(modules)}/168 个 Linear、{len(rows):,}/304,128 个输出行，其中复用 v43 的25模块/40,064行，本次新完成 {len(new)} 模块/{sum(r['rows'] for r in new):,} 行。完整模型未完成，未生成最终R checkpoint、部署就绪锁或R的WT2/C4 PPL。历史正式M1/M2基线已核查，但R的相对质量变化尚不可计算。

v43的三个失败候选按原完整4864行FP32批量路径仍被拒绝；拒绝行code/p/loss/接受数逐位保留，其他4,827行合法更新通过。一状态内禁止重复被拒坐标＋offset，接受后才清除该行拒绝集合，评价数不重置。8项CPU确定性测试通过。固定失败状态回归使用测试上限1次接受，仅验证处理流程；正式修复仍是32次上限。原部分v43计数未存储，回归从0计数，不称为原轨迹完整恢复。

恢复边界输入SHA及C SHA与v43相同；全部保存张量键核对证明25个已提交权重之外的参数仍等于原M2。正式从原M2重建，复用已提交组，失败up/gate从原M2码重新开始。该组两模块已完整修复并提交，随后在自身修复前缀产生的down输入上触发新的停止。

'''
 if diag:
  d=diag['rows'][0]
  text+=f'''block3 down_proj第2次候选批次，第{d['row']}行、坐标{d['coordinate']}、offset={d['offset']}：预测Δ={d['saved_prediction']:.12g}，原FP32批量直接Δ={d['saved_FP32_direct_delta']:.12g}，两者都下降；绝对差{d['absolute_discrepancy']:.12g}，超过原容差{d['original_tolerance']:.12g}（{d['error_over_original_tolerance']:.6f}倍）。这不是允许拒绝后继续的“FP32持平/上升”分支，而是原误差检查失败。未放宽容差、未把实际下降作为绕过误差检查的理由、未提交该批或down组。

独立诊断重放同一前缀，核对input/C哈希、整数码及实际网格E/proposal精确一致，无非有限值。原完整896行FP32批量旧/新loss逐位复现；p重算最大漂移{diag['p_max_absolute_drift']:.6g}，该候选新预测−保存预测={d['fresh_minus_saved_prediction']:.6g}。仅用于定位的FP64配对输出增量与新预测差{d['FP64_paired_minus_prediction']:.6g}。证据指向FP32直接累加与二次型增量之间的差异，而非码/网格恢复或p漂移；诊断精度不用于接受或继续运行。用户规定超过原容差仍停止，因此本轮以D结束。

'''
 text+='## 正式质量基线\n\n'
 for ds in ['wikitext','c4_fixed_budget']:
  mm={r['method']:r['ppl'] for r in read('reports/BASELINE_AUDIT.json')['baselines'] if r['dataset']==ds}
  text+=f"- {ds}：M1={mm['M1']:.8f}，M2={mm['M2']:.8f}；R未测量，R−M1/M2未验证。\n"
 text+=f'''
## 明确的v43流程差异

原v43遇到任一预测下降但FP32直接持平/上升候选，就终止Linear及运行。v44仅在原误差容差通过时，拒绝该候选、恢复该行tentative code/E，保持p/loss/接受数，黑名单当前状态的坐标＋offset，继续其他候选；其他行合法更新独立保留。该行下一次接受后清空黑名单；全部重新评价计账，维持128×输入维度上限。无剩余预测下降候选、评价上限或32次接受上限分别结束行。非有限值、网格/恢复错误、超过原容差的数值不一致仍停止。v44是显式流程修订，不称与v43原算法逐步等价。v43原协议、NUMERICAL_STOP、失败和诊断文件及14个组的来源哈希均保持不变。

评分仍为原FP32完整输出行批量matmul，4096-token流式块、FP64 SSE累积；未换单行matvec、FP64接受评分或批量形状。候选预测与p为未damping独立全FP64 C；实际δ来自FP16网格值差。原预测阈值max(1e−14,1e−8×原始本行直接loss)、误差容差max(2e−6×当前loss,1e−3×|预测Δ|)、实际直接严格下降、首次最小值tie-breaking（坐标升序、−1再+1）不变。完整邻域容纳不下剩余预算时按原顺序实际评分到预算，停止该行，不以部分邻域最小值冒充全邻域选择。

## 覆盖、拒绝和正确性

已提交模块接受 {sum(r['accepted_updates'] for r in modules):,} 次修改、终态改变 {sum(r['modified_coordinates'] for r in modules):,} 个码位、候选评价 {sum(r['actual_candidate_evaluations'] for r in modules):,} 次，含复用v43原始计数。本次新提交部分单列在DECISION.json：{sum(r['accepted_updates'] for r in new):,}次接受、{sum(r['actual_candidate_evaluations'] for r in new):,}次评价。固定各组自身输入的局部MSE合计 {initial:.12g}→{final:.12g}（下降{100*(initial-final)/initial:.4f}%），仅是已提交组诊断，不能代表完整模型目标或PPL。

已提交正式候选拒绝 {rejects['count']} 次，服务于 {rejects['unique_rows']} 个不同模块行。预测下降绝对量约2.21e−11至1.49e−10，相对当前loss约1.19e−8至7.21e−8；原FP32直接上升约2.83e−12至6.84e−11，相对loss约1.28e−9至4.14e−8。逐个Δ及相对当前loss量级见reports/REJECTION_EVENTS.csv和NUMERICAL_INFORMATION.json。停止计数（仅已完成行）：{json.dumps(stopped,ensure_ascii=False)}。未完成down的896行不冒充终止行；第1批已接受896次、失败第2批候选也已计账，状态单独保存在CRITICAL_FAILURE和PARTIAL_FAILURE_AUDIT，不计入已提交覆盖/质量。回归测试评价/接受与正式计数分开。

CPU审计逐次重放接受修改、拒绝集合及清除、实际重新评价和上限，核对终态整数码、pack/unpack和FP16权重逐位一致，scale/zero/g_idx与原M2不变。全局168模块零修改部署证据来自v42只读继承；本次已提交组再次落盘核查。完整模型保存/reload及最终输出/PPL链未执行，不把已有packing局部证据冒充最终模型部署通过。辅助u4 packing用于码存储；部署协议仍是原dense FP16/SDPA。逐行、逐模块、逐层、原始输入/C哈希和提交状态均保留。

## 成本：预测与实测

启动前保守预测为21个剩余完整块上界×v42实测block0 {projection['v42_measured_block0_seconds']:.6f}秒×原安全系数1.35＋准备30＋验证120＋收尾360 = {projection['conservative_total_projection_seconds']:.6f}秒（{projection['conservative_total_projection_seconds']/60:.4f}分钟）；这是预测，非本次实测。真实形状仍为q/o 896×896、k/v 128×896、up/gate 4864×896、down 896×4864，24块共168模块，不按代表行倍乘。不降低安全系数、不减更新次数。180分钟硬上限保持。

本次失败修复作业墙时 {status['elapsed_seconds']:.3f}秒；诊断结束时同一原始预算时钟累计 {globalwall:.3f}秒，包含CPU调度/记录及失败到诊断间隔。同步阶段实测总计 {sum(cost.values()):.3f}秒，其中 {json.dumps(modes,ensure_ascii=False)}；未分项墙时 {globalwall-sum(cost.values()):.3f}秒。同步阶段墙时包含CPU操作/IO及GPU同步，并非纯GPU kernel时间。C构建、单点候选评分、接受/恢复、直接复算、提交、依赖刷新、packing/IO分项见原始STAGE_TIMINGS.csv及reports/COST_BILL.csv。PPL与最终保存/reload本次未执行，不能用预测数冒充实测。

验证链历史已发生成本：v42总预算墙时 {historical42:.3f}秒（其中block0 pilot {projection['historical_v42_pilot_block_seconds']:.3f}秒，子项不重复相加）；v43总预算墙时 {historical43:.3f}秒（含失败尝试 {projection['historical_v43_repair_attempt_wall_seconds']:.3f}秒及诊断/间隔）；v44 {globalwall:.3f}秒。v42+v43+v44累计 {cumulative:.3f}秒（{cumulative/60:.3f}分钟）。原M2构建 {projection['historical_M2_build_seconds']:.3f}秒另外列出，加原M2总链成本 {(cumulative+projection['historical_M2_build_seconds'])/60:.3f}分钟。完整新增修复成本尚未测完；所有失败成本保留，候选评价次数不等于运行时间或GPTQ完整Linear执行数。

峰值allocated {peak/2**30:.3f}GiB、reserved {reserved/2**30:.3f}GiB。v44失败{len(failures)}次、失败后正式恢复{len(recoveries)}次；未自动恢复误差停止，前缀重放仅为诊断。

## 推进判断与交付

拒绝流程在固定失败状态和本次正式up/gate中通过，但完整模型遭遇原误差容差停止。因此不能判断完整PPL迁移；本轮停止，不推进质量扩展、不自动调整容差/评分路径、不启动下一版本。这是已有离散修复思路的应用验证，不是新论文方法、不是官方ReQuant复现；未读取P3，未根据PPL调参，未生成第二个模型。局部MSE下降不保证PPL下降，也不作PPL预测或跨seed稳定性结论。

冻结PROTOCOL、原字节run_v42/audit_sources、v44修复实现/协调器、CPU测试、失败状态验证、来源/恢复边界哈希、formal/groups checkpoint、逐行/模块/层统计、数值拒绝与误差诊断、成本账单、DECISION与复现命令均位于本独立目录。v43历史状态保持NUMERICAL_STOP_FROZEN_RULE。用户要求Sol high，接口无法确认或切换；未声称已确认设置。
'''
 (OUT/'REPORT.md').write_text(text,encoding='utf8');(OUT/'reports/REPORT.md').write_text(text,encoding='utf8')
 dump('DELIVERY_MANIFEST.json',{'files':[{'path':str(p.relative_to(OUT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file() and 'original_codes' not in p.parts and 'formal' not in p.parts and p.name not in ['DELIVERY_MANIFEST.json','finalize.log']],'group_checkpoint_hashes_location':'reports/OUTPUT_AUDIT.json','CPU_final_audit_seconds':time.perf_counter()-t,'category':category})
 print(json.dumps(decision,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
