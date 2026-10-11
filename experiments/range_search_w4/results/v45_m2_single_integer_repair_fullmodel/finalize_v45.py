"""CPU-only final persisted state-machine, baseline/window, lock and cost audit."""
import ast,csv,json,math,time
from collections import Counter,defaultdict
from pathlib import Path
import torch
from safetensors import safe_open
from audit_sources import OUT,V18,sha,th,pack,unpack
import audit_helpers as ah
from precision_check import gamma
V44=OUT.parent/'v44_m2_single_integer_repair_fullmodel'
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
def verify_v45(s,loss,rr,pred,actual,accept,attempt,refs):
 threshold=torch.maximum(s['initial_mse'][rr]*1e-8,torch.full_like(pred,1e-14))
 assert bool(torch.isfinite(pred).all() and torch.isfinite(actual).all() and (pred < -threshold).all())
 if accept:assert bool((actual<0).all())
 errors=(actual-pred).abs();tols=torch.maximum(loss[rr]*2e-6,pred.abs()*1e-3)
 special=(actual<0)&(errors>tols)
 present={row for (roundnum,row) in refs if roundnum==attempt}
 assert present&set(rr.tolist())==set(rr[special].tolist())
 if accept:assert bool(((errors<=tols)|special).all())
 for j in special.nonzero().flatten().tolist():
  row=int(rr[j])
  error=abs(float(actual[j]-pred[j]));tol=max(float(loss[row])*2e-6,abs(float(pred[j]))*1e-3);ref=refs.get((attempt,row))
  if float(actual[j])>=0:
   assert not accept and ref is None
  elif error<=tol:
   assert accept and ref is None
  else:
   assert ref is not None and ref['same_precision_pass'] and ref['accepted_after_FP64']==accept
   assert ref['predict64']==float(pred[j]) and ref['direct_FP32_delta']==float(actual[j]) and ref['old_loss']==float(loss[row])
   assert ref['same_precision_absolute_error']<=ref['same_precision_allowed_error']
   assert abs(ref['predict64']-ref['direct64_from_raw_X'])==ref['same_precision_absolute_error']
   assert (ref['direct64_from_raw_X']<0)==accept
   assert ref['relative_tolerance']==32*gamma(s['codes'].shape[1]+262144+64)
   lower=ref['relative_tolerance']*(ref['operation_scale']+max(abs(ref['predict64']),abs(ref['direct64_from_raw_X'])))
   assert ref['same_precision_allowed_error']>=lower*(1-1e-12)
   assert ref['cache_max_error_over_tolerance']<=1

def replay_v45(s,o):
 # Exact v44 replay bookkeeping; only event numerical checks use frozen v45 branches.
 code=o['codes'].long().clone();rows,d=code.shape;cap=128*d
 counts=torch.zeros(rows,dtype=torch.long);updates=counts.clone();rejections=counts.clone();status=counts.clone();loss=s['initial_mse'].clone()
 blocked=torch.zeros((rows,d,2),dtype=torch.bool)
 moves={m['round']:m for m in s['moves']};events={e['attempt']:e for e in s['rejection_events']}
 refs={(r['attempt'],r['row']):r for r in s['recheck_events']};assert len(refs)==len(s['recheck_events'])
 assert len(moves)==len(s['moves']) and len(events)==len(s['rejection_events'])
 attempts=set();rejection_rows=[]
 for sp in s['score_passes']:
  attempt=sp['attempt'];assert attempt not in attempts;attempts.add(attempt)
  status[(status==0)&(updates>=32)]=2;status[(status==0)&(counts>=cap)]=3
  active=status==0;legal=torch.stack((code>0,code<15),-1);eligible=(legal&~blocked).sum((1,2));remaining=cap-counts
  expected=torch.where(active,torch.minimum(eligible,remaining),torch.zeros_like(counts));assert torch.equal(expected,sp['evaluations_added'])
  counts+=expected;after=sp['status_after_scoring'];assert bool(((after>=0)&(after<=3)).all()) and torch.equal(after[~active],status[~active])
  assert bool((after[active&(eligible==0)]==1).all() and (after[active&(eligible>remaining)]==3).all())
  m=moves.get(attempt);e=events.get(attempt)
  accepted=m['rows'] if m is not None else torch.empty(0,dtype=torch.long);rejected=e['rows'] if e is not None else torch.empty(0,dtype=torch.long)
  proposed=torch.cat((accepted,rejected));assert proposed.unique().numel()==proposed.numel() and torch.equal(torch.sort(proposed).values,(after==0).nonzero().flatten())
  for obj,rr,is_accept in [(m,accepted,True),(e,rejected,False)]:
   if obj is None:continue
   col,off=obj['columns'],obj['offsets'];assert bool(((off==-1)|(off==1)).all())
   kk=(off+1)//2;assert not bool(blocked[rr,col,kk].any()) and bool(legal[rr,col,kk].all()) and torch.equal(obj['evaluations'],counts[rr])
   verify_v45(s,loss,rr,obj['predicted_delta'],obj['direct_delta'],is_accept,attempt,refs)
   if is_accept:
    code[rr,col]+=off;loss[rr]+=obj['direct_delta'];updates[rr]+=1;blocked[rr]=False;assert torch.equal(obj['updates_after'],updates[rr])
   else:
    assert obj['restored'] and torch.equal(obj['old_loss'],loss[rr]) and torch.equal(obj['state_accept_count'],updates[rr])
    blocked[rr,col,kk]=True;rejections[rr]+=1
    for k,row in enumerate(rr.tolist()):
     old=float(loss[row]);pred=float(obj['predicted_delta'][k]);actual=float(obj['direct_delta'][k])
     rejection_rows.append({'attempt':attempt,'row':row,'coordinate':int(col[k]),'offset':int(off[k]),'state_accept_count':int(updates[row]),'evaluations':int(counts[row]),'predicted_delta':pred,'direct_delta':actual,'old_loss':old,'relative_predicted_delta':pred/old if old else 0,'relative_direct_delta':actual/old if old else 0,'reason':obj['reasons'][k],'restored':True})
  status=after.clone();assert bool(((code>=0)&(code<=15)).all() and (counts<=cap).all() and (updates<=32).all())
 status[(status==0)&(updates>=32)]=2;status[(status==0)&(counts>=cap)]=3
 assert set(moves)|set(events)<=attempts
 assert torch.equal(code.byte(),s['codes']) and torch.equal(counts,s['evaluations']) and torch.equal(updates,s['updates']) and torch.equal(rejections,s['rejections']) and torch.equal(status,s['status'])
 assert not bool((status==0).any()) and bool(torch.isclose(loss,s['final_mse'],rtol=2e-6,atol=1e-12).all())
 return counts,updates,rejections,rejection_rows

def persisted_audit():
 for fn,h in read('IMPLEMENTATION_LOCK.json').items():assert sha(OUT/fn)==h
 start=read('GPU_START.json')
 for fn,key in [('PROTOCOL.json','protocol_sha256'),('precision_check.py','precision_sha256'),('run_v42.py','legacy_runtime_sha256')]:assert sha(OUT/fn)==start[key]
 # Verify extracted CPU helper bodies exactly equal their read-only source bodies.
 source=(V44/'finalize_v44.py').read_text();local=(OUT/'audit_helpers.py').read_text()
 bodies=lambda s:{n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
 a,b=bodies(source),bodies(local)
 for name in ['verify_delta','replay_legacy','replay_revision','ppl_audit']:assert a[name]==b[name]
 audit=read('SOURCE_AUDIT.json');reuse=read('SOURCE_REUSE_AUDIT.json')
 for s in audit['sources']:assert sha(s['source'])==s['source_sha256'] and sha(s['derived'])==s['derived_sha256']
 assert sha(V18/'M2/checkpoint/final/model.safetensors')==audit['model_sha256'] and sha(V18/'CALIBRATION_MANIFEST.json')==reuse['calibration_sha256']
 for s in reuse['historical_watched']:assert sha(s['path'])==s['sha256']
 for s in reuse['inherited_groups']:assert sha(s['source_path'])==s['source_sha256']
 modules=[];rowrecords=[];rejectrecords=[];refs=[];locks=[]
 for path in sorted((OUT/'formal/groups').glob('*.pt')):
  ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['committed'] and ck['protocol_sha256']==start['protocol_sha256']
  orig=torch.load(OUT/f'original_codes/layer_{ck["layer"]:02d}.pt',map_location='cpu',weights_only=True);inherited=ck.get('reused_without_new_repair',False)
  if inherited:
   assert sha(ck['source_v44_path'])==ck['source_v44_sha256'];old=torch.load(ck['source_v44_path'],map_location='cpu',weights_only=False);assert ck['input_sha256']==old['input_sha256'] and ck['C_sha256']==old['C_sha256']
  for name,s in ck['modules'].items():
   o=orig[name];c=s['codes'];assert bool((c<=15).all()) and torch.equal(pack(c),s['packed_u4']) and torch.equal(unpack(s['packed_u4']),c)
   for key in ['scales','zeros','g_idx']:assert torch.equal(s[key],o[key])
   q=(s['scales'][:,s['g_idx'].long()]*(c.float()-8)).half();assert torch.equal(q.view(torch.int16),s['weight'].view(torch.int16))
   assert bool(torch.isfinite(s['initial_mse']).all() and torch.isfinite(s['final_mse']).all() and (s['final_mse']<=s['initial_mse']).all())
   if inherited:
    for key in ['codes','weight','scales','zeros','g_idx','updates','evaluations','status','initial_mse','final_mse']:assert torch.equal(s[key],old['modules'][name][key])
    if 'score_passes' in s:counts,updates,rejects,recs=ah.replay_revision(s,o)
    else:counts,updates,rejects=ah.replay_legacy(s,o);recs=[]
   else:
    counts,updates,rejects,recs=replay_v45(s,o);refs+=s['recheck_events']
   rejectrecords.extend({'layer':ck['layer'],'module':name,'inherited':inherited,**r} for r in recs)
   n,d=c.shape;initial=float(s['initial_mse'].sum());final=float(s['final_mse'].sum())
   modules.append({'layer':ck['layer'],'module':name,'inherited_from_v44':inherited,'rows':n,'input_dim':d,'accepted_updates':int(updates.sum()),'modified_coordinates':int((c!=o['codes']).sum()),'actual_candidate_evaluations':int(counts.sum()),'rejected_candidates':int(rejects.sum()),'rows_with_rejection':int((rejects>0).sum()),'FP64_recheck_candidates':len(s.get('recheck_events',[])),'FP64_recheck_accepted':sum(r['accepted_after_FP64'] for r in s.get('recheck_events',[])),'initial_search_mse_sum':initial,'final_search_mse_sum':final,'search_gain_percent':100*(initial-final)/initial,'initial_pooled_SSE':initial*262144,'final_pooled_SSE':final*262144,'max_p_drift':s['max_p_drift'],'codes_sha256':th(c),'scale_sha256':th(s['scales']),'zero_sha256':th(s['zeros']),'g_idx_sha256':th(s['g_idx']),'input_sha256':ck['input_sha256'],'C_sha256':ck['C_sha256']})
   for i in range(n):rowrecords.append({'layer':ck['layer'],'module':name,'inherited_from_v44':inherited,'row':i,'input_dim':d,'initial_search_mse':float(s['initial_mse'][i]),'final_search_mse':float(s['final_mse'][i]),'accepted_updates':int(updates[i]),'evaluations':int(counts[i]),'eval_cap':128*d,'rejections':int(rejects[i]),'status':{1:'NO_REMAINING_PREDICTED_DESCENT',2:'UPDATE_CAP',3:'EVALUATION_CAP'}[int(s['status'][i])]})
  locks.append({'path':str(path),'sha256':sha(path),'layer':ck['layer'],'group':ck['group'],'inherited_from_v44':inherited,'input_sha256':ck['input_sha256'],'C_sha256':ck['C_sha256']})
 csvwrite('reports/MODULE_RESULTS.csv',modules);csvwrite('reports/ROW_RESULTS.csv',rowrecords);csvwrite('reports/REJECTION_EVENTS.csv',rejectrecords);csvwrite('reports/NUMERICAL_RECHECKS.csv',refs)
 labels=['NO_REMAINING_PREDICTED_DESCENT','EVALUATION_CAP','UPDATE_CAP'];stops={k:sum(r['status']==k for r in rowrecords) for k in labels}
 dump('reports/OUTPUT_AUDIT.json',{'pass':True,'modules':len(modules),'rows':len(rowrecords),'grid_unchanged':True,'codes_and_budget_replayed':True,'rejection_blacklist_and_state_restoration_replayed':True,'same_precision_rechecks_linked_to_accepted_or_rejected_moves':True,'FP32_strict_descent_all_accepts':True,'historical_source_hashes_unchanged':True,'stop_counts_committed_rows':stops,'checkpoints':locks})
 return modules,rowrecords,rejectrecords,refs

def final_deployment_audit(complete):
 if not complete:return {'pass':False,'not_reached':True}
 lock=read('DEPLOYMENT_LOCK.json');assert lock['configuration_pass'] and lock['reload_output_pass']
 assert sha(OUT/'R/checkpoint/final/model.safetensors')==lock['final_model_sha256'] and sha(OUT/'R/checkpoint/final/config.json')==lock['config_sha256']
 for rec in lock['group_checkpoints']:assert sha(rec['path'])==rec['sha256']
 old=json.loads((V18/'M2/checkpoint/final/config.json').read_text());new=json.loads((OUT/'R/checkpoint/final/config.json').read_text());ignore={'_name_or_path','transformers_version'}
 assert {k:v for k,v in old.items() if k not in ignore}=={k:v for k,v in new.items() if k not in ignore}
 untouched=0;repaired=0
 with safe_open(str(V18/'M2/checkpoint/final/model.safetensors'),framework='pt',device='cpu') as sf0,safe_open(str(OUT/'R/checkpoint/final/model.safetensors'),framework='pt',device='cpu') as sf:
  lookup={}
  for path in sorted((OUT/'formal/groups').glob('*.pt')):
   ck=torch.load(path,map_location='cpu',weights_only=False)
   for name,s in ck['modules'].items():lookup[f'model.layers.{ck["layer"]}.{name}.weight']=s['weight']
  assert set(sf.keys())==set(sf0.keys())
  for key in sf.keys():
   actual=sf.get_tensor(key);expected=lookup[key] if key in lookup else sf0.get_tensor(key)
   assert torch.equal(actual.view(torch.int16),expected.view(torch.int16)),key
   if key in lookup:repaired+=1
   else:untouched+=1
 assert repaired==168
 checks=[json.loads(l) for l in (OUT/'CORRECTNESS.jsonl').read_text().splitlines()]
 regression=[r for r in checks if r['kind']=='final_reload']
 assert len(regression)==1 and regression[0]['pass_all']
 result={'pass':True,'repaired_Linear_parameters':repaired,'untouched_model_parameters_verified':untouched,'dense_FP16_weight_grid_bitwise':True,'configuration_unchanged':True,'reload_output_regression':regression[0],'final_model_sha256':lock['final_model_sha256'],'locked_before_PPL':True}
 dump('reports/DEPLOYMENT_AUDIT.json',result);return result

def main():
 t=time.perf_counter();status=read('RUN_STATUS.json');complete=status['status']=='COMPLETE'
 modules,rows,rejections,refs=persisted_audit();comparisons=ah.ppl_audit();deployment=final_deployment_audit(complete)
 if complete:assert len(modules)==168 and len(rows)==304128 and len(comparisons)==2
 category='D' if not complete else 'A' if all(r['improved_vs_M2'] for r in comparisons) else 'B' if any(r['improved_vs_M2'] for r in comparisons) else 'C'
 verdict={'A':'两语料PPL均改善，本次完整模型质量迁移得到支持','B':'两语料方向不一致，本次质量迁移不稳定','C':'两语料均未改善','D':'完整修复或评测未完成，质量未验证'}[category]
 times=csvread('STAGE_TIMINGS.csv');cost=defaultdict(float)
 for r in times:cost[(r['mode'],r['stage'])]+=float(r['seconds'])
 inclusive=dict(cost)
 nested=cost.get(('formal','independent_FP64_increment_recheck'),0)
 # v45 verification is nested inside the row transaction timer. Keep raw inclusive
 # measurements, but subtract the child from the parent for additive stage totals.
 if nested:
  cost[('formal','accepted_update_and_rejection_restore')]-=nested
  assert cost[('formal','accepted_update_and_rejection_restore')]>=-1e-6
 csvwrite('reports/COST_BILL.csv',[{'mode':m,'stage':s,'actual_inclusive_synchronized_wall_seconds':inclusive[(m,s)],'actual_exclusive_synchronized_wall_seconds':v,'type':'measurement; child verification removed from parent for additive total'} for (m,s),v in sorted(cost.items())])
 modes={m:sum(v for (mm,s),v in cost.items() if mm==m) for m in set(m for m,s in cost)}
 stage_recheck=cost.get(('formal','independent_FP64_increment_recheck'),0)
 formalblocks=sum(float(r['seconds']) for r in csvread('BLOCK_TIMINGS.csv') if int(r['new_groups'])>0)
 new=[r for r in modules if not r['inherited_from_v44']];inherited=[r for r in modules if r['inherited_from_v44']]
 initial=sum(r['initial_search_mse_sum'] for r in modules);final=sum(r['final_search_mse_sum'] for r in modules)
 totals=lambda rr:{'modules':len(rr),'rows':sum(r['rows'] for r in rr),'accepted_updates':sum(r['accepted_updates'] for r in rr),'modified_coordinates':sum(r['modified_coordinates'] for r in rr),'candidate_evaluations':sum(r['actual_candidate_evaluations'] for r in rr),'rejected_candidates':sum(r['rejected_candidates'] for r in rr)}
 layers=[]
 for li in sorted(set(r['layer'] for r in modules)):
  rr=[r for r in modules if r['layer']==li];a=sum(r['initial_search_mse_sum'] for r in rr);b=sum(r['final_search_mse_sum'] for r in rr)
  layers.append({'layer':li,**totals(rr),'initial_search_mse_sum':a,'final_search_mse_sum':b,'gain_percent':100*(a-b)/a})
 csvwrite('reports/LAYER_RESULTS.csv',layers)
 rawrefs=csvread('NUMERICAL_RECHECKS.csv') if (OUT/'NUMERICAL_RECHECKS.csv').exists() else []
 if complete:assert len(rawrefs)==len(refs),'raw rechecks must match committed event records for complete run'
 else:assert len(rawrefs)>=len(refs)
 rejected_reasons=Counter(r.get('reason','FP32_NON_DESCENT') for r in rejections)
 new_rejection_count=sum(r['rejected_candidates'] for r in new)
 failures=(OUT/'FAILED_ATTEMPTS.jsonl').read_text().splitlines() if (OUT/'FAILED_ATTEMPTS.jsonl').exists() else []
 recoveries=(OUT/'RECOVERY_LOG.jsonl').read_text().splitlines() if (OUT/'RECOVERY_LOG.jsonl').exists() else []
 critical=list(OUT.glob('CRITICAL_FAILURE*.pt'))
 checks=[json.loads(l) for l in (OUT/'CORRECTNESS.jsonl').read_text().splitlines()]
 if complete:assert not critical and not failures and all(r.get('pass_all',True) for r in checks)
 numerical={'formal_recheck_candidates':len(refs),'formal_recheck_batches':sum(r['mode']=='formal' and r['stage']=='independent_FP64_increment_recheck' for r in times),'formal_recheck_accepts':sum(r['accepted_after_FP64'] for r in refs),'formal_recheck_rejects':sum(not r['accepted_after_FP64'] for r in refs),'formal_recheck_actual_synchronized_seconds':stage_recheck,'new_v45_candidate_rejects':new_rejection_count,'inherited_rejects':sum(r['rejected_candidates'] for r in inherited),'all_committed_reject_reasons':dict(rejected_reasons),'genuine_correctness_failure_count':len(critical),'stopped_process_count':len(failures),'maximum_same_precision_error_over_allowed':max((r['same_precision_absolute_error']/r['same_precision_allowed_error'] for r in refs),default=0),'maximum_FP32_difference_minus_predict_magnitude':max((r['old_cross_precision_error'] for r in refs),default=0),'criterion_frozen_before_formal':read('NUMERICAL_PROCESS_LOCK.json'),'criterion_empirical_not_strict_theorem':True,'FP32_strict_decline_never_overridden':True,'stop_counts_committed_rows':read('reports/OUTPUT_AUDIT.json')['stop_counts_committed_rows'],'initial_failure_diagnosis':read('validation/V44_DIAGNOSIS.json'),'CPU_test_count':len(read('CPU_TESTS.json')['checks'])}
 dump('reports/NUMERICAL_INFORMATION.json',numerical)
 proj=read('COST_PROJECTION.json');historical={k:proj[k] for k in ['historical_v42_GPU_budget_wall_seconds','historical_v43_GPU_budget_wall_seconds','historical_v44_GPU_budget_wall_seconds']}
 newwall=status['elapsed_seconds'];chain=sum(historical.values())+newwall
 peak=max(int(r['gpu_peak_allocated_bytes']) for r in times);reserved=max(int(r['gpu_peak_reserved_bytes']) for r in times)
 decision={'category':category,'verdict':verdict,'status':status,'coverage':totals(modules),'new_v45':totals(new),'inherited_v44':totals(inherited),'local_search_MSE_sum_gain_percent':100*(initial-final)/initial,'PPL_comparison':comparisons,'numerical_rechecks':numerical,'deployment':deployment,'actual_v45_GPU_budget_wall_seconds':newwall,'actual_v45_synchronized_stage_seconds_by_mode':modes,'actual_formal_blocks_new_groups_seconds':formalblocks,'actual_formal_increment_recheck_seconds':stage_recheck,'actual_stage_sum_seconds':sum(cost.values()),'raw_inclusive_stage_sum_seconds_contains_nested_verification':sum(inclusive.values()),'nested_verification_seconds_removed_from_parent':nested,'unitemized_wall_seconds':newwall-sum(cost.values()),**historical,'historical_v42_pilot_block_seconds_subset_of_v42_wall':proj['historical_v42_pilot_block_seconds'],'v42_to_v45_cumulative_GPU_budget_wall_seconds':chain,'historical_M2_creation_seconds_separate':proj['historical_M2_build_seconds'],'M2_plus_validation_chain_wall_seconds':chain+proj['historical_M2_build_seconds'],'pre_run_projection_seconds_prediction_only':proj['conservative_total_projection_seconds'],'peak_GPU_allocated_bytes':peak,'peak_GPU_reserved_bytes':reserved,'failed_attempts':len(failures),'formal_recoveries':len(recoveries),'no_automatic_followup':True,'model_setting':'Sol high request retained; interface cannot confirm or switch.'}
 dump('DECISION.json',decision);dump('reports/CUMULATIVE_COST.json',{k:v for k,v in decision.items() if 'seconds' in k or 'stage_' in k or 'peak_' in k})
 text=f'''# v45 REPORT

## 结果与推进判断

**{category}：{verdict}。** 覆盖 {len(modules)}/168 个Transformer Linear、{len(rows):,}/304,128 个输出行。其中复用v44已提交27模块/49,792行，本次新增{len(new)}模块/{sum(r['rows'] for r in new):,}行，不重复修复已提交组。

'''
 for r in comparisons:text+=f"- {r['dataset']}：M1={r['M1_ppl']:.8f}，M2={r['M2_ppl']:.8f}，R={r['R_ppl']:.8f}。相对M2：{r['R_minus_M2']:+.8f}（{r['R_vs_M2_relative_percent']:+.4f}%）；相对M1：{r['R_minus_M1']:+.8f}（{r['R_vs_M1_relative_percent']:+.4f}%）。\n"
 text+='\n'+('两语料同方向改善；该固定运行支持已有单点修复路线的完整模型迁移。' if category=='A' else '两语料方向不一致；本次完整模型迁移不稳定。' if category=='B' else '两语料均未改善；本次完整模型迁移没有质量支持。' if category=='C' else '完整质量验证未完成，不能用局部MSE或诊断通过替代PPL。')+'\n'
 text+=f'''
## v44失败定位与冻结数值流程

从原M2装入15个v44已提交组，重放自身前缀。恢复边界X哈希与C哈希一致，完整失败状态的code、实际FP16网格值、pristine BF16→FP16目标W、e、C、p和实际δ逐项检查。全部dtype及原始X/N见validation/V44_DIAGNOSIS.json。参考增量只使用原始X：a=X.float64@e，d=X.float64@实际dense δ，分块累积mean(2*a*d+d*d)，不借助C/p，不使用两个FP32 loss相减作正确性参考。原失败批次896个候选全部通过同精度验证。第211行独立Δ64=−9.852102460456627e−7，预测−9.852102460456653e−7，误差2.54e−21；原FP32两个loss相减仍为−9.84100198216819e−7。独立复核保留了全部失败用例和跨精度差异。

正式恢复前冻结PROTOCOL和NUMERICAL_PROCESS_LOCK，之后没有逐次调阈值。搜索仍是原二次型argmin、原预测下降阈值、±1单坐标、坐标升序/−1先+1、128×输入维度评价上限、最多32次接受，批量形状和4096-token FP32直接评分路径不变：

1. FP32直接评分持平或上升：拒绝，恢复该行code/E，保持p/loss/接受计数，黑名单当前状态的坐标＋offset，继续其他候选。
2. FP32直接评分严格下降且原误差检查通过：按原规则接受。
3. FP32严格下降但原跨精度误差检查不通过：从原始X独立FP64增量、完整δ公式及p缓存核验；同精度检查通过且Δ64<0才接受。Δ64不下降则拒绝。FP64核验不能接受FP32持平/上升。

原跨精度误差容差max(2e−6×当前loss,1e−3×|预测Δ|)没有扩大。同精度经验准则：u=2^-53，gamma(k)=ku/(1−ku)，k=d+N+ceil(N/4096)，rtol=32gamma(k)；atol=32gamma(k)×mean(2(|X|@|e|)(|X|@|δ|)+(|X|@|δ|)^2)+64u×mean(a²+d²)。要求|Δ_predict64−Δ_direct64|≤atol+rtol×max(|Δ_predict64|,|Δ_direct64|)。绝对项结合原始运算尺度，相对项结合维度和累积长度；这是预先固定的经验核验判据，不是严格误差定理。p缓存另用|e|行L1和C列最大绝对值构造尺度，在原漂移抽样节点及每次复核检查。非有限值、码越界、参数污染、同精度公式/缓存失败仍停止。

15项确定性测试包含Decimal80固定参考、取消误差/近零量级、原FP32上升拒绝、拒绝恢复/集合清除/计数上限、注入真实缓存和公式错误、完整小型批量修复终态与逐行参考一致。v43三个原上升候选仍拒绝。CPU测试仅为正确性证据，不作为质量证据；不重跑完整pilot。

## 数值复核与修复统计

正式跨精度复核触发 {len(refs):,} 个候选、{numerical['formal_recheck_batches']:,} 个批次，同步实测 {stage_recheck:.3f}秒。复核接受 {numerical['formal_recheck_accepts']:,}，复核后拒绝 {numerical['formal_recheck_rejects']:,}。本次新修复候选拒绝 {new_rejection_count:,}；继承v44拒绝单列 {sum(r['rejected_candidates'] for r in inherited):,}。真正正确性失败 {len(critical)}，进程失败 {len(failures)}，恢复 {len(recoveries)}。全部逐次预测/FP32差值/独立Δ64/尺度容差/缓存误差/接受结果见NUMERICAL_RECHECKS.csv及reports/。

全链已提交接受 {sum(r['accepted_updates'] for r in modules):,} 次，终态改变 {sum(r['modified_coordinates'] for r in modules):,} 个码位，实际候选评价 {sum(r['actual_candidate_evaluations'] for r in modules):,} 次；本次新增接受 {sum(r['accepted_updates'] for r in new):,} 次。该评价预算计数指原邻域二次型搜索评分；FP32直接核验及独立FP64增量/缓存/公式核验是额外计算，次数和成本单独记录，不称计算次数完全相同。无剩余预测下降候选/评价上限/更新上限的逐行统计见OUTPUT_AUDIT。固定各依赖组自身X下局部MSE合计 {initial:.12g}→{final:.12g}（下降{100*(initial-final)/initial:.4f}%），这是局部诊断合计，不代表端到端目标，不转换成PPL预测。

## 来源、传播与部署

模型revision、正式128×2048校准token及哈希、原M2 checkpoint和168个Linear网格继承原协议。所有历史参数/结果只读，v43/v44停止状态没有改写为PASS；源哈希及v44→v43链再次核验。未提交down从原始M2参数重新运行，保存失败tentative状态只用于验证。每组sibling修复后完整提交，再捕获真实依赖的下游输入：qkv→o→up/gate→down→下一block，来自本次R自身前缀。

每组原子checkpoint含整数码、scale/zero/g_idx、权重、输入/C哈希、预算与提交状态。CPU审计重放每次接受与拒绝集合，核对实际评价数、所有预算、合法码及网格不变。完整保存后核验全部168模块及其他原M2参数，pack/unpack实际FP16网格逐位一致；沿用原dense FP16/SDPA部署，辅助u4 packing不是替换kernel。reload与固定64-token输出回归使用原rtol1e−5/atol1e−6；配置不变，锁定最终checkpoint后才各运行一次WT2/C4。最终哈希见DEPLOYMENT_LOCK及reports/DEPLOYMENT_AUDIT。

PPL沿用2048窗口/512 stride、原token及valid-token计分规则，原始window_nll/valid_tokens全部保存并重新加权汇总。M1/M2正式历史模型、数据manifest和窗口来源均已核查，不重新量化基线，不读取P3，不选择性删除窗口，不以PPL调参。C4是固定预算validation PPL，不是完整标准C4。

## 实际成本与历史链成本

启动前保守预测 {proj['conservative_total_projection_seconds']/60:.3f}分钟：21剩余完整block上界×v42实测226.106660秒×原1.35安全系数，准备30、诊断验证180、新增FP64复核规划 allowance1200、收尾360。1200秒是单列的规划假设，非实测；在线保守成本门保留1.35和360秒预留。所有真实矩阵形状继承，未按代表行倍乘。

本次GPU预算墙时 {newwall:.3f}秒（{newwall/60:.3f}分钟），包含初始诊断、验证、CPU调度/记录间隔、完整恢复、失败、保存/reload和正式评测。同步阶段去除嵌套重复计数后实测合计 {sum(cost.values()):.3f}秒，分模式 {json.dumps(modes,ensure_ascii=False)}；未分项墙时 {newwall-sum(cost.values()):.3f}秒。原始阶段inclusive合计{sum(inclusive.values()):.3f}秒，包含父接受/恢复阶段内嵌的FP64复核{nested:.3f}秒；账单保留inclusive实测并从父阶段扣除子项，exclusive阶段求和不重复相加。正式含新组block墙时 {formalblocks:.3f}秒，新增独立FP64复核 {stage_recheck:.3f}秒。C构建、修复初始化、候选评分、接受/拒绝恢复、p核验、直接复算、提交、依赖刷新、packing/IO、最终保存/reload和PPL分项见STAGE_TIMINGS/COST_BILL。同步墙时含CPU操作和IO，不等于纯GPU kernel时间；评价次数也不等于时间或GPTQ完整Linear执行数。

历史成本分别为v42 {historical['historical_v42_GPU_budget_wall_seconds']:.3f}秒、v43 {historical['historical_v43_GPU_budget_wall_seconds']:.3f}秒、v44 {historical['historical_v44_GPU_budget_wall_seconds']:.3f}秒，包含各版本失败/诊断及间隔，未隐藏失败开销。v42 block0 pilot {proj['historical_v42_pilot_block_seconds']:.3f}秒是v42总墙时子项，不重复相加。v42至v45验证链累计 {chain:.3f}秒（{chain/60:.3f}分钟）；原M2构建 {proj['historical_M2_build_seconds']:.3f}秒另列，含M2构建总链 {(chain+proj['historical_M2_build_seconds'])/60:.3f}分钟。峰值allocated {peak/2**30:.3f}GiB、reserved {reserved/2**30:.3f}GiB。

## 边界与交付

最终以本次两语料PPL判断迁移。已有离散修复应用及数值核验修订，不是新论文方法，不是官方ReQuant复现，也不自动构成原创贡献。一次固定运行不能声称跨seed稳定。保持原模型/网格/目标，没有配对、缩放、正则化、残差目标、alpha重搜或新质量策略。完成后停止，不开启新规则/版本。

PROTOCOL、precision_check/repair_v45/run_v45、15项CPU检查、完整失败状态诊断、来源链、原始阶段计时、逐模块/层/行结果、checkpoint/hash、PPL原始窗口、DECISION和REPRODUCE均在本独立目录。Sol high要求仍保留；接口无法确认或切换，不声称已确认设置。
'''
 (OUT/'REPORT.md').write_text(text,encoding='utf8');(OUT/'reports/REPORT.md').write_text(text,encoding='utf8')
 dump('DELIVERY_MANIFEST.json',{'files':[{'path':str(p.relative_to(OUT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file() and 'original_codes' not in p.parts and 'formal' not in p.parts and p.name not in ['DELIVERY_MANIFEST.json','finalize.log']],'group_checkpoint_hashes_location':'reports/OUTPUT_AUDIT.json;DEPLOYMENT_LOCK.json','CPU_final_audit_seconds':time.perf_counter()-t,'category':category})
 print(json.dumps({k:v for k,v in decision.items() if k not in ['numerical_rechecks','deployment']},indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
