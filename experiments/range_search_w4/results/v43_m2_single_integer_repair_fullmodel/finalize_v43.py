"""CPU-only persisted move/grid, provenance, timing and PPL-window audit/report."""
import csv,hashlib,json,math,time
from collections import defaultdict
from pathlib import Path
import torch
from audit_sources import OUT,V18,sha,th,pack,unpack
V42=OUT.parent/'v42_m2_single_integer_repair_fullmodel'
torch.set_num_threads(4)
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def dump(n,v):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def csvread(n):return list(csv.DictReader((OUT/n).open(encoding='utf8')))
def csvwrite(n,rr):
 if not rr:return
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def persisted_audit():
 identity=read('GPU_START.json');assert sha(OUT/'run_v42.py')==identity['implementation_sha256']==sha(V42/'run_v42.py')
 assert sha(OUT/'run_v43.py')==identity['coordinator_sha256'] and sha(OUT/'PROTOCOL.json')==identity['protocol_sha256']
 source=read('SOURCE_AUDIT.json')
 for s in source['sources']:assert sha(s['source'])==s['source_sha256'] and sha(s['derived'])==s['derived_sha256']
 for s in read('SOURCE_REUSE_AUDIT.json')['original_codes_copies']:assert sha(s['path'])==s['sha256']
 assert sha(V18/'M2/checkpoint/final/model.safetensors')==source['model_sha256']
 modules=[];locks=[];rowrecords=[]
 for path in sorted((OUT/'formal/groups').glob('*.pt')):
  ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['committed'] and ck['protocol_sha256']==identity['protocol_sha256']
  orig=torch.load(OUT/f'original_codes/layer_{ck["layer"]:02d}.pt',map_location='cpu',weights_only=True)
  for name,s in ck['modules'].items():
   c=s['codes'];o=orig[name];assert bool((c<=15).all()) and torch.equal(unpack(s['packed_u4']),c) and torch.equal(pack(c),s['packed_u4'])
   for key in ['scales','zeros','g_idx']:assert torch.equal(s[key],o[key])
   q=(s['scales'][:,s['g_idx'].long()]*(c.float()-8)).half();assert torch.equal(q.view(torch.int16),s['weight'].view(torch.int16))
   assert bool((s['final_mse']<=s['initial_mse']).all())
   replay=o['codes'].long().clone();rows,dim=replay.shape;counts=torch.zeros(rows,dtype=torch.long);updates=counts.clone();loss=s['initial_mse'].clone();active=torch.ones(rows,dtype=torch.bool);moves={m['round']:m for m in s['moves']}
   for roundnum in range(1,33):
    if not active.any():break
    counts[active]+=(replay[active]>0).sum(1)+(replay[active]<15).sum(1);m=moves.get(roundnum);active[:]=False
    if m is None:break
    rr,col,off=m['rows'],m['columns'],m['offsets'];assert rr.unique().numel()==rr.numel() and bool(((off==-1)|(off==1)).all())
    assert torch.equal(m['evaluations'],counts[rr]);pred,actual=m['predicted_delta'],m['direct_delta']
    threshold=torch.maximum(s['initial_mse'][rr]*1e-8,torch.full_like(pred,1e-14));assert bool((pred < -threshold).all() and (actual<0).all())
    assert bool(((actual-pred).abs()<=torch.maximum(loss[rr]*2e-6,pred.abs()*1e-3)).all())
    replay[rr,col]+=off;assert bool(((replay>=0)&(replay<=15)).all());loss[rr]+=actual;updates[rr]+=1;active[rr]=True
   assert torch.equal(replay.byte(),c) and torch.equal(counts,s['evaluations']) and torch.equal(updates,s['updates'])
   assert bool((counts<=128*dim).all() and (updates<=32).all()) and torch.isclose(loss,s['final_mse'],rtol=2e-6,atol=1e-12).all()
   initial=float(s['initial_mse'].sum());final=float(s['final_mse'].sum())
   modules.append({'layer':ck['layer'],'module':name,'rows':rows,'input_dim':dim,'accepted_updates':int(updates.sum()),'modified_coordinates':int((c!=o['codes']).sum()),'actual_candidate_evaluations':int(counts.sum()),'initial_search_mse_sum':initial,'final_search_mse_sum':final,'search_gain_percent':100*(initial-final)/initial,'initial_pooled_SSE':initial*262144,'final_pooled_SSE':final*262144,'max_p_drift':s['max_p_drift'],'codes_sha256':th(c),'scale_sha256':th(s['scales']),'zero_sha256':th(s['zeros']),'g_idx_sha256':th(s['g_idx']),'input_sha256':ck['input_sha256'],'C_sha256':ck['C_sha256'],'persisted_moves_replayed':True})
   for row in range(rows):rowrecords.append({'layer':ck['layer'],'module':name,'row':row,'input_dim':dim,'initial_search_mse':float(s['initial_mse'][row]),'final_search_mse':float(s['final_mse'][row]),'updates':int(updates[row]),'evaluations':int(counts[row]),'eval_cap':128*dim,'status':{1:'NO_STRICT_DESCENT',2:'UPDATE_CAP'}[int(s['status'][row])]})
  locks.append({'path':str(path),'sha256':sha(path),'layer':ck['layer'],'group':ck['group'],'input_sha256':ck['input_sha256'],'C_sha256':ck['C_sha256']})
 csvwrite('reports/MODULE_RESULTS.csv',modules);csvwrite('reports/ROW_RESULTS.csv',rowrecords)
 # Formal block0 is naturally repaired once; compare it with inherited pilot, no rerun.
 block0=[]
 for group in ['qkv','o','up_gate','down']:
  current=OUT/f'formal/groups/layer_00_{group}.pt'
  if not current.exists():continue
  a=torch.load(current,map_location='cpu',weights_only=False);b=torch.load(V42/f'pilot/groups/layer_00_{group}.pt',map_location='cpu',weights_only=False)
  assert a['input_sha256']==b['input_sha256'] and a['C_sha256']==b['C_sha256']
  for n,s in a['modules'].items():
   assert torch.equal(s['codes'],b['modules'][n]['codes']) and torch.equal(s['updates'],b['modules'][n]['updates']) and torch.equal(s['evaluations'],b['modules'][n]['evaluations'])
  block0.append({'group':group,'input_C_hashes_equal':True,'codes_updates_evaluations_equal':True})
 dump('reports/OUTPUT_AUDIT.json',{'pass':True,'modules':len(modules),'rows':sum(r['rows'] for r in modules),'persisted_integer_moves_replayed':True,'grid_unchanged':True,'budget_caps_verified':True,'historical_source_hashes_unchanged':True,'runtime_byte_identical_to_v42':True,'formal_block0_matches_inherited_pilot':block0,'checkpoints':locks})
 return modules
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
 dump('reports/BASELINE_AUDIT.json',{'pass':True,'baselines':baseline});comparisons=[]
 for ds in ['wikitext','c4_fixed_budget']:
  p=OUT/f'R_{ds}.json'
  if not p.exists():continue
  r=json.loads(p.read_text());wr=csvread(f'R_{ds}_WINDOWS.csv');nt=sum(int(w['valid_tokens']) for w in wr);nll=sum(float(w['window_nll']) for w in wr)/nt
  assert nt==r['scored_tokens']==r['input_tokens']-1 and len(wr)==r['windows'] and abs(nll-r['nll'])<1e-10 and abs(math.exp(nll)-r['ppl'])<1e-10
  ready=read('DEPLOYMENT_LOCK.json');assert r['model_sha256']==ready['final_model_sha256']==sha(OUT/'R/checkpoint/final/model.safetensors')
  mm={b['method']:b['ppl'] for b in baseline if b['dataset']==ds}
  comparisons.append({'dataset':ds,'M1_ppl':mm['M1'],'M2_ppl':mm['M2'],'R_ppl':r['ppl'],'R_minus_M1':r['ppl']-mm['M1'],'R_vs_M1_relative_percent':100*(r['ppl']/mm['M1']-1),'R_minus_M2':r['ppl']-mm['M2'],'R_vs_M2_relative_percent':100*(r['ppl']/mm['M2']-1),'improved_vs_M2':r['ppl']<mm['M2'],'scored_tokens':nt,'windows':len(wr),'raw_window_NLL_reaggregation_pass':True})
 csvwrite('reports/PPL_COMPARISON.csv',comparisons);return comparisons
def main():
 t=time.perf_counter();status=read('RUN_STATUS.json');modules=persisted_audit();comparison=ppl_audit();complete=status['status']=='COMPLETE'
 if complete:assert len(modules)==168 and sum(r['rows'] for r in modules)==304128 and len(comparison)==2
 category='D' if not complete else ('A' if all(r['improved_vs_M2'] for r in comparison) else 'B' if any(r['improved_vs_M2'] for r in comparison) else 'C')
 verdict={'A':'两语料 PPL 均改善，已有离散修复路线的本次迁移得到支持','B':'两语料方向不一致，本次迁移不稳定','C':'两语料均未改善','D':'完整运行或评测未完成，质量未验证'}[category]
 times=csvread('STAGE_TIMINGS.csv');cost=defaultdict(float)
 for r in times:cost[(r['mode'],r['stage'])]+=float(r['seconds'])
 csvwrite('reports/COST_BILL.csv',[{'mode':m,'stage':s,'actual_synchronized_wall_seconds':v,'type':'measured'} for (m,s),v in sorted(cost.items())])
 blocks=csvread('BLOCK_TIMINGS.csv') if (OUT/'BLOCK_TIMINGS.csv').exists() else []
 layer=[]
 for li in sorted(set(r['layer'] for r in modules)):
  rr=[r for r in modules if r['layer']==li];a=sum(r['initial_search_mse_sum'] for r in rr);b=sum(r['final_search_mse_sum'] for r in rr)
  layer.append({'layer':li,'modules':len(rr),'rows':sum(r['rows'] for r in rr),'accepted_updates':sum(r['accepted_updates'] for r in rr),'modified_coordinates':sum(r['modified_coordinates'] for r in rr),'actual_evaluations':sum(r['actual_candidate_evaluations'] for r in rr),'initial_search_mse_sum':a,'final_search_mse_sum':b,'gain_percent':100*(a-b)/a})
 csvwrite('reports/LAYER_RESULTS.csv',layer)
 init=sum(r['initial_search_mse_sum'] for r in modules);final=sum(r['final_search_mse_sum'] for r in modules);accepted=sum(r['accepted_updates'] for r in modules)
 peak=max(int(r['gpu_peak_allocated_bytes']) for r in times);reserved=max(int(r['gpu_peak_reserved_bytes']) for r in times)
 checks=[json.loads(l) for l in (OUT/'CORRECTNESS.jsonl').read_text().splitlines()]
 rejected=[r for r in checks if not r.get('pass_all',True)]
 if complete:assert not rejected
 else:
  assert not (OUT/'R/checkpoint/final/model.safetensors').exists() and not comparison
  # Preserve rejected acceptance checks; they are not silently called passes.
  assert all(not (OUT/f'formal/groups/layer_{r["layer"]:02d}_up_gate.pt').exists() for r in rejected)
 diagnosis=read('NUMERICAL_DIAGNOSIS.json') if (OUT/'NUMERICAL_DIAGNOSIS.json').exists() else None
 dump('reports/NUMERICAL_REJECTIONS.json',{'rejected_acceptance_checks':rejected,'diagnosis':diagnosis,'rejected_group_not_committed':True,'frozen_acceptance_rule_unchanged':True})
 projection=read('COST_PROJECTION.json');failure=(OUT/'FAILED_ATTEMPTS.jsonl').read_text().splitlines() if (OUT/'FAILED_ATTEMPTS.jsonl').exists() else [];recovery=(OUT/'RECOVERY_LOG.jsonl').read_text().splitlines() if (OUT/'RECOVERY_LOG.jsonl').exists() else []
 formal_block_seconds=sum(float(r['seconds']) for r in blocks);save_reload=sum(v for (m,s),v in cost.items() if m=='deployment');eval_seconds=sum(v for (m,s),v in cost.items() if m=='evaluation')
 decision={'category':category,'verdict':verdict,'status':status,'formal_modules':len(modules),'formal_rows':sum(r['rows'] for r in modules),'accepted_updates':accepted,'modified_coordinates':sum(r['modified_coordinates'] for r in modules),'candidate_evaluations':sum(r['actual_candidate_evaluations'] for r in modules),'counts_scope':'committed dependency groups only; incomplete/rejected up_gate excluded and preserved separately','local_reconstruction_gain_percent':100*(init-final)/init if init else None,'formal_block_measured_seconds':formal_block_seconds,'formal_all_attempt_stage_seconds':sum(v for (m,s),v in cost.items() if m=='formal'),'diagnostic_measured_stage_seconds':sum(v for (m,s),v in cost.items() if m=='diagnostic'),'deployment_measured_stage_seconds':save_reload,'evaluation_measured_stage_seconds':eval_seconds,'this_run_hard_budget_wall_seconds':status['elapsed_seconds'],'v42_prior_block_pilot_seconds':projection['v42_prior_pilot_block_seconds'],'v42_prior_GPU_wall_seconds':projection['v42_prior_total_GPU_budget_wall_seconds'],'historical_M2_creation_seconds':2115.9353341000096,'peak_GPU_allocated_bytes':peak,'peak_GPU_reserved_bytes':reserved,'failed_attempts':len(failure),'recoveries':len(recovery),'rejected_acceptance_checks':len(rejected),'PPL_comparison':comparison,'conservative_pre_run_forecast_seconds':projection['conservative_total_projection_seconds'],'no_automatic_followup':True,'model_setting':'Sol high requested; interface cannot confirm/switch.'}
 dump('DECISION.json',decision)
 text=f'''# v43 REPORT

## 结果

**{category}：{verdict}。** {status['status']}。正式覆盖 {len(modules)}/168 个 Linear、{sum(r['rows'] for r in modules):,}/304,128 个输出行。

'''
 for r in comparison:text+=f"- {r['dataset']}：M1 {r['M1_ppl']:.8f}，M2 {r['M2_ppl']:.8f}，R {r['R_ppl']:.8f}；R−M2 {r['R_minus_M2']:+.8f}（{r['R_vs_M2_relative_percent']:+.4f}%），R−M1 {r['R_minus_M1']:+.8f}（{r['R_vs_M1_relative_percent']:+.4f}%）。\n"
 if complete:text+='\n两语料方向'+('一致，均改善。' if category=='A' else '不一致。' if category=='B' else '均未改善。')+'此判断使用各一次冻结正式评测，无跨 seed 稳定性结论。\n'
 else:
  text+='\n完整修复或规定评测未完成，不能用局部重构下降代替质量验证。详见 RUN_STATUS.json 和失败记录。\n'
  if diagnosis:
   text+='\n停止原因为 block3 mlp.up_proj 第28轮的3行（1558、2907、3725）未通过冻结批量FP32直接评分的严格下降判据。预测Δ为负，但FP32批量实际Δ为正（约1.38e−11至6.84e−11）；该步及整个up/gate组未提交。\n\n同一当前R前缀输入、相同网格/码状态的独立诊断显示：重新计算p的预测与失败时预测之差最大约1.30e−22；FP64直接增量与预测之差最大约1.73e−18，均真实下降；FP32单行matvec也下降，但FP32三行/原批量matmul仍上升。根因是近零增量的FP32累加路径差异，非p漂移、网格/整数码或公式错误。诊断不授权替换冻结评分或接受上升，未放宽容差、未修改修复代码，未自动恢复相同失败或换评分继续。\n'
  hist=read('reports/BASELINE_AUDIT.json')['baselines']
  text+='\n已核查并复用的历史正式基线（R未评测，相对变化不可计算）：\n\n'
  for ds in ['wikitext','c4_fixed_budget']:
   mm={r['method']:r['ppl'] for r in hist if r['dataset']==ds};text+=f"- {ds}：M1={mm['M1']:.8f}，M2={mm['M2']:.8f}；R/相对变化未测量。\n"
 text+=f'''
## 实现与正确性

从原正式 M2 初始化一次独立完整流程；block0 在正式流程正常修复一次。v42 run_v42.py/audit_sources.py 逐字节复制且哈希相同；新协调入口只将硬预算改为180分钟、跳过已完成的 v41复现/pilot、设置新输出目录和增加最终配置/就绪锁核验。搜索、缓存、直接重算、C构建和传播函数未改动，无新增缓存优化。

v42 已通过的96条A轨迹、4条逐行参考、168模块零修改回环和全部block0证据以只读哈希方式继承，不重新执行。正式block0落盘码、评价/接受数、输入和C哈希与继承pilot逐项比较。每组提交后才捕获真正依赖该组的新输入，qkv→o→up/gate→down→next block，均来自本次R自身前缀。

固定 W4A16 symmetric G128、原FP32 scale、zero=8、g_idx、合法整数码0..15。只评分±1单坐标邻域，每行最多32次接受、128×输入维度评价上限。固定首次最小值顺序坐标升序/−1再+1，沿用v41预测阈值和直接严格下降/误差容差。实际δ来自原FP16网格值之差。全C为独立未damping的FP64 Gram，固定4096-token累积，全非对角/跨组保留。所有数值接受检查见 CORRECTNESS.jsonl。

接受 {accepted:,} 次修改；终态改变 {sum(r['modified_coordinates'] for r in modules):,} 个码位；实际评价 {sum(r['actual_candidate_evaluations'] for r in modules):,} 个候选。固定各组输入下的局部MSE合计 {init:.12g}→{final:.12g}（下降 {100*(init-final)/init if init else 0:.4f}%）。跨模块合计仅为局部诊断，不是全模型端到端目标。逐模块/层/行见 reports/。

CPU落盘核查重放每一步修改并核对候选评价数、更新上限、码合法、scale/zero/g_idx不变、pack/reload逐位网格值及来源哈希。正式部署仍为原dense FP16/SDPA；辅助unsigned4 packing仅存储码，不换部署kernel。完整时先保存、全168模块grid/code与配置检查、reload固定64-token输出回归（原rtol1e−5/atol1e−6），再写DEPLOYMENT_LOCK锁，之后各一次PPL。未放宽容差。

## 成本：预测与实测分开

v42实测完整block0为 {projection['v42_measured_block0_seconds']:.6f} 秒。真实形状（输出行×输入列）：q/o=896×896，k/v=128×896，up/gate=4864×896，down=896×4864；24块均已核对同形。原预测公式：226.1066603×24×1.35+30 = {projection['formal_repair_conservative_projection_seconds']:.6f} 秒（122.60分钟）；再预留360秒收尾，总 {projection['conservative_total_projection_seconds']/60:.4f} 分钟。原安全系数1.35保持；各阶段及未分项block墙时见 COST_PROJECTION.json。既有正式M2两语料评测实测130.580秒，未出现要求超过360秒预留的新来源证据。

本次完整完成block修复实测合计 {formal_block_seconds:.3f} 秒（不包含未完成block）；所有正式尝试同步阶段实测 {sum(v for (m,s),v in cost.items() if m=='formal'):.3f} 秒，包含失败block的已执行阶段。数值诊断同步阶段实测 {sum(v for (m,s),v in cost.items() if m=='diagnostic'):.3f} 秒；保存/reload等部署阶段同步实测 {save_reload:.3f} 秒；正式评测同步阶段实测 {eval_seconds:.3f} 秒。全轮预算墙时 {status['elapsed_seconds']:.3f}/10800秒，包含预热、失败、诊断及间隔与全部执行阶段。分阶段总计 {sum(cost.values()):.3f} 秒；未分项墙时 {status['elapsed_seconds']-sum(cost.values()):.3f} 秒（CPU调度/记录及失败到诊断的间隔等，不等同GPU kernel时间）。原始阶段、block墙时和失败/恢复日志保留。

v42 已发生成本独立列出：block0 pilot {projection['v42_prior_pilot_block_seconds']:.3f} 秒，v42总GPU预算墙时 {projection['v42_prior_total_GPU_budget_wall_seconds']:.3f} 秒。本轮不重复计为正式修复。原历史M2构建成本2115.935秒单列；新修复是新增成本。候选评价次数不等于时间，也不等于GPTQ完整Linear执行数。

峰值 allocated {peak/2**30:.3f} GiB、reserved {reserved/2**30:.3f} GiB。失败 {len(failure)} 次、恢复 {len(recovery)} 次；恢复已提交组只复制参数/重放前缀，不重复修复。

## 推进判断与边界

本轮为已有离散修复路线应用验证，不是官方ReQuant复现，也不自动构成原创贡献。局部MSE下降不能替代PPL，不将其转换为PPL预测；PPL未用于调参。没有P3、配对、缩放、正则化、改更新次数、挑层或新缓存优化。{'本次迁移获得两语料支持，但一次固定运行不能声称跨seed稳定或原创性。' if category=='A' else '按本次结果分类记录结论，不扩大质量实验或调整冻结参数。'} 完成后停止，无自动下一版本。

Sol high为用户请求；接口不能确认或切换，故不声明已确认。

## 交付

PROTOCOL.json、SOURCE_REUSE_AUDIT.json、run_v42.py（原字节）、run_v43.py（协调器）、原始STAGE_TIMINGS.csv/BLOCK_TIMINGS.csv/CORRECTNESS.jsonl、formal/groups已提交 checkpoint及输入哈希/预算/提交状态、reports下已提交模块/层/行结果与成本/基线审计、DECISION.json、REPRODUCE.md。{'另有最终checkpoint/hash/部署锁、两语料原始窗口NLL/valid_tokens和PPL汇总。' if complete else '本轮未产生完整R checkpoint、DEPLOYMENT_LOCK或R的PPL窗口。NUMERICAL_FAILURE状态、NUMERICAL_DIAGNOSIS、失败原始状态与日志完整保留；局部checkpoint不称为完整模型。'}
'''
 (OUT/'REPORT.md').write_text(text,encoding='utf8');(OUT/'reports/REPORT.md').write_text(text,encoding='utf8')
 dump('DELIVERY_MANIFEST.json',{'files':[{'path':str(p.relative_to(OUT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file() and 'original_codes' not in p.parts and 'formal' not in p.parts and p.name not in ['DELIVERY_MANIFEST.json','finalize.log']],'checkpoint_hashes_location':'reports/OUTPUT_AUDIT.json;DEPLOYMENT_LOCK.json','CPU_final_audit_seconds':time.perf_counter()-t,'category':category})
 print(json.dumps(decision,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
