"""CPU-only v52 audit, provenance, costs and quality reporting. No search or GPU calls."""
import csv,hashlib,json,math,time
from collections import defaultdict,Counter
from pathlib import Path
import torch
O=Path(__file__).resolve().parent;B=O.parent;V51=B/'v51_m2_fast_integer_repair_fullmodel';V46=B/'v46_m1_single_integer_repair_fullmodel';V49=B/'v49_fullmodel_fast_integer_repair';V18=B/'range_w4_fullmodel_v18'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf8'))) if Path(p).exists() else []
def outcsv(n,rr):
 if not rr:(O/n).write_text('',encoding='utf8');return
 fields=list(dict.fromkeys(k for r in rr for k in r))
 with (O/n).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def quality_source(model,ds,rec):
 assert sha(rec['result_path'])==rec['result_sha256'] and sha(rec['windows_path'])==rec['windows_sha256']
 windows=rows(rec['windows_path']);tokens=sum(int(w['valid_tokens']) for w in windows);avg=sum(float(w['window_nll']) for w in windows)/tokens
 assert tokens==int(rec['valid_tokens']) and abs(avg-float(rec['nll']))<1e-10
 assert abs(math.exp(avg)-float(rec['ppl']))/float(rec['ppl'])<1e-10
 return dict(model=model,dataset=ds,PPL=float(rec['ppl']),NLL=avg,valid_tokens=tokens,windows=len(windows),quality_source='historical audited',result_path=rec['result_path'],result_sha256=rec['result_sha256'],windows_path=rec['windows_path'],windows_sha256=rec['windows_sha256'])
def main():
 status=read(O/'RUN_STATUS.json');complete=bool(status.get('complete_model') and (O/'COMPLETE.json').exists());ledger=read(O/'ACTIVE_JOB_LEDGER.json');assert all('charged_seconds' in s for s in ledger['segments'])
 newwall=sum(s['charged_seconds'] for s in ledger['segments']);total=ledger['v51_charged_seconds']+newwall;assert total<=10800+.2
 for n,h in read(O/'FROZEN_VALIDATION_LOCK.json')['files'].items():assert sha(O/n)==h,f'frozen v52 file changed: {n}'
 im=[]
 for x in read(O/'HISTORY_IMMUTABILITY_BEFORE.json'):
  h=sha(x['path']);assert h==x['sha256'];im.append(dict(path=x['path'],before_sha256=x['sha256'],after_sha256=h,unchanged=True))
 assert read(V51/'FINAL_DECISION.json')['status']=='D';dump('HISTORY_IMMUTABILITY_AFTER.json',dict(**{'pass':True},files=im,v51_still_D=True))
 groupfiles=sorted((O/'formal/groups').glob('layer_*.pt'));mods=[];events=[];hist=[];updates_dist=Counter();stop_dist=Counter();inherited=0
 for path in groupfiles:
  cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['committed'];old=bool(cp.get('inherited_from'));inherited+=old
  for name,r in cp['modules'].items():
   assert bool((r['codes']<=15).all() and (r['updates']<=32).all() and (r['evaluations']<=128*r['codes'].shape[1]).all())
   counts=Counter(r['updates'].tolist());updates_dist.update(counts);stop_dist.update(r['status'].tolist())
   info=r.get('initialization_check',{});mods.append(dict(block=cp['layer'],group=cp['group'],module=name,rows=len(r['codes']),input_dim=r['codes'].shape[1],origin='inherited v51' if old else 'v52',accepted_updates=int(r['updates'].sum()),candidate_evaluations=int(r['evaluations'].sum()),candidate_rejections=int(r['rejections'].sum()),candidate_FP64_rechecks=len(r['recheck_events']),initialization_failing_rows=info.get('original_failed_rows',0),initialization_FP64_trigger=int(bool(info.get('fallback'))),raw64_initial_loss_sum=float(r['terminal_initial_raw64'].sum()),raw64_final_loss_sum=float(r['terminal_final_raw64'].sum()),max_p_drift=r['max_p_drift'],input_sha256=cp['input_sha256'],C_sha256=cp['C_sha256'],checkpoint_sha256=sha(path)))
   for c,num in sorted(counts.items()):hist.append(dict(block=cp['layer'],group=cp['group'],module=name,updates=c,rows=num))
   for e in info.get('events',[]):events.append(dict(block=cp['layer'],group=cp['group'],module=name,**e))
 outcsv('MODULE_SUMMARY.csv',mods);outcsv('INITIALIZATION_REVIEW_EVENTS.csv',events);outcsv('UPDATE_DISTRIBUTION_BY_MODULE.csv',hist);outcsv('UPDATE_DISTRIBUTION_GLOBAL.csv',[dict(updates=c,rows=n) for c,n in sorted(updates_dist.items())]);outcsv('STOP_DISTRIBUTION.csv',[dict(status=c,meaning={0:'still active',1:'no remaining improving candidate',2:'32 update cap',3:'candidate evaluation cap'}[c],rows=n) for c,n in sorted(stop_dist.items())])
 stages=rows(O/'STAGE_TIMINGS.csv');summary=defaultdict(list)
 for x in stages:summary[(x['mode'],x['stage'])].append(x)
 outcsv('STAGE_COSTS.csv',[dict(mode=m,stage=s,calls=len(v),synchronized_seconds=sum(float(x['seconds']) for x in v),peak_allocated_bytes=max(int(x['gpu_peak_allocated_bytes']) for x in v),note='some timers nest; do not sum all stage rows as exclusive wall') for (m,s),v in sorted(summary.items())])
 blocks=rows(O/'BLOCK_TIMINGS.csv');repair=sum(float(x['seconds']) for x in blocks if int(x['new_groups'])>0);newrepair=sum(float(x['seconds']) for x in blocks if int(x['new_groups'])>0 and int(x['layer'])>0);restore=sum(float(x['seconds']) for x in blocks if int(x['new_groups'])==0)
 historicalM2=read(V18/'FORMAL_RUNTIME.json')['m2']['wall_seconds_including_load_replay_save'];newdiag=sum(s['charged_seconds'] for s in ledger['segments'] if 'regression' in s['label']);formal=sum(s['charged_seconds'] for s in ledger['segments'] if 'formal' in s['label']);timed=lambda names:sum(float(x['seconds']) for x in stages if x['mode']!='diagnostic' and x['stage'] in names)
 costs=[dict(item='historical M2 construction',seconds=historicalM2,basis='v18 actual; outside shared repair GPU cap'),dict(item='historical v51 charged job',seconds=ledger['v51_charged_seconds'],basis='original measured657.304609 rounded up; contains failure/recovery/diagnosis'),dict(item='v52 regression/source-validation active job',seconds=newdiag,basis='active segment wall, includes GPU and supporting IO'),dict(item='v52 formal active job',seconds=formal,basis='active segment wall including restoration/repair/save/reload/quality and failures if any'),dict(item='v52 total added active-job wall',seconds=newwall,basis='sum active job segments; app idle excluded'),dict(item='v51+v52 shared charged wall',seconds=total,basis='cap10800; CPU environment diagnosis separate'),dict(item='successful full 24block repair stages',seconds=repair,basis='v51 block0 once + new v52 block1-23; completed repair only'),dict(item='v52 new block repair stages',seconds=newrepair,basis='actual block wall; excludes inherited block0'),dict(item='v52 block0 restore-only run_layer wall',seconds=restore,basis='no repair; prefix-validation outer timer is separate'),dict(item='formal Gram construction timers',seconds=timed(['C_construction']),basis='unchanged FP64 builder; synchronized measured timers'),dict(item='initialization fallback timers including checks/hash',seconds=timed(['initial_rawX_FP64_review']),basis='required fallback cost; separate diagnostic-mode costs in STAGE_COSTS'),dict(item='provenance wrapper timers',seconds=timed(['v52_input_source_hash','v52_input_pre_Gram_source_check','v52_Gram_source_hash']),basis='actual synchronized source/hash costs; may lie within restore outer timer'),dict(item='deployment save/reload/tensor comparison timers',seconds=timed(['final_save','pack_reload_all_grid_check','reload','named_tensor_comparison_vs_original_R2']),basis='actual measured stages'),dict(item='fresh PPL timers',seconds=timed(['PPL_evaluation']),basis='zero if exact model equivalence permits audited reuse'),dict(item='historical M2 build + complete validation chain spent',seconds=historicalM2+total,basis='includes all historical v51 failures and new active jobs; not original R2 net repair speed estimate')]
 continuation=sum(v['charged_seconds'] for v in ledger['segments'] if 'deployment continuation' in v['label'])
 prefixdiag=sum(v['charged_seconds'] for v in ledger['segments'] if 'prefix compatibility diagnosis' in v['label'])
 costs += [dict(item='deployment continuation jobs including failures and interrupted PPL',seconds=continuation,basis='all3 active segments; forced end conservatively bounded; no completed PPL result'),dict(item='actual block0 prefix compatibility diagnostic',seconds=prefixdiag,basis='bounded GPU source diagnosis only; no repair or PPL'),dict(item='CPU current source re-verification',seconds=read(O/'SOURCE_REVERIFICATION.json')['CPU_wall_seconds'],basis='CPU-only evidence audit; not additional GPU time; may overlap active-job wall'),dict(item='CPU environment/tool diagnosis',seconds=None,basis='NOT_MEASURED independent total; separate from GPU jobs'),dict(item='interrupted PPL stage exclusive time',seconds=None,basis='NOT_MEASURED: forced stop before outer timer flushed; included within continuation wall, preserve partial log')]
 outcsv('ACTIVE_SEGMENTS.csv',ledger['segments'])
 outcsv('COST_LEDGER.csv',costs)
 baseline=read(V46/'PPL_SOURCE_AUDIT.json')['historical_source_records'];audit=read(O/'SOURCE_AUDIT.json');qr=[]
 for ds,key in [('WT2 test','wikitext'),('C4 fixed validation','c4_fixed_budget')]:
  for model,prefix in [('M1','M1'),('M2','M2'),('original R2','R2')]:qr.append(quality_source(model,ds,baseline[f'{prefix}_{key}']))
  qr.append(quality_source('v49 M1+fast',ds,audit['v49_r1_ppl_sources'][key]))
  if complete:
   rp=O/f'evaluation/M2_FAST_{key}.json';wp=O/f'evaluation/M2_FAST_{key}_WINDOWS.csv';j=read(rp);ww=rows(wp);tokens=sum(int(w['valid_tokens']) for w in ww);nll=sum(float(w['window_nll']) for w in ww)/tokens;assert abs(nll-j['nll'])<1e-10
   qr.append(dict(model='v52 M2+fast',dataset=ds,PPL=j['ppl'],NLL=nll,valid_tokens=tokens,windows=len(ww),quality_source='audited original R2 reuse after deployment equality' if status['PPL_reused'] else 'one new frozen evaluation',result_path=str(rp),result_sha256=sha(rp),windows_path=str(wp),windows_sha256=sha(wp)))
  else:qr.append(dict(model='v52 M2+fast',dataset=ds,PPL=None,NLL=None,quality_source='D: quality not completed'))
 for x in qr:
  if x['PPL'] is None:continue
  for ref in ['M1','M2','original R2','v49 M1+fast']:
   base=next(y for y in qr if y['dataset']==x['dataset'] and y['model']==ref);tag=ref.replace(' ','_')
   x[f'PPL_delta_vs_{tag}']=x['PPL']-base['PPL'];x[f'PPL_change_vs_{tag}_pct']=(x['PPL']/base['PPL']-1)*100;x[f'NLL_delta_vs_{tag}']=x['NLL']-base['NLL']
 outcsv('QUALITY_COMPARISON.csv',qr)
 if complete:assert len(groupfiles)==96 and len(mods)==168 and sum(m['rows'] for m in mods)==304128 and inherited==4
 result=dict(category='COMPLETE' if complete else 'D',full_model=complete,coverage=dict(groups=len(groupfiles),inherited_v51_groups=inherited,modules=len(mods),rows=sum(m['rows'] for m in mods)),accepted_updates=sum(m['accepted_updates'] for m in mods),candidate_evaluations=sum(m['candidate_evaluations'] for m in mods),candidate_rejections=sum(m['candidate_rejections'] for m in mods),candidate_FP64_rechecks=sum(m['candidate_FP64_rechecks'] for m in mods),initialization_review_triggers=sum(m['initialization_FP64_trigger'] for m in mods),initialization_review_rows=len(events),historical_v51_D_preserved=True,shared_charged_seconds=total,v52_added_seconds=newwall,M2_build_seconds=historicalM2,quality_evaluations=2 if complete and not status['PPL_reused'] else 0,quality_reused=bool(status.get('PPL_reused')),equivalent_to_original_R2=status.get('equivalent_to_original_R2'),frozen_source_verified=True,no_theoretical_monotonicity_claim=True)
 prefix=read(O/'PREFIX_COMPATIBILITY_DIAGNOSIS.json')
 result.update(category='D',full_model=False,all168_matrices_executed=True,correct_full_model_propagation=False,quality_verified=False,initialization_fallback_regression_pass=True,hard_correctness_error='Tensor return indexed as old tuple; first token broadcast to entire next-block sequence',layer1_and_later_results_invalid_for_frozen_protocol=True,partial_PPL_interrupted=True,new_completed_PPL_evaluations=0,remaining_shared_budget_seconds=10800-total)
 dump('FINAL_DECISION.json',result)
 dump('MODEL_VALIDITY.json',{'valid_formal_v52_model':False,'diagnostic_saved_model_path':str(O/'R1/checkpoint/final/model.safetensors'),'model_sha256':sha(O/'R1/checkpoint/final/model.safetensors'),'deployment_grid_and_reload_checks_passed':True,'true_prefix_propagation_failed':True,'must_not_use_for_quality_comparison_or_as_resumption_after_layer0':True,'evidence':'PREFIX_COMPATIBILITY_DIAGNOSIS.json','historical_files_unchanged':True})
 lines=['v52：M2 快速整数修复的初始化跨精度独立复核。','',('完整修复、部署及质量验证完成。' if complete else '结果 D：完整运行或质量评测未完成。停止原因见 RUN_STATUS.json。'),'',f"覆盖 {len(groupfiles)}/96 依赖组、{len(mods)}/168 Linear、{sum(m['rows'] for m in mods)}/304128 行。其中 block0 四组来自已核验 v51，只恢复并重放；layer1 qkv 从原 M2 开始。所有新增输入来自本次修复模型前缀。",'',f"唯一协议变化是初始化检查失败后的 raw-X FP64 复核。原 2e-5 相对/1e-12 绝对损失尺度容差保留，原 FP32 initial、quadratic state loss、p/码/阈值不变。row122 原失败重现，独立 FP64 通过且状态逐字节不变；8 个CPU用例覆盖正常路径、错误C/输入和非有限值。具体 diff、数据流及冻结哈希见 LOCAL_REPAIR.diff、DATAFLOW.md、FROZEN_VALIDATION_LOCK.json。",'',f"新增初始化复核触发 {result['initialization_review_triggers']} 个矩阵、{len(events)} 行；正式必要复核与来源检查时间 {timed(['initial_rawX_FP64_review']):.3f}s，原候选FP64复核 {result['candidate_FP64_rechecks']} 次。接受 {result['accepted_updates']} 次、候选拒绝 {result['candidate_rejections']} 次、增量评价 {result['candidate_evaluations']} 次。终态 raw-X FP64 核验覆盖每个完成矩阵的全部行；这不构成全量逐步数学单调性证明。",'','质量（各语料 valid-token 加权 NLL，语料间不混合）：']
 for x in qr:lines.append(f"- {x['dataset']} / {x['model']}: "+(f"PPL {x['PPL']:.9f}; NLL {x['NLL']:.12f}; {x['valid_tokens']} valid tokens; {x['quality_source']}" if x['PPL'] is not None else '未评测'))
 if complete:
  lines+=['',f"与原 R2 部署张量及推理配置等价：{status['equivalent_to_original_R2']}。新评测次数 {result['quality_evaluations']}；复用需部署等价证明和历史窗口来源hash通过。完整张量/配置比较与模型锁见 DEPLOYMENT_COMPARISON.json、CHECKPOINT_LOCK.json。"]
  for ds in ['WT2 test','C4 fixed validation']:
   x=next(y for y in qr if y['model']=='v52 M2+fast' and y['dataset']==ds)
   lines.append(f"{ds}: 相对M2 PPL {x['PPL_change_vs_M2_pct']:+.6f}%; 相对原R2 {x['PPL_change_vs_original_R2_pct']:+.6f}%; 相对v49 {x['PPL_change_vs_v49_M1+fast_pct']:+.6f}%。")
 lines+=['',f"成本：v51原账本657.304609s按657.305s保守计入；v52新增活跃作业 {newwall:.3f}s（其中局部回归 {newdiag:.3f}s、正式作业 {formal:.3f}s）；共同累计 {total:.3f}/10800s。关闭应用的等待不计入。历史M2构建 {historicalM2:.3f}s单列；构建加验证链累计 {historicalM2+total:.3f}s。",f"成功24block修复阶段合计 {repair:.3f}s，其中新block1–23为 {newrepair:.3f}s、v51 block0一次187.084840s；恢复-only block0 run_layer为 {restore:.3f}s，另有来源/前缀核验计时。原R2没有可靠完整净修复计时，不计算精确加速比。v49使用不同初始化及输入，成本差不能全部归因于实现优化。",'','STAGE_COSTS.csv 汇总真实打点；部分timer嵌套，不将全表时间相加冒充独立总wall。C构建、初始化、候选评分、更新/接受、候选复核、p刷新、完整终态核验、checkpoint/IO、保存/reload及PPL均保留原始阶段记录。新增来源hash/包装有独立实测计时，C hash只核验来源，无复用或路径改变。','', '工具故障：普通沙盒仍失败，已获授权的沙盒外执行入口可用。CPU环境诊断与GPU作业分开；没有修改全局权限，后续未继续结束辅助进程。Sol high 为用户请求，接口不能确认实际模型设置。','', '历史v51保持D且只读，已复核涉及来源文件的前后hash。未读取P3，未使用v50混合Gram，未调候选接受条件或预算。这是数值实现修订与应用验证，不作理论安全保证、官方ReQuant复现或原创贡献声明。完成后停止。']
 lines += ['', '最终硬停止：前缀接口不兼容（不是初始化数值失败）。', '', '当前 Transformers5.16.1 的 Qwen2DecoderLayer.forward 返回 [1,2048,896] Tensor；冻结 runtime_v46.advance_block 按旧 tuple 接口使用 [0][0]，得到 [896] 向量，并被 NumPy 广播至 [2048,896]。实际block0诊断证明128/128文档受影响。错误广播边界同时匹配v51和v52已记录hash，真实完整序列边界则不同；layer1已提交qkv输入hash匹配错误路径，与真实前缀输入不匹配。详见 PREFIX_COMPATIBILITY_DIAGNOSIS.json。历史输入hash一致只证明错误路径重复，并不能证明真实前缀正确。', '', '因此168个矩阵虽均执行完毕、全部局部终态检查与网格/pack/reload输出回归通过，layer1及之后的输入来源不符合冻结目标。FORMAL_REPAIR_COMPLETE.json是执行覆盖记录，不是有效完整实验PASS；已保存模型只作诊断产物，MODEL_VALIDITY.json明确禁止作为质量对照或后续有效恢复起点。', '', '原配置断言停止记录保留于diagnostics/CONFIG_ASSERTION_STOP.json。配置旧新字段经相同AutoConfig全部加载字段一致（只忽略原有两项metadata），仅通过序列化接入检查；这不能补救decoder返回类型不兼容。收尾适配器两次接入失败共27.718s，均在第一个PPL前向之前停止，源码、原锁和日志已保留。第三次发现实际输出损失异常后定位并停止，保留全部进程日志；仅有每25窗口进度记录，未生成完整窗口表，不能反推缺失窗口或汇总PPL。WT2未完成，C4未启动。没有删除窗口、重新挑模型或复用旧PPL作为本次结果。', '', f'部署收尾三个活跃作业（含失败及中断PPL）共{continuation:.3f}s；真实前缀诊断{prefixdiag:.3f}s；共同账本总计{total:.3f}/10800s，剩余{10800-total:.3f}s。强制停止段结束时间明确为保守上界，不声称精确GPU kernel时间。STAGE_TIMINGS中PPL外层timer未在强停前写完，独占PPL阶段时间标为NOT_MEASURED，其成本已包含在活跃作业账本。CPU环境诊断不冒充GPU运行；当前来源58项哈希复核CPU耗时另列。', '', '最终结果D，质量未验证。不扩大本轮唯一初始化检查修订的权限，不自行重写advance_block、改变库环境、重新修复或重新评测。要完成有效完整运行，必须先获得恢复原受支持运行库或修订前缀接口的授权，并排除错误传播后的检查点；本轮在已确认正确性错误后停止，不自动开启版本。']
 (O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 (O/'REPRODUCE.md').write_text('''v52 frozen reproduction commands (PowerShell, E:/quant-thesis)

CPU local checks:
    .\\.venv\\Scripts\\python.exe experiments/range_search_w4/results/v52_m2_fast_integer_repair_fp64_initial_review/validate_cpu.py

Original GPU regression entry (already run; do not repeat to select results):
    .\\.venv\\Scripts\\python.exe experiments/range_search_w4/results/v52_m2_fast_integer_repair_fp64_initial_review/validate_row122.py

Historical frozen formal entry (currently blocked by confirmed prefix incompatibility; do not execute):
    .\\.venv\\Scripts\\python.exe experiments/range_search_w4/results/v52_m2_fast_integer_repair_fp64_initial_review/run_v52.py

The following historical recovery command is NOT valid for the current layer1+ checkpoints; retained for source reproduction only:
    .\\.venv\\Scripts\\python.exe experiments/range_search_w4/results/v52_m2_fast_integer_repair_fp64_initial_review/run_v52.py --resume

Confirmed D stop is immutable. Do not use the diagnostic final checkpoint for inference-quality claims. No more GPU calls without explicit handling of the prefix incompatibility. Do not recreate protocol, stage inherited groups again, reset active-job ledger, or rerun PPL. Check frozen lock and inherited source hashes first.

CPU final evidence aggregation:
    .\\.venv\\Scripts\\python.exe experiments/range_search_w4/results/v52_m2_fast_integer_repair_fp64_initial_review/finalize_v52.py

Prefix source diagnostic already executed: diagnose_prefix.py (do not repeat for result selection). Deployment adapters: finish_v52.py; their failures are preserved and no completed evaluation exists.

Active GPU budget starts in budget.start() for each GPU process. Its durable segments carry prior657.305s and all new work; idle applications and CPU tool diagnosis are excluded. See ACTIVE_JOB_LEDGER.json.
''',encoding='utf8')
 manifest=[]
 for p in sorted(O.rglob('*')):
  if p.is_file() and p.name not in ['DELIVERY_MANIFEST.json','finalize.log'] and '__pycache__' not in p.parts:manifest.append(dict(path=str(p.relative_to(O)),sha256=sha(p),bytes=p.stat().st_size))
 dump('DELIVERY_MANIFEST.json',dict(status=result['category'],files=manifest,history_read_only_verified=True))
 print('V52_FINALIZED',json.dumps(result),flush=True)
if __name__=='__main__':main()
