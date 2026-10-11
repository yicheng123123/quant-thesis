"""CPU-only evidence audit and report; no model imports, scoring or selection."""
import csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
OUT=Path(__file__).resolve().parent;V18=OUT.parent/'range_w4_fullmodel_v18'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def read(n):return json.loads((OUT/n).read_text())
def csvread(n):return list(csv.DictReader((OUT/n).open(encoding='utf8')))
def csvwrite(n,rr):
 if not rr:return
 with (OUT/n).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def audit_baselines():
 baseline=[]
 for m in ['M0','M1','M2']:
  for ds,mfile,key in [('wikitext','WIKITEXT_TEST_MANIFEST.json','v16_wikitext_manifest_sha256'),('c4_fixed_budget','C4_EVAL_MANIFEST.json','v16_c4_manifest_sha256')]:
   path=V18/f'evaluation/raw/{m}_{ds}.json';r=json.loads(path.read_text());manifest=V18/'evaluation/manifests'/mfile
   assert sha(manifest)==r['evaluation_manifest_sha256']==read('PROTOCOL.json')['v18']['source_protocol'][key]
   assert r['status']=='PASS' and r['max_length']==2048 and r['stride']==512 and not r['add_special_tokens'] and not r['chat_template']
   source=Path(r['model_source'])
   for wf in r['model_weight_files']:assert sha(source/wf['name'])==wf['sha256']
   windows=list(csv.DictReader((V18/r['window_csv']).open()))
   nt=sum(int(w['valid_tokens']) for w in windows);nll=sum(float(w['window_nll']) for w in windows)/nt
   assert nt==r['evaluated_tokens']==r['input_tokens']-1 and len(windows)==r['window_count']
   assert abs(nll-r['average_nll'])<1e-10 and abs(math.exp(nll)-r['perplexity'])<1e-10
   baseline.append({'method':m,'dataset':ds,'ppl':r['perplexity'],'scored_tokens':nt,'model_files_sha256_verified':True,'manifest_sha256_verified':True,'raw_windows_reaggregated':True,'raw_result_path':str(path),'raw_result_sha256':sha(path)})
 dump('BASELINE_PROVENANCE_AUDIT.json',{'pass':True,'baselines':baseline});return baseline
def main():
 status=read('RUN_STATUS.json');sources=read('SOURCE_AUDIT.json');projection=read('COST_PROJECTION.json') if (OUT/'COST_PROJECTION.json').exists() else None
 assert read('OUTPUT_AUDIT.json')['pass'],'persisted output audit required'
 baseline=audit_baselines();complete=status['status']=='COMPLETE'
 tt=csvread('STAGE_TIMINGS.csv');cost=defaultdict(float)
 for r in tt:cost[(r['mode'],r['stage'])]+=float(r['seconds'])
 csvwrite('COST_BILL.csv',[{'mode':m,'stage':s,'measured_seconds':v,'type':'actual synchronized wall'} for (m,s),v in sorted(cost.items())])
 checks=[json.loads(l) for l in (OUT/'CORRECTNESS.jsonl').read_text().splitlines()];assert all(c.get('pass_all',True) for c in checks)
 reproduction=read('V41_A_REPRODUCTION.json');assert reproduction['pass'] and len(reproduction['trajectories'])==96
 modules=csvread('MODULE_RESULTS.csv') if (OUT/'MODULE_RESULTS.csv').exists() else [];pilot=[r for r in modules if r['mode']=='pilot'];formal=[r for r in modules if r['mode']=='formal']
 layer=[]
 for mode in ['pilot','formal']:
  for li in sorted(set(int(r['layer']) for r in modules if r['mode']==mode)):
   rr=[r for r in modules if r['mode']==mode and int(r['layer'])==li];a=sum(float(r['initial_search_mse_sum']) for r in rr);b=sum(float(r['final_search_mse_sum']) for r in rr)
   layer.append({'mode':mode,'layer':li,'modules':len(rr),'rows':sum(int(r['rows']) for r in rr),'accepted_updates':sum(int(r['accepted_updates']) for r in rr),'modified_coordinates':sum(int(r['modified_coordinates']) for r in rr),'candidate_evaluations':sum(int(r['candidate_evaluations']) for r in rr),'initial_local_mse_sum':a,'final_local_mse_sum':b,'gain_fraction':(a-b)/a})
 csvwrite('LAYER_RESULTS.csv',layer)
 comparison=[];category='D'
 if complete:
  assert len(formal)==168 and sum(int(r['rows']) for r in formal)==304128
  lock=read('CHECKPOINT_LOCK.json');assert sha(OUT/'R/checkpoint/final/model.safetensors')==lock['final_model_sha256']
  signs=[]
  for ds in ['wikitext','c4_fixed_budget']:
   r=read(f'R_{ds}.json');assert r['status']=='PASS' and r['model_sha256']==lock['final_model_sha256']
   wr=csvread(f'R_{ds}_WINDOWS.csv');assert sum(int(w['valid_tokens']) for w in wr)==r['scored_tokens'];assert abs(math.exp(sum(float(w['window_nll']) for w in wr)/r['scored_tokens'])-r['ppl'])<1e-10
   mm={b['method']:b['ppl'] for b in baseline if b['dataset']==ds}
   signs.append(r['ppl']<mm['M2'])
   comparison.append({'dataset':ds,'M0_ppl':mm['M0'],'M1_ppl':mm['M1'],'M2_ppl':mm['M2'],'R_ppl':r['ppl'],'R_minus_M1':r['ppl']-mm['M1'],'R_vs_M1_relative_percent':100*(r['ppl']/mm['M1']-1),'R_minus_M2':r['ppl']-mm['M2'],'R_vs_M2_relative_percent':100*(r['ppl']/mm['M2']-1),'improved_vs_M2':signs[-1]})
  category='A' if all(signs) else 'B' if any(signs) else 'C'
  csvwrite('PPL_COMPARISON.csv',comparison)
 verdict={'A':'两语料 PPL 均改善，已有修复路线迁移得到支持','B':'质量方向不一致，迁移不稳定','C':'两语料均未改善','D':'正确性或预算限制，质量未验证'}[category]
 selected=formal if complete else pilot
 initial=sum(float(r['initial_search_mse_sum']) for r in selected);final=sum(float(r['final_search_mse_sum']) for r in selected)
 accepted=sum(int(r['accepted_updates']) for r in selected);rows=sum(int(r['rows']) for r in selected);evals=sum(int(r['candidate_evaluations']) for r in selected)
 maxalloc=max(int(r['gpu_peak_allocated_bytes']) for r in tt);maxres=max(int(r['gpu_peak_reserved_bytes']) for r in tt)
 gpuwall=status['elapsed_seconds'];timedsum=sum(cost.values())
 decision={'category':category,'verdict':verdict,'status':status,'complete_model':complete,'full_model_modules':len(formal),'full_model_rows':sum(int(r['rows']) for r in formal),'pilot_modules':len(pilot),'pilot_rows':sum(int(r['rows']) for r in pilot),'scope_accepted_updates':accepted,'scope_candidate_evaluations':evals,'scope_local_reconstruction_gain_percent':100*(initial-final)/initial if initial else None,'PPL_comparison':comparison,'projection':projection,'measured_GPU_budget_wall_seconds':gpuwall,'measured_stage_sum_seconds':timedsum,'unitemized_wall_seconds':gpuwall-timedsum,'CPU_source_audit_seconds':sources['CPU_seconds'],'historical_M2_quantization_seconds':2115.9353341000096,'peak_allocated_bytes':maxalloc,'peak_reserved_bytes':maxres,'no_automatic_followup':True,'model_setting':'Sol high requested; interface could not confirm/switch.'}
 dump('DECISION.json',decision)
 text=f'''# v42 REPORT

## 结果与推进判断

**{category}：{verdict}。** 状态：{status['status']}。完成后停止，无自动后续实验。

正式修复覆盖 {len(formal)}/168 Linear、{sum(int(r['rows']) for r in formal):,}/304,128 行。完整 block0 pilot 覆盖 {len(pilot)}/7 Linear、{sum(int(r['rows']) for r in pilot):,}/12,672 行。

'''
 if complete:
  for r in comparison:text+=f"- {r['dataset']}：M1={r['M1_ppl']:.8f}，M2={r['M2_ppl']:.8f}，R={r['R_ppl']:.8f}；R−M2={r['R_minus_M2']:+.8f}（{r['R_vs_M2_relative_percent']:+.4f}%），R−M1={r['R_minus_M1']:+.8f}（{r['R_vs_M1_relative_percent']:+.4f}%）。\n"
 else:
  text+='未生成经部署核对的完整 R，未运行 R 的正式 PPL。不能判断局部信号是否迁移。\n\n已核查模型文件哈希、冻结 token manifest 和原始窗口重新汇总后复用的历史值：\n\n'
  for ds in ['wikitext','c4_fixed_budget']:
   mm={b['method']:b['ppl'] for b in baseline if b['dataset']==ds}
   text+=f"- {ds}：M0={mm['M0']:.8f}，M1={mm['M1']:.8f}，M2={mm['M2']:.8f}；R 及其相对变化未测量。\n"
 if projection:
  text+=f"\n完整 block0 实测 {projection['measured_complete_block0_seconds']:.3f} 秒；按全部真实相同矩阵形状，24 block ×1.35 +30 秒，正式修复保守预测 {projection['predicted_formal_repair_seconds']:.3f} 秒，另预留保存/reload/PPL {projection['save_reload_and_PPL_reserved_seconds']} 秒；门控 {'通过' if projection['go'] else '预算不支持，pilot 后停止'}。这是成本预测，不是全模型实测。\n"
 text+=f'''
## 修复与正确性

本报告统计范围为 {'完整正式修复' if complete else '已完成 pilot'}：{rows:,} 行，接受 {accepted:,} 次单坐标修改，实际候选增量评价 {evals:,} 次。同一冻结本组输入下的局部重构 MSE 合计 {initial:.12g} → {final:.12g}（下降 {100*(initial-final)/initial if initial else 0:.4f}%）。各组输入来自当前 R 前缀；跨模块合计仅是局部诊断，不是端到端损失。

v41 原始代码/协议一致：坐标升序、−1 再+1、首次最小值；预测下降阈值 max(1e−14,1e−8×初始直接 MSE)，每次接受必须直接重算严格下降且增量误差满足原容差。每行最多32次接受、128×输入维度候选评价上限；边界过滤。没有更换算法或放宽容差。

96 条 v41 A 轨迹的终态、实际评价次数、接受次数及5个预算节点状态逐一一致；4 条原始逐行参考与批量路径一致。模型搜索主循环跨行批量，非逐行 Python 搜索。每次接受后 p 按实际 FP16 网格增量更新。

正式 M2 本来是 dense FP16 部署，不存在历史 packed kernel。整数码来自可信保存的 FP32 scale/zero/g_idx，枚举全部16个网格值唯一逐位匹配，不是近似除法恢复。辅助4位打包仅用于存储与 roundtrip；部署结构和 kernel 沿用正式 dense FP16/SDPA。全部168模块零修改参数权重逐位一致，64-token 输出回归逐位一致。C 为独立未 damping 的完整 FP64 Gram，保留跨组/非对角项。

每次接受直接输出重算、固定1/8/16/24/32更新点 p 漂移和初始/最终核对见 CORRECTNESS.jsonl；原始移动、整数码、固定网格参数、输入/C哈希、实际预算及提交状态见各依赖组 checkpoint。CPU 落盘审计重新加载全部本阶段 checkpoint，逐步重放修改，核对整数码、网格、packing、实际候选评价与更新上限，并确认历史来源哈希及冻结实现/协议哈希均未改变（OUTPUT_AUDIT.json）。{'最终 checkpoint 先锁定，再 pack/reload 全168模块核对与输出容差核对，然后仅一次评测各语料。' if complete else '完整模型的最终 pack/reload 与 PPL 迁移未验证。'}

## 成本

GPU 硬预算整体实测墙时 {gpuwall:.3f} 秒 /5400秒，包含预热、v41复现、pilot及所有已执行诊断；本轮无失败重试。同步分阶段合计 {timedsum:.3f} 秒；未分项墙时 {gpuwall-timedsum:.3f} 秒单列为调度/CPU/记录等开销，不能冒称 GPU kernel 时间。峰值 allocated {maxalloc/2**30:.3f} GiB、reserved {maxres/2**30:.3f} GiB。

pilot 主要阶段实测：C 构建 {cost.get(('pilot','C_construction'),0):.3f} 秒、候选评分 {cost.get(('pilot','candidate_scoring'),0):.3f} 秒、接受更新 {cost.get(('pilot','accepted_update'),0):.3f} 秒、直接重算 {cost.get(('pilot','direct_recompute'),0):.3f} 秒。所有成本预测单独保存在 COST_PROJECTION.json。

CPU 来源审计实测 {sources['CPU_seconds']:.3f} 秒，在 GPU 总预算开始前。历史正式 M2 校准/量化成本 2115.935 秒单列复用；本轮新增成本见 STAGE_TIMINGS.csv、COST_BILL.csv。候选评价次数不是完整 Linear/GPTQ 执行节省，也不等于运行时间。校准修复和 PPL 评测分阶段记账。

## 边界

局部重构下降不保证 PPL 下降；本轮不是 ReQuant 官方复现，也不证明论文原创贡献。一次评测不代表跨 seed 稳定。不读取 P3、不重跑 GPTQ、不再搜 alpha、不改 scale/zero/g_idx、无配对/残差目标/新局部 held gate。历史 v41 是旧文章探索，本轮正式评测未用于调参。没有把局部 MSE 转换成 PPL 预测。

Sol high 为用户请求；当前接口不能确认或切换模型设置，故不声明已确认。

## 交付

PROTOCOL.json、IMPLEMENTATION_LOCK.json、SOURCE_AUDIT.json、BASELINE_PROVENANCE_AUDIT.json、run_v42.py、V41_A_REPRODUCTION.json、STAGE_TIMINGS.csv、COST_BILL.csv、MODULE_RESULTS.csv、LAYER_RESULTS.csv、各阶段 ROW_RESULTS.csv、CORRECTNESS.jsonl、group checkpoints、COST_PROJECTION.json、DECISION.json；完整时另有 CHECKPOINT_LOCK.json、R checkpoint、R 两语料原始窗口和 PPL_COMPARISON.csv。复现命令见 REPRODUCE.md。
'''
 (OUT/'REPORT.md').write_text(text,encoding='utf8')
 dump('DELIVERY_MANIFEST.json',{'files':[{'path':str(p.relative_to(OUT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.glob('*')) if p.is_file() and p.name not in ['DELIVERY_MANIFEST.json','finalize.log']],'excluded_active_log':'finalize.log','status':status['status'],'category':category})
 print(json.dumps(decision,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
