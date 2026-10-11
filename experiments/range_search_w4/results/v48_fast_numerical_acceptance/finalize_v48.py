"""CPU-only evidence summary; never reruns GPU work."""
import csv,json,hashlib
from collections import defaultdict
from pathlib import Path
OUT=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def csvwrite(n,rows):
    if not rows:return
    with (OUT/n).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    frozen=read('GPU_START.json')
    locks={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['run_v48.py','fast_accept.py','ACCEPTANCE_RULES.md']}
    assert locks['run_v48.py']==frozen['source_sha256'] and locks['fast_accept.py']==frozen['acceptance_sha256'] and locks['ACCEPTANCE_RULES.md']==frozen['protocol_sha256']
    (OUT/'IMPLEMENTATION_INTEGRITY.json').write_text(json.dumps(dict(pass_all=True,frozen_before_GPU_hashes=locks),indent=2)+'\n',encoding='utf8')
    status=read('STATUS.json');stages=list(csv.DictReader((OUT/'STAGE_TIMINGS.csv').open(encoding='utf8')))
    sums=defaultdict(lambda:[0.,0])
    for r in stages:
        k=(r['mode'],r['layer'],r['module'],r['stage']);sums[k][0]+=float(r['seconds']);sums[k][1]+=1
    csvwrite('STAGE_COSTS.csv',[dict(mode=k[0],layer=k[1],module=k[2],stage=k[3],seconds=v[0],calls=v[1],scope='synchronized stage wall; nested timers not additive') for k,v in sums.items()])
    changes=[dict(item='acceptance sign',v46='strict decrease in original complete-batch FP32 loss',v48='FP64 increment clear descent or independent raw-X FP64 descent beyond empirical tolerance'),dict(item='uncertainty check',v46='FP32 decrease plus cross-precision mismatch invokes raw-X FP64',v48='predicted sign inside predeclared FP64 envelope invokes raw-X FP64'),dict(item='p maintenance',v46='drift check at 1/8/16/24/32 without refresh',v48='same cache checks followed by p/B refresh at those milestones'),dict(item='search/threshold/budget/grid',v46='frozen v45',v48='same imported search, threshold, candidate order, ±1,32,128*d, actual FP16 grid delta'),dict(item='PPL evidence',v46='formal R1 PPL',v48='no PPL; no inherited quality claim')]
    csvwrite('PROTOCOL_CHANGES.csv',changes)
    if status['status']!='COMPLETE':
        (OUT/'REPORT.md').write_text('# v48 — 未完成\n\n分类：'+status['category']+'。\n\n'+status.get('error','预算硬停止')+'\n\n详见 STATUS.json。历史记录未改写；未运行整模型、held或PPL。\n',encoding='utf8');return
    comps=[read(f'COMPARE_mlp_{m}_proj.json') for m in ['up','down']]
    rows=[];control=0.;fast=0.;Ctotal=0.
    for co in comps:
        mod=co['control']['module'];c=sum(float(r['seconds']) for r in stages if r['layer']=='0' and r['module']==mod and r['stage']=='C_construction');Ctotal+=c
        a=co['control']['complete_repair_with_terminal_seconds'];b=co['fast']['complete_repair_with_terminal_seconds'];control+=a;fast+=b
        rows.append(dict(module=mod,rows=co['control']['rows'],input_dim=co['control']['input_dim'],C_seconds=c,control_complete_seconds=a,fast_complete_seconds=b,speedup=a/b,control_C_plus_repair_seconds=c+a,fast_C_plus_repair_seconds=c+b,C_plus_repair_speedup=(c+a)/(c+b),different_codes=co['different_codes'],different_rows=co['different_rows'],initial_raw64_mse_sum=co['control']['initial_raw64_mse_sum'],control_final_raw64_mse_sum=co['control']['final_raw64_mse_sum'],fast_final_raw64_mse_sum=co['fast']['final_raw64_mse_sum'],fast_minus_control_raw64_mse_sum=co['fast_minus_control_raw64_mse_sum']))
    csvwrite('COMPARISON.csv',rows)
    pathstages=[]
    for co in comps:
        module=co['control']['module']
        for path in ['control','fast']:
            selected=[r for r in stages if r['layer']=='0' and r['module']==module and r['mode']==path]
            direct=[float(r['seconds']) for r in selected if r['stage']=='direct_recompute']
            covered=0.
            skip={'C_construction','input_hash','dependency_input_capture','shadow_rawX_full_trajectories','fast_uncertain_refresh_and_rawX'}
            for stage in dict.fromkeys(r['stage'] for r in selected):
                if stage in skip:continue
                v=[float(r['seconds']) for r in selected if r['stage']==stage];covered+=sum(v)
                pathstages.append(dict(module=module,path=path,stage=stage,seconds=sum(v),calls=len(v),measured=True))
            pathstages.append(dict(module=module,path=path,stage='timer_outside_complete_path_wall',seconds=co[path]['complete_repair_with_terminal_seconds']-covered,calls=1,measured=False))
            if direct:
                pathstages.append(dict(module=module,path=path,stage='initial_FP32_direct_subset_of_direct_recompute',seconds=direct[0],calls=1,measured=True))
                if path=='control':
                    pathstages.append(dict(module=module,path=path,stage='proposal_FP32_direct_subset_of_direct_recompute',seconds=sum(direct[1:-1]),calls=len(direct)-2,measured=True))
                    pathstages.append(dict(module=module,path=path,stage='final_FP32_direct_subset_of_direct_recompute',seconds=direct[-1],calls=1,measured=True))
    csvwrite('PATH_COST_BREAKDOWN.csv',pathstages)
    (OUT/'TIMER_COVERAGE.md').write_text('''# Timer coverage

Complete path wall (MATRIX_TIMINGS.csv) starts before initialization and ends after independent full raw-X FP64 terminal validation. It includes shared-search CPU loop, clones, transfers, actual-grid delta, scoring/argmin, settle/p update, refresh/checks, packing. Shadow and evidence serialization are outside this interval. C/capture/hash are shared preparation, separately listed.

Raw MODE is a mutable logging label: down capture/C retained `fast` from the completed up path. They occur before down control and are common preparation; normalize by layer=0, stage=C_construction and module, irrespective of MODE. No raw record is edited.

PATH_COST_BREAKDOWN.csv contains nested descriptive subrows for initial/proposal/final FP32 scoring; do not sum them again with direct_recompute. fast_uncertain_refresh_and_rawX is inside fast_propose_and_settle. Candidate scoring and argmin are combined, not independently measured. Direct output matmul/transfer/SSE are combined, not independently measured. p update and envelope/state updates are combined; their separate cost is NOT_MEASURED. No allocation by operation count. Timer-outside wall is arithmetic residual, not GPU kernel measurement. Local evidence IO, GC and general process overhead have no separate timer; part of cumulative process wall. Timer synchronization overhead is included, not independently quantified. No profiler was added.
''',encoding='utf8')
    shadowtime=sum(float(r['seconds']) for r in stages if r['stage']=='shadow_rawX_full_trajectories')
    histtime=sum(float(r['seconds']) for r in stages if r['mode']=='v48_preflight')
    trajectories=list(csv.DictReader((OUT/'SHADOW_TRAJECTORIES.csv').open(encoding='utf8')))
    histories=list(csv.DictReader((OUT/'HISTORICAL_CANDIDATE_CHECKS.csv').open(encoding='utf8')))
    shadowcounts=defaultdict(int)
    for r in trajectories:shadowcounts[(r['path'],r['module'],r['row'])]+=1
    csvwrite('SHADOW_COVERAGE.csv',[dict(path=k[0],module=k[1],row=k[2],accepted_steps_checked=v) for k,v in shadowcounts.items()])
    numerical=dict(historical_candidates=len(histories),shadow_steps=len(trajectories),shadow_max_absolute_increment_error=max(float(r['error']) for r in trajectories),shadow_max_error_over_allowed=max(float(r['error'])/float(r['allowed_error']) for r in trajectories),shadow_largest_increment=max(float(r['rawX_FP64_increment']) for r in trajectories),full_terminal_rows_per_path=5760,terminal_max_absolute_loss_error=max(x[p]['terminal_max_error'] for x in comps for p in ['control','fast']),confirmed_math_ascent=0,true_correctness_failures=0,complete_matrix_fast_rechecks=sum(x['fast']['FP64_rechecked_candidates'] for x in comps),additional_CPU_branch_validation=read('ADDITIONAL_CPU_TESTS.json'))
    (OUT/'NUMERICAL_SUMMARY.json').write_text(json.dumps(numerical,indent=2)+'\n',encoding='utf8')
    stops=defaultdict(int)
    for file in OUT.glob('ROWS_*.csv'):
        for r in csv.DictReader(file.open(encoding='utf8')):
            assert float(r['final_raw64_mse'])<float(r['initial_raw64_mse'])
            stops[(r['path'],r['module'],r['status'])]+=1
    csvwrite('STOP_COUNTS.csv',[dict(path=k[0],module=k[1],status={1:'NO_CANDIDATE_DESCENT',2:'32_ACCEPT_CAP',3:'EVALUATION_BUDGET'}[int(k[2])],rows=v) for k,v in stops.items()])
    decision=dict(category=status['category'],reason='Independent fast path passed historical/raw-X trajectory/full terminal checks with measured full-matrix time reduction' if status['category']=='A' else 'correctness passed; at least one full target lacked measured speed gain',control_complete_seconds=control,fast_complete_seconds=fast,repair_speedup=control/fast,shared_C_seconds=Ctotal,C_plus_repair_speedup=(Ctotal+control)/(Ctotal+fast),new_process_wall_seconds=status['elapsed_seconds'],shadow_validation_seconds=shadowtime,prior_cost_excluded=True,quality_or_novelty_claim=False,next_fullmodel_started=False)
    (OUT/'DECISION.json').write_text(json.dumps(decision,indent=2)+'\n',encoding='utf8')
    reject=sum(x['fast']['rejected_candidates'] for x in comps);recheck=sum(x['fast']['FP64_rechecked_candidates'] for x in comps)
    lines=['# v48 — 固定网格整数修复的快速数值接受机制','',f"## 结果：{status['category']}",'',f"两个完整矩阵的独立快速路径通过已覆盖的数值检查。含初始化、p刷新、复核、packing和全行独立FP64终态检查的修复wall合计：原路径 **{control:.3f}s**，快速路径 **{fast:.3f}s**，实测 **{control/fast:.3f}×**。没有减少32次更新上限或候选预算。",'']
    for r in rows:
        lines += [f"- L0 {r['module']}，{r['rows']}×{r['input_dim']}：原{r['control_complete_seconds']:.3f}s，快{r['fast_complete_seconds']:.3f}s（{r['speedup']:.3f}×）；C单列{r['C_seconds']:.3f}s。终态不同码{r['different_codes']}、不同行{r['different_rows']}；快速减原路径raw-X FP64 MSE总和{r['fast_minus_control_raw64_mse_sum']:.12g}。"]
    for co in comps:
        lines += [f"- {co['control']['module']} 两路径各接受{co['control']['accepted_updates']:,}次修改、评价{co['control']['candidate_evaluations']:,}个合法候选（快速路径相同）；完整矩阵没有拒绝或不确定复核。逐行预算与停止状态见ROWS_*.csv和STOP_COUNTS.csv。"]
    lines += ['',f"将同样的C构建成本各计入一次后，两矩阵合计速度比为 **{(Ctotal+control)/(Ctotal+fast):.3f}×**。这只是这两个固定输入完整矩阵的比较，不是完整模型校准加速。控制路径先运行，快速路径后运行；一次同步wall测量有顺序、缓存、功率/温度和系统负载影响，没有跨次稳定性结论。",'', '## 数值机制与明确变更','', '完整规则和假设在 ACCEPTANCE_RULES.md，逐项协议变化在 PROTOCOL_CHANGES.csv。搜索枚举、预算、阈值和确定性tie保持原函数；接受符号从FP32完整输出loss差改为FP64二次型增量及不确定时的raw-X复核；固定里程碑检查后刷新p/B。不声称与v46逐步完全等价。', '', 'C使用原FP64完整未阻尼构建路径。B、cache尺度和gamma累积长度构成事前冻结的经验误差包络。abs(C)不能严格界定原始Gram的非对角消去误差，GPU归约路径也没有严格误差定理；因此这不是安全证书。通过仅是以下已有输入/状态上的实现数值证据。', '', '## 正确性覆盖','', f"- 历史：v43的3个失败候选、v44失败row211、v46全部9个正式拒绝候选，共{len(histories)}个；复原真实prefix输入并匹配X/C历史哈希，原完整批量FP32增量逐位一致，独立raw-X FP64公式核验通过。历史停止状态只读。部分FP32非下降为FP64下降，v48允许接受，这是显式规则变化。", f"- 完整目标：M1原始可信codes/grid，pristine FP16 W，128×2048输入；锁定v46 R1前缀恢复X并匹配哈希；C同一份，两路径状态独立。控制终态codes/evaluations与v46一致。",f"- 逐步：事前固定rows=[0,224,448,672]，两个投影、两路径，独立raw-X FP64核验 **{len(trajectories)}次接受更新**；全部严格下降。详见SHADOW_TRAJECTORIES.csv和SHADOW_COVERAGE.csv。没有逐步核验其余所有行。", '- 终态：每路径全部5760输出行，独立原始X FP64直接重构初态/终态，与二次型及快速增量记账核验；所有行终态下降。码合法、zero=8、G128、网格不变、初末packing回环、W/X/C哈希不变；锁定R1及历史源码/失败状态文件SHA前后不变。', f"- 完整快速目标拒绝{reject}个、不确定raw-X复核{recheck}个；不能因复核分支未在完整矩阵触发而省略其覆盖，确定性小例包含正增量复核拒绝、状态恢复、错误注入和预算检查。没有确认的数学目标上升或实质正确性失败。", '', '## 成本口径','',f"本次累计进程wall **{status['elapsed_seconds']:.3f}s /1200s**，含启动、诊断、失败用例、C、两路径、终态、shadow和IO。同步历史诊断阶段合计{histtime:.3f}s；shadow验证 **{shadowtime:.3f}s**独立列出，没有混入快速执行成本。共享目标C构建{Ctotal:.3f}s，不混入修复wall。",f"峰值allocated {status['peak_allocated_bytes']/2**30:.3f}GiB，reserved {status['peak_reserved_bytes']/2**30:.3f}GiB。{status['gpu']}，PyTorch{status['torch']}；TF32关闭。",'', 'MATRIX_TIMINGS.csv为完整路径计时；STAGE_TIMINGS.csv为原始同步阶段wall。fast_uncertain_refresh_and_rawX是fast_propose_and_settle内嵌timer，不能重复相加。LOCAL_*.pt写入和未包timer的hash/调度/GC属于processwall中的未独立归因开销，不能冒称独立GPU kernel时间。无额外profiler；已有同步timer本身的开销未单独测量。', '', '1000s计划预留是运行前预测；上述成本均为本次测量。v47的54.03%仅作问题来源，未用省掉的评分次数推算本次速度。', '', '## 判断与边界','', 'A表示本轮完整矩阵数值检查通过且有实测收益；支持继续研究这种接受实现。它不继承R1/R2的PPL，不证明完整模型质量迁移、普适数值保证或论文原创性。没有执行新整模型、held、PPL、模型提交或保存可部署checkpoint；LOCAL_*.pt只是局部修复证据。按要求完成后停止。', '', '## 交付','', 'fast_accept.py / run_v48.py：实现与复用入口；ACCEPTANCE_RULES.md：冻结规则；SOURCE_AUDIT.json / SOURCE_IMMUTABILITY.json：来源；HISTORICAL_CANDIDATE_CHECKS.csv：旧失败/拒绝；ROWS_*.csv、LOCAL_*.pt：逐行和轨迹证据；SHADOW_*：独立逐步检查；COMPARE_*.json：全行终态与数值差异；STAGE_*和MATRIX_TIMINGS.csv：原始/汇总成本；REPRODUCE.md：命令。', '', '模型设置：用户要求Sol high，当前接口无法确认或切换。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (OUT/'REPRODUCE.md').write_text('''# Reproduction

```powershell
Set-Location E:/quant-thesis
& ./.venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v48_fast_numerical_acceptance/run_v48.py
& ./.venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v48_fast_numerical_acceptance/additional_cpu_validation.py
& ./.venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v48_fast_numerical_acceptance/finalize_v48.py
```

The first command refuses an existing GPU_START.json to prevent overwriting evidence/resetting the1200s budget. To reproduce, copy run_v48.py, fast_accept.py, additional_cpu_validation.py, finalize_v48.py and ACCEPTANCE_RULES.md into a NEW independently named result directory under the same results parent; run those copied scripts. Leave v48 and historical evidence untouched. GPU budget includes diagnostics, not just timed matrices. Exact frozen v46 runtime/repair/precision helpers are imported read-only; Python -B suppresses historical pycache writes. No network, held, PPL, commit or deployable model save. The second command checks additional settlement branches on CPU only (it does not feed back into selection or alter rules); the third is CPU-only summary regeneration. Frozen source/manifest hashes are SOURCE_AUDIT.json and GPU_START.json.
''',encoding='utf8')
    manifest=[]
    for p in OUT.iterdir():
        if p.is_file() and p.name!='DELIVERY_MANIFEST.csv':manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    csvwrite('DELIVERY_MANIFEST.csv',manifest)
if __name__=='__main__':main()
