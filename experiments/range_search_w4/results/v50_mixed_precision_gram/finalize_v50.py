"""CPU-only closeout of the externally interrupted frozen v50 attempt."""
import csv,hashlib,json,math,time
from pathlib import Path
from collections import defaultdict
OUT=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def writecsv(n,rows):
    with (OUT/n).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    start=read('GPU_START.json');protocol=read('PROTOCOL.json');hist=read('HISTORICAL_COST_EVIDENCE.json')
    assert not (OUT/'STATUS.json').exists(),'do not overwrite an execution status'
    audit=read('SOURCE_AUDIT.json');assert all(sha(x['path'])==x['sha256'] for x in audit)
    assert sha(OUT/'run_v50.py')==start['runner_sha256']==protocol['runner_sha256']
    assert sha(OUT/'NUMERICAL_RULES.md')==start['rules_sha256'] and sha(OUT/'mixed_gram.py')==protocol['new_path_sha256']
    dump('SOURCE_IMMUTABILITY.json',dict(pass_all=True,historical_files=audit,frozen_algorithm_and_rules_unchanged=True))
    stages=list(csv.DictReader((OUT/'STAGE_TIMINGS.csv').open(encoding='utf8')));matrices=list(csv.DictReader((OUT/'MATRIX_TIMINGS.csv').open(encoding='utf8')))
    assert len(matrices)==1 and matrices[0]['path']=='baseline' and matrices[0]['module']=='mlp.up_proj'
    last=max(float(r['elapsed_seconds']) for r in stages);group=defaultdict(list)
    for r in stages:group[(r['mode'],r['module'],r['stage'])].append(r)
    cost=[]
    for (path,module,stage),rr in group.items():cost.append(dict(path=path,module=module,stage=stage,calls=len(rr),seconds=sum(float(x['seconds']) for x in rr),nested_in='fast_propose_and_settle' if stage=='fast_uncertain_refresh_and_rawX' else '',complete_matrix=(path=='baseline' and module=='mlp.up_proj')))
    writecsv('STAGE_COSTS.csv',cost)
    lines=(OUT/'run.log').read_text(encoding='utf8').splitlines();moves=[]
    for line in lines:
        a=line.split()
        if len(a)==8 and a[:2]==['FAST','mlp.up_proj']:moves.append(dict(attempt=int(a[2]),accepted=int(a[4]),rejected=int(a[6])))
        elif len(a)==7 and a[:2]==['FAST','mlp.up_proj']:moves.append(dict(attempt=int(a[2]),accepted=int(a[4]),rejected=int(a[6])))
    baseline_logs=[x for x in moves[:32]];mixed_logs=[x for x in moves[32:]]
    assert len(baseline_logs)==32 and len(mixed_logs)==23 and mixed_logs[-1]['attempt']==23
    raw=sum(float(r['seconds']) for r in stages if r['mode']=='mixed' and r['stage']=='fast_uncertain_refresh_and_rawX')
    bgram=sum(float(r['seconds']) for r in stages if r['mode']=='baseline' and r['stage']=='C_construction')
    mgram=sum(float(r['seconds']) for r in stages if r['mode']=='mixed' and r['stage']=='C_mixed_construction_and_statistics')
    shadows=list(csv.DictReader((OUT/'SHADOW_TRAJECTORIES.csv').open(encoding='utf8')))
    interruption=dict(status='EXTERNALLY_INTERRUPTED_NO_FINAL_HANDLER_RECORD',cause='UNKNOWN; no traceback or watchdog status was written',original_pid=start['pid'],process_absent_confirmed=True,confirmed_local='2026-10-08T22:06:05+08:00',last_run_log_local='2026-10-08T03:24:06+08:00',OS_last_boot_local='2026-10-05T05:39:39+08:00',last_completed_timer_elapsed_seconds=last,original_cap_seconds=1200,actual_process_end_time='NOT_RECORDED',full_GPU_active_wall_seconds='NOT_MEASURED',inflight_work_after_last_timer='NOT_MEASURED',external_wall_gap_is_not_GPU_compute_time=True,remaining_GPU_budget_cannot_be_reliably_reconstructed=True,no_second_GPU_attempt=True)
    dump('INTERRUPTION_AUDIT.json',interruption)
    coverage=[dict(path='baseline',module='mlp.up_proj',full_repair=True,all_row_terminal_rows=4864,sampled_trajectory_rows=4,sampled_accepted_updates=len(shadows),final_codes_saved=True),dict(path='mixed',module='mlp.up_proj',full_repair=False,all_row_terminal_rows=0,sampled_trajectory_rows=0,sampled_accepted_updates=0,final_codes_saved=False),dict(path='baseline',module='mlp.down_proj',full_repair=False,all_row_terminal_rows=0,sampled_trajectory_rows=0,sampled_accepted_updates=0,final_codes_saved=False),dict(path='mixed',module='mlp.down_proj',full_repair=False,all_row_terminal_rows=0,sampled_trajectory_rows=0,sampled_accepted_updates=0,final_codes_saved=False)]
    writecsv('VALIDATION_COVERAGE.csv',coverage)
    missing=[dict(item=x,status='NOT_COMPLETED') for x in ['mixed up_proj terminal codes and all-row raw-X check','mixed up_proj sampled full trajectory check','both down_proj repair paths','C numerical error versus reference','selected candidate/acceptance/code comparison for completed mixed matrix','complete two-matrix total wall/peak comparison']]
    writecsv('MISSING_EVIDENCE.csv',missing)
    decision=dict(category='D',full_comparison_complete=False,independent_acceptance_design=True,independent_complete_matrix_execution_verified=False,numerical_checks_pass='baseline up_proj only; mixed whole-matrix coverage incomplete',new_Gram_kernel_has_measured_speedup=bgram/mgram,observed_complete_matrix_gain=False,mixed_partial_raw_rechecks_seconds=raw,baseline_complete_up_pipeline_seconds=float(matrices[0]['pipeline_seconds']),worth_full_model_validation=False,reason='Mixed up_proj raw-X rechecks already exceed complete baseline up_proj cost; missing terminal/down evidence plus unrecorded process termination. Do not infer full-matrix speedup or numerical pass.',algorithm_or_threshold_adjusted_after_results=False,held=False,PPL=False,model_committed=False)
    dump('DECISION.json',decision)
    dump('STATUS.json',dict(status='CLOSED_INCOMPLETE',category='D',execution_end='external unclean termination; exact cause/time unknown',last_measured_elapsed_seconds=last,cap_seconds=1200,complete_paths=1,planned_paths=4,mixed_logged_accepted_rounds=23,mixed_logged_accepted_updates=sum(x['accepted'] for x in mixed_logs),mixed_logged_rejected_updates=sum(x['rejected'] for x in mixed_logs),mixed_candidate_evaluations='NOT_SAVED',mixed_rechecked_candidates='NOT_SAVED; 23 timed recheck calls',GPU_actual_total_seconds='NOT_MEASURED',no_budget_reset=True,second_GPU_attempt=False))
    b=matrices[0];hshare=100*hist['v49_C_successful_seconds']/hist['v49_formal_block_wall_seconds']
    text=f'''# v50 混合精度 Gram 构建试验：D，完整比较未完成

冻结流程可独立构建混合精度 C 并进行候选接受，不使用原 FP64 C 作生产选择。实际运行完成了 baseline up_proj，并执行到 mixed up_proj 第23轮；进程随后无终态/异常处理记录地退出，精确原因与退出时间未知。2026-10-08 22:06 检查时原 PID 已不存在。日志/计时最后更新在03:24。系统未在这段时间重新启动，不能将中间时间差当作 GPU 计算时间。

不能重建确切的剩余累计 GPU 预算，故没有重启或重置1200秒预算。冻结代码和规则未改变。本轮的交付结果是缺失范围明确的 D，不把不完整路径写成数值PASS。

## 已观测的成本结论

up_proj 的同步 C 构建阶段：原 FP64 {bgram:.6f}s；FP32块内Gram＋FP64累加与列能量统计 {mgram:.6f}s，Gram阶段实测比 {bgram/mgram:.3f}×。但是混合路径已完成23次不确定候选raw-X复核调用，嵌套同步时间合计 {raw:.6f}s，已超过原路径完整up_proj的C＋修复＋终态检查 {float(b['pipeline_seconds']):.6f}s。即使剩余工作耗时为零，混合up_proj也不能在本次规则下取得完整矩阵总成本收益。down_proj尚未开始，不能将up结果推广成down或整模型加速。

最终建议：**不进入下一轮整模型验证**。当前不确定性流程的必要复核开销吞没了已观测的 Gram 节省，且混合完整矩阵数值覆盖未完成。未调整触发规则或用验证最大误差调阈值；也不自动开展新方案。

## v49成本与局部/全模型差距

v49完整24block净修复4543.075288s；全部原始C事件97次/1755.010978s，含失败qkv一次/2.741256s。有效正式96次C为1752.269722s，占净修复{hshare:.2f}%。真实输入N=262144，4096-token块：72次896×896 C（qkv、o、up_gate各24次）合计186.528726s；24次4864×4864 C（down）合计1565.740995s。完整调用清单见V49_GRAM_CALLS.csv。

sibling已经在run_layer内共享C：qkv三个投影共用一次，up/gate两个投影共用一次。不是本轮新增优化。v47的54.03%是v46 proposal直接评分占比，不能代替v49实测份额。

v48局部1.508×来自两个L0 MLP完整矩阵并计入C；v46/v49完整模型净block wall比是1.207×。投影形状/输出行混合和分母不同。v48局部control为比较增加了全行独立FP64终态检查；v46原正式路径的终态使用FP32直接loss，v49则为168个Linear执行完整raw-X FP64终态检查，有效成功阶段2123.186194s（原始169条事件2124.068145s，含一次失败目标）。这项检查和有效C合计约85.30%的v49净block wall，限制了全模型收益。局部倍率不能直接外推；这些是不同覆盖、相同口径核查后的成本证据，不是因果分摊或运算量估计。

## 数值规则与独立性

唯一新C路径固定4096-token块、FP32块内完整GEMM、FP64累加/归一化/对称化；TF32 off且autocast禁用。FP32舍入不会被随后转FP64恢复。非对角与跨G128项全部保留。数值定义、运算假设、gamma32/gamma64及列能量h、rho均在GPU前冻结于NUMERICAL_RULES.md/PROTOCOL.json。

附加Gram不确定性为rho*(2|delta|h_j sum_i |e_i|h_i + delta²h_j²)。它仅依赖当前X的能量统计及当前e/真实FP16网格delta，无需先算FP64 C。继承v48搜索函数、256输出行chunk、确定性坐标/-1/+1顺序、合法码、32接受上限、128*d候选评价、拒绝集合与p刷新里程碑。

预测明显下降时用附加包络判断；不确定时独立raw-X FP64计算mean(2ab+b²)。原same-C缓存/公式检查保持严格；Gram误差只用于预测与raw-X的比较允许量，不用于放宽raw-X下降的符号判定。它依赖未经逐kernel证明的FP32归约假设，是经验数值流程，不是安全证书或全程理论单调性保证。

原FP64 C只作为baseline及之后计划的验证参考，未传入混合路径生产选择。本次混合未结束，因此C误差对照尚未生成。

## 来源与实际数值覆盖

锁定v49模型SHA为8a4e2aed53ed0f72c20780ca80ad7a2f751c8babb1c509d8f532054f150ecaf8。L0 up/down输入从该模型前缀重放，两份X哈希均匹配正式提交组；输入来自正式128×2048文章/token。兩路径使用同一可信原始M1 codes/grid和只读pristine FP16 W，目标模型未提交或更改。

CPU确定性检查通过：真实下降接受、正增量拒绝、拒绝状态/blacklist恢复、same-C缓存及公式错误注入、原更新/评价上限、固定构造混合Gram参照。baseline up_proj C哈希、终态整数码与评价次数匹配v49，全部4864行终态raw-X FP64检查通过；事前固定[0,224,448,672]四行完整轨迹的{len(shadows)}次接受更新全部经过独立raw-X核验。

baseline up_proj：接受{b['accepted_updates']}，候选评价{b['candidate_evaluations']}，拒绝{b['rejected_candidates']}，不确定复核{b['rechecks']}；raw-X FP64初末MSE总和{float(b['initial_raw64_sum']):.12f}→{float(b['final_raw64_sum']):.12f}。最大终态误差{float(b['terminal_max_error']):.3g}，max p漂移{float(b['max_p_drift']):.3g}。C、初始化/修复、全行终态实测分别为{float(b['C_seconds']):.6f}、{float(b['repair_seconds']):.6f}、{float(b['terminal_seconds']):.6f}s；其完整pipeline wall为{float(b['pipeline_seconds']):.6f}s。独立shadow成本单列于STAGE_COSTS.csv。

mixed up_proj仅有23轮接受日志、24次已完成候选评分timer及23次已完成raw-X复核timer；没有完整终态codes、精确累计候选评价/复核行数或shadow结果。无已记录的数值异常不等于PASS。down两条路径未执行。终態码差异、C误差、完整接受差异、完整矩阵总成本/峰值比较均标NOT_COMPLETED，见MISSING_EVIDENCE.csv/VALIDATION_COVERAGE.csv。

## 预算与计时覆盖

最后完整timer的GPU_START elapsed为{last:.6f}s；精确进程结束时间及之后在途工作时间未记录，不能将此数写成完整GPU总成本。启动上限1200s的watchdog和所有deadline保持原样；没有第二次启动。中间约18小时的外部时间间隔不是GPU消耗测量。

STAGE_TIMINGS.csv保留已完成阶段原始同步计时；STAGE_COSTS.csv标明fast_uncertain_refresh_and_rawX内嵌于fast_propose_and_settle，不能重加。C计时包含混合路径列能量/触发统计；矩阵pipeline含初始化、原FP32阈值回归、搜索、刷新、必要复核及全行终态。验证shadow、输入重放/cache IO、本地证据IO另列。未关闭的timer没有数据；不按操作数补算。原FP64 baseline C拟用于后续验证，但本次没有额外构建验证C或完成C差值计算。

## 交付与复现

源码：mixed_gram.py、run_v50.py；冻结规则：NUMERICAL_RULES.md、PROTOCOL.json；原始/汇总计时：STAGE_TIMINGS.csv、MATRIX_TIMINGS.csv、STAGE_COSTS.csv；来源与不变检查：SOURCE_AUDIT.json、SOURCE_IMMUTABILITY.json；输入哈希：INPUT_SOURCE_CHECKS.json；现存轨迹：SHADOW_TRAJECTORIES.csv；中断/缺失/判断：INTERRUPTION_AUDIT.json、VALIDATION_COVERAGE.csv、MISSING_EVIDENCE.csv、DECISION.json。

复现入口见REPRODUCE.md。现有目录保留GPU_START，runner会拒绝静默重跑/预算重置。本轮不运行整模型修复、held、PPL，不保存可部署的新量化模型，不自动启动下一版。Sol high模型设置无法通过本接口确认或切换。
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf8')
    (OUT/'REPRODUCE.md').write_text('''# v50 reproduction

CPU preparation (only before the first GPU start):

```powershell
& E:/quant-thesis/.venv-gptq/Scripts/python.exe -B -X utf8 E:/quant-thesis/experiments/range_search_w4/results/v50_mixed_precision_gram/prepare_v50.py
```

Frozen bounded GPU entry:

```powershell
& E:/quant-thesis/.venv-gptq/Scripts/python.exe -u -B -X utf8 E:/quant-thesis/experiments/range_search_w4/results/v50_mixed_precision_gram/run_v50.py
```

The existing evidence directory refuses another launch because GPU_START.json exists. Do not delete the start marker, reset the budget or overwrite this attempt to make a fresh run appear completed. CPU closeout uses finalize_v50.py; it refuses overwriting an existing STATUS.json.
''',encoding='utf8')
    manifest=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.suffix not in ['.npy'] and p.name!='DELIVERY_MANIFEST.csv':manifest.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    writecsv('DELIVERY_MANIFEST.csv',manifest)
    print(json.dumps(decision,ensure_ascii=False))
if __name__=='__main__':main()
