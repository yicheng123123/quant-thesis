"""CPU report of locked pilot outputs. No experimental selection."""
import csv,json,math,statistics
from collections import defaultdict,Counter
from pathlib import Path
OUT=Path(__file__).resolve().parent
def read(n):
    with (OUT/n).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
    with (OUT/n).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def metric(pairs,init,split,objective,comparison,node):
    base=math.fsum(b*32768 for _,b,_,_ in pairs);new=math.fsum(n*32768 for _,_,n,_ in pairs)
    gains=[(b-n)/b for _,b,n,_ in pairs]
    w=sum(b-n>eps for _,b,n,eps in pairs);l=sum(n-b>eps for _,b,n,eps in pairs)
    bytarget=defaultdict(lambda:[0,0])
    for target,b,n,eps in pairs:
        bytarget[target][0]+=b*32768;bytarget[target][1]+=n*32768
    largest=max(bytarget,key=lambda t:bytarget[t][0]-bytarget[t][1]);lb,ln=bytarget[largest]
    return {'init':init,'split':split,'objective':objective,'comparison':comparison,'node_over_dim':node,'rows':len(pairs),'base_SSE':base,'new_SSE':new,'gain_percent':100*(base-new)/base,'W':w,'T':len(pairs)-w-l,'L':l,'median_gain_percent':100*statistics.median(gains),'worst_gain_percent':100*min(gains),'degradation_gt10percent':sum(g<-.1 for g in gains),'largest_contribution_target':largest,'largest_contribution_SSE':lb-ln,'leave_largest_out_gain_percent':100*((base-lb)-(new-ln))/(base-lb),'positive_targets':sum(b>n for b,n in bytarget.values()),'target_count':len(bytarget)}
def main():
    rows=read('ROW_RESULTS.csv');budget=read('BUDGET_RESULTS.csv');summ=[];contributions=[]
    lookup={(r['unit_id'],r['init'],r['algorithm']):r for r in rows}
    for init in ['alpha1','scalar10']:
      for split in ['S0','S1','ALL']:
       selected=[r for r in rows if r['init']==init and r['algorithm']=='C' and (split=='ALL' or r['split']==split)]
       for obj in ['search','held']:
        for comparison in ['A_vs_init','B_vs_init','C_vs_init','C_vs_A','C_vs_B']:
          alg,basealg=comparison.split('_vs_');pairs=[]
          for cr in selected:
            r=lookup[(cr['unit_id'],init,alg)];target=f'L{r["layer"]}_{r["projection"]}'
            b=float(r[f'{obj}_initial_mse']) if basealg=='init' else float(lookup[(cr['unit_id'],init,basealg)][f'{obj}_final_mse'])
            n=float(r[f'{obj}_final_mse']);pairs.append((target,b,n,float(r['epsilon'])))
          summ.append(metric(pairs,init,split,obj,comparison,'FINAL'))
          bytarget=defaultdict(list)
          for p in pairs:bytarget[p[0]].append(p)
          for target,pp in bytarget.items():
            b=sum(x[1]*32768 for x in pp);n=sum(x[2]*32768 for x in pp)
            contributions.append({'init':init,'split':split,'objective':obj,'comparison':comparison,'target':target,'base_SSE':b,'new_SSE':n,'saving_SSE':b-n,'gain_percent':100*(b-n)/b})
    bl={(r['unit_id'],r['init'],r['algorithm'],r['node_over_dim']):r for r in budget}
    for node in ['16','32','64','96','128']:
      for init in ['alpha1','scalar10']:
       for split in ['S0','S1','ALL']:
        for obj in ['search','held']:
         for comp in ['C_vs_A','C_vs_B']:
          basealg=comp[-1];pairs=[]
          for r in budget:
            if r['init']!=init or r['algorithm']!='C' or r['node_over_dim']!=node or (split!='ALL' and r['split']!=split):continue
            b=bl.get((r['unit_id'],init,basealg,node))
            if b is None:continue
            target=r['target'].rsplit('_',1)[0];pairs.append((target,float(b[f'{obj}_mse']),float(r[f'{obj}_mse']),float(r['epsilon'])))
          if pairs:summ.append(metric(pairs,init,split,obj,comp,node))
    write('POOLED_SUMMARY.csv',summ);write('TARGET_CONTRIBUTIONS.csv',contributions)
    costs=[dict(r,attempt='successful') for r in read('COST_BILL.csv')];failed_bill=defaultdict(float)
    for p in sorted(OUT.glob('failed_attempt_*/COST_BILL.csv')):
        with p.open(encoding='utf8',newline='') as f:
            for r in csv.DictReader(f):costs.append(dict(r,attempt=p.parent.name));failed_bill[(r['stage'],r['algorithm'])]+=float(r['seconds'])
    wall_guard=json.loads((OUT/'GPU_WALL_BUDGET_GUARD.json').read_text(encoding='utf-8-sig')) if (OUT/'GPU_WALL_BUDGET_GUARD.json').exists() else None
    if wall_guard:costs.append({'stage':'between_attempt_diagnostics','target':'serialization_key_fix','algorithm':'','seconds':wall_guard['between_attempt_diagnostic_seconds'],'kind':'measured wall time between GPU attempts','attempt':'diagnostics'})
    write('COST_ALL_ATTEMPTS.csv',costs);bill=defaultdict(float)
    for r in costs:bill[(r['stage'],r['algorithm'])]+=float(r['seconds'])
    write('COST_SUMMARY.csv',[{'stage':s,'algorithm':a,'measured_seconds':v,'failed_attempt_seconds':failed_bill[(s,a)],'successful_attempt_seconds':v-failed_bill[(s,a)] if s!='between_attempt_diagnostics' else 0,'estimated_seconds':''} for (s,a),v in bill.items()])
    mech=read('MECHANISM.csv');checks=read('NUMERICAL_CHECKS.csv');status=json.loads((OUT/'RUN_STATUS.json').read_text());audit=json.loads((OUT/'SOURCE_AUDIT.json').read_text())
    def get(init,split,obj,comp,node='FINAL'):return next(r for r in summ if (r['init'],r['split'],r['objective'],r['comparison'],str(r['node_over_dim']))==(init,split,obj,comp,str(node)))
    whole_wall=wall_guard['whole_round_wall_seconds'] if wall_guard else status['elapsed_seconds']
    lines=['# 固定网格整数码联合修复：有界探索报告','',f'状态：{status["status"]}；288 条算法轨迹；GPU 全轮墙钟实测 {whole_wall:.3f} 秒（含失败、诊断及退出观测；硬上限 900 秒）。活动尝试累计 {status["elapsed_seconds"]:.3f} 秒。','',
      '## 结果','', '以下百分比为 pooled SSE 相对收益，正数代表损失下降；旧文章、代表行、固定 M2-prefix 输入，属于探索性诊断。']
    for init in ['alpha1','scalar10']:
      lines.extend(['',f'### 初始化 {init}',''])
      for split in ['S0','S1','ALL']:
        values=[get(init,split,'held',f'{a}_vs_init') for a in ['A','B','C']]
        ca=get(init,split,'held','C_vs_A');cb=get(init,split,'held','C_vs_B')
        lines.append(f'- {split} held：A/B/C 相对初始化 {values[0]["gain_percent"]:.4f}% / {values[1]["gain_percent"]:.4f}% / {values[2]["gain_percent"]:.4f}%；C 相对 A {ca["gain_percent"]:.4f}%，相对 B {cb["gain_percent"]:.4f}%。')
      for comp in ['C_vs_init','C_vs_A','C_vs_B']:
        h=get(init,'ALL','held',comp);s=get(init,'ALL','search',comp)
        lines.append(f'- {comp}：search {s["gain_percent"]:.4f}%；held W/T/L={h["W"]}/{h["T"]}/{h["L"]}，行收益中位数 {h["median_gain_percent"]:.4f}%，最差 {h["worst_gain_percent"]:.4f}%，held >10% 退化 {h["degradation_gt10percent"]} 行。最大贡献 {h["largest_contribution_target"]}；移除后 held {h["leave_largest_out_gain_percent"]:.4f}%（{h["positive_targets"]}/{h["target_count"]} target 为正）。')
    lines.extend(['','## 共同候选预算','', '候选增量评分在 16/32/64/96/128 × 输入维度节点精确暂停，未完成邻域保留当前状态，完整邻域结束才接受。提前停止或 32 次更新上限不虚填评价次数。共同节点比较匹配的是预算上限；实际使用次数和未完成邻域另存 SEARCH_TRAJECTORIES.csv，不能把同上限称为实际耗时相同。'])
    for init in ['alpha1','scalar10']:
      for node in [16,32,64,96,128]:
        ca=get(init,'ALL','search','C_vs_A',node);cb=get(init,'ALL','search','C_vs_B',node);ha=get(init,'ALL','held','C_vs_A',node);hb=get(init,'ALL','held','C_vs_B',node)
        lines.append(f'- {init}，{node}d：C 对 A/B search {ca["gain_percent"]:.4f}%/{cb["gain_percent"]:.4f}%；锁定状态 held {ha["gain_percent"]:.4f}%/{hb["gain_percent"]:.4f}%。')
    pair_ct=Counter((r['algorithm'],r['both_singles_non_decreasing']) for r in mech)
    maxerr=max((float(r['delta_error']) for r in checks if r.get('delta_error')),default=0);maxdrift=max((float(r['p_drift_max']) for r in checks if r.get('p_drift_max')),default=0)
    lines.extend(['','## 机制与数值正确性','',f'- 接受双点更新 B={sum(r["algorithm"]=="B" for r in mech)}，C={sum(r["algorithm"]=="C" for r in mech)}；两个单独修改都不下降而联合下降：B={pair_ct[("B","True")]}, C={pair_ct[("C","True")]}。只说明当前状态的交叉项效应，不说明逃离单点局部最优。',f'- 增量公式对直接输出复算最大绝对差 {maxerr:.6g} MSE；缓存 p 最大漂移 {maxdrift:.6g}。每个接受步骤严格下降检查，初始/最终直接核对，固定更新抽样 p 核对；无容差放宽。',f'- {audit["candidate_banks"]} 个历史 bank 缓存逐位唯一匹配 16 个原生网格值，整数码来源为精确有效网格逆查；历史前向整数轨迹未存，不宣称恢复。scale/zero/g_idx 冻结，零更新权重逐位一致。零方差列与其误差见 ZERO_VARIANCE.csv。'])
    lines.extend(['','## 实测成本',''])
    for stage in sorted(set(s for s,a in bill)):
        lines.append(f'- {stage}：{sum(v for (s,a),v in bill.items() if s==stage):.3f} 秒。')
    measured=sum(bill.values());lines.append(f'- 全轮 {whole_wall:.3f} 秒；分段记录和 {measured:.3f} 秒，余量 {whole_wall-measured:.3f} 秒为 Python 编排、未单独计时诊断/初始化、记录及退出观测等。候选评价实际 {status["total_evaluations"]:,} 次。CPU 来源准备实测 {audit["cpu_prep_wall_seconds"]:.3f} 秒。未估算整模型时间、GPTQ 节省或 PPL。')
    if wall_guard:lines.append(f'- 首次尝试因统计缓存键冲突失败：配对边字段覆盖了 C 字段，发生在任何候选评价之前；全部目标初始检查已通过。失败活跃阶段 55.489 秒、中间 CPU 诊断/修正 {wall_guard["between_attempt_diagnostic_seconds"]:.3f} 秒纳入全轮。失败源码、检查点与账单保存于 failed_attempt_01；重试独立重建所有 C，未删除目标、未改变算法或接受规则。正常成功阶段成本与失败成本在 COST_SUMMARY.csv 分列，COST_ALL_ATTEMPTS.csv 保留每项实际记录。')
    for alg in ['A','B','C']:
        rr=[r for r in rows if r['algorithm']==alg];lines.append(f'- {alg}：增量候选 {sum(int(r["evals"]) for r in rr):,} 次；候选评分 {bill[("candidate_scoring",alg)]:.3f} 秒；接受更新 {bill[("accepted_update",alg)]:.3f} 秒；直接复算 {bill[("direct_recompute",alg)]:.3f} 秒。构建 C 与图为共享开销，另列，不能把公平次数等同公平时间。')
    signal=any(all(get(i,s,'held',f'{a}_vs_init')['gain_percent']>0 for i in ['alpha1','scalar10'] for s in ['S0','S1']) for a in ['A','B','C'])
    advantage=all(get(i,s,'held',c)['gain_percent']>0 for i in ['alpha1','scalar10'] for s in ['S0','S1'] for c in ['C_vs_A','C_vs_B'])
    tails=any(get(i,'ALL','held','C_vs_init')['degradation_gt10percent']>0 for i in ['alpha1','scalar10'])
    decision='已有修复路线有信号，本轮配对策略未显示优势' if signal and not advantage else ('相关性配对具有局部探索信号，仍需独立验证' if advantage else '修复的 held 迁移证据不足')
    independent_of_top=all(get(i,s,'held',c)['leave_largest_out_gain_percent']>0 for i in ['alpha1','scalar10'] for s in ['S0','S1','ALL'] for c in ['C_vs_init','C_vs_A','C_vs_B'])
    compute=lambda a:sum(bill[(st,a)] for st in ['candidate_enumeration','candidate_scoring','accepted_update'])
    overhead_ratio=compute('C')/compute('A') if compute('A') else float('inf')
    cost_verdict='有理由继续做局部实现验证，仍不支持整模型质量外推' if advantage and independent_of_top and not tails else '当前不支持扩展相关性配对的质量实验或整模型实现'
    lines.extend(['','## 推进判断','',decision+'。',f'1. 整数码修复相对初始化的两初始化 S0/S1 held 信号：{signal}。',f'2. 128d 候选预算上限下，相关性配对对 A/B 的两初始化两 split 一致 held 优势：{advantage}；较低预算节点的结果已逐项报告，不能把终点优势外推到所有预算。',f'3. 移除最大贡献 target 后，在两初始化、S0/S1/ALL、三个 C 比较中均仍为正：{independent_of_top}；C 相对初始化存在 >10% held 退化：{tails}。',f'4. C 候选枚举/评分/接受更新实测为 A 的 {overhead_ratio:.3f} 倍；成功尝试相关图构建 {bill[("pair_graph","C")]-failed_bill[("pair_graph","C")]:.3f} 秒（含失败的全账 {bill[("pair_graph","C")]:.3f} 秒）；{cost_verdict}。', '5. 剩余差异是相关性筛选的冻结配对图；固定网格与联合下降路线属于已有思路。', '本轮结束，不自动启动下一版本或整模型。', '', '## 与已有工作的差异','',
      '固定网格修复、二次型增量、单点下降和双点搜索不是本轮原创。A/B 为受论文启发的受限实现，不是官方完整复现。',
      'ReQuant 使用固定网格上的局部整数更新和缓存二次统计，包含 FP 输入与量化前缀输入不一致的重建项，采用坐标扫描。本轮同一个历史 M2-prefix X 定义 teacher/error，未实现其激活失配项，采用全邻域贪心、±1 和有界预算。[ReQuant](https://arxiv.org/html/2608.07019v1)',
      'CDQuant 提出贪心单坐标与随机块下降，块搜索枚举合法码；还处理组尺度与 clipping 初始化。本轮冻结既有 bank 参数，±1 邻域，随机图修复前固定且与相关图边数相同，允许重叠跨组边。剩余被检验的差异是仅由 search C 的归一化相关性 top2 构造冻结配对图，尚未证明论文创新成立。[CDQuant](https://arxiv.org/html/2406.17542v3)',
      '', '## 来源与复现','', '历史 row split 48 个，两初始化 96 个，三个算法 288 条轨迹；只覆盖六个 Linear 的四个代表输出行，不能外推整模型。局部 alpha1 初始化不称正式 M1。使用旧文章，是探索分析，非新数据验证。GPU 等 v40 作业空闲后启动；当前会话工具不支持确认 Sol high 模型切换，执行设置未获确认。',
      'PROTOCOL.json + FREEZE_LOCK.json 冻结协议和实现；SOURCE_AUDIT.json 来源/精确码审计；SELECTION_LOCK.json 在 held 前锁全部码和预算快照；CHECKPOINT.pt 留存状态。',
      '复现命令见 REPRODUCE.md，包含源码、CPU 检查与全轮预算监督器。OUTPUT_AUDIT.json 核对全部码/参数哈希、锁定时序、图和预算；ROW_COMPARISONS.csv 保存每个比较的逐行负结果。预算 16d/32d 时三个算法每行实际增量评价次数均精确相同；128d 终点 A 因 32 次更新上限提前停止，不能称实际评价次数相同。'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');(OUT/'DECISION.json').write_text(json.dumps({'conclusion':decision,'repair_signal':signal,'pair_advantage':advantage,'independent_of_largest_target':independent_of_top,'severe_tail':tails,'C_compute_time_ratio_to_A':overhead_ratio,'advance_verdict':cost_verdict,'no_automatic_followup':True},indent=2,ensure_ascii=False),encoding='utf8')
    try:
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
      fig,axes=plt.subplots(2,2,figsize=(10,7))
      for ax,(init,split) in zip(axes.flat,[(i,s) for i in ['alpha1','scalar10'] for s in ['S0','S1']]):
        for alg,color in [('A','#555555'),('B','#999999'),('C','#225ea8')]:
          xx=[0];yy=[1.]
          for node in [16,32,64,96,128]:
            rr=[r for r in budget if r['init']==init and r['split']==split and r['algorithm']==alg and int(r['node_over_dim'])==node]
            if rr:
              denom=sum(float(lookup[(r['unit_id'],init,alg)]['search_initial_mse']) for r in rr);xx.append(node);yy.append(sum(float(r['search_mse']) for r in rr)/denom)
          ax.plot(xx,yy,marker='o',label=alg,color=color)
        ax.set_title(f'{init} / {split}');ax.set_xlabel('Candidate evaluation allowance / input dimension');ax.set_ylabel('Pooled search SSE / initial SSE');ax.legend();ax.grid(alpha=.2)
      fig.suptitle('Historical cached search: common budget checkpoints');fig.tight_layout();fig.savefig(OUT/'SEARCH_BUDGET_CURVES.svg');plt.close(fig)
    except ImportError:pass
    print(json.dumps({'status':status,'decision':decision,'final_held':[r for r in summ if r['objective']=='held' and r['node_over_dim']=='FINAL' and r['split']=='ALL']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
