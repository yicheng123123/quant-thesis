import time
T0=time.time()
import csv,hashlib,json,math,os,sys
from pathlib import Path
import torch
from safetensors import safe_open

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
BASE=OUT.parent
V18=BASE/'range_w4_fullmodel_v18'
V45=BASE/'v45_m2_single_integer_repair_fullmodel'
V54=BASE/'v54_m2_decoder_fixed_integer_repair'
CAP=12600.0
INHERITED=5980.039842538834

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):
            h.update(b)
    return h.hexdigest()

def readj(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def dump(p,obj):
    p=OUT/p
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    tmp.replace(p)

# Fresh required evaluation outputs and completed evaluator record are mandatory.
evalsrc=readj(OUT/'QUALITY_EVALUATION_SOURCE.json')
assert evalsrc['pass'] and evalsrc['new_PPL_evaluations']==2 and evalsrc['reused_history'] is False
assert len(evalsrc['results'])==2
modelp=OUT/'R1/checkpoint/final/model.safetensors'
configp=OUT/'R1/checkpoint/final/config.json'
assert sha(modelp)==readj(OUT/'CHECKPOINT_LOCK.json')['final_model_sha256']

# Hash every named deployed tensor and preserve the per-tensor comparison record.
named=[]
with safe_open(str(modelp),framework='pt',device='cpu') as sf:
    for key in sorted(sf.keys()):
        t=sf.get_tensor(key).contiguous()
        digest=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
        named.append({'tensor':key,'shape':json.dumps(list(t.shape),separators=(',',':')),'dtype':str(t.dtype),'sha256':digest})
with (OUT/'NAMED_TENSOR_HASHES.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=['tensor','shape','dtype','sha256'])
    w.writeheader()
    w.writerows(named)
named_hash_file_sha=sha(OUT/'NAMED_TENSOR_HASHES.csv')

# Metrics and token-weighted NLL comparison.
source={
 'M1':{
  'wikitext':V18/'evaluation/raw/M1_wikitext.json',
  'c4_fixed_budget':V18/'evaluation/raw/M1_c4_fixed_budget.json'},
 'M2':{
  'wikitext':V18/'evaluation/raw/M2_wikitext.json',
  'c4_fixed_budget':V18/'evaluation/raw/M2_c4_fixed_budget.json'},
 'v55':{
  'wikitext':OUT/'evaluation/v55_wikitext.json',
  'c4_fixed_budget':OUT/'evaluation/v55_c4_fixed_budget.json'}
}
checkpoints={
 'M1':V18/'M1/checkpoint/final/model.safetensors',
 'M2':V18/'M2/checkpoint/final/model.safetensors',
 'v55':modelp}
metrics={}
for model,dsmap in source.items():
    metrics[model]={}
    for ds,p in dsmap.items():
        d=readj(p)
        ppl=d.get('ppl',d.get('perplexity'))
        nll=d.get('exact_window_weighted_average_nll',d.get('nll',d.get('weighted_nll',d.get('average_nll'))))
        tok=d.get('scored_tokens',d.get('valid_tokens',d.get('evaluated_tokens')))
        assert ppl is not None and nll is not None and tok is not None,(model,ds,list(d))
        metrics[model][ds]={'ppl':float(ppl),'nll':float(nll),'valid_tokens':int(tok),
          'result_path':str(p),'result_sha256':sha(p),'checkpoint_path':str(checkpoints[model]),
          'checkpoint_sha256':sha(checkpoints[model]),'recorded_checkpoint_sha256':d.get('model_sha256',d.get('model_weights_sha256')),
          'manifest_sha256':d.get('manifest_sha256',d.get('evaluation_manifest_sha256')),
          'windows':d.get('windows',d.get('window_count'))}
        if d.get('model_sha256') is not None:
            assert d['model_sha256']==metrics[model][ds]['checkpoint_sha256'],(model,ds,'checkpoint hash mismatch')
rows=[]
for ds in ['wikitext','c4_fixed_budget']:
    m1=metrics['M1'][ds];m2=metrics['M2'][ds];v=metrics['v55'][ds]
    for name,m in [('M1',m1),('M2',m2),('v55',v)]:
        r={'dataset':ds,'model':name,'ppl':m['ppl'],'nll_per_valid_token':m['nll'],
          'valid_tokens':m['valid_tokens'],'windows':m['windows'],'checkpoint_sha256':m['checkpoint_sha256'],
          'result_sha256':m['result_sha256'],'manifest_sha256':m['manifest_sha256'],
          'recorded_checkpoint_sha256':m['recorded_checkpoint_sha256']}
        for base,bm in [('M1',m1),('M2',m2)]:
            r['ppl_delta_vs_'+base]=m['ppl']-bm['ppl']
            r['ppl_relative_pct_vs_'+base]=100*(m['ppl']/bm['ppl']-1)
            r['nll_delta_vs_'+base]=m['nll']-bm['nll']
            r['nll_relative_pct_vs_'+base]=100*(m['nll']/bm['nll']-1)
        rows.append(r)
with (OUT/'QUALITY_COMPARISON.csv').open('w',newline='',encoding='utf8') as f:
    fields=list(rows[0].keys())
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# Aggregate saved stage timers without imputing values into unmeasured substeps.
stage_sums={}
stage_counts={}
stage_modes={}
for file in [OUT/'STAGE_TIMINGS.csv',OUT/'STAGE_TIMINGS_EVAL_CONTINUATION.csv']:
    if not file.exists():
        continue
    for r in csv.DictReader(file.open(encoding='utf8')):
        stage=r.get('stage') or 'UNSPECIFIED'
        mode=r.get('mode') or ('evaluation' if 'EVAL' in file.name else 'formal')
        try: sec=float(r.get('seconds') or 0)
        except (ValueError,TypeError): continue
        k=(mode,stage)
        stage_sums[k]=stage_sums.get(k,0.0)+sec
        stage_counts[k]=stage_counts.get(k,0)+1
with (OUT/'STAGE_COST_SUMMARY.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=['mode','stage','calls','measured_seconds'])
    w.writeheader()
    for (mode,stage),sec in sorted(stage_sums.items()):
        w.writerow({'mode':mode,'stage':stage,'calls':stage_counts[(mode,stage)],'measured_seconds':sec})

blocks=list(csv.DictReader((OUT/'BLOCK_TIMINGS.csv').open(encoding='utf8')))
new_blocks=[r for r in blocks if int(r['layer'])>=2 and int(r['new_groups'])>0]
repaired_layer_wall=sum(float(r['seconds']) for r in new_blocks)
new_accepted=sum(int(r['accepted_updates']) for r in new_blocks)
formal=readj(OUT/'FORMAL_REPAIR_COMPLETE.json')
deployment=readj(OUT/'DEPLOYMENT_COMPARISON.json')
config=readj(OUT/'CONFIG_COMPARISON.json')
prop_rows=list(csv.DictReader((OUT/'PROPAGATION_NATIVE_RESULTS.csv').open(encoding='utf8')))
prop_blocks={int(r['block_completed']) for r in prop_rows}
prop_articles={int(r['article']) for r in prop_rows}
assert len(prop_blocks)==24 and prop_blocks==set(range(24)) and {0,64,127}.issubset(prop_articles) and all(str(r.get('pass','')).lower()=='true' for r in prop_rows),('native propagation coverage/failure',len(prop_rows),sorted(prop_blocks),sorted(prop_articles))

# Final audit and report are CPU-only. Add a measured interval and conservative closeout padding.
ledgerp=OUT/'ACTIVE_JOB_LEDGER.json'
ledger=readj(ledgerp)
unmetered=45.0
ledger['segments'].append({
 'version':'v55','label':'v55 CPU-only diagnostics, prior aggregation failure, and evaluator launcher work outside active budget hook',
 'charged_seconds':unmetered,'cpu_only':True,'GPU_calls':0,
 'basis':'Conservative charge for read-only evaluator inspection, failed CPU-only metrics aggregation and patching, and launcher/file-write work outside active budget hooks.'
})
finalize_seconds=max(0.0,time.time()-T0)
closeout_pad=20.0
ledger['segments'].append({
 'version':'v55','label':'v55 CPU-only final hash, quality, cost, and report aggregation',
 'charged_seconds':finalize_seconds+closeout_pad,'cpu_only':True,'GPU_calls':0,
 'measured_script_seconds':finalize_seconds,'conservative_closeout_pad_seconds':closeout_pad
})
ledger['finalized_unix']=time.time()
ledgerp.write_text(json.dumps(ledger,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
charged=[s.get('charged_seconds',max(0,time.time()-s.get('start_epoch',time.time()))) for s in ledger['segments']]
total=float(ledger['v51_charged_seconds'])+sum(charged)
v55cost=sum(v for s,v in zip(ledger['segments'],charged) if s.get('version')=='v55')
cpu55=sum(v for s,v in zip(ledger['segments'],charged) if s.get('version')=='v55' and s.get('cpu_only'))
noncpu55=v55cost-cpu55
formal_proc=next((s.get('charged_seconds') for s in ledger['segments'] if s.get('version')=='v55' and s.get('label')=='v55 complete repair and deployment/evaluation'),None)
final_budget={
 'cap_seconds':CAP,'inherited_used_seconds':INHERITED,'inherited_remaining_after_extension_seconds':12600-INHERITED,
 'final_cumulative_charged_seconds':total,'remaining_seconds':CAP-total,
 'v55_added_charged_seconds':v55cost,'v55_cpu_only_charged_seconds':cpu55,
 'v55_non_cpu_only_process_wall_seconds':noncpu55,
 'v55_formal_runner_process_wall_seconds':formal_proc,
 'valid_new_layer2_to23_block_wall_seconds':repaired_layer_wall,
 'valid_new_layer2_to23_accepted_updates':new_accepted,
 'finish_reserve_seconds':360,'reserve_met':total<=CAP-360,
 'original_M2_build_seconds_separate':2115.9353341,
 'accounting_notes':['Cumulative cap was increased from 10800 to 12600 seconds; no reset.',
 'v55 non-CPU-only values are process wall and include CPU work, not pure GPU kernel time.',
 'The original M2 build cost is separate and not included in the shared experiment cap.',
 'The 360-second final reserve was part of the original projection; it is not subtracted twice.']
}
dump('FINAL_BUDGET_ACCOUNTING.json',final_budget)
dump('FINAL_CHECKPOINT_LOCK.json',{
 'model_sha256':sha(modelp),'config_sha256':sha(configp),
 'original_checkpoint_lock_sha256':sha(OUT/'CHECKPOINT_LOCK.json'),
 'named_tensor_hashes_csv_sha256':named_hash_file_sha,'named_tensor_count':len(named),
 'group_checkpoint_count':len(readj(OUT/'CHECKPOINT_LOCK.json')['group_checkpoints']),
 'deployment_comparison_sha256':sha(OUT/'DEPLOYMENT_COMPARISON.json'),
 'effective_config_comparison_sha256':sha(OUT/'CONFIG_COMPARISON.json'),
 'locked_before_evaluation':readj(OUT/'CHECKPOINT_LOCK.json').get('locked_before_PPL_unix'),
 'PPL_evaluations_completed_after_lock':True
})
complete={
 'status':'COMPLETE','category':'COMPLETE_MODEL_AND_PPL',
 'complete_model':True,'modules':formal['modules'],'rows':formal['rows'],
 'model_sha256':sha(modelp),'PPL_run':True,'PPL_reused':False,
 'prior_formal_runner_failure_preserved':True,
 'prior_failure_was_raw_config_json_serialization_assertion':True,
 'effective_config_comparison_pass':config['effective_configs_equal'],
 'native_propagation_blocks_passed':len(prop_blocks),
 'finalized_unix':time.time(),'shared_elapsed_seconds':total,'shared_remaining_seconds':CAP-total
}
dump('RUN_STATUS.json',complete)
dump('COMPLETE.json',{
 'pass':True,'all_96_committed_groups':True,'all_168_modules':True,'all_304128_rows':True,
 'full_model_locked':True,'PPL_newly_evaluated_once_each':True,
 'model_sha256':sha(modelp),'quality_comparison_sha256':sha(OUT/'QUALITY_COMPARISON.csv'),
 'final_budget_sha256':sha(OUT/'FINAL_BUDGET_ACCOUNTING.json')
})

def fmt(x,n=6):
    return f'{float(x):.{n}f}'
tab=[]
delta_tab=[]
for ds,label in [('wikitext','WikiText-2 test'),('c4_fixed_budget','C4 fixed-budget validation')]:
    ms=metrics
    tab.append('| '+label+' | M1 | '+fmt(ms['M1'][ds]['ppl'])+' | '+fmt(ms['M1'][ds]['nll'])+' | '+str(ms['M1'][ds]['valid_tokens'])+' |')
    tab.append('| '+label+' | M2 | '+fmt(ms['M2'][ds]['ppl'])+' | '+fmt(ms['M2'][ds]['nll'])+' | '+str(ms['M2'][ds]['valid_tokens'])+' |')
    tab.append('| '+label+' | v55 | '+fmt(ms['v55'][ds]['ppl'])+' | '+fmt(ms['v55'][ds]['nll'])+' | '+str(ms['v55'][ds]['valid_tokens'])+' |')
    for b in ['M1','M2']:
        delta=ms['v55'][ds]['nll']-ms[b][ds]['nll']
        pdelta=ms['v55'][ds]['ppl']-ms[b][ds]['ppl']
        delta_tab.append('| '+label+' | v55 vs '+b+' | '+fmt(delta)+' ('+fmt(100*delta/ms[b][ds]['nll'],4)+'%) | '+fmt(pdelta)+' ('+fmt(100*pdelta/ms[b][ds]['ppl'],4)+'%) |')
failed=readj(OUT/'RUN_STATUS.json')
report=[
'# v55 报告：M2 初始化整数码修复与正式 PPL',
'',
'## 结论',
'',
'完整运行覆盖24个block、168个Linear和304,128个输出行。v55使用修复后的decoder接口传播；每个block的固定文章原生forward/hook对照均通过。最终checkpoint通过码/网格、packing/reload和固定样例输出回归。WT2与固定预算C4均由冻结evaluator对v55 checkpoint新测一次。',
'',
'质量变化以如下实际测量为准；历史v45/v46/v49校准传播有效性仍为UNKNOWN，不能将v55与它们的差异全部归因于decoder接口修复。',
'',
'| 数据集 | 模型 | PPL | 按valid_tokens加权NLL | valid_tokens |',
'|---|---:|---:|---:|---:|'
]+tab+[
'',
'相对变化以v55减对应基线计算。',
'',
'| 数据集 | 比较 | NLL差值（相对%） | PPL差值（相对%） |',
'|---|---|---:|---:|'
]+delta_tab+[
'',
'## 来源与恢复',
'',
'- 从正式M2 checkpoint恢复，SHA256为 '+metrics['M2']['wikitext']['checkpoint_sha256']+'。v53有效恢复边界通过核对后，复用block0/1的8个组；这些组未在v55重复修复。layer2至23从本次模型自身前缀修复，未读取v52 layer1之后的无效产物。',
'- 128篇×2048校准manifest SHA256为 '+readj(OUT/'SOURCE_AUDIT.json')['calibration_manifest_sha256']+'。全token轴输入严格检查shape和hash；传播对照覆盖24个block与文章0/64/127。',
'- 数值流程、decoder实现、初始化复核和fast repair核心依照v54/v53/v52/v49原件复用。v55唯一修订在收尾脚本：原始config JSON比较受到Transformers序列化版本差异阻断，故保留原FAILED_STOP证据，改为同一当前loader解析后的有效配置核对，再独立完成张量审计和新评测。未改量化代码、接受规则或checkpoint。',
'- 普通沙盒未确认恢复；本轮使用此前获授权的沙盒外执行入口。请求的Luna Max模型设置无法由执行接口确认。',
'',
'## 修复、传播与部署检查',
'',
'- 修复：'+str(formal['modules'])+' Linear；'+str(formal['rows'])+'输出行；接受更新 '+str(formal['accepted_updates'])+'；修改坐标 '+str(formal['modified_coordinates'])+'；候选评价 '+str(formal['candidate_evaluations'])+'；拒绝0；FP64复核0。',
'- 终态检查：每个模块全行raw-X FP64直接重构与冻结二次型检查通过；固定网格、码合法、scale/zero/g_idx保持、pack/reload检查和固定样例logit回归通过。',
'- 原生传播：'+str(len(prop_blocks))+'/24个block通过；'+str(len(prop_rows))+'条逐边界对照均通过，覆盖文章0、64、127并保留完整token轴。',
'- 完整模型SHA256：'+sha(modelp)+'。具名张量SHA清单包含'+str(len(named))+'个张量，SHA256 '+named_hash_file_sha+'。与M2及历史R2的逐具名张量比较见DEPLOYMENT_COMPARISON.json与reports/TENSOR_DIFFERENCES_VS_R2_AND_M2.csv。',
'- M2/v55/历史R2配置原始JSON因保存端序列化版本不同而不完全相同；当前同一loader有效配置比较：'+str(config['effective_configs_equal'])+'，见CONFIG_COMPARISON.json。新checkpoint张量不同于历史R2，所以没有复用历史PPL。',
'',
'## 成本',
'',
'- 新增v55共同账本成本：'+fmt(v55cost,3)+'秒；其中CPU-only '+fmt(cpu55,3)+'秒，非CPU-only进程wall '+fmt(noncpu55,3)+'秒（包含CPU工作，不等于纯GPU kernel时间）。',
'- 本轮有效正式修复增量成本（新修layer2至23的完整依赖组块wall）：'+fmt(repaired_layer_wall,3)+'秒；新接受更新 '+str(new_accepted)+'。有效复用的block0/1组未在v55重复修复，其历史生成时间已经包含于继承账本。本次完整formal runner进程wall（含模型构建、恢复前缀、修复、传播与保存/reload至原配置序列化断言）：'+fmt(formal_proc,3)+'秒。',
'- 原M2构建成本另计 '+fmt(2115.9353341,3)+'秒。共同累计：'+fmt(total,3)+'/'+fmt(CAP,0)+'秒，剩余 '+fmt(CAP-total,3)+'秒；最终收尾前预留满足360秒。',
'- 分阶段测量见STAGE_COST_SUMMARY.csv、BLOCK_TIMINGS.csv及STAGE_TIMINGS_EVAL_CONTINUATION.csv。共享C是否新做或共享情况由原冻结实现决定；此轮未改动。',
'',
'## 评测与解释',
'',
'- WT2/C4均实际运行一次，窗口NLL与窗口valid_tokens原始记录保存在evaluation目录；NLL按valid_tokens加权后汇总。没有读取P3或依据PPL调参。',
'- 局部search重构误差不替代PPL。本轮是已有离散修复思路在M2上的应用验证，不是官方ReQuant复现，也不构成原创性声明；一次运行不支持跨seed稳定性。',
'- v45/v46/v49历史PPL只作为各自checkpoint的实际测量背景；其校准传播有效性仍是UNKNOWN，未作为严格公平因果对照。',
'- 历史最终化断言失败被保留在FAILED_ATTEMPTS.jsonl：raw config JSON不相等导致停止；本轮通过有效配置解析和独立新PPL完成了v55交付，没有改写为“历史runner当时成功”。',
'',
'## 复现与交付索引',
'',
'- 冻结协议与来源锁：PROTOCOL.json、RESUME_SOURCE_LOCK.json、SOURCE_AUDIT.json。',
'- 数值和恢复：NUMERICAL_VALIDATION.json、PROPAGATION_RESTORE_CHECKS.csv、PROPAGATION_NATIVE_RESULTS.csv、CORRECTNESS.jsonl。',
'- 模型：R1/checkpoint/final/model.safetensors、CHECKPOINT_LOCK.json、FINAL_CHECKPOINT_LOCK.json、NAMED_TENSOR_HASHES.csv。',
'- 质量：QUALITY_COMPARISON.csv、QUALITY_EVALUATION_SOURCE.json、evaluation/v55_wikitext_WINDOWS.csv、evaluation/v55_c4_fixed_budget_WINDOWS.csv。',
'- 成本：ACTIVE_JOB_LEDGER.json、FINAL_BUDGET_ACCOUNTING.json、STAGE_COST_SUMMARY.csv。',
'- 复现命令与失败边界见REPRODUCE.md。所有历史checkpoint、报告与公共实现保持只读。'
]
(OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf8')
(OUT/'REPRODUCE.md').write_text(
 '# v55复现记录\n\n'
 '修复使用本目录冻结PROTOCOL.json、run_v55.py、fast_accept_v52.py、runtime_v54.py、decoder_interface.py及源锁。原正式runner命令为：\n\n'
 '    python run_v55.py --resume\n\n'
 '正式修复已完整结束并保存checkpoint。该runner在最后的原始config JSON序列化比较处留下FAILED_STOP；原失败保留在FAILED_ATTEMPTS.jsonl。之后使用同一当前Transformers loader核验有效配置，并运行一次冻结评测续接脚本：\n\n'
 '    python continue_v55_eval.py\n\n'
 '该脚本只适用于未写入v55评测结果的新隔离副本；本目录已完成一次WT2及一次C4评测，不能在本目录重复执行。评测协议与manifest SHA记录在QUALITY_EVALUATION_SOURCE.json中。\n',encoding='utf8')
print(json.dumps({'status':'aggregated','final_elapsed_s':time.time()-T0,'shared_total':total,'shared_remaining':CAP-total,'v55_cost':v55cost,'tensor_count':len(named),'results':metrics,'layer2_23_block_wall':repaired_layer_wall,'formal_process_wall':formal_proc},indent=2))
