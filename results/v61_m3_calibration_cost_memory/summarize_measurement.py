from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, time, math

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
HIST=REPO/'experiments/range_search_w4/results/range_w4_omniquant_resume_v21_4/M3'

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def put(name,value): (ROOT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def write_csv(name,rows):
    if not rows: return
    with (ROOT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

DESCRIPTIONS={
    'runtime_setup':'CUDA/runtime and source bootstrap',
    'source_validation':'frozen config, pristine snapshot, tokens, implementation/runtime checks; imports, callback setup and initial checkpoint lookup (merged)',
    'model_load':'original loader: FP model load, cuda transfer and identity checks; runtime serialization',
    'data_prepare':'tokenizer and frozen token tensor list; original calibration-call bootstrap',
    'initial_activation_capture':'128 initial embedding/decoder-input captures; activation buffers and copies',
    'block_setup':'native layer clone/wrapper and quantization-state setup',
    'teacher_capture':'128 FP teacher block outputs',
    'optimization':'20-epoch AMP/AdamW LWC loop including optimizer setup, original correctness checks and logging',
    'quantization_finalize':'half cast, learned range application, scale/zero registration and CPU layer commit checks',
    'student_activation_refresh':'128 quantized student block outputs for next block',
    'checkpoint_save':'cumulative LWC parameters + RNG, hashes, fsync and block commit metadata',
    'block_cleanup':'block cleanup and cache release',
    'calibration_cleanup':'activation cleanup plus post-call final checkpoint validation and summary IO (merged)',
    'export_finalize':'original 64-token reload-regression reference; replace quantized wrappers by dense FP16 Linear; CPU move',
    'model_save':'save_pretrained, tokenizer, export file hashing and manifest',
    'reload_check':'load final dense export and original 64-token logits regression; final status IO',
}

def main():
    wall0=time.monotonic(); cpu0=time.process_time()
    processes=[load(p) for p in sorted(ROOT.glob('PROCESS_ACCOUNTING_*.json'))]
    meters=[load(p) for p in sorted((ROOT/'timings').glob('summary_*.json'))]
    all_rows=[]
    for m in meters:
        with Path(m['stage_csv']).open(newline='') as f: all_rows.extend(csv.DictReader(f))
    complete=[m for m in meters if m['status']=='COMPLETE']
    status='COMPLETE' if len(complete)==1 and any(p['status']=='COMPLETE' for p in processes) else 'PARTIAL_OR_FAILED'
    successful_rows=[r for r in all_rows if complete and r['attempt']==complete[0]['attempt']]
    stage_totals=defaultdict(lambda:dict(wall_seconds=0.,cpu_process_seconds=0.,allocated_peak_bytes=0,reserved_peak_bytes=0,segments=0))
    for r in successful_rows:
        a=stage_totals[r['phase']]
        for k in ['wall_seconds','cpu_process_seconds']: a[k]+=float(r[k])
        for k in ['allocated_peak_bytes','reserved_peak_bytes']: a[k]=max(a[k],int(r[k]))
        a['segments']+=1
    stage_csv=[]
    for phase,a in stage_totals.items():
        stage_csv.append(dict(phase=phase,description=DESCRIPTIONS[phase],**a,
            allocated_peak_GiB=a['allocated_peak_bytes']/2**30,reserved_peak_GiB=a['reserved_peak_bytes']/2**30,
            scope='complete invocation',measurement='synchronized wall, not GPU kernel busy time'))
    write_csv('STAGE_TIMINGS.csv',stage_csv)
    write_csv('ALL_ATTEMPT_STAGE_TIMINGS.csv',all_rows)
    block_totals=defaultdict(lambda:defaultdict(float))
    for r in successful_rows:
        if int(r['block'])>=0: block_totals[int(r['block'])][r['phase']]+=float(r['wall_seconds'])
    phases=['block_setup','teacher_capture','optimization','quantization_finalize','student_activation_refresh','checkpoint_save','block_cleanup']
    write_csv('BLOCK_TIMINGS.csv',[dict(block=i,**{k:a.get(k,0) for k in phases},total_seconds=sum(a.values())) for i,a in sorted(block_totals.items())])
    memory_results=[]; sampled_phase=defaultdict(lambda:dict(device_peak_bytes=0,main_rss_peak_bytes=0,child_rss_peak_bytes=0,samples=0))
    for p in processes:
        m=load(Path(p['memory_summary']))
        matched=[x for x in all_rows if int(x['start_monotonic_ns'])>=p['launch_monotonic_ns'] and int(x['start_monotonic_ns'])<=p['end_monotonic_ns']]
        model_pid=int(matched[0]['attempt'].rsplit('_',1)[1]) if matched else None
        for z in m['process_rss_peaks']:
            z['effective_role']='MODEL_MAIN' if z['pid']==model_pid else 'LAUNCHER_OR_CHILD'
        m['model_main_pid']=model_pid
        m['effective_role_source']='actual meter os.getpid; initial venv launcher/main labels normalize by workload PID, not filename or inferred memory size'
        memory_results.append(dict(attempt=p['attempt'],status=p['status'],**m))
        if p['status']!='COMPLETE': continue
        gf=ROOT/'memory'/('gpu_'+p['attempt']+'.csv')
        cf=ROOT/'memory'/('rss_'+p['attempt']+'.csv')
        stamps=[]; baseline=None
        with gf.open(newline='') as f:
            for r in csv.DictReader(f):
                stamps.append(int(r['monotonic_ns']))
                if baseline is None and r['device_used_bytes']: baseline=int(r['device_used_bytes'])
                z=sampled_phase[r['phase']]; z['samples']+=1
                if r['device_used_bytes']: z['device_peak_bytes']=max(z['device_peak_bytes'],int(r['device_used_bytes']))
        intervals=sorted((b-a)/1e9 for a,b in zip(stamps,stamps[1:]))
        m['pre_run_device_baseline_bytes']=baseline
        m['pre_run_device_baseline_GiB']=baseline/2**30 if baseline is not None else None
        m['actual_interval_median_seconds']=intervals[len(intervals)//2] if intervals else None
        m['actual_interval_p95_seconds']=intervals[int(.95*(len(intervals)-1))] if intervals else None
        m['actual_interval_max_seconds']=max(intervals,default=None)
        memory_results[-1].update({k:m[k] for k in ['pre_run_device_baseline_bytes','pre_run_device_baseline_GiB',
            'actual_interval_median_seconds','actual_interval_p95_seconds','actual_interval_max_seconds']})
        with cf.open(newline='') as f:
            for r in csv.DictReader(f):
                z=sampled_phase[r['phase']]; key='main_rss_peak_bytes' if int(r['pid'])==model_pid else 'child_rss_peak_bytes'
                z[key]=max(z[key],int(r['rss_bytes']))
    write_csv('SAMPLED_PHASE_PEAKS.csv',[dict(phase=k,**a,
        device_peak_GiB=a['device_peak_bytes']/2**30,main_rss_peak_GiB=a['main_rss_peak_bytes']/2**30,
        child_rss_peak_GiB=a['child_rss_peak_bytes']/2**30,peak_type='sampled, not exact instantaneous') for k,a in sampled_phase.items()])
    put('MEMORY_PEAKS.json',dict(status=status,attempts=memory_results,allocator_attempts=meters,
        overall_peak_allocated_bytes=max([x['peak_allocated_bytes'] for x in meters],default=None),
        overall_peak_reserved_bytes=max([x['peak_reserved_bytes'] for x in meters],default=None),
        device_peak_scope='device total, including desktop and all other processes; baseline not subtracted',
        process_gpu_unavailable_not_zero=True))
    training=load(ROOT/'M3/M3_TRAINING_SUMMARY.json') if (ROOT/'M3/M3_TRAINING_SUMMARY.json').exists() else None
    costs=dict(status=status,attempts=processes,main_stage_meters=meters,
        total_all_attempt_wall_seconds=sum(x['end_to_end_wall_seconds'] for x in processes),
        failure_wall_seconds=sum(x['end_to_end_wall_seconds'] for x in processes if x['status']!='COMPLETE'),
        full_invocation_wall_seconds=sum(x['end_to_end_wall_seconds'] for x in processes if x['status']=='COMPLETE'),
        optimization_loop_with_original_checks_seconds=stage_totals['optimization']['wall_seconds'],
        bare_kernels_or_AdamW_only_seconds='NOT_MEASURED',
        original_omniquant_call_wall_seconds=training['calibration_wall_seconds'] if training else None,
        data_to_finalized_model_pipeline_seconds=sum(a['wall_seconds'] for k,a in stage_totals.items() if k in [
            'data_prepare','initial_activation_capture','block_setup','teacher_capture','optimization','quantization_finalize',
            'student_activation_refresh','checkpoint_save','block_cleanup','calibration_cleanup','export_finalize']),
        pipeline_includes='data preparation, initial input capture, all teacher and student propagation, block setup, LWC optimization+existing diagnostics, quantization finalization, block checkpoints, cleanup/final checkpoint validation and dense finalization',
        pipeline_excludes='source/bootstrap/model loading; final dense serialization; reload regression',
        final_model_save_seconds=stage_totals['model_save']['wall_seconds'],reload_and_regression_seconds=stage_totals['reload_check']['wall_seconds'],
        cpu_main_stage_seconds=sum(x['cpu_process_seconds'] for x in meters),
        cpu_supervisor_seconds=sum(x['supervisor_cpu_process_seconds'] for x in processes),
        main_stage_cpu_excludes='Python imports before StageMeter creation; full main/child sampled CPU snapshots are in RSS timelines',
        startup_shutdown_unattributed_wall_seconds=sum(x['end_to_end_wall_seconds'] for x in processes)-sum(x['wall_seconds'] for x in meters),
        no_gpu_busy_time_claim=True,background_M1_wall_seconds=648.8761009,background_M2_wall_seconds=2115.9353341,
        fair_speed_ratios=None,background_reason='different methods, runtime/diagnostic coverage and timer scopes; historic M3 internal wall missing')
    put('COST_ACCOUNTING.json',costs)
    prechecks=load(ROOT/'INSTRUMENTATION_FAILURE_AND_PRECHECK.json')
    tensor_comparison=load(ROOT/'HISTORICAL_M3_TENSOR_COMPARISON.json') if (ROOT/'HISTORICAL_M3_TENSOR_COMPARISON.json').exists() else None
    costs['CPU_NVML_precheck_internal_wall_seconds']=prechecks['total_precheck_internal_wall_seconds']
    costs['CPU_NVML_precheck_cpu_process_seconds']=prechecks['total_precheck_cpu_process_seconds']
    costs['post_tensor_comparison_cpu_seconds']=tensor_comparison['cpu_process_seconds'] if tensor_comparison else None
    costs['post_tensor_comparison_internal_wall_seconds']=tensor_comparison['wall_seconds'] if tensor_comparison else None
    costs['calibration_sampler_wall_overhead']='NOT_MEASURED; no uninstrumented control run; supervisor CPU is measured, cannot subtract it from wall'
    put('COST_ACCOUNTING.json',costs)
    if training:
        corrected=dict(training)
        corrected['raw_runner_peak_fields_scope']='phase-local after measurement-owned resets; preserved raw summary unchanged'
        corrected['whole_invocation_peak_allocated_bytes']=complete[0]['peak_allocated_bytes'] if complete else None
        corrected['whole_invocation_peak_reserved_bytes']=complete[0]['peak_reserved_bytes'] if complete else None
        put('TRAINING_SUMMARY_WITH_MEASUREMENT_SCOPE.json',corrected)
        steps=list(csv.DictReader((ROOT/'M3/logs/training_steps.csv').open(newline='')))
        unique={(int(r['layer']),int(r['epoch_index']),int(r['sample_index'])) for r in steps}
        assert len(steps)==61440 and len(unique)==61440
        assert unique=={(i,e,j) for i in range(24) for e in range(20) for j in range(128)}
        success=sum(r['optimizer_step_executed']=='True' for r in steps)
        skips=sum(r['optimizer_step_skipped']=='True' for r in steps)
        assert success+skips==61440 and success==training['successful_optimizer_steps']
        manifest=load(ROOT/'M3/checkpoints/block_23/CHECKPOINT_MANIFEST.json')
        assert manifest['state_file_sha256']==sha(ROOT/'M3/checkpoints/block_23/omni_parameters.pth')
        dense=load(ROOT/'M3/DENSE_EXPORT_MANIFEST.json')
        assert all(sha(ROOT/'M3/dense_export'/r['path'])==r['sha256'] for r in dense['files'])
        put('COMPLETENESS.json',dict(status=status,blocks=24,Linear=168,scheduled_iterations=61440,
            unique_locations=61440,successful_optimizer_steps=success,gradscaler_skips=skips,
            resume=False,from_pristine_FP=True,token_manifest_sha256=training['calibration_manifest_sha256'],
            ordered_token_sha256=training['ordered_token_hashes_sha256'],new_ppl=False,
            checkpoint_manifest_sha256=sha(ROOT/'M3/checkpoints/block_23/CHECKPOINT_MANIFEST.json'),
            final_dense_weight_sha256=sha(ROOT/'M3/dense_export/model.safetensors'),
            reload=load(ROOT/'M3/DENSE_EXPORT_RELOAD_CHECK.json')))
    # Verify all historical inputs actually read by this task are unchanged.
    audit=load(ROOT/'SOURCE_AUDIT.json'); checks=[]
    for rec in audit['sources']:
        checks.append(dict(path=rec['path'],sha256=rec['sha256'],unchanged=sha(Path(rec['path']))==rec['sha256']))
    assert all(x['unchanged'] for x in checks)
    put('HISTORICAL_READONLY_CHECK.json',dict(status='PASS',files=checks))
    implemented=load(ROOT/'MEASUREMENT_IMPLEMENTATION_LOCK.json')
    assert all(sha(Path(r['path']))==r['sha256'] for r in implemented['files'])
    put('IMPLEMENTATION_LOCK_CHECK.json',dict(status='PASS',unchanged_files=implemented['files']))
    runtime=load(ROOT/'ACTUAL_RUNTIME.json') if (ROOT/'ACTUAL_RUNTIME.json').exists() else {}
    mcomplete=next((m for m in memory_results if m['status']=='COMPLETE'),None)
    mainpeaks=[p['peak_rss_bytes'] for p in mcomplete['process_rss_peaks'] if p['effective_role']=='MODEL_MAIN'] if mcomplete else []
    mainrss=max(mainpeaks,default=None)
    a=complete[0]['peak_allocated_bytes'] if complete else None; r=complete[0]['peak_reserved_bytes'] if complete else None
    device=mcomplete['device_sampled_peak_bytes'] if mcomplete else None
    fmt=lambda v: 'NOT_AVAILABLE' if v is None else f'{v:,} bytes / {v/2**30:.6f} GiB'
    report=f'''# v61：仓库 M3/LWC 完整校准成本与内存

状态：{status}。本轮从原始 FP 模型开始；不加载历史训练参数，不运行 PPL。

## 实际方法及来源

采用历史 v21.4 实际 runner 及其 Qwen2 原生 decoder 组合适配器，依赖 OmniQuant commit feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14 的量化/LWC代码。名称为“仓库 M3/LWC 实现”，不称完整官方 OmniQuant 复现。

模型 Qwen2.5-0.5B-Instruct，revision 7ae557604adf67be50417f59c2c2f167def9a775。正式 128×2048 WT2-train 冻结 token，原 manifest SHA 664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3；逐篇和有序 token hash 已核验。W4A16、G128、每输出行连续输入组，非对称 affine [0,15]，zero 非固定8；最终为 dense dequant FP16，real_quant=False。

只更新每个权重组的 lowbound_factor/upbound_factor，初值4，sigmoid调节组内min/max；LET、aug_loss关闭，优化器不持有其他模型参数。目标为 FP 教师前缀的 block 输出与当前量化学生前缀 block 输出的 MSE。AdamW，LWC lr=0.01，weight_decay=0，betas=(0.9,0.999)，eps=1e-8；batch=1，20 epoch，固定篇次序，seed=20261005。每block新建AdamW和动态GradScaler，不补偿/重试overflow跳步。FP16模型/输入/teacher，优化前qlayer.float，原CUDA FP16 autocast及原GradScaler，提交前half。

历史完成链：v21.1提交block0–20，v21.4从block20恢复并完成21–23；共61440计划步，历史成功61438、跳步2。v21.5仅恢复最终摘要、导出/评测证据。历史0–20的成功步数来自有限owned梯度日志与scaler源码推断，缺少直接step hook/found_inf快照；本轮全部成功/跳步则由实际hook计数。旧内部总timer不可恢复，3052秒仅时间戳区间和，不能作为完整实测wall。本轮不缩减epoch，也不使用该旧区间替代测量。

本轮全程保留v21.4详细AMP/有限值/日志诊断；历史0–20采用较简诊断。优化时间包含这套实际执行检查，不能宣称是历史每段逐行执行的同一成本。仅新增阶段标记及外部采样，核心去除标记后的AST与历史源码一致。原末尾KeyError修复仅为返回已读manifest的摘要元数据；不改变模型/优化。IMPLEMENTATION.diff列出全部更改。

## 实测成本

- 纯优化循环及原检查：{costs['optimization_loop_with_original_checks_seconds']:.6f} 秒。包括优化器创建/训练、原有限值/AMP检查及epoch日志；裸GPU kernel/纯AdamW子项 NOT_MEASURED。
- 原omniquant完整调用：{costs['original_omniquant_call_wall_seconds']} 秒，包括初始捕获、24个block教师/学生传播、训练、参数应用及组级检查点。
- 数据准备至最终化的完整校准流程：{costs['data_to_finalized_model_pipeline_seconds']:.6f} 秒；{costs['pipeline_includes']}。不含加载、最终save及reload。
- 完整成功进程端到端（包括加载/检查、保存和reload输出回归）：{costs['full_invocation_wall_seconds']:.6f} 秒。
- 最终保存：{costs['final_model_save_seconds']:.6f} 秒；reload及64-token输出回归：{costs['reload_and_regression_seconds']:.6f} 秒。
- 失败开销：{costs['failure_wall_seconds']:.6f} 秒；所有正式尝试累计：{costs['total_all_attempt_wall_seconds']:.6f} 秒。首次计量通知文件与采样读取发生Windows共享冲突，优化0步、提交0block；现场/账本保留，通知改为追加式日志，经200次并发读写检查后从FP重启。
- 主进程已覆盖阶段CPU时间（所有尝试）：{costs['cpu_main_stage_seconds']:.6f} 秒；采样监督进程CPU：{costs['cpu_supervisor_seconds']:.6f} 秒。CPU时间不等于wall；阶段之外的Python导入/进程启动退出开销单列COST_ACCOUNTING，RSS时间线保留各进程CPU快照。
- CPU/NVML计量预检内部wall：{costs['CPU_NVML_precheck_internal_wall_seconds']:.6f} 秒、CPU：{costs['CPU_NVML_precheck_cpu_process_seconds']:.6f} 秒；不含Python导入和工具启动。后处理CPU账本另列。没有无计量对照，采样器对wall的额外影响NOT_MEASURED，不能拿采样进程CPU时间从wall中扣除。

阶段CSV同步计时，原训练代码已有逐步同步仍保留，新计量只在阶段边界同步，不称wall为GPU计算时间。calibration_cleanup合并cleanup、最终检查点读回及摘要IO，不按代码/运算量分摊。M1 648.8761秒、M2 2115.9353秒仅作背景，不给公平加速比。

## 内存

- 完整成功尝试PyTorch peak allocated：{fmt(a)}。
- 完整成功尝试PyTorch peak reserved：{fmt(r)}。
- GPU设备总显存采样峰值：{fmt(device)}，含桌面及其他进程，不是本进程独占，也未把基线扣除后冒充allocated。
- 运行前设备显存基线：{fmt(mcomplete['pre_run_device_baseline_bytes'] if mcomplete else None)}。实际采样间隔中位数/P95/最大值：{mcomplete['actual_interval_median_seconds'] if mcomplete else None} / {mcomplete['actual_interval_p95_seconds'] if mcomplete else None} / {mcomplete['actual_interval_max_seconds'] if mcomplete else None} 秒。
- 主模型进程RSS采样峰值：{fmt(mainrss)}。venv启动器及可见子进程分别记录；并发树RSS峰值与逐进程峰值见MEMORY_PEAKS，不将各独立峰值相加。
- GPU进程显存：NOT_AVAILABLE（当前Windows/WDDM返回不可用，空值不是0）。NVML设备总显存约200ms采样；CPU RSS同时采样。采样峰值不是精确瞬时峰值；短于采样间隔的子进程可能遗漏，明确记录该覆盖限制。
- 每阶段PyTorch峰值在reset前持久化，整程为所有阶段峰值最大值；失败阶段也保存。设备/RSS分阶段峰值和原始时间线独立保留。失败及正常退出均flush已有记录。
- 原runner摘要中的peak字段在新增分阶段reset后只代表末尾阶段，不作为整程峰值。原摘要保留，TRAINING_SUMMARY_WITH_MEASUREMENT_SCOPE补充口径，整程峰值以本报告和MEMORY_PEAKS的阶段累积为准。

## 环境、完整性及质量边界

实际进程使用 {runtime.get('python','UNKNOWN').splitlines()[0]}；PyTorch {runtime.get('torch','UNKNOWN')}；Transformers {runtime.get('transformers','UNKNOWN')}；CUDA {runtime.get('cuda_runtime','UNKNOWN')}；GPU {runtime.get('gpu','UNKNOWN')}。完整解释器、源码SHA、tokenizer、SDPA策略、TF32及有效配置见ACTUAL_RUNTIME。操作系统现为Win11 build26300，历史记录build26200；不抹去差异。接口无法确认Sol high设置。

COMPLETENESS记录24block/168Linear、全部61440计划位置的唯一性、成功/跳步、最终检查点SHA与原输出回归。最终模型为本轮新产物；即使局部/文件hash相同也不据此继承历史M3 PPL。本轮没有新质量结果，历史PPL不作为新模型测量。

独立CPU逐具名保存张量比较：{tensor_comparison['different_tensor_count'] if tensor_comparison else 'NOT_MEASURED'} 个张量不同，详见HISTORICAL_M3_TENSOR_COMPARISON；本轮PPL未测量、不继承。正式输入文章/截断token及有序hash见CALIBRATION_SOURCE和CALIBRATION_SOURCE_ROWS；文章hash来自锁定manifest，本轮不重新解析原Arrow文章。实际Python工作进程映像与venv解释器路径分别见PROCESS_IMAGE_IDENTITY与ACTUAL_RUNTIME。

SOURCE_AUDIT、HISTORICAL_READONLY_CHECK与IMPLEMENTATION_LOCK_CHECK核对历史来源及冻结代码；不覆盖历史文件。成本只适用于这一次现有运行时下的完整校准及计量路径，不声称重复测量稳定性。

## 复现

已完成目录不能重新执行校准以覆盖结果。执行REPRODUCE_FRESH.ps1会在results下创建新的独立测量目录，复制生成器/计量源码，再从FP生成相同训练入口；不自行执行该脚本。现有目录/导出已存在时拒绝覆盖。

```powershell
& 'E:/quant-thesis/results/v61_m3_calibration_cost_memory/REPRODUCE_FRESH.ps1'
# 仅在有已提交block且运行未完成时恢复；账本保留全部PROCESS_ACCOUNTING文件：
& 'E:/quant-thesis/.venv-gptq/Scripts/python.exe' -B 'E:/quant-thesis/results/v61_m3_calibration_cost_memory/measure_once.py' --resume
# 仅CPU重算报告：
& 'E:/quant-thesis/.venv-gptq/Scripts/python.exe' -B 'E:/quant-thesis/results/v61_m3_calibration_cost_memory/summarize_measurement.py'
```

完整测量后停止，不启动其他方法重测或PPL。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    put('CPU_REPORT_ACCOUNTING.json',dict(cpu_process_seconds=time.process_time()-cpu0,wall_seconds=time.monotonic()-wall0,
        scope='post-run CPU aggregation/hash audit only; not merged into calibration wall',no_model_forward=True))
    print(json.dumps(dict(status=status,costs=costs,allocated=a,reserved=r,device=device,main_rss=mainrss),indent=2))

if __name__=='__main__': main()
