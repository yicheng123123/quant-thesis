"""CPU-only evidence validation and report generation. Never resume training."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESULTS=HERE.parent
UP=RESULTS/'range_w4_fullmodel_baseline_v16/official_source'
M3=RESULTS/'range_w4_omniquant_full_v21_1/M3'
AUDIT12=RESULTS/'range_w4_omniquant_nan_audit_v21_2'


def load(path):
    def reject(value):
        raise ValueError(f'nonfinite literal: {value}')
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=reject)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def md(name,value):
    (HERE/name).write_text(value.strip()+'\n',encoding='utf-8')


def excerpt(path,start,end):
    lines=path.read_text(encoding='utf-8').splitlines()
    return (f'文件：`{path}`；SHA256：`{sha(path)}`；lines {start}–{end}。\n\n```python\n'+
            '\n'.join(f'{i}: {lines[i-1]}' for i in range(start,end+1))+'\n```\n')


def main():
    raw=load(HERE/'OFFICIAL_OVERFLOW_HANDLING_REPLAY.json');single=raw['record']
    sequence=[load(HERE/f'SEQUENCE_SAMPLE_{i}.json') for i in range(4)]
    micro=load(HERE/'GRADSCALER_BEHAVIOR_CHECK.json')
    history=load(HERE/'HISTORICAL_UPDATE_COUNT_AUDIT.json')
    sources=load(HERE/'SOURCE_SHA256.json')
    protected=load(HERE/'PROTECTED_HASHES.json')
    prior=load(AUDIT12/'REPLAY_1.json')
    bad={g['parameter_name']:g['gradient'] for g in prior['gradients'] if not g['gradient']['all_finite']}
    checks={
        'exact_upstream_and_runtime_source_hashes':all(sha(Path(p))==value for p,value in sources.items()),
        'parent_and_v21_2_artifacts_unchanged':all(sha(Path(p))==value for p,value in protected.items()),
        'torch_micro_skip_and_backoff':micro['status']=='PASS' and micro['iterations'][0]['found_inf']>0 and not micro['iterations'][0]['optimizer_step_executed'] and micro['iterations'][0]['scale_after']==micro['iterations'][0]['scale_before']*0.5,
        'torch_micro_next_finite_update':micro['iterations'][1]['found_inf']==0 and micro['iterations'][1]['optimizer_step_executed'],
        'single_exact_historical_loss':single['loss']==0.15326565504074097,
        'single_found_inf_skip_backoff':single['found_inf']>0 and single['optimizer_step_skipped'] and single['scale_after']==32768.0,
        'single_no_parameter_or_optimizer_mutation':not single['LWC_parameters_changed'] and not single['optimizer_state_changed'],
        'single_sample1_not_entered':raw['sample1_entered'] is False,
        'same_four_nonfinite_parameter_names_and_counts':set(bad)=={name for name,value in single['unscaled_gradients'].items() if not value['all_finite']} and all(
            bad[name][key]==single['unscaled_gradients'][name][key] for name in bad for key in ('nan_count','posinf_count','neginf_count','finite_count')),
        'continuous_sequence_exactly_four_ordered_samples':[r['sample'] for r in sequence]==[0,1,2,3],
        'sequence_scaler_state_continuity':all(sequence[i]['scaler_state_after']==sequence[i+1]['scaler_state_before'] for i in range(3)),
        'sequence_optimizer_state_continuity':all(sequence[i]['optimizer_state_after']['sha256']==sequence[i+1]['optimizer_state_before']['sha256'] for i in range(3)),
        'sequence_three_recovered_finite_updates':all(r['found_inf']==0 and r['optimizer_step_executed'] and r['LWC_parameters_changed'] for r in sequence[1:]),
        'sequence_parameters_and_optimizer_finite':all(r['LWC_finite'] and r['all_named_parameters_finite'] and r['optimizer_state_after']['all_finite'] for r in sequence),
        'historical_complete_ordered_53760_finite_owned_gradients':history['status']=='PASS' and history['committed_prefix_iterations']==53760 and history['prefix_nonfinite_LWC_gradient_rows']==0,
        'formal_commits_zero':all(r['formal_committed_updates']==0 for r in [single]+sequence),
        'frozen_baseline_not_changed':load(HERE/'PROTOCOL.json')['precision']=='formal autocast FP16, FP32 LWC; no forced FP32 or manually fixed scale',
    }
    with (HERE/'DYNAMIC_SCALE_RECOVERY_TRACE.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    checks['CSV_matches_raw_sequence']=len(rows)==4 and all(
        int(row['sample'])==r['sample'] and float(row['loss'])==r['loss'] and
        float(row['scale_after'])==r['scale_after'] and row['optimizer_step_skipped']==str(r['optimizer_step_skipped'])
        for row,r in zip(rows,sequence))
    assert all(checks.values()),checks
    write('correctness_checks.json',{'status':'PASS','scope':'bounded AMP semantics audit; not formal M3 completion',
        'checks':checks,'limitations':['historical successful steps inferred from finite optimizer-owned gradients and exact native control flow, not directly hooked',
                                      'four-sample recovery does not prove whole epoch or remaining layers stable',
                                      'v21.1 STOP was post-scaler, not pre-scaler']})
    write('FINAL_STATUS.json',{'verdict':'STOP_GATE_TOO_STRICT_RESUME_ALLOWED',
        'overflow_classification':'RECOVERABLE_DYNAMIC_LOSS_SCALE_OVERFLOW',
        'stop_gate_classification':'STOP_GATE_TOO_STRICT','upstream_case':'A: dynamic GradScaler',
        'actual_STOP_boundary':'after native scaler.step and scaler.update; extra runner callback stopped next sample',
        'resume_allowed_meaning':'source/control-flow justification and plan only; execution requires next-round instruction',
        'correctness':'PASS','formal_resume_executed':False,'formal_committed_updates':0,
        'last_formal_checkpoint':'block_20','single_diagnostic_optimizer_updates':0,
        'continuous_diagnostic_optimizer_updates':3,'FP32_baseline_adopted':False,
        'T4_executed':False,'WT2_C4_export_executed':False})
    costs=load(HERE/'AUDIT_COST.json')
    costs['max_layer_peak_allocated_bytes']=max(r['peak_allocated_bytes'] for r in [single]+sequence)
    costs['max_layer_peak_reserved_bytes']=max(r['peak_reserved_bytes'] for r in [single]+sequence)
    write('AUDIT_COST.json',costs)
    torch_scaler_source=Path(next(p for p in sources if p.replace('\\','/').endswith('/torch/amp/grad_scaler.py')))

    md('UPSTREAM_AMP_CONTROL_FLOW.md',f'''
# Frozen upstream AMP control flow

官方仓库 OpenGVLab/OmniQuant，commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`。直接使用保存的该commit源码，不用最新README、其他commit或runner封装代替官方证据。全部257个源码manifest条目哈希验证通过；精确文件哈希见SOURCE_SHA256.json。

## Case A：官方使用 dynamic GradScaler

weight-only LWC 路径位于 quantize/omniquant.py 的 `omniquant`。LET off 时优化器LET组为空，另一组为LWC；activation bits16不做激活量化。W4默认 deactive_amp=False，student训练使用 CUDA autocast；qlayer.float()保存FP32参数。每个block新建AdamW和NativeScaler，每个epoch/sample不重建scaler。

{excerpt(UP/'main.py',225,225)}
{excerpt(UP/'main.py',246,247)}
{excerpt(UP/'quantize/omniquant.py',103,110)}

## 精确 training loop

函数：`omniquant`。先autocast forward/loss；非有限loss进入官方pdb诊断；finite loss时zero_grad并调用NativeScaler。20 epochs按for循环迭代samples，不根据successful steps重试。

{excerpt(UP/'quantize/omniquant.py',243,274)}

## 精确 backward / unscale / step / update

类：`NativeScalerWithGradNormCount`，方法`__init__`、`__call__`。不传init_scale参数；调用当前torch默认dynamic scaler。正式调用没有clip_grad，使用else分支：unscale后计算norm（可为NaN），不因norm非有限中止，然后执行scaler.step及update。没有独立的裸optimizer.step调用。

{excerpt(UP/'utils.py',27,47)}

## 当前安装torch实现与实际检查

torch2.6.0+cu124。默认scale65536、growth_factor2、backoff_factor0.5、growth_interval2000，见本地torch/amp/grad_scaler.py:119–125。CUDA兼容包装继承torch.amp.GradScaler，没有覆盖这些默认参数。

unscale_只检查optimizer-owned grads，通过`_amp_foreach_non_finite_check_and_unscale_`记录found_inf；_maybe_opt_step仅在found_inf总和为0时调用optimizer.step；update使用found_inf和backoff/growth参数更新scale、清空本iteration的optimizer tracking。

{excerpt(torch_scaler_source,277,283)}
{excerpt(torch_scaler_source,343,353)}
{excerpt(torch_scaler_source,521,531)}
本地实现路径和SHA256见SOURCE_SHA256；关键可执行代码区段：torch/amp/grad_scaler.py:338–341（检测状态）、343–353（有条件step）、521–531（dynamic update与下一iteration状态清理）。不是仅引用文档：GRADSCALER_BEHAVIOR_CHECK.json在当前GPU/torch运行了overflow和后续finite两次iteration，观察到了skip/backoff和后续成功更新。

官方loop不会因一次scaled-gradient overflow退出；梯度norm非有限只进入日志。下一iteration的optimizer.zero_grad清除优化器参数的旧梯度，低scale继续参与backward。正常的skip并不保证之后任意sample/epoch都稳定。
''')
    md('RUNNER_VS_UPSTREAM_AMP_DIFF.md',f'''
# v21.1 runner 与 frozen upstream 的 AMP 对照

结论：AMP/backward/unscale/step/update相同；额外梯度fatal gate不属于官方算法。**实际STOP发生在scaler处理之后，必须纠正“pre-scaler STOP”的假设。**

## 按执行顺序对比

1. upstream omniquant.py:257–262在traincast内量化、forward、loss；port:360–370同样操作，Qwen调用原生block_forward。未换训练目标。
2. upstream:263–265仅对非有限loss进入pdb；port:373–374改为raise。当前loss有限，未触发。
3. upstream:268 optimizer.zero_grad；port:383同样zero_grad。
4. upstream:269调用NativeScaler；port:399同样调用。utils.py的SHA完全相同（`b209ca43c737e583eb4e23c8bdcd9292c592e7378c6f5c5aac87a8700600b868`）。两者都是scale(loss).backward → unscale_ → norm → scaler.step → scaler.update。
5. 在owned gradient非有限时，官方scaler已经skip/backoff。然后port:400–401才调用v21_step_check_callback。
6. runner install_source_callbacks.step_check:424–442检查**此时仍非有限的unscaled LWC gradients**，日志写入后raise，阻止下一sample。scaler不会把Inf/NaN修复成有限梯度；安全性来自跳过更新。这种post-scaler finite-gradient绝对要求同样会拒绝合法AMP event。
7. upstream:270–274保存/汇总norm并写日志；runner epoch_cb:414–422额外要求norm_mean有限。一旦合法skip产生NaN norm，该gate仍会在epoch末STOP。
8. runner:575额外要求finite_gradients==steps；:307/:579/:587等将iteration乘积命名为effective_optimizer_updates。这些都要在未来runner修正方案中一起处理，只改一处callback不足。

## 关键时序代码

{excerpt(M3/'official_port/quantize/omniquant.py',397,403)}
{excerpt(M3/'run_formal_m3.py',424,442)}

原始formal runner SHA：`d69a93b55090d40b75011ed37115c563fa367db7e39b37d4824847f58fbd81f6`；port driver SHA：`7319c46bf73595b95796c6cf894f823ae862b12718249cda020c9af7ac66db60`。

v21.2正确定位了scaled HALF gradient overflow。本轮补充了严重度判断：该首次event在官方control flow下可恢复，不能等同fatal参数损坏。v21.1/v21.2历史文档没有覆盖，新结论保存在v21.3。
''')
    md('FORMAL_STEP_SEMANTICS.md',f'''
# Formal iteration 与 successful optimizer-step 语义

## 官方20 epochs按样本循环定义

upstream quantize/omniquant.py:251–254是`for epochs in range(args.epochs)`和`for j in range(args.nsamples//args.batch_size)`。没有成功step计数条件、重试队列或skip后补跑。同一block optimizer/scaler跨sample/epoch保留，下一block重新创建。

当前配置24 blocks×20 epochs×128 samples/batch1 = **61440 scheduled training iterations**。successful optimizer steps = iterations − scaler skips；不保证61440成功更新，不可为满足旧计数标签额外补跑sample。

checkpoint persistence是另一含义：只有完成整block并通过参数/量化检查才commit。诊断中的成功step不等于持久checkpoint更新。

## 历史block0–20的53760

主attempt `20261001_183359_6c00bf87`，逐行核对21×20×128=53760个prefix iterations，layer/epoch/sample顺序唯一且完整。每行loss有限，14个owned LWC grads在native scaler之后全有限；LET off，optimizer没有其他参数。按exact native scaler控制流，可推断这些iterations都成功step，因此本已完成prefix下iterations与successful steps均为53760。

证据限制：历史没有保存step hook/found_inf/scaler snapshots，**53760 successful steps是基于完整gradient日志与源码语义的推断，不是新拿到的直接step计数**。没有证据支持prefix曾发生owned-gradient overflow，也不把它扩展为所有未优化intermediate均无Inf。

training_epochs主attempt420行，loss/norm/参数有限。主attempt第53761行是L21/sample0失败，未计入block20。最初checkpoint IO失败attempt的2560行也不属于当前持久state，不能相加。

## 原manifest标签问题

runner save_block_checkpoint:307直接写`completed_layers * EPOCHS * SAMPLES`，不是optimizer.step计数器；FORMAL_PROTOCOL_LOCK的total_optimizer_updates=61440也是schedule乘积。历史prefix没有skip时标签数值恰好相等；允许dynamic skip之后标签的普遍含义必须修正。

未来新版本同时记录：attempted_iterations、successful_optimizer_steps、scaler_skips、checkpoint_committed_blocks，并保持每sample只消费一次。历史manifest/协议锁不修改，在新审计和新resume metadata中解释其旧标签。源文件、CSV SHA见SOURCE_SHA256和PROTECTED_HASHES；逐行核验结果见HISTORICAL_UPDATE_COUNT_AUDIT.json。
''')
    md('STOP_GATE_ASSESSMENT.md','''
# STOP gate assessment

分类：`STOP_GATE_TOO_STRICT`。

最终：`STOP_GATE_TOO_STRICT_RESUME_ALLOWED`，仅意味着控制流审计支持下一轮修正runner并准备resume，不是本轮执行授权，也不保证完整训练通过。

官方single-step：finite loss0.15326565504074097 → found_inf1 → optimizer.step未执行 → LWC及optimizer state不变 → scale65536→32768 → 函数正常返回。属于`RECOVERABLE_DYNAMIC_LOSS_SCALE_OVERFLOW`。

时序更正：v21.1并未阻止scaler处理，本来已经完成skip/backoff；随后post-scaler callback把残留的非有限grad当fatal，阻止官方下一sample。因此结论“too strict”成立，但理由是**拒绝已合法处理的overflow**，不是“在scaler前中断”。

同一disposable L21 optimizer/scaler连续samples0–3：一次skip，之后三次finite update，全部参数和AdamW state保持有限。每样本只跑一次，无重试，无手动scale，无LR/precision/quantizer变更。

v21.2的根因定位仍有效；“formal M3遇到梯度gate而STOP”是历史事实；将该event解释为官方算法本身fatal不成立。可以保留非有限grad/norm日志，不把known scaler skip的诊断Inf/NaN当参数损坏。

本轮没有patch runner、改变冻结协议、保存新formal checkpoint、执行正式resume。block20仍为最后合法checkpoint。完整epoch/后续层稳定性尚未验证；无需T4解释已验证的scaler行为。
''')
    md('FORMAL_RESUME_PLAN.md','''
# Formal resume plan — 下一轮执行方案，尚未实施

## 依据与边界

此次修改应是runner correctness fix：官方GradScaler原本允许skip/backoff，额外runner gate和update计数应准确观察该控制流。保留官方NativeScaler/AdamW、autocast、loss、LWC参数化、LR、epochs20、样本/顺序、量化器、teacher和prefix。禁止手动指定scale、retry或补跑skip样本。

当前实际gate为post-scaler，未来要调整它；不能声称删除了不存在的pre-scaler fatal gate。原v21.1源码/锁/checkpoint/结果保留，用新目录、新runner与新审计锁记录最小差异。

## 必要runner修正

1. 在optimizer_init边界新建官方AdamW/GradScaler，如原生每block语义。observer读取unscale之后、update清空tracking之前的found_inf，并用实际optimizer.step hook统计成功调用，不依赖AdamW返回值（成功step通常也返回None）。observer不得更改梯度、返回值、scale或调用次数。
2. step callback仅在官方wrapper返回后检查：loss有限、scaler状态有限、所有实际参数有限、optimizer state有限。known found_inf>0时要求step跳过、参数/状态未变、官方backoff已发生；将非有限grad/norm作为overflow诊断。found_inf=0但owned gradient非有限/实际step状态不一致，应STOP。
3. 对实际成功step要求owned unscaled grads及更新后参数/state有限；不得直接optimizer.step或额外unscale。missing LWC gradients等可保留一致性gate；LET仍off。
4. epoch_cb不能再要求所有norm_mean有限：合法skip可能使raw norm_mean为NaN。保留loss/参数安全gate，另记skip数、finite norm统计和raw norm状态；JSON用null+finite flag表达非有限统计，不偷偷丢弃events。upstreamnorm是诊断量，不影响optimizer。
5. 移除最终finite_gradients==steps的绝对要求，改为iterations=successful_steps+scaler_skips，并核对61440 schedule。scaler state reset/growth均交给官方，不为保证固定successful step数增补sample。
6. 新checkpoint metadata分别记录completed_blocks、scheduled/attempted iterations、successful steps、skips、计数provenance。继承block20的53760成功step以历史推断标记，不冒充hook实测。若未来支持层内恢复，必须同时保存optimizer/scaler/iteration/RNG；当前只需block边界恢复，不能继续失败进程内的半成品状态。

## 恢复步骤

先验证原block20 model/omni/RNG/config/source/token哈希。按已有native resume逻辑重构已完成0..20的FP teacher与learned quant prefix；从checkpoint RNG恢复，fresh L21 LWC初始化、fresh AdamW、fresh dynamic GradScaler。不要加载本轮一次性diagnostic的三个更新。

从L21 epoch0/sample0开始完整官方20×128循环；sample0应消费一次，允许scaler skip，并在sample1使用backoff后的scale。结束L21后按原算法量化/传播/整block检查再commit。L22/23尚未运行，仍按原生fresh per-block scaler处理，不预先承诺不会overflow。

## 下一轮执行前的门禁

先以同一冻结fixture核对新runner与本轮four-sample trace的loss/scale/found_inf/step/参数状态，及正常样本路径不变；review所有新增observer和gate差异。冻结新runner hash与iteration计数定义，保留当前cost/memory边界并监测剩余正式校准成本。该计划只允许以后继续calibration，export/WT2/C4须按后续明确授权和既有验收顺序。

## 本轮状态

没有实现或执行上述runner修正。当前M3不完整，正式STOP保持。审计verdict中的RESUME_ALLOWED指“可作为下一轮的官方语义一致方案”，不是已恢复/已完成。
''')
    lines='\n'.join(f"- sample{r['sample']}：loss={r['loss']:.17g}，scale {r['scale_before']:.0f}→{r['scale_after']:.0f}，found_inf={r['found_inf']:.0f}，skip={r['optimizer_step_skipped']}，参数变化={r['LWC_parameters_changed']}。" for r in sequence)
    md('REPORT.md',f'''
# v21.3 — Official AMP / GradScaler Semantics Audit

日期2026-10-01。Correctness **PASS**；最终分类 **STOP_GATE_TOO_STRICT_RESUME_ALLOWED**。

overflow类别：`RECOVERABLE_DYNAMIC_LOSS_SCALE_OVERFLOW`。本轮仅完成审计、single-step与最多4 samples的disposable诊断，**正式M3未resume，仍停在block20**。

## 逐项回答

1. **官方是否使用GradScaler？** Case A。冻结commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` 的omniquant.py:249创建NativeScaler；utils.py:31创建torch.cuda.amp.GradScaler。不是只使用autocast。
2. **dynamic还是fixed？** Dynamic。未显式传init_scale；当前torch2.6.0+cu124默认65536、growth2、backoff0.5、interval2000。per-block fresh，block内持续状态。
3. **hard STOP属于官方算法？** 不属于。官方只有nonfinite loss诊断与scaler overflow处理；额外LWC-gradient/epoch-norm/final-all-finite-count gates属于thesis runner。
4. **L21/sample0在官方flow下发生什么？** loss严格复现0.15326565504074097，四个gate/up upper/lower梯度counts与v21.2相同；found_inf1；原生wrapper正常完成step/update。single-step后立即停止，未进入sample1。
5. **optimizer step是否skip？** 是。prehook计数0，AdamW state仍空，LWC参数未变。scaler.step被调用与AdamW.step实际执行是两回事。
6. **scale是否backoff？** 65536→32768。没有手动指定scale，也没有改AMP、LR、optimizer或quantizer。
7. **下一sample是否允许继续？** 官方for-loop允许；本轮授权的四步诊断已观察到samples1–3恢复finite updates。
8. **20 epochs语义？** 固定epoch/sample iteration循环，不是固定successful optimizer-step数量。61440是24×20×128 scheduled iterations；skip不额外补跑。
9. **runner是否把recoverable当fatal？** 是，但实际STOP时序必须纠正：port:399先返回loss_scaler，:400–401才执行STOP callback；此前skip/backoff已经发生。是post-scaler额外gate阻止继续，不是pre-scaler拦截。
10. **可否不改算法从block20 resume？** 控制流审计支持下一轮修正runner gate/计数后恢复，见FORMAL_RESUME_PLAN。当前未patch或执行，不保证完整epoch或L22/23通过。
11. **最终分类？** `STOP_GATE_TOO_STRICT_RESUME_ALLOWED`，含义为审计与计划允许；下一轮执行指令仍必要。

## 原始连续序列

一次性L21对象，跨sample保留同一optimizer/scaler/参数，不重置、不retry。与v21.2每样本fresh-state诊断不同；sample2/3的loss含前面合法更新影响，不将二者混为同一个实验。

{lines}

四步1次overflow skip、3次实际AdamW更新，LWC/其他named parameters/optimizer state均有限。只在内存诊断对象更新，正式checkpoint committed updates=0。Independent micro-test也实际验证finite loss overflow→skip/backoff→下一finite update，见GRADSCALER_BEHAVIOR_CHECK。

## 历史53760计数

主attempt block0..20有完整唯一顺序的53760行，所有14个owned LWC grads在native scaler之后全有限，420个epoch日志有限。按exact scaler语义推断53760 iterations=53760 successful steps。历史未保存actual step hook或found_inf，因此该成功step数是源码+日志推断，不标为直接实测。

排除早期IO失败attempt的2560行，以及L21失败行。历史manifest用乘积命名effective_optimizer_updates，不能通用于含skip的后续正式结果；旧manifest不改，新resume元数据须区分iterations/successful steps/skips/commit。

## 证据与交付

- UPSTREAM_AMP_CONTROL_FLOW：exact官方文件、函数、行号、代码区段、SHA256；257-file frozen source manifest全部通过。
- RUNNER_VS_UPSTREAM_AMP_DIFF：逐阶段flow和真实callback顺序。
- OFFICIAL_OVERFLOW_HANDLING_REPLAY：单步scaled/unscaled梯度、found_inf、scale、参数/optimizer变化。
- DYNAMIC_SCALE_RECOVERY_TRACE.csv及SEQUENCE_SAMPLE_0..3.json：连续恢复轨迹。
- FORMAL_STEP_SEMANTICS、HISTORICAL_UPDATE_COUNT_AUDIT：官方loop和历史逐行计数。
- STOP_GATE_ASSESSMENT、FORMAL_RESUME_PLAN：gate修正依据与下一轮计划。

本轮复用v21.2已验证native checkpoint恢复产生的L21 fixture；再次核对block20、配置、源码与fixture哈希，没有重跑已完成prefix。所有父目录与v21.2 artifacts哈希保持一致。主要算法helper沿用已有代码，直接加载冻结**纯upstream utils.py**的NativeScaler执行，不调用v21.1 fatal callback。

## 成本与异常

真实layer诊断5次（single1 + continuous4），合计instrumented {costs['sum_layer_replay_seconds']:.6f}秒；synthetic micro-test2次。此时间不含加载、CPU哈希审计、IO，不是完整校准时间或速度估计。layer peak allocated {costs['max_layer_peak_allocated_bytes']} bytes、reserved {costs['max_layer_peak_reserved_bytes']} bytes。

single/sample0的raw grad norm非有限，JSON以null+norm_finite=False记录；这是合法overflow诊断，不能直接触发epoch-norm fatal gate。官方旧AMP API产生deprecation warnings；原Qwen SDPA sliding-window warning仍存在，未改backend或forward。它们不阻塞本轮semantics验证。

未执行T4、full epoch、正式训练、FP32 baseline、export、WT2/C4。没有假定其余所有samples/epochs都会恢复，也没有自动变更baseline。

**STOP：只交付审计结论与resume plan，等待下一轮执行指令。**
''')
    md('REPRO_COMMANDS.md','''
# Reproducibility

工作目录E:\\quant-thesis，已有.venv-gptq，torch2.6.0+cu124/RTX4060 Laptop。单步状态来源为v21.2验证过的checkpoint20恢复fixture；checkpoint、fixture、source哈希在PROTOCOL/SOURCE_SHA256/PROTECTED_HASHES。

已执行命令（cached raw结果存在时run_amp_audit.py拒绝重跑和覆盖）：

```powershell
& '.venv-gptq\\Scripts\\python.exe' -B -X utf8 'experiments\\range_search_w4\\results\\range_w4_omniquant_amp_audit_v21_3\\run_amp_audit.py'
```

CPU-only汇总与验证，不执行forward/backward：

```powershell
& '.venv-gptq\\Scripts\\python.exe' -B -X utf8 'experiments\\range_search_w4\\results\\range_w4_omniquant_amp_audit_v21_3\\finalize_audit.py'
```

本轮没有安装依赖、下载模型、调用正式run_calibration、写新正式checkpoint。保留CONSOLE.log。不得删除历史raw以强制重跑；独立重执行应另行授权并使用新的隔离目录及同一协议。
''')
    parsed=[load(p) for p in HERE.glob('*.json')]
    for p in HERE.glob('*.py'):
        compile(p.read_text(encoding='utf-8'),str(p),'exec')
    write('CODE_HASHES.json',{p.name:sha(p) for p in HERE.glob('*.py')})
    write('ARTIFACT_INTEGRITY_CHECK.json',{'status':'PASS','JSON_files_parsed':len(parsed),
        'CSV_rows_checked':len(rows),'source_files_checked':len(sources),'historical_artifacts_checked':len(protected),
        'diagnostic_script_syntax':'PASS','GPU_forward_backward_executed_by_finalizer':False})
    write('PROTECTED_ARTIFACT_CHECK.json',{'status':'PASS','files':len(protected),'all_hashes_unchanged':True})
    print('FINAL AMP AUDIT PASS: STOP_GATE_TOO_STRICT_RESUME_ALLOWED; formal remains STOP')


if __name__=='__main__':
    main()
