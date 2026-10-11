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

- sample0：loss=0.15326565504074097，scale 65536→32768，found_inf=1，skip=True，参数变化=False。
- sample1：loss=0.17120549082756042，scale 32768→32768，found_inf=0，skip=False，参数变化=True。
- sample2：loss=0.16788706183433533，scale 32768→32768，found_inf=0，skip=False，参数变化=True。
- sample3：loss=0.16902445256710052，scale 32768→32768，found_inf=0，skip=False，参数变化=True。

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

真实layer诊断5次（single1 + continuous4），合计instrumented 1.020965秒；synthetic micro-test2次。此时间不含加载、CPU哈希审计、IO，不是完整校准时间或速度估计。layer peak allocated 500600320 bytes、reserved 595591168 bytes。

single/sample0的raw grad norm非有限，JSON以null+norm_finite=False记录；这是合法overflow诊断，不能直接触发epoch-norm fatal gate。官方旧AMP API产生deprecation warnings；原Qwen SDPA sliding-window warning仍存在，未改backend或forward。它们不阻塞本轮semantics验证。

未执行T4、full epoch、正式训练、FP32 baseline、export、WT2/C4。没有假定其余所有samples/epochs都会恢复，也没有自动变更baseline。

**STOP：只交付审计结论与resume plan，等待下一轮执行指令。**
