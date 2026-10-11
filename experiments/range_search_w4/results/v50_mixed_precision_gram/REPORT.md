# v50 混合精度 Gram 构建试验：D，完整比较未完成

冻结流程可独立构建混合精度 C 并进行候选接受，不使用原 FP64 C 作生产选择。实际运行完成了 baseline up_proj，并执行到 mixed up_proj 第23轮；进程随后无终态/异常处理记录地退出，精确原因与退出时间未知。2026-10-08 22:06 检查时原 PID 已不存在。日志/计时最后更新在03:24。系统未在这段时间重新启动，不能将中间时间差当作 GPU 计算时间。

不能重建确切的剩余累计 GPU 预算，故没有重启或重置1200秒预算。冻结代码和规则未改变。本轮的交付结果是缺失范围明确的 D，不把不完整路径写成数值PASS。

## 已观测的成本结论

up_proj 的同步 C 构建阶段：原 FP64 2.659039s；FP32块内Gram＋FP64累加与列能量统计 0.318169s，Gram阶段实测比 8.357×。但是混合路径已完成23次不确定候选raw-X复核调用，嵌套同步时间合计 209.968691s，已超过原路径完整up_proj的C＋修复＋终态检查 27.829580s。即使剩余工作耗时为零，混合up_proj也不能在本次规则下取得完整矩阵总成本收益。down_proj尚未开始，不能将up结果推广成down或整模型加速。

最终建议：**不进入下一轮整模型验证**。当前不确定性流程的必要复核开销吞没了已观测的 Gram 节省，且混合完整矩阵数值覆盖未完成。未调整触发规则或用验证最大误差调阈值；也不自动开展新方案。

## v49成本与局部/全模型差距

v49完整24block净修复4543.075288s；全部原始C事件97次/1755.010978s，含失败qkv一次/2.741256s。有效正式96次C为1752.269722s，占净修复38.57%。真实输入N=262144，4096-token块：72次896×896 C（qkv、o、up_gate各24次）合计186.528726s；24次4864×4864 C（down）合计1565.740995s。完整调用清单见V49_GRAM_CALLS.csv。

sibling已经在run_layer内共享C：qkv三个投影共用一次，up/gate两个投影共用一次。不是本轮新增优化。v47的54.03%是v46 proposal直接评分占比，不能代替v49实测份额。

v48局部1.508×来自两个L0 MLP完整矩阵并计入C；v46/v49完整模型净block wall比是1.207×。投影形状/输出行混合和分母不同。v48局部control为比较增加了全行独立FP64终态检查；v46原正式路径的终态使用FP32直接loss，v49则为168个Linear执行完整raw-X FP64终态检查，有效成功阶段2123.186194s（原始169条事件2124.068145s，含一次失败目标）。这项检查和有效C合计约85.30%的v49净block wall，限制了全模型收益。局部倍率不能直接外推；这些是不同覆盖、相同口径核查后的成本证据，不是因果分摊或运算量估计。

## 数值规则与独立性

唯一新C路径固定4096-token块、FP32块内完整GEMM、FP64累加/归一化/对称化；TF32 off且autocast禁用。FP32舍入不会被随后转FP64恢复。非对角与跨G128项全部保留。数值定义、运算假设、gamma32/gamma64及列能量h、rho均在GPU前冻结于NUMERICAL_RULES.md/PROTOCOL.json。

附加Gram不确定性为rho*(2|delta|h_j sum_i |e_i|h_i + delta²h_j²)。它仅依赖当前X的能量统计及当前e/真实FP16网格delta，无需先算FP64 C。继承v48搜索函数、256输出行chunk、确定性坐标/-1/+1顺序、合法码、32接受上限、128*d候选评价、拒绝集合与p刷新里程碑。

预测明显下降时用附加包络判断；不确定时独立raw-X FP64计算mean(2ab+b²)。原same-C缓存/公式检查保持严格；Gram误差只用于预测与raw-X的比较允许量，不用于放宽raw-X下降的符号判定。它依赖未经逐kernel证明的FP32归约假设，是经验数值流程，不是安全证书或全程理论单调性保证。

原FP64 C只作为baseline及之后计划的验证参考，未传入混合路径生产选择。本次混合未结束，因此C误差对照尚未生成。

## 来源与实际数值覆盖

锁定v49模型SHA为8a4e2aed53ed0f72c20780ca80ad7a2f751c8babb1c509d8f532054f150ecaf8。L0 up/down输入从该模型前缀重放，两份X哈希均匹配正式提交组；输入来自正式128×2048文章/token。兩路径使用同一可信原始M1 codes/grid和只读pristine FP16 W，目标模型未提交或更改。

CPU确定性检查通过：真实下降接受、正增量拒绝、拒绝状态/blacklist恢复、same-C缓存及公式错误注入、原更新/评价上限、固定构造混合Gram参照。baseline up_proj C哈希、终态整数码与评价次数匹配v49，全部4864行终态raw-X FP64检查通过；事前固定[0,224,448,672]四行完整轨迹的128次接受更新全部经过独立raw-X核验。

baseline up_proj：接受140498，候选评价253768708，拒绝0，不确定复核0；raw-X FP64初末MSE总和3.852174952391→3.275073240744。最大终态误差6.07e-18，max p漂移4.44e-16。C、初始化/修复、全行终态实测分别为2.663529、2.889161、22.268533s；其完整pipeline wall为27.829580s。独立shadow成本单列于STAGE_COSTS.csv。

mixed up_proj仅有23轮接受日志、24次已完成候选评分timer及23次已完成raw-X复核timer；日志记载108,818次接受、0次拒绝，但没有完整终态codes、精确累计候选评价/复核行数或shadow结果。无已记录的数值异常不等于PASS。down两条路径未执行。终态码差异、C误差、完整接受差异、完整矩阵总成本/峰值比较均标NOT_COMPLETED，见MISSING_EVIDENCE.csv/VALIDATION_COVERAGE.csv；C_ERROR_STATUS.json明确两种C误差数据未完成的原因。

已记录的显存：baseline完整up_proj峰值allocated 880,304,640 bytes、reserved 1,484,783,616 bytes；mixed部分up_proj已关闭timer中最大allocated 941,505,536 bytes、reserved 1,392,508,928 bytes。后一项不覆盖未关闭的在途操作，不能当作完整混合矩阵峰值。输入捕获阶段已记录峰值allocated 1,079,519,744 bytes。见OBSERVED_PEAK_MEMORY.csv。

## 预算与计时覆盖

最后完整timer的GPU_START elapsed为251.405280s；精确进程结束时间及之后在途工作时间未记录，不能将此数写成完整GPU总成本。启动上限1200s的watchdog和所有deadline保持原样；没有第二次启动。中间约18小时的外部时间间隔不是GPU消耗测量。

STAGE_TIMINGS.csv保留已完成阶段原始同步计时；STAGE_COSTS.csv标明fast_uncertain_refresh_and_rawX内嵌于fast_propose_and_settle，不能重加。C计时包含混合路径列能量/触发统计；矩阵pipeline含初始化、原FP32阈值回归、搜索、刷新、必要复核及全行终态。验证shadow、输入重放/cache IO、本地证据IO另列。未关闭的timer没有数据；不按操作数补算。原FP64 baseline C拟用于后续验证，但本次没有额外构建验证C或完成C差值计算。

## 交付与复现

源码：mixed_gram.py、run_v50.py；冻结规则：NUMERICAL_RULES.md、PROTOCOL.json；原始/汇总计时：STAGE_TIMINGS.csv、MATRIX_TIMINGS.csv、STAGE_COSTS.csv；来源与不变检查：SOURCE_AUDIT.json、SOURCE_IMMUTABILITY.json；输入哈希：INPUT_SOURCE_CHECKS.json；现存轨迹：SHADOW_TRAJECTORIES.csv；中断/缺失/判断：INTERRUPTION_AUDIT.json、VALIDATION_COVERAGE.csv、MISSING_EVIDENCE.csv、DECISION.json。

复现入口见REPRODUCE.md。现有目录保留GPU_START，runner会拒绝静默重跑/预算重置。本轮不运行整模型修复、held、PPL，不保存可部署的新量化模型，不自动启动下一版。Sol high模型设置无法通过本接口确认或切换。
