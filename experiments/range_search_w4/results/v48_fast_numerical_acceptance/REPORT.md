# v48 — 固定网格整数修复的快速数值接受机制

## 结果：A

两个完整矩阵的独立快速路径通过已覆盖的数值检查。含初始化、p刷新、复核、packing和全行独立FP64终态检查的修复wall合计：原路径 **130.323s**，快速路径 **62.585s**，实测 **2.082×**。没有减少32次更新上限或候选预算。

- L0 mlp.up_proj，4864×896：原57.940s，快30.272s（1.914×）；C单列3.046s。终态不同码0、不同行0；快速减原路径raw-X FP64 MSE总和0。
- L0 mlp.down_proj，896×4864：原72.383s，快32.313s（2.240×）；C单列67.642s。终态不同码0、不同行0；快速减原路径raw-X FP64 MSE总和0。
- mlp.up_proj 两路径各接受140,498次修改、评价253,768,708个合法候选（快速路径相同）；完整矩阵没有拒绝或不确定复核。逐行预算与停止状态见ROWS_*.csv和STOP_COUNTS.csv。
- mlp.down_proj 两路径各接受28,672次修改、评价277,386,920个合法候选（快速路径相同）；完整矩阵没有拒绝或不确定复核。逐行预算与停止状态见ROWS_*.csv和STOP_COUNTS.csv。

将同样的C构建成本各计入一次后，两矩阵合计速度比为 **1.508×**。这只是这两个固定输入完整矩阵的比较，不是完整模型校准加速。控制路径先运行，快速路径后运行；一次同步wall测量有顺序、缓存、功率/温度和系统负载影响，没有跨次稳定性结论。

## 数值机制与明确变更

完整规则和假设在 ACCEPTANCE_RULES.md，逐项协议变化在 PROTOCOL_CHANGES.csv。搜索枚举、预算、阈值和确定性tie保持原函数；接受符号从FP32完整输出loss差改为FP64二次型增量及不确定时的raw-X复核；固定里程碑检查后刷新p/B。不声称与v46逐步完全等价。

C使用原FP64完整未阻尼构建路径。B、cache尺度和gamma累积长度构成事前冻结的经验误差包络。abs(C)不能严格界定原始Gram的非对角消去误差，GPU归约路径也没有严格误差定理；因此这不是安全证书。通过仅是以下已有输入/状态上的实现数值证据。

## 正确性覆盖

- 历史：v43的3个失败候选、v44失败row211、v46全部9个正式拒绝候选，共13个；复原真实prefix输入并匹配X/C历史哈希，原完整批量FP32增量逐位一致，独立raw-X FP64公式核验通过。历史停止状态只读。部分FP32非下降为FP64下降，v48允许接受，这是显式规则变化。
- 完整目标：M1原始可信codes/grid，pristine FP16 W，128×2048输入；锁定v46 R1前缀恢复X并匹配哈希；C同一份，两路径状态独立。控制终态codes/evaluations与v46一致。
- 逐步：事前固定rows=[0,224,448,672]，两个投影、两路径，独立raw-X FP64核验 **512次接受更新**；全部严格下降。详见SHADOW_TRAJECTORIES.csv和SHADOW_COVERAGE.csv。没有逐步核验其余所有行。
- 终态：每路径全部5760输出行，独立原始X FP64直接重构初态/终态，与二次型及快速增量记账核验；所有行终态下降。码合法、zero=8、G128、网格不变、初末packing回环、W/X/C哈希不变；锁定R1及历史源码/失败状态文件SHA前后不变。
- 完整快速目标拒绝0个、不确定raw-X复核0个；不能因复核分支未在完整矩阵触发而省略其覆盖，确定性小例包含正增量复核拒绝、状态恢复、错误注入和预算检查。没有确认的数学目标上升或实质正确性失败。

## 成本口径

本次累计进程wall **412.810s /1200s**，含启动、诊断、失败用例、C、两路径、终态、shadow和IO。同步历史诊断阶段合计104.575s；shadow验证 **23.685s**独立列出，没有混入快速执行成本。共享目标C构建70.688s，不混入修复wall。
峰值allocated 1.833GiB，reserved 2.523GiB。NVIDIA GeForce RTX 4060 Laptop GPU，PyTorch2.6.0+cu124；TF32关闭。

MATRIX_TIMINGS.csv为完整路径计时；STAGE_TIMINGS.csv为原始同步阶段wall。fast_uncertain_refresh_and_rawX是fast_propose_and_settle内嵌timer，不能重复相加。LOCAL_*.pt写入和未包timer的hash/调度/GC属于processwall中的未独立归因开销，不能冒称独立GPU kernel时间。无额外profiler；已有同步timer本身的开销未单独测量。

1000s计划预留是运行前预测；上述成本均为本次测量。v47的54.03%仅作问题来源，未用省掉的评分次数推算本次速度。

## 判断与边界

A表示本轮完整矩阵数值检查通过且有实测收益；支持继续研究这种接受实现。它不继承R1/R2的PPL，不证明完整模型质量迁移、普适数值保证或论文原创性。没有执行新整模型、held、PPL、模型提交或保存可部署checkpoint；LOCAL_*.pt只是局部修复证据。按要求完成后停止。

## 交付

fast_accept.py / run_v48.py：实现与复用入口；ACCEPTANCE_RULES.md：冻结规则；SOURCE_AUDIT.json / SOURCE_IMMUTABILITY.json：来源；HISTORICAL_CANDIDATE_CHECKS.csv：旧失败/拒绝；ROWS_*.csv、LOCAL_*.pt：逐行和轨迹证据；SHADOW_*：独立逐步检查；COMPARE_*.json：全行终态与数值差异；STAGE_*和MATRIX_TIMINGS.csv：原始/汇总成本；REPRODUCE.md：命令。

模型设置：用户要求Sol high，当前接口无法确认或切换。
