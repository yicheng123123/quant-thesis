# v39 Robust Hessian selector 数学可行性预检

## 决策：B — 存在不同选择空间，但统计规则尚未成立

目标 b+ρ||e||² 与v30完全相同，也与RQC未加权目标在归一化后重复。谱范数worst-case解释成立，但不能单独作为新算法。任意矩阵级search-onlyρ与共同τ射线的选择空间不同；当前没有有依据的半径估计规则，**统计建模未完成，停止实验推进**。不扫τ、不选历史held最优区间、不提出新质量运行。

## 先处理的M3冲突

已解决：v38遗漏v21.4续跑/v21.5最终化，错误把v21.1停止状态作为最终M3状态。v25引用的完成M3有效；原始窗口和导出哈希复核通过。WT2 PPL=13.545256974943815；冻结预算C4 PPL=21.44542796230099。v38原报告保留，详见 `M3_CORRECTION.md`。M2已有正式PPL亦有效，不能把本次纠正说成新质量收益。

## 候选与选择空间：只读search

L0/L12/L23 up/down，4个固定行[0,224,448,672]，S0/S1每组16篇search，每篇2048；共享已有120个候选tensor文件，480个row-alpha项，48单位。只做文件身份与已存标量运算，未加载tensor值。

- 31/48单位有共同输出/weight-error最小候选，任意非负ρ不换选。
- 17/48单位各有2个有效下包络候选；共65个有效候选区间，415个从不获选。
- 复用v30完整search路径：18个共同ρ开区间、17个交点；共同τ=ρ/κ有18个开区间配置。本轮没有重复τ=.01/.1/1的实验。
- 12个矩阵×split各自用一个ρ（4个固定行共享），开区间配置的笛卡尔积为7200种。它是数学可达空间，不是7200次实验，不是新结果，更不是允许分别给每行ρ的证明。各矩阵路径见 `MATRIX_RHO_PATH.csv`。
- 可改变单位分布：L0 up S0/S1为2/4；L12 down为1/1，up为1/0；L23 down为4/2，up为1/1；L0 down两split均0。
- 切换多数是0.8→0.9；另有L0 up R672 S1的1.0→0.9，以及L23 up R0 S1的0.9→0.8。ρ并不必然把alpha推向1。

每个候选的精确有效区间、端点归属、等效τ区间见 `CANDIDATE_SELECTION_INTERVALS.csv`；所有单位概览见 `UNIT_SELECTION_SPACE.csv`。将保存的小数转为精确有理数并相交所有线性不等式，未离散扫描ρ。端点精确并列优先较大alpha，与旧路径一致；旧60位Decimal交点没有造成边界选择差异。原ρ=0选择复现。

v30关于预固定正τ的负结果没有被扩大为“所有ρ都失败”。本轮不读取其held候选表/路径held汇总；因此不报告哪个ρ有held收益，也没有据此选统计规则。局部4行的7200配置更不代表完整矩阵或PPL泛化。

## 数学与统计边界

`OBJECTIVE_EQUIVALENCE.md`证明 C PSD、对称谱球、ρ≥0下的精确最坏情况，包含e=0及矩阵共享扰动；使用未阻尼、未中心化C。保留原直接分数的FP执行口径，不消除v28数值不一致历史结论。

`STATISTICAL_ASSUMPTIONS.md`将总体平均、未来单篇、跨领域三个对象分开。n=16文章，不是32768独立tokens。已有trace/κ不足以估计文章级谱范数置信半径，前缀/候选数据依赖及尾部假设未解决。经验文章偏差分位数不能直接称置信界。

`PRIOR_WORK_COMPARISON.md`核对[RQC/SARQC原文v2](https://arxiv.org/html/2605.05693v2)，区分目标重复、robust解释、候选生成区别与可能但尚未完成的统计新增。当前不提出可执行ρ规则，后续实验协议状态为NOT_PROPOSED_STATISTICAL_MODEL_UNFINISHED。

## 正确性、成本与复现

最终检查2709/2709 PASS：来源hash、M3窗口复算、候选metadata/search分数一致、MSE/SSE/weight归一化、48个ρ0选择、全部旧下包络序列/区间。PASS范围是CPU证据审计，不是新GPU parity或质量门槛。

标准库预检执行2次：首次CPU2.0625s / wall3.894671s；补充矩阵级路径及前缀来源核验后的最终执行CPU1.765625s / wall1.7860876s。合计预检CPU3.828125s；源文件阅读和联网查重时间另列，不称实验运行成本。CPU上限300s；未构建二阶矩、未做特征分解，所有GPU/前向/候选/新held/PPL均0，P3内容未读取。

```powershell
Set-Location E:/quant-thesis
.venv-gptq/Scripts/python.exe -B experiments/range_search_w4/results/range_w4_robust_hessian_precheck_v39/precheck_v39.py
```

复现脚本只写本目录，带CPU与保守执行wall双300s停止检查；失败写STOP.json。仅导入Python标准库，不导入torch/模型。来源清单明确读取文件；所有历史产物和公共实现保持只读。大checkpoint本轮仅流式hash，不载入模型，也不覆盖原产物。

尚未运行：ρ估计、文章级矩统计、统计覆盖验证、新文章质量、完整Linear全部行质量、全模型量化/PPL。数学可达≠统计可靠，检查通过≠方法有收益。本轮最终分类B，不自动追加实验。
