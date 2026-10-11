# v11 full10 独立科学审查与下一阶段决定

审查日期：2026-09-30。对象：Qwen2.5-0.5B-Instruct 的 96 个预冻结行，L0/L12/L23 × q/o/up/down 投影，每个投影 8 行，S00–S09。本文是已有产物的只读复算与描述性分析；没有新量化、PPL 或 predictor 训练。审查程序为 [`review_full10_v11.py`](review_full10_v11.py)，派生数值在 [`SOL_REVIEW_DERIVED_METRICS.json`](SOL_REVIEW_DERIVED_METRICS.json)。

## 1. Correctness audit：PASS，范围有界

- 本审查重新计算了 96-row manifest 和 v9 split manifest SHA-256，分别为 `c56eeaa1d1af6e4ed8dd78f811c811c66758b215659a98bf8550052c786e6106` 与 `34a24ef8602b8ce1e38e7a284f112acf7f766962d87d737f69148b78bcfaba78`。十个 split 均为四篇 search、四篇 held-out，同 split 无文章重叠；全体 search 40 篇与 held-out 40 篇也互不重叠。逐个 raw 记录的文章索引与冻结 manifest 相符。
- 重读 960 个 raw row/split JSON，核对每条的 10 个 alpha 候选、search MSE `argmin`、D/C search 与 held-out SSE，以及 9,600 条 landscape CSV 数值；重算 96 个冻结标签、pooled SSE、固定 alpha 汇总，全部相符。S00–S02 与 S03–S09 只按冻结 split 连接；没有改变目标或选择规则。每个候选的 `heldout_sse` 是事后评价字段，源码选择语句只读 `search_output_mse`。
- 根据现存 [`FULL10_CORRECTNESS.json`](FULL10_CORRECTNESS.json) 与 preflight 的原生比较，S03 对四种投影各有一个 alpha=1 spot parity：FP16 最终权重、scale、zero、search/held-out SSE 均相符；记录中的权重不匹配元素数和损失差为零。这支持该实现中 **D = dynamic GPTQ(alpha=1)** 的局部等价，不等于全 96 行逐一与 native 实现重跑的证明。源码 [`dynamic_core.py`](../../dynamic_core.py) 显示每个候选从原始权重 clone，在组入口从已补偿权重估计范围、按 alpha 缩放 scale，并保留 GPTQ 补偿。
- 本审查也重新哈希 v8/v9/v10 三棵历史结果树，与 v11 前的快照逐文件比对；重新哈希 Stage 1 冻结的 630 个文件。当前均相符。它们没有被本轮审查重写。
- 协议为 FP16 权重、FP32 GPTQ/损失、symmetric W4A16 G128、block 128、damping 0.1、`desc_act=False`，输入来自 FP 模型。模型修订为 `7ae557604adf67be50417f59c2c2f167def9a775`。无 quantized-prefix、整模型 PPL 或下游任务证据。Transformers 的 SDPA/sliding-window warning 与未做其他 attention backend parity 仍是跨实现解释限制。

若要复算本审查的 CSV 和 JSON（不会运行 GPTQ）：

```powershell
& '.venv-gptq\Scripts\python.exe' 'experiments/range_search_w4/results/range_w4_96row_v11/review_full10_v11.py'
```

## 2. 冻结分类与四个规则空档

按 [`STAGE2_PROTOCOL_AMENDMENT.md`](STAGE2_PROTOCOL_AMENDMENT.md) 原规则：Robust-positive 26/96、Mixed-risk 61/96、Neutral 5/96、Unclassified 4/96。四个空档都只出现正增益或零增益，没有负增益；零点都来自选中 alpha=1。其 10 次选择均在 0.9/1.0，连续三档的稳定计数都是 10/10。这是规则未覆盖“间歇性改进但本轮无损失”的模式，不能将它们事后归为 robust，也不能把本轮无负样本解释为未来无风险。

- `L0_down_proj_R517`：6 正、4 零；median +18.87%，worst 0%；alpha=0.9 六次、1.0 四次。逐 split gain：`[0,0,0,25.10%,0,29.06%,17.44%,24.15%,20.30%,23.34%]`。
- `L12_down_proj_R496`：5 正、5 零；median +2.36%，worst 0%；0.9/1.0 各五次。gain：`[26.10%,36.86%,0,0,28.29%,24.23%,4.72%,0,0,0]`。
- `L12_o_proj_R827`：7 正、3 零；median +9.08%，worst 0%；0.9 七次、1.0 三次。gain：`[7.99%,0,31.97%,33.11%,0,18.92%,32.81%,5.78%,10.17%,0]`。
- `L23_q_proj_R696`：7 正、3 零；median +7.40%，worst 0%；0.9 七次、1.0 三次。gain：`[0,9.60%,35.55%,7.31%,0,0,7.49%,6.18%,12.89%,43.37%]`。

它们是**可描述的行为候选**，但目前最稳妥的结论是冻结规则不完备。未来可预注册一个“nonnegative/intermittent”标签，再在新数据检验；不能据此改写 v11 的 26/61/5/4 prevalence。

## 3. 主结论：结束 blanket clipping 的“安全统一方法”表述

96 行中只有 26 行满足预定 robust 标准，61 行为 mixed-risk。960 次 row/split 中 178 次 held-out gain <0，97 次 <−10%，51 次 <−20%，22 次 <−40%；最差 `L23_up_proj_R3814/S00` 为 −115.19%，虽然同次 search gain 为 +29.21%。候选包含 alpha=1，因此 search loss 不能比 D 差；这对 held-out 不成立。所有行无条件执行 shared-alpha 搜索作为**安全的统一方法**，现有证据不支持。下一阶段问题应是 **“Which rows should be clipped?”**；保留 D 作为可选回退点。

这不否定局部收益：96 行的 row-median-gain 中位数为 +10.28%，960 次 row/split gain 中位数为 +11.25%。行层面 80 行的十 split 净 SSE 改善、11 行净恶化、5 行持平。研究问题由“能否在某些行减小局部输出误差”转为“能否在未知 held-out 结果前识别收益并限制尾部风险”。

## 4. 投影和层：结构显著，成因未识别

四个投影各 24 行。q：10 robust、8 mixed、5 neutral、1 gap；row-median-gain 中位 +8.54%，23/240 次负 gain、13/240 次 <−10%，alpha=1 被选 82/240 次，D SSE 总量 1101.55，pooled gain +22.65%。o：2 robust、21 mixed、1 gap；row-median-gain 中位反而 +19.25%，50/240 次负 gain、35/240 次 <−10%，alpha=1 仅 35/240 次，D SSE 总量 46.43，pooled gain +21.81%。up：6 robust、18 mixed；中位 +10.12%，56/240 次负 gain、26/240 次 <−10%，alpha=1 为 27/240，D SSE 664.86，pooled +9.75%。down：8 robust、14 mixed、2 gap；中位 +9.84%，49/240 次负 gain、23/240 次 <−10%，alpha=1 为 32/240，D SSE 692.16，pooled +17.44%。最差尾部分别为 q −50.82%、o −59.92%、up −115.19%、down −91.98%。

因此 o_proj 的低 robust 比例并非“没有收益”：它常选择 alpha<1，典型 gain 较大，同时分布两侧也更宽、尾部更差。q_proj 较常选择 D，负例更少，并含 5 个 neutral。它们在当前三个层呈明显的**观测性 projection-specific pattern**；每投影只有 24 行且同层行共享输入/Hessian，不能据此断言投影类型本身为因果机制，也不能排除采样不确定性。

从实际选择分布看，q 的 alpha=0.6/0.7/0.8/0.9/1.0 次数为 13/25/37/83/82；o 为 5/48/66/86/35；up 为 0/1/80/132/27；down 为 2/22/82/102/32。D 的每次 held-out SSE 中位数分别约 1.616、0.0819、0.923、0.170；不能把 o 的小绝对 SSE 当作“更安全”，它的相对严重回归频率反而最高。

层 0/12/23 的 robust 分别为 9/32、7/32、10/32，差别小于 q 与 o 的 10/24 对 2/24。row-median-gain 中位分别 +7.21%、+10.67%、+12.22%；<−10% 次数分别为 19、36、42。L23 的 D SSE 2065.66，占全部 D SSE 的 82.46%，其 pooled gain +19.47% 且 top5 净贡献行都在 L23；不能从较高 pooled 或 median 推断“深层普遍更适合 clipping”。L0 q 的 pooled gain 甚至约 −0.12%，显示层与投影有交互。

L0/L12/L23 的 alpha=1 选择次数分别为 104/320、42/320、30/320；深层更常选较小 alpha，但这与层间输入统计、投影组成及行抽样共同变化，不能当作单独的层效应。

## 5. pooled gain 与固定 alpha

D held-out SSE 为 2504.9973，搜索所选 C 为 2059.8041，pooled gain +17.77%。这是按 baseline SSE 加权的总体重构误差下降，**不是平均行收益，也不是 PPL 改善**。按每行十 split 的 D−C 净 SSE 排序，top1、top5、top10、top20 分别贡献总净下降的 35.41%、62.57%、78.94%、94.39%；去掉 top1 仍为 +13.67%，去掉 top5 仍为 +9.61%。故收益并非仅一行造成，但高度集中，高 SSE 行显著影响总体值。正贡献合计 458.31 SSE，11 个净恶化行抵消 13.12 SSE。

固定 alpha 对照是描述性反事实，未据此在 held-out 上选择新的 universal alpha：0.7 pooled −70.53%、673/960 次回归、worst −1644.84%；0.8 pooled −12.68%、404/960 次回归、worst −915.50%；0.9 pooled +6.63%，但仍有 291/960 次回归、worst −279.63%。因此已测固定 alpha 不适合作为当前安全主方法；不能证明所有未来 universal scheme 均不可行。

## 6. Stage 1 三 split 筛选：富集，不能硬过滤

将 Stage 1 `candidate-positive` 当阳性、full10 robust-positive 当阳性：TP=25、FP=22、FN=1、TN=48；precision 25/47=53.19%，recall 25/26=96.15%，FPR 22/70=31.43%，NPV 48/49=97.96%。22 个 FP 包括 19 mixed 与 3 gap；FN 是 1 个 Stage1 mixed 但 full10 robust 的行。

这个混淆矩阵**不是独立验证**：full10 标签含 Stage1 的 S00–S02 三个 held-out split，Stage1 标签本身也使用这三次 held-out gain；它不是纯 search-side gate。作为额外描述，S03–S09 的文章与 S00–S02 完全不重叠：Stage1 candidate-positive 的 47 行在后七 split 的 gain 中位 +21.23%、负率 48/329=14.59%、严重负率 25/329=7.60%；Stage1 mixed-risk 的 48 行对应 +0.01%、70/336=20.83%、38/336=11.31%。但前一组仍有 18/47 行在后七 split 至少一次 <−10%。三 split 筛选可作为后续研究的 **enrichment** 线索，不能硬过滤，也不能叫低成本 search-only 安全规则；它本身用到了 2,880 个候选槽位中的 2,850 次新 GPTQ 运行。

## 7. feature/outcome：探索性，未训练 selector

[`feature_outcome_analysis.csv`](feature_outcome_analysis.csv) 提供 48 个预指定特征的 robust（n=26）与 mixed/neutral（n=66，四个 gap 单列）中位数、Q1/Q3、`(median_robust−median_other)/两组 IQR 均值`，及与 96 行 median held gain、负例频率、严重负例频率、worst gain 的 Spearman rho。[`feature_stratified_summary.csv`](feature_stratified_summary.csv) 给出每个投影的同样描述。相关只用于假设生成；48 项高度相关且未做多重比较校正，不报告显著性或选择精度。所有纳入特征来自权重或 search 侧；现有 feature 表中的 `activation_heldout_*` 和 held-out baseline 未用于候选规则。`hinv_cholesky_diag_*` 是所存逆矩阵 Cholesky 对角摘要，**不是 Hessian condition number**。

- Weight：robust 的 max-abs/RMS 中位 3.73，对照 4.05；excess kurtosis 0.252 对 0.485。但对应 median gain 的全体 rho 仅 −0.18、−0.16，max-abs/RMS 的分投影 rho 为 q −0.55、o +0.03、up +0.32、down −0.32，方向并不稳定；没有足够证据给出静态 weight gate。
- Activation/Hessian：例如 robust 的 search activation feature-max-RMS 中位较高，但同一 layer/projection 的八行共享完全相同的 activation 与 Hessian 汇总（在 12 个 layer/projection 单元内每列 `nunique=1`）。这些字段只能刻画单元差异，不能区分该单元内的 robust 与 mixed 行；跨行关联容易被 layer/projection 混杂。Hessian 均值与 activation RMS 还由同一输入统计诱导，不能当独立证据。
- GPTQ/landscape：每行十 split 的 search gain 中位，robust 为 27.50%（IQR 20.48–38.17%），对照 13.28%（3.70–19.98%）；与 held-out median gain 的 rho=+0.771，四投影内为 +0.635 至 +0.854。`alpha=1` 在 search 上距最优不超过 2% 的频率与 median held gain rho=−0.681；selected-alpha mean 为 −0.697。这是**典型收益的信号**，但 search gain 对负例率、严重负例率的 rho 仅 −0.140、−0.116。
- 搜索第一与第二名的相对 margin 中位，robust 0.0843、对照 0.0805；它与行负例率 rho=−0.367、严重负例率 rho=−0.318，四投影内严重率 rho 都为负（约 −0.16 至 −0.42），可作为风险假设。但 `L23_down_proj_R647` 的行 median margin 仍有 0.1474，同一行出现 −91.98% 回归，因此 margin 不能单独证明安全。基于“距最佳搜索 SSE 不超过 2% 的候选个数”定义的局部 flatness 与风险的相关很弱。
- 分 split 的 search gain 与 held-out gain rho=+0.466；只看 alpha<1 的 784 次为 +0.327。最差的 `L23_up_proj_R3814/S00` search gain +29.21% 却 held-out −115.19%；`L23_o_proj_R546/S07` 为 +43.21% 对 −59.92%。故不能把较大 search 改善当作风险保证，也不能仅凭 row-level rho=0.771 宣称已能预测 held-out magnitude。

## 8. 匹配案例与可检验机制假设

[`matched_case_analysis.csv`](matched_case_analysis.csv) 用透明规则取四投影各一个接近该投影 robust median 的行，再加高 gain 行，并在同一 layer/projection 中按 log10 D search SSE 最近匹配 mixed；另补同层同投影的 severe/near-neutral 反向配对。这些是审查后选的解释性案例，**不是预注册验证样本**。同层同投影的 activation/Hessian 数值相同，例如 L23 q 的 robust `R446`（median +31.25%，worst +14.62%）与 mixed `R633`（median −0.45%，worst −22.07%），其 max-abs/RMS 为 3.72 对 5.32，search gain median 43.83% 对 11.07%。L0 up 的 `R3557` 与 `R2047` 在相同输入/H 下，search gain median 41.88% 对 9.86%，worst +9.90% 对 −11.69%。这支持检查行内权重形状及 search landscape，而不是把共有的 H 值作为行选择器。

反例同样重要：L23 down `R647` median held gain +15.10%，但 worst −91.98%，search gain median +34.75%、margin 0.1474；与匹配的 robust `R138`（worst −6.86%）相比，单看较大的 search gain/margin 会选错。L12 o 的 `R89` 与 `R769` 都有正 median（+40.83% 对 +19.54%），后者仍有 −19.70% 的 worst。这说明“收益”和“尾部风险”应是两个目标。

以下均为**待验证机制假设**，不是因果发现：

1. 某些行较高的 group/全行极值与 kurtosis 使最佳 alpha 随文章变化；固定同一行权重但改变 search/held-out 文章组成，应能揭示范围选择边界的漂移。q/down 中的部分权重趋势值得定向检验，o/up 的反向或弱关系要求投影分层。
2. 搜索目标存在多个近邻候选时，微小的校准扰动会切换 alpha；margin 可能反映这一选择稳定性。应比较同一行不同冻结文档上的候选排序变化，并检验 margin 是否真正预测尾部，而非只预测投影类型。
3. 有些高 search gain、高 margin 行仍发生极端 held-out 损失，可能来自输入方向与权重误差对齐改变，而不是 loss surface 平坦。可在**已有保存的候选权重及输入**上分析误差向量投影；该机制分析尚未执行。
4. o_proj 典型收益高而 robust 少，可能与该投影的输入统计和被搜索到的较小 alpha 频率有关；需要新层重复观察，不能以当前 24 行给出架构因果解释。

## 9. 下一步只推荐一个独立验证实验

**决策：Case B，设计 search-side 的 clip/alpha=1 回退门控；现阶段不发布 selector。** 静态 weight 特征跨投影不稳，activation/H 不能在同一单元区分行；search gain 对典型收益有信号，margin 对风险有弱到中等信号，但失败反例明确。门控应使用搜索已经产生的指标，不查看 held-out；它只可能节省应用错误范围的风险，**不会加速既有十候选离线搜索**。在新数据运行前，须把 `search_gain_median`、`search_margin_median` 及其门槛或规则从 v11 discovery 数据一次冻结，明确 alpha=1 回退；新验证期间不得调门槛。候选可以从这两个连续量开始，但本审查没有声称某个数值阈值已经有效。

唯一推荐验证：在同一 checkpoint 的**未研究层 L4/L8/L20**，每个层×四投影事前用固定 seed 从四个等宽输出索引段各抽一行，共 48 个全新行；使用与旧 80 篇文档无交集的新完整文章，冻结 10 个 search/held-out split，每 split 各四篇、128 token。保持 W4A16、G128、动态范围时机、十个共享 alpha、原生补偿和 FP 输入不变。预注册三种配对策略 D（全部 alpha=1）、C（全部 search-best）、Gate（search 指标决定 C 或 D），记录每个 row/split 的 held-out gain、<−10% 与 <−40% 率、按 SSE 加权收益、各投影和层结果，以及 GPU 同步的总成本。建议在运行前冻结**共同通过条件**：Gate 的 <−10% 次数相对同次 C 至少减半；Gate 相对 D 的 pooled gain >0，且当 C 对 D 为正时至少保留 C 的一半 pooled gain；所有投影均报告净值及最差 case，不能以总体正值掩盖一个投影的严重尾部。这些门槛是下一轮的提议，不是 v11 已验证阈值。统计不应把 480 个高度相关 row/split 当 480 个独立模型；按行/层投影单元报告不确定性。失败则保留负结果，回到机制分析，不用同批验证样本再调门槛。

此设计为 48×10×10=4,800 个候选槽位。按 v11 两阶段正式 runner 合计 12,917 秒/9,600 槽位粗略折半约 1.8 小时，但 Stage2 实测比其带 1.5 倍裕量的预估仍慢 1.61 倍，且新文章输入捕获、预检和缓存不包含在这个折半值里；因此只作为预算量级，正式运行前重新做 spot parity、GPU 内存与耗时预检。v11 的两阶段正式 runner 共 3.59 小时、9,500 次新候选加 100 次严格复用；Stage2 峰值 CUDA reserved 1052 MiB（含捕获/原生 parity），这些均为离线校准成本，不是在线推理加速。

## 10. PPL gate 与研究路线

**Full-model PPL：NO-GO。** 当前只有 FP 输入的代表行局部输出误差；没有通过独立验证的风险门控，也没有量化前缀输入下的迁移证据。进入现有标准 PPL 前至少需要新层/新文档验证出相对 D 的收益与相对 C 的尾部改善，预先固定如何处理全模型其余行，并预算全模型候选 GPTQ 的成本。PPL 必须用同 checkpoint、token 计数与评估协议比较；局部 17.77% 不可写成模型级收益。

三条论文路线的当前判断：

- **A. Uniform GPTQ range search**：已有局部 pooled 与典型收益、native alpha=1 parity；最大缺口是 61/96 mixed 与灾难尾部，统一安全方法的现有证据不成立。下一步若坚持应先解释并控制分布迁移，而非直接全模型 PPL。发表风险高，尤其“搜索总能改进”只能在相同 search 数据/目标成立。
- **B. Selective Range Optimization**：最贴近 “Which rows should be clipped?”；不同投影与行存在异质性。最大缺口是可外推、低风险的行选择准则。下一步是上述唯一新层/新文档验证。发表风险在 selector 泛化、候选校准成本与现有方法差异说明；不能声称新颖性已确立。
- **C. Calibration-risk-aware fallback to alpha=1**：B 的更窄、可证伪工程路径；D 为严格候选，search landscape 的收益信号与 margin 的风险线索可供门控，但严重反例说明目前未验证。下一步同样是上述验证，不另起实验。发表风险在同一校准语料上过拟合阈值以及“安全”无法被最坏案例支持。

**最可信的当前研究问题是 B/C 的交集**：能否仅凭 search-side 诊断，在未见文章和未研究层上决定保留 clipped GPTQ 还是回退 native dynamic GPTQ，从而保留部分重构收益并减少严重退化？v11 已验证的是该问题值得检验，而不是 selector、PPL、泛化机制或论文新颖性已经成立。
