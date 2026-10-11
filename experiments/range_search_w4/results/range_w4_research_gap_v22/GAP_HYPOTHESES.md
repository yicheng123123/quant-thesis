# Gap hypotheses：证据强度与可判别性

“优先级”是下一次实验的信息价值，不是已经证明的因果排名。当前没有单因素消融能分解 M3−M2。

| 假设 | 支持证据 | 反证/限制 | 判别实验 | 优先级 |
|---|---|---|---|---|
| H1：Linear-local 目标太局部 | v18 search 行均值收益12.826748%，WT2 PPL仅0.891748%；M3优化block并更好；非线性/下游计算缺失 | 两个百分比单位不同，不能直接作为“低转化率”；没有同候选/同格式目标消融；格式与teacher trajectory混杂 | E1同候选、同row干预、同teacher输入，local与block排名/held-out比较 | 1 / PRIMARY |
| H2：共享α限制容量 | M2一行1变量；M3每组2端点；本轮54.63%行组间端点平均系数跨度>.05，深层明显更大 | LWC模式来自不同格式/目标/优化；不证明这些组差异必要；更多自由度可能放大v11风险 | E2仅2个预设region，保持symmetric/GPTQ/local目标，匹配预算scalar对照 | 2 / BACKUP |
| H3：.1步长是瓶颈 | M2 .8/.9/1合计98.81%；未评价.85/.95等，可能有量化状态跳变 | histogram不含曲率/中间候选信息；M3多项变化，不能推断细网格可补齐gap；v14粗网格排名稳定也不排除细网格 | E3原10点、包含原点的16点、10点评价预算对照 | 3 / 便宜排除项 |
| H4：projection联合优化重要 | attention的QK与V、SwiGLU的gate×up明确耦合；M3七投影共用loss | 共享输入不等于优化可分或不可分；没有交互消融，M3 joint≠joint贡献证明 | E1后续固定q/k或gate/up两行的2×2 block SSE interaction，先不联合穷举 | 4 / H1后续机制 |
| H5：compensation与clipping的下游交互 | α改变group量化误差及后续补偿轨迹；v13方向×geometry；局部误差方向可能被下游放大 | M2已对每候选完整重跑GPTQ并评分实际Q，**不是RTN proxy没有评估补偿**；没有算法bug证据 | E1记录候选error direction和下游block排名；以后固定补偿有/无另做单因素消融 | 4 / H1机制，非修复项 |
| H6：更多compute是主要来源 | M3多轮联合连续更新，对比M2有限候选 | 61,438 optimizer steps与1,680 GPTQ executions不等价；M3 exact wall缺失；计算/格式/容量/目标未匹配 | 同纯搜索格式/容量、固定评价预算，比较block/local；必要时以后匹配优化预算做学习对照 | 5 / 尚不可识别 |

## 不应遗漏的额外混杂

H_format：M3 asymmetric affine、M2 symmetric；即使目标完全相同，网格/zero差异仍能改变PPL。它不进入新的主线算法，但任何“解释M3优势”的因果结论都需要未来格式控制（小型固定格式RTN/GPTQ消融，随后才考虑模型级）。当前没有该证据，不给贡献百分比。

H_teacher：M3 teacher输入来自FP prefix，student来自quantized prefix；M2 teacher和student共享当前Linear的X。E1先固定同一block输入，隔离块内目标效应；成功后再单独测teacher alignment，不能一次更改两项。

## 必须保留的否定证据

- v11：96预选行×10splits非总体随机样本，132/960 gain<−5%，97<−10%，51<−20%；最差−115.19%；top1/top5/top20贡献35.41/62.57/94.39%净收益。pooled17.77%不能抹去长尾。
- v12：独立新48行Gate FAIL；仅保留23.82% C收益，严重失败下降42.03%，都没达到注册的50%门槛；最差−412.09%仍被接受。不重调gate阈值。
- v13：最坏case四篇search都偏好α=.7，rank agreement=.970；held-out单文档贡献88.63% excess SSE。误差方向与H漂移的精确恒等式支持机制描述，不能将使用held-out的方向指标作为预测器。文章投票不是已有答案。
- v14：K4→K32 native baseline改善35.51%，但预注册clipping风险gate失败；严重数1→1，最差−12.99→−22.31%。更多coverage不是本轮主要补救，不否认其对nativeGPTQ价值。
- v15：QPREFIX pooled held-out15.81% vs FP13.87%，33/48 α一致、47/48差≤一格。信号不是纯FP输入伪象；不同H和候选error direction仍混杂，不能据此证明prefix风险已解决。

数据来源为以上各轮原报告与 v19独立审计；新读参数统计见 `M3_RANGE_DIAGNOSTIC.md`。这些证据提高H1/H2研究优先级，仍不构成它们的验证。
